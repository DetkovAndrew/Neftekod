"""
Агент экономики (ARCHITECTURE.md §6.7).

Переводит состояние процесса и каждый кандидат Агента оптимизации из
безразмерных прокси в РЕАЛЬНЫЕ производственные единицы:

* выпуск -- тонны в сутки (тег `242000:F17`, массовый расход
  гидроочищенного ДТ в цех №8);
* тепло -- Гкал/ч (печь П-3 АВТ и нагрев сырья гидроочистки, через
  опубликованную корреляцию теплоёмкости Kesler-Lee, `thermo.py`);
* электроэнергия -- кВт (политропная мощность циркуляционного
  компрессора ЦК-201 по фактическим расходу и перепаду давления);
* водород -- нм3/ч свежего ВСГ;
* деньги -- рубли в сутки по ценам из `config/economics.yaml`.

Два принципа, которые здесь соблюдаются жёстко
-----------------------------------------------
1. **Физика отделена от цен.** Все натуральные величины считаются из
   телеметрии и не зависят ни от одной цены. Цены -- отдельный,
   помеченный как допущение слой; заменяются правкой YAML. В карточке
   оператора всегда присутствуют обе части, поэтому видно, что именно
   является допущением, а что -- измерением.
2. **Приращение важнее абсолюта.** Для кандидата считается РАЗНИЦА
   с текущим режимом. Абсолютная себестоимость потребовала бы данных,
   которых нам не выдали (состав топливного газа, КПД конкретных машин,
   амортизация), а приращение от изменения уставки считается корректно
   и именно оно нужно для сравнения вариантов.

Экономика не имеет права перевесить качество или жёсткое ограничение:
она подключается к ранжированию УЖЕ допустимых кандидатов, после
Guard'а и hard-фильтров (ключевой принцип ТЗ, §0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neftekod_mas.economics.thermo import (
    compressor_power_kw,
    heating_duty_gcal_h,
    is_extrapolation,
)
from neftekod_mas.schemas import ControlCandidate, ProcessState

HOURS_PER_DAY = 24.0


@dataclass
class EnergyBreakdown:
    """Натуральные величины -- без единого рубля."""

    product_rate_t_h: float | None = None
    furnace_p3_duty_gcal_h: float | None = None
    ht_feed_heating_gcal_h: float | None = None
    recycle_compressor_kw: float | None = None
    makeup_hydrogen_nm3_h: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in {
            "product_rate_t_h": self.product_rate_t_h,
            "furnace_p3_duty_gcal_h": self.furnace_p3_duty_gcal_h,
            "ht_feed_heating_gcal_h": self.ht_feed_heating_gcal_h,
            "recycle_compressor_kw": self.recycle_compressor_kw,
            "makeup_hydrogen_nm3_h": self.makeup_hydrogen_nm3_h,
        }.items() if v is not None}


class EconomicsAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.prices = {k: float(v["value"]) for k, v in self.cfg.get("prices", {}).items()}
        self.constants = {k: float(v["value"]) for k, v in self.cfg.get("physical_constants", {}).items()}
        self.streams = self.cfg.get("streams", {})

    @property
    def available(self) -> bool:
        return bool(self.prices and self.streams)

    # -- вспомогательное -------------------------------------------------

    @staticmethod
    def _tag(state: ProcessState, tag: str | None) -> float | None:
        if not tag:
            return None
        reading = state.kip.get(tag)
        if reading is None or reading.value is None:
            return None
        return float(reading.value)

    # -- натуральные величины -------------------------------------------

    def energy_breakdown(self, state: ProcessState) -> EnergyBreakdown:
        out = EnergyBreakdown()

        prod = self.streams.get("product_diesel", {})
        out.product_rate_t_h = self._tag(state, prod.get("mass_flow_tag"))
        if out.product_rate_t_h is None:
            out.notes.append(
                f"Нет выпуска по тегу {prod.get('mass_flow_tag')} -- производительность в тоннах не считается"
            )

        furnace = self.streams.get("avt_vacuum_furnace_p3", {})
        flow = self._tag(state, furnace.get("mass_flow_tag"))
        t_in = self._tag(state, furnace.get("inlet_temp_tag"))
        t_out = self._tag(state, furnace.get("outlet_temp_tag"))
        if None not in (flow, t_in, t_out) and t_out > t_in:
            mean_t = 0.5 * (t_in + t_out)
            duty = heating_duty_gcal_h(
                flow, t_out - t_in,
                float(furnace["typical_density_kg_m3"]),
                float(furnace["typical_mean_boiling_point_c"]),
                mean_t,
            )
            out.furnace_p3_duty_gcal_h = duty / max(self.constants.get("furnace_efficiency", 1.0), 1e-6)
            if is_extrapolation(mean_t):
                out.notes.append(
                    f"Теплоёмкость печи П-3 считается экстраполяцией Kesler-Lee (средняя T={mean_t:.0f} °C)"
                )
        else:
            out.notes.append("Печь П-3: нет полного набора тегов (расход/вход/выход) -- тепло не считается")

        gas = self.streams.get("ht_recycle_gas", {})
        f_gas = self._tag(state, gas.get("flow_tag"))
        p_in = self._tag(state, gas.get("suction_pressure_tag"))
        p_out = self._tag(state, gas.get("discharge_pressure_tag"))
        if None not in (f_gas, p_in, p_out):
            out.recycle_compressor_kw = compressor_power_kw(
                f_gas, p_in, p_out,
                suction_temperature_c=self.constants.get("compressor_suction_temp_c", 40.0),
                polytropic_efficiency=self.constants.get("compressor_polytropic_efficiency", 0.75),
                k_ratio=self.constants.get("recycle_gas_k_ratio", 1.38),
            )
        else:
            out.notes.append("ЦК-201: нет расхода или давлений -- электроэнергия не считается")

        h2 = self.streams.get("ht_makeup_hydrogen", {})
        out.makeup_hydrogen_nm3_h = self._tag(state, h2.get("flow_tag"))
        return out

    # -- приращение по кандидату ----------------------------------------

    def candidate_effect(
        self, state: ProcessState, modified_state: ProcessState, candidate: ControlCandidate
    ) -> dict:
        """Экономический эффект кандидата в натуре и в рублях за сутки.

        Отдельно считается нагрев сырья гидроочистки: когда кандидат
        двигает уставку температуры входа в реактор, ровно это и есть
        дополнительное тепло, и оно считается точно -- Q = m*Cp*dT по
        фактическому расходу сырья, без всяких предположений о печи.
        """
        base = self.energy_breakdown(state)
        new = self.energy_breakdown(modified_state)

        deltas: dict[str, float] = {}
        for key in ("product_rate_t_h", "furnace_p3_duty_gcal_h",
                    "recycle_compressor_kw", "makeup_hydrogen_nm3_h"):
            b, n = getattr(base, key), getattr(new, key)
            if b is not None and n is not None:
                deltas[key] = n - b

        notes = list(dict.fromkeys(base.notes + new.notes))
        reactor_duty = self._reactor_heating_delta(state, candidate)
        if reactor_duty is not None:
            deltas["ht_feed_heating_gcal_h"] = reactor_duty

        money = self._to_rubles_per_day(deltas)
        return {
            "baseline_physical": base.as_dict(),
            "delta_physical": {k: round(v, 4) for k, v in deltas.items()},
            "delta_rub_per_day": {k: round(v, 1) for k, v in money.items()},
            "net_rub_per_day": round(sum(money.values()), 1) if money else None,
            "notes": notes,
            "prices_are_assumptions": True,
        }

    def _reactor_heating_delta(self, state: ProcessState, candidate: ControlCandidate) -> float | None:
        """Дополнительное тепло на нагрев сырья при сдвиге уставки
        температуры входа в реактор. Считается только для тегов
        температуры реакторной секции 24-2000 -- для остальных рычагов
        честно возвращается None (нечем измерить), а не 0."""
        feed = self.streams.get("ht_feed", {})
        flow = self._tag(state, feed.get("mass_flow_tag"))
        if flow is None:
            return None
        delta_t = 0.0
        for action in candidate.actions:
            if action.tag in ("242000:T5", "242000:T11", "242000:T6") and action.unit.startswith("°"):
                delta_t += action.recommended_value - action.current_value
        if abs(delta_t) < 1e-9:
            return None
        current = self._tag(state, "242000:T5") or 370.0
        return heating_duty_gcal_h(
            flow, delta_t,
            float(feed.get("typical_density_kg_m3", 845.0)),
            float(feed.get("typical_mean_boiling_point_c", 280.0)),
            current + 0.5 * delta_t,
        ) / max(self.constants.get("furnace_efficiency", 1.0), 1e-6)

    def _to_rubles_per_day(self, deltas: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        if "product_rate_t_h" in deltas and "diesel_rub_per_t" in self.prices:
            out["product"] = deltas["product_rate_t_h"] * HOURS_PER_DAY * self.prices["diesel_rub_per_t"]
        gcal = sum(deltas.get(k, 0.0) for k in ("furnace_p3_duty_gcal_h", "ht_feed_heating_gcal_h"))
        if gcal and "fuel_gas_rub_per_gcal" in self.prices:
            out["fuel_gas"] = -gcal * HOURS_PER_DAY * self.prices["fuel_gas_rub_per_gcal"]
        if "recycle_compressor_kw" in deltas and "electricity_rub_per_kwh" in self.prices:
            out["electricity"] = -deltas["recycle_compressor_kw"] * HOURS_PER_DAY * self.prices["electricity_rub_per_kwh"]
        if "makeup_hydrogen_nm3_h" in deltas and "hydrogen_rub_per_nm3" in self.prices:
            out["hydrogen"] = -deltas["makeup_hydrogen_nm3_h"] * HOURS_PER_DAY * self.prices["hydrogen_rub_per_nm3"]
        return {k: v for k, v in out.items() if abs(v) > 1e-9}
