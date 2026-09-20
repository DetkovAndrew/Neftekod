"""
Агент блендинга (ARCHITECTURE.md §6.6).

Отвечает за дизельный пул установки АВТ: из каких компонентов он
собран сейчас, каким будет качество смеси при других долях и какие
изменения долей допустимо предлагать как управляющее воздействие.

Почему это полноценный контур, а не декларация
----------------------------------------------
1. **Компоненты установлены по данным, а не назначены**: боковые погоны
   фр. 240-290 (`avt:F32`) и фр. 290-350 (`avt:F30`), качество каждого --
   отдельная точка отбора ЛИМС. Соответствие подтверждено тем, что
   показатели смеси лежат строго между компонентами, а доля, при которой
   правила смешения воспроизводят смесь, совпадает с фактическим
   отношением расходов. Независимое подтверждение из выданных материалов:
   производственная ВАК-формула `AVT6:240-350:D15` содержит ровно
   `F30/(F32+F30)`.
2. **Правила смешения -- физика и литература**, а не подгонка
   (`blending/rules.py`).
3. **Каждое правило проверено вне выборки** forward-chaining бэктестом
   (`scripts/compute_blend_model.py` -> `config/blend_model.yaml`) и
   используется, только если оно устойчиво лучше константы. Правила,
   не прошедшие проверку (плотность, температура помутнения), в контур
   НЕ подключаются -- агент по ним отказывается оценивать.
4. **Влияние на качество замкнуто до нормируемого показателя продукта**:
   доли -> показатель сырья гидроочистки -> показатель товарного ДТ
   через измеренный на истории коэффициент переноса.
5. **Σ долей = 100 % соблюдается по построению**: кандидат перераспределяет
   доли при неизменном суммарном расходе пула, поэтому ограничение не
   "проверяется постфактум", а не может быть нарушено. Guard всё равно
   проверяет его независимо.

Чего агент НЕ делает: не управляет долями товарного блендинга
(присадки, вовлечение сторонних компонентов) -- в выданных данных нет
ни их расходов, ни качества, и выдумывать их агент не станет.
"""

from __future__ import annotations

from dataclasses import dataclass

from neftekod_mas.blending.rules import (
    Component,
    blend_cloud_point,
    blend_density,
    blend_distillation,
    blend_mass_linear,
    volume_fractions,
)
from neftekod_mas.schemas import (
    BlendAssessment,
    BlendComponentState,
    ConfidenceLevel,
    ProcessState,
)

# Показатели, которые агент умеет считать, и правило для каждого.
_CURVE_TARGET = {"t50_c": 50.0, "t90_c": 90.0, "t95_c": 95.0}
_MIN_FLOW_TPH = 1.0  # ниже -- поток остановлен или брак КИП


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    description: str
    flow_tag: str
    lims_point: str
    density_kg_m3: float
    sulfur_pct_mass: float | None


class BlendingAgent:
    """Читает `config/blend_model.yaml`; ничего не обучает в рантайме."""

    def __init__(self, blend_model: dict, lims_curves: dict[str, dict[str, float]] | None = None):
        self.model = blend_model
        self.components = [
            ComponentSpec(
                key=key,
                description=block["description"],
                flow_tag=block["flow_tag"],
                lims_point=block["lims_point"],
                density_kg_m3=float(block["density_kg_m3"]),
                sulfur_pct_mass=self._component_sulfur(key),
            )
            for key, block in blend_model.get("components", {}).items()
        ]
        self.validation = blend_model.get("validation", {})
        self.sensitivity = blend_model.get("share_sensitivity", {})
        self.propagation = blend_model.get("feed_to_product_propagation", {})
        # Типовые кривые разгонки компонентов -- агрегированная статистика
        # из того же конфига (медианы ЛИМС, см. compute_blend_model.py).
        # Агент никогда не читает сырые файлы сам; при необходимости
        # кривые можно подменить снаружи (свежий ЛИМС по компоненту).
        self.lims_curves = lims_curves if lims_curves is not None else {
            key: {k: v for k, v in block.items() if not k.endswith("__n")}
            for key, block in blend_model.get("typical_component_curves", {}).items()
        }

    # -- вспомогательное ------------------------------------------------

    def _component_sulfur(self, key: str) -> float | None:
        cs = self.model.get("component_sulfur", {})
        if cs.get("status") != "ok":
            return None
        value = cs.get(f"{key}_pct_mass")
        return float(value) if value is not None else None

    def supported_metrics(self) -> list[str]:
        """Показатели, по которым модель смешения принята бэктестом."""
        return sorted(m for m, v in self.validation.items() if v.get("accepted"))

    def _flows(self, state: ProcessState) -> dict[str, float] | None:
        flows = {}
        for spec in self.components:
            reading = state.kip.get(spec.flow_tag)
            if reading is None or reading.value is None:
                return None
            if not (reading.value > _MIN_FLOW_TPH):
                return None
            flows[spec.key] = float(reading.value)
        return flows or None

    def _as_components(self, flows: dict[str, float]) -> list[Component]:
        total = sum(flows.values())
        return [
            Component(spec.key, flows[spec.key] / total, spec.density_kg_m3)
            for spec in self.components
        ]

    # -- оценка текущего состояния пула ---------------------------------

    def assess(self, state: ProcessState) -> BlendAssessment:
        flows = self._flows(state)
        if flows is None:
            return BlendAssessment(
                decision_at=state.decision_at,
                components=[],
                total_flow_t_h=0.0,
                share_sum_pct=0.0,
                blend_quality={},
                supported_metrics=[],
                available=False,
                unavailable_reason=(
                    "Нет достоверных расходов компонентов пула "
                    + ", ".join(spec.flow_tag for spec in self.components)
                ),
                confidence=ConfidenceLevel.REFUSE,
            )

        comps = self._as_components(flows)
        vol = volume_fractions(comps)
        total = sum(flows.values())
        component_states = [
            BlendComponentState(
                key=spec.key,
                description=spec.description,
                flow_tag=spec.flow_tag,
                flow_t_h=round(flows[spec.key], 3),
                mass_share_pct=round(100.0 * c.mass_fraction, 3),
                volume_share_pct=round(100.0 * v, 3),
                density_kg_m3=spec.density_kg_m3,
                sulfur_pct_mass=spec.sulfur_pct_mass,
            )
            for spec, c, v in zip(self.components, comps, vol, strict=True)
        ]

        return BlendAssessment(
            decision_at=state.decision_at,
            components=component_states,
            total_flow_t_h=round(total, 3),
            share_sum_pct=round(sum(s.mass_share_pct for s in component_states), 6),
            blend_quality=self.blend_quality(comps),
            supported_metrics=self.supported_metrics(),
            available=True,
            confidence=ConfidenceLevel.MEDIUM,
        )

    # -- качество смеси --------------------------------------------------

    def blend_quality(self, comps: list[Component]) -> dict[str, float]:
        """Показатели смеси (сырья гидроочистки) по принятым правилам.
        Показатели, не прошедшие бэктест, сюда не попадают."""
        out: dict[str, float] = {}
        accepted = set(self.supported_metrics())
        curves = [self.lims_curves.get(spec.key) for spec in self.components]

        if all(c for c in curves):
            targets = tuple(_CURVE_TARGET[m] for m in _CURVE_TARGET if m in accepted)
            if targets:
                mixed = blend_distillation(comps, [dict(c) for c in curves], targets=targets)
                for metric, q in _CURVE_TARGET.items():
                    if metric in accepted and q in mixed:
                        out[metric] = round(mixed[q] + self._bias(metric), 3)

        if "density_kg_m3" in accepted:
            out["density_kg_m3"] = round(blend_density(comps) + self._bias("density_kg_m3"), 3)

        sulfurs = [spec.sulfur_pct_mass for spec in self.components]
        if all(s is not None for s in sulfurs):
            out["feed_sulfur_pct_mass"] = round(blend_mass_linear(comps, sulfurs), 4)

        clouds = [self.lims_curves.get(spec.key, {}).get("CloudPoint") for spec in self.components]
        if "cloud_point_c" in accepted and all(c is not None for c in clouds):
            out["cloud_point_c"] = round(blend_cloud_point(comps, clouds) + self._bias("cloud_point_c"), 3)

        return out

    def _bias(self, metric: str) -> float:
        return float(self.validation.get(metric, {}).get("bias", 0.0) or 0.0)

    # -- эффект изменения долей -----------------------------------------

    def predict_product_effect(
        self, state: ProcessState, new_flows: dict[str, float]
    ) -> dict[str, float]:
        """Эффект перераспределения долей на нормируемые показатели ПРОДУКТА.

        Считается как приращение: показатель смеси при новых долях минус
        при текущих, перенесённое на продукт измеренным коэффициентом.
        Уровень показателя продукта берётся Агентом качества из самой
        точной доступной оценки -- здесь возвращается только сдвиг, чтобы
        систематическая ошибка модели смешения не попадала в сравнение
        с пределом (тот же delta-метод, что в quality_agent.predict_effect).
        """
        flows = self._flows(state)
        if flows is None:
            return {}
        base = self.blend_quality(self._as_components(flows))
        new = self.blend_quality(self._as_components(new_flows))

        deltas: dict[str, float] = {}
        for metric, slope_block in self.propagation.items():
            if slope_block.get("status") != "ok" or metric not in base or metric not in new:
                continue
            deltas[metric] = float(slope_block["slope"]) * (new[metric] - base[metric])
        return deltas

    # -- кандидатные изменения долей ------------------------------------

    def share_candidates(self, state: ProcessState, steps_pct=(-6.0, -4.5, -3.0, -1.5, 1.5, 3.0, 4.5, 6.0)):
        """Перераспределение долей при НЕИЗМЕННОМ суммарном расходе пула.

        Шаг задаётся в процентных пунктах доли тяжёлого компонента.
        Кандидат отбрасывается, если выводит долю за исторически
        наблюдавшийся диапазон (p05..p95 из `config/blend_model.yaml`) --
        то же правило границ, что и для остальных рычагов (§7).
        """
        flows = self._flows(state)
        if flows is None or len(self.components) != 2:
            return []
        light, heavy = self.components[0], self.components[1]
        total = sum(flows.values())
        w_heavy = flows[heavy.key] / total

        obs = self.model.get("observed_shares", {})
        lo, hi = float(obs.get("p05", 0.0)), float(obs.get("p95", 1.0))

        out = []
        for step in steps_pct:
            w_new = w_heavy + step / 100.0
            if not (lo <= w_new <= hi):
                continue
            new_flows = {heavy.key: total * w_new, light.key: total * (1.0 - w_new)}
            if any(v <= _MIN_FLOW_TPH for v in new_flows.values()):
                continue
            out.append({
                "heavy_share_pct": round(100.0 * w_new, 3),
                "current_heavy_share_pct": round(100.0 * w_heavy, 3),
                "flows": {k: round(v, 3) for k, v in new_flows.items()},
                "share_sum_pct": 100.0,
                "product_effect": self.predict_product_effect(state, new_flows),
            })
        return out
