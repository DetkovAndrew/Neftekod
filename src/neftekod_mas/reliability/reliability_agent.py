"""
Агент надёжности (ARCHITECTURE.md §6.3).

Оценивает тяжесть режима установки 24-2000 по прокси-показателям --
перепад давления на реакторе Р-202 (`P8`, индикатор закоксовывания
катализатора: чем выше перепад при том же расходе, тем сильнее
забита насадка) и температуры реакторной секции (`T5`, `T6`, `T11`).

Границы -- НЕ паспортные пределы оборудования (их не передали, см.
Q&A-сессию и ARCHITECTURE.md §13), а консервативные перцентили истории
(config/reliability_bounds.yaml, посчитанные scripts/compute_reliability_bounds.py
на статистике, а не на обученной модели). Это явно проговорённое
допущение, а не скрытая точность.

Две вещи агент берёт уже не из перцентилей, а из подтверждённых данными
свойств установки (`config/equipment_limits.yaml`, §6.3.1):

1. **Цензурирование главного индикатора.** Перепад давления `P8` упирается
   в 0.25 МПа: в верхнем проценте у него всего 11 разных значений на 1814
   точек -- это предел шкалы датчика, а не реальный максимум. Значение на
   упоре означает "не меньше 0.25", а не "ровно 0.25", и именно там, где
   риск важнее всего. Поэтому такой отсчёт не оценивается как обычное
   измерение: фактор помечается цензурированным и получает консервативную
   тяжесть не ниже критической.
2. **Ресурс катализатора.** Скорость дезактивации (рост температуры,
   нужной для удержания той же серы) измерена по кампаниям между
   остановами: около +1 °C/мес, устойчиво по всем кампаниям. Отсюда
   остаток температурного запаса до потолка и срок, на который его
   хватит, -- это ресурсный показатель, а не перцентиль.
"""

from __future__ import annotations

from neftekod_mas.schemas import EquipmentRiskAssessment, ProcessState, RiskClass, RiskFactor

# Вес каждого признака в итоговом индексе тяжести режима -- ΔP считается
# главным индикатором (закоксовывание необратимо и критично для ресурса
# катализатора), температуры -- вторичным (обратимый режимный параметр).
FACTOR_WEIGHTS: dict[str, float] = {"P8": 0.5, "T5": 0.17, "T6": 0.17, "T11": 0.16}

# Относительный допуск, в пределах которого отсчёт считается стоящим на
# упоре. 0.5 % от величины предела -- шум АЦП, а не отход от упора.
SATURATION_TOLERANCE_FRAC = 0.005
# Тяжесть, приписываемая цензурированному отсчёту: истинное значение
# может быть сколь угодно выше предела шкалы, поэтому режим считается
# как минимум критическим, а не "ровно на пределе".
CENSORED_SEVERITY = 1.0

FACTOR_DESCRIPTIONS: dict[str, str] = {
    "P8": "Перепад давления на реакторе Р-202 -- прокси закоксовывания катализатора",
    "T5": "Температура ГСС на выходе Р-201 (вход в реакторную секцию)",
    "T6": "Температура ГСС на входе Р-202",
    "T11": "Температура на выходе из Р-202",
}


def _tag_severity(value: float, bounds: dict) -> float:
    """0 на уровне p90 истории и ниже, линейно растёт до 1 на p99,
    экстраполируется выше 1, если значение превышает исторический максимум."""
    p90, p99, p100 = bounds["0.9"], bounds["0.99"], bounds["1.0"]
    if value <= p90:
        return 0.0
    span = max(p99 - p90, 1e-9)
    sev = (value - p90) / span
    if value > p100:
        sev = max(sev, 1.0 + (value - p100) / max(p100 - p99, 1e-9))
    return sev


class ReliabilityAgent:
    def __init__(self, bounds: dict, equipment_limits: dict | None = None):
        self.bounds = {k: v for k, v in bounds.items() if k != "_meta"}
        self.is_assumption = bounds.get("_meta", {}).get("is_assumption", True)
        self.assumption_note = bounds.get("_meta", {}).get("note", "")
        limits = equipment_limits or {}
        self.saturation = {
            tag: block for tag, block in (limits.get("saturation") or {}).items()
            if block.get("looks_saturated")
        }
        self.deactivation = limits.get("catalyst_deactivation") or {}
        self.temp_ceiling = (limits.get("reactor_temperature_ceiling_c") or {}).get("value")

    def _censored(self, tag: str, value: float) -> bool:
        block = self.saturation.get(tag)
        if not block:
            return False
        ceiling = float(block["observed_max"])
        return value >= ceiling * (1.0 - SATURATION_TOLERANCE_FRAC)

    def _catalyst_factor(self, state: ProcessState) -> RiskFactor | None:
        """Остаток температурного запаса катализатора и срок его исчерпания.

        Это единственный фактор агента, выраженный в ресурсе, а не в
        перцентиле: скорость дезактивации измерена по кампаниям истории.
        Вклад в индекс тяжести растёт по мере исчерпания запаса и
        достигает 1.0, когда запаса не осталось.
        """
        rate = self.deactivation.get("median_c_per_month")
        reading = state.kip.get("242000:T5")
        if not rate or rate <= 0 or reading is None or self.temp_ceiling is None:
            return None
        headroom = float(self.temp_ceiling) - float(reading.value)
        months = headroom / float(rate)
        contribution = min(max(1.0 - headroom / max(float(self.temp_ceiling) - 350.0, 1e-6), 0.0), 1.0)
        return RiskFactor(
            tag_id="242000:T5",
            description=(
                f"Ресурс катализатора: запас по температуре {headroom:.1f} °C до потолка "
                f"{self.temp_ceiling:.1f} °C при дезактивации {rate:+.2f} °C/мес "
                f"-- хватит примерно на {months:.1f} мес"
            ),
            contribution=round(contribution, 3),
            is_assumption=True,
            affects_index=False,
            assumption_note=(
                "Потолок температуры -- 99-й перцентиль истории, а не паспортный предел "
                "end-of-run; скорость дезактивации измерена по кампаниям между остановами."
            ),
        )

    def assess(self, state: ProcessState) -> EquipmentRiskAssessment:
        factors: list[RiskFactor] = []
        weighted_sum = 0.0
        weight_total = 0.0

        missing_inputs: list[str] = []
        for tag, weight in FACTOR_WEIGHTS.items():
            qid = f"242000:{tag}"
            reading = state.kip.get(qid)
            if reading is None or tag not in self.bounds:
                missing_inputs.append(qid)
                continue
            sev = _tag_severity(reading.value, self.bounds[tag])
            description = FACTOR_DESCRIPTIONS.get(tag, tag)
            note = self.assumption_note if self.is_assumption else None
            if self._censored(tag, reading.value):
                # Отсчёт на упоре шкалы означает "не меньше", а не "ровно":
                # истинное значение может быть выше, и занижать тяжесть
                # режима здесь недопустимо.
                sev = max(sev, CENSORED_SEVERITY)
                description += " -- ОТСЧЁТ НА УПОРЕ ШКАЛЫ (цензурирован)"
                note = (
                    f"Значение {reading.value:.4g} достигло предела шкалы "
                    f"{self.saturation[tag]['observed_max']}: истинная величина может быть выше. "
                    "Тяжесть режима принята консервативно (config/equipment_limits.yaml)."
                )
            factors.append(
                RiskFactor(
                    tag_id=qid,
                    description=description,
                    contribution=round(min(sev, 1.5), 3),
                    is_assumption=True if note else self.is_assumption,
                    assumption_note=note,
                )
            )
            weighted_sum += weight * sev
            weight_total += weight

        catalyst = self._catalyst_factor(state)
        if catalyst is not None:
            factors.append(catalyst)

        coverage = weight_total / sum(FACTOR_WEIGHTS.values())
        if missing_inputs:
            return EquipmentRiskAssessment(
                decision_at=state.decision_at, severity_index=0.0,
                risk_class=RiskClass.UNKNOWN, factors=factors, hard_stop=True,
                coverage=coverage, missing_inputs=missing_inputs,
            )
        weighted_avg = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        # Итоговый индекс -- максимум из взвешенного среднего и худшего
        # отдельного фактора: единичный фактор на пределе (напр. только
        # ΔP резко выросло) не должен "размываться" усреднением с
        # нормальными остальными -- по аналогии с тем, как в Ta & Liu
        # (2027) именно отдельный bed-level exotherm-constraint, а не
        # агрегат, определял связывающее ограничение (§3.3.2-3.3.4).
        # В индекс тяжести режима входят только факторы, помеченные
        # affects_index. Ресурс катализатора сюда не входит намеренно:
        # исчерпание температурного запаса -- повод планировать перегрузку,
        # а не признак того, что режим тяжёлый прямо сейчас. Если смешать
        # их в одном числе, система начнёт отказывать в оптимизации из-за
        # планового события, до которого ещё месяцы.
        max_single = max((f.contribution for f in factors if f.affects_index), default=0.0)
        severity_index = max(0.0, min(max(weighted_avg, max_single), 1.5))

        if severity_index >= 1.0:
            risk_class = RiskClass.CRITICAL
        elif severity_index >= 0.6:
            risk_class = RiskClass.HIGH
        elif severity_index >= 0.3:
            risk_class = RiskClass.MEDIUM
        else:
            risk_class = RiskClass.LOW

        hard_stop = risk_class in (RiskClass.HIGH, RiskClass.CRITICAL)

        return EquipmentRiskAssessment(
            decision_at=state.decision_at,
            severity_index=round(severity_index, 3),
            risk_class=risk_class,
            factors=factors,
            hard_stop=hard_stop,
            coverage=coverage,
            missing_inputs=[],
        )
