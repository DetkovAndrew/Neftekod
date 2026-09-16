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
"""

from __future__ import annotations

from neftekod_mas.schemas import EquipmentRiskAssessment, ProcessState, RiskClass, RiskFactor

# Вес каждого признака в итоговом индексе тяжести режима -- ΔP считается
# главным индикатором (закоксовывание необратимо и критично для ресурса
# катализатора), температуры -- вторичным (обратимый режимный параметр).
FACTOR_WEIGHTS: dict[str, float] = {"P8": 0.5, "T5": 0.17, "T6": 0.17, "T11": 0.16}

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
    def __init__(self, bounds: dict):
        self.bounds = {k: v for k, v in bounds.items() if k != "_meta"}
        self.is_assumption = bounds.get("_meta", {}).get("is_assumption", True)
        self.assumption_note = bounds.get("_meta", {}).get("note", "")

    def assess(self, state: ProcessState) -> EquipmentRiskAssessment:
        factors: list[RiskFactor] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for tag, weight in FACTOR_WEIGHTS.items():
            qid = f"242000:{tag}"
            reading = state.kip.get(qid)
            if reading is None or tag not in self.bounds:
                continue
            sev = _tag_severity(reading.value, self.bounds[tag])
            factors.append(
                RiskFactor(
                    tag_id=qid,
                    description=FACTOR_DESCRIPTIONS.get(tag, tag),
                    contribution=round(min(sev, 1.5), 3),
                    is_assumption=self.is_assumption,
                    assumption_note=self.assumption_note if self.is_assumption else None,
                )
            )
            weighted_sum += weight * sev
            weight_total += weight

        weighted_avg = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        # Итоговый индекс -- максимум из взвешенного среднего и худшего
        # отдельного фактора: единичный фактор на пределе (напр. только
        # ΔP резко выросло) не должен "размываться" усреднением с
        # нормальными остальными -- по аналогии с тем, как в Ta & Liu
        # (2027) именно отдельный bed-level exotherm-constraint, а не
        # агрегат, определял связывающее ограничение (§3.3.2-3.3.4).
        max_single = max((f.contribution for f in factors), default=0.0)
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
        )
