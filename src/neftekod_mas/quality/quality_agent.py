"""
Агент качества (ARCHITECTURE.md §6.2).

Оценивает нормируемые показатели товарного ДТ (после гидроочистки --
до блендинга, для которого исторических данных нет, см. §13) по
приоритету источников ЛИМС -> ПАК -> ВАК-формула (ТЗ, "Правила работы
с источниками качества"), с явным учётом возраста и понижением
доверия при деградации источника.

Контрольная точка. В ЛИМС финальное качество гидроочищенного ДТ
находится в точке отбора 2 установки "Гидроочистка" (после реактора,
готовый продукт) -- подтверждено Q&A-сессией: "по 242000 всё понятнее,
чем АВТ: точки отбора перед установкой и после установки". Точка 1 --
это вход в гидроочистку (ФРАКЦ_ДИЗ), точка 2 -- выход (готовое ДТ).
Обратите внимание на реальную опечатку в исходных данных (двойная
точка после 'Гидроочистка'..') -- она воспроизведена дословно, иначе
джойн с ЛИМС не сработает.
"""

from __future__ import annotations

from dataclasses import dataclass

from neftekod_mas.quality import vak_formulas as vak
from neftekod_mas.schemas import (
    ConfidenceLevel,
    DataSource,
    ProcessState,
    QualityAssessment,
    QualityMetricEstimate,
    RiskClass,
    SpecViolationRisk,
)

GODT_POINT2 = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"

# Критический порог возраста ЛИМС, после которого показатель считается
# ненадёжным настолько, что System обязана понизить confidence до REFUSE,
# если это единственный источник (нет ПАК/ВАК-фолбэка) -- см. ARCHITECTURE.md §13.
CRITICAL_LIMS_AGE_MINUTES = 7 * 24 * 60  # 7 суток


@dataclass
class MetricSpec:
    metric: str
    unit: str
    lims_param: str | None
    pak_param: str | None  # уже с префиксом "24-2000:..."
    vak_formula: str | None  # ключ в vak.HT242000_FORMULAS
    vak_needs_lims: bool  # формула сама использует LIMS.* как вход


# Явный реестр -- какой из 4 показателей чем поддержан. Обратите внимание:
# для серы и цетанового числа ВАК-формулы НЕТ вовсе (в переданном наборе
# формул её не оказалось) -- честно отражено как vak_formula=None, а не
# скрыто "заглушкой". Для серы есть независимый ПАК; для цетанового
# числа нет НИ ПАК, НИ формулы -- это самый хрупкий с точки зрения
# доступности показатель в системе, и это важно показывать оператору.
METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("sulfur_mg_kg", "мг/кг", "Mg.Sulfur", "24-2000:Mg.Sulfur", None, False),
    MetricSpec("t95_c", "°C", "95%.T", None, "24-2000:GODT:T95", True),
    MetricSpec("cetane_number", "ед.цет.ч.", "CetaneNumber", None, None, False),
    MetricSpec("cfpp_c", "°C", "CFPP", None, "24-2000:GODT:CFPP", False),
    MetricSpec("density_kg_m3", "кг/м3", "D15", "24-2000:D15", "24-2000:GODT:D15", True),
]


def _ht_tag_dict(state: ProcessState) -> dict[str, float]:
    prefix = "242000:"
    return {
        qid[len(prefix):]: r.value
        for qid, r in state.kip.items()
        if qid.startswith(prefix)
    }


def _confidence_from_age(age_minutes: float, stale_minutes: float) -> ConfidenceLevel:
    if age_minutes <= stale_minutes:
        return ConfidenceLevel.HIGH
    if age_minutes <= CRITICAL_LIMS_AGE_MINUTES:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


class QualityAgent:
    def __init__(self, hard_constraints: dict, stale_lims_minutes: float = 24 * 60):
        self.hard_constraints = hard_constraints
        self.stale_lims_minutes = stale_lims_minutes

    def assess(self, state: ProcessState) -> QualityAssessment:
        estimates: list[QualityMetricEstimate] = []

        for spec in METRIC_SPECS:
            est = self._estimate_metric(spec, state)
            if est is not None:
                estimates.append(est)

        violations = self._check_violations(estimates)

        if not estimates:
            overall = ConfidenceLevel.REFUSE
        else:
            levels = [e.confidence for e in estimates]
            if ConfidenceLevel.LOW in levels:
                overall = ConfidenceLevel.LOW
            elif ConfidenceLevel.MEDIUM in levels:
                overall = ConfidenceLevel.MEDIUM
            else:
                overall = ConfidenceLevel.HIGH
            # если хотя бы один ОБЯЗАТЕЛЬНЫЙ (жёстко нормируемый) показатель
            # не удалось оценить вообще -- это повод для отказа, не для
            # "среднего" доверия по оставшимся.
            covered = {e.metric for e in estimates}
            required = {m for m in self.hard_constraints.get("product_diesel", {})}
            missing_required = {m for m in required if m in {s.metric for s in METRIC_SPECS}} - covered
            if missing_required:
                overall = ConfidenceLevel.REFUSE

        return QualityAssessment(
            decision_at=state.decision_at,
            current=estimates,
            violations=violations,
            overall_confidence=overall,
        )

    def _estimate_metric(self, spec: MetricSpec, state: ProcessState) -> QualityMetricEstimate | None:
        # 1) ЛИМС -- контрольный факт по приоритету ТЗ
        if spec.lims_param is not None:
            point_id = f"{GODT_POINT2}|{spec.lims_param}"
            lims = state.lab_points.get(point_id)
            if lims is not None:
                return QualityMetricEstimate(
                    metric=spec.metric,
                    value=lims.value,
                    unit=spec.unit,
                    source=DataSource.LIMS,
                    age_minutes=lims.age_minutes,
                    confidence=_confidence_from_age(lims.age_minutes, self.stale_lims_minutes),
                )

        # 2) ПАК -- поточный, ниже приоритетом, но не требует расчёта
        if spec.pak_param is not None:
            pak = state.lab_points.get(spec.pak_param)
            if pak is not None:
                return QualityMetricEstimate(
                    metric=spec.metric,
                    value=pak.value,
                    unit=spec.unit,
                    source=DataSource.PAK,
                    age_minutes=pak.age_minutes,
                    confidence=ConfidenceLevel.HIGH if pak.age_minutes <= 60 else ConfidenceLevel.MEDIUM,
                )

        # 3) ВАК-формула -- расчётный слой, самый низкий приоритет
        if spec.vak_formula is not None:
            ht_tags = _ht_tag_dict(state)
            fn = vak.HT242000_FORMULAS[spec.vak_formula]
            try:
                if spec.vak_needs_lims:
                    # формула использует последнее ЛИМС-значение как один
                    # из входов (см. модуль vak_formulas) -- если его нет
                    # вообще, формула недоступна.
                    lims_key = "24-2000.Pipeline.D15" if spec.metric == "density_kg_m3" else "95%.T"
                    point_id = f"{GODT_POINT2}|{spec.lims_param}"
                    lims = state.lab_points.get(point_id)
                    if lims is None:
                        return None
                    value = fn(ht_tags, {lims_key: lims.value})
                    confidence = ConfidenceLevel.LOW  # частично опирается на потенциально старый ЛИМС
                else:
                    value = fn(ht_tags)
                    confidence = ConfidenceLevel.MEDIUM
            except KeyError:
                return None  # не хватает тегов КИП -- честно не оцениваем
            return QualityMetricEstimate(
                metric=spec.metric,
                value=value,
                unit=spec.unit,
                source=DataSource.VAK_FORMULA,
                age_minutes=None,
                confidence=confidence,
            )

        return None

    def _check_violations(self, estimates: list[QualityMetricEstimate]) -> list[SpecViolationRisk]:
        violations: list[SpecViolationRisk] = []
        limits = self.hard_constraints.get("product_diesel", {})
        by_metric = {e.metric: e for e in estimates}

        for metric, cfg in limits.items():
            if cfg.get("limit") is None:
                continue
            est = by_metric.get(metric)
            if est is None:
                continue
            op = cfg["op"]
            limit = float(cfg["limit"])
            margin = (limit - est.value) if op == "<=" else (est.value - limit)
            if margin < 0:
                risk = RiskClass.CRITICAL
            elif margin < 0.1 * abs(limit):
                risk = RiskClass.HIGH
            elif margin < 0.25 * abs(limit):
                risk = RiskClass.MEDIUM
            else:
                risk = RiskClass.LOW
            violations.append(SpecViolationRisk(metric=metric, limit=limit, margin=margin, risk_class=risk))

        return violations
