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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neftekod_mas.quality import astm_correlations as astm
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

if TYPE_CHECKING:
    from neftekod_mas.quality.soft_sensors import SoftSensorService

GODT_POINT2 = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"
# Точка 1 -- вход в гидроочистку (ФРАКЦ_ДИЗ), содержит Mass.Sulfur
# (% масс.) -- сера СЫРЬЯ, нужна для расчёта severity ГДС
# (literature_proxies.hds_severity), см. ARCHITECTURE.md §6.4.
HT_FEED_POINT = "Установка 'Гидроочистка'. Точка отбора '1'. Продукт 'ФРАКЦ_ДИЗ'."

# Критический порог возраста ЛИМС, после которого показатель считается
# ненадёжным настолько, что System обязана понизить confidence до REFUSE,
# если это единственный источник (нет ПАК/ВАК-фолбэка) -- см. ARCHITECTURE.md §13.
CRITICAL_LIMS_AGE_MINUTES = 7 * 24 * 60  # 7 суток

# Замер ЛИМС моложе этого -- фактически текущее измерение, он приоритетнее
# soft-sensor'а. Старше -- soft-sensor точнее: по forward-chaining CV
# (config/soft_sensor_selection.yaml) "последнее значение ЛИМС" как прогноз
# на момент следующей пробы хуже якорной модели (сера 5.6 против 1.2 мг/кг,
# T95 4.9 против 4.5°C, плотность 1.33 против 1.14 кг/м3), а сама модель уже
# включает этот ЛИМС через поправку смещения.
FRESH_LIMS_OVERRIDES_SOFT_SENSOR_MINUTES = 60

# Классы риска по вероятности превышения предела, когда ошибка оценки
# измерена (typical_error = MAE по бэктесту/CV). Ошибка считается нормальной:
# sigma = MAE * sqrt(pi/2). Фиксированные risk_margin_* из hard_constraints.yaml
# остаются только для оценок без измеренной ошибки (свежий ЛИМС).
# Пороги -- допущение; обоснование: завод штатно держит серу 7.7-9.2 мг/кг при
# пределе 10, и фиксированный запас 2.5 мг/кг объявлял "риском" ~80% суток.
EXCEED_P_HIGH = 0.20
EXCEED_P_MEDIUM = 0.05
_MAE_TO_SIGMA = math.sqrt(math.pi / 2.0)


def exceed_probability(margin: float, typical_error: float) -> float:
    sigma = max(typical_error * _MAE_TO_SIGMA, 1e-9)
    return 0.5 * math.erfc(margin / (sigma * math.sqrt(2.0)))


@dataclass
class MetricSpec:
    metric: str
    unit: str
    lims_param: str | None
    pak_param: str | None  # уже с префиксом "24-2000:..."
    vak_formula: str | None  # ключ в vak.HT242000_FORMULAS
    vak_needs_lims: bool  # формула сама использует LIMS.* как вход


# Явный реестр -- какой из 4 показателей чем поддержан. Для серы есть
# независимый ПАК; для цетанового числа нет НИ ПАК, НИ производственной
# ВАК-формулы (vak_formula=None, честно, не заглушка) -- но есть
# опубликованный отраслевой стандарт ASTM D976 (astm_correlations.py),
# подключённый отдельным механизмом ниже (_estimate_cetane_via_d976),
# т.к. ему нужны сразу ДВА других показателя (D15 и T50), а не один
# тег КИП, как у остальных формул -- не укладывается в единообразный
# паттерн MetricSpec.vak_formula.
METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("sulfur_mg_kg", "мг/кг", "Mg.Sulfur", "24-2000:Mg.Sulfur", None, False),
    MetricSpec("t95_c", "°C", "95%.T", None, "24-2000:GODT:T95", True),
    MetricSpec("cetane_number", "ед.цет.ч.", "CetaneNumber", None, None, False),
    MetricSpec("cfpp_c", "°C", "CFPP", None, "24-2000:GODT:CFPP", False),
    MetricSpec("density_kg_m3", "кг/м3", "D15", "24-2000:D15", "24-2000:GODT:D15", True),
]

# Внутренний, не публикуемый в QualityAssessment.current спецификатор --
# T50 не является нормируемым показателем сам по себе, нужен только как
# вход формулы D976 (см. ниже).
_T50_SPEC = MetricSpec("t50_c", "°C", "50%.T", None, "24-2000:GODT:T50", False)
_DENSITY_SPEC = next(s for s in METRIC_SPECS if s.metric == "density_kg_m3")


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
    def __init__(
        self,
        hard_constraints: dict,
        stale_lims_minutes: float = 24 * 60,
        formula_accuracy: dict | None = None,
        astm_accuracy: dict | None = None,
        soft_sensors: "SoftSensorService | None" = None,
    ):
        self.hard_constraints = hard_constraints
        self.stale_lims_minutes = stale_lims_minutes
        # Бэктест точности ВАК-формул против ЛИМС (scripts/compute_vak_formula_accuracy.py,
        # ARCHITECTURE.md §5.3) -- статистика, не обучение. Опционально: без
        # него формульные оценки просто не получают typical_error/доп.
        # понижения confidence, остальной пайплайн не ломается.
        self.formula_accuracy = {k: v for k, v in (formula_accuracy or {}).items() if k != "_meta"}
        # Бэктест ASTM D976 против реальных CetaneNumber этого завода
        # (scripts/compute_astm_cetane_accuracy.py) -- также опционально.
        self.astm_accuracy = {k: v for k, v in (astm_accuracy or {}).items() if k != "_meta"}
        # Soft-sensor'ы, отобранные scripts/benchmark_anchored.py (quality/soft_sensors.py).
        # Опционально: без них приоритет источников прежний (ЛИМС -> ПАК -> формула).
        self.soft_sensors = soft_sensors

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
        lims = None
        if spec.lims_param is not None:
            lims = state.lab_points.get(f"{GODT_POINT2}|{spec.lims_param}")

        # 1) Soft-sensor -- если для показателя он отобран и ЛИМС не "только что"
        # измерен (см. FRESH_LIMS_OVERRIDES_SOFT_SENSOR_MINUTES).
        lims_is_current = lims is not None and lims.age_minutes <= FRESH_LIMS_OVERRIDES_SOFT_SENSOR_MINUTES
        if not lims_is_current:
            soft = self._estimate_via_soft_sensor(spec, state)
            if soft is not None:
                return soft

        # 2) ЛИМС -- контрольный факт по приоритету ТЗ
        if lims is not None:
            return QualityMetricEstimate(
                metric=spec.metric,
                value=lims.value,
                unit=spec.unit,
                source=DataSource.LIMS,
                age_minutes=lims.age_minutes,
                confidence=_confidence_from_age(lims.age_minutes, self.stale_lims_minutes),
                # ошибка "последнего ЛИМС" как прогноза на сейчас -- по тому же CV,
                # только если ЛИМС не свежий (свежий -- это измерение, а не прогноз)
                typical_error=(
                    None if lims_is_current or self.soft_sensors is None
                    else self.soft_sensors.lims_typical_error(spec.metric)
                ),
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

        # 3a) Цетановое число -- особый случай: нет ни ПАК, ни
        # производственной ВАК-формулы, но есть опубликованный
        # отраслевой стандарт ASTM D976 (нужны сразу D15 И T50).
        if spec.metric == "cetane_number":
            return self._estimate_cetane_via_d976(state, for_prediction=False)

        # 3b) ВАК-формула -- расчётный слой, самый низкий приоритет
        return self._estimate_via_formula(spec, state)

    def _estimate_via_soft_sensor(self, spec: MetricSpec, state: ProcessState) -> QualityMetricEstimate | None:
        if self.soft_sensors is None:
            return None
        est = self.soft_sensors.estimate(spec.metric, state)
        if est is None:
            return None
        margin_medium = self.hard_constraints.get("product_diesel", {}).get(spec.metric, {}).get("risk_margin_medium")
        last_age = (
            None if est.last_lims_at is None
            else (state.decision_at - est.last_lims_at).total_seconds() / 60.0
        )
        if est.n_bias_samples < 3 or last_age is None or last_age > CRITICAL_LIMS_AGE_MINUTES:
            confidence = ConfidenceLevel.LOW  # поправка не подкреплена свежей лабораторией
        elif margin_medium is not None and est.typical_error >= margin_medium:
            confidence = ConfidenceLevel.LOW
        elif margin_medium is not None and est.typical_error < 0.5 * margin_medium:
            confidence = ConfidenceLevel.HIGH
        else:
            confidence = ConfidenceLevel.MEDIUM
        return QualityMetricEstimate(
            metric=spec.metric,
            value=est.value,
            unit=spec.unit,
            source=DataSource.SOFT_SENSOR,
            age_minutes=last_age,  # возраст самого свежего ЛИМС в поправке
            confidence=confidence,
            typical_error=est.typical_error,
        )

    def _estimate_cetane_via_d976(self, state: ProcessState, for_prediction: bool) -> QualityMetricEstimate | None:
        """ASTM D976 (astm_correlations.py) -- опубликованный, не
        заводской источник. Нужны ОДНОВРЕМЕННО D15 и T50: для текущей
        оценки берутся по обычному приоритету ЛИМС->ПАК->формула
        (`_estimate_metric`), для прогноза эффекта кандидата -- ТОЛЬКО
        через формулу (`_estimate_via_formula`), по тем же причинам,
        что и в `predict_effect` (ЛИМС/ПАК не реагируют на гипотетическое
        решение)."""
        if for_prediction:
            d15 = self._estimate_via_formula(_DENSITY_SPEC, state)
            t50 = self._estimate_via_formula(_T50_SPEC, state)
        else:
            d15 = self._estimate_metric(_DENSITY_SPEC, state)
            t50 = self._estimate_metric(_T50_SPEC, state)

        if d15 is None or t50 is None or t50.value <= 0 or d15.value <= 0:
            return None

        value = astm.cetane_index_d976(d15.value / 1000.0, t50.value)

        acc = self.astm_accuracy.get("cetane_number_d976")
        typical_error = None
        if acc and acc.get("n", 0) > 0:
            # Коррекция измеренного систематического смещения (bias) --
            # НЕ переобучение формулы, просто вычитание уже посчитанной
            # константы (scripts/compute_astm_cetane_accuracy.py). Без
            # неё цепочка GODT:D15 -> GODT:T50 -> D976 систематически
            # занижала цетановое число (~1 ед.), из-за чего почти все
            # кандидаты ложно отбраковывались как нарушающие предел 51 --
            # найдено на реальном прогоне, см. ARCHITECTURE.md §5.2.
            value -= acc["bias"]
            typical_error = acc.get("std_after_bias_correction", acc["mae"])

        # MEDIUM, а не HIGH: валидированный стандарт (см. astm_correlations.py,
        # бэктест на данных этого завода дал MAE~1.7 при пороге риска 2.0),
        # но не прямое измерение и не заводская формула -- ЛИМС всё равно
        # приоритетнее, если доступен свежий (см. п.1 выше).
        confidence = ConfidenceLevel.MEDIUM
        margin_medium = self.hard_constraints.get("product_diesel", {}).get("cetane_number", {}).get("risk_margin_medium")
        if typical_error is not None and margin_medium is not None and typical_error >= margin_medium:
            confidence = ConfidenceLevel.LOW

        return QualityMetricEstimate(
            metric="cetane_number",
            value=value,
            unit="ед.цет.ч.",
            source=DataSource.PUBLISHED_CORRELATION,
            age_minutes=None,
            confidence=confidence,
            typical_error=typical_error,
        )

    def _estimate_via_formula(self, spec: MetricSpec, state: ProcessState) -> QualityMetricEstimate | None:
        if spec.vak_formula is None:
            return None
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

        typical_error = None
        acc = self.formula_accuracy.get(spec.metric)
        if acc and acc.get("n", 0) > 0:
            # Коррекция измеренного систематического смещения -- см.
            # тот же приём и то же обоснование в _estimate_cetane_via_d976.
            # Особенно важно для CFPP: bias≈-MAE (формула была ПОЧТИ
            # ЦЕЛИКОМ систематической ошибкой, не шумом) -- после
            # коррекции typical_error падает с 11.4°C до остаточного
            # разброса std_after_bias_correction, см. ARCHITECTURE.md §5.3.
            value -= acc["bias"]
            typical_error = acc.get("std_after_bias_correction", acc["mae"])
            margin_medium = self.hard_constraints.get("product_diesel", {}).get(spec.metric, {}).get("risk_margin_medium")
            if margin_medium is not None and typical_error >= margin_medium:
                confidence = ConfidenceLevel.LOW

        return QualityMetricEstimate(
            metric=spec.metric,
            value=value,
            unit=spec.unit,
            source=DataSource.VAK_FORMULA,
            age_minutes=None,
            confidence=confidence,
            typical_error=typical_error,
        )

    def predict_effect(
        self,
        hypothetical_state: ProcessState,
        baseline_state: ProcessState | None = None,
        baseline_quality: QualityAssessment | None = None,
    ) -> list[QualityMetricEstimate]:
        """Прогноз эффекта ГИПОТЕТИЧЕСКОГО состояния (кандидата Агента
        оптимизации) -- принципиально ТОЛЬКО через формульный/ML слой.

        ЛИМС/ПАК описывают уже случившийся факт и не могут "среагировать"
        на предполагаемое, ещё не принятое решение -- использовать их
        текущее значение как прогноз эффекта было бы неявной и неверной
        подменой факта прогнозом. Метрики без формулы/корреляции (сера --
        см. §5.2.1 ARCHITECTURE.md) здесь принципиально отсутствуют:
        система не делает вид, что умеет прогнозировать то, для чего в
        материалах нет расчётной модели. Цетановое число -- ИСКЛЮЧЕНИЕ
        с 2026-09: ASTM D976 (astm_correlations.py) даёт прогноз через
        предсказываемые D15/T50, см. _estimate_cetane_via_d976.

        Приращение (delta-метод). Если переданы baseline_state и
        baseline_quality, прогноз = текущая лучшая оценка (ЛИМС/soft-sensor)
        + [формула(кандидат) - формула(текущее)]. Формула отвечает только
        за НАПРАВЛЕНИЕ и ВЕЛИЧИНУ изменения, а уровень берётся из самой
        точной оценки -- систематическая ошибка формулы (например, CFPP,
        bias ~ -11°C) при этом сокращается, и сравнение с пределом ведётся
        от реального, а не формульного уровня."""
        results = []
        for spec in METRIC_SPECS:
            est = self._estimate_via_formula(spec, hypothetical_state)
            if est is not None:
                base = None if baseline_state is None else self._estimate_via_formula(spec, baseline_state)
                results.append(self._as_increment(est, base, baseline_quality))

        cetane_est = self._estimate_cetane_via_d976(hypothetical_state, for_prediction=True)
        if cetane_est is not None:
            base = None if baseline_state is None else self._estimate_cetane_via_d976(baseline_state, for_prediction=True)
            results.append(self._as_increment(cetane_est, base, baseline_quality))

        return results

    @staticmethod
    def _as_increment(
        predicted: QualityMetricEstimate,
        formula_baseline: QualityMetricEstimate | None,
        baseline_quality: QualityAssessment | None,
    ) -> QualityMetricEstimate:
        if formula_baseline is None or baseline_quality is None:
            return predicted
        current = next((e for e in baseline_quality.current if e.metric == predicted.metric), None)
        if current is None:
            return predicted
        delta = predicted.value - formula_baseline.value
        return predicted.model_copy(update={"value": current.value + delta})

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
            margin_high = cfg.get("risk_margin_high")
            margin_medium = cfg.get("risk_margin_medium")
            p_exceed = None if not est.typical_error else exceed_probability(margin, est.typical_error)
            if margin < 0:
                risk = RiskClass.CRITICAL
            elif p_exceed is not None:
                risk = (
                    RiskClass.HIGH if p_exceed >= EXCEED_P_HIGH
                    else RiskClass.MEDIUM if p_exceed >= EXCEED_P_MEDIUM
                    else RiskClass.LOW
                )
            elif margin_high is not None and margin < margin_high:
                risk = RiskClass.HIGH
            elif margin_medium is not None and margin < margin_medium:
                risk = RiskClass.MEDIUM
            else:
                risk = RiskClass.LOW
            violations.append(SpecViolationRisk(
                metric=metric, limit=limit, margin=margin, op=op, risk_class=risk, exceed_probability=p_exceed,
            ))

        return violations
