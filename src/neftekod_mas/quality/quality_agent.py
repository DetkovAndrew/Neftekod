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

GODT_POINT2 = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"
# Точка 1 -- вход в гидроочистку (ФРАКЦ_ДИЗ), содержит Mass.Sulfur
# (% масс.) -- сера СЫРЬЯ, нужна для расчёта severity ГДС
# (literature_proxies.hds_severity), см. ARCHITECTURE.md §6.4.
HT_FEED_POINT = "Установка 'Гидроочистка'. Точка отбора '1'. Продукт 'ФРАКЦ_ДИЗ'."

# Критический порог возраста ЛИМС, после которого показатель считается
# ненадёжным настолько, что System обязана понизить confidence до REFUSE,
# если это единственный источник (нет ПАК/ВАК-фолбэка) -- см. ARCHITECTURE.md §13.
CRITICAL_LIMS_AGE_MINUTES = 7 * 24 * 60  # 7 суток
DEFAULT_USABLE_PAK_MINUTES = 3 * 60


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
        usable_lims_minutes: float = CRITICAL_LIMS_AGE_MINUTES,
        usable_pak_minutes: float = DEFAULT_USABLE_PAK_MINUTES,
    ):
        self.hard_constraints = hard_constraints
        self.stale_lims_minutes = stale_lims_minutes
        self.usable_lims_minutes = usable_lims_minutes
        self.usable_pak_minutes = usable_pak_minutes
        # Бэктест точности ВАК-формул против ЛИМС (scripts/compute_vak_formula_accuracy.py,
        # ARCHITECTURE.md §5.3) -- статистика, не обучение. Опционально: без
        # него формульные оценки просто не получают typical_error/доп.
        # понижения confidence, остальной пайплайн не ломается.
        self.formula_accuracy = {k: v for k, v in (formula_accuracy or {}).items() if k != "_meta"}
        # Бэктест ASTM D976 против реальных CetaneNumber этого завода
        # (scripts/compute_astm_cetane_accuracy.py) -- также опционально.
        self.astm_accuracy = {k: v for k, v in (astm_accuracy or {}).items() if k != "_meta"}

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
            required = {m for m, cfg in self.hard_constraints.get("product_diesel", {}).items() if cfg.get("limit") is not None}
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
            if lims is not None and lims.age_minutes <= self.usable_lims_minutes:
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
            if pak is not None and pak.age_minutes <= self.usable_pak_minutes:
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
                if lims is None or lims.age_minutes > self.usable_lims_minutes:
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

    def predict_effect(self, hypothetical_state: ProcessState) -> list[QualityMetricEstimate]:
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
        предсказываемые D15/T50, см. _estimate_cetane_via_d976."""
        results = []
        for spec in METRIC_SPECS:
            est = self._estimate_via_formula(spec, hypothetical_state)
            if est is not None:
                results.append(est)

        cetane_est = self._estimate_cetane_via_d976(hypothetical_state, for_prediction=True)
        if cetane_est is not None:
            results.append(cetane_est)

        return results

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
            if margin < 0:
                risk = RiskClass.CRITICAL
            elif margin_high is not None and margin < margin_high:
                risk = RiskClass.HIGH
            elif margin_medium is not None and margin < margin_medium:
                risk = RiskClass.MEDIUM
            else:
                risk = RiskClass.LOW
            violations.append(SpecViolationRisk(metric=metric, limit=limit, margin=margin, risk_class=risk))

        return violations
