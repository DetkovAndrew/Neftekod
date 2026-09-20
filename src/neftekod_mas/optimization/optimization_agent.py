"""
Агент оптимизации (ARCHITECTURE.md §6.4).

Генерирует кандидатные изменения режима вокруг текущей точки по
кандидатному набору управляющих воздействий (config/control_variables.yaml,
только записи с одиночным `tag` и `active_in_optimizer_v1 != false` --
переменные блендинга/соотношений без единого тега вынесены за рамки v1,
см. ARCHITECTURE.md §13), оценивает каждый кандидат Агентом качества
(предиктивный, формульный путь -- `predict_effect`, НЕ текущий ЛИМС/ПАК,
см. docstring `quality_agent.predict_effect`) и Агентом надёжности,
отбрасывает недопустимые, ранжирует оставшиеся взвешенной суммой
(ТЗ п.4).

Это ЧИСТО алгоритмическая оптимизация (перебор сетки кандидатов +
взвешенное ранжирование) поверх уже готовых детерминированных
предикторов -- обучения модели здесь нет.
"""

from __future__ import annotations

import copy
from datetime import datetime

from neftekod_mas.reliability.reliability_agent import ReliabilityAgent
from neftekod_mas.quality import literature_proxies as litproxy
from neftekod_mas.quality.quality_agent import HT_FEED_POINT, QualityAgent
from neftekod_mas.schemas import (
    ConfidenceLevel,
    ControlAction,
    ControlCandidate,
    DataSource,
    EquipmentRiskAssessment,
    OptimizationResult,
    ProcessState,
    QualityAssessment,
    QualityMetricEstimate,
    RiskClass,
    TagReading,
    metric_name,
)

# Теги, для которых включается литературный прокси серы (см.
# quality/literature_proxies.py) -- температура входа/выхода реакторной
# секции 24-2000, единственные теги, к которым Q&A-сессия прямо привязала
# зависимость "температура реактора -> сера".
SULFUR_TEMP_PROXY_TAGS = {"242000:T5", "242000:T11"}

DELTA_STEPS = [-2, -1, 1, 2]
# Прирост запаса меньше этого (в единицах показателя) -- "не сдвигает"
# (численный шум формулы, а не эффект).
_MIN_GAIN = 1e-3  # шагов сетки в обе стороны от текущей точки

_FLOW_UNITS = {"т/ч", "м³/ч", "м3/ч"}


def _flow_throughput_proxy(action: ControlAction) -> float | None:
    """Прокси 'влияние на выпуск' (ТЗ п.4: 'производительность/выпуск' --
    одна из 4 вещей, которые система должна оптимизировать). Осмыслен
    только для переменных типа расход (F30/F32/F15) -- для температуры
    и давления изменение выпуска не считается напрямую в имеющихся
    материалах, поэтому честно возвращается None, а не 0 (0 означало бы
    "проверили, эффекта нет", а не "нечем измерить")."""
    if action.unit not in _FLOW_UNITS:
        return None
    return action.recommended_value - action.current_value


_INSTALLATION_KEY_TO_GRAPH_PREFIX = {"avt": "avt", "hydrotreating_242000": "242000"}


def _flatten_active_variables(control_variables_cfg: dict) -> list[dict]:
    """Возвращает переменные с уже КВАЛИФИЦИРОВАННЫМ тегом ('avt:T55'),
    т.к. control_variables.yaml хранит голые коды колонок (как в CSV), а
    ProcessState.kip ключуется составным installation:tag_id
    (см. tags/pid_graph.py docstring про коллизию T6/F9 между установками)."""
    out = []
    for installation_key, block in control_variables_cfg.get("installations", {}).items():
        prefix = _INSTALLATION_KEY_TO_GRAPH_PREFIX.get(installation_key)
        for var in block.get("variables", []):
            if var.get("active_in_optimizer_v1") is False:
                continue
            if not var.get("tag") or prefix is None:  # только одиночный тег -- см. docstring модуля
                continue
            qualified = dict(var)
            qualified["tag"] = f"{prefix}:{var['tag']}"
            out.append(qualified)
    return out


class OptimizationAgent:
    def __init__(
        self,
        control_variables_cfg: dict,
        control_bounds: dict,
        hard_constraints: dict,
        objective_weights: dict,
        quality_agent: QualityAgent,
        reliability_agent: ReliabilityAgent,
        blending_agent=None,
        economics_agent=None,
        sulfur_temp_response: dict | None = None,
    ):
        # Агент блендинга опционален: без config/blend_model.yaml система
        # обязана работать ровно как раньше (ARCHITECTURE.md §6.6).
        self.blending_agent = blending_agent
        # Агент экономики тоже опционален: без config/economics.yaml
        # ранжирование остаётся на прежних безразмерных прокси (§6.7).
        self.economics_agent = economics_agent
        # Заводская калибровка отклика серы на температуру (§6.4.1).
        # Пусто -- работает только литературный прокси, и карточка это скажет.
        self.sulfur_temp_response = sulfur_temp_response or {}
        self.active_variables = _flatten_active_variables(control_variables_cfg)
        self.bounds = {k: v for k, v in control_bounds.items() if k != "_meta"}
        self.hard_constraints = hard_constraints.get("product_diesel", {})
        self.weights = objective_weights
        self.quality_agent = quality_agent
        self.reliability_agent = reliability_agent

    # -- генерация кандидатов -----------------------------------------

    def _candidate_values(self, tag: str, current: float) -> list[float]:
        b = self.bounds.get(tag)
        if b is None:
            return []
        step = max((b["p95"] - b["p05"]) / 8.0, 1e-6)
        values: list[float] = []
        for k in DELTA_STEPS:
            v = round(min(max(current + k * step, b["p05"]), b["p95"]), 4)
            # клиппинг к границе может свести разные шаги к одному и тому же
            # значению -- дедуплицируем, иначе в альтернативах появляются
            # визуально одинаковые кандидаты с разными candidate_id.
            if abs(v - current) > 1e-9 and v not in values:
                values.append(v)
        return values

    def _apply_actions(self, state: ProcessState, actions: list[ControlAction]) -> ProcessState:
        """Кандидат может менять СРАЗУ НЕСКОЛЬКО тегов -- так устроен
        рычаг блендинга: доли перераспределяются между компонентами пула
        одновременно, при неизменном суммарном расходе (ARCHITECTURE.md
        §6.6). Одиночное действие -- частный случай списка из одного."""
        new_state = copy.deepcopy(state)
        for action in actions:
            tag = action.tag
            old = new_state.kip.get(tag)
            unit = old.unit if old else ""
            new_state.kip[tag] = TagReading(
                tag_id=tag, value=action.recommended_value, unit=unit,
                timestamp=state.decision_at,
                source=old.source if old else state.kip[tag].source,
            )
        return new_state

    # -- оценка кандидата ------------------------------------------------

    def _evaluate(
        self,
        candidate_id: str,
        state: ProcessState,
        actions: list[ControlAction] | ControlAction,
        baseline_risk: EquipmentRiskAssessment,
        quality_baseline: QualityAssessment,
        violated_metrics: set[str],
        external_quality_deltas: dict[str, float] | None = None,
        external_caveats: list[str] | None = None,
        throughput_proxy: float | None = None,
    ) -> ControlCandidate:
        if isinstance(actions, ControlAction):
            actions = [actions]
        action = actions[0]  # ведущее действие: по нему нормируется ранг
        modified_state = self._apply_actions(state, actions)
        predicted_quality = self.quality_agent.predict_effect(modified_state, state, quality_baseline)

        caveats: list[str] = list(external_caveats or [])

        # Приращения, которые формульный слой 24-2000 увидеть не может:
        # рычаг блендинга меняет теги АВТ (F30/F32), а ВАК-формулы
        # продукта зависят только от тегов 24-2000. Эффект приходит
        # от Агента блендинга уже как СДВИГ показателя продукта
        # (delta-метод, см. blending_agent.predict_product_effect).
        if external_quality_deltas:
            by_metric = {e.metric: e for e in predicted_quality}
            for metric, delta in external_quality_deltas.items():
                est = by_metric.get(metric)
                if est is None:
                    est = next((e for e in quality_baseline.current if e.metric == metric), None)
                    if est is None:
                        continue
                    est = est.model_copy(update={"confidence": ConfidenceLevel.LOW})
                    predicted_quality.append(est)
                    by_metric[metric] = est
                est.value = est.value + float(delta)

        # Литературный прокси серы -- см. quality/literature_proxies.py.
        # Включается ТОЛЬКО для температуры реактора 24-2000 и ТОЛЬКО
        # если сера сейчас реально под угрозой (иначе не нужен).
        # Предпочитается физически обоснованная Аррениус-модель (нужна
        # свежая сера СЫРЬЯ из ЛИМС, точка 1); при её отсутствии --
        # резервная плоская оценка (7.5%/°C, Q&A вопрос №32).
        if action.tag in SULFUR_TEMP_PROXY_TAGS and "sulfur_mg_kg" in violated_metrics:
            baseline_sulfur = next((e for e in quality_baseline.current if e.metric == "sulfur_mg_kg"), None)
            if baseline_sulfur is not None:
                delta_t = action.recommended_value - action.current_value
                feed_reading = state.lab_points.get(f"{HT_FEED_POINT}|Mass.Sulfur")
                model_note = None
                proxy_value = None
                if feed_reading is not None:
                    try:
                        severity = litproxy.hds_severity(feed_reading.value, baseline_sulfur.value)
                        proxy_value = litproxy.sulfur_after_reactor_temp_change_arrhenius(
                            baseline_sulfur.value, severity, action.current_value, delta_t,
                        )
                        model_note = (
                            f"физически обоснованная модель Аррениуса кинетики ГДС "
                            f"(Ea=55 кДж/моль по открытой литературе, severity={severity:.2f} "
                            f"по факт. сере сырья {feed_reading.value:.3g}% масс., "
                            f"возраст ЛИМС {feed_reading.age_minutes:.0f} мин)"
                        )
                    except ValueError:
                        pass
                if proxy_value is None:
                    proxy_value = litproxy.sulfur_after_reactor_temp_change_flat(baseline_sulfur.value, delta_t)
                    model_note = "резервная плоская оценка 7.5%/°C (нет свежего ЛИМС по сере сырья, точка 1)"

                # Заводская калибровка (§6.4.1): на истории ЭТОЙ установки
                # измеренная чувствительность серы к температуре в разы
                # МЕНЬШЕ литературной. Обещать литературный эффект нельзя --
                # берём ту из двух оценок, что обещает меньше улучшения.
                plant_pct = self.sulfur_temp_response.get("plant_calibrated_pct_per_c")
                if plant_pct is not None:
                    plant_value = litproxy.sulfur_after_reactor_temp_change_plant_calibrated(
                        baseline_sulfur.value, delta_t, float(plant_pct),
                    )
                    conservative = litproxy.conservative_sulfur_prediction(
                        proxy_value, plant_value, baseline_sulfur.value,
                    )
                    if conservative != proxy_value:
                        model_note = (
                            f"{model_note}; прогноз ОГРАНИЧЕН заводской калибровкой "
                            f"{float(plant_pct):+.2f} %/°C, измеренной по отклику поточного "
                            f"анализатора Q21 на ступени температуры "
                            f"(литературная модель дала бы {proxy_value:.3g} мг/кг -- оптимистичнее)"
                        )
                    proxy_value = conservative

                predicted_quality = [*predicted_quality, QualityMetricEstimate(
                    metric="sulfur_mg_kg", value=proxy_value, unit="мг/кг",
                    source=DataSource.LITERATURE_PROXY, age_minutes=None, confidence=ConfidenceLevel.LOW,
                )]
                caveats.append(
                    f"Прогноз эффекта на серу -- ЛИТЕРАТУРНЫЙ прокси ({model_note}), НЕ подтверждён экспертом "
                    "завода и НЕ является производственной ВАК-формулой. Требует проверки "
                    "технологом перед применением."
                )

        # Для показателей без расчётной связи с данным рычагом сохраняем
        # текущую лучшую оценку и прикладываем как минимум эталонную ошибку
        # из hard_constraints. Это явное предположение локальной неизменности
        # на малом шаге, которое затем независимо проверяет Guard. Кандидат без
        # оценки обязательного показателя публиковать нельзя.
        predicted_metrics = {estimate.metric for estimate in predicted_quality}
        baseline_by_metric = {estimate.metric: estimate for estimate in quality_baseline.current}
        for metric, cfg in self.hard_constraints.items():
            if cfg.get("limit") is None or metric in predicted_metrics:
                continue
            baseline = baseline_by_metric.get(metric)
            if baseline is None:
                continue
            predicted_quality.append(baseline.model_copy(update={
                "confidence": ConfidenceLevel.LOW,
                "typical_error": baseline.typical_error or float(cfg.get("reference_error", 0.0)),
            }))
            caveats.append(
                f"Для {metric_name(metric)} нет модели отклика на {action.tag}; "
                "применено консервативное предположение локальной неизменности."
            )

        predicted_risk = self.reliability_agent.assess(modified_state)
        _draft = ControlCandidate(
            candidate_id=candidate_id, actions=actions, predicted_quality=[],
            predicted_risk=predicted_risk, feasible=True,
        )

        # Кандидат -- решение нарушения, только если по прогнозу он СДВИГАЕТ
        # нарушенный показатель к норме. "Прогноз есть, но равен исходному"
        # (формула ВАК не содержит этого тега: T95 зависит от F9/F2/T6, ни один
        # из которых не рычаг v1) -- не решение, хотя раньше проходил проверку
        # "показатель прогнозируется" и выигрывал ранг за счёт бонуса выпуска.
        # Ухудшать уже нарушенный показатель нельзя ни при каких условиях.
        gains = self._margin_gains(predicted_quality, quality_baseline, violated_metrics)
        improved = sorted(m for m, g in gains.items() if g > _MIN_GAIN)
        worsened = sorted(m for m, g in gains.items() if g < -_MIN_GAIN)
        unaddressed = sorted(violated_metrics - set(improved))

        feasible = True
        rejection_reason = None
        required_metrics = {
            metric for metric, cfg in self.hard_constraints.items()
            if cfg.get("limit") is not None
        }
        predicted_metrics = {estimate.metric for estimate in predicted_quality}
        missing_required = sorted(required_metrics - predicted_metrics)
        if missing_required:
            feasible = False
            rejection_reason = (
                "Нет прогноза обязательных показателей: "
                + ", ".join(map(metric_name, missing_required))
            )
        elif predicted_risk.risk_class == RiskClass.UNKNOWN:
            feasible = False
            rejection_reason = "Недостаточно признаков для оценки риска оборудования (UNKNOWN)."
        elif violated_metrics and not improved:
            feasible = False
            rejection_reason = (
                "Не сдвигает к норме нарушенный показатель " + ", ".join(map(metric_name, sorted(violated_metrics)))
                + " -- у тега нет расчётной связи с ним в имеющихся моделях"
            )
        elif worsened:
            feasible = False
            rejection_reason = "Ухудшает уже нарушенный показатель " + ", ".join(map(metric_name, worsened))
        elif unaddressed:
            caveats.append(
                "ВНИМАНИЕ: вариант не влияет на " + ", ".join(map(metric_name, unaddressed)) +
                " -- этот риск останется, нужен отдельный рычаг."
            )

        for est in predicted_quality if feasible else []:
            cfg = self.hard_constraints.get(est.metric)
            if not cfg or cfg.get("limit") is None:
                continue
            op = cfg["op"]
            limit = float(cfg["limit"])
            error = est.typical_error or 0.0
            conservative_value = est.value + error if op == "<=" else est.value - error
            margin = (limit - conservative_value) if op == "<=" else (conservative_value - limit)
            if margin < 0:
                feasible = False
                rejection_reason = (
                    f"Консервативный прогноз {est.metric}={conservative_value:.3g} "
                    f"(оценка {est.value:.3g}, ошибка {error:.3g}) нарушает жёсткий предел {op}{limit}"
                )
                break

        if feasible and predicted_risk.hard_stop:
            if predicted_risk.severity_index > baseline_risk.severity_index:
                feasible = False
                rejection_reason = (
                    f"Индекс тяжести режима растёт ({baseline_risk.severity_index:.2f} -> "
                    f"{predicted_risk.severity_index:.2f}) при уже критичном режиме -- "
                    "ключевой принцип запрещает такой компромисс (ARCHITECTURE.md §0)"
                )

        economics = None
        if self.economics_agent is not None and self.economics_agent.available:
            try:
                economics = self.economics_agent.candidate_effect(state, modified_state, _draft)
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                # Экономика -- вспомогательный слой: её отказ не должен
                # ронять цикл принятия решения.
                caveats.append(f"Экономический эффект не посчитан: {exc}")

        if throughput_proxy is None:
            per_action = [_flow_throughput_proxy(a) for a in actions]
            measurable = [x for x in per_action if x is not None]
            # None означает "нечем измерить", 0.0 -- "измерили, эффекта нет".
            throughput_proxy = float(sum(measurable)) if measurable else None

        return ControlCandidate(
            candidate_id=candidate_id,
            actions=actions,
            predicted_quality=predicted_quality,
            predicted_risk=predicted_risk,
            throughput_proxy=throughput_proxy,
            energy_cost_proxy=max(abs(a.recommended_value - a.current_value) for a in actions),
            feasible=feasible,
            rejection_reason=rejection_reason,
            caveats=caveats,
            economics=economics,
        )

    def _blending_candidates(
        self,
        state: ProcessState,
        baseline_risk: EquipmentRiskAssessment,
        quality_baseline: QualityAssessment,
        violated_metrics: set[str],
        start_id: int,
    ) -> list[ControlCandidate]:
        """Кандидаты рычага блендинга: перераспределение долей компонентов
        дизельного пула АВТ при НЕИЗМЕННОМ суммарном расходе.

        Это единственный рычаг системы, который сдвигает T95 продукта:
        ВАК-формула T95 зависит от F9/F2/T6, а доли пула действуют через
        фракционный состав сырья, и цепочка "доли -> T95 сырья -> T95
        продукта" измерена на истории (ARCHITECTURE.md §6.6). Ровно
        поэтому раньше при риске по T95 система была вынуждена
        отказываться "нет рычага".
        """
        if self.blending_agent is None:
            return []
        assessment = self.blending_agent.assess(state)
        if not assessment.available:
            return []

        by_key = {c.key: c for c in assessment.components}
        out: list[ControlCandidate] = []
        cid = start_id
        for option in self.blending_agent.share_candidates(state):
            effect = option.get("product_effect") or {}
            if not effect:
                continue
            cid += 1
            actions = []
            for key, new_flow in option["flows"].items():
                comp = by_key.get(key)
                if comp is None:
                    continue
                actions.append(ControlAction(
                    variable_name=f"blend_share_{key}",
                    tag=comp.flow_tag,
                    current_value=comp.flow_t_h,
                    recommended_value=round(float(new_flow), 3),
                    unit="т/ч",
                ))
            if len(actions) != len(option["flows"]):
                continue
            # Ведущим делаем тяжёлый компонент -- именно его доля
            # фигурирует в объяснении и в модели чувствительности.
            actions.sort(key=lambda a: a.variable_name != "blend_share_avt_fr_290_350")
            caveat = (
                f"Рычаг блендинга: доля тяжёлого компонента (фр. 290-350) "
                f"{option['current_heavy_share_pct']:.2f}% -> {option['heavy_share_pct']:.2f}% масс., "
                f"суммарный расход пула не меняется, сумма долей = {option['share_sum_pct']:.0f}%. "
                "Эффект на качество -- по модели смешения, проверенной на истории "
                "(config/blend_model.yaml), уверенность средняя."
            )
            out.append(self._evaluate(
                f"b{cid}", state, actions, baseline_risk, quality_baseline, violated_metrics,
                external_quality_deltas=effect,
                external_caveats=[caveat],
                # Суммарный расход пула неизменен по построению -- это
                # измеренный ноль, а не "нечем измерить".
                throughput_proxy=0.0,
            ))
        return out

    def _score(
        self,
        candidate: ControlCandidate,
        baseline_quality_margins: dict[str, float],
        violated_metrics: set[str],
        quality_baseline: QualityAssessment,
    ) -> float:
        # Улучшение маржи нормируется на масштаб предела (safe % от limit),
        # иначе разница в °C (T95) и в мг/кг (сера) складывались бы напрямую
        # в одну сумму, что физически бессмысленно.
        margin_improve = 0.0
        for est in candidate.predicted_quality:
            base = baseline_quality_margins.get(est.metric)
            if base is None:
                continue
            cfg = self.hard_constraints.get(est.metric, {})
            if not cfg or cfg.get("limit") is None:
                continue
            op = cfg["op"]
            limit = float(cfg["limit"])
            new_margin = (limit - est.value) if op == "<=" else (est.value - limit)
            margin_improve += (new_margin - base) / max(abs(limit), 1e-6)

        # Частичное решение (сдвигает к норме не все нарушенные показатели)
        # штрафуется за каждый оставшийся без рычага показатель.
        gains = self._margin_gains(candidate.predicted_quality, quality_baseline, violated_metrics)
        unaddressed_penalty = 1.0 * sum(1 for m in violated_metrics if gains.get(m, 0.0) <= _MIN_GAIN)

        # energy_cost_proxy и throughput_proxy -- в СЫРЫХ единицах самой
        # переменной (°C, МПа, т/ч вперемешку). Нормируем на масштаб
        # диапазона (p95-p05) этой же переменной, иначе сдвиг давления на
        # 0.02 МПа и сдвиг температуры на 4°C складывались бы по весу так,
        # будто они сопоставимы -- та же ошибка, что уже была исправлена
        # для margin_improve чуть выше.
        tag = candidate.actions[0].tag
        b = self.bounds.get(tag or "")
        scale = max((b["p95"] - b["p05"]), 1e-6) if b else 1.0
        energy_norm = (candidate.energy_cost_proxy or 0.0) / scale
        throughput_norm = (candidate.throughput_proxy or 0.0) / scale

        w = self.weights
        score = (
            w["quality_margin_improvement"] * margin_improve
            - w["equipment_risk_severity"] * candidate.predicted_risk.severity_index
            - unaddressed_penalty
        )

        # Экономика в реальных деньгах вытесняет безразмерные прокси --
        # но ТОЛЬКО там, где её удалось посчитать. Где не удалось,
        # остаётся прежнее, уже проверенное поведение (§6.7).
        net_rub = (candidate.economics or {}).get("net_rub_per_day")
        if net_rub is not None and "economic_value" in w:
            reference = float(w.get("economic_reference_rub_per_day", 1e6)) or 1e6
            score += w["economic_value"] * (float(net_rub) / reference)
        else:
            score += -w["energy_cost_proxy"] * energy_norm + w["throughput_proxy"] * throughput_norm
        return score

    # -- главный вход ------------------------------------------------

    def run(
        self,
        state: ProcessState,
        baseline_quality_margins: dict[str, float],
        baseline_risk: EquipmentRiskAssessment,
        quality_baseline: QualityAssessment,
        violated_metrics: set[str],
    ) -> OptimizationResult:
        candidates: list[ControlCandidate] = []
        cid = 0

        for var in self.active_variables:
            tag = var["tag"]
            current_reading = state.kip.get(tag)
            if current_reading is None:
                continue
            for new_value in self._candidate_values(tag, current_reading.value):
                cid += 1
                action = ControlAction(
                    variable_name=var["name"],
                    tag=tag,
                    current_value=current_reading.value,
                    recommended_value=new_value,
                    unit=var.get("unit", ""),
                )
                candidate = self._evaluate(
                    f"c{cid}", state, action, baseline_risk, quality_baseline, violated_metrics
                )
                candidates.append(candidate)

        candidates.extend(
            self._blending_candidates(
                state, baseline_risk, quality_baseline, violated_metrics, start_id=cid
            )
        )

        # Если НИ ОДИН кандидат в принципе не прогнозирует эффект ни на
        # один из реально нарушенных показателей -- у системы просто нет
        # рычага с известной количественной (или литературной прокси-)
        # связью с проблемой. Предлагать в этом случае "лучший по прочим
        # метрикам" вариант было бы нечестной подменой решения проблемы
        # оптимизацией несвязанных показателей -- честнее отказаться
        # (ТЗ: "при отсутствии допустимого варианта система должна
        # сообщить об этом").
        if violated_metrics and not any(
            any(g > _MIN_GAIN for g in self._margin_gains(c.predicted_quality, quality_baseline, violated_metrics).values())
            for c in candidates
        ):
            return OptimizationResult(
                decision_at=state.decision_at,
                candidates_evaluated=len(candidates),
                feasible_candidates=[],
                rejected_candidates=[c for c in candidates if not c.feasible],
                no_feasible_solution=True,
                no_feasible_reason=(
                    f"Ни один из {len(candidates)} вариантов по кандидатным рычагам не увеличивает "
                    f"запас до норматива по показателю {', '.join(map(metric_name, sorted(violated_metrics)))}: "
                    + self._why_no_lever(candidates, state, quality_baseline, violated_metrics)
                    + " Количественно обоснованную рекомендацию сформировать нечем -- нужны решение "
                    "технолога и свежий анализ ЛИМС."
                ),
            )

        feasible = [c for c in candidates if c.feasible]
        for c in feasible:
            c.score = self._score(c, baseline_quality_margins, violated_metrics, quality_baseline)
        feasible.sort(key=lambda c: c.score, reverse=True)

        if not feasible:
            return OptimizationResult(
                decision_at=state.decision_at,
                candidates_evaluated=len(candidates),
                feasible_candidates=[],
                rejected_candidates=[c for c in candidates if not c.feasible],
                no_feasible_solution=True,
                no_feasible_reason=(
                    "Ни один из "
                    f"{len(candidates)} рассмотренных вариантов не проходит жёсткие "
                    "ограничения, не устраняет нарушенный показатель или не улучшает "
                    "уже критичный режим надёжности." + self._best_attempts(candidates, violated_metrics)
                ),
            )

        pareto_ids = _pareto_front(feasible)

        return OptimizationResult(
            decision_at=state.decision_at,
            candidates_evaluated=len(candidates),
            feasible_candidates=feasible,
            rejected_candidates=[c for c in candidates if not c.feasible],
            pareto_front_ids=pareto_ids,
            recommended_candidate_id=feasible[0].candidate_id,
        )


    def _margin_gains(
        self,
        predicted_quality: list[QualityMetricEstimate],
        quality_baseline: QualityAssessment,
        metrics: set[str],
    ) -> dict[str, float]:
        """Прирост запаса до норматива (в единицах показателя) по каждому из
        metrics; показатель без прогноза в словарь не попадает."""
        base = {e.metric: e.value for e in quality_baseline.current}
        gains = {}
        for e in predicted_quality:
            if e.metric not in metrics or e.metric not in base:
                continue
            cfg = self.hard_constraints.get(e.metric) or {}
            sign = -1.0 if cfg.get("op", "<=") == "<=" else 1.0
            gains[e.metric] = sign * (e.value - base[e.metric])
        return gains

    def _why_no_lever(self, candidates, state, quality_baseline, violated_metrics) -> str:
        """Различает "рычага нет вообще" и "рычаг есть, но упёрся в границу
        модельного диапазона" -- для оператора это разные ситуации."""
        bits = []
        for metric in sorted(violated_metrics):
            moving = [
                c for c in candidates
                if abs(self._margin_gains(c.predicted_quality, quality_baseline, {metric}).get(metric, 0.0)) > _MIN_GAIN
            ]
            if not moving:
                bits.append(
                    f"{metric_name(metric)} -- в имеющихся расчётных моделях (ВАК-формулы, ASTM D976, "
                    "литературный прокси серы) у рычагов нет связи с этим показателем"
                )
                continue
            for tag in dict.fromkeys(c.actions[0].tag for c in moving):
                reading, b = state.kip.get(tag), self.bounds.get(tag)
                var = next(c.actions[0] for c in moving if c.actions[0].tag == tag)
                if reading is not None and b is not None and not (b["p05"] < reading.value < b["p95"]):
                    bits.append(
                        f"{metric_name(metric)} -- {var.variable_name} ({tag}) = {reading.value:.4g} {var.unit} "
                        f"уже на границе модельного диапазона [{b['p05']:.4g}; {b['p95']:.4g}], "
                        "сдвиг в нужную сторону не допускается"
                    )
                else:
                    bits.append(f"{metric_name(metric)} -- все расчётные шаги {var.variable_name} ({tag}) ухудшают показатель")
        return "; ".join(bits) + "."

    def _best_attempts(self, candidates: list[ControlCandidate], violated_metrics: set[str]) -> str:
        """Для отказа: лучший достижимый прогноз по каждому нарушенному показателю --
        оператор видит, насколько не хватает рычагов, а не только сам факт отказа."""
        bits = []
        for metric in sorted(violated_metrics):
            cfg = self.hard_constraints.get(metric) or {}
            if cfg.get("limit") is None:
                continue
            sign = 1.0 if cfg["op"] == "<=" else -1.0
            tries = [
                (e.value, c) for c in candidates for e in c.predicted_quality if e.metric == metric
            ]
            if not tries:
                continue
            value, c = min(tries, key=lambda t: sign * t[0])
            a = c.actions[0]
            bits.append(
                f"лучший вариант по показателю «{metric_name(metric)}» -- {a.variable_name} ({a.tag}) до "
                f"{a.recommended_value:.4g} {a.unit}: прогноз {value:.4g} при нормативе "
                f"{cfg['op']} {float(cfg['limit']):.4g}"
                + (f" ({c.rejection_reason})" if c.rejection_reason and metric not in c.rejection_reason else "")
            )
        text = "; ".join(bits)
        return (" " + text[0].upper() + text[1:] + ".") if bits else ""


def _pareto_front(candidates: list[ControlCandidate]) -> list[str]:
    """Недоминируемые по (качество-маржа не хуже, риск не выше, энергия
    не выше) -- простая O(n^2) реализация, кандидатов мало (десятки)."""

    def dominates(a: ControlCandidate, b: ControlCandidate) -> bool:
        a_risk, b_risk = a.predicted_risk.severity_index, b.predicted_risk.severity_index
        a_energy = a.energy_cost_proxy or 0.0
        b_energy = b.energy_cost_proxy or 0.0
        not_worse = (a.score or 0) >= (b.score or 0) and a_risk <= b_risk and a_energy <= b_energy
        strictly_better = (a.score or 0) > (b.score or 0) or a_risk < b_risk or a_energy < b_energy
        return not_worse and strictly_better

    front = []
    for c in candidates:
        if not any(dominates(other, c) for other in candidates if other.candidate_id != c.candidate_id):
            front.append(c.candidate_id)
    return front
