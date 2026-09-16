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
from neftekod_mas.quality.quality_agent import QualityAgent
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
)

# Теги, для которых включается литературный прокси серы (см.
# quality/literature_proxies.py) -- температура входа/выхода реакторной
# секции 24-2000, единственные теги, к которым Q&A-сессия прямо привязала
# зависимость "температура реактора -> сера".
SULFUR_TEMP_PROXY_TAGS = {"242000:T5", "242000:T11"}

DELTA_STEPS = [-2, -1, 1, 2]  # шагов сетки в обе стороны от текущей точки


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
    ):
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

    def _apply_action(self, state: ProcessState, tag: str, new_value: float) -> ProcessState:
        new_state = copy.deepcopy(state)
        old = new_state.kip.get(tag)
        unit = old.unit if old else ""
        new_state.kip[tag] = TagReading(
            tag_id=tag, value=new_value, unit=unit, timestamp=state.decision_at,
            source=old.source if old else state.kip[tag].source,
        )
        return new_state

    # -- оценка кандидата ------------------------------------------------

    def _evaluate(
        self,
        candidate_id: str,
        state: ProcessState,
        action: ControlAction,
        baseline_risk: EquipmentRiskAssessment,
        quality_baseline: QualityAssessment,
        violated_metrics: set[str],
    ) -> ControlCandidate:
        modified_state = self._apply_action(state, action.tag, action.recommended_value)
        predicted_quality = self.quality_agent.predict_effect(modified_state)

        caveats: list[str] = []

        # Литературный прокси серы -- см. quality/literature_proxies.py.
        # Включается ТОЛЬКО для температуры реактора 24-2000 и ТОЛЬКО
        # если сера сейчас реально под угрозой (иначе не нужен).
        if action.tag in SULFUR_TEMP_PROXY_TAGS and "sulfur_mg_kg" in violated_metrics:
            baseline_sulfur = next((e for e in quality_baseline.current if e.metric == "sulfur_mg_kg"), None)
            if baseline_sulfur is not None:
                delta_t = action.recommended_value - action.current_value
                proxy_value = litproxy.sulfur_after_reactor_temp_change(baseline_sulfur.value, delta_t)
                predicted_quality = [*predicted_quality, QualityMetricEstimate(
                    metric="sulfur_mg_kg", value=proxy_value, unit="мг/кг",
                    source=DataSource.LITERATURE_PROXY, age_minutes=None, confidence=ConfidenceLevel.LOW,
                )]
                caveats.append(
                    "Оценка серы -- ЛИТЕРАТУРНЫЙ прокси (7.5%/°C, середина диапазона 5-10%, "
                    "Q&A вопрос №32), НЕ подтверждён экспертом и НЕ является производственной "
                    "ВАК-формулой. Требует проверки технологом перед применением."
                )

        predicted_risk = self.reliability_agent.assess(modified_state)

        predictable_metrics = {e.metric for e in predicted_quality}
        # Честно: какие из РЕАЛЬНО НАРУШЕННЫХ сейчас показателей этот
        # кандидат вообще не в состоянии спрогнозировать -- раньше это
        # проверялось только для тегов 24-2000, из-за чего кандидаты по
        # АВТ-тегам молча выглядели так, будто "решают" проблему серы,
        # хотя вообще её не затрагивают.
        unaddressed = violated_metrics - predictable_metrics
        if unaddressed:
            caveats.append(
                "ВНИМАНИЕ: этот кандидат НЕ прогнозирует эффект на " + ", ".join(sorted(unaddressed)) +
                " -- у выбранного тега нет расчётной связи с этим показателем в имеющихся материалах. "
                "Нарушение по нему могло остаться неисправленным."
            )

        feasible = True
        rejection_reason = None

        for est in predicted_quality:
            cfg = self.hard_constraints.get(est.metric)
            if not cfg or cfg.get("limit") is None:
                continue
            op = cfg["op"]
            limit = float(cfg["limit"])
            margin = (limit - est.value) if op == "<=" else (est.value - limit)
            if margin < 0:
                feasible = False
                rejection_reason = f"Прогноз {est.metric}={est.value:.3g} нарушает жёсткий предел {op}{limit}"
                break

        if feasible and predicted_risk.hard_stop:
            if predicted_risk.severity_index > baseline_risk.severity_index:
                feasible = False
                rejection_reason = (
                    f"Индекс тяжести режима растёт ({baseline_risk.severity_index:.2f} -> "
                    f"{predicted_risk.severity_index:.2f}) при уже критичном режиме -- "
                    "ключевой принцип запрещает такой компромисс (ARCHITECTURE.md §0)"
                )

        return ControlCandidate(
            candidate_id=candidate_id,
            actions=[action],
            predicted_quality=predicted_quality,
            predicted_risk=predicted_risk,
            throughput_proxy=None,
            energy_cost_proxy=abs(action.recommended_value - action.current_value),
            feasible=feasible,
            rejection_reason=rejection_reason,
            caveats=caveats,
        )

    def _score(
        self,
        candidate: ControlCandidate,
        baseline_quality_margins: dict[str, float],
        violated_metrics: set[str],
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

        # Кандидат, который вообще не прогнозирует эффект на реально
        # нарушенный показатель, штрафуется отдельно и сильно -- иначе он
        # может обойти по рангу кандидата, честно пытающегося решить
        # именно заявленную проблему (см. docstring _evaluate).
        predictable_metrics = {e.metric for e in candidate.predicted_quality}
        unaddressed_penalty = 1.0 * len(violated_metrics - predictable_metrics)

        w = self.weights
        score = (
            w["quality_margin_improvement"] * margin_improve
            - w["equipment_risk_severity"] * candidate.predicted_risk.severity_index
            - w["energy_cost_proxy"] * (candidate.energy_cost_proxy or 0.0)
            - unaddressed_penalty
        )
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

        # Если НИ ОДИН кандидат в принципе не прогнозирует эффект ни на
        # один из реально нарушенных показателей -- у системы просто нет
        # рычага с известной количественной (или литературной прокси-)
        # связью с проблемой. Предлагать в этом случае "лучший по прочим
        # метрикам" вариант было бы нечестной подменой решения проблемы
        # оптимизацией несвязанных показателей -- честнее отказаться
        # (ТЗ: "при отсутствии допустимого варианта система должна
        # сообщить об этом").
        if violated_metrics and not any(
            violated_metrics & {e.metric for e in c.predicted_quality} for c in candidates
        ):
            return OptimizationResult(
                decision_at=state.decision_at,
                candidates_evaluated=len(candidates),
                feasible_candidates=[],
                no_feasible_solution=True,
                no_feasible_reason=(
                    "Ни один из кандидатных управляющих тегов не имеет расчётной "
                    "(ВАК-формула/литературный прокси) связи с нарушенным показателем "
                    f"{', '.join(sorted(violated_metrics))} -- количественно обоснованную "
                    "рекомендацию сформировать нечем."
                ),
            )

        feasible = [c for c in candidates if c.feasible]
        for c in feasible:
            c.score = self._score(c, baseline_quality_margins, violated_metrics)
        feasible.sort(key=lambda c: c.score, reverse=True)

        if not feasible:
            return OptimizationResult(
                decision_at=state.decision_at,
                candidates_evaluated=len(candidates),
                feasible_candidates=[],
                no_feasible_solution=True,
                no_feasible_reason=(
                    "Ни один из "
                    f"{len(candidates)} рассмотренных вариантов не проходит жёсткие "
                    "ограничения или не улучшает уже критичный режим надёжности."
                ),
            )

        pareto_ids = _pareto_front(feasible)

        return OptimizationResult(
            decision_at=state.decision_at,
            candidates_evaluated=len(candidates),
            feasible_candidates=feasible,
            pareto_front_ids=pareto_ids,
            recommended_candidate_id=feasible[0].candidate_id,
        )


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
