"""
Оркестратор (ARCHITECTURE.md §6.5) -- проводит полный цикл принятия
решения (ТЗ, 8 шагов, см. ARCHITECTURE.md §8) и формирует итоговую
Recommendation или мотивированный отказ.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from neftekod_mas.data.sync import build_process_state
from neftekod_mas.optimization.optimization_agent import OptimizationAgent
from neftekod_mas.orchestrator.explain import explain_no_action, explain_recommendation, explain_refusal
from neftekod_mas.orchestrator.guard import Guard
from neftekod_mas.quality.quality_agent import QualityAgent
from neftekod_mas.reliability.reliability_agent import ReliabilityAgent
from neftekod_mas.schemas import (
    ConfidenceLevel,
    GuardVerdict,
    Recommendation,
    RiskClass,
)

ACTION_NEEDED_QUALITY_RISK = {RiskClass.MEDIUM, RiskClass.HIGH, RiskClass.CRITICAL}
ACTION_NEEDED_EQUIPMENT_RISK = {RiskClass.MEDIUM, RiskClass.HIGH, RiskClass.CRITICAL}


class Orchestrator:
    def __init__(
        self,
        quality_agent: QualityAgent,
        reliability_agent: ReliabilityAgent,
        optimization_agent: OptimizationAgent,
        guard: Guard,
        kip_bounds: dict | None = None,
    ):
        self.quality_agent = quality_agent
        self.reliability_agent = reliability_agent
        self.optimization_agent = optimization_agent
        self.guard = guard
        self.kip_bounds = kip_bounds

    def run_cycle(
        self,
        decision_at: datetime,
        avt_kip: pd.DataFrame,
        ht_kip: pd.DataFrame,
        lims_long: pd.DataFrame,
        pak_long: pd.DataFrame,
    ) -> Recommendation:
        # Шаги 1-2 ТЗ: получить состояние, проверить полноту/актуальность/согласованность
        state = build_process_state(
            decision_at, avt_kip, ht_kip, lims_long, pak_long, kip_bounds=self.kip_bounds
        )

        if not state.quality_report.sync_ok:
            return self._refusal(
                decision_at, state.quality_report.flags,
                "нет снимка КИП в пределах допуска на запрошенный момент времени.",
            )

        # Шаг 3: качество
        quality = self.quality_agent.assess(state)
        if quality.overall_confidence == ConfidenceLevel.REFUSE:
            return self._refusal(
                decision_at, state.quality_report.flags,
                "не удалось получить оценку одного или нескольких обязательных "
                "показателей качества (нет свежих ЛИМС/ПАК и нет расчётной формулы).",
            )

        # Шаг 4: надёжность
        risk = self.reliability_agent.assess(state)

        quality_needs_action = any(v.risk_class in ACTION_NEEDED_QUALITY_RISK for v in quality.violations)
        equipment_needs_action = risk.risk_class in ACTION_NEEDED_EQUIPMENT_RISK

        key_state = {
            e.metric: e.value for e in quality.current
        } | {"severity_index": risk.severity_index}

        if not quality_needs_action and not equipment_needs_action:
            return Recommendation(
                decision_at=decision_at,
                key_state=key_state,
                problem_or_risk="Риска не обнаружено.",
                proposed_actions=[],
                expected_effect={},
                constraints_checked=[],
                confidence=quality.overall_confidence,
                confidence_warnings=self._warnings(state),
                explanation=explain_no_action(quality, risk),
                is_refusal=False,
            )

        # Шаги 5-7: генерация, отбор, ранжирование вариантов
        baseline_margins = {v.metric: v.margin for v in quality.violations}
        violated_metrics = {v.metric for v in quality.violations if v.risk_class in ACTION_NEEDED_QUALITY_RISK}
        opt_result = self.optimization_agent.run(state, baseline_margins, risk, quality, violated_metrics)

        if opt_result.no_feasible_solution:
            return self._refusal(decision_at, state.quality_report.flags, opt_result.no_feasible_reason or "нет допустимого варианта.", key_state=key_state)

        # Шаг 8: выбрать и объяснить -- с независимой Guard-проверкой,
        # при BLOCK пробуем следующего по рангу кандидата (defense in depth).
        # Альтернативы (ТЗ п.3, роль Оркестратора: "...альтернативы...") --
        # остальные точки Парето-фронта, кроме выбранной.
        pareto_ids = set(opt_result.pareto_front_ids)
        by_id = {c.candidate_id: c for c in opt_result.feasible_candidates}

        for candidate in opt_result.feasible_candidates:
            guard_report = self.guard.review(candidate, decision_at)
            if guard_report.final_verdict != GuardVerdict.BLOCK:
                alternatives = [
                    by_id[cid] for cid in pareto_ids
                    if cid != candidate.candidate_id and cid in by_id
                ]
                return Recommendation(
                    decision_at=decision_at,
                    key_state=key_state,
                    problem_or_risk=self._problem_text(quality, risk),
                    proposed_actions=candidate.actions,
                    expected_effect={e.metric: e.value for e in candidate.predicted_quality},
                    constraints_checked=guard_report.checks,
                    confidence=quality.overall_confidence,
                    confidence_warnings=self._warnings(state) + candidate.caveats,
                    explanation=explain_recommendation(candidate, quality, risk),
                    is_refusal=False,
                    alternatives=alternatives,
                )

        return self._refusal(
            decision_at, state.quality_report.flags,
            "все допустимые по Агенту оптимизации варианты не прошли независимую "
            "P&ID-проверку Guard (см. ARCHITECTURE.md §6.5).",
            key_state=key_state,
        )

    def _problem_text(self, quality, risk) -> str:
        bits = []
        if quality.violations:
            worst = min(quality.violations, key=lambda v: v.margin)
            bits.append(f"риск нарушения {worst.metric} (margin={worst.margin:.3g})")
        if risk.risk_class in ACTION_NEEDED_EQUIPMENT_RISK:
            bits.append(f"тяжёлый режим оборудования (индекс {risk.severity_index:.2f})")
        return "; ".join(bits) if bits else "см. объяснение"

    def _warnings(self, state) -> list[str]:
        return [f"{f.code}: {f.tag_or_point} -- {f.detail}" for f in state.quality_report.flags]

    def _refusal(self, decision_at, flags, reason: str, key_state: dict | None = None) -> Recommendation:
        return Recommendation(
            decision_at=decision_at,
            key_state=key_state or {},
            problem_or_risk=reason,
            proposed_actions=[],
            expected_effect={},
            constraints_checked=[],
            confidence=ConfidenceLevel.REFUSE,
            confidence_warnings=[f"{f.code}: {f.tag_or_point} -- {f.detail}" for f in flags],
            explanation=explain_refusal(reason),
            is_refusal=True,
        )
