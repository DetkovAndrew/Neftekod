"""
Оркестратор (ARCHITECTURE.md §6.5) -- проводит полный цикл принятия
решения (ТЗ, 8 шагов, см. ARCHITECTURE.md §8) и формирует итоговую
Recommendation или мотивированный отказ.

Каждый обмен между агентами фиксируется на шине (orchestrator/bus.py) и
попадает в Recommendation.trace. LLM Monitor (orchestrator/llm_monitor.py),
если подключён, вызывается ПОСЛЕ того, как карточка полностью сформирована
и проверена Guard'ом, и может добавить только llm_commentary.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from neftekod_mas.data.sync import build_process_state
from neftekod_mas.optimization.optimization_agent import OptimizationAgent
from neftekod_mas.orchestrator.bus import MessageBus
from neftekod_mas.orchestrator.explain import (
    describe_violation,
    worst_violation,
    explain_no_action,
    explain_recommendation,
    explain_refusal,
)
from neftekod_mas.orchestrator.guard import Guard
from neftekod_mas.quality.quality_agent import GODT_POINT2, METRIC_SPECS, QualityAgent
from neftekod_mas.reliability.reliability_agent import ReliabilityAgent
from neftekod_mas.schemas import (
    ConfidenceLevel,
    GuardVerdict,
    Recommendation,
    RiskClass,
)
from neftekod_mas.utils.logging_run import NullRunLogger

# MEDIUM по качеству -- "под наблюдением" (в карточке, без управляющего действия):
# показатель между порогом наблюдения и порогом действия (config/hard_constraints.yaml).
ACTION_NEEDED_QUALITY_RISK = {RiskClass.HIGH, RiskClass.CRITICAL}
ACTION_NEEDED_EQUIPMENT_RISK = {RiskClass.MEDIUM, RiskClass.HIGH, RiskClass.CRITICAL}
# Флаги Data & Sync, при которых рекомендация не формируется вообще.
REFUSAL_FLAG_CODES = {"transient_regime"}
# Устаревшие лабораторные точки, которые не участвуют в оценке нормируемых
# показателей (другие точки отбора АВТ и т.п.), в карточку сворачиваются в
# одну строку -- иначе десятки однотипных предупреждений заслоняют важные.
# Полный список остаётся в журнале цикла (01_process_state.json).
_CONTROL_LAB_POINTS = {f"{GODT_POINT2}|{s.lims_param}" for s in METRIC_SPECS if s.lims_param} | {
    s.pak_param for s in METRIC_SPECS if s.pak_param
}


class Orchestrator:
    def __init__(
        self,
        quality_agent: QualityAgent,
        reliability_agent: ReliabilityAgent,
        optimization_agent: OptimizationAgent,
        guard: Guard,
        kip_bounds: dict | None = None,
        run_logger=None,
        llm_monitor=None,
        stale_lims_minutes: int = 24 * 60,
        stale_pak_minutes: int = 60,
    ):
        self.quality_agent = quality_agent
        self.reliability_agent = reliability_agent
        self.optimization_agent = optimization_agent
        self.guard = guard
        self.kip_bounds = kip_bounds
        self.run_logger = run_logger or NullRunLogger()
        self.llm_monitor = llm_monitor
        self.stale_lims_minutes = stale_lims_minutes
        self.stale_pak_minutes = stale_pak_minutes

    def run_cycle(
        self,
        decision_at: datetime,
        avt_kip: pd.DataFrame,
        ht_kip: pd.DataFrame,
        lims_long: pd.DataFrame,
        pak_long: pd.DataFrame,
    ) -> Recommendation:
        bus = MessageBus()
        rec = self._decide(bus, decision_at, avt_kip, ht_kip, lims_long, pak_long)
        rec = rec.model_copy(update={"trace": list(bus.messages)})
        if self.llm_monitor is not None:
            rec = self.llm_monitor.annotate(rec)
            bus.send("llm_monitor", "operator", "commentary", rec.llm_status or "")
            rec = rec.model_copy(update={"trace": list(bus.messages)})
        self.run_logger.log(decision_at, "00_trace", bus.messages)
        self.run_logger.log(decision_at, "05_recommendation", rec)
        return rec

    def _decide(self, bus, decision_at, avt_kip, ht_kip, lims_long, pak_long) -> Recommendation:
        # Шаги 1-2 ТЗ: получить состояние, проверить полноту/актуальность/согласованность
        state = build_process_state(
            decision_at, avt_kip, ht_kip, lims_long, pak_long,
            kip_bounds=self.kip_bounds,
            stale_lims_minutes=self.stale_lims_minutes,
            stale_pak_minutes=self.stale_pak_minutes,
        )
        self.last_cycle_context = None  # каждый цикл начинается с чистого контекста
        self.run_logger.log(decision_at, "01_process_state", state)
        flags = state.quality_report.flags
        bus.send(
            "data_sync", "orchestrator", "ProcessState",
            f"{len(state.kip)} тегов КИП, {len(state.lab_points)} лаб. точек, {len(flags)} флагов, "
            f"sync_ok={state.quality_report.sync_ok}, стационарный режим={state.steady_regime}",
        )

        if not state.quality_report.sync_ok:
            return self._refusal(bus, decision_at, flags, "нет снимка КИП в пределах допуска на запрошенный момент времени.")

        blocking = [f for f in flags if f.code in REFUSAL_FLAG_CODES]
        if blocking:
            return self._refusal(
                bus, decision_at, flags,
                "реактор гидроочистки в переходном режиме (пуск/останов/разгон температуры) -- "
                "вне области применимости моделей качества и оптимизации; режим ведётся по "
                "регламенту пуска/останова. " + blocking[0].detail,
            )

        # Шаг 3: качество
        bus.send("orchestrator", "quality", "ProcessState", "оценить показатели качества")
        quality = self.quality_agent.assess(state)
        self.run_logger.log(decision_at, "02_quality_assessment", quality)
        bus.send(
            "quality", "orchestrator", "QualityAssessment",
            "; ".join(f"{e.metric}={e.value:.4g} [{e.source.value}, {e.confidence.value}]" for e in quality.current)
            + f" | общая уверенность {quality.overall_confidence.value}",
        )
        if quality.overall_confidence == ConfidenceLevel.REFUSE:
            return self._refusal(
                bus, decision_at, flags,
                "не удалось получить оценку одного или нескольких обязательных "
                "показателей качества (нет свежих ЛИМС/ПАК и нет расчётной формулы).",
            )

        # Шаг 4: надёжность
        bus.send("orchestrator", "reliability", "ProcessState", "оценить тяжесть режима")
        risk = self.reliability_agent.assess(state)
        self.run_logger.log(decision_at, "03_reliability_assessment", risk)
        bus.send(
            "reliability", "orchestrator", "EquipmentRiskAssessment",
            f"индекс {risk.severity_index:.2f} ({risk.risk_class.value}), hard_stop={risk.hard_stop}",
        )
        if risk.risk_class == RiskClass.UNKNOWN:
            return self._refusal(
                bus, decision_at, flags,
                "нет обязательных признаков для оценки риска оборудования: "
                + ", ".join(risk.missing_inputs),
            )

        quality_needs_action = any(v.risk_class in ACTION_NEEDED_QUALITY_RISK for v in quality.violations)
        equipment_needs_action = risk.risk_class in ACTION_NEEDED_EQUIPMENT_RISK

        key_state = {
            e.metric: e.value for e in quality.current
        } | {"severity_index": risk.severity_index}

        if not quality_needs_action and not equipment_needs_action:
            bus.send("orchestrator", "operator", "Recommendation", "риска нет -- действий не требуется")
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
        bus.send(
            "orchestrator", "optimization", "OptimizationRequest",
            f"нарушения/риски: {', '.join(sorted(violated_metrics)) or 'нет'}; "
            f"риск оборудования {risk.risk_class.value}",
        )
        opt_result = self.optimization_agent.run(state, baseline_margins, risk, quality, violated_metrics)
        self.run_logger.log(decision_at, "04_optimization_result", opt_result)
        bus.send(
            "optimization", "orchestrator", "OptimizationResult",
            f"оценено {opt_result.candidates_evaluated}, допустимо {len(opt_result.feasible_candidates)}, "
            f"Парето {len(opt_result.pareto_front_ids)}",
        )

        if opt_result.no_feasible_solution:
            # сначала само нарушение, затем почему его нечем устранить
            problem = self._problem_text(quality, risk)
            return self._refusal(
                bus, decision_at, flags,
                problem[0].upper() + problem[1:] + ". " + (opt_result.no_feasible_reason or "нет допустимого варианта."),
                key_state=key_state,
            )

        # Шаг 8: выбрать и объяснить -- с независимой Guard-проверкой,
        # при BLOCK пробуем следующего по рангу кандидата (defense in depth).
        # Альтернативы (ТЗ п.3, роль Оркестратора: "...альтернативы...") --
        # остальные точки Парето-фронта, кроме выбранной.
        # Контекст цикла сохраняется целиком -- по нему LLM-оркестратор
        # (§9.1) работает через инструменты, не пересчитывая ничего сам.
        # Детерминированный результат при этом уже получен и остаётся
        # источником истины.
        self.last_cycle_context = {
            "decision_at": decision_at,
            "state": state,
            "quality": quality,
            "risk": risk,
            "optimization": opt_result,
            "all_candidates": list(getattr(opt_result, "feasible_candidates", [])) + self._rejected_candidates(opt_result),
            "violated_metrics": violated_metrics,
            "key_state": key_state,
            "blending": self._blend_assessment(state),
        }

        for candidate in opt_result.feasible_candidates:
            bus.send("orchestrator", "guard", "ControlCandidate", f"проверить {candidate.candidate_id}")
            built = self.recommendation_for(self.last_cycle_context, candidate)
            recommendation, verdict = built
            bus.send("guard", "orchestrator", "GuardReport", f"{candidate.candidate_id}: {verdict.value}")
            if verdict != GuardVerdict.BLOCK:
                bus.send("orchestrator", "operator", "Recommendation",
                         f"рекомендован {candidate.candidate_id}, альтернатив {len(recommendation.alternatives)}")
                return recommendation

        return self._refusal(
            bus, decision_at, flags,
            "все допустимые по Агенту оптимизации варианты не прошли независимую "
            "P&ID-проверку Guard (см. ARCHITECTURE.md §6.5).",
            key_state=key_state,
        )

    def recommendation_for(self, context: dict, candidate):
        """Собирает карточку для КОНКРЕТНОГО кандидата, прогнав его через
        независимый Guard. Возвращает (карточка, вердикт Guard).

        Единственный путь превращения кандидата в рекомендацию -- и
        детерминированный цикл, и LLM-оркестратор идут именно через него.
        Поэтому выбор модели не может обойти Guard: другого способа
        получить карточку в системе просто нет.
        """
        decision_at = context["decision_at"]
        state, quality, risk = context["state"], context["quality"], context["risk"]
        opt_result = context["optimization"]
        guard_report = self.guard.review(candidate, decision_at, state)

        by_id = {c.candidate_id: c for c in opt_result.feasible_candidates}
        alternatives = [
            by_id[cid] for cid in opt_result.pareto_front_ids
            if cid != candidate.candidate_id and cid in by_id
        ]
        recommendation = Recommendation(
            decision_at=decision_at,
            key_state=context["key_state"],
            problem_or_risk=self._problem_text(quality, risk),
            proposed_actions=candidate.actions,
            expected_effect={
                **{e.metric: e.value for e in candidate.predicted_quality},
                "equipment_risk_severity": candidate.predicted_risk.severity_index,
                "energy_cost_proxy": candidate.energy_cost_proxy or 0.0,
            },
            constraints_checked=guard_report.checks,
            confidence=quality.overall_confidence,
            confidence_warnings=self._warnings(state) + candidate.caveats,
            explanation=explain_recommendation(candidate, quality, risk, opt_result.feasible_candidates),
            economic_effect=candidate.economics,
            is_refusal=False,
            alternatives=alternatives,
        )
        return recommendation, guard_report.final_verdict

    @staticmethod
    def _rejected_candidates(opt_result) -> list:
        """Отклонённые варианты для инструмента explain_rejections.
        Агент оптимизации не хранит их в OptimizationResult (контракт
        отдаёт только допустимые), поэтому берём из необязательного поля,
        если оно есть -- иначе список пуст и инструмент честно это скажет."""
        return list(getattr(opt_result, "rejected_candidates", []) or [])

    def _blend_assessment(self, state):
        agent = getattr(self.optimization_agent, "blending_agent", None)
        if agent is None:
            return None
        try:
            return agent.assess(state)
        except (KeyError, ValueError, ZeroDivisionError):
            return None

    def _problem_text(self, quality, risk) -> str:
        bits = []
        if quality.violations:
            # все показатели на пороге действия, худший первым; если таких нет
            # (действие из-за оборудования) -- самый близкий к пределу
            worst = worst_violation(quality)
            acting = [v for v in quality.violations if v.risk_class in ACTION_NEEDED_QUALITY_RISK and v is not worst]
            bits.append("риск нарушения: " + "; ".join(describe_violation(quality, v) for v in [worst, *acting]))
        if risk.risk_class in ACTION_NEEDED_EQUIPMENT_RISK:
            bits.append(f"тяжёлый режим оборудования (индекс {risk.severity_index:.2f})")
        return "; ".join(bits) if bits else "см. объяснение"

    def _warnings(self, state) -> list[str]:
        return operator_warnings(state.quality_report.flags)

    def _refusal(self, bus, decision_at, flags, reason: str, key_state: dict | None = None) -> Recommendation:
        bus.send("orchestrator", "operator", "Refusal", reason[:200])
        return Recommendation(
            decision_at=decision_at,
            key_state=key_state or {},
            problem_or_risk=reason,
            proposed_actions=[],
            expected_effect={},
            constraints_checked=[],
            confidence=ConfidenceLevel.REFUSE,
            confidence_warnings=operator_warnings(flags),
            explanation=explain_refusal(reason),
            is_refusal=True,
        )


def operator_warnings(flags) -> list[str]:
    shown, folded = [], 0
    for f in flags:
        if f.code.startswith("stale_") and f.tag_or_point not in _CONTROL_LAB_POINTS:
            folded += 1
            continue
        shown.append(f"{f.code}: {f.tag_or_point} -- {f.detail}")
    if folded:
        shown.append(
            f"stale_other: ещё {folded} лабораторных точек устарели (не используются для оценки "
            "нормируемых показателей; полный список -- в журнале цикла 01_process_state)"
        )
    return shown
