"""Тесты LLM-оркестратора (ARCHITECTURE.md §9.1).

Проверяется главное свойство: что бы ни выдала модель -- в том числе
враждебное или бессмысленное -- она не может ни выпустить непроверенную
рекомендацию, ни обойти Guard, ни ухудшить детерминированный результат.
Модель здесь подменяется заглушкой со сценарием ответов: тесты не
зависят ни от какой запущенной LLM.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from neftekod_mas.orchestrator.llm_orchestrator import (
    LLMOrchestrator,
    _finalize_from_content,
    _local_opener,
)
from neftekod_mas.schemas import (
    ConfidenceLevel,
    ControlAction,
    ControlCandidate,
    DataQualityReport,
    EquipmentRiskAssessment,
    GuardVerdict,
    OptimizationResult,
    ProcessState,
    QualityAssessment,
    QualityMetricEstimate,
    DataSource,
    Recommendation,
    RiskClass,
)

AT = datetime(2025, 12, 1, 20, 0, 0)


def _candidate(cid: str) -> ControlCandidate:
    return ControlCandidate(
        candidate_id=cid,
        actions=[ControlAction(variable_name="ht_inlet_temp_c", tag="242000:T5",
                               current_value=376.3, recommended_value=380.2, unit="°C")],
        predicted_quality=[QualityMetricEstimate(
            metric="sulfur_mg_kg", value=6.59, unit="мг/кг",
            source=DataSource.LITERATURE_PROXY, confidence=ConfidenceLevel.LOW)],
        predicted_risk=EquipmentRiskAssessment(decision_at=AT, severity_index=0.0,
                                               risk_class=RiskClass.LOW, factors=[], hard_stop=False),
        feasible=True, score=1.0,
    )


class FakeDeterministic:
    """Заглушка детерминированного оркестратора: отдаёт готовую карточку
    и контекст, как настоящий."""

    def __init__(self, *, is_refusal: bool = False, guard_verdict=GuardVerdict.PASS):
        self.guard_verdict = guard_verdict
        self.baseline = Recommendation(
            decision_at=AT, key_state={"sulfur_mg_kg": 9.97},
            problem_or_risk="сера близко к пределу",
            proposed_actions=[] if is_refusal else _candidate("c20").actions,
            expected_effect={}, constraints_checked=[],
            confidence=ConfidenceLevel.MEDIUM, confidence_warnings=[],
            explanation="детерминированное объяснение", is_refusal=is_refusal,
        )
        candidates = [] if is_refusal else [_candidate("c20")]
        self.last_cycle_context = {
            "decision_at": AT,
            "state": ProcessState(decision_at=AT, kip={}, lab_points={},
                                  quality_report=DataQualityReport(decision_at=AT, sync_ok=True),
                                  steady_regime=True),
            "quality": QualityAssessment(decision_at=AT, current=[], violations=[],
                                         overall_confidence=ConfidenceLevel.MEDIUM),
            "risk": EquipmentRiskAssessment(decision_at=AT, severity_index=0.0,
                                            risk_class=RiskClass.LOW, factors=[], hard_stop=False),
            "optimization": OptimizationResult(
                decision_at=AT, candidates_evaluated=30,
                feasible_candidates=candidates,
                pareto_front_ids=[c.candidate_id for c in candidates],
                recommended_candidate_id=candidates[0].candidate_id if candidates else None,
                no_feasible_solution=is_refusal),
            "all_candidates": candidates,
            "violated_metrics": {"sulfur_mg_kg"},
            "key_state": {"sulfur_mg_kg": 9.97},
            "blending": None,
        }
        self.built_for: list[str] = []

    def run_cycle(self, *args, **kwargs):
        return self.baseline

    def recommendation_for(self, context, candidate):
        self.built_for.append(candidate.candidate_id)
        rec = self.baseline.model_copy(deep=True)
        rec.proposed_actions = candidate.actions
        return rec, self.guard_verdict


class ScriptedClient:
    """Возвращает заранее заданные ответы модели по очереди."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = 0

    def __call__(self, messages, tools):
        self.calls += 1
        if not self.messages:
            return {"content": "", "tool_calls": []}
        return self.messages.pop(0)


def _tool_call(name, arguments, cid="1"):
    return {"content": "", "tool_calls": [
        {"id": cid, "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
    ]}


def test_valid_choice_is_accepted_and_marked_as_llm_led():
    det = FakeDeterministic()
    client = ScriptedClient([
        _tool_call("assess_quality", {}),
        _tool_call("list_candidates", {}),
        _tool_call("finalize", {"action": "recommend", "candidate_id": "c20",
                                "rationale": "сера у предела, вариант её снижает"}),
    ])
    rec = LLMOrchestrator(det, client).run_cycle()
    assert rec.orchestration_mode == "llm"
    assert rec.llm_status == "ok"
    assert det.built_for == ["c20"], "карточка обязана собираться тем же кодом, что и обычно"


def test_invented_candidate_id_is_refused():
    """Модель не может предложить уставку мимо Агента оптимизации:
    единственный канал -- candidate_id из проверенного списка."""
    det = FakeDeterministic()
    client = ScriptedClient([
        _tool_call("finalize", {"action": "recommend", "candidate_id": "c999",
                                "rationale": "поднять температуру до 500"}),
    ])
    rec = LLMOrchestrator(det, client).run_cycle()
    assert rec.orchestration_mode == "deterministic"
    assert "не из числа проверенных" in rec.llm_status


def test_guard_block_overrides_model_choice():
    det = FakeDeterministic(guard_verdict=GuardVerdict.BLOCK)
    client = ScriptedClient([
        _tool_call("finalize", {"action": "recommend", "candidate_id": "c20", "rationale": "ок"}),
    ])
    rec = LLMOrchestrator(det, client).run_cycle()
    assert rec.orchestration_mode == "deterministic"
    assert "Guard" in rec.llm_status


def test_model_cannot_refuse_when_verified_solution_exists():
    """Отказ модели при наличии проверенного решения -- её ошибка, а не
    новая информация: все проверки уже сделаны детерминированно."""
    det = FakeDeterministic()
    client = ScriptedClient([
        _tool_call("finalize", {"action": "refuse", "rationale": "мне кажется, лучше ничего не делать"}),
    ])
    rec = LLMOrchestrator(det, client).run_cycle()
    assert rec.orchestration_mode == "deterministic"
    assert rec.proposed_actions, "детерминированная рекомендация обязана сохраниться"


def test_model_refusal_is_accepted_when_deterministic_also_refuses():
    det = FakeDeterministic(is_refusal=True)
    client = ScriptedClient([
        _tool_call("finalize", {"action": "refuse", "rationale": "нет рычага"}),
    ])
    rec = LLMOrchestrator(det, client).run_cycle()
    assert rec.is_refusal is True
    assert rec.llm_status == "refused"


def test_unavailable_model_falls_back_to_deterministic_card():
    det = FakeDeterministic()

    def broken(messages, tools):
        raise TimeoutError("нет связи")

    rec = LLMOrchestrator(det, broken).run_cycle()
    assert rec.orchestration_mode == "deterministic"
    assert rec.proposed_actions == det.baseline.proposed_actions
    assert "недоступна" in rec.llm_status


def test_endless_loop_is_cut_off_by_step_budget():
    det = FakeDeterministic()
    client = ScriptedClient([_tool_call("assess_quality", {}) for _ in range(50)])
    rec = LLMOrchestrator(det, client, max_tool_calls=4).run_cycle()
    assert rec.orchestration_mode == "deterministic"
    assert client.calls <= 4


def test_tool_calls_are_recorded_in_trace():
    det = FakeDeterministic()
    client = ScriptedClient([
        _tool_call("assess_quality", {}),
        _tool_call("finalize", {"action": "recommend", "candidate_id": "c20", "rationale": "ок"}),
    ])
    rec = LLMOrchestrator(det, client).run_cycle()
    tool_messages = [m for m in rec.trace if m.topic == "ToolCall"]
    assert [m.recipient for m in tool_messages] == ["assess_quality", "finalize"]


def test_missing_cycle_context_falls_back():
    det = FakeDeterministic()
    det.last_cycle_context = None
    rec = LLMOrchestrator(det, ScriptedClient([])).run_cycle()
    assert rec.orchestration_mode == "deterministic"


def test_finalize_written_as_text_is_parsed():
    parsed = _finalize_from_content(
        'Выбираю вариант. finalize:\n```json\n{"action": "recommend", "selected_candidate_id": "c20"}\n```'
    )
    assert parsed["action"] == "recommend"
    assert parsed["candidate_id"] == "c20"
    assert "```" not in parsed["rationale"]


def test_plain_text_is_not_mistaken_for_finalize():
    assert _finalize_from_content("Сера близка к пределу, думаю дальше.") is None


def test_local_endpoint_bypasses_system_proxy(monkeypatch):
    """В закрытом контуре с заданным http_proxy запрос к локальной модели
    не должен уходить в прокси -- иначе приходит 502 вместо ответа."""
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:7897")

    def proxies_of(url):
        opener = _local_opener(url)
        return [h.proxies for h in opener.handlers if type(h).__name__ == "ProxyHandler"]

    # Для локального адреса обработчик прокси вообще не регистрируется.
    assert proxies_of("http://127.0.0.1:11434/v1") == []
    assert proxies_of("http://localhost:11434/v1") == []
    # Для удалённого сервера (модель на другом узле кластера) прокси
    # остаётся в силе -- отключается только локальный случай.
    remote = proxies_of("http://gpu-node.cluster:11434/v1")
    assert remote and remote[0].get("http") == "http://proxy.invalid:7897"
