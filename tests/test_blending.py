"""Тесты контура блендинга (ARCHITECTURE.md §6.6).

Проверяется не "код запускается", а свойства, на которых держится
безопасность контура: доли всегда дают 100 %, правила смешения ведут
себя физически, Guard независимо ловит подделку долей.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from neftekod_mas.blending.blending_agent import BlendingAgent
from neftekod_mas.blending.rules import (
    BlendingError,
    Component,
    blend_cloud_point,
    blend_density,
    blend_distillation,
    blend_mass_linear,
    volume_fractions,
)
from neftekod_mas.schemas import ControlAction, ControlCandidate, DataSource, ProcessState, TagReading

LIGHT, HEAVY = 832.7, 856.8


def _comps(w_heavy: float):
    return [Component("light", 1.0 - w_heavy, LIGHT), Component("heavy", w_heavy, HEAVY)]


def test_shares_must_sum_to_one():
    with pytest.raises(BlendingError):
        blend_density([Component("a", 0.4, LIGHT), Component("b", 0.4, HEAVY)])


def test_blend_density_is_between_components_and_monotone():
    values = [blend_density(_comps(w)) for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values[0] == pytest.approx(LIGHT)
    assert values[-1] == pytest.approx(HEAVY)
    assert values == sorted(values), "плотность обязана расти с долей тяжёлого компонента"


def test_volume_fractions_differ_from_mass_fractions():
    """Объёмная доля лёгкого компонента больше массовой -- он менее плотный.
    Если эти доли перепутать, плотность смеси поедет на единицы кг/м3."""
    mass_w = 0.5
    vol = volume_fractions(_comps(mass_w))
    assert vol[0] > mass_w > vol[1]


def test_mass_linear_blending_is_exact_for_sulfur():
    """Сера в % масс. смешивается по массе -- это сохранение массы, не корреляция."""
    assert blend_mass_linear(_comps(0.5), [0.4318, 1.2466]) == pytest.approx(0.8392)


def test_distillation_blend_is_bounded_by_components():
    light = {"IBP.T": 202, "50%.T": 251.5, "90%.T": 283, "95%.T": 296, "EBP.T": 310}
    heavy = {"IBP.T": 195, "50%.T": 305, "90%.T": 349, "95%.T": 363, "EBP.T": 370}
    mixed = blend_distillation(_comps(0.5), [light, heavy], targets=(50.0, 95.0))
    assert light["50%.T"] <= mixed[50.0] <= heavy["50%.T"]
    assert light["95%.T"] <= mixed[95.0] <= heavy["95%.T"]


def test_distillation_blend_rises_with_heavy_share():
    light = {"IBP.T": 202, "50%.T": 251.5, "90%.T": 283, "95%.T": 296, "EBP.T": 310}
    heavy = {"IBP.T": 195, "50%.T": 305, "90%.T": 349, "95%.T": 363, "EBP.T": 370}
    low = blend_distillation(_comps(0.4), [light, heavy], targets=(95.0,))[95.0]
    high = blend_distillation(_comps(0.7), [light, heavy], targets=(95.0,))[95.0]
    assert high > low


def test_cloud_point_index_is_not_linear_average():
    """Низкотемпературные свойства смешиваются нелинейно: смесь ближе к
    худшему (более высокому) компоненту, чем простое среднее. Ровно это и
    ловит индекс смешения, ради чего он и вводится."""
    values = [-23.0, 3.0]
    indexed = blend_cloud_point(_comps(0.5), values)
    linear = sum(w * v for w, v in zip(volume_fractions(_comps(0.5)), values, strict=True))
    assert indexed > linear


def test_cloud_point_rejects_impossible_temperature():
    with pytest.raises(BlendingError):
        blend_cloud_point(_comps(0.5), [-300.0, 0.0])


# --- Агент -----------------------------------------------------------

MODEL = {
    "components": {
        "light": {"description": "лёгкий", "flow_tag": "avt:F32", "lims_point": "L", "density_kg_m3": LIGHT},
        "heavy": {"description": "тяжёлый", "flow_tag": "avt:F30", "lims_point": "H", "density_kg_m3": HEAVY},
    },
    "component_sulfur": {"status": "ok", "light_pct_mass": 0.43, "heavy_pct_mass": 1.25},
    "typical_component_curves": {
        "light": {"IBP.T": 202, "50%.T": 251.5, "90%.T": 283, "95%.T": 296, "EBP.T": 310},
        "heavy": {"IBP.T": 195, "50%.T": 305, "90%.T": 349, "95%.T": 363, "EBP.T": 370},
    },
    "validation": {"t95_c": {"accepted": True, "bias": 0.0}, "density_kg_m3": {"accepted": False}},
    "share_sensitivity": {},
    "feed_to_product_propagation": {"t95_c": {"status": "ok", "slope": 0.8285}},
    "observed_shares": {"p05": 0.50, "p95": 0.70},
}


def _state(f30: float, f32: float) -> ProcessState:
    at = datetime(2025, 12, 1, 20, 0, 0)
    kip = {
        "avt:F30": TagReading(tag_id="avt:F30", value=f30, unit="т/ч", timestamp=at, source=DataSource.KIP),
        "avt:F32": TagReading(tag_id="avt:F32", value=f32, unit="т/ч", timestamp=at, source=DataSource.KIP),
    }
    from neftekod_mas.schemas import DataQualityReport
    return ProcessState(decision_at=at, kip=kip, lab_points={},
                        quality_report=DataQualityReport(decision_at=at, sync_ok=True), steady_regime=True)


def test_agent_shares_always_sum_to_100():
    agent = BlendingAgent(MODEL)
    for f30, f32 in ((127.0, 90.0), (10.0, 200.0), (150.0, 60.0)):
        assert agent.assess(_state(f30, f32)).share_sum_pct == pytest.approx(100.0)


def test_agent_refuses_when_flow_is_missing_or_dead():
    agent = BlendingAgent(MODEL)
    assessment = agent.assess(_state(0.0, 90.0))
    assert assessment.available is False
    assert assessment.unavailable_reason


def test_agent_only_reports_metrics_that_passed_backtest():
    """Плотность бэктест не прошла -- агент не имеет права её выдавать,
    даже несмотря на то, что физически считает её точно."""
    agent = BlendingAgent(MODEL)
    quality = agent.assess(_state(127.0, 90.0)).blend_quality
    assert "t95_c" in quality
    assert "density_kg_m3" not in quality


def test_candidates_preserve_total_flow_and_stay_in_observed_range():
    agent = BlendingAgent(MODEL)
    state = _state(127.0, 90.0)
    total = 127.0 + 90.0
    options = agent.share_candidates(state)
    assert options, "должен быть хотя бы один кандидат"
    for opt in options:
        assert sum(opt["flows"].values()) == pytest.approx(total)
        assert opt["share_sum_pct"] == pytest.approx(100.0)
        assert 50.0 <= opt["heavy_share_pct"] <= 70.0


def test_more_heavy_component_raises_predicted_t95():
    agent = BlendingAgent(MODEL)
    state = _state(127.0, 90.0)
    for opt in agent.share_candidates(state):
        delta = opt["product_effect"]["t95_c"]
        if opt["heavy_share_pct"] > opt["current_heavy_share_pct"]:
            assert delta > 0
        else:
            assert delta < 0


# --- Guard: независимая проверка суммы долей -------------------------

def _guard():
    from neftekod_mas.orchestrator.guard import Guard
    from neftekod_mas.tags.pid_graph import TagGraph
    graph = TagGraph()
    cfg = {"installations": {"blending": {"variables": [
        {"name": "blend_share_heavy", "tag": "F30"},
        {"name": "blend_share_light", "tag": "F32"},
    ]}}}
    return Guard(graph, cfg, {}, hard_constraints={})


def _candidate(actions):
    from neftekod_mas.schemas import EquipmentRiskAssessment, RiskClass
    at = datetime(2025, 12, 1, 20, 0, 0)
    return ControlCandidate(
        candidate_id="b1", actions=actions, predicted_quality=[],
        predicted_risk=EquipmentRiskAssessment(
            decision_at=at, severity_index=0.0, risk_class=RiskClass.LOW, factors=[], hard_stop=False),
        feasible=True,
    )


def _action(name, tag, cur, new):
    return ControlAction(variable_name=name, tag=tag, current_value=cur, recommended_value=new, unit="т/ч")


def _blend_check(report):
    return next(c for c in report.checks if c.check_name == "blend_sum_100")


def test_guard_passes_pure_share_redistribution():
    report = _guard().review(_candidate([
        _action("blend_share_heavy", "avt:F30", 127.0, 133.0),
        _action("blend_share_light", "avt:F32", 90.0, 84.0),
    ]), datetime(2025, 12, 1, 20, 0, 0))
    assert _blend_check(report).verdict.value == "pass"


def test_guard_blocks_when_pool_flow_changes():
    """Если суммарный расход уехал, это уже не перераспределение долей,
    а изменение загрузки -- скрывать такое под видом блендинга нельзя."""
    report = _guard().review(_candidate([
        _action("blend_share_heavy", "avt:F30", 127.0, 140.0),
        _action("blend_share_light", "avt:F32", 90.0, 84.0),
    ]), datetime(2025, 12, 1, 20, 0, 0))
    assert _blend_check(report).verdict.value == "block"


def test_guard_blocks_single_sided_share_change():
    report = _guard().review(_candidate([
        _action("blend_share_heavy", "avt:F30", 127.0, 133.0),
    ]), datetime(2025, 12, 1, 20, 0, 0))
    assert _blend_check(report).verdict.value == "block"


def test_guard_ignores_non_blending_candidates():
    report = _guard().review(_candidate([
        _action("ht_inlet_temp_c", "242000:T5", 376.0, 380.0),
    ]), datetime(2025, 12, 1, 20, 0, 0))
    assert _blend_check(report).verdict.value == "not_applicable"
