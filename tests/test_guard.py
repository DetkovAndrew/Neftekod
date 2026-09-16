from datetime import datetime

import numpy as np

from neftekod_mas.data.sync import DataQualityReport
from neftekod_mas.optimization.joint_envelope import JointEnvelopeChecker
from neftekod_mas.orchestrator.guard import Guard
from neftekod_mas.schemas import (
    ConfidenceLevel,
    ControlAction,
    ControlCandidate,
    EquipmentRiskAssessment,
    GuardVerdict,
    ProcessState,
    RiskClass,
    TagReading,
    DataSource,
)
from neftekod_mas.tags.pid_graph import TagGraph, TagNode

CONTROL_VARIABLES = {
    "installations": {
        "avt": {
            "variables": [
                {"name": "avt_furnace_outlet_temp_c", "tag": "T55", "unit": "°C", "confidence": "confirmed_from_pid"},
            ]
        }
    }
}
CONTROL_BOUNDS = {"_meta": {}, "avt:T55": {"p05": 375.0, "p50": 381.0, "p95": 385.0}}


def _graph():
    g = TagGraph()
    g.add(TagNode(
        tag_id="T55", installation="avt", stage="K10", description="", unit="°C",
        physical_quantity="temperature", node_kind="controller", actuatable=True,
        confidence="confirmed_from_pid", controller_type="TC",
    ))
    return g


def _risk(decision_at):
    return EquipmentRiskAssessment(decision_at=decision_at, severity_index=0.0, risk_class=RiskClass.LOW, factors=[], hard_stop=False)


def _candidate(tag_name, tag, value):
    now = datetime(2023, 1, 1, 10, 0)
    action = ControlAction(variable_name=tag_name, tag=tag, current_value=380.0, recommended_value=value, unit="°C")
    return ControlCandidate(
        candidate_id="c1", actions=[action], predicted_quality=[], predicted_risk=_risk(now),
        feasible=True,
    )


def test_pass_for_known_variable_within_bounds():
    guard = Guard(_graph(), CONTROL_VARIABLES, CONTROL_BOUNDS)
    candidate = _candidate("avt_furnace_outlet_temp_c", "avt:T55", 382.0)
    report = guard.review(candidate, datetime(2023, 1, 1, 10, 0))
    assert report.final_verdict == GuardVerdict.PASS


def test_block_for_variable_not_in_registry():
    guard = Guard(_graph(), CONTROL_VARIABLES, CONTROL_BOUNDS)
    candidate = _candidate("some_unregistered_variable", "avt:P999", 5.0)
    report = guard.review(candidate, datetime(2023, 1, 1, 10, 0))
    assert report.final_verdict == GuardVerdict.BLOCK


def test_block_for_value_outside_validated_bounds():
    guard = Guard(_graph(), CONTROL_VARIABLES, CONTROL_BOUNDS)
    candidate = _candidate("avt_furnace_outlet_temp_c", "avt:T55", 500.0)  # далеко за p95
    report = guard.review(candidate, datetime(2023, 1, 1, 10, 0))
    assert report.final_verdict == GuardVerdict.BLOCK
    bounds_check = next(c for c in report.checks if c.check_name == "within_bounds")
    assert bounds_check.verdict == GuardVerdict.BLOCK


def test_block_when_no_candidate():
    guard = Guard(_graph(), CONTROL_VARIABLES, CONTROL_BOUNDS)
    report = guard.review(None, datetime(2023, 1, 1, 10, 0))
    assert report.final_verdict == GuardVerdict.BLOCK


def test_joint_envelope_warns_far_from_history_but_does_not_block():
    now = datetime(2023, 1, 1, 10, 0)
    points = np.array([[0.95]], dtype="float32")  # история сосредоточена у верхней границы
    je = JointEnvelopeChecker(points=points, tags=["avt:T55"], bounds=CONTROL_BOUNDS)
    guard = Guard(_graph(), CONTROL_VARIABLES, CONTROL_BOUNDS, joint_envelope=je)

    state = ProcessState(
        decision_at=now,
        kip={"avt:T55": TagReading(tag_id="avt:T55", value=380.0, unit="°C", timestamp=now, source=DataSource.KIP)},
        lab_points={},
        quality_report=DataQualityReport(decision_at=now, flags=[], sync_ok=True),
    )
    candidate = _candidate("avt_furnace_outlet_temp_c", "avt:T55", 375.0)  # далеко от истории (p05)

    report = guard.review(candidate, now, state)
    joint_check = next(c for c in report.checks if c.check_name == "joint_envelope")
    assert joint_check.verdict == GuardVerdict.WARN
    assert report.final_verdict == GuardVerdict.WARN  # WARN, не BLOCK -- см. docstring Guard
