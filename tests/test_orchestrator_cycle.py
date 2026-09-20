"""
Сквозные тесты полного цикла Оркестратора на синтетических (не реальных)
данных -- три обязательных демо-сценария из ТЗ п.6:
  1. стабильный период -> нет лишних действий
  2. риск ухудшения качества -> полная карточка рекомендации
  3. неполные/устаревшие данные -> мотивированный отказ
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from neftekod_mas.optimization.optimization_agent import OptimizationAgent
from neftekod_mas.orchestrator.guard import Guard
from neftekod_mas.orchestrator.orchestrator import Orchestrator
from neftekod_mas.quality.quality_agent import GODT_POINT2, QualityAgent
from neftekod_mas.reliability.reliability_agent import ReliabilityAgent
from neftekod_mas.tags.pid_graph import TagGraph, TagNode

HARD_CONSTRAINTS = {
    "product_diesel": {
        "sulfur_mg_kg": {"op": "<=", "limit": 10.0, "risk_margin_high": 1.0, "risk_margin_medium": 2.5},
        "t95_c": {"op": "<=", "limit": 360.0, "risk_margin_high": 5.0, "risk_margin_medium": 15.0},
        "cetane_number": {"op": ">=", "limit": 51.0, "risk_margin_high": 1.0, "risk_margin_medium": 2.0},
        "cfpp_c": {"op": "<=", "limit": None, "risk_margin_high": 3.0, "risk_margin_medium": 8.0},
    }
}

CONTROL_VARIABLES = {
    "installations": {
        "hydrotreating_242000": {
            "variables": [
                {"name": "ht_inlet_temp_c", "tag": "T5", "unit": "°C", "confidence": "assumption"},
            ]
        }
    }
}

CONTROL_BOUNDS = {"_meta": {}, "242000:T5": {"p05": 350.0, "p50": 365.0, "p95": 385.0}}

RELIABILITY_BOUNDS = {
    "_meta": {"is_assumption": True, "note": "test"},
    "P8": {"0.9": 0.21, "0.95": 0.22, "0.99": 0.24, "1.0": 0.25},
    "T5": {"0.9": 381.0, "0.95": 384.0, "0.99": 388.0, "1.0": 393.0},
    "T6": {"0.9": 376.0, "0.95": 380.0, "0.99": 385.0, "1.0": 391.0},
    "T11": {"0.9": 375.0, "0.95": 379.0, "0.99": 383.0, "1.0": 388.0},
}

OBJECTIVE_WEIGHTS = {
    "quality_margin_improvement": 1.0,
    "equipment_risk_severity": 1.5,
    "throughput_proxy": 0.3,
    "energy_cost_proxy": 0.3,
}


def _empty_df(cols):
    return pd.DataFrame(columns=cols)


def _kip_df(start, n, tags):
    idx = pd.date_range(start, periods=n, freq="10min")
    return pd.DataFrame(tags, index=idx)


def _make_orchestrator():
    graph = TagGraph()
    graph.add(TagNode(
        tag_id="T5", installation="242000", stage="reactor", description="",
        unit="°C", physical_quantity="temperature", node_kind="sensor",
        actuatable=False, confidence="assumption",
    ))
    qa = QualityAgent(HARD_CONSTRAINTS)
    ra = ReliabilityAgent(RELIABILITY_BOUNDS)
    oa = OptimizationAgent(CONTROL_VARIABLES, CONTROL_BOUNDS, HARD_CONSTRAINTS, OBJECTIVE_WEIGHTS, qa, ra)
    guard = Guard(graph, CONTROL_VARIABLES, CONTROL_BOUNDS)
    return Orchestrator(qa, ra, oa, guard)


def test_stable_period_no_action():
    start = datetime(2023, 1, 1, 0, 0)
    n = 10
    avt = _kip_df(start, n, {})
    ht = _kip_df(start, n, {"T5": [365.0] * n, "T6": [363.0] * n, "T11": [364.0] * n, "P8": [0.17] * n, "W7": [0.17] * n})
    decision_at = start + timedelta(minutes=30)
    lims = pd.DataFrame({
        "point_label": [GODT_POINT2] * 4,
        "param": ["Mg.Sulfur", "95%.T", "CetaneNumber", "CFPP"],
        "unit": ["мг/кг", "°C", "ед.цет.ч.", "°C"],
        "measured_at": [decision_at - timedelta(hours=1)] * 4,
        "value": [3.0, 340.0, 55.0, -10.0],  # все далеко от пределов
    })
    pak = _empty_df(["param", "unit", "measured_at", "value"])

    orch = _make_orchestrator()
    rec = orch.run_cycle(decision_at, avt, ht, lims, pak)

    assert rec.is_refusal is False
    assert rec.proposed_actions == []
    assert "нарушений не обнаружено" in rec.explanation or "не создаёт" in rec.explanation or "не требуется" in rec.explanation


def test_quality_risk_with_incomplete_prediction_refuses():
    start = datetime(2023, 1, 1, 0, 0)
    n = 10
    avt = _kip_df(start, n, {})
    ht = _kip_df(start, n, {"T5": [365.0] * n, "T6": [363.0] * n, "T11": [364.0] * n, "P8": [0.17] * n, "W7": [0.17] * n})
    decision_at = start + timedelta(minutes=30)
    lims = pd.DataFrame({
        "point_label": [GODT_POINT2] * 4,
        "param": ["Mg.Sulfur", "95%.T", "CetaneNumber", "CFPP"],
        "unit": ["мг/кг", "°C", "ед.цет.ч.", "°C"],
        "measured_at": [decision_at - timedelta(hours=1)] * 4,
        "value": [9.5, 340.0, 55.0, -10.0],  # сера почти на пределе -> HIGH risk
    })
    pak = _empty_df(["param", "unit", "measured_at", "value"])

    orch = _make_orchestrator()
    rec = orch.run_cycle(decision_at, avt, ht, lims, pak)

    assert rec.is_refusal is True
    assert rec.proposed_actions == []


def test_stale_data_period_refuses():
    start = datetime(2023, 1, 1, 0, 0)
    n = 10
    avt = _kip_df(start, n, {})
    ht = _kip_df(start, n, {"T5": [365.0] * n, "T6": [363.0] * n, "T11": [364.0] * n, "P8": [0.17] * n, "W7": [0.17] * n})
    decision_at = start + timedelta(minutes=30)
    # ЛИМС/ПАК полностью отсутствуют -> ни одного показателя не оценить
    lims = _empty_df(["point_label", "param", "unit", "measured_at", "value"])
    pak = _empty_df(["param", "unit", "measured_at", "value"])

    orch = _make_orchestrator()
    rec = orch.run_cycle(decision_at, avt, ht, lims, pak)

    assert rec.is_refusal is True
    assert rec.confidence.value == "refuse"
    assert "Надёжной рекомендации нет" in rec.explanation
