"""Тесты эксплуатационных границ и ресурса катализатора (§6.3.1).

Проверяется то, ради чего эти данные вообще извлекались: цензурированный
отсчёт не должен выглядеть как нормальное измерение, а ресурс катализатора
не должен подменять собой оценку тяжести режима.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from neftekod_mas.reliability.reliability_agent import ReliabilityAgent
from neftekod_mas.schemas import DataQualityReport, DataSource, ProcessState, RiskClass, TagReading

AT = datetime(2025, 12, 1, 20, 0, 0)

BOUNDS = {
    "_meta": {"is_assumption": True, "note": "перцентили истории"},
    "P8": {"0.9": 0.20, "0.99": 0.23, "1.0": 0.25},
    "T5": {"0.9": 378.0, "0.99": 385.0, "1.0": 390.0},
    "T6": {"0.9": 372.0, "0.99": 380.0, "1.0": 385.0},
    "T11": {"0.9": 374.0, "0.99": 382.0, "1.0": 388.0},
}

LIMITS = {
    "saturation": {
        "P8": {"observed_max": 0.2503, "looks_saturated": True,
               "description": "Перепад давления на реакторе Р-202"},
        "F9": {"observed_max": 283.0, "looks_saturated": False, "description": "Расход сырья"},
    },
    "catalyst_deactivation": {"status": "ok", "median_c_per_month": 1.003},
    "reactor_temperature_ceiling_c": {"value": 388.1},
}


def _state(**tags) -> ProcessState:
    base = {"242000:P8": 0.15, "242000:T5": 370.0, "242000:T6": 365.0, "242000:T11": 366.0}
    base.update(tags)
    kip = {k: TagReading(tag_id=k, value=v, unit="", timestamp=AT, source=DataSource.KIP)
           for k, v in base.items()}
    return ProcessState(decision_at=AT, kip=kip, lab_points={},
                        quality_report=DataQualityReport(decision_at=AT, sync_ok=True))


def _factor(assessment, tag):
    return next(f for f in assessment.factors if f.tag_id == tag and f.affects_index)


def test_normal_reading_is_not_marked_censored():
    a = ReliabilityAgent(BOUNDS, equipment_limits=LIMITS).assess(_state())
    f = _factor(a, "242000:P8")
    assert "УПОР" not in f.description
    assert f.contribution == 0.0


def test_reading_at_instrument_ceiling_is_treated_as_censored():
    """0.25 МПа -- предел шкалы датчика, а не реальный максимум. Значение
    на упоре означает 'не меньше', поэтому тяжесть берётся консервативно."""
    a = ReliabilityAgent(BOUNDS, equipment_limits=LIMITS).assess(_state(**{"242000:P8": 0.2503}))
    f = _factor(a, "242000:P8")
    assert "УПОР" in f.description
    assert f.contribution >= 1.0
    assert f.is_assumption is True
    assert "может быть выше" in (f.assumption_note or "")
    assert a.risk_class == RiskClass.CRITICAL


def test_censoring_is_not_applied_to_unsaturated_tags():
    limits = {**LIMITS, "saturation": {"F9": LIMITS["saturation"]["F9"]}}
    a = ReliabilityAgent(BOUNDS, equipment_limits=limits).assess(_state(**{"242000:P8": 0.2503}))
    f = _factor(a, "242000:P8")
    assert "УПОР" not in f.description


def test_catalyst_resource_is_reported_but_does_not_drive_severity():
    """Исчерпание запаса по температуре -- повод планировать перегрузку,
    а не признак тяжёлого режима прямо сейчас."""
    agent = ReliabilityAgent(BOUNDS, equipment_limits=LIMITS)
    spacious = agent.assess(_state(**{"242000:T5": 360.0}))
    tight = agent.assess(_state(**{"242000:T5": 386.0}))

    cat_spacious = next(f for f in spacious.factors if not f.affects_index)
    cat_tight = next(f for f in tight.factors if not f.affects_index)
    assert cat_tight.contribution > cat_spacious.contribution
    assert "мес" in cat_tight.description
    # T5=386 выше p99, поэтому сам режим действительно тяжелее -- но
    # проверяем, что вклад вносит именно температурный фактор, а не ресурс.
    assert spacious.severity_index == pytest.approx(0.0)


def test_catalyst_factor_absent_without_limits_config():
    a = ReliabilityAgent(BOUNDS).assess(_state())
    assert all(f.affects_index for f in a.factors)
    assert a.severity_index == pytest.approx(0.0)


def test_censoring_changes_verdict_where_percentiles_alone_would_not():
    """Ключевой случай: отсчёт стоит на упоре шкалы, но по перцентилям
    истории он ещё "нормальный". Без знания об упоре система занизила бы
    тяжесть режима именно там, где риск важнее всего."""
    # Перцентили, при которых 0.25 -- обычное значение, а не экстремум.
    wide_bounds = {**BOUNDS, "P8": {"0.9": 0.30, "0.99": 0.40, "1.0": 0.50}}
    state = _state(**{"242000:P8": 0.2503})

    without = ReliabilityAgent(wide_bounds).assess(state)
    assert _factor(without, "242000:P8").contribution == 0.0
    assert without.risk_class == RiskClass.LOW

    with_limits = ReliabilityAgent(wide_bounds, equipment_limits=LIMITS).assess(state)
    assert _factor(with_limits, "242000:P8").contribution >= 1.0
    assert with_limits.risk_class == RiskClass.CRITICAL


def test_agent_works_without_limits_config():
    """Контур обязан работать и на одних перцентилях: файл границ
    необязателен, его отсутствие не должно ронять оценку."""
    a = ReliabilityAgent(BOUNDS).assess(_state())
    assert a.risk_class == RiskClass.LOW
    assert a.factors and all(f.affects_index for f in a.factors)
