from datetime import datetime

from neftekod_mas.data.sync import DataQualityReport
from neftekod_mas.reliability.reliability_agent import ReliabilityAgent
from neftekod_mas.schemas import DataSource, ProcessState, RiskClass, TagReading

BOUNDS = {
    "_meta": {"is_assumption": True, "note": "test bounds"},
    "P8": {"0.01": 0.1, "0.05": 0.12, "0.5": 0.17, "0.9": 0.21, "0.95": 0.22, "0.99": 0.24, "1.0": 0.25},
    "T5": {"0.01": 345.0, "0.05": 353.0, "0.5": 370.0, "0.9": 381.0, "0.95": 384.0, "0.99": 388.0, "1.0": 393.0},
    "T6": {"0.01": 338.0, "0.05": 346.0, "0.5": 363.0, "0.9": 376.0, "0.95": 380.0, "0.99": 385.0, "1.0": 391.0},
    "T11": {"0.01": 341.0, "0.05": 349.0, "0.5": 364.0, "0.9": 375.0, "0.95": 379.0, "0.99": 383.0, "1.0": 388.0},
}


def _state(**kip_values) -> ProcessState:
    now = datetime(2023, 6, 1, 10, 0)
    kip = {
        f"242000:{k}": TagReading(tag_id=f"242000:{k}", value=v, unit="", timestamp=now, source=DataSource.KIP)
        for k, v in kip_values.items()
    }
    return ProcessState(
        decision_at=now, kip=kip, lab_points={},
        quality_report=DataQualityReport(decision_at=now, flags=[], sync_ok=True),
    )


def test_low_severity_at_median_operating_point():
    state = _state(P8=0.17, T5=370.0, T6=363.0, T11=364.0)
    risk = ReliabilityAgent(BOUNDS).assess(state)
    assert risk.risk_class == RiskClass.LOW
    assert risk.hard_stop is False


def test_high_severity_and_hard_stop_on_elevated_delta_p():
    # P8 около p99 -> подозрение на закоксовывание катализатора
    state = _state(P8=0.245, T5=370.0, T6=363.0, T11=364.0)
    risk = ReliabilityAgent(BOUNDS).assess(state)
    assert risk.risk_class in (RiskClass.HIGH, RiskClass.CRITICAL)
    assert risk.hard_stop is True
    p8_factor = next(f for f in risk.factors if f.tag_id == "242000:P8")
    assert p8_factor.is_assumption is True


def test_missing_tags_yield_zero_severity_not_crash():
    state = _state()  # пусто -- ни одного тега нет в состоянии
    risk = ReliabilityAgent(BOUNDS).assess(state)
    assert risk.severity_index == 0.0
    assert risk.factors == []
