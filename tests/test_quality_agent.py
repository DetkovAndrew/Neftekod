from datetime import datetime, timedelta

from neftekod_mas.data.sync import DataQualityReport
from neftekod_mas.quality.quality_agent import GODT_POINT2, QualityAgent
from neftekod_mas.schemas import ConfidenceLevel, DataSource, LabPointReading, ProcessState

HARD_CONSTRAINTS = {
    "product_diesel": {
        "sulfur_mg_kg": {"op": "<=", "limit": 10.0},
        "t95_c": {"op": "<=", "limit": 360.0},
        "cetane_number": {"op": ">=", "limit": 51.0},
        "cfpp_c": {"op": "<=", "limit": None},
    }
}


def _state(decision_at: datetime, lab_points: dict, kip: dict | None = None) -> ProcessState:
    return ProcessState(
        decision_at=decision_at,
        kip=kip or {},
        lab_points=lab_points,
        quality_report=DataQualityReport(decision_at=decision_at, flags=[], sync_ok=True),
    )


def _lims(point, param, value, measured_at, decision_at, unit="u"):
    return LabPointReading(
        point_id=f"{point}|{param}", value=value, unit=unit,
        measured_at=measured_at, decision_at=decision_at, source=DataSource.LIMS,
    )


def test_lims_takes_priority_over_pak():
    now = datetime(2023, 6, 1, 10, 0)
    lab_points = {
        f"{GODT_POINT2}|Mg.Sulfur": _lims(GODT_POINT2, "Mg.Sulfur", 8.0, now, now),
        "24-2000:Mg.Sulfur": LabPointReading(
            point_id="24-2000:Mg.Sulfur", value=100.0, unit="ppm",
            measured_at=now, decision_at=now, source=DataSource.PAK,
        ),
    }
    state = _state(now, lab_points)
    qa = QualityAgent(HARD_CONSTRAINTS).assess(state)
    sulfur = next(e for e in qa.current if e.metric == "sulfur_mg_kg")
    assert sulfur.source == DataSource.LIMS
    assert sulfur.value == 8.0


def test_falls_back_to_pak_when_lims_missing():
    now = datetime(2023, 6, 1, 10, 0)
    lab_points = {
        "24-2000:Mg.Sulfur": LabPointReading(
            point_id="24-2000:Mg.Sulfur", value=6.5, unit="ppm",
            measured_at=now, decision_at=now, source=DataSource.PAK,
        ),
    }
    state = _state(now, lab_points)
    qa = QualityAgent(HARD_CONSTRAINTS).assess(state)
    sulfur = next(e for e in qa.current if e.metric == "sulfur_mg_kg")
    assert sulfur.source == DataSource.PAK
    assert sulfur.value == 6.5


def test_falls_back_to_vak_formula_for_cfpp():
    now = datetime(2023, 6, 1, 10, 0)
    from neftekod_mas.schemas import TagReading
    kip = {
        f"242000:{tag}": TagReading(tag_id=f"242000:{tag}", value=v, unit="", timestamp=now, source=DataSource.KIP)
        for tag, v in {"T23": 234.20, "P8": 0.117, "F9": 171.09, "W7": 0.143, "P24": 0.595}.items()
    }
    state = _state(now, {}, kip)
    qa = QualityAgent(HARD_CONSTRAINTS).assess(state)
    cfpp = next(e for e in qa.current if e.metric == "cfpp_c")
    assert cfpp.source == DataSource.VAK_FORMULA
    assert cfpp.confidence == ConfidenceLevel.MEDIUM


def test_no_source_available_metric_absent_and_overall_low_or_refuse():
    now = datetime(2023, 6, 1, 10, 0)
    state = _state(now, {})
    qa = QualityAgent(HARD_CONSTRAINTS).assess(state)
    assert not any(e.metric == "cetane_number" for e in qa.current)
    assert qa.overall_confidence == ConfidenceLevel.REFUSE


def test_violation_detection_sulfur_over_limit():
    now = datetime(2023, 6, 1, 10, 0)
    lab_points = {f"{GODT_POINT2}|Mg.Sulfur": _lims(GODT_POINT2, "Mg.Sulfur", 12.0, now, now)}
    state = _state(now, lab_points)
    qa = QualityAgent(HARD_CONSTRAINTS).assess(state)
    v = next(v for v in qa.violations if v.metric == "sulfur_mg_kg")
    assert v.margin < 0
    assert v.risk_class.value == "critical"


def test_alert_thresholds_in_units_shift_earlier_for_less_accurate_estimates():
    from neftekod_mas.schemas import ConfidenceLevel, DataSource, QualityMetricEstimate, RiskClass

    cfg = {"op": "<=", "limit": 10.0, "watch_at": 7.5, "act_at": 8.5, "reference_error": 1.18,
           "risk_margin_high": 1.0, "risk_margin_medium": 2.5}
    qa = QualityAgent({"product_diesel": {"sulfur_mg_kg": cfg}})

    def check(value, err):
        est = QualityMetricEstimate(metric="sulfur_mg_kg", value=value, unit="мг/кг", source=DataSource.SOFT_SENSOR,
                                    confidence=ConfidenceLevel.HIGH, typical_error=err)
        return qa._check_violations([est])[0]

    assert check(7.0, 1.18).risk_class == RiskClass.LOW
    assert check(8.0, 1.18).risk_class == RiskClass.MEDIUM  # под наблюдением
    assert check(8.6, 1.18).risk_class == RiskClass.HIGH
    assert check(10.2, 1.18).risk_class == RiskClass.CRITICAL
    assert check(8.0, None).risk_class == RiskClass.MEDIUM  # свежий ЛИМС -- без сдвига
    # оценка вдвое грубее -> пороги сдвинуты к норме на 1.18: действие уже с 7.32
    worse = check(8.0, 2.36)
    assert worse.risk_class == RiskClass.HIGH
    assert abs(worse.act_at - 7.32) < 1e-9


def test_alert_thresholds_for_lower_limit_metrics():
    from neftekod_mas.quality.quality_agent import alert_thresholds

    cetane = {"op": ">=", "limit": 51.0, "watch_at": 53.0, "act_at": 52.0, "reference_error": 1.23}
    assert alert_thresholds(cetane, 1.23) == (52.0, 53.0)
    act, watch = alert_thresholds(cetane, 2.23)
    assert abs(act - 53.0) < 1e-9 and abs(watch - 54.0) < 1e-9
    assert alert_thresholds({"op": "<=", "limit": 1.0}, 0.5) is None
