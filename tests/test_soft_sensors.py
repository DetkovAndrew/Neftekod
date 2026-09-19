from datetime import datetime, timedelta

import pandas as pd

from neftekod_mas.quality.quality_agent import GODT_POINT2, METRIC_SPECS, QualityAgent
from neftekod_mas.quality.soft_sensors import SoftSensorService
from neftekod_mas.schemas import DataQualityReport, DataSource, LabPointReading, ProcessState, TagReading

T0 = datetime(2024, 1, 10, 12, 0)
SELECTION = {
    "_meta": {},
    "sulfur_mg_kg": {"selected": "anchor+bias_k10", "cv_mean_mae": 1.18},
    "cfpp_c": {"selected": "last_lims", "cv_mean_mae": 1.17},
}
SULFUR_SPEC = next(s for s in METRIC_SPECS if s.metric == "sulfur_mg_kg")
HC = {"product_diesel": {"sulfur_mg_kg": {"op": "<=", "limit": 10.0, "risk_margin_high": 1.0, "risk_margin_medium": 2.5}}}


def _state(q21=5.0, steady=True, lims_age_h=None, at=T0):
    kip = {"242000:Q21": TagReading(tag_id="242000:Q21", value=q21, unit="", timestamp=at, source=DataSource.KIP)}
    lab = {}
    if lims_age_h is not None:
        lab[f"{GODT_POINT2}|Mg.Sulfur"] = LabPointReading(
            point_id="x", value=8.0, unit="", measured_at=at - timedelta(hours=lims_age_h), decision_at=at, source=DataSource.LIMS)
    return ProcessState(decision_at=at, kip=kip, lab_points=lab,
                        quality_report=DataQualityReport(decision_at=at, sync_ok=True), steady_regime=steady)


def _service(residuals):
    idx = [T0 - timedelta(days=len(residuals) - i) for i in range(len(residuals))]
    return SoftSensorService(SELECTION, {"sulfur_mg_kg": pd.Series(residuals, index=idx)})


def test_estimate_is_anchor_plus_median_of_past_residuals():
    est = _service([1.0, 2.0, 3.0, 100.0, 2.0]).estimate("sulfur_mg_kg", _state(q21=5.0))
    assert est.value == 5.0 + 2.0
    assert est.n_bias_samples == 5


def test_future_residuals_are_ignored():
    svc = _service([1.0, 1.0, 1.0])
    svc.residuals["sulfur_mg_kg"].loc[pd.Timestamp(T0 + timedelta(hours=1))] = 50.0
    assert svc.estimate("sulfur_mg_kg", _state()).bias == 1.0


def test_no_estimate_outside_steady_state_or_invalid_anchor():
    svc = _service([1.0, 1.0, 1.0])
    assert svc.estimate("sulfur_mg_kg", _state(steady=False)) is None
    assert svc.estimate("sulfur_mg_kg", _state(q21=24.9)) is None  # анализатор в насыщении


def test_quality_agent_prefers_soft_sensor_over_stale_lims_but_not_over_current():
    qa = QualityAgent(HC, soft_sensors=_service([1.0, 1.0, 1.0]))
    stale = qa._estimate_metric(SULFUR_SPEC, _state(lims_age_h=12))
    assert stale.source == DataSource.SOFT_SENSOR and stale.value == 6.0
    assert stale.confidence.value == "high"  # 1.18 < 0.5 * 2.5
    fresh = qa.assess(_state(lims_age_h=0.5)).current[0]
    assert fresh.source == DataSource.LIMS and fresh.value == 8.0
