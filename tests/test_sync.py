from datetime import datetime, timedelta

import pandas as pd

from neftekod_mas.data.sync import build_process_state
from neftekod_mas.schemas import DataSource


def _kip_df(start: datetime, n: int, tags: dict[str, list[float]]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="10min")
    return pd.DataFrame(tags, index=idx)


def test_kip_snapshot_and_freshness_and_lab_join():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 10, {"T1": [100.0 + i for i in range(10)], "P2": [1.0] * 10})
    ht = _kip_df(start, 10, {"T5": [50.0] * 10})

    lims = pd.DataFrame(
        {
            "point_label": ["ptA", "ptA"],
            "param": ["sulfur", "sulfur"],
            "unit": ["mg/kg", "mg/kg"],
            "measured_at": [start - timedelta(hours=1), start - timedelta(hours=30)],
            "value": [8.0, 20.0],
        }
    )
    pak = pd.DataFrame(
        {
            "param": ["sulfur_pak"],
            "unit": ["ppm"],
            "measured_at": [start - timedelta(minutes=5)],
            "value": [7.5],
        }
    )

    decision_at = start + timedelta(minutes=30)
    state = build_process_state(decision_at, avt, ht, lims, pak)

    assert state.quality_report.sync_ok
    assert "avt:T1" in state.kip
    assert state.kip["avt:T1"].value == 103.0  # индекс 3 -> 00:30
    assert "242000:T5" in state.kip

    # берётся ПОСЛЕДНЕЕ значение ЛИМС на момент decision_at (8.0, не 20.0)
    lims_reading = state.lab_points["ptA|sulfur"]
    assert lims_reading.value == 8.0
    assert lims_reading.source == DataSource.LIMS

    pak_reading = state.lab_points["sulfur_pak"]
    assert pak_reading.value == 7.5


def test_stuck_sensor_detected():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 10, {"T1": [100.0] * 10})  # неизменно все 10 точек
    ht = _kip_df(start, 10, {"T5": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    lims = pd.DataFrame(columns=["point_label", "param", "unit", "measured_at", "value"])
    pak = pd.DataFrame(columns=["param", "unit", "measured_at", "value"])

    decision_at = start + timedelta(minutes=90)
    state = build_process_state(decision_at, avt, ht, lims, pak)

    stuck_codes = [f for f in state.quality_report.flags if f.code == "stuck_sensor"]
    assert any(f.tag_or_point == "avt:T1" for f in stuck_codes)
    assert not any(f.tag_or_point == "242000:T5" for f in stuck_codes)


def test_out_of_range_detected_when_bounds_supplied():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 5, {"T1": [100.0, 101.0, 102.0, 5000.0, 103.0]})  # выброс на 4-й точке
    ht = _kip_df(start, 5, {"T5": [50.0] * 5})
    lims = pd.DataFrame(columns=["point_label", "param", "unit", "measured_at", "value"])
    pak = pd.DataFrame(columns=["param", "unit", "measured_at", "value"])
    bounds = {"avt:T1": {"low": 50.0, "high": 150.0}}

    decision_at = start + timedelta(minutes=30)  # -> индекс 3, значение 5000.0
    state = build_process_state(decision_at, avt, ht, lims, pak, kip_bounds=bounds)

    out_of_range = [f for f in state.quality_report.flags if f.code == "out_of_range"]
    assert len(out_of_range) == 1
    assert out_of_range[0].tag_or_point == "avt:T1"


def test_missing_kip_snapshot_beyond_tolerance():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 5, {"T1": [1.0] * 5})
    ht = _kip_df(start, 5, {"T5": [1.0] * 5})
    lims = pd.DataFrame(columns=["point_label", "param", "unit", "measured_at", "value"])
    pak = pd.DataFrame(columns=["param", "unit", "measured_at", "value"])

    far_future = start + timedelta(days=1)
    state = build_process_state(far_future, avt, ht, lims, pak)

    assert not state.quality_report.sync_ok
    assert any(f.code == "missing_kip_snapshot" for f in state.quality_report.flags)
