from datetime import datetime

import numpy as np
import pandas as pd

from neftekod_mas.ml.anchored import candidate_predictions, prior_lims, rolling_bias
from neftekod_mas.ml.dataset import TrainingTable
from neftekod_mas.quality.quality_agent import GODT_POINT2


def _lims(times, param, values):
    return pd.DataFrame({"point_label": GODT_POINT2, "param": param, "unit": "u",
                         "measured_at": pd.to_datetime(times), "value": values})


def test_prior_lims_is_strictly_before():
    t0, t1 = datetime(2024, 1, 1, 10), datetime(2024, 1, 2, 10)
    lims = _lims([t0, t1], "D15", [830.0, 840.0])
    times = pd.DatetimeIndex([t0, t1, t1 + pd.Timedelta(minutes=1)])
    out = prior_lims(lims, "D15", times)
    assert np.isnan(out.iloc[0])  # значение в тот же момент не видно
    assert out.iloc[1] == 830.0
    assert out.iloc[2] == 840.0


def test_rolling_bias_ignores_current_and_future_labels():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    y = pd.Series(np.arange(30, dtype=float), index=idx)
    anchor = pd.Series(0.0, index=idx)
    b1 = rolling_bias(y, anchor, 5)
    y2 = y.copy()
    y2.iloc[20:] += 1000.0
    b2 = rolling_bias(y2, anchor, 5)
    pd.testing.assert_series_equal(b1.iloc[:21], b2.iloc[:21])  # точка 20 видит только 15..19


def _table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="12h")
    q = rng.uniform(5, 12, n)
    feats = pd.DataFrame({"242000:Q21": q, "242000:T11": 360.0, "242000:T5": 370.0}, index=idx)
    y = pd.Series(q + 0.5 + rng.normal(0, 0.3, n), index=idx)
    return TrainingTable("sulfur_mg_kg", feats, y, pd.Series(0.0, index=idx))


def test_future_labels_do_not_change_past_predictions():
    table = _table()
    lims = _lims([], "Mg.Sulfur", [])
    cutoff = table.target.index[200]
    fit_mask = pd.Series(table.target.index < cutoff, index=table.target.index)
    first = {c.name: c.pred for c in candidate_predictions("sulfur_mg_kg", table, lims, fit_mask)}

    changed = TrainingTable(table.metric, table.features, table.target.copy(), table.target_age_minutes)
    changed.target[changed.target.index >= cutoff] += 1000.0
    second = {c.name: c.pred for c in candidate_predictions("sulfur_mg_kg", changed, lims, fit_mask)}

    past = table.target.index < cutoff
    for name in first:
        pd.testing.assert_series_equal(first[name][past], second[name][past], check_names=False)


def test_anchor_with_bias_recovers_constant_offset():
    table = _table()
    lims = _lims([], "Mg.Sulfur", [])
    fit_mask = pd.Series(True, index=table.target.index)
    preds = {c.name: c.pred for c in candidate_predictions("sulfur_mg_kg", table, lims, fit_mask)}
    tail = table.target.index[50:]
    mae_raw = (preds["anchor"][tail] - table.target[tail]).abs().mean()
    mae_bias = (preds["anchor+bias_k10"][tail] - table.target[tail]).abs().mean()
    assert mae_bias < mae_raw  # смещение +0.5 анализатора снято подстройкой по лаборатории
