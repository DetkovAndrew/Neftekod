from datetime import datetime

import pandas as pd

from neftekod_mas.ml.dataset import build_feature_frame, build_training_table
from neftekod_mas.ml.split import chronological_split
from neftekod_mas.quality.quality_agent import GODT_POINT2


def _kip_df(start, n, value_fn):
    idx = pd.date_range(start, periods=n, freq="10min")
    return pd.DataFrame({"T1": [value_fn(i) for i in range(n)]}, index=idx)


def test_build_training_table_uses_only_past_kip_no_lookahead():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 20, lambda i: 100.0 + i)
    ht = _kip_df(start, 20, lambda i: 200.0 + i)
    ff = build_feature_frame(avt, ht)

    # измерение ЛИМС в момент между 5-й и 6-й точкой КИП (00:55)
    measured_at = start + pd.Timedelta(minutes=55)
    lims = pd.DataFrame({
        "point_label": [GODT_POINT2],
        "param": ["Mg.Sulfur"],
        "unit": ["мг/кг"],
        "measured_at": [measured_at],
        "value": [8.0],
    })

    table = build_training_table("sulfur_mg_kg", lims, ff)
    assert len(table.target) == 1
    # backward asof -> берётся точка 00:50 (индекс 5), НЕ 01:00 (индекс 6, из будущего)
    assert table.features["avt:T1"].iloc[0] == 105.0
    assert table.target.iloc[0] == 8.0
    assert table.target_age_minutes.iloc[0] == 5.0


def test_build_training_table_drops_observations_beyond_tolerance():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 3, lambda i: 100.0 + i)
    ht = _kip_df(start, 3, lambda i: 200.0 + i)
    ff = build_feature_frame(avt, ht)

    far_future = start + pd.Timedelta(hours=5)  # далеко за пределами КИП-истории и допуска
    lims = pd.DataFrame({
        "point_label": [GODT_POINT2],
        "param": ["Mg.Sulfur"],
        "unit": ["мг/кг"],
        "measured_at": [far_future],
        "value": [8.0],
    })
    table = build_training_table("sulfur_mg_kg", lims, ff)
    assert len(table.target) == 0


def test_chronological_split_no_shuffle_and_no_overlap():
    start = datetime(2023, 1, 1, 0, 0)
    avt = _kip_df(start, 100, lambda i: 100.0 + i)
    ht = _kip_df(start, 100, lambda i: 200.0 + i)
    ff = build_feature_frame(avt, ht)

    measured_ats = [start + pd.Timedelta(minutes=10 * i) for i in range(50)]
    lims = pd.DataFrame({
        "point_label": [GODT_POINT2] * 50,
        "param": ["Mg.Sulfur"] * 50,
        "unit": ["мг/кг"] * 50,
        "measured_at": measured_ats,
        "value": list(range(50)),
    })
    table = build_training_table("sulfur_mg_kg", lims, ff)
    split = chronological_split(table, train_fraction=0.8)

    assert len(split.train.target) + len(split.test.target) == len(table.target)
    assert split.train.features.index.max() < split.split_at
    assert split.test.features.index.min() >= split.split_at
    # никакого перемешивания -- train целиком раньше test по времени
    assert split.train.features.index.max() < split.test.features.index.min()
