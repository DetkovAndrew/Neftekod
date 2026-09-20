"""
Юнит-тест на ПРОГРАММНУЮ корректность (форма данных, save/load,
интерфейс QualityPredictor) обучающего кода baseline_gbm.py --
НЕ на качество модели. Обучение здесь -- на синтетических случайных
данных, одноразовое, в памяти теста, никакого артефакта не
производится и не сохраняется как часть решения. Настоящее обучение
на реальных данных завода этим тестом не запускается (см.
scripts/train_quality_models.py, который тоже не запущен).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from neftekod_mas.ml.baseline_gbm import train_all_metrics, train_gbm_on_shared_split
from neftekod_mas.ml.dataset import build_feature_frame, build_training_table, TARGET_METRICS
from neftekod_mas.ml.split import shared_time_split
from neftekod_mas.quality.quality_agent import GODT_POINT2
from neftekod_mas.schemas import DataSource, ProcessState, TagReading


def _synthetic_kip(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(datetime(2023, 1, 1), periods=n, freq="10min")
    return pd.DataFrame({f"T{i}": rng.normal(300, 10, n) for i in range(1, 4)}, index=idx)


def _synthetic_lims(n_obs: int, kip_index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = sorted(rng.choice(kip_index[10:-10], size=n_obs, replace=False))
    return pd.DataFrame({
        "point_label": [GODT_POINT2] * n_obs,
        "param": ["Mg.Sulfur"] * n_obs,
        "unit": ["мг/кг"] * n_obs,
        "measured_at": pd.to_datetime(times),
        "value": rng.normal(7.0, 1.5, n_obs),
    })


@pytest.mark.parametrize("seed", [0])
def test_train_predict_save_load_roundtrip(tmp_path, seed):
    avt = _synthetic_kip(500, seed)
    ht = _synthetic_kip(500, seed + 1)
    lims_sulfur = _synthetic_lims(60, avt.index, seed)
    # остальные 4 метрики отсутствуют -- проверяем, что train_all_metrics
    # честно падает с понятной ошибкой на пустой цели, а не молча создаёт мусорную модель
    lims = lims_sulfur

    with pytest.raises(ValueError):
        train_all_metrics(avt, ht, lims)  # для t95_c/cetane/cfpp/d15 нет наблюдений вовсе


def test_train_predict_save_load_roundtrip_all_metrics(tmp_path):
    seed = 1
    avt = _synthetic_kip(800, seed)
    ht = _synthetic_kip(800, seed + 1)
    frames = []
    for param, base in [("Mg.Sulfur", 7.0), ("95%.T", 345.0), ("CetaneNumber", 53.0), ("CFPP", -10.0), ("D15", 835.0)]:
        rng = np.random.default_rng(hash(param) % 1000)
        times = sorted(rng.choice(avt.index[10:-10], size=40, replace=False))
        frames.append(pd.DataFrame({
            "point_label": [GODT_POINT2] * 40,
            "param": [param] * 40,
            "unit": ["u"] * 40,
            "measured_at": pd.to_datetime(times),
            "value": rng.normal(base, 1.0, 40),
        }))
    lims = pd.concat(frames, ignore_index=True)

    predictor = train_all_metrics(avt, ht, lims, lgb_params={"n_estimators": 20})
    assert set(predictor.models) == {"sulfur_mg_kg", "t95_c", "cetane_number", "cfpp_c", "density_kg_m3"}
    for mm in predictor.models.values():
        assert mm.n_train > 0 and mm.n_test > 0
        assert mm.test_mae >= 0

    now = avt.index[-1].to_pydatetime()
    kip = {}
    for tag in ["T1", "T2", "T3"]:
        kip[f"avt:{tag}"] = TagReading(tag_id=f"avt:{tag}", value=300.0, unit="", timestamp=now, source=DataSource.KIP)
        kip[f"242000:{tag}"] = TagReading(tag_id=f"242000:{tag}", value=300.0, unit="", timestamp=now, source=DataSource.KIP)
    from neftekod_mas.data.sync import DataQualityReport
    state = ProcessState(decision_at=now, kip=kip, lab_points={}, quality_report=DataQualityReport(decision_at=now, flags=[], sync_ok=True))

    predictions = predictor.predict(state)
    assert len(predictions) == 5
    assert all(p.source == DataSource.ML_MODEL for p in predictions)

    save_dir = tmp_path / "gbm_models"
    predictor.save(save_dir)
    reloaded = predictor.__class__.load(save_dir)
    reloaded_predictions = reloaded.predict(state)
    assert {p.metric: round(p.value, 4) for p in predictions} == {p.metric: round(p.value, 4) for p in reloaded_predictions}


def test_gbm_shared_time_split_keeps_test_out_of_model_selection():
    avt = _synthetic_kip(800, 4)
    ht = _synthetic_kip(800, 5)
    frames = []
    for number, (param, base) in enumerate([("Mg.Sulfur", 7.), ("95%.T", 345.), ("CetaneNumber", 53.), ("CFPP", -10.), ("D15", 835.)]):
        times = avt.index[30 + number::18][:40]
        frames.append(pd.DataFrame({"point_label": GODT_POINT2, "param": param, "unit": "u",
                                    "measured_at": times, "value": base + np.arange(len(times)) * .01}))
    lims = pd.concat(frames, ignore_index=True)
    features = build_feature_frame(avt, ht)
    tables = {m: build_training_table(m, lims, features) for m in TARGET_METRICS}
    result = train_gbm_on_shared_split(tables, shared_time_split(tables), {"n_estimators": 20})
    assert set(result.models) == set(TARGET_METRICS)
    assert all(rep["n_train"] and rep["n_validation"] and rep["n_test"] for rep in result.report.values())
