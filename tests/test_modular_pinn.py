"""
Юнит-тест на ПРОГРАММНУЮ корректность модульной PI-архитектуры (формы
тензоров, отсутствие ошибок в forward/backward, save) -- НЕ на качество
модели. Обучение здесь -- 5 эпох на маленьких синтетических данных,
только чтобы убедиться, что граф вычислений строится и градиенты текут
без исключений. Настоящее обучение на данных завода этим тестом не
запускается (см. scripts/train_quality_models.py --model pinn).
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from neftekod_mas.ml.modular_pinn import train_modular_pinn  # noqa: E402
from neftekod_mas.quality.quality_agent import GODT_POINT2  # noqa: E402


def _synthetic_kip(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(datetime(2023, 1, 1), periods=n, freq="10min")
    return pd.DataFrame({f"T{i}": rng.normal(300, 10, n) for i in range(1, 4)}, index=idx)


def _synthetic_lims_all_metrics(kip_index: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for param, base, n in [
        ("Mg.Sulfur", 7.0, 30), ("95%.T", 345.0, 25), ("CetaneNumber", 53.0, 15),
        ("CFPP", -10.0, 20), ("D15", 835.0, 25),
    ]:
        rng = np.random.default_rng(hash(param) % 1000)
        times = sorted(rng.choice(kip_index[10:-10], size=n, replace=False))
        frames.append(pd.DataFrame({
            "point_label": [GODT_POINT2] * n, "param": [param] * n, "unit": ["u"] * n,
            "measured_at": pd.to_datetime(times), "value": rng.normal(base, 1.0, n),
        }))
    return pd.concat(frames, ignore_index=True)


def test_modular_pinn_trains_without_errors_and_produces_finite_predictions():
    avt = _synthetic_kip(500, seed=1)
    ht = _synthetic_kip(500, seed=2)
    lims = _synthetic_lims_all_metrics(avt.index)

    result = train_modular_pinn(avt, ht, lims, epochs=5)

    assert set(result.metrics_report) == {"sulfur_mg_kg", "t95_c", "cetane_number", "cfpp_c", "density_kg_m3"}
    for metric, rep in result.metrics_report.items():
        if rep.get("n_test", 0) == 0:
            continue
        assert np.isfinite(rep["test_mae"])
        assert np.isfinite(rep["test_rmse"])


def test_modular_pinn_save_roundtrip(tmp_path):
    avt = _synthetic_kip(300, seed=3)
    ht = _synthetic_kip(300, seed=4)
    lims = _synthetic_lims_all_metrics(avt.index)

    result = train_modular_pinn(avt, ht, lims, epochs=3)
    out_dir = tmp_path / "pinn_model"
    result.save(out_dir)

    assert (out_dir / "model.pt").exists()
    assert (out_dir / "meta.json").exists()
