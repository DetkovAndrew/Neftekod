#!/usr/bin/env python3
"""
Честный отбор soft-sensor'ов для 5 показателей (ml/anchored.py) против
baseline'ов: forward-chaining валидация по 4 последовательным временным
блокам (модель каждый раз видит только прошлое), выбор кандидата по
среднему MAE блоков, итоговый test (тот же период, что в MODEL_AUDIT.md,
с 2025-11-26) в выборе не участвует.

Оценка -- на стационарном режиме (ml/anchored.steady_state_mask);
отдельно для прозрачности -- на всём тесте, включая переходные режимы.
Там, где кандидат неприменим (например, анализатор Q21 вне шкалы),
подставляется медиана train -- так работала бы система (fallback), и так
сравнение между кандидатами остаётся честным по покрытию.

    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python3 scripts/benchmark_anchored.py --out runs/models/anchored_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip, load_lims  # noqa: E402
from neftekod_mas.ml.anchored import candidate_predictions, steady_state_mask  # noqa: E402
from neftekod_mas.ml.dataset import TARGET_METRICS, build_feature_frame, build_training_table  # noqa: E402

TEST_AT = pd.Timestamp("2025-11-26 10:00:00")
N_BLOCKS = 4
VALIDATION_SHARE_OF_PRETEST = 0.6
# Правило простоты (аналог one-standard-error rule): из кандидатов, чей CV MAE
# не хуже лучшего более чем на SIMPLICITY_TOLERANCE, берётся самый простой --
# выигрыш в пределах шума не оправдывает лишних степеней свободы.
SIMPLICITY_TOLERANCE = 0.02


def _complexity(name: str) -> int:
    if name == "median":
        return 0
    if name == "last_lims":
        return 1
    if name == "anchor":
        return 2
    if name.endswith("gbm_resid"):
        return 4
    return 3  # anchor+bias_k*


def _mae(pred: pd.Series, y: pd.Series, mask: pd.Series, fallback: float) -> tuple[float, float]:
    p = pred[mask]
    coverage = float(p.notna().mean()) if len(p) else float("nan")
    err = (p.fillna(fallback) - y[mask]).abs()
    return float(err.mean()) if len(err) else float("nan"), coverage


def evaluate_metric(metric, table, lims, ht):
    y = table.target.astype(float)
    steady = steady_state_mask(table, ht)
    pre = y.index < TEST_AT
    pre_times = y.index[pre & steady.to_numpy()].sort_values()
    start = pre_times[int(len(pre_times) * (1 - VALIDATION_SHARE_OF_PRETEST))]
    edges = pd.date_range(start, TEST_AT, periods=N_BLOCKS + 1)

    block_scores: dict[str, list[float]] = {}
    for b0, b1 in zip(edges[:-1], edges[1:]):
        fit_mask = steady & (y.index < b0)
        eval_mask = steady & (y.index >= b0) & (y.index < b1)
        if eval_mask.sum() == 0:
            continue
        fallback = float(y[fit_mask].median())
        for cand in candidate_predictions(metric, table, lims, fit_mask):
            mae, _ = _mae(cand.pred, y, eval_mask, fallback)
            block_scores.setdefault(cand.name, []).append(mae)

    cv = {name: float(np.mean(v)) for name, v in block_scores.items() if len(v) == max(map(len, block_scores.values()))}
    best = min(cv.values())
    near_best = [n for n, v in cv.items() if v <= best * (1 + SIMPLICITY_TOLERANCE)]
    selected = min(near_best, key=lambda n: (_complexity(n), cv[n]))

    fit_mask = steady & (y.index < TEST_AT)
    test_steady = steady & (y.index >= TEST_AT)
    test_all = y.index >= TEST_AT
    fallback = float(y[fit_mask].median())
    final = {}
    for cand in candidate_predictions(metric, table, lims, fit_mask):
        mae_s, cov = _mae(cand.pred, y, test_steady, fallback)
        mae_a, _ = _mae(cand.pred, y, pd.Series(test_all, index=y.index), fallback)
        final[cand.name] = {"test_mae_steady": mae_s, "test_mae_all": mae_a, "coverage": cov}

    return {
        "n_total": int(len(y)),
        "n_transient_excluded": int((~steady).sum()),
        "n_test_steady": int(test_steady.sum()),
        "cv_blocks": [str(e) for e in edges],
        "cv_mean_mae": cv,
        "selected": selected,
        "selected_test_mae_steady": final[selected]["test_mae_steady"],
        "median_test_mae_steady": final["median"]["test_mae_steady"],
        "final": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kip-dir", type=Path, default=None)
    parser.add_argument("--lims", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kip_dir = args.kip_dir or data_dir()
    lims_path = args.lims or data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx"
    avt, ht = load_kip(kip_dir / "avt_tags.csv"), load_kip(kip_dir / "242000_tags.csv")
    lims = load_lims(lims_path)
    features = build_feature_frame(avt, ht)

    report = {"test_at": str(TEST_AT), "selection_rule": "min mean MAE over forward-chaining validation blocks; test never used for selection", "metrics": {}}
    for metric in TARGET_METRICS:
        table = build_training_table(metric, lims, features)
        r = evaluate_metric(metric, table, lims, ht)
        report["metrics"][metric] = r
        print(f"{metric:15s} selected={r['selected']:32s} test MAE={r['selected_test_mae_steady']:.3f} "
              f"| median={r['median_test_mae_steady']:.3f} | n_test={r['n_test_steady']} "
              f"| excluded transient={r['n_transient_excluded']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")

    # Агрегированный итог отбора (без сырых данных) -- конфиг для Агента качества.
    selection = {
        "_meta": {
            "source": "scripts/benchmark_anchored.py",
            "selection_rule": report["selection_rule"] + f"; simplest within {SIMPLICITY_TOLERANCE:.0%} of best",
            "scope": "steady-state reactor only (ml/anchored.steady_state_mask); outside -> refuse",
            "test_at": report["test_at"],
        },
        **{
            m: {
                "selected": r["selected"],
                "cv_mean_mae": round(r["cv_mean_mae"][r["selected"]], 4),
                "test_mae_steady": round(r["selected_test_mae_steady"], 4),
                "median_baseline_test_mae": round(r["median_test_mae_steady"], 4),
                "n_test_steady": r["n_test_steady"],
            }
            for m, r in report["metrics"].items()
        },
    }
    sel_path = REPO_ROOT / "config" / "soft_sensor_selection.yaml"
    import yaml  # noqa: PLC0415
    sel_path.write_text(yaml.safe_dump(selection, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"-> {sel_path}")


if __name__ == "__main__":
    main()
