"""Train and select quality-model candidates without using the final holdout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neftekod_mas.data.loaders import load_kip, load_lims
from neftekod_mas.ml.baseline_gbm import train_gbm_on_shared_split
from neftekod_mas.ml.dataset import TARGET_METRICS, build_feature_frame, build_training_table
from neftekod_mas.ml.split import shared_time_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kip-dir", type=Path, required=True)
    parser.add_argument("--lims", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Use a new output directory to preserve previous experiments")
    avt = load_kip(args.kip_dir / "avt_tags.csv")
    ht = load_kip(args.kip_dir / "242000_tags.csv")
    lims = load_lims(args.lims)
    features = build_feature_frame(avt, ht)
    tables = {m: build_training_table(m, lims, features) for m in TARGET_METRICS}
    split = shared_time_split(tables)
    gbm = train_gbm_on_shared_split(tables, split)

    comparison = {}
    for metric in TARGET_METRICS:
        result = gbm.report[metric]
        if "validation_mae" not in result:
            comparison[metric] = {"selected_by_validation": "insufficient_data", **result}
            continue
        candidates = {"median": result["validation_baseline_mae"], "lightgbm": result["validation_mae"]}
        final_mae = {"median": result["baseline_median_mae"], "lightgbm": result["test_mae"]}
        winner = min(candidates, key=candidates.get)
        comparison[metric] = {"selected_by_validation": winner,
                              "selection_validation_mae": candidates[winner],
                              "selected_test_mae": final_mae[winner],
                              "candidates_validation_mae": candidates}
    report = {"feature_mode": "snapshot",
              "feature_count": int(features.shape[1]),
              "split": {"validation_at": str(split.validation_at), "test_at": str(split.test_at)},
              "selection_rule": "lowest validation MAE; final test is never used to choose a model",
              "selected": comparison, "lightgbm": gbm.report}
    (args.out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
