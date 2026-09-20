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
from neftekod_mas.ml.modular_pinn import train_modular_pinn
from neftekod_mas.ml.split import shared_time_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kip-dir", type=Path, required=True)
    parser.add_argument("--lims", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--feature-mode", choices=["snapshot", "causal_temporal"], action="append",
                        help="Can be passed repeatedly; default benchmarks both modes")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Use a new output directory to preserve previous experiments")
    torch.set_num_threads(4)
    avt = load_kip(args.kip_dir / "avt_tags.csv")
    ht = load_kip(args.kip_dir / "242000_tags.csv")
    lims = load_lims(args.lims)
    modes = args.feature_mode or ["snapshot", "causal_temporal"]
    per_mode = {}
    for mode in dict.fromkeys(modes):
        features = build_feature_frame(avt, ht, mode=mode)
        tables = {m: build_training_table(m, lims, features) for m in TARGET_METRICS}
        split = shared_time_split(tables)
        gbm = train_gbm_on_shared_split(tables, split)
        pinn = train_modular_pinn(avt, ht, lims, epochs=args.epochs, feature_mode=mode)
        pinn.save(args.out_dir / mode / "pinn")
        per_mode[mode] = {"feature_count": int(features.shape[1]), "pinn": pinn.metrics_report, "lightgbm": gbm.report}

    comparison = {}
    baseline_mode = next(iter(per_mode))
    for metric in TARGET_METRICS:
        candidates = {"median": per_mode[baseline_mode]["pinn"][metric]["validation_baseline_mae"]}
        final_mae = {"median": per_mode[baseline_mode]["pinn"][metric]["baseline_median_mae"]}
        for mode, result in per_mode.items():
            candidates[f"{mode}:pinn"] = result["pinn"][metric]["validation_mae"]
            candidates[f"{mode}:lightgbm"] = result["lightgbm"][metric]["validation_mae"]
            final_mae[f"{mode}:pinn"] = result["pinn"][metric]["test_mae"]
            final_mae[f"{mode}:lightgbm"] = result["lightgbm"][metric]["test_mae"]
        winner = min(candidates, key=candidates.get)
        comparison[metric] = {"selected_by_validation": winner,
                              "selection_validation_mae": candidates[winner],
                              "selected_test_mae": final_mae[winner],
                              "candidates_validation_mae": candidates}
    report = {"feature_modes": list(per_mode),
              "split": {"validation_at": str(split.validation_at), "test_at": str(split.test_at)},
              "selection_rule": "lowest validation MAE; final test is never used to choose a model",
              "selected": comparison, "per_mode": per_mode}
    (args.out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
