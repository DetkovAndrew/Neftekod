#!/usr/bin/env python3
"""
Обучение ML-слоя Агента качества (ARCHITECTURE.md §5.2) -- LightGBM
baseline по умолчанию, модульная PI-архитектура (`--model pinn`) как
альтернатива для сравнения на chronological holdout.

ЭТОТ СКРИПТ НЕ ЗАПУСКАЛСЯ на реальных данных завода в рамках текущей
сессии -- по явной договорённости, обучение начинается отдельно.
Написан и готов к запуску: вся ETL/синхронизация/split уже проверены
на реальных данных (ml/dataset.py, ml/split.py), обучающий код проверен
на синтетике (tests/test_baseline_gbm.py).

Запуск (когда будет дано добро):
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/train_quality_models.py --model gbm
    python scripts/train_quality_models.py --model pinn   # после установки torch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip, load_lims  # noqa: E402
from neftekod_mas.ml.baseline_gbm import train_all_metrics  # noqa: E402


def train_gbm(out_dir: Path) -> None:
    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")

    predictor = train_all_metrics(avt, ht, lims)

    report = {}
    for metric, mm in predictor.models.items():
        report[metric] = {
            "test_mae": mm.test_mae,
            "test_rmse": mm.test_rmse,
            "n_train": mm.n_train,
            "n_test": mm.n_test,
        }
        print(f"{metric}: MAE={mm.test_mae:.4f}  RMSE={mm.test_rmse:.4f}  "
              f"(train n={mm.n_train}, test n={mm.n_test})")

    predictor.save(out_dir)
    (out_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nМодели и отчёт -> {out_dir}")
    print(
        "\nСравните test_mae с config/vak_formula_accuracy.yaml (std_after_bias_correction) "
        "-- ML-слой оправдан только там, где он заметно точнее уже действующих ВАК-формул/"
        "ASTM-корреляций, а не просто 'ещё одна модель'."
    )


def train_pinn(out_dir: Path) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyTorch не установлен. pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu (CPU-версия -- см. Q&A: продакшен без интернета "
            "и GPU не гарантирован, ARCHITECTURE.md §9)."
        )
    from neftekod_mas.ml.modular_pinn import train_modular_pinn  # noqa: PLC0415

    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")

    result = train_modular_pinn(avt, ht, lims)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save(out_dir)
    print(f"Модель и отчёт -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gbm", "pinn"], default="gbm")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "models"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.model
    if args.model == "gbm":
        train_gbm(out_dir)
    else:
        train_pinn(out_dir)


if __name__ == "__main__":
    main()
