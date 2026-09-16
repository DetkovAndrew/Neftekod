#!/usr/bin/env python3
"""
Бэктест точности ВАК-формул 24-2000 против реальных значений ЛИМС
(ARCHITECTURE.md §5.3 -- закрывает документированный пробел "формула и
лаборатория не сверяются перекрёстно"). Это СТАТИСТИКА (сравнение уже
готовых, не обучаемых формул с фактом), не обучение модели -- ничего
не минимизируется, ни один коэффициент формулы не меняется.

Методология для T95/D15 (формулы, которым нужен ЛИМС как один из
входов, см. quality_agent.py MetricSpec.vak_needs_lims): используется
ПРЕДЫДУЩЕЕ известное значение ЛИМС (measured_at < текущей точки), а не
значение В ТОЙ ЖЕ точке -- иначе бэктест был бы обманчиво оптимистичным
(формула "знала" бы ответ заранее). Это ровно то, как формула
реально используется в проде: прогноз на момент T использует последний
ИЗВЕСТНЫЙ на тот момент ЛИМС, а не будущий.

Результат -- config/vak_formula_accuracy.yaml: MAE/RMSE/bias/n по
каждой из 3 формул (T95, CFPP, D15), используемых Агентом качества.
Не оптимизирован никакой параметр формулы -- задача только измерить,
насколько ей можно доверять, и передать эту неопределённость оператору.

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/compute_vak_formula_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip, load_lims  # noqa: E402
from neftekod_mas.quality import vak_formulas as vak  # noqa: E402
from neftekod_mas.quality.quality_agent import GODT_POINT2  # noqa: E402

# (metric_name, lims_param, formula_key, needs_lims, lims_key_for_formula)
BACKTEST_SPECS = [
    ("t95_c", "95%.T", "24-2000:GODT:T95", True, "95%.T"),
    ("cfpp_c", "CFPP", "24-2000:GODT:CFPP", False, None),
    ("density_kg_m3", "D15", "24-2000:GODT:D15", True, "24-2000.Pipeline.D15"),
]

KIP_TOLERANCE = pd.Timedelta(minutes=30)


def _kip_row_asof(ht: pd.DataFrame, ts) -> dict | None:
    idx = ht.index
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0 or (ts - idx[pos]) > KIP_TOLERANCE:
        return None
    row = ht.iloc[pos]
    if row.isna().any():
        return None
    return row.to_dict()


def backtest_metric(ht: pd.DataFrame, lims_point: pd.DataFrame, formula_key, needs_lims, lims_key) -> dict:
    fn = vak.HT242000_FORMULAS[formula_key]
    lims_point = lims_point.sort_values("measured_at").reset_index(drop=True)

    errors = []
    for i, row in lims_point.iterrows():
        ts = row["measured_at"]
        kip_row = _kip_row_asof(ht, ts)
        if kip_row is None:
            continue

        if needs_lims:
            prior = lims_point[lims_point["measured_at"] < ts]
            if prior.empty:
                continue
            lims_input_value = prior.iloc[-1]["value"]
            try:
                pred = fn(kip_row, {lims_key: lims_input_value})
            except KeyError:
                continue
        else:
            try:
                pred = fn(kip_row)
            except KeyError:
                continue

        errors.append(pred - row["value"])

    if not errors:
        return {"n": 0}
    s = pd.Series(errors)
    rmse = float((s ** 2).mean() ** 0.5)
    bias = float(s.mean())
    return {
        "n": int(len(s)),
        "mae": round(float(s.abs().mean()), 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        # см. compute_astm_cetane_accuracy.py -- та же идея: остаточный
        # разброс после вычитания измеренного смещения.
        "std_after_bias_correction": round(max(rmse ** 2 - bias ** 2, 0.0) ** 0.5, 4),
    }


def main() -> None:
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")

    results: dict[str, dict] = {
        "_meta": {
            "methodology": "Backtest: prediction at time T uses only LIMS known BEFORE T "
            "(no look-ahead). Statistics, not model fitting -- formula coefficients unchanged.",
        }
    }
    for metric, lims_param, formula_key, needs_lims, lims_key in BACKTEST_SPECS:
        lims_point = lims[(lims["point_label"] == GODT_POINT2) & (lims["param"] == lims_param)]
        stats = backtest_metric(ht, lims_point, formula_key, needs_lims, lims_key)
        results[metric] = {"formula": formula_key, **stats}
        print(f"{metric} ({formula_key}): {stats}")

    out_path = REPO_ROOT / "config" / "vak_formula_accuracy.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, allow_unicode=True, sort_keys=False)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
