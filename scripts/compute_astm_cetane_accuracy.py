#!/usr/bin/env python3
"""
Бэктест ASTM D976 (цетановый индекс из плотности и T50) против реальных
измеренных ЛИМС CetaneNumber на точке 'Гидроочистка.2' -- ARCHITECTURE.md
§5.2. Статистика, не обучение: коэффициенты D976 не меняются, проверяется
только применимость стандартного метода к конкретному заводу/сырью.

В отличие от бэктеста ВАК-формул (compute_vak_formula_accuracy.py), здесь
НЕТ проблемы "заглядывания в будущее": D976 использует D15/T50 как
независимые входы, а не предыдущее значение самого цетанового числа --
поэтому корректно использовать D15/T50, измеренные В ТОЙ ЖЕ точке
отбора (ближайшие по времени к моменту замера CetaneNumber), а не
предшествующие.

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/compute_astm_cetane_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_lims  # noqa: E402
from neftekod_mas.quality.astm_correlations import cetane_index_d976  # noqa: E402
from neftekod_mas.quality.quality_agent import GODT_POINT2  # noqa: E402

MATCH_TOLERANCE = pd.Timedelta(hours=6)  # D15/T50 должны быть измерены близко по времени к CetaneNumber


def nearest_value(df: pd.DataFrame, param: str, ts, tolerance: pd.Timedelta) -> float | None:
    sub = df[df["param"] == param]
    if sub.empty:
        return None
    diffs = (sub["measured_at"] - ts).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > tolerance:
        return None
    return float(sub.loc[idx, "value"])


def main() -> None:
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    point = lims[lims["point_label"] == GODT_POINT2]
    cetane_actual = point[point["param"] == "CetaneNumber"].sort_values("measured_at")

    errors = []
    for _, row in cetane_actual.iterrows():
        ts = row["measured_at"]
        d15_kg_m3 = nearest_value(point, "D15", ts, MATCH_TOLERANCE)
        t50_c = nearest_value(point, "50%.T", ts, MATCH_TOLERANCE)
        if d15_kg_m3 is None or t50_c is None:
            continue
        if t50_c <= 0 or d15_kg_m3 <= 0:
            continue  # физически недостоверное показание (см. фильтр простоя в compute_reliability_bounds.py)
        pred = cetane_index_d976(d15_kg_m3 / 1000.0, t50_c)
        errors.append(pred - row["value"])

    result = {"n": len(errors)}
    if errors:
        s = pd.Series(errors)
        rmse = float((s ** 2).mean() ** 0.5)
        bias = float(s.mean())
        result.update({
            "mae": round(float(s.abs().mean()), 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            # остаточный разброс ПОСЛЕ вычитания систематического смещения
            # (bias) -- используется как typical_error вместо сырого MAE,
            # см. quality_agent.py: value скорректировано на -bias.
            "std_after_bias_correction": round(max(rmse ** 2 - bias ** 2, 0.0) ** 0.5, 3),
        })
    print("ASTM D976 cetane index backtest:", result)

    out = {
        "_meta": {
            "methodology": "D15/T50 matched to CetaneNumber measurement within "
            f"{MATCH_TOLERANCE}; same-time matching is valid here (unlike VAK "
            "formula backtest) since D976 does not use cetane itself as an input.",
            "source": "ASTM D976-91, archive.org/stream/gov.law.astm.d976.1991",
        },
        "cetane_number_d976": result,
    }
    out_path = REPO_ROOT / "config" / "astm_cetane_accuracy.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
