#!/usr/bin/env python3
"""
Считает 5-95 перцентильные границы валидированной истории для тегов из
config/control_variables.yaml (ARCHITECTURE.md §5.3, по методике Ta & Liu
2027 §3.3.1: "Variable bounds were set at the 5th and 95th percentiles of
their respective training distributions"). Статистика, не обучение модели.

Пишет config/control_bounds.yaml отдельно от control_variables.yaml,
чтобы не смешивать рукописное обоснование (rationale) с генерируемыми
числами -- числа пересчитываются этим скриптом, текст обоснования правится
руками.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip  # noqa: E402

RUNNING_STATE_MIN_C = 200.0
# Исключает переходные периоды пуска/останова внутри "рабочего" диапазона --
# см. подробное обоснование в compute_reliability_bounds.py (событие
# 2024-04-23: разгон реактора 202->341°C проходил бы порог T11>200°C уже на
# первой точке, хотя режим ещё явно не стационарный).
MAX_RATE_OF_CHANGE_C_PER_10MIN = 2.0


def bounds_for(series, running_mask=None) -> dict:
    s = series.dropna()
    if running_mask is not None:
        s = s[running_mask.reindex(s.index, fill_value=False)]
    q = s.quantile([0.05, 0.50, 0.95])
    return {"p05": round(float(q[0.05]), 4), "p50": round(float(q[0.50]), 4), "p95": round(float(q[0.95]), 4)}


def main() -> None:
    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    ht_running = (ht["T11"] > RUNNING_STATE_MIN_C) & (ht["T11"].diff().abs() <= MAX_RATE_OF_CHANGE_C_PER_10MIN)

    out: dict[str, dict] = {
        "_meta": {
            "is_assumption": True,
            "source": "5-95 перцентиль истории (scripts/compute_control_bounds.py), "
            "методика Ta & Liu (2027), §3.3.1",
        }
    }

    avt_tags = ["T55", "P22", "F25", "F26", "F27", "F28", "F30", "F32"]
    for tag in avt_tags:
        out[f"avt:{tag}"] = bounds_for(avt[tag])

    ht_tags = ["F15", "F25", "T5", "T11", "P13"]
    for tag in ht_tags:
        out[f"242000:{tag}"] = bounds_for(ht[tag], running_mask=ht_running)

    out_path = REPO_ROOT / "config" / "control_bounds.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    print(f"Записано -> {out_path}")
    for k, v in out.items():
        if k != "_meta":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
