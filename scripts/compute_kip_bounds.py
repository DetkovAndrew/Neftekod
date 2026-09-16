#!/usr/bin/env python3
"""
Считает широкие (0.5-99.5 перцентиль) границы правдоподобия для ВСЕХ
тегов КИП обеих установок -- используется Data & Sync Agent для флага
`out_of_range` (ARCHITECTURE.md §2, §6.1). Статистика, не обучение.

Сознательно ШИРЕ, чем reliability_bounds.yaml (которые используют
режимный фильтр "установка в работе" и узкий диапазон 90-99%
специально для реакторных тегов): здесь цель -- поймать явно
недостоверные показания датчика (обрыв, выброс), а не отклонения
режима, для которых уже есть отдельный, более тонкий Агент надёжности.
Поэтому фильтр "установка в работе" НЕ применяется -- иначе периоды
пуска/останова сами были бы объявлены "недостоверными", что неверно:
это реальные, просто нечастые состояния процесса.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip  # noqa: E402

QUANTILES = (0.005, 0.995)


def main() -> None:
    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")

    bounds: dict[str, dict] = {
        "_meta": {
            "is_assumption": True,
            "note": "0.5-99.5 перцентиль полной истории, без режимного фильтра. "
            "Ловит явные сбои датчика, а не отклонения режима (для этого -- reliability_bounds.yaml).",
        }
    }
    for prefix, df in (("avt", avt), ("242000", ht)):
        for tag in df.columns:
            q = df[tag].dropna().quantile(list(QUANTILES))
            lo, hi = float(q.iloc[0]), float(q.iloc[1])
            margin = 0.1 * max(abs(hi - lo), 1.0)
            bounds[f"{prefix}:{tag}"] = {"low": round(lo - margin, 4), "high": round(hi + margin, 4)}

    out_path = REPO_ROOT / "config" / "kip_bounds.json"
    out_path.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано {len(bounds) - 1} тегов -> {out_path}")


if __name__ == "__main__":
    main()
