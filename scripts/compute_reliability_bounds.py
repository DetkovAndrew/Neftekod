#!/usr/bin/env python3
"""
Считает консервативные перцентильные границы для признаков Агента
надёжности из истории 242000_tags.csv (ARCHITECTURE.md §2 "Правило
границ", §6.3).

Это статистика/калибровка (перцентили), НЕ обучение модели -- никакой
функции потерь не минимизируется, ничего не "учится" предсказывать.
Тем не менее результат -- допущение, а не паспортный предел, и явно
так помечен в config/reliability_bounds.yaml.

Фильтр "установка в работе" (T11 > 200°C) введён потому, что сырые
данные содержат периоды пуска/останова/near-zero значений (T5/T6/T11
падают почти до 0°C, P8 уходит в отрицательные и завышенные до ~7 МПа
значения) -- без фильтра перцентили были бы искажены нерабочими
состояниями, что прямо противоречит рекомендации Ta & Liu (2027, §2.2)
исключать данные пуска/останова/физически неправдоподобные значения
перед статистическими оценками.

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/compute_reliability_bounds.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip  # noqa: E402

RELIABILITY_TAGS = ["P8", "T5", "T6", "T11", "W7"]
RUNNING_STATE_TAG = "T11"
RUNNING_STATE_MIN_C = 200.0
QUANTILES = [0.01, 0.05, 0.50, 0.90, 0.95, 0.99, 1.0]


def main() -> None:
    src = data_dir() / "242000_tags.csv"
    if not src.exists():
        raise SystemExit(f"Не найден {src}. Установите NEFTEKOD_DATA_DIR.")

    ht = load_kip(src)
    running = ht[RUNNING_STATE_TAG] > RUNNING_STATE_MIN_C
    ht_running = ht[running]

    bounds: dict[str, dict] = {
        "_meta": {
            "source": "242000_tags.csv, локальный расчёт (scripts/compute_reliability_bounds.py)",
            "is_assumption": True,
            "note": (
                "Паспортных ограничений оборудования не передано (Q&A-сессия). "
                "Перцентили посчитаны по истории с фильтром "
                f"'{RUNNING_STATE_TAG} > {RUNNING_STATE_MIN_C}°C' (прокси 'установка в работе'), "
                f"доля данных после фильтра: {running.mean():.3f}."
            ),
        }
    }

    for tag in RELIABILITY_TAGS:
        s = ht_running[tag].dropna()
        if tag == "P8":
            # перепад давления физически не может быть <= 0 -- отбрасываем
            # как заведомо недостоверные показания датчика/простоя.
            s = s[s > 0]
        q = s.quantile(QUANTILES)
        bounds[tag] = {str(k): round(float(v), 4) for k, v in q.items()}
        bounds[tag]["n"] = int(len(s))

    out_path = REPO_ROOT / "config" / "reliability_bounds.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(bounds, f, allow_unicode=True, sort_keys=False)
    print(f"Записано -> {out_path}")
    for tag in RELIABILITY_TAGS:
        print(f"  {tag}: {bounds[tag]}")


if __name__ == "__main__":
    main()
