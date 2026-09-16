#!/usr/bin/env python3
"""
Строит опорное облако точек совместного (joint) исторического состояния
активных управляющих переменных -- закрывает пробел ARCHITECTURE.md §5.3:
Guard.within_bounds раньше проверял только МАРЖИНАЛЬНЫЙ диапазон каждой
переменной по отдельности, что не ловит случай (Ta & Liu 2027, §3.3.3),
когда каждая переменная по отдельности в допустимых пределах, но их
СОВМЕСТНАЯ комбинация в истории не встречалась.

Артефакт -- НЕ аггрегированная статистика вроде перцентилей, а фактически
проекция реальной телеметрии на 7 измерений (пусть и без временных меток
и не всех тегов) -- поэтому, в отличие от config/*_bounds.yaml,
НЕ коммитится в git (см. .gitignore: config/joint_envelope.npz)
и генерируется локально этим скриптом при необходимости.

Метод: даунсемплинг (не обучение) до NUM_POINTS строк, нормализация
каждой переменной на её собственный [p05, p95] (control_bounds.yaml),
сохранение как numpy-массив для brute-force nearest-neighbor поиска в
optimization/joint_envelope.py. Это НЕ ML-модель -- ближайший сосед по
уже посчитанным координатам, без какой-либо оптимизации параметров.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip  # noqa: E402
from neftekod_mas.utils.config import load_control_bounds  # noqa: E402

NUM_POINTS = 20_000
SEED = 42

# Порядок координат ФИКСИРОВАН -- тот же порядок используется при
# построении вектора кандидата в optimization/joint_envelope.py.
ACTIVE_QUALIFIED_TAGS = [
    "avt:T55", "avt:P22", "avt:F30", "avt:F32",
    "242000:F15", "242000:T5", "242000:P13",
]


def main() -> None:
    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    bounds = load_control_bounds()

    cols = {}
    for qtag in ACTIVE_QUALIFIED_TAGS:
        installation, tag = qtag.split(":")
        src = avt if installation == "avt" else ht
        cols[qtag] = src[tag]

    # Общий индекс по обеим установкам (они синхронны по 10-мин сетке,
    # но пересекаем на всякий случай явно, а не предполагаем совпадение).
    common_index = cols[ACTIVE_QUALIFIED_TAGS[0]].index
    for qtag in ACTIVE_QUALIFIED_TAGS[1:]:
        common_index = common_index.intersection(cols[qtag].index)

    matrix = np.column_stack([cols[qtag].reindex(common_index).to_numpy() for qtag in ACTIVE_QUALIFIED_TAGS])
    mask = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[mask]

    for i, qtag in enumerate(ACTIVE_QUALIFIED_TAGS):
        b = bounds[qtag]
        span = max(b["p95"] - b["p05"], 1e-9)
        matrix[:, i] = (matrix[:, i] - b["p05"]) / span

    rng = np.random.default_rng(SEED)
    if len(matrix) > NUM_POINTS:
        idx = rng.choice(len(matrix), size=NUM_POINTS, replace=False)
        matrix = matrix[idx]

    out_path = REPO_ROOT / "config" / "joint_envelope.npz"
    np.savez_compressed(out_path, points=matrix.astype("float32"), tags=np.array(ACTIVE_QUALIFIED_TAGS))
    print(f"Записано {matrix.shape[0]} точек x {matrix.shape[1]} измерений -> {out_path}")


if __name__ == "__main__":
    main()
