"""
Проверка совместного (joint) исторического состояния активных
управляющих переменных -- см. scripts/compute_joint_envelope.py и
ARCHITECTURE.md §5.3. Brute-force nearest-neighbor по нормализованному
облаку точек истории; НЕ модель, ничего не обучается и не подгоняется --
чистый геометрический поиск ближайшего соседа по уже готовым координатам.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class JointEnvelopeChecker:
    def __init__(self, points: np.ndarray, tags: list[str], bounds: dict):
        self.points = points  # (N, D), уже нормализовано на [p05,p95] при построении
        self.tags = tags
        self.bounds = {k: v for k, v in bounds.items() if k != "_meta"}

    @classmethod
    def from_npz(cls, path: Path, bounds: dict) -> "JointEnvelopeChecker":
        data = np.load(path, allow_pickle=True)
        return cls(points=data["points"], tags=list(data["tags"]), bounds=bounds)

    def _normalize(self, tag: str, value: float) -> float:
        b = self.bounds[tag]
        span = max(b["p95"] - b["p05"], 1e-9)
        return (value - b["p05"]) / span

    def nearest_distance(self, current_values: dict[str, float]) -> float | None:
        """current_values -- полный вектор ЗНАЧЕНИЙ ПОСЛЕ применения
        кандидата (изменённый тег + текущие значения остальных активных
        тегов). Возвращает евклидово расстояние (в нормализованных
        единицах) до ближайшей исторически наблюдавшейся точки, или
        None, если не все координаты доступны (тег отсутствует в
        состоянии -- проверка тогда пропускается, не блокирует)."""
        vec = []
        for tag in self.tags:
            if tag not in current_values:
                return None
            vec.append(self._normalize(tag, current_values[tag]))
        v = np.array(vec, dtype="float32")
        dists = np.linalg.norm(self.points - v, axis=1)
        return float(dists.min())
