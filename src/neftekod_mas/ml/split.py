"""
Хронологический train/test split (ТЗ п.6: "Случайное перемешивание строк
при train/test split для временных рядов не допускается"; ARCHITECTURE.md
§12; практика обеих "полезных статей" -- Ta & Liu 2027 §2.5 и обычная
практика оценки soft sensor'ов).

Никакого перемешивания, никакой случайности -- первые `train_fraction`
наблюдений ПО ВРЕМЕНИ идут в train, остаток -- в test. Это разбиение
по МОМЕНТУ ИЗМЕРЕНИЯ ЦЕЛИ (ЛИМС), а не по времени КИП-снимка (они почти
совпадают по построению dataset.py, но именно момент цели -- то, что
физически нельзя знать заранее).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from neftekod_mas.ml.dataset import TrainingTable


@dataclass
class ChronoSplit:
    train: TrainingTable
    test: TrainingTable
    split_at: pd.Timestamp


def chronological_split(table: TrainingTable, train_fraction: float = 0.8) -> ChronoSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction должен быть в (0, 1)")

    ordered_index = table.target.sort_index().index
    n = len(ordered_index)
    if n < 10:
        raise ValueError(f"Слишком мало наблюдений ({n}) для содержательного split")

    split_pos = int(n * train_fraction)
    split_at = ordered_index[split_pos]

    train_mask = table.features.index < split_at
    test_mask = ~train_mask

    train = TrainingTable(
        metric=table.metric,
        features=table.features[train_mask],
        target=table.target[train_mask],
        target_age_minutes=table.target_age_minutes[train_mask],
    )
    test = TrainingTable(
        metric=table.metric,
        features=table.features[test_mask],
        target=table.target[test_mask],
        target_age_minutes=table.target_age_minutes[test_mask],
    )
    return ChronoSplit(train=train, test=test, split_at=split_at)
