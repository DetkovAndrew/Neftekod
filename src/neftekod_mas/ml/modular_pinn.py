"""
Модульная process-aware архитектура Агента качества (ARCHITECTURE.md
§5.2), по образцу Ta & Liu (2027, Fuel) -- предсказатель зеркалит
топологию процесса (АВТ -> гидроочистка), а не один монолитный "вход
-> выход" чёрный ящик.

Название класса `ModularPINN` сохранено для обратной совместимости, но модель
не заявляется как PINN: в её функции потерь пока нет физического уравнения.

ОБУЧЕНИЕ НЕ ЗАПУСКАЕТСЯ этим модулем при импорте. `train_modular_pinn`
нужно вызвать явно (scripts/train_quality_models.py --model pinn),
что НЕ было сделано в этой сессии -- по договорённости, реальное
обучение откладывается до следующей сессии с бОльшими вычислительными
ресурсами (см. ARCHITECTURE.md §5.2, критерий целесообразности ML).

Архитектура -- shared trunk + раздельные головы по показателям
(multi-task hard parameter sharing), А НЕ единая многовыходная сеть с
общим таргет-вектором, как в оригинальной статье. Причина: в статье
все целевые показатели измеряются СИНХРОННО (почасовые данные завода),
а у нас измерения ЛИМС для разных показателей АСИНХРОННЫ и
разрежены по-разному (сера ~1458 наблюдений за 3.5 года, цетан ~42 --
см. ARCHITECTURE.md §5.2.1) -- нет единого момента времени, где были бы
известны одновременно все 5 целей. Раздельные головы позволяют учить
общий "ствол" (AVT-модуль -> HT-модуль -> общее скрытое представление)
на объединении всех наблюдений всех показателей, а конкретную голову --
только на наблюдениях ЕЁ показателя, без необходимости в синхронных
метках.

    avt_features --[AVTModule]--> avt_latent
    concat(avt_latent, ht_features) --[HTModule]--> shared_hidden
    shared_hidden --[per-metric Linear head]--> предсказание показателя
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from neftekod_mas.ml.dataset import TARGET_METRICS, build_feature_frame, build_training_table
from neftekod_mas.ml.split import chronological_split
from neftekod_mas.schemas import ConfidenceLevel, DataSource, ProcessState, QualityMetricEstimate

AVT_LATENT_DIM = 12
SHARED_HIDDEN_DIM = 24


class AVTModule(nn.Module):
    """Отображает состояние КИП установки АВТ в латентное представление
    прямогонной дизельной фракции -- физическая аналогия: то, что
    формулы AVT6:* (vak_formulas.py) считают явно для T50/D15/CFPP/EBP,
    модуль должен уметь восстанавливать неявно, в общем латентном виде."""

    def __init__(self, n_avt_features: int, latent_dim: int = AVT_LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_avt_features, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, avt_x: torch.Tensor) -> torch.Tensor:
        return self.net(avt_x)


class HTModule(nn.Module):
    """Принимает латентное состояние прямогонной фракции (выход
    AVTModule) вместе с состоянием КИП установки 24-2000 -- физическая
    аналогия: сырьё гидроочистки действительно определяется состоянием
    АВТ, а не независимо от него (см. общую схему процесса,
    Ustanovka_AVT_merged.pdf)."""

    def __init__(self, n_ht_features: int, avt_latent_dim: int = AVT_LATENT_DIM, hidden_dim: int = SHARED_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(avt_latent_dim + n_ht_features, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, avt_latent: torch.Tensor, ht_x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([avt_latent, ht_x], dim=-1))


class ModularPINN(nn.Module):
    def __init__(self, n_avt_features: int, n_ht_features: int, metrics: list[str]):
        super().__init__()
        self.avt_module = AVTModule(n_avt_features)
        self.ht_module = HTModule(n_ht_features)
        # раздельные головы -- см. docstring модуля про асинхронность ЛИМС
        self.heads = nn.ModuleDict({m: nn.Linear(SHARED_HIDDEN_DIM, 1) for m in metrics})

    def forward(self, avt_x: torch.Tensor, ht_x: torch.Tensor, metric: str) -> torch.Tensor:
        avt_latent = self.avt_module(avt_x)
        shared = self.ht_module(avt_latent, ht_x)
        return self.heads[metric](shared).squeeze(-1)


@dataclass
class PINNTrainingResult:
    model: ModularPINN
    avt_columns: list[str]
    ht_columns: list[str]
    feature_mean: pd.Series
    feature_std: pd.Series
    metrics_report: dict[str, dict]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        meta = {
            "avt_columns": self.avt_columns,
            "ht_columns": self.ht_columns,
            "feature_mean": self.feature_mean.to_dict(),
            "feature_std": self.feature_std.to_dict(),
            "metrics_report": self.metrics_report,
        }
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return (df - mean) / std.replace(0, 1.0)


def train_modular_pinn(
    avt_kip: pd.DataFrame,
    ht_kip: pd.DataFrame,
    lims_long: pd.DataFrame,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> PINNTrainingResult:
    """НЕ вызывается автоматически нигде в пайплайне. Обучение --
    round-robin по метрикам на каждой эпохе (общий "ствол" обновляется
    от каждой метрики по очереди, голова -- только от своей), полный
    batch (датасеты малы -- сотни-тысячи наблюдений на метрику, см.
    ARCHITECTURE.md §5.2.1), Adam, MSE на нормированные (train-only
    статистики) цели и признаки. Хронологический split -- ml/split.py,
    без перемешивания (ТЗ п.6)."""
    torch.manual_seed(seed)

    feature_frame = build_feature_frame(avt_kip, ht_kip)
    avt_columns = [c for c in feature_frame.columns if c.startswith("avt:")]
    ht_columns = [c for c in feature_frame.columns if c.startswith("242000:")]

    feature_mean = feature_frame.mean()
    feature_std = feature_frame.std()

    splits = {}
    for metric in TARGET_METRICS:
        table = build_training_table(metric, lims_long, feature_frame)
        splits[metric] = chronological_split(table, train_fraction=0.8)

    model = ModularPINN(len(avt_columns), len(ht_columns), list(TARGET_METRICS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def to_tensors(features: pd.DataFrame, target: pd.Series):
        norm = _normalize(features, feature_mean, feature_std)
        avt_x = torch.tensor(norm[avt_columns].to_numpy(), dtype=torch.float32)
        ht_x = torch.tensor(norm[ht_columns].to_numpy(), dtype=torch.float32)
        y = torch.tensor(target.to_numpy(), dtype=torch.float32)
        return avt_x, ht_x, y

    train_tensors = {m: to_tensors(splits[m].train.features, splits[m].train.target) for m in TARGET_METRICS}
    test_tensors = {m: to_tensors(splits[m].test.features, splits[m].test.target) for m in TARGET_METRICS}

    model.train()
    for _epoch in range(epochs):
        for metric in TARGET_METRICS:
            avt_x, ht_x, y = train_tensors[metric]
            if len(y) == 0:
                continue
            optimizer.zero_grad()
            pred = model(avt_x, ht_x, metric)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()
    report = {}
    with torch.no_grad():
        for metric in TARGET_METRICS:
            avt_x, ht_x, y = test_tensors[metric]
            if len(y) == 0:
                report[metric] = {"n_test": 0}
                continue
            pred = model(avt_x, ht_x, metric)
            errors = (pred - y).numpy()
            report[metric] = {
                "n_train": len(train_tensors[metric][2]),
                "n_test": len(y),
                "test_mae": float(np.abs(errors).mean()),
                "test_rmse": float(np.sqrt((errors ** 2).mean())),
            }
            print(f"{metric}: MAE={report[metric]['test_mae']:.4f} RMSE={report[metric]['test_rmse']:.4f}")

    return PINNTrainingResult(
        model=model, avt_columns=avt_columns, ht_columns=ht_columns,
        feature_mean=feature_mean, feature_std=feature_std, metrics_report=report,
    )
