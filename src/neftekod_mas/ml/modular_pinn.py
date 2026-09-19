"""
Модульная Physics-Informed архитектура Агента качества (ARCHITECTURE.md
§5.2), по образцу Ta & Liu (2027, Fuel) -- предсказатель зеркалит
топологию процесса (АВТ -> гидроочистка), а не один монолитный "вход
-> выход" чёрный ящик.

Обучение запускается явно через train_modular_pinn. Это сеть с топологией
процесса: физических уравнений или physics loss в текущей реализации нет.

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
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from neftekod_mas.ml.dataset import FeatureMode, TARGET_METRICS, TemporalFeatureSpec, build_feature_frame, build_training_table
from neftekod_mas.ml.split import shared_time_split
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
    target_mean: dict[str, float] = field(default_factory=dict)
    target_std: dict[str, float] = field(default_factory=dict)
    training_info: dict = field(default_factory=dict)

    def predict(self, features: pd.DataFrame, metric: str) -> np.ndarray:
        """Return predictions in original laboratory units."""
        device = next(self.model.parameters()).device
        norm = _normalize(features, self.feature_mean, self.feature_std)
        if self.training_info.get("feature_clip") is not None:
            norm = norm.clip(-self.training_info["feature_clip"], self.training_info["feature_clip"])
        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                torch.tensor(norm[self.avt_columns].to_numpy(dtype="float32"), device=device),
                torch.tensor(norm[self.ht_columns].to_numpy(dtype="float32"), device=device),
                metric,
            ).cpu().numpy()
        return pred * self.target_std.get(metric, 1.0) + self.target_mean.get(metric, 0.0)

    @classmethod
    def load(cls, directory: Path) -> "PINNTrainingResult":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        model = ModularPINN(len(meta["avt_columns"]), len(meta["ht_columns"]), list(meta["metrics_report"]))
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        return cls(model=model, avt_columns=meta["avt_columns"], ht_columns=meta["ht_columns"],
                   feature_mean=pd.Series(meta["feature_mean"]), feature_std=pd.Series(meta["feature_std"]),
                   metrics_report=meta["metrics_report"], target_mean=meta.get("target_mean", {}),
                   target_std=meta.get("target_std", {}), training_info=meta.get("training_info", {}))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        meta = {
            "avt_columns": self.avt_columns,
            "ht_columns": self.ht_columns,
            "feature_mean": self.feature_mean.to_dict(),
            "feature_std": self.feature_std.to_dict(),
            "metrics_report": self.metrics_report,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "training_info": self.training_info,
            "format_version": 2,
        }
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return ((df.replace([np.inf, -np.inf], np.nan) - mean) / std.replace(0, 1.0)).fillna(0.0)


def train_modular_pinn(
    avt_kip: pd.DataFrame,
    ht_kip: pd.DataFrame,
    lims_long: pd.DataFrame,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 30,
    weight_decay: float = 1e-4,
    feature_clip: float = 8.0,
    feature_mode: FeatureMode = "snapshot",
    temporal_spec: TemporalFeatureSpec = TemporalFeatureSpec(),
) -> PINNTrainingResult:
    """Shared 60/20/20 time split, train-only scaling, balanced Huber loss.

    Select the checkpoint by validation MAE in normalized units. All test
    metrics are computed only after selection, in original laboratory units.
    """
    if epochs < 1 or patience < 1 or lr <= 0 or weight_decay < 0 or feature_clip <= 0:
        raise ValueError("epochs/patience/lr must be positive and weight_decay nonnegative")
    torch.manual_seed(seed)

    feature_frame = build_feature_frame(avt_kip, ht_kip, mode=feature_mode, temporal_spec=temporal_spec)
    avt_columns = [c for c in feature_frame.columns if c.startswith("avt:")]
    ht_columns = [c for c in feature_frame.columns if c.startswith("242000:")]

    tables = {}
    for metric in TARGET_METRICS:
        table = build_training_table(metric, lims_long, feature_frame)
        table.target = pd.to_numeric(table.target, errors="coerce")
        valid = np.isfinite(table.target.to_numpy(dtype=float))
        table.features = table.features.loc[valid]
        table.target = table.target.loc[valid]
        tables[metric] = table

    # Shared trunk must never train on a later label from another head.
    split = shared_time_split(tables)
    validation_at, test_at = split.validation_at, split.test_at
    masks = {m: split.masks(t) for m, t in tables.items()}
    for m in tables:
        if masks[m]["train"].sum() < 2 or not masks[m]["validation"].any():
            raise ValueError(f"{m}: need >=2 training and >=1 validation observations at shared cutoffs")
    train_features = pd.concat([t.features.loc[masks[m]["train"]] for m, t in tables.items()])
    train_features = train_features.loc[~train_features.index.duplicated()].replace([np.inf, -np.inf], np.nan)
    feature_mean = train_features.mean().fillna(0.0)
    feature_std = train_features.std().fillna(1.0).replace(0, 1.0)
    # Legacy field names mean/std now hold robust center/scale (median/IQR).
    target_mean = {m: float(t.target.loc[masks[m]["train"]].median()) for m, t in tables.items()}
    target_std = {m: max(float(t.target.loc[masks[m]["train"]].quantile(.75) -
                               t.target.loc[masks[m]["train"]].quantile(.25)), 1.0)
                  for m, t in tables.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModularPINN(len(avt_columns), len(ht_columns), list(TARGET_METRICS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    def to_tensors(features: pd.DataFrame, target: pd.Series):
        norm = _normalize(features, feature_mean, feature_std).clip(-feature_clip, feature_clip)

        avt_x = torch.tensor(norm[avt_columns].to_numpy(dtype="float32"), dtype=torch.float32, device=device)
        ht_x = torch.tensor(norm[ht_columns].to_numpy(dtype="float32"), dtype=torch.float32, device=device)

        y_np = pd.to_numeric(target, errors="coerce").to_numpy(dtype="float32", na_value=np.nan)
        y = torch.tensor(y_np, dtype=torch.float32, device=device)

        return avt_x, ht_x, y

    tensors = {part: {m: to_tensors(t.features.loc[masks[m][part]],
                        (t.target.loc[masks[m][part]] - target_mean[m]) / target_std[m])
                      for m, t in tables.items()} for part in ("train", "validation", "test")}
    train_tensors = tensors["train"]
    test_tensors = tensors["test"]
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    history = []

    model.train()
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        losses = []
        for metric in TARGET_METRICS:
            avt_x, ht_x, y = train_tensors[metric]
            if len(y) == 0:
                continue
            pred = model(avt_x, ht_x, metric)
            losses.append(loss_fn(pred, y))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.stack([nn.functional.l1_loss(model(a, h, m), y)
                            for m, (a, h, y) in tensors["validation"].items()]).mean())
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": _epoch + 1, "train_huber": float(loss.detach()), "validation_mae_normalized": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, _epoch + 1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)

    model.eval()
    report = {}
    with torch.no_grad():
        for metric in TARGET_METRICS:
            avt_x, ht_x, y = test_tensors[metric]
            if len(y) == 0:
                report[metric] = {"n_test": 0}
                continue
            pred = model(avt_x, ht_x, metric)
            errors = (pred - y).detach().cpu().numpy() * target_std[metric]
            train_y = tables[metric].target.loc[masks[metric]["train"]].to_numpy(dtype=float)
            test_y = tables[metric].target.loc[masks[metric]["test"]].to_numpy(dtype=float)
            baseline_mae = float(np.abs(test_y - np.median(train_y)).mean())
            va, vh, vy = tensors["validation"][metric]
            validation_mae = float((model(va, vh, metric) - vy).abs().mean()) * target_std[metric]
            validation_baseline_mae = float(vy.abs().mean()) * target_std[metric]
            report[metric] = {
                "n_train": len(train_tensors[metric][2]),
                "n_test": len(y),
                "test_mae": float(np.abs(errors).mean()),
                "test_rmse": float(np.sqrt((errors ** 2).mean())),
                "n_validation": len(tensors["validation"][metric][2]),
                "baseline_median_mae": baseline_mae,
                "beats_median_baseline": bool(np.abs(errors).mean() < baseline_mae),
                "limited_test_support": len(y) < 30,
                "validation_mae": validation_mae,
                "validation_baseline_mae": validation_baseline_mae,
                "validation_prefers_network": validation_mae < validation_baseline_mae,
            }
            print(f"{metric}: MAE={report[metric]['test_mae']:.4f} RMSE={report[metric]['test_rmse']:.4f}")

    return PINNTrainingResult(
        model=model, avt_columns=avt_columns, ht_columns=ht_columns,
        feature_mean=feature_mean, feature_std=feature_std, metrics_report=report,
        target_mean=target_mean, target_std=target_std,
        training_info={"validation_at": str(validation_at), "test_at": str(test_at),
                       "best_epoch": best_epoch, "epochs_run": len(history), "seed": seed,
                       "lr": lr, "patience": patience, "weight_decay": weight_decay,
                       "feature_clip": feature_clip, "target_scaling": "median_iqr",
                       "feature_mode": feature_mode,
                       "history": history},
    )
