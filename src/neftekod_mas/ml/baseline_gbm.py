"""
Базовая ML-модель Агента качества -- градиентный бустинг (LightGBM),
один регрессор на показатель (ARCHITECTURE.md §5.2). Простой, быстрый,
стандартный baseline для soft sensor'ов -- прежде чем переходить к
модульной PI-архитектуре (modular_pinn.py), нужна точка отсчёта:
физически-информированная модель имеет смысл только если она бьёт
обычный бустинг на chronological holdout, иначе сложность не оправдана.

ОБУЧЕНИЕ НЕ ЗАПУСКАЕТСЯ этим модулем при импорте -- `train_all_metrics`
нужно вызвать явно (см. scripts/train_quality_models.py, тоже не
запущен). Реализует протокол `QualityPredictor` (schemas.py), поэтому
после обучения взаимозаменяем с `VAKFormulaPredictor` без изменения
Агента качества/оптимизации.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from neftekod_mas.ml.dataset import TARGET_METRICS, build_feature_frame, build_training_table, TrainingTable
from neftekod_mas.ml.split import SharedTimeSplit, chronological_split
from neftekod_mas.schemas import ConfidenceLevel, DataSource, ProcessState, QualityMetricEstimate

# eval_set -> eval_X/eval_y появился только в lightgbm 4.7; старый API оставлен ради
# совместимости с окружением кластера, предупреждение о депрекации глушится точечно.
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")

DEFAULT_LGB_PARAMS: dict = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,  # немного (сотни-тысячи наблюдений на метрику, см. ml/dataset.py) -- небольшое дерево против переобучения
    "min_child_samples": 10,
    "verbosity": -1,
}

def _sanitize_feature_name(col: str) -> str:
    """LightGBM запрещает 'специальные JSON-символы' в именах признаков
    (найдено тестом: обучение падало с LightGBMError на любом теге вида
    'avt:T1' -- двоеточие из нашей же составной квалификации installation:tag,
    см. tags/pid_graph.py). Заменяем безопасно и обратимо не требуется --
    исходные (канонические) имена остаются в MetricModel.feature_columns
    для сопоставления с ProcessState.kip, санитизация применяется только
    непосредственно перед вызовом LightGBM."""
    return col.replace(":", "__")


METRIC_UNITS = {
    "sulfur_mg_kg": "мг/кг",
    "t95_c": "°C",
    "cetane_number": "ед.цет.ч.",
    "cfpp_c": "°C",
    "density_kg_m3": "кг/м3",
}


@dataclass
class MetricModel:
    metric: str
    booster: lgb.Booster  # предсказание всегда через Booster.predict -- не зависит от private-состояния LGBMRegressor при save/load
    feature_columns: list[str]
    test_mae: float
    test_rmse: float
    n_train: int
    n_test: int


@dataclass
class GBMExperimentResult:
    """Fitted models and untouched-period metrics for one common time split."""

    models: dict[str, lgb.LGBMRegressor]
    report: dict[str, dict]


def train_gbm_on_shared_split(
    tables: dict[str, TrainingTable], split: SharedTimeSplit, lgb_params: dict | None = None,
) -> GBMExperimentResult:
    """Fit one GBM per metric using train only; validation chooses tree count.

    The returned test metrics are reporting-only and do not influence fitting or
    model selection.
    """
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    models, report = {}, {}
    for metric, table in tables.items():
        target = pd.to_numeric(table.target, errors="coerce")
        valid = np.isfinite(target.to_numpy(dtype=float))
        features, target = table.features.loc[valid], target.loc[valid]
        masks = split.masks(TrainingTable(metric, features, target, table.target_age_minutes.loc[valid]))
        if masks["train"].sum() < 2 or not masks["validation"].any() or not masks["test"].any():
            raise ValueError(f"{metric}: insufficient observations for common train/validation/test split")
        train_x = features.loc[masks["train"]].rename(columns=_sanitize_feature_name)
        validation_x = features.loc[masks["validation"]].rename(columns=_sanitize_feature_name)
        test_x = features.loc[masks["test"]].rename(columns=_sanitize_feature_name)
        model = lgb.LGBMRegressor(**params)
        model.fit(train_x, target.loc[masks["train"]],
                  eval_set=[(validation_x, target.loc[masks["validation"]])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        validation_pred = model.predict(validation_x)
        test_pred = model.predict(test_x)
        validation_y, test_y = target.loc[masks["validation"]].to_numpy(), target.loc[masks["test"]].to_numpy()
        median = float(np.median(target.loc[masks["train"]]))
        report[metric] = {
            "n_train": int(masks["train"].sum()), "n_validation": int(masks["validation"].sum()),
            "n_test": int(masks["test"].sum()), "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
            "validation_mae": float(np.abs(validation_pred - validation_y).mean()),
            "validation_baseline_mae": float(np.abs(validation_y - median).mean()),
            "test_mae": float(np.abs(test_pred - test_y).mean()),
            "test_rmse": float(np.sqrt(((test_pred - test_y) ** 2).mean())),
            "baseline_median_mae": float(np.abs(test_y - median).mean()),
        }
        models[metric] = model
    return GBMExperimentResult(models=models, report=report)


@dataclass
class GBMQualityPredictor:
    """Реализует schemas.QualityPredictor. Один регрессор на показатель."""

    models: dict[str, MetricModel] = field(default_factory=dict)

    def predict(self, state: ProcessState) -> list[QualityMetricEstimate]:
        row = {qid: r.value for qid, r in state.kip.items()}
        results = []
        for metric, mm in self.models.items():
            x = pd.DataFrame([{_sanitize_feature_name(c): row.get(c, np.nan) for c in mm.feature_columns}])
            if x.isna().any(axis=None):
                continue  # не хватает тегов КИП -- честно не оцениваем, как и ВАК-формулы
            value = float(mm.booster.predict(x)[0])
            results.append(QualityMetricEstimate(
                metric=metric, value=value, unit=METRIC_UNITS[metric],
                source=DataSource.ML_MODEL, age_minutes=None,
                confidence=ConfidenceLevel.MEDIUM, typical_error=mm.test_mae,
            ))
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {}
        for metric, mm in self.models.items():
            mm.booster.save_model(str(directory / f"{metric}.txt"))
            meta[metric] = {
                "feature_columns": mm.feature_columns,
                "test_mae": mm.test_mae,
                "test_rmse": mm.test_rmse,
                "n_train": mm.n_train,
                "n_test": mm.n_test,
            }
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "GBMQualityPredictor":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        models = {}
        for metric, m in meta.items():
            booster = lgb.Booster(model_file=str(directory / f"{metric}.txt"))
            models[metric] = MetricModel(
                metric=metric, booster=booster, feature_columns=m["feature_columns"],
                test_mae=m["test_mae"], test_rmse=m["test_rmse"],
                n_train=m["n_train"], n_test=m["n_test"],
            )
        return cls(models=models)


def train_all_metrics(
    avt_kip: pd.DataFrame,
    ht_kip: pd.DataFrame,
    lims_long: pd.DataFrame,
    lgb_params: dict | None = None,
) -> GBMQualityPredictor:
    """Обучает по одному регрессору LightGBM на каждый из 5 показателей
    (TARGET_METRICS), на хронологическом split (ml/split.py). НЕ
    вызывается автоматически нигде в пайплайне -- явный шаг
    scripts/train_quality_models.py."""
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    feature_frame = build_feature_frame(avt_kip, ht_kip)

    predictor = GBMQualityPredictor()
    for metric in TARGET_METRICS:
        table = build_training_table(metric, lims_long, feature_frame)
        split = chronological_split(table, train_fraction=0.8)

        original_columns = list(split.train.features.columns)  # канонические имена (installation:tag) -- хранятся в MetricModel
        train_x = split.train.features.rename(columns=_sanitize_feature_name)
        test_x = split.test.features.rename(columns=_sanitize_feature_name)

        model = lgb.LGBMRegressor(**params)
        model.fit(train_x, split.train.target)

        pred = model.predict(test_x)
        errors = pred - split.test.target.to_numpy()
        mae = float(np.abs(errors).mean())
        rmse = float(np.sqrt((errors ** 2).mean()))

        predictor.models[metric] = MetricModel(
            metric=metric, booster=model.booster_, feature_columns=original_columns,
            test_mae=mae, test_rmse=rmse, n_train=len(split.train.target), n_test=len(split.test.target),
        )
    return predictor
