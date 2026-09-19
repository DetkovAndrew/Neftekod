"""
Построение обучающего датасета для будущего ML-слоя Агента качества
(ARCHITECTURE.md §5.2 -- "Опциональный ML/PI-слой"). ОБУЧЕНИЕ НЕ
ЗАПУСКАЕТСЯ этим модулем -- только подготовка признаков/целей.

В отличие от `data/sync.py` (Data & Sync Agent, синхронизация ОДНОГО
момента времени для принятия решения в проде), здесь нужна эффективная
ВЕКТОРИЗОВАННАЯ синхронизация ВСЕЙ истории сразу -- вызывать
`build_process_state` по одному разу на каждую из ~190k точек КИП было
бы на порядки медленнее. Используется `pandas.merge_asof` (backward),
та же логика (только последнее известное значение на момент T, никогда
не из будущего), но за один проход.

Единица наблюдения -- каждое РЕАЛЬНОЕ измерение ЛИМС целевого
показателя (не искусственно на каждые 10 минут: апсемплинг цели по
несуществующим измерениям создал бы иллюзию точности и дублировал
метки). Для сети сера ~1462 наблюдения, T95 ~1288, D15 ~1011, CFPP
~414 -- это отражает реальную частоту лабораторного контроля, не
искусственно раздуто.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from neftekod_mas.quality.quality_agent import GODT_POINT2

# metric -> (lims_param в точке GODT_POINT2, unit)
TARGET_METRICS: dict[str, str] = {
    "sulfur_mg_kg": "Mg.Sulfur",
    "t95_c": "95%.T",
    "cetane_number": "CetaneNumber",
    "cfpp_c": "CFPP",
    "density_kg_m3": "D15",
}

KIP_MATCH_TOLERANCE = pd.Timedelta(minutes=30)

# Chosen from the process ontology: crude feed density/cuts on AVT and reactor,
# hydrogen, feed/product flows and online sulphur on hydrotreater.  The list is
# deliberately small so sparse LIMS labels cannot be overwhelmed by hundreds of
# arbitrary lagged tags.
TEMPORAL_TAGS: tuple[str, ...] = (
    "avt:D10", "avt:F30", "avt:F32", "avt:F34", "avt:T33", "avt:T48",
    "avt:T55", "242000:F9", "242000:F14", "242000:F15", "242000:F17",
    "242000:F25", "242000:P8", "242000:P13", "242000:Q20", "242000:Q21",
    "242000:T5", "242000:T6", "242000:T11", "242000:T23",
)


@dataclass(frozen=True)
class TemporalFeatureSpec:
    """Causal history features sampled on the native 10-minute KIP grid."""

    tags: tuple[str, ...] = TEMPORAL_TAGS
    lags: tuple[pd.Timedelta, ...] = (pd.Timedelta(minutes=30), pd.Timedelta(hours=2))
    rolling_windows: tuple[pd.Timedelta, ...] = (pd.Timedelta(minutes=30), pd.Timedelta(hours=2))


FeatureMode = Literal["snapshot", "causal_temporal"]


@dataclass
class TrainingTable:
    metric: str
    features: pd.DataFrame  # index = measured_at (ЛИМС), колонки = "avt:<tag>"/"242000:<tag>"
    target: pd.Series  # index = measured_at, значение целевого показателя
    target_age_minutes: pd.Series  # для справки/анализа, НЕ признак


def _prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.add_prefix(f"{prefix}:")


_KIP_TIME_COL = "kip_time"


def build_feature_frame(
    avt_kip: pd.DataFrame,
    ht_kip: pd.DataFrame,
    mode: FeatureMode = "snapshot",
    temporal_spec: TemporalFeatureSpec = TemporalFeatureSpec(),
) -> pd.DataFrame:
    """Объединённая по времени (inner join индекса, обе установки
    синхронны по 10-мин сетке, см. ТЗ) таблица признаков КИП. Имя
    индекса устанавливается ЯВНО (не полагаемся на то, что вызывающий
    код прогнал данные через loaders.load_kip, где оно "date" по
    побочному эффекту set_index) -- иначе build_training_table ломается
    на любых данных, не прошедших именно через тот загрузчик."""
    avt = _prefixed(avt_kip, "avt")
    ht = _prefixed(ht_kip, "242000")
    merged = avt.join(ht, how="inner").sort_index()
    merged.index.name = _KIP_TIME_COL
    if mode == "snapshot":
        return merged
    if mode != "causal_temporal":
        raise ValueError(f"Unknown feature mode: {mode}")

    missing = set(temporal_spec.tags) - set(merged.columns)
    if missing:
        raise ValueError(f"Tags configured for temporal features are absent: {sorted(missing)}")
    selected = merged.loc[:, list(temporal_spec.tags)]
    extras: dict[str, pd.Series] = {}
    # shift/rolling use only the present or preceding KIP states. No value after
    # the decision timestamp can enter a feature.
    for lag in temporal_spec.lags:
        label = f"lag_{int(lag.total_seconds() // 60)}m"
        for tag in temporal_spec.tags:
            extras[f"{tag}__{label}"] = selected[tag].shift(freq=lag)
    for window in temporal_spec.rolling_windows:
        label = f"mean_{int(window.total_seconds() // 60)}m"
        averages = selected.rolling(window=window, min_periods=1, closed="both").mean()
        for tag in temporal_spec.tags:
            extras[f"{tag}__{label}"] = averages[tag]
    temporal = pd.DataFrame(extras, index=merged.index, dtype="float32")
    return pd.concat([merged, temporal], axis=1)


def build_training_table(
    metric: str,
    lims_long: pd.DataFrame,
    feature_frame: pd.DataFrame,
    lims_point: str = GODT_POINT2,
) -> TrainingTable:
    if metric not in TARGET_METRICS:
        raise ValueError(f"Неизвестная метрика {metric}, ожидается одна из {list(TARGET_METRICS)}")
    lims_param = TARGET_METRICS[metric]

    targets = (
        lims_long[(lims_long["point_label"] == lims_point) & (lims_long["param"] == lims_param)]
        .sort_values("measured_at")
        .set_index("measured_at")["value"]
    )
    if targets.empty:
        raise ValueError(f"Нет ЛИМС-записей для {lims_point} / {lims_param}")

    ff = feature_frame.sort_index()
    if ff.index.name != _KIP_TIME_COL:
        raise ValueError(
            f"feature_frame должен быть построен через build_feature_frame() "
            f"(ожидалось имя индекса '{_KIP_TIME_COL}', получено {ff.index.name!r})"
        )
    targets_df = targets.rename("target").reset_index()  # колонки: [measured_at, target]
    kip_df = ff.reset_index()  # колонка _KIP_TIME_COL уже названа правильно

    # merge_asof: для каждого времени измерения ЛИМС берём ближайшее ПРЕДЫДУЩЕЕ
    # (direction='backward') состояние КИП в пределах допуска -- никогда из будущего.
    merged = pd.merge_asof(
        targets_df,
        kip_df,
        left_on="measured_at",
        right_on=_KIP_TIME_COL,
        direction="backward",
        tolerance=KIP_MATCH_TOLERANCE,
    )
    merged = merged.dropna(subset=[_KIP_TIME_COL]).copy()  # нет КИП в допуске -- выбрасываем наблюдение; .copy() убирает pandas-предупреждение о фрагментации при вставке новой колонки ниже
    merged["age_minutes"] = (merged["measured_at"] - merged[_KIP_TIME_COL]).dt.total_seconds() / 60.0

    feature_cols = [c for c in ff.columns]
    features = merged.set_index("measured_at")[feature_cols]
    target = merged.set_index("measured_at")["target"]
    age = merged.set_index("measured_at")["age_minutes"]

    return TrainingTable(metric=metric, features=features, target=target, target_age_minutes=age)
