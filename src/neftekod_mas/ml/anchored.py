"""
"Якорные" soft-sensor'ы: физический/приборный якорь + поправка смещения
по ПРОШЛЫМ лабораторным замерам (+ опционально сильно регуляризованная
коррекция остатка). Ответ на переобучение монолитных моделей (MODEL_AUDIT.md):
вместо того чтобы заставлять сеть/бустинг заново выучивать из ~100 признаков
то, что уже известно физически, модель стартует с якоря и учит только
малую, медленно меняющуюся поправку.

Якоря по показателям (всё -- только по данным, доступным на момент замера):
- сера: поточный анализатор КИП 242000:Q21 ("анализатор серы в г/о ДТ"),
  валиден в пределах шкалы [0.2, 24.5] мг/кг (у верхней границы -- насыщение,
  у нуля -- выключен);
- T95: ВАК 24-2000:GODT:T95 (КИП + ПРЕДЫДУЩЕЕ значение ЛИМС T95);
- CFPP: ВАК 24-2000:GODT:CFPP (только КИП);
- плотность: ВАК 24-2000:GODT:D15 (КИП + предыдущее ЛИМС D15);
- цетановое число: ASTM D976 от предыдущих ЛИМС D15 и T50.

Поправка смещения -- скользящая медиана (y - якорь) по K предыдущим
лабораторным замерам, строго до текущего (shift(1)): так поддерживают
промышленные анализаторы/ВАК -- подстройкой по лаборатории, без переобучения.

Область применимости -- стационарный режим реактора (steady_state_mask):
замеры во время пусков/остановов (например, сера 2120 мг/кг при разгоне
реактора 2024-04-23, ARCHITECTURE.md §12.1) не используются ни для
подгонки, ни для основной оценки, и отчитываются отдельно -- в эти
периоды система должна отказываться, а не предсказывать.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from neftekod_mas.ml.dataset import TrainingTable
from neftekod_mas.quality import vak_formulas as vak
from neftekod_mas.quality.quality_agent import GODT_POINT2

# eval_set -> eval_X/eval_y появился только в lightgbm 4.7; старый API оставлен ради
# совместимости с окружением кластера, предупреждение о депрекации глушится точечно.
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")

Q21_VALID_RANGE = (0.2, 24.5)
STEADY_T11_MIN_C = 340.0
STEADY_T11_MAX_CHANGE_2H_C = 10.0

RESIDUAL_GBM_PARAMS: dict = {
    "objective": "huber",
    "alpha": 1.0,
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 30,
    "reg_lambda": 10.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 0,
    "deterministic": True,
}

# Небольшой физически мотивированный набор для коррекции остатка --
# реакторная секция и поток; сотни признаков на ~1000 наблюдений = переобучение.
RESIDUAL_FEATURES: tuple[str, ...] = (
    "242000:T5", "242000:T6", "242000:T11", "242000:P13", "242000:P8",
    "242000:F9", "242000:F15", "242000:F25", "242000:T23", "242000:Q21",
    "avt:T55", "avt:F30", "avt:F32",
)


def steady_state_mask(table: TrainingTable, ht_kip: pd.DataFrame) -> pd.Series:
    """True -- замер сделан в стационарном режиме реактора (T11 >= 340°C и
    изменение T11 за предыдущие 2 ч не более 10°C). Использует только
    текущие/прошлые значения КИП."""
    t11 = ht_kip["T11"]
    change_2h = (t11 - t11.shift(freq=pd.Timedelta(hours=2)).reindex(t11.index)).abs()
    x = table.features["242000:T11"]
    change = change_2h.reindex(table.target.index, method="ffill")
    return (x >= STEADY_T11_MIN_C) & (change <= STEADY_T11_MAX_CHANGE_2H_C)


def prior_lims(lims_long: pd.DataFrame, param: str, times: pd.DatetimeIndex) -> pd.Series:
    """Последнее значение ЛИМС (точка GODT_POINT2, показатель param),
    измеренное СТРОГО раньше каждого момента times."""
    s = (lims_long[(lims_long["point_label"] == GODT_POINT2) & (lims_long["param"] == param)]
         .sort_values("measured_at")[["measured_at", "value"]])
    left = pd.DataFrame({"measured_at": times}).sort_values("measured_at")
    merged = pd.merge_asof(left, s, on="measured_at", direction="backward", allow_exact_matches=False)
    return pd.Series(merged["value"].to_numpy(dtype=float), index=merged["measured_at"]).reindex(times)


def _ht(features: pd.DataFrame) -> dict[str, pd.Series]:
    return {c.split(":", 1)[1]: features[c].astype(float) for c in features.columns if c.startswith("242000:")}


def compute_anchor(metric: str, table: TrainingTable, lims_long: pd.DataFrame) -> pd.Series:
    x = table.features
    times = table.target.index
    if metric == "sulfur_mg_kg":
        q = x["242000:Q21"].astype(float)
        return q.where(q.between(*Q21_VALID_RANGE))
    ht = _ht(x)
    if metric == "t95_c":
        return vak.ht_godt_t95(ht, {"95%.T": prior_lims(lims_long, "95%.T", times)})
    if metric == "cfpp_c":
        return vak.ht_godt_cfpp(ht)
    if metric == "density_kg_m3":
        return vak.ht_godt_d15(ht, {"24-2000.Pipeline.D15": prior_lims(lims_long, "D15", times)})
    if metric == "cetane_number":
        d = prior_lims(lims_long, "D15", times) / 1000.0
        b = prior_lims(lims_long, "50%.T", times).where(lambda v: v > 0)
        return (454.74 - 1641.416 * d + 774.74 * d ** 2 - 0.554 * b + 97.803 * np.log10(b) ** 2)
    raise ValueError(f"Нет якоря для {metric}")


def rolling_bias(y: pd.Series, anchor: pd.Series, k: int) -> pd.Series:
    """Скользящая медиана (y - якорь) по K предыдущим замерам (строго до
    текущего). Пока истории меньше 3 валидных точек -- 0."""
    res = (y - anchor).sort_index()
    return res.shift(1).rolling(k, min_periods=3).median().fillna(0.0).reindex(y.index)


@dataclass
class CandidatePrediction:
    name: str
    pred: pd.Series  # NaN там, где кандидат неприменим (например, якорь невалиден)


def candidate_predictions(
    metric: str,
    table: TrainingTable,
    lims_long: pd.DataFrame,
    fit_mask: pd.Series,
    ks: tuple[int, ...] = (5, 10, 20),
    residual_k: int = 10,
) -> list[CandidatePrediction]:
    """Все кандидаты для одного показателя. fit_mask -- наблюдения,
    разрешённые для подгонки (обучающая часть и стационарный режим).
    Никакой кандидат не использует y за пределами fit_mask, кроме
    поправки смещения, которая по построению берёт только строго прошлые
    лабораторные замеры -- ровно то, что доступно в эксплуатации."""
    y = table.target.astype(float)
    anchor = compute_anchor(metric, table, lims_long)
    train_median = float(y[fit_mask].median())

    out = [
        CandidatePrediction("median", pd.Series(train_median, index=y.index)),
        CandidatePrediction("last_lims", y.sort_index().shift(1).reindex(y.index)),
        CandidatePrediction("anchor", anchor),
    ]
    biases = {}
    for k in ks:
        biases[k] = rolling_bias(y, anchor, k)
        out.append(CandidatePrediction(f"anchor+bias_k{k}", anchor + biases[k]))

    base = anchor + biases[residual_k]
    cols = [c for c in RESIDUAL_FEATURES if c in table.features.columns]
    feats = table.features[cols].astype(float).rename(columns=lambda c: c.replace(":", "__"))
    feats["anchor"] = anchor
    feats["bias"] = biases[residual_k]
    resid = y - base
    ok = fit_mask & resid.notna()
    if ok.sum() >= 50:
        order = resid[ok].sort_index().index
        cut = order[int(len(order) * 0.8)]
        inner_tr = ok & (y.index < cut)
        inner_va = ok & (y.index >= cut)
        model = lgb.LGBMRegressor(**RESIDUAL_GBM_PARAMS)
        model.fit(feats[inner_tr], resid[inner_tr], eval_set=[(feats[inner_va], resid[inner_va])],
                  eval_metric="l1", callbacks=[lgb.early_stopping(40, verbose=False)])
        corr = pd.Series(model.predict(feats), index=y.index)
        out.append(CandidatePrediction(f"anchor+bias_k{residual_k}+gbm_resid", base + corr))
    return out
