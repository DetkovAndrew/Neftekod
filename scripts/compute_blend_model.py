#!/usr/bin/env python3
"""
Исторический бэктест модели блендинга дизельного пула АВТ
(ARCHITECTURE.md §6.6). Закрывает пробел "ограничение суммы долей есть,
но нет исторически проверенной модели компонентов и влияния блендинга
на качество".

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ
---------------------
Дизельный пул установки АВТ собирается из двух боковых погонов:

    фр. 240-290  -- расход `avt:F32`, качество -- ЛИМС точка отбора '2'
    фр. 290-350  -- расход `avt:F30`, качество -- ЛИМС точка отбора '2.1'

Смесь -- это то, что уходит с установки как дизельная фракция и
приходит на гидроочистку: ЛИМС точка отбора '3' (контроль) и
'Гидроочистка т.1' (перекрёстная проверка).

Соответствие "точка 2 = лёгкий погон, точка 2.1 = тяжёлый погон, точка
3 = их смесь" не постулировано, а установлено ПО ДАННЫМ: медианы
показателей точки 3 лежат строго между точками 2 и 2.1 (T50: 250 /
280 / 304 °C, плотность: 832 / 845 / 857 кг/м3), а доля, при которой
линейное смешение по объёму воспроизводит плотность смеси, совпадает
с фактическим отношением расходов F30/(F30+F32). Независимое
подтверждение из выданных материалов: производственная ВАК-формула
`AVT6:240-350:D15` в `формулы_ВАК.xlsx` содержит ровно выражение
`F30/(F32+F30)` -- то есть сам завод описывает эту фракцию как смесь
этих двух потоков в этих же долях.

МЕТОДОЛОГИЯ
-----------
* Ничего не обучается сверх ОДНОЙ константы смещения на показатель.
  Правила смешения (`blending/rules.py`) -- физика и литература.
* Смещение оценивается ТОЛЬКО по прошлым блокам (forward chaining по 4
  последовательным блокам времени), проверяется на следующем блоке.
  Утечки из будущего нет -- тест `test_blend_bias_uses_only_past`.
* Опорная модель для сравнения -- константа (медиана прошлых замеров
  смеси). Правило принимается в модель, только если оно устойчиво
  лучше константы вне выборки.
* Сера компонентов не измеряется лабораторией напрямую. Она
  ИДЕНТИФИЦИРУЕТСЯ по 132 замерам серы сырья гидроочистки при
  естественной вариации долей (0.16-0.71) как решение линейной задачи
  наименьших квадратов S_смеси = w240*S240 + w290*S290. Результат
  сохраняется вместе с доверительным интервалом и честной пометкой,
  что точечный прогноз серы сырья при этом не улучшается -- ценность
  в физически состоятельной ЧУВСТВИТЕЛЬНОСТИ к долям.

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/compute_blend_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.blending.rules import (  # noqa: E402
    CLOUD_POINT_INDEX_EXPONENT,
    Component,
    blend_cloud_point,
    blend_density,
    blend_distillation,
)
from neftekod_mas.data.loaders import data_dir, load_kip, load_lims  # noqa: E402

LIGHT_POINT = "Установка 'АВТ'. Точка отбора '2'. Продукт 'Дизельное топливо'"
HEAVY_POINT = "Установка 'АВТ'. Точка отбора '2.1'. Продукт 'Дизельное топливо'"
BLEND_POINT = "Установка 'АВТ'. Точка отбора '3'. Продукт 'Дизельное топливо'"
HT_FEED_POINT = "Установка 'Гидроочистка'. Точка отбора '1'. Продукт 'ФРАКЦ_ДИЗ'."
GODT_PRODUCT_POINT = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"

LIGHT_FLOW, HEAVY_FLOW = "F32", "F30"

# Физически допустимые окна для точек разгонки дизельных фракций.
# Вне их -- брак ввода/не тот продукт; такие значения выбрасываются до
# всякого счёта (в ЛИМС точки 2.1 по 95%.T std = 91 °C из-за единичных
# выбросов -- без чистки они определяют весь результат).
PHYSICAL_RANGES = {
    "IBP.T": (120.0, 260.0),
    "50%.T": (200.0, 340.0),
    "90%.T": (250.0, 380.0),
    "95%.T": (260.0, 400.0),
    "EBP.T": (280.0, 420.0),
    "D15": (780.0, 920.0),
    "CloudPoint": (-40.0, 20.0),
}

CURVE_POINTS = ("IBP.T", "50%.T", "90%.T", "95%.T", "EBP.T")
FLOW_WINDOW = pd.Timedelta("2h")  # проба представляет режим за предшествующие 2 ч
LIMS_TOLERANCE = pd.Timedelta("6h")  # компоненты и смесь отбираются в одну смену
KIP_TOLERANCE = pd.Timedelta("1h")
N_BLOCKS = 4  # forward chaining: блок k предсказывается по блокам < k
MIN_BLOCK_OBS = 8


def series(lims: pd.DataFrame, point: str, param: str) -> pd.Series:
    sub = lims[(lims.point_label == point) & (lims.param == param)]
    s = sub.dropna(subset=["value"]).groupby("measured_at")["value"].mean().sort_index()
    lo, hi = PHYSICAL_RANGES.get(param, (-np.inf, np.inf))
    return s[(s >= lo) & (s <= hi)]


def component_frame(lims: pd.DataFrame, point: str, params) -> pd.DataFrame:
    cols = {p: series(lims, point, p) for p in params}
    return pd.concat(cols, axis=1, sort=True)


def shares(avt: pd.DataFrame) -> pd.DataFrame:
    """Средние расходы боковых погонов за окно до отбора пробы.
    Значения <= 1 т/ч отбрасываются: это остановленный поток или брак КИП
    (в сырых данных встречаются отрицательные расходы)."""
    flows = avt[[LIGHT_FLOW, HEAVY_FLOW]]
    return flows.where(flows > 1.0).rolling(FLOW_WINDOW).mean()


def join_asof(left: pd.DataFrame, right: pd.DataFrame, tol: pd.Timedelta) -> pd.DataFrame:
    return pd.merge_asof(
        left.sort_values("measured_at"),
        right.sort_values("measured_at"),
        on="measured_at",
        tolerance=tol,
        direction="nearest",
    )


def build_dataset(lims: pd.DataFrame, avt: pd.DataFrame) -> pd.DataFrame:
    params = (*CURVE_POINTS, "D15", "CloudPoint")
    light = component_frame(lims, LIGHT_POINT, params).add_prefix("l_")
    heavy = component_frame(lims, HEAVY_POINT, params).add_prefix("h_")
    blend = component_frame(lims, BLEND_POINT, params).add_prefix("b_")
    flow = shares(avt)

    df = blend.reset_index(names="measured_at")
    df = join_asof(df, light.reset_index(names="measured_at"), LIMS_TOLERANCE)
    df = join_asof(df, heavy.reset_index(names="measured_at"), LIMS_TOLERANCE)
    df = join_asof(df, flow.reset_index(names="measured_at"), KIP_TOLERANCE)
    return df.dropna(subset=[LIGHT_FLOW, HEAVY_FLOW]).sort_values("measured_at").reset_index(drop=True)


def components_at(row, rho_light: float, rho_heavy: float) -> list[Component]:
    """Доли -- по фактическим расходам. Плотность компонента берётся из
    сопоставленного замера ЛИМС, если он есть; иначе -- медиана по истории
    (D15 компонентов измеряется заметно реже разгонки: ~120 замеров против ~200)."""
    total = row[LIGHT_FLOW] + row[HEAVY_FLOW]
    rl = float(row.l_D15) if np.isfinite(row.l_D15) else rho_light
    rh = float(row.h_D15) if np.isfinite(row.h_D15) else rho_heavy
    return [
        Component("avt_fr_240_290", float(row[LIGHT_FLOW] / total), rl),
        Component("avt_fr_290_350", float(row[HEAVY_FLOW] / total), rh),
    ]


def predict_row(row, rho_light: float, rho_heavy: float, prop: str):
    comps = components_at(row, rho_light, rho_heavy)
    if prop == "D15":
        return blend_density(comps)
    if prop == "CloudPoint":
        if not np.isfinite([row.l_CloudPoint, row.h_CloudPoint]).all():
            return np.nan
        return blend_cloud_point(comps, [float(row.l_CloudPoint), float(row.h_CloudPoint)])
    curves = []
    for pref in ("l_", "h_"):
        curve = {p: row[pref + p] for p in CURVE_POINTS}
        if not np.isfinite(list(curve.values())).all():
            return np.nan
        curves.append(curve)
    target = {"50%.T": 50.0, "90%.T": 90.0, "95%.T": 95.0}[prop]
    return blend_distillation(comps, curves, targets=(target,))[target]


def forward_chaining(df: pd.DataFrame, prop: str, rho_light: float, rho_heavy: float) -> dict:
    """Оценка вне выборки: смещение считается по прошлым блокам, ошибка --
    на следующем. Опорная модель -- медиана смеси по тем же прошлым блокам."""
    col = "b_" + prop
    raw = df[["measured_at", col]].copy()
    raw["pred"] = [predict_row(r, rho_light, rho_heavy, prop) for _, r in df.iterrows()]
    raw = raw.dropna()
    if len(raw) < N_BLOCKS * MIN_BLOCK_OBS:
        return {"n": int(len(raw)), "status": "insufficient_data"}

    blocks = np.array_split(np.arange(len(raw)), N_BLOCKS)
    err_model, err_base, biases = [], [], []
    for k in range(1, N_BLOCKS):
        past = raw.iloc[np.concatenate(blocks[:k])]
        cur = raw.iloc[blocks[k]]
        bias = float((past[col] - past["pred"]).median())
        baseline = float(past[col].median())
        biases.append(bias)
        err_model.extend((cur[col] - (cur["pred"] + bias)).abs().tolist())
        err_base.extend((cur[col] - baseline).abs().tolist())

    mae_model, mae_base = float(np.mean(err_model)), float(np.mean(err_base))
    full_bias = float((raw[col] - raw["pred"]).median())
    resid = raw[col] - (raw["pred"] + full_bias)
    return {
        "n": int(len(raw)),
        "n_out_of_sample": int(len(err_model)),
        "bias": round(full_bias, 4),
        "bias_stability": round(float(np.std(biases)), 4),
        "mae_out_of_sample": round(mae_model, 4),
        "mae_baseline_constant": round(mae_base, 4),
        "improvement_vs_constant_pct": round(100.0 * (mae_base - mae_model) / mae_base, 1),
        "residual_std": round(float(resid.std()), 4),
        "accepted": bool(mae_model < mae_base),
        "status": "ok",
    }


def typical_component_curves(lims: pd.DataFrame) -> dict:
    """Типовые (медианные) кривые разгонки и температура помутнения
    компонентов -- агрегированная статистика, а не сырые ряды, поэтому
    её можно держать в конфиге и в git (тот же принцип, что у
    `*_bounds.yaml`, ARCHITECTURE.md §11).

    Нужна затем, что в момент решения свежего ЛИМС по компоненту почти
    никогда нет (по 200 замеров на 3.5 года): агент берёт типовую кривую
    компонента и считает эффект ПЕРЕРАСПРЕДЕЛЕНИЯ долей. Уровень
    показателя при этом всё равно приходит из самой точной оценки
    Агента качества -- модель смешения отвечает только за приращение.
    """
    out = {}
    for key, point in (("avt_fr_240_290", LIGHT_POINT), ("avt_fr_290_350", HEAVY_POINT)):
        block = {}
        for param in (*CURVE_POINTS, "CloudPoint", "D15"):
            s = series(lims, point, param)
            if len(s) >= 10:
                block[param] = round(float(s.median()), 3)
                block[f"{param}__n"] = int(len(s))
        out[key] = block
    return out


def identify_component_sulfur(lims: pd.DataFrame, avt: pd.DataFrame) -> dict:
    """Сера компонентов по вариации долей (МНК на 2 неизвестных).
    Сера в % масс. смешивается по МАССЕ -- поэтому здесь массовые доли."""
    s = series(lims, HT_FEED_POINT, "Mass.Sulfur")
    flow = shares(avt)
    s_df = s.rename("S").rename_axis("measured_at").reset_index()
    df = join_asof(s_df, flow.reset_index(names="measured_at"), KIP_TOLERANCE).dropna()
    if len(df) < 30:
        return {"n": int(len(df)), "status": "insufficient_data"}

    w_heavy = (df[HEAVY_FLOW] / (df[HEAVY_FLOW] + df[LIGHT_FLOW])).to_numpy()
    X = np.column_stack([1.0 - w_heavy, w_heavy])
    coef, *_ = np.linalg.lstsq(X, df["S"].to_numpy(), rcond=None)
    pred = X @ coef
    resid = df["S"].to_numpy() - pred
    dof = len(df) - 2
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "n": int(len(df)),
        "status": "ok",
        "share_range": [round(float(w_heavy.min()), 3), round(float(w_heavy.max()), 3)],
        "avt_fr_240_290_pct_mass": round(float(coef[0]), 4),
        "avt_fr_240_290_ci95": [round(float(coef[0] - 1.96 * se[0]), 4), round(float(coef[0] + 1.96 * se[0]), 4)],
        "avt_fr_290_350_pct_mass": round(float(coef[1]), 4),
        "avt_fr_290_350_ci95": [round(float(coef[1] - 1.96 * se[1]), 4), round(float(coef[1] + 1.96 * se[1]), 4)],
        "ratio_heavy_to_light": round(float(coef[1] / coef[0]), 2),
        "mae": round(float(np.abs(resid).mean()), 4),
        "mae_baseline_constant": round(float(np.abs(df["S"] - df["S"].median()).mean()), 4),
    }


def share_sensitivity(df: pd.DataFrame, prop: str, rho_light: float, rho_heavy: float) -> dict:
    """Чувствительность показателя смеси к доле тяжёлого компонента.

    Для контура управления важна не столько абсолютная точность прогноза,
    сколько ПРОИЗВОДНАЯ: на сколько сдвинется показатель, если изменить
    доли. Здесь она считается двумя независимыми способами и сверяется:

    1. `model` -- численная производная самой модели смешения: доля
       тяжёлого компонента сдвигается на +-1 п.п. при тех же качествах
       компонентов, разница делится на 0.02.
    2. `empirical` -- наклон регрессии фактического значения смеси на
       фактическую долю по истории (МНК, с доверительным интервалом).

    Совпадение знака и порядка величины -- это и есть историческое
    подтверждение того, что рычаг блендинга действительно работает так,
    как его описывает модель.
    """
    col = "b_" + prop
    rows, preds_lo, preds_hi = [], [], []
    for _, r in df.iterrows():
        base = predict_row(r, rho_light, rho_heavy, prop)
        if not np.isfinite(base):
            continue
        total = r[LIGHT_FLOW] + r[HEAVY_FLOW]
        w_heavy = float(r[HEAVY_FLOW] / total)
        if not (0.02 < w_heavy < 0.98):
            continue
        shifted = []
        for dw in (-0.01, +0.01):
            rr = r.copy()
            rr[HEAVY_FLOW] = (w_heavy + dw) * total
            rr[LIGHT_FLOW] = (1.0 - w_heavy - dw) * total
            shifted.append(predict_row(rr, rho_light, rho_heavy, prop))
        if not np.isfinite(shifted).all():
            continue
        preds_lo.append(shifted[0]); preds_hi.append(shifted[1])
        rows.append((w_heavy, float(r[col])))

    if len(rows) < 30:
        return {"n": len(rows), "status": "insufficient_data"}

    d_model = (np.array(preds_hi) - np.array(preds_lo)) / 0.02
    w = np.array([x for x, _ in rows]); y = np.array([v for _, v in rows])
    ok = np.isfinite(y)
    w, y = w[ok], y[ok]
    X = np.column_stack([np.ones_like(w), w])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - 2, 1)
    cov = (float(resid @ resid) / dof) * np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(cov[1, 1]))
    slope = float(coef[1])
    model_slope = float(np.median(d_model))
    ci = [round(slope - 1.96 * se_slope, 3), round(slope + 1.96 * se_slope, 3)]
    return {
        "n": int(len(y)),
        "status": "ok",
        "unit": "единиц показателя на единицу доли (0..1)",
        "model_slope": round(model_slope, 3),
        "empirical_slope": round(slope, 3),
        "empirical_ci95": ci,
        "signs_agree": bool(np.sign(model_slope) == np.sign(slope)),
        "model_slope_within_empirical_ci95": bool(ci[0] <= model_slope <= ci[1]),
        "empirical_slope_significant": bool(ci[0] * ci[1] > 0),
    }


def feed_to_product_propagation(lims: pd.DataFrame) -> dict:
    """Перенос показателя с сырья гидроочистки на товарный продукт.

    Рычаг блендинга меняет качество СЫРЬЯ гидроочистки. Чтобы он стал
    рычагом по нормируемому показателю ПРОДУКТА, нужен подтверждённый
    коэффициент переноса. Здесь он измеряется регрессией продукта
    (ЛИМС точка 'Гидроочистка т.2') на сырьё (точка 'Гидроочистка т.1')
    по сопоставленным пробам.

    Гидроочистка почти не меняет фракционный состав (реакция идёт по
    сере, не по длине цепи), поэтому физически ожидается наклон,
    близкий к 1 -- и это именно то, что показывают данные. Для серы
    такой регрессии НЕ строится: сера как раз и удаляется в реакторе,
    зависимость нелинейна и описывается кинетикой (§6.4), а не
    переносом.
    """
    out = {}
    for param, metric, rng in (("95%.T", "t95_c", (260.0, 400.0)),
                               ("50%.T", "t50_c", (200.0, 340.0)),
                               ("D15", "density_kg_m3", (780.0, 920.0))):
        a = lims[(lims.point_label == HT_FEED_POINT) & (lims.param == param)]
        b = lims[(lims.point_label == GODT_PRODUCT_POINT) & (lims.param == param)]
        fa = a.groupby("measured_at")["value"].mean().sort_index()
        fb = b.groupby("measured_at")["value"].mean().sort_index()
        fa = fa[(fa >= rng[0]) & (fa <= rng[1])]
        fb = fb[(fb >= rng[0]) & (fb <= rng[1])]
        if min(len(fa), len(fb)) < 50:
            out[metric] = {"n": int(min(len(fa), len(fb))), "status": "insufficient_data"}
            continue
        j = join_asof(fb.rename("product").rename_axis("measured_at").reset_index(),
                      fa.rename("feed").rename_axis("measured_at").reset_index(),
                      LIMS_TOLERANCE).dropna()
        y = j["product"].to_numpy()
        X = np.column_stack([np.ones(len(j)), j["feed"].to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = max(len(j) - 2, 1)
        cov = (float(resid @ resid) / dof) * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(cov[1, 1]))
        out[metric] = {
            "n": int(len(j)),
            "status": "ok",
            "intercept": round(float(coef[0]), 4),
            "slope": round(float(coef[1]), 4),
            "slope_ci95": [round(float(coef[1] - 1.96 * se), 4), round(float(coef[1] + 1.96 * se), 4)],
            "pearson_r": round(float(np.corrcoef(j["feed"], y)[0, 1]), 4),
            "mae": round(float(np.abs(resid).mean()), 4),
            "mae_baseline_constant": round(float(np.abs(y - np.median(y)).mean()), 4),
        }
    return out


def cross_check_ht_feed(lims: pd.DataFrame) -> dict:
    """Точка 3 АВТ и точка 1 гидроочистки должны описывать одну струю."""
    out = {}
    for param in ("50%.T", "95%.T", "CloudPoint"):
        a = series(lims, BLEND_POINT, param).rename("avt3").rename_axis("measured_at").reset_index()
        b = series(lims, HT_FEED_POINT, param).rename("ht1").rename_axis("measured_at").reset_index()
        j = join_asof(a, b, pd.Timedelta("2h")).dropna()
        if len(j) > 30:
            out[param] = {"n": int(len(j)),
                          "median_abs_diff": round(float((j.avt3 - j.ht1).abs().median()), 3)}
    return out


def main() -> None:
    d = data_dir()
    lims = load_lims(d / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    avt = load_kip(d / "avt_tags.csv")

    rho_light = float(series(lims, LIGHT_POINT, "D15").median())
    rho_heavy = float(series(lims, HEAVY_POINT, "D15").median())

    df = build_dataset(lims, avt)
    props = {"D15": "density_kg_m3", "50%.T": "t50_c", "90%.T": "t90_c",
             "95%.T": "t95_c", "CloudPoint": "cloud_point_c"}
    validation = {props[p]: forward_chaining(df, p, rho_light, rho_heavy) for p in props}
    sensitivity = {props[p]: share_sensitivity(df, p, rho_light, rho_heavy) for p in props}

    w_heavy = (df[HEAVY_FLOW] / (df[HEAVY_FLOW] + df[LIGHT_FLOW])).dropna()
    model = {
        "_meta": {
            "methodology": (
                "Blend model for the AVT diesel pool. Mixing rules are physics/literature "
                "(volumetric for density, mass for sulfur, volumetric distillation-curve "
                "blending, power blending index for low-temperature properties). Only a "
                "single bias constant per property is fitted, estimated on PAST blocks only "
                "(forward chaining, 4 blocks). Component identity established from data, and "
                "independently confirmed by the plant's own VAK formula AVT6:240-350:D15, "
                "which contains F30/(F32+F30)."
            ),
            "blend_target": BLEND_POINT,
            "cross_check_point": HT_FEED_POINT,
            "lims_tolerance": str(LIMS_TOLERANCE),
            "flow_window": str(FLOW_WINDOW),
            "cloud_point_index_exponent": CLOUD_POINT_INDEX_EXPONENT,
        },
        "components": {
            "avt_fr_240_290": {
                "description": "Боковой погон АВТ, фракция 240-290 °C (лёгкий компонент пула)",
                "flow_tag": f"avt:{LIGHT_FLOW}",
                "flow_unit": "т/ч",
                "lims_point": LIGHT_POINT,
                "density_kg_m3": round(rho_light, 2),
                "density_source": "медиана ЛИМС D15 точки отбора '2'",
            },
            "avt_fr_290_350": {
                "description": "Боковой погон АВТ, фракция 290-350 °C (тяжёлый компонент пула)",
                "flow_tag": f"avt:{HEAVY_FLOW}",
                "flow_unit": "т/ч",
                "lims_point": HEAVY_POINT,
                "density_kg_m3": round(rho_heavy, 2),
                "density_source": "медиана ЛИМС D15 точки отбора '2.1'",
            },
        },
        "observed_shares": {
            "note": "Массовая доля тяжёлого компонента F30/(F30+F32) за всю историю",
            "n": int(len(w_heavy)),
            "p05": round(float(w_heavy.quantile(0.05)), 4),
            "p50": round(float(w_heavy.quantile(0.50)), 4),
            "p95": round(float(w_heavy.quantile(0.95)), 4),
        },
        "component_sulfur": identify_component_sulfur(lims, avt),
        "typical_component_curves": typical_component_curves(lims),
        "validation": validation,
        "share_sensitivity": sensitivity,
        "feed_to_product_propagation": feed_to_product_propagation(lims),
        "cross_check_avt3_vs_ht_feed": cross_check_ht_feed(lims),
        "matched_samples": int(len(df)),
    }

    out = REPO_ROOT / "config" / "blend_model.yaml"
    out.write_text(yaml.safe_dump(model, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Записано: {out}")
    print(f"Сопоставленных проб (компоненты + смесь + расходы): {len(df)}")
    print(f"Плотности компонентов: лёгкий {rho_light:.1f}, тяжёлый {rho_heavy:.1f} кг/м3")
    print("\nВалидация вне выборки (forward chaining):")
    for name, v in validation.items():
        if v.get("status") != "ok":
            print(f"  {name:16s} {v['status']} (n={v['n']})")
            continue
        mark = "принято" if v["accepted"] else "ОТКЛОНЕНО"
        print(f"  {name:16s} n={v['n']:4d}  MAE={v['mae_out_of_sample']:7.3f}  "
              f"константа={v['mae_baseline_constant']:7.3f}  "
              f"({v['improvement_vs_constant_pct']:+.1f}%)  {mark}")
    print("\nЧувствительность к доле тяжёлого компонента (модель против истории):")
    for name, s in sensitivity.items():
        if s.get("status") != "ok":
            print(f"  {name:16s} {s['status']} (n={s['n']})")
            continue
        agree = "совпадает" if s["model_slope_within_empirical_ci95"] else (
            "знак совпадает" if s["signs_agree"] else "РАСХОЖДЕНИЕ")
        print(f"  {name:16s} модель={s['model_slope']:+8.2f}  история={s['empirical_slope']:+8.2f} "
              f"CI95 {s['empirical_ci95']}  {agree}")

    print("\nПеренос показателя сырьё -> продукт (регрессия по ЛИМС):")
    for name, s in model["feed_to_product_propagation"].items():
        if s.get("status") != "ok":
            print(f"  {name:16s} {s['status']} (n={s['n']})"); continue
        print(f"  {name:16s} наклон={s['slope']:.4f} CI95 {s['slope_ci95']}  r={s['pearson_r']:.3f}  "
              f"n={s['n']}  MAE={s['mae']:.3f} (константа {s['mae_baseline_constant']:.3f})")

    cs = model["component_sulfur"]
    if cs.get("status") == "ok":
        print(f"\nСера компонентов (идентификация по {cs['n']} замерам, доли {cs['share_range']}):")
        print(f"  фр.240-290 = {cs['avt_fr_240_290_pct_mass']:.3f} %масс. CI95 {cs['avt_fr_240_290_ci95']}")
        print(f"  фр.290-350 = {cs['avt_fr_290_350_pct_mass']:.3f} %масс. CI95 {cs['avt_fr_290_350_ci95']}")
        print(f"  отношение тяжёлый/лёгкий = {cs['ratio_heavy_to_light']}")


if __name__ == "__main__":
    main()
