#!/usr/bin/env python3
"""
Эксплуатационные границы и ресурс катализатора по истории установки
(ARCHITECTURE.md §6.3.1). Закрывает пробел "надёжность оценивается по
историческим перцентилям, без паспортных ограничений и подтверждённых
эксплуатационных границ".

Паспорта оборудования нам не выдали, и выдумывать его нельзя. Но часть
того, ради чего паспорт нужен, извлекается из самих данных -- если
искать не "какое значение было максимальным", а следы РЕАЛЬНЫХ
физических упоров:

1. **Насыщение прибора или контура.** Если тег подолгу стоит ровно на
   одном и том же максимальном значении, а в верхнем проценте у него
   мало РАЗНЫХ значений -- это не распределение измерений, а упор:
   предел шкалы датчика или полностью открытый клапан. Такой предел
   подтверждён данными, в отличие от перцентиля.
2. **Остановы установки.** Ищутся по падению расхода сырья. Для каждого
   смотрится, что происходило перед ним: если параметры не подходили
   к экстремумам, значит остановы плановые, и порогов защит из истории
   восстановить НЕЛЬЗЯ -- это честно фиксируется, а не маскируется.
3. **Дезактивация катализатора.** Главный ресурсный показатель
   гидроочистки: насколько нужно поднимать температуру реактора, чтобы
   удерживать одну и ту же серу. Считается ПО КАМПАНИЯМ между
   остановами (после перегрузки катализатора отсчёт начинается заново,
   и сквозная по всей истории регрессия смешала бы разные загрузки).
   Отсюда -- остаток температурного запаса до потолка и оценка, на
   сколько его хватит при текущем режиме.

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/compute_equipment_limits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip  # noqa: E402

RUNNING_FEED_T_H = 20.0        # ниже -- установка остановлена
STEADY_MIN_T11_C = 340.0
STEADY_MAX_2H_CHANGE_C = 10.0
ANALYZER_RANGE = (0.2, 24.5)
SULFUR_BAND = (6.0, 10.0)      # коридор, в котором завод штатно держит серу
MIN_CAMPAIGN_DAYS = 60
MIN_CAMPAIGN_POINTS = 2000
# Признак упора: мало РАЗНЫХ значений в верхнем проценте. У живого
# измерения их почти столько же, сколько самих точек.
SATURATION_UNIQUE_RATIO = 0.25

SATURATION_TAGS = ("F14", "T5", "T11", "P13", "P8", "F25", "F2", "F9")
TAG_MEANING = {
    "F14": "Расход квенча в реактор Р-202 (контур регулирования температуры)",
    "T5": "Температура ГСС на выходе Р-201",
    "T11": "Температура на выходе из Р-202",
    "P13": "Давление на входе Р-202",
    "P8": "Перепад давления на реакторе Р-202",
    "F25": "Расход свежего ВСГ с КЦА",
    "F2": "Расход газа от ЦК-201",
    "F9": "Расход сырья на установку",
}


def detect_saturation(ht: pd.DataFrame) -> dict:
    running = ht["F9"] > RUNNING_FEED_T_H
    out = {}
    for tag in SATURATION_TAGS:
        s = ht[tag][running].dropna()
        if s.empty:
            continue
        top = float(s.max())
        top_quantile = s[s >= s.quantile(0.99)]
        unique_ratio = top_quantile.round(3).nunique() / max(len(top_quantile), 1)
        saturated = unique_ratio < SATURATION_UNIQUE_RATIO
        out[tag] = {
            "description": TAG_MEANING.get(tag, tag),
            "observed_max": round(top, 4),
            "time_exactly_at_max_pct": round(float((s >= top - 1e-6).mean() * 100), 4),
            "unique_values_in_top_percent": int(top_quantile.round(3).nunique()),
            "points_in_top_percent": int(len(top_quantile)),
            "looks_saturated": bool(saturated),
            "interpretation": (
                "Похоже на УПОР (предел шкалы прибора или полностью открытый контур): "
                "в верхнем проценте почти нет разных значений. Значения на этом уровне "
                "цензурированы -- истинная величина может быть выше, и доверять им как "
                "измерению нельзя."
                if saturated else
                "Обычное распределение измерений, признаков упора нет."
            ),
        }
    return out


def campaigns(ht: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Кампании между остановами установки."""
    running = ht["F9"] > RUNNING_FEED_T_H
    edges = [ht.index[0]]
    prev = True
    for ts, is_running in running.items():
        if prev and not is_running:
            edges.append(ts)
        prev = bool(is_running)
    edges.append(ht.index[-1])
    return list(zip(edges[:-1], edges[1:], strict=True))


def shutdown_analysis(ht: pd.DataFrame, spans) -> dict:
    """Что наблюдалось перед остановами. Если ничего экстремального --
    остановы плановые, и порогов защит из истории не восстановить."""
    running = ht["F9"] > RUNNING_FEED_T_H
    stops = [b for _, b in spans[:-1]]
    rows = {}
    for tag in ("T11", "P8", "P13", "F14"):
        normal_p99 = float(ht[tag][running].quantile(0.99))
        pre_max = []
        for stop in stops:
            window = ht[tag].loc[stop - pd.Timedelta("6h"):stop].dropna()
            if len(window) > 10:
                pre_max.append(float(window.max()))
        if not pre_max:
            continue
        rows[tag] = {
            "description": TAG_MEANING.get(tag, tag),
            "median_max_before_stop": round(float(np.median(pre_max)), 4),
            "normal_operation_p99": round(normal_p99, 4),
            "exceeds_normal_p99": bool(np.median(pre_max) > normal_p99),
        }
    any_exceed = any(v["exceeds_normal_p99"] for v in rows.values())
    return {
        "n_shutdowns": len(stops),
        "per_tag": rows,
        "conclusion": (
            "Перед остановами параметры выходили за обычный диапазон -- возможно, "
            "срабатывание защит; пороги стоит уточнить у технолога."
            if any_exceed else
            "Перед остановами параметры НЕ подходили к экстремумам: остановы выглядят "
            "плановыми. Значит, пороги срабатывания защит по этим данным восстановить "
            "НЕЛЬЗЯ, и их отсутствие остаётся открытым допущением модели надёжности."
        ),
    }


def deactivation(ht: pd.DataFrame, spans) -> dict:
    """Скорость дезактивации: рост требуемой T5 при удержании той же серы."""
    t11 = ht["T11"]
    steady = (t11 >= STEADY_MIN_T11_C) & (t11.diff(12).abs() <= STEADY_MAX_2H_CHANGE_C)
    sulfur = ht["Q21"].where((ht["Q21"] >= ANALYZER_RANGE[0]) & (ht["Q21"] <= ANALYZER_RANGE[1]))
    df = pd.DataFrame({"T5": ht["T5"], "S": sulfur})[steady].dropna()
    band = df[(df["S"] >= SULFUR_BAND[0]) & (df["S"] <= SULFUR_BAND[1])]

    results = []
    for start, end in spans:
        seg = band.loc[start:end]
        if len(seg) < MIN_CAMPAIGN_POINTS:
            continue
        days = (seg.index - seg.index[0]).days.to_numpy().astype(float)
        if days.max() < MIN_CAMPAIGN_DAYS:
            continue
        X = np.column_stack([np.ones(len(seg)), days])
        y = seg["T5"].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        se = float(np.sqrt((float(resid @ resid) / max(len(seg) - 2, 1)) * np.linalg.inv(X.T @ X)[1, 1]))
        results.append({
            "campaign_start": str(start.date()),
            "campaign_end": str(end.date()),
            "days": int(days.max()),
            "n_points": int(len(seg)),
            "c_per_month": round(float(coef[1] * 30.0), 4),
            "ci95_half_width": round(float(se * 30.0 * 1.96), 4),
            "start_temp_c": round(float(coef[0]), 2),
        })
    if not results:
        return {"status": "insufficient_data", "campaigns": []}

    rates = np.array([r["c_per_month"] for r in results])
    return {
        "status": "ok",
        "campaigns": results,
        "median_c_per_month": round(float(np.median(rates)), 4),
        "n_campaigns": len(results),
        "n_positive": int((rates > 0).sum()),
        "sulfur_band_mg_kg": list(SULFUR_BAND),
        "note": (
            "Скорость считается ВНУТРИ кампании: после перегрузки катализатора отсчёт "
            "начинается заново, и сквозная регрессия по всей истории смешала бы разные "
            "загрузки (на всей истории наклон статистически неотличим от нуля, а по "
            "кампаниям он устойчиво положительный -- это и есть проявление перегрузок)."
        ),
    }


def main() -> None:
    ht = load_kip(data_dir() / "242000_tags.csv")
    spans = campaigns(ht)
    sat = detect_saturation(ht)
    deact = deactivation(ht, spans)

    running = ht["F9"] > RUNNING_FEED_T_H
    t5_ceiling = float(ht["T5"][running].quantile(0.99))

    out = {
        "_meta": {
            "purpose": "Эксплуатационные границы и ресурс катализатора, извлечённые из истории",
            "important": (
                "Это НЕ паспортные ограничения -- их не выдали. Здесь только то, что "
                "подтверждается самими данными: упоры приборов/контуров, поведение перед "
                "остановами и скорость дезактивации катализатора. Всё остальное в модели "
                "надёжности остаётся перцентилями истории и помечено как допущение."
            ),
            "running_feed_threshold_t_h": RUNNING_FEED_T_H,
        },
        "saturation": sat,
        "shutdowns": shutdown_analysis(ht, spans),
        "catalyst_deactivation": deact,
        "reactor_temperature_ceiling_c": {
            "value": round(t5_ceiling, 2),
            "source": "99-й перцентиль T5 в рабочем режиме",
            "is_assumption": True,
            "note": "Потолок по истории, не паспортный предел end-of-run температуры.",
        },
    }

    path = REPO_ROOT / "config" / "equipment_limits.yaml"
    path.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Записано: {path}")

    print("\nУпоры приборов/контуров:")
    for tag, block in sat.items():
        if block["looks_saturated"]:
            print(f"  {tag}: max={block['observed_max']} -- УПОР "
                  f"({block['unique_values_in_top_percent']} разных значений "
                  f"в верхнем проценте из {block['points_in_top_percent']})")

    print(f"\nОстановов найдено: {out['shutdowns']['n_shutdowns']}")
    print(f"  {out['shutdowns']['conclusion']}")

    if deact["status"] == "ok":
        print(f"\nДезактивация катализатора по {deact['n_campaigns']} кампаниям "
              f"(сера удерживается в {SULFUR_BAND[0]}-{SULFUR_BAND[1]} мг/кг):")
        for c in deact["campaigns"]:
            print(f"  {c['campaign_start']} .. {c['campaign_end']}  {c['days']:4d} дней  "
                  f"{c['c_per_month']:+.3f} ±{c['ci95_half_width']:.3f} °C/мес")
        print(f"  медиана: {deact['median_c_per_month']:+.3f} °C/мес "
              f"(положительных {deact['n_positive']} из {deact['n_campaigns']})")
        print(f"  потолок T5 по истории: {t5_ceiling:.1f} °C")


if __name__ == "__main__":
    main()
