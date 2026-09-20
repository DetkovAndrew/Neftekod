"""
Soft-sensor'ы в эксплуатации (ARCHITECTURE.md §5.2): оценка показателя на
момент decision_at по модели, отобранной scripts/benchmark_anchored.py
(config/soft_sensor_selection.yaml).

Эксплуатационная формула та же, что и в отборе (ml/anchored.py), без
каких-либо отличий в данных:

    оценка(t) = якорь(t) + median_{последние K замеров ЛИМС с measured_at <= t}(y_i - якорь_i)

- якорь(t) -- из текущего снимка ProcessState (анализатор Q21 или ВАК-формула
  от текущего КИП и последнего известного ЛИМС);
- остатки y_i - якорь_i посчитаны заранее ровно той же функцией
  compute_anchor, что и в бенчмарке, на истории до decision_at (история
  из будущего отсекается фильтром по времени при каждом вызове).

Если по бенчмарку лучшей оказалась "last_lims" (цетановое число, CFPP),
soft-sensor не нужен -- сервис возвращает None, а Агент качества берёт ЛИМС,
приписывая ему измеренную ошибку прогноза "последним значением".

Вне стационарного режима реактора сервис не оценивает (область применимости).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from neftekod_mas.ml.anchored import Q21_VALID_RANGE, compute_anchor
from neftekod_mas.ml.dataset import TARGET_METRICS, build_feature_frame, build_training_table
from neftekod_mas.quality import vak_formulas as vak
from neftekod_mas.quality.quality_agent import GODT_POINT2
from neftekod_mas.schemas import ProcessState

MIN_BIAS_HISTORY = 3  # как rolling(min_periods=3) в ml/anchored.rolling_bias


@dataclass(frozen=True)
class SoftSensorEstimate:
    metric: str
    value: float
    anchor: float
    bias: float
    n_bias_samples: int
    last_lims_at: datetime | None  # самый свежий замер ЛИМС, вошедший в поправку
    typical_error: float  # средний MAE forward-chaining CV (не тест)
    model: str


def _parse_k(model: str) -> int | None:
    if not model.startswith("anchor+bias_k") or "gbm" in model:
        return None
    return int(model.removeprefix("anchor+bias_k"))


class SoftSensorService:
    def __init__(self, selection: dict, residuals: dict[str, pd.Series]):
        self.selection = {k: v for k, v in selection.items() if k != "_meta"}
        # metric -> Series(y - якорь) по времени замера ЛИМС, отсортирована
        self.residuals = {m: r.sort_index() for m, r in residuals.items()}

    @classmethod
    def from_history(
        cls,
        selection: dict,
        avt_kip: pd.DataFrame,
        ht_kip: pd.DataFrame,
        lims_long: pd.DataFrame,
    ) -> "SoftSensorService":
        features = build_feature_frame(avt_kip, ht_kip)
        residuals: dict[str, pd.Series] = {}
        for metric, cfg in selection.items():
            if metric == "_meta" or metric not in TARGET_METRICS or _parse_k(cfg.get("selected", "")) is None:
                continue
            table = build_training_table(metric, lims_long, features)
            anchor = compute_anchor(metric, table, lims_long)
            residuals[metric] = (table.target.astype(float) - anchor).rename(metric)
        return cls(selection, residuals)

    def available_metrics(self) -> set[str]:
        return set(self.residuals)

    def lims_typical_error(self, metric: str) -> float | None:
        """Если отобрана "last_lims" -- её CV MAE и есть типичная ошибка ЛИМС как прогноза."""
        cfg = self.selection.get(metric)
        if cfg and cfg.get("selected") == "last_lims":
            return float(cfg["cv_mean_mae"])
        return None

    def estimate(self, metric: str, state: ProcessState) -> SoftSensorEstimate | None:
        cfg = self.selection.get(metric)
        if cfg is None or metric not in self.residuals:
            return None
        if state.steady_regime is False:
            return None
        k = _parse_k(cfg["selected"])
        anchor = self._anchor_now(metric, state)
        if anchor is None or not np.isfinite(anchor):
            return None

        res = self.residuals[metric]
        # Строго: только замеры, известные на decision_at. Окно -- последние K
        # строк (включая те, где якорь был невалиден), как rolling(k) в бенчмарке.
        past = res[res.index <= pd.Timestamp(state.decision_at)].iloc[-k:]
        valid = past.dropna()
        bias = float(valid.median()) if len(valid) >= MIN_BIAS_HISTORY else 0.0
        return SoftSensorEstimate(
            metric=metric,
            value=anchor + bias,
            anchor=anchor,
            bias=bias,
            n_bias_samples=int(len(valid)),
            last_lims_at=valid.index[-1].to_pydatetime() if len(valid) else None,
            typical_error=float(cfg["cv_mean_mae"]),
            model=cfg["selected"],
        )

    @staticmethod
    def _anchor_now(metric: str, state: ProcessState) -> float | None:
        ht = {qid.split(":", 1)[1]: r.value for qid, r in state.kip.items() if qid.startswith("242000:")}
        try:
            if metric == "sulfur_mg_kg":
                q = ht["Q21"]
                return q if Q21_VALID_RANGE[0] <= q <= Q21_VALID_RANGE[1] else None
            if metric == "t95_c":
                lims = state.lab_points.get(f"{GODT_POINT2}|95%.T")
                return None if lims is None else float(vak.ht_godt_t95(ht, {"95%.T": lims.value}))
            if metric == "density_kg_m3":
                lims = state.lab_points.get(f"{GODT_POINT2}|D15")
                return None if lims is None else float(vak.ht_godt_d15(ht, {"24-2000.Pipeline.D15": lims.value}))
            if metric == "cfpp_c":
                return float(vak.ht_godt_cfpp(ht))
        except KeyError:
            return None
        return None
