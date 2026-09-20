"""
Режим реактора гидроочистки: стационарный или переходный (пуск/останов,
разгон температуры). Область применимости soft-sensor'ов и всей логики
рекомендаций -- стационарный режим (ARCHITECTURE.md §12.1, MODEL_AUDIT.md):
во время пусков качество определяется регламентом пуска, а не оптимизацией,
и лабораторные значения там на порядки выходят за рабочий диапазон
(сера 2120 мг/кг при разгоне 2024-04-23).

Критерий используется одинаково в отборе моделей (ml/anchored.steady_state_mask)
и в эксплуатации (data/sync.build_process_state), чтобы модель никогда не
применялась вне той области, на которой её оценивали.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

STEADY_T11_MIN_C = 340.0
STEADY_T11_MAX_CHANGE_2H_C = 10.0
REGIME_TAG = "T11"  # 242000:T11 -- температура на входе в реактор
REGIME_WINDOW = pd.Timedelta(hours=2)


@dataclass(frozen=True)
class RegimeStatus:
    steady: bool | None  # None -- нет данных T11 для вывода
    t11_c: float | None
    change_2h_c: float | None

    @property
    def detail(self) -> str:
        if self.steady is None:
            return f"нет данных {REGIME_TAG} для определения режима"
        change = "н/д" if self.change_2h_c is None else f"{self.change_2h_c:.1f}°C"
        return (
            f"T11={self.t11_c:.1f}°C (порог >= {STEADY_T11_MIN_C:.0f}°C), изменение за 2 ч {change} "
            f"(порог <= {STEADY_T11_MAX_CHANGE_2H_C:.0f}°C)"
        )


def regime_at(ht_kip: pd.DataFrame, decision_at: datetime, tolerance: pd.Timedelta = pd.Timedelta(minutes=15)) -> RegimeStatus:
    """Режим на момент decision_at -- только по текущему и прошлым значениям T11."""
    if REGIME_TAG not in ht_kip.columns:
        return RegimeStatus(None, None, None)
    t11 = ht_kip[REGIME_TAG].dropna()
    t = pd.Timestamp(decision_at)
    pos = t11.index.searchsorted(t, side="right") - 1
    if pos < 0 or t - t11.index[pos] > tolerance:
        return RegimeStatus(None, None, None)
    now = float(t11.iloc[pos])
    past_pos = t11.index.searchsorted(t11.index[pos] - REGIME_WINDOW, side="right") - 1
    change = None
    if past_pos >= 0 and (t11.index[pos] - REGIME_WINDOW) - t11.index[past_pos] <= tolerance:
        change = abs(now - float(t11.iloc[past_pos]))
    steady = now >= STEADY_T11_MIN_C and (change is None or change <= STEADY_T11_MAX_CHANGE_2H_C)
    return RegimeStatus(steady, now, change)
