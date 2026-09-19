"""
Загрузка сырых источников хакатона в единообразный вид.

Ничего здесь не "обучается" -- это чистый ETL (ARCHITECTURE.md §2).
Файлы читаются из локального каталога (см. `NEFTEKOD_DATA_DIR`),
который НЕ входит в репозиторий.

Форматы источников (см. ARCHITECTURE.md §1 для описания каждого):

- avt_tags.csv / 242000_tags.csv: синхронный 10-минутный ряд, колонки
  `Unnamed: 0[.1]` -- служебные индексы (отбрасываются), `date` --
  временной ключ, остальное -- коды тегов КИП.

- ЛИМСы...xlsx: широкая разреженная таблица. Каждая пара колонок
  (дата, значение) -- один лабораторный показатель одной точки отбора.
  Строка 0 -- метка "Установка/Точка отбора/Продукт" (задана только в
  первой колонке пары начала каждой группы точек -- остальные ячейки
  группы пустые, требуют forward-fill). Строка 1 -- имя показателя
  (задано в каждой паре). Строка 2 -- единица измерения (в каждой
  паре). Строка 3 -- служебный счётчик количества значений. Данные --
  с 4-й строки, ряды внутри пары независимы по длине (хвост дополнен
  пустыми ячейками).

- Выгрузка ПАК...xlsx: тот же принцип пар (дата, значение), но между
  парами есть НАСТОЯЩИЕ пустые колонки-разделители (а не только внутри
  одной группы) -- значит здесь имя серии дано в каждой паре без
  forward-fill, но позиции пар не выровнены по чётности индекса.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return Path(os.environ.get("NEFTEKOD_DATA_DIR", repo_root / ".." / "neftekod")).resolve()


# ---------------------------------------------------------------------------
# КИП-телеметрия (10-минутные ряды)
# ---------------------------------------------------------------------------


# Служебный код историана "нет данных / плохое качество". Точное значение
# 307.0 встречается в 64 из 71 тегов АВТ и 15 из 26 тегов 24-2000 --
# одновременно у температур, давлений (МПа), расходов и плотности, что
# физически невозможно как реальное общее значение. Тег D10 (плотность
# нефти) равен 307 в 99.99% истории. Без замены на NaN код попадал в
# признаки моделей, в перцентильные границы и маскировал себя от
# детектора out_of_range (границы считались по тем же испорченным данным).
HISTORIAN_BAD_QUALITY_SENTINELS: tuple[float, ...] = (307.0,)


def load_kip(csv_path: Path, replace_sentinels: bool = True) -> pd.DataFrame:
    """Читает avt_tags.csv / 242000_tags.csv, отбрасывает служебные
    Unnamed-колонки, приводит `date` к datetime-индексу, заменяет
    служебные коды историана на NaN (см. HISTORIAN_BAD_QUALITY_SENTINELS)."""
    df = pd.read_csv(csv_path)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # float32 достаточно для телеметрии КИП и вдвое экономнее по памяти
    # на ~189k строк x ~70 тегов, чем float64 по умолчанию.
    df = df.astype("float32")
    if replace_sentinels:
        for sentinel in HISTORIAN_BAD_QUALITY_SENTINELS:
            df = df.mask((df - sentinel).abs() < 1e-3)
    return df


# ---------------------------------------------------------------------------
# Общий разбор "парных" таблиц ЛИМС/ПАК
# ---------------------------------------------------------------------------


@dataclass
class PairSeries:
    group_label: str | None  # для ЛИМС -- "Установка/точка/продукт", для ПАК -- None
    param_label: str
    unit: str | None
    date_col: int
    value_col: int


def _iter_pairs(
    header_rows: list[tuple],
    marker_row_idx: int,
    unit_row_idx: int | None,
    group_row_idx: int | None,
) -> Iterator[PairSeries]:
    """ВАЖНО: пустая ячейка после pd.read_excel -- это float NaN, а не
    Python None (обычная ловушка openpyxl/pandas: колонка с текстом и
    пропусками читается как object-массив, где пропуск -- всё равно
    numpy.nan). Проверка `cell is not None` пропускает такие ячейки как
    валидные -- поэтому здесь используется `pd.notna()`."""
    marker_row = header_rows[marker_row_idx]
    unit_row = header_rows[unit_row_idx] if unit_row_idx is not None else None
    group_row = header_rows[group_row_idx] if group_row_idx is not None else None

    group_filled: list[str | None] = []
    if group_row is not None:
        last = None
        for cell in group_row:
            if pd.notna(cell):
                last = cell
            group_filled.append(last)

    n_cols = len(marker_row)
    i = 0
    while i < n_cols - 1:
        label = marker_row[i]
        if pd.notna(label):
            unit_cell = unit_row[i] if unit_row is not None else None
            yield PairSeries(
                group_label=group_filled[i] if group_row is not None else None,
                param_label=str(label),
                unit=(str(unit_cell) if pd.notna(unit_cell) else None),
                date_col=i,
                value_col=i + 1,
            )
            i += 2
        else:
            i += 1


def load_lims(xlsx_path: Path, sheet_name: str = "Лист1") -> pd.DataFrame:
    """Возвращает длинный (long) формат:
    columns = [point_label, param, unit, measured_at, value]
    Строка 0 = группа (точка отбора, forward-fill), строка 1 = показатель,
    строка 2 = единица, строка 3 = счётчик (пропускается), данные с 4-й строки."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    header_rows = [tuple(raw.iloc[r]) for r in range(4)]
    data = raw.iloc[4:].reset_index(drop=True)

    frames = []
    for pair in _iter_pairs(header_rows, marker_row_idx=1, unit_row_idx=2, group_row_idx=0):
        sub = data.iloc[:, [pair.date_col, pair.value_col]].copy()
        sub.columns = ["measured_at", "value"]
        sub = sub.dropna(how="any")
        if sub.empty:
            continue
        sub["point_label"] = pair.group_label
        sub["param"] = pair.param_label
        sub["unit"] = pair.unit
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["point_label", "param", "unit", "measured_at", "value"])

    out = pd.concat(frames, ignore_index=True)
    out["measured_at"] = pd.to_datetime(out["measured_at"], errors="coerce")
    out = out.dropna(subset=["measured_at"])
    return out[["point_label", "param", "unit", "measured_at", "value"]].sort_values("measured_at")


def load_pak(xlsx_path: Path, sheet_name: str = "Лист1") -> pd.DataFrame:
    """Возвращает длинный формат: columns = [param, unit, measured_at, value].
    Строка 0 = имя серии (без forward-fill, между парами есть настоящие
    пустые колонки-разделители), строка 1 = единица, данные с 2-й строки."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    header_rows = [tuple(raw.iloc[r]) for r in range(2)]
    data = raw.iloc[2:].reset_index(drop=True)

    frames = []
    for pair in _iter_pairs(header_rows, marker_row_idx=0, unit_row_idx=1, group_row_idx=None):
        sub = data.iloc[:, [pair.date_col, pair.value_col]].copy()
        sub.columns = ["measured_at", "value"]
        sub = sub.dropna(how="any")
        if sub.empty:
            continue
        sub["param"] = pair.param_label
        sub["unit"] = pair.unit
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["param", "unit", "measured_at", "value"])

    out = pd.concat(frames, ignore_index=True)
    out["measured_at"] = pd.to_datetime(out["measured_at"], errors="coerce")
    out = out.dropna(subset=["measured_at"])
    return out[["param", "unit", "measured_at", "value"]].sort_values("measured_at")
