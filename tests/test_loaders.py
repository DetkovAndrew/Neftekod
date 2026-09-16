"""
Тесты на data/loaders.py -- модуль, где уже дважды находились реальные
баги на настоящих данных (Unnamed-колонки в CSV, NaN vs None в
_iter_pairs при разборе ЛИМС/ПАК, см. git log). До сих пор не имел
собственного автоматического теста, только ручную проверку при разборе
материалов -- закрываем этот пробел.
"""

import math

import pytest

from neftekod_mas.data.loaders import _iter_pairs, load_kip


def test_load_kip_drops_unnamed_columns_and_indexes_by_date(tmp_path):
    csv_path = tmp_path / "avt_tags.csv"
    csv_path.write_text(
        "Unnamed: 0.1,Unnamed: 0,date,T1,P2\n"
        "0,0,2023-01-01 00:00:00,130.5,3.5\n"
        "1,1,2023-01-01 00:10:00,130.3,3.4\n",
        encoding="utf-8",
    )
    df = load_kip(csv_path)
    assert list(df.columns) == ["T1", "P2"]
    assert df.index.name == "date"
    assert df.iloc[0]["T1"] == pytest.approx(130.5, abs=1e-3)


def test_iter_pairs_pak_style_skips_nan_spacer_columns_not_data():
    """Регрессия на реальный найденный баг: пустая ячейка после
    pd.read_excel -- float NaN, не Python None. Проверка `is not None`
    (старая версия кода) пропускала NaN как валидный маркер и портила
    разбор второй пары после настоящей пустой колонки-разделителя."""
    header_rows = [
        ("24-2000:Mg.Sulfur", math.nan, math.nan, "24-2000:D15", math.nan),
        ("ppm", math.nan, math.nan, "кг/м3", math.nan),
    ]
    pairs = list(_iter_pairs(header_rows, marker_row_idx=0, unit_row_idx=1, group_row_idx=None))
    assert [(p.param_label, p.date_col, p.value_col) for p in pairs] == [
        ("24-2000:Mg.Sulfur", 0, 1),
        ("24-2000:D15", 3, 4),
    ]


def test_iter_pairs_lims_style_forward_fills_group_label():
    header_rows = [
        ("Точка 1", None, None, "Точка 2", None, None),  # группа -- разреженная, требует ffill
        ("CFPP", None, "D15", None, "T50", None),  # показатель -- задан в каждой паре
        ("°C", None, "кг/м3", None, "°C", None),
    ]
    pairs = list(_iter_pairs(header_rows, marker_row_idx=1, unit_row_idx=2, group_row_idx=0))
    assert [(p.group_label, p.param_label) for p in pairs] == [
        ("Точка 1", "CFPP"),
        ("Точка 1", "D15"),
        ("Точка 2", "T50"),
    ]


def test_iter_pairs_empty_marker_row_yields_nothing():
    header_rows = [(math.nan, math.nan, math.nan)]
    assert list(_iter_pairs(header_rows, marker_row_idx=0, unit_row_idx=None, group_row_idx=None)) == []
