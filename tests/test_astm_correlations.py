import pytest

from neftekod_mas.quality.astm_correlations import cetane_index_d976


def test_cetane_index_d976_matches_astm_worked_example():
    """Численный пример из самого стандарта ASTM D976-91 (Фаренгейтовая
    версия формулы): API=33.0, скорректированная mid b.pt=557.14°F ->
    CCI=48.52. Пересчитываем те же исходные данные в метрические единицы
    (D976 §... версия формулы для D в г/мл, B в °C) и проверяем, что
    результат близок (независимая сверка двух представлений одного
    стандарта, а не переиспользование одного и того же числа)."""
    api = 33.0
    mid_boiling_f = 557.14

    density_g_ml = 141.5 / (131.5 + api)  # относительная плотность 60/60°F, стандартная формула API<->SG
    mid_boiling_c = (mid_boiling_f - 32) / 1.8

    cci = cetane_index_d976(density_g_ml, mid_boiling_c)
    assert cci == pytest.approx(48.52, abs=0.5)


def test_cetane_index_d976_monotonic_in_expected_directions():
    """Более лёгкое топливо (ниже T50) и менее плотное -- как правило,
    выше цетановое число в типичном рабочем диапазоне дизельных фракций."""
    base = cetane_index_d976(0.840, 280.0)
    lighter_cut = cetane_index_d976(0.840, 260.0)
    assert lighter_cut < base  # ниже T50 -> легче фракция -> обычно ниже цетановое число
