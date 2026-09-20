import pytest

from neftekod_mas.quality import literature_proxies as litproxy


def test_hds_severity_positive_for_real_desulfurization():
    # сырьё 0.5% масс. (5000 мг/кг) -> продукт 8 мг/кг
    severity = litproxy.hds_severity(0.5, 8.0)
    assert severity == pytest.approx(6.4378, abs=1e-3)


def test_hds_severity_rejects_nonpositive_inputs():
    with pytest.raises(ValueError):
        litproxy.hds_severity(0.5, 0.0)


def test_arrhenius_heating_reduces_sulfur():
    severity = litproxy.hds_severity(0.5, 8.0)
    reduced = litproxy.sulfur_after_reactor_temp_change_arrhenius(8.0, severity, 370.0, delta_t_c=2.0)
    assert reduced < 8.0


def test_arrhenius_cooling_increases_sulfur():
    severity = litproxy.hds_severity(0.5, 8.0)
    increased = litproxy.sulfur_after_reactor_temp_change_arrhenius(8.0, severity, 370.0, delta_t_c=-2.0)
    assert increased > 8.0


def test_arrhenius_matches_qa_cited_range_order_of_magnitude():
    """Перекрёстная проверка с Q&A вопросом №32 (5-10% относительного
    снижения серы на 1°C) -- см. docstring literature_proxies.py.
    Не точное совпадение (независимые источники), а проверка того, что
    оценка лежит в разумном порядке величины этого диапазона."""
    severity = litproxy.hds_severity(0.5, 8.0)
    new_value = litproxy.sulfur_after_reactor_temp_change_arrhenius(8.0, severity, 370.0, delta_t_c=1.0)
    relative_change = (8.0 - new_value) / 8.0
    assert 0.02 < relative_change < 0.20  # разумный запас вокруг 5-10%


def test_flat_fallback_never_negative():
    assert litproxy.sulfur_after_reactor_temp_change_flat(5.0, delta_t_c=100.0) == 0.0


# --- Заводская калибровка отклика серы (ARCHITECTURE.md §6.4.1) -------

from neftekod_mas.quality.literature_proxies import (  # noqa: E402
    conservative_sulfur_prediction,
    sulfur_after_reactor_temp_change_plant_calibrated,
)


def test_plant_calibrated_response_has_correct_direction():
    base = 10.0
    warmer = sulfur_after_reactor_temp_change_plant_calibrated(base, +4.0, -2.29)
    cooler = sulfur_after_reactor_temp_change_plant_calibrated(base, -4.0, -2.29)
    assert warmer < base < cooler


def test_plant_calibrated_response_is_multiplicative():
    """Кинетика ГДС мультипликативна: два шага по 2 °C обязаны дать то же,
    что один шаг на 4 °C. Аддитивная модель здесь была бы ошибкой."""
    one = sulfur_after_reactor_temp_change_plant_calibrated(10.0, 4.0, -2.29)
    two = sulfur_after_reactor_temp_change_plant_calibrated(
        sulfur_after_reactor_temp_change_plant_calibrated(10.0, 2.0, -2.29), 2.0, -2.29)
    assert one == pytest.approx(two)


def test_conservative_choice_never_promises_more_than_the_weaker_model():
    """Из литературной и заводской оценок берётся та, что обещает МЕНЬШЕ
    улучшения -- завышение эффекта опаснее занижения."""
    assert conservative_sulfur_prediction(6.59, 9.12, 9.97) == pytest.approx(9.12)
    assert conservative_sulfur_prediction(11.0, 10.4, 9.97) == pytest.approx(11.0)


def test_plant_calibration_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        sulfur_after_reactor_temp_change_plant_calibrated(0.0, 4.0, -2.29)
