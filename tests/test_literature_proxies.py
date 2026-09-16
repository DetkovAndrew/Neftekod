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
