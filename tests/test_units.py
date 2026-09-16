import pytest

from neftekod_mas.utils import units


def test_mpa_mmhg_roundtrip():
    assert units.mmhg_to_mpa(units.mpa_to_mmhg(1.0)) == pytest.approx(1.0, rel=1e-9)


def test_mpa_to_mmhg_conversion_factor():
    """1 МПа = 7500.6168 мм рт.ст. -- физический коэффициент, объясняющий
    пару P50 (МПа)/P51 (мм рт.ст.) в теги АВТ_24-2000.xlsx: одна и та же
    величина вакуума верха К-10, пересчитанная дважды (units.py docstring)."""
    assert units.mpa_to_mmhg(1.0) == pytest.approx(7500.6168)


def test_kgf_cm2_to_mpa():
    assert units.kgf_cm2_to_mpa(1.0) == pytest.approx(0.0980665)


def test_celsius_kelvin_roundtrip():
    assert units.kelvin_to_celsius(units.celsius_to_kelvin(25.0)) == pytest.approx(25.0)


def test_celsius_to_kelvin_known_point():
    assert units.celsius_to_kelvin(0.0) == pytest.approx(273.15)
