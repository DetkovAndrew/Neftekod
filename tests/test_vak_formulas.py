"""
Сверка ВАК-формул с примерами расчёта из формулы_ВАК.xlsx.

Это не тест ML-модели (обучения не было) -- это регрессионный тест на
то, что коэффициенты формул перенесены в код без опечаток. Входные
значения и ожидаемые результаты взяты дословно из столбца
"Пример расчёта" исходного файла. Допуск (abs=0.2) отражает то, что
сам исходный файл показывает промежуточные суммы с округлением.
"""

import pytest

from neftekod_mas.quality import vak_formulas as vak


def test_avt6_d15_240_350():
    t = {"F65": 789.54, "F32": 79.00, "F30": 98.97, "T66": 255.04, "T33": 335.41}
    assert vak.avt6_d15_240_350(t) == pytest.approx(849.83, abs=0.2)


def test_avt6_t50_240_350():
    t = {"F7": 217.54, "F30": 98.97, "F34": 60.02, "F45": 16.45, "F59": 196.85, "F63": 88.16}
    assert vak.avt6_t50_240_350(t) == pytest.approx(271.82, abs=0.2)


def test_avt6_cfpp_240_350():
    t = {"T33": 335.41, "P67": 1.10, "P4": 3.85, "F65": 789.54, "F32": 79.00, "F30": 98.97}
    assert vak.avt6_cfpp_240_350(t) == pytest.approx(-5.53, abs=0.2)


def test_avt6_ebp_240_350():
    t = {"F30": 98.97, "T33": 335.41, "F36": 139.29, "T37": 68.18, "T40": 180.86, "T58": 66.04}
    assert vak.avt6_ebp_240_350(t) == pytest.approx(264.42, abs=0.2)


def test_avt6_viscosity_k_350_500():
    t = {
        "T6": 234.73, "T13": 89.20, "T18": 158.40, "T20": 113.80, "L43": 60.08,
        "T48": 354.43, "P50": 0.96, "F53": 80.13, "P51": 42.90, "F59": 196.85, "T61": 193.06,
    }
    assert vak.avt6_viscosity_k_350_500(t) == pytest.approx(4.18, abs=0.2)


def test_avt6_cfpp_350():
    # F31/F57 = 21.0124 в исходном примере; конкретные F31/F57 не даны,
    # берём пару с тем же отношением -- формула зависит только от него.
    t = {"T48": 354.43, "T40": 180.86, "F31": 210.124, "F57": 10.0}
    assert vak.avt6_cfpp_350(t) == pytest.approx(-2.10, abs=0.2)


def test_avt6_t50_350():
    t = {"T42": 275.79, "T48": 355.21, "F31": 498.33, "F57": 20.53, "T66": 256.08, "T33": 336.20}
    assert vak.avt6_t50_350(t) == pytest.approx(299.67, abs=0.2)


def test_avt6_i350_350():
    t = {"L43": 60.08, "T6": 234.73, "T18": 158.40, "F64": 104.56, "T15": 135.58, "T11": 56.14}
    assert vak.avt6_i350_350(t) == pytest.approx(88.82, abs=0.2)


def test_avt6_d15_350():
    t = {"T42": 275.99, "T48": 354.43, "F31": 500.83, "F57": 23.84}
    assert vak.avt6_d15_350(t) == pytest.approx(878.25, abs=0.2)


def test_ht_godt_t90():
    t = {"T12": 175.10, "F15": 2787.38, "W7": 0.143, "T23": 234.20, "F1": 2.613, "F26": 201.23}
    assert vak.ht_godt_t90(t) == pytest.approx(324.92, abs=0.2)


def test_ht_godt_t50():
    t = {"P13": 3.759, "F9": 171.09, "T6": 360.13}
    assert vak.ht_godt_t50(t) == pytest.approx(263.86, abs=0.2)


def test_ht_godt_i250():
    t = {"T5": 365.14, "T11": 363.60, "F25": 13242.36, "F14": 5.824, "T23": 234.20, "T16": 30.011}
    assert vak.ht_godt_i250(t) == pytest.approx(20.70, abs=0.2)


def test_ht_godt_d15():
    t = {"F22": 10507.55, "T11": 363.60}
    lims = {"24-2000.Pipeline.D15": 840.0}
    assert vak.ht_godt_d15(t, lims) == pytest.approx(837.08, abs=0.2)


def test_ht_godt_cloud_point():
    t = {"F22": 10507.55, "W7": 0.143, "F25": 13242.36, "F1": 2.613, "T6": 360.13, "F9": 171.09, "T16": 30.011}
    assert vak.ht_godt_cloud_point(t) == pytest.approx(-0.91, abs=0.2)


def test_ht_godt_cfpp():
    t = {"T23": 234.20, "P8": 0.117, "F9": 171.09, "W7": 0.143, "P24": 0.595}
    assert vak.ht_godt_cfpp(t) == pytest.approx(-17.33, abs=0.2)


def test_ht_godt_t95():
    t = {"F9": 171.09, "F2": 88476.80, "T6": 360.13}
    lims = {"95%.T": 347.5}
    assert vak.ht_godt_t95(t, lims) == pytest.approx(343.54, abs=0.2)


def test_ht_godt_ibp():
    t = {
        "F26": 201.23, "F22": 10507.55, "P13": 3.759, "P24": 0.595,
        "F14": 5.824, "W4": 1.818, "T23": 234.20, "T16": 30.011,
    }
    assert vak.ht_godt_ibp(t) == pytest.approx(197.73, abs=0.2)


def test_registries_are_complete():
    assert len(vak.AVT6_FORMULAS) == 9
    assert len(vak.HT242000_FORMULAS) == 8
    assert vak.HT242000_FORMULAS_NEED_LIMS == {"24-2000:GODT:D15", "24-2000:GODT:T95"}
