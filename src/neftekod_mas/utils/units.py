"""
Явный реестр единиц измерения и конверсий (ARCHITECTURE.md §2, правило 4).

Ничего не конвертируется "по наитию" -- каждая используемая в проекте пара
единиц с коэффициентом переноса объявлена здесь и покрыта тестом.
Пример реального дублирования величины в разных единицах в исходных
данных: P50 (МПа) и P51 (мм рт. ст.) -- это одно и то же давление верха
К-10, пересчитанное дважды (см. теги АВТ_24-2000.xlsx: "P51 -- пересчёт
на мм рт.ст. с кгс/см2 от AVT6:PIR0825.P").
"""

from __future__ import annotations

MPA_TO_MMHG = 7500.6168  # 1 МПа = 7500.6168 мм рт. ст.
KGF_CM2_TO_MPA = 0.0980665  # 1 кгс/см2 = 0.0980665 МПа


def mpa_to_mmhg(value_mpa: float) -> float:
    return value_mpa * MPA_TO_MMHG


def mmhg_to_mpa(value_mmhg: float) -> float:
    return value_mmhg / MPA_TO_MMHG


def kgf_cm2_to_mpa(value_kgf_cm2: float) -> float:
    return value_kgf_cm2 * KGF_CM2_TO_MPA


def celsius_to_kelvin(value_c: float) -> float:
    return value_c + 273.15


def kelvin_to_celsius(value_k: float) -> float:
    return value_k - 273.15


# Канонические единицы по типу физической величины -- всё внутри пайплайна
# приводится к этому набору при загрузке (см. data/loaders.py).
CANONICAL_UNITS: dict[str, str] = {
    "temperature": "°C",
    "pressure": "МПа",
    "flow_mass": "т/ч",
    "flow_volume": "м³/ч",
    "density": "кг/м³",
    "level": "%",
    "concentration_mass": "мг/кг",
    "concentration_wt_pct": "% масс.",
    "concentration_ppm": "ppm",
}
