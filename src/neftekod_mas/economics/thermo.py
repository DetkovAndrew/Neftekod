"""
Теплофизика нефтяных фракций -- ровно столько, сколько нужно, чтобы
выражать энергозатраты в Гкал, а не в безразмерном "прокси"
(ARCHITECTURE.md §6.7).

Единственная корреляция здесь -- Kesler-Lee для теплоёмкости жидких
нефтяных фракций в форме из Riazi, "Characterization and Properties of
Petroleum Fractions" (ASTM MNL50), ур. 7.40. Это опубликованная
отраслевая корреляция, а не подгонка под наши данные: ни один её
коэффициент мы не трогаем.

Характеризующий фактор Ватсона Kw = Tb^(1/3)/SG (Tb -- средняя
температура кипения в °R) считается из фактических ЛИМС-данных
установки (T50 и плотность), а не берётся "типовой из справочника".
"""

from __future__ import annotations

KCAL_PER_KJ = 0.2388458966
GCAL_PER_GJ = 1.0 / 4.1868
WATER_DENSITY_15C = 999.016  # кг/м3, для перевода плотности в SG

# Границы применимости корреляции Kesler-Lee по приведённой температуре.
# Вне их значение всё равно возвращается, но помечается как экстраполяция.
KESLER_LEE_T_MIN_K = 233.0
KESLER_LEE_T_MAX_K = 700.0


def specific_gravity(density_kg_m3: float) -> float:
    """SG 15/15 из плотности при 15 °C."""
    if density_kg_m3 <= 0:
        raise ValueError("Плотность должна быть положительной")
    return density_kg_m3 / WATER_DENSITY_15C


def watson_k(mean_boiling_point_c: float, density_kg_m3: float) -> float:
    """Характеризующий фактор Ватсона. Tb переводится в градусы Ренкина."""
    t_rankine = (mean_boiling_point_c + 273.15) * 1.8
    if t_rankine <= 0:
        raise ValueError("Температура кипения ниже абсолютного нуля")
    return t_rankine ** (1.0 / 3.0) / specific_gravity(density_kg_m3)


def liquid_heat_capacity_kj_kg_k(
    temperature_c: float, density_kg_m3: float, mean_boiling_point_c: float
) -> float:
    """Теплоёмкость жидкой нефтяной фракции, кДж/(кг·К).

    Kesler-Lee в форме Riazi (ур. 7.40):
        Cp = (0.28299 + 0.23605*Kw)
             * [0.645 - 0.05959*SG + (2.32056 - 0.94752*SG)*(T/1000 - 0.25537)]
    T -- в кельвинах.
    """
    sg = specific_gravity(density_kg_m3)
    kw = watson_k(mean_boiling_point_c, density_kg_m3)
    t_k = temperature_c + 273.15
    a = 0.28299 + 0.23605 * kw
    b = 0.645 - 0.05959 * sg + (2.32056 - 0.94752 * sg) * (t_k / 1000.0 - 0.25537)
    return a * b


def is_extrapolation(temperature_c: float) -> bool:
    t_k = temperature_c + 273.15
    return not (KESLER_LEE_T_MIN_K <= t_k <= KESLER_LEE_T_MAX_K)


def heating_duty_gcal_h(
    mass_flow_t_h: float,
    delta_t_c: float,
    density_kg_m3: float,
    mean_boiling_point_c: float,
    mean_temperature_c: float,
) -> float:
    """Тепловая нагрузка на нагрев потока, Гкал/ч.

    Q = m * Cp * dT. Теплоёмкость берётся при СРЕДНЕЙ температуре
    участка нагрева -- Cp нефтяных фракций заметно растёт с
    температурой, и подстановка Cp на входе завысила бы/занизила бы
    результат на единицы процентов.
    """
    cp = liquid_heat_capacity_kj_kg_k(mean_temperature_c, density_kg_m3, mean_boiling_point_c)
    duty_kj_h = mass_flow_t_h * 1000.0 * cp * delta_t_c
    return duty_kj_h * KCAL_PER_KJ / 1e6


def compressor_power_kw(
    volumetric_flow_nm3_h: float,
    suction_pressure_mpa_abs: float,
    discharge_pressure_mpa_abs: float,
    suction_temperature_c: float = 40.0,
    polytropic_efficiency: float = 0.75,
    k_ratio: float = 1.38,
) -> float:
    """Политропная мощность центробежного компрессора, кВт.

    Классическое выражение через работу политропного сжатия идеального
    газа. Коэффициент адиабаты k=1.38 -- для водородсодержащего газа
    (преимущественно H2 с примесью лёгких УВ), КПД 0.75 -- типовое
    значение для центробежной машины; оба помечены как допущения в
    config/economics.yaml и не выводятся из наших данных.
    """
    if volumetric_flow_nm3_h <= 0 or suction_pressure_mpa_abs <= 0:
        return 0.0
    ratio = discharge_pressure_mpa_abs / suction_pressure_mpa_abs
    if ratio <= 1.0:
        return 0.0
    n = k_ratio / (k_ratio - 1.0)
    # Мольный расход из нормального объёма (0 °C, 101.325 кПа).
    mol_per_h = volumetric_flow_nm3_h * 1000.0 / 22.414
    r = 8.314  # Дж/(моль·К)
    t_suction_k = suction_temperature_c + 273.15
    work_j_per_h = n * mol_per_h * r * t_suction_k * (ratio ** (1.0 / n) - 1.0)
    return work_j_per_h / 3.6e6 / polytropic_efficiency
