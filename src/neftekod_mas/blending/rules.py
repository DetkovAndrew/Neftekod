"""
Правила смешения компонентов дизельного пула.

Здесь только математика смешения -- никакой загрузки данных и никаких
подогнанных под наши данные коэффициентов, кроме явно передаваемых
снаружи поправок (bias), измеренных бэктестом в
`scripts/compute_blend_model.py` и сохранённых в `config/blend_model.yaml`.

Для каждого показателя используется то правило, которое физически
корректно для этой величины, а не "линейная комбинация по умолчанию":

* **Плотность** смешивается линейно по ОБЪЁМУ (сохранение массы и объёма
  при смешении близких по природе нефтяных фракций). Это точное
  соотношение, а не корреляция.
* **Сера, % масс.** смешивается линейно по МАССЕ -- тоже точное
  соотношение (сохранение массы серы).
* **Разгонка (T50/T90/T95)** НЕ смешивается линейно по точкам: смесь
  двух фракций имеет собственную кривую отгона. Правильный приём --
  сложить кривые отгона компонентов по объёму на общей температурной
  сетке и прочитать нужные точки обратной интерполяцией
  (`blend_distillation`). Линейное усреднение самих температур даёт
  ошибку в разы больше -- это показано бэктестом на реальных данных
  (см. ARCHITECTURE.md §6.6).
* **Температура помутнения / CFPP** смешиваются через степенной индекс
  смешения (blending index) -- классический приём нефтепереработки для
  низкотемпературных свойств, которые сильно нелинейны по составу:
  `BI = T_K^(1/n)`, смешивается BI, затем обратное преобразование.
  Показатель степени n не подбирается на наших данных заново -- берётся
  литературное значение n=0.05 (Chevron/Nelson blending index для
  pour/cloud point), а бэктест только ПРОВЕРЯЕТ его пригодность
  и измеряет остаточное смещение.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Показатель степени индекса смешения для низкотемпературных свойств
# (температура помутнения, CFPP). Литературное значение, не подгонка.
CLOUD_POINT_INDEX_EXPONENT = 0.05

# Доли отгона, которым соответствуют точки разгонки в ЛИМС.
DISTILLATION_POINTS: tuple[tuple[str, float], ...] = (
    ("IBP.T", 0.0),
    ("50%.T", 50.0),
    ("90%.T", 90.0),
    ("95%.T", 95.0),
    ("EBP.T", 100.0),
)

KELVIN = 273.15


class BlendingError(ValueError):
    """Некорректный набор компонентов (доли не суммируются, пустой список)."""


@dataclass(frozen=True)
class Component:
    """Один компонент пула на момент смешения."""

    key: str
    mass_fraction: float  # доля по массе, 0..1
    density_kg_m3: float  # плотность при 15 °C

    @property
    def volume_per_mass(self) -> float:
        return self.mass_fraction / self.density_kg_m3


def _check(components: list[Component], tol: float = 1e-6) -> None:
    if not components:
        raise BlendingError("Пустой список компонентов")
    total = sum(c.mass_fraction for c in components)
    if abs(total - 1.0) > tol:
        raise BlendingError(f"Сумма массовых долей = {total:.6f}, должна быть 1.0")
    if any(c.density_kg_m3 <= 0 for c in components):
        raise BlendingError("Плотность компонента должна быть положительной")


def volume_fractions(components: list[Component]) -> list[float]:
    """Переводит массовые доли в объёмные через плотности компонентов."""
    _check(components)
    v = [c.volume_per_mass for c in components]
    total = sum(v)
    return [x / total for x in v]


def blend_density(components: list[Component]) -> float:
    """Плотность смеси: масса / объём. Точное соотношение."""
    _check(components)
    return 1.0 / sum(c.volume_per_mass for c in components)


def blend_mass_linear(components: list[Component], values: list[float]) -> float:
    """Линейное смешение по массе -- для массовых концентраций (сера, % масс.)."""
    _check(components)
    return float(sum(c.mass_fraction * v for c, v in zip(components, values, strict=True)))


def blend_volume_linear(components: list[Component], values: list[float]) -> float:
    """Линейное смешение по объёму -- для объёмных свойств."""
    w = volume_fractions(components)
    return float(sum(wi * v for wi, v in zip(w, values, strict=True)))


def blend_distillation(
    components: list[Component],
    curves: list[dict[str, float]],
    targets: tuple[float, ...] = (50.0, 90.0, 95.0),
) -> dict[float, float]:
    """Смешение кривых разгонки по объёму.

    `curves[i]` -- разгонка i-го компонента: {имя точки ЛИМС -> °C}.
    Возвращает {доля отгона, % -> температура смеси, °C}.

    Кривые приводятся к монотонному виду (`np.maximum.accumulate`):
    в лабораторных данных изредка встречается T90 > T95 из-за
    округления/разных методов, и без этого обратная интерполяция
    давала бы разрывы.
    """
    w = volume_fractions(components)
    temps: list[np.ndarray] = []
    for curve in curves:
        try:
            t = np.array([float(curve[name]) for name, _ in DISTILLATION_POINTS], dtype=float)
        except KeyError as exc:  # неполная кривая -- смешивать нечего
            raise BlendingError(f"В кривой разгонки нет точки {exc}") from exc
        if not np.all(np.isfinite(t)):
            raise BlendingError("Кривая разгонки содержит NaN")
        temps.append(np.maximum.accumulate(t))

    fracs = np.array([f for _, f in DISTILLATION_POINTS], dtype=float)
    lo = min(float(t[0]) for t in temps) - 5.0
    hi = max(float(t[-1]) for t in temps) + 5.0
    grid = np.linspace(lo, hi, 400)

    evaporated = np.zeros_like(grid)
    for wi, t in zip(w, temps, strict=True):
        evaporated += wi * np.interp(grid, t, fracs)

    # Обратная интерполяция: доля отгона -> температура. `np.interp`
    # требует возрастающей оси x, а сумма монотонных кривых монотонна.
    return {q: float(np.interp(q, evaporated, grid)) for q in targets}


def blend_cloud_point(
    components: list[Component],
    values_c: list[float],
    exponent: float = CLOUD_POINT_INDEX_EXPONENT,
) -> float:
    """Смешение температуры помутнения / CFPP через индекс смешения.

    BI = T_K^(1/n); индексы смешиваются по объёму; обратно T = BI^n.
    Температуры ниже абсолютного нуля физически невозможны и отсекаются.
    """
    w = volume_fractions(components)
    idx = 0.0
    for wi, t_c in zip(w, values_c, strict=True):
        t_k = float(t_c) + KELVIN
        if t_k <= 0:
            raise BlendingError(f"Температура {t_c} °C ниже абсолютного нуля")
        idx += wi * t_k ** (1.0 / exponent)
    return float(idx**exponent - KELVIN)
