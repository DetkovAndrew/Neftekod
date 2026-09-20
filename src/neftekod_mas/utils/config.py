"""Загрузка YAML/JSON-конфигов допущений (config/*) -- см. ARCHITECTURE.md §4, §7."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


def load_hard_constraints() -> dict:
    with open(CONFIG_DIR / "hard_constraints.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_control_variables() -> dict:
    with open(CONFIG_DIR / "control_variables.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_control_bounds() -> dict:
    path = CONFIG_DIR / "control_bounds.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден. Запустите scripts/compute_control_bounds.py "
            "(требуется NEFTEKOD_DATA_DIR с avt_tags.csv/242000_tags.csv)."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_objective_weights() -> dict:
    with open(CONFIG_DIR / "objective_weights.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vak_formula_accuracy() -> dict:
    path = CONFIG_DIR / "vak_formula_accuracy.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_astm_accuracy() -> dict:
    path = CONFIG_DIR / "astm_cetane_accuracy.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_freshness() -> dict:
    """Допустимый возраст источников для синхронизации и оценки качества."""
    with open(CONFIG_DIR / "freshness.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_soft_sensor_selection() -> dict:
    """Итог scripts/benchmark_anchored.py; пусто -- soft-sensor'ы выключены."""
    path = CONFIG_DIR / "soft_sensor_selection.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_kip_bounds() -> dict:
    path = CONFIG_DIR / "kip_bounds.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден. Запустите scripts/compute_kip_bounds.py (требуется NEFTEKOD_DATA_DIR)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_reliability_bounds() -> dict:
    path = CONFIG_DIR / "reliability_bounds.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден. Запустите scripts/compute_reliability_bounds.py "
            "(требуется NEFTEKOD_DATA_DIR с 242000_tags.csv)."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_blend_model() -> dict:
    """Модель блендинга дизельного пула (ARCHITECTURE.md §6.6).

    Отсутствие файла не является ошибкой: блендинг -- отдельный контур,
    и система обязана работать и без него (возвращается пустой словарь,
    Агент блендинга при этом честно сообщает о недоступности, а не
    подставляет выдуманные доли). Пересчёт --
    `scripts/compute_blend_model.py` (нужен NEFTEKOD_DATA_DIR).
    """
    path = CONFIG_DIR / "blend_model.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_economics() -> dict:
    """Цены и физические константы для экономических критериев
    (ARCHITECTURE.md §6.7). Отсутствие файла не ошибка: без него система
    работает, просто не показывает рубли."""
    path = CONFIG_DIR / "economics.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sulfur_temp_response() -> dict:
    """Интервенционная калибровка отклика серы на температуру реактора
    по истории этой установки (ARCHITECTURE.md §6.4.1). Отсутствие файла
    не ошибка: тогда используется только литературный прокси, и карточка
    честно сообщает, что заводского подтверждения величины нет."""
    path = CONFIG_DIR / "sulfur_temp_response.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_equipment_limits() -> dict:
    """Эксплуатационные границы из истории: упоры приборов, поведение
    перед остановами, скорость дезактивации катализатора
    (ARCHITECTURE.md §6.3.1). Отсутствие файла не ошибка -- Агент
    надёжности тогда работает на одних перцентилях, как раньше."""
    path = CONFIG_DIR / "equipment_limits.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
