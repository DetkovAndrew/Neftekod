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
