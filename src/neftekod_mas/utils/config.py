"""Загрузка YAML-конфигов допущений (config/*.yaml) -- см. ARCHITECTURE.md §4, §7."""

from __future__ import annotations

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


def load_reliability_bounds() -> dict:
    path = CONFIG_DIR / "reliability_bounds.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден. Запустите scripts/compute_reliability_bounds.py "
            "(требуется NEFTEKOD_DATA_DIR с 242000_tags.csv)."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
