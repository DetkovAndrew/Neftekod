"""
Сохранение промежуточных артефактов каждого цикла принятия решения на
диск (ARCHITECTURE.md §6, §10, ТЗ: "система должна сохранять входные
данные, оценки агентов и итоговую рекомендацию так, чтобы логику
решения можно было проверить" -- это прямое требование
воспроизводимости/объяснимости, а не диагностический лог "на всякий
случай").

Каталог `runs/` НЕ входит в git (см. .gitignore) -- артефакты содержат
значения, производные от реальной технологической телеметрии.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return json.loads(obj.model_dump_json())
    if isinstance(obj, list):
        return [_to_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


class RunLogger:
    """Пишет по одному JSON-файлу на каждый артефакт цикла в
    `runs/<decision_at>/<NN_name>.json`. Порядок имён (01_, 02_...)
    соответствует порядку шагов ТЗ п.1 (ARCHITECTURE.md §8), чтобы
    директорию можно было просто читать сверху вниз как трассу решения."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def _run_dir(self, decision_at: datetime) -> Path:
        d = self.base_dir / decision_at.strftime("%Y%m%dT%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log(self, decision_at: datetime, name: str, obj: Any) -> None:
        path = self._run_dir(decision_at) / f"{name}.json"
        path.write_text(
            json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


class NullRunLogger:
    """No-op логгер по умолчанию -- цикл работает и без записи на диск
    (напр. в юнит-тестах или при массовом прогоне run_history_scan.py,
    где сотни артефактов на диске не нужны)."""

    def log(self, decision_at: datetime, name: str, obj: Any) -> None:
        return None
