#!/usr/bin/env python3
"""Проверяет, что приватные исходные данные не попали в рабочее дерево.

Скрипт не читает телеметрию и не передаёт данные в сеть. Он проверяет только
имена файлов, видимые Git, поэтому его безопасно запускать перед коммитом или
перед передачей архива проекта.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".npz"}
FORBIDDEN_TOP_LEVEL_DIRS = {"runs", "raw", "data", "neftekod"}


def _git_paths(*args: str) -> list[Path]:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _changed_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        if len(line) >= 4:
            paths.append(Path(line[3:].strip()))
    return paths


def _is_forbidden(path: Path) -> bool:
    return path.suffix.lower() in FORBIDDEN_SUFFIXES or (
        bool(path.parts) and path.parts[0].lower() in FORBIDDEN_TOP_LEVEL_DIRS
    )


def main() -> int:
    paths = set(_git_paths("ls-files")) | set(_changed_paths())
    forbidden = sorted(path.as_posix() for path in paths if _is_forbidden(path))
    if forbidden:
        print("ОШИБКА: найдены файлы, которые могут содержать приватные данные:")
        print("\n".join(f"  - {path}" for path in forbidden))
        return 1
    print("Privacy check passed: приватные наборы данных не найдены среди файлов Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
