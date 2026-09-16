#!/usr/bin/env python3
"""
Строит config/tag_ontology.json -- цифровой P&ID (граф тегов) из
справочника "теги АВТ_24-2000.xlsx" (листы АВТ / 24-2000) и ручной
разметки actuatable-статуса по сканам P&ID (pid_annotations.py).

Артефакт config/tag_ontology.json НЕ является сырыми данными хакатона --
это производный справочник (код тега -> описание/единица/actuatable),
явно предназначенный организаторами для использования в решениях
участников (см. ARCHITECTURE.md §11), поэтому он коммитится в git.

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod   # или используется дефолт ../neftekod
    python scripts/build_tag_ontology.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import openpyxl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.tags.pid_graph import Confidence, NodeKind, TagGraph, TagNode  # noqa: E402
from neftekod_mas.tags import pid_annotations as pid  # noqa: E402


def data_dir() -> Path:
    return Path(os.environ.get("NEFTEKOD_DATA_DIR", REPO_ROOT / ".." / "neftekod")).resolve()


# Грубая разбивка тегов по стадиям процесса -- по описаниям в справочнике
# и по тому, на каком из 4 сканов P&ID тег физически присутствует
# (ARCHITECTURE.md §3). Список сформирован вручную один раз при разборе
# материалов; при появлении официальной схемы 24-2000 должен быть
# пересобран из неё, а не редактироваться "на глаз".
AVT_STAGE_BY_TAG: dict[str, str] = {
    **{t: "K1" for t in ["T1", "P2", "F3", "P4", "F5", "T6", "F7", "F8", "F9", "D10"]},
    **{
        t: "K2"
        for t in [
            "T11", "F12", "T13", "F14", "T15", "F16", "T17", "T18", "F19", "T20",
            "P21", "P22", "P23", "T24", "F25", "F26", "F27", "F28", "F29", "F30",
            "F31", "F32", "T33", "F34", "F64", "F65", "T66", "T71", "W70", "P67",
        ]
    },
    **{
        t: "K10"
        for t in [
            "F35", "F36", "T37", "T38", "T39", "T40", "F41", "T42", "L43", "P44",
            "F45", "F46", "T47", "T48", "T49", "P50", "P51", "P52", "F53", "F54",
            "T55", "F56", "F57", "T58", "F59", "F60", "T61", "F62", "F63", "F68", "F69",
        ]
    },
}

HT242000_STAGE_BY_TAG: dict[str, str] = {
    "F1": "unit_general", "F2": "stabilization", "P3": "reactor", "W4": "stabilization",
    "T5": "reactor", "T6": "reactor", "W7": "stabilization", "P8": "reactor",
    "F9": "unit_general", "W10": "unit_general", "T11": "reactor", "T12": "stabilization",
    "P13": "reactor", "F14": "reactor", "F15": "unit_general", "T16": "stabilization",
    "F17": "unit_general", "T18": "stabilization", "F19": "stabilization", "Q20": "reactor",
    "Q21": "unit_general", "F22": "stabilization", "T23": "stabilization", "P24": "stabilization",
    "F25": "reactor", "F26": "unit_general",
}


def physical_quantity_from_unit_text(text: str) -> str:
    text = (text or "").lower()
    if "температура" in text:
        return "temperature"
    if "давление" in text or "вакуум" in text or "перепад" in text:
        return "pressure"
    if "плотность" in text:
        return "density"
    if "уровень" in text:
        return "level"
    if "т/ч" in text or "масс.расход" in text:
        return "flow_mass"
    if "м3/ч" in text or "м³/ч" in text or "нм3/ч" in text or "нм³/ч" in text:
        return "flow_volume"
    if "сера" in text or "ppm" in text:
        return "concentration_ppm"
    return "unknown"


def load_sheet_rows(path: Path, sheet_name: str) -> list[tuple]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    return rows[1:]  # skip header


def build() -> TagGraph:
    src = data_dir() / "теги АВТ_24-2000.xlsx"
    if not src.exists():
        raise SystemExit(
            f"Не найден {src}. Установите NEFTEKOD_DATA_DIR на каталог с материалами хакатона."
        )

    graph = TagGraph()

    for tag_code, desc, unit_text in load_sheet_rows(src, "АВТ"):
        if not tag_code:
            continue
        stage = AVT_STAGE_BY_TAG.get(tag_code, "unknown")
        controller_type = pid.CONFIRMED_ACTUATABLE_AVT.get(tag_code)
        actuatable = controller_type is not None
        graph.add(TagNode(
            tag_id=tag_code,
            installation="avt",
            stage=stage,
            description=desc or "",
            unit=unit_text or "",
            physical_quantity=physical_quantity_from_unit_text(unit_text),
            node_kind=NodeKind.CONTROLLER if actuatable else NodeKind.SENSOR,
            actuatable=actuatable,
            confidence=Confidence.CONFIRMED_FROM_PID if (actuatable or tag_code in pid.CONFIRMED_INDICATOR_AVT) else Confidence.ASSUMPTION,
            controller_type=controller_type,
        ))

    for tag_code, desc, unit_text in load_sheet_rows(src, "24-2000"):
        if not tag_code:
            continue
        stage = HT242000_STAGE_BY_TAG.get(tag_code, "unknown")
        controller_type = pid.CONFIRMED_ACTUATABLE_242000.get(tag_code)
        actuatable = controller_type is not None
        graph.add(TagNode(
            tag_id=tag_code,
            installation="242000",
            stage=stage,
            description=desc or "",
            unit=unit_text or "",
            physical_quantity=physical_quantity_from_unit_text(unit_text),
            node_kind=NodeKind.CONTROLLER if actuatable else NodeKind.SENSOR,
            actuatable=actuatable,
            # P&ID для 24-2000 не выдан -- вся разметка здесь по умолчанию assumption.
            confidence=Confidence.ASSUMPTION,
            controller_type=controller_type,
        ))

    return graph


def main() -> None:
    graph = build()
    out_path = REPO_ROOT / "config" / "tag_ontology.json"
    graph.to_json(out_path)

    n_actuatable = sum(1 for n in graph.nodes.values() if n.actuatable)
    n_avt = sum(1 for n in graph.nodes.values() if n.installation == "avt")
    n_242000 = sum(1 for n in graph.nodes.values() if n.installation == "242000")
    print(f"Записано {len(graph.nodes)} тегов -> {out_path}")
    print(f"  АВТ: {n_avt}, 24-2000: {n_242000}")
    print(f"  actuatable (управляемые): {n_actuatable}")
    print(f"  indicator (только измерение): {len(graph.nodes) - n_actuatable}")


if __name__ == "__main__":
    main()
