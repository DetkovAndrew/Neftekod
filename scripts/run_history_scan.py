#!/usr/bin/env python3
"""
Прогоняет Оркестратор по диапазону истории с заданным шагом и собирает
результаты в самодостаточный офлайн-дашборд (ARCHITECTURE.md, "приветствуется:
интерактивная визуализация"). Никакого обучения -- просто много повторных
вызовов уже готового детерминированного цикла принятия решения.

Дашборд -- один HTML-файл с данными, встроенными прямо в него (никаких
fetch()/CORS, никаких внешних CDN -- открывается двойным кликом в
браузере без доступа к интернету, что прямо соответствует требованию
Q&A-сессии "решение должно работать локально без доступа к Интернету").

Результаты сканирования (JSON/HTML) НЕ коммитятся в git -- это
производные от реальных данных хакатона (см. .gitignore, runs/).

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/run_history_scan.py --start 2023-01-01 --end 2023-04-01 --step-hours 12
    # результат: runs/dashboard.html -- открыть в браузере
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard import render as render_dashboard  # noqa: E402
from run_demo_cycle import build_orchestrator  # noqa: E402
from neftekod_mas.data.loaders import data_dir, load_kip, load_lims, load_pak  # noqa: E402
from neftekod_mas.quality.soft_sensors import SoftSensorService  # noqa: E402
from neftekod_mas.utils.config import load_soft_sensor_selection  # noqa: E402


def scan(start: datetime, end: datetime, step: timedelta, use_soft_sensors: bool = True) -> list[dict]:
    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    pak = load_pak(data_dir() / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx")
    selection = load_soft_sensor_selection() if use_soft_sensors else {}
    soft_sensors = SoftSensorService.from_history(selection, avt, ht, lims) if selection else None
    # per-cycle артефакты (runs/<timestamp>/*.json) отключены при массовом
    # прогоне -- иначе сотни точек сканирования оставили бы тысячи файлов
    # на диске; для точечного разбора конкретного момента см. run_demo_cycle.py
    orchestrator = build_orchestrator(enable_run_logging=False, soft_sensors=soft_sensors)

    rows: list[dict] = []
    ts = start
    while ts <= end:
        rec = orchestrator.run_cycle(ts, avt, ht, lims, pak)
        decision = "refuse" if rec.is_refusal else ("recommend" if rec.proposed_actions else "no_action")
        rows.append({
            "t": ts.isoformat(),
            "decision": decision,
            "confidence": rec.confidence.value,
            "problem": rec.problem_or_risk,
            "key_state": rec.key_state,
            "n_warnings": len(rec.confidence_warnings),
            "transient": any(w.startswith("transient_regime") for w in rec.confidence_warnings),
            "stopped_at": rec.trace[-1].sender if rec.trace else None,
        })
        ts += step
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-hours", type=float, default=12.0)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--no-soft-sensors", action="store_true")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    step = timedelta(hours=args.step_hours)

    rows = scan(start, end, step, use_soft_sensors=not args.no_soft_sensors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "scan.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    period = f"{start:%Y-%m-%d} — {end:%Y-%m-%d}, шаг {args.step_hours:g} ч"
    html = render_dashboard(rows, period)
    html_path = out_dir / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    n_no_action = sum(1 for r in rows if r["decision"] == "no_action")
    n_recommend = sum(1 for r in rows if r["decision"] == "recommend")
    n_refuse = sum(1 for r in rows if r["decision"] == "refuse")
    n_transient = sum(1 for r in rows if r["transient"])
    print(f"Прогнано {len(rows)} точек: no_action={n_no_action}, recommend={n_recommend}, "
          f"refuse={n_refuse} (из них переходный режим: {n_transient})")
    print(f"Дашборд -> {html_path} (открыть в браузере)")


if __name__ == "__main__":
    main()
