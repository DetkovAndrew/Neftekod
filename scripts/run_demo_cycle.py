#!/usr/bin/env python3
"""
Прогоняет один полный цикл принятия решения (ТЗ, 8 шагов) на заданный
момент времени и печатает карточку рекомендации оператору
(ARCHITECTURE.md §10). Ничего не обучает -- собирает уже готовые
детерминированные агенты (ETL, ВАК-формулы, статистика, правила).

Запуск:
    export NEFTEKOD_DATA_DIR=/home/acid/neftekod
    python scripts/build_tag_ontology.py            # один раз
    python scripts/compute_reliability_bounds.py    # один раз
    python scripts/compute_control_bounds.py        # один раз
    python scripts/run_demo_cycle.py --timestamp "2023-03-15 10:00:00"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.data.loaders import data_dir, load_kip, load_lims, load_pak  # noqa: E402
from neftekod_mas.optimization.optimization_agent import OptimizationAgent  # noqa: E402
from neftekod_mas.orchestrator.guard import Guard  # noqa: E402
from neftekod_mas.orchestrator.orchestrator import Orchestrator  # noqa: E402
from neftekod_mas.quality.quality_agent import QualityAgent  # noqa: E402
from neftekod_mas.reliability.reliability_agent import ReliabilityAgent  # noqa: E402
from neftekod_mas.tags.pid_graph import TagGraph  # noqa: E402
from neftekod_mas.utils.logging_run import RunLogger  # noqa: E402
from neftekod_mas.utils.config import (  # noqa: E402
    CONFIG_DIR,
    load_control_bounds,
    load_control_variables,
    load_hard_constraints,
    load_kip_bounds,
    load_objective_weights,
    load_reliability_bounds,
    load_vak_formula_accuracy,
)


def build_orchestrator(enable_run_logging: bool = True) -> Orchestrator:
    hc = load_hard_constraints()
    cv = load_control_variables()
    cb = load_control_bounds()
    rb = load_reliability_bounds()
    ow = load_objective_weights()
    graph = TagGraph.from_json(CONFIG_DIR / "tag_ontology.json")
    formula_accuracy = load_vak_formula_accuracy()

    quality_agent = QualityAgent(hc, formula_accuracy=formula_accuracy)
    reliability_agent = ReliabilityAgent(rb)
    optimization_agent = OptimizationAgent(cv, cb, hc, ow, quality_agent, reliability_agent)
    guard = Guard(graph, cv, cb)
    try:
        kip_bounds = load_kip_bounds()
    except FileNotFoundError:
        kip_bounds = None
    run_logger = RunLogger(REPO_ROOT / "runs") if enable_run_logging else None
    return Orchestrator(
        quality_agent, reliability_agent, optimization_agent, guard,
        kip_bounds=kip_bounds, run_logger=run_logger,
    )


def print_recommendation_card(rec) -> None:
    print("=" * 78)
    print(f"Момент принятия решения: {rec.decision_at}")
    print("-" * 78)
    print("Время и состояние:")
    for k, v in rec.key_state.items():
        print(f"    {k}: {v:.4g}")
    print("-" * 78)
    print(f"Проблема/риск: {rec.problem_or_risk}")
    print("-" * 78)
    print("Предлагаемое действие:")
    if not rec.proposed_actions:
        print("    (нет)")
    for a in rec.proposed_actions:
        print(f"    {a.variable_name} [{a.tag}]: {a.current_value:.4g} -> {a.recommended_value:.4g} {a.unit}")
    print("-" * 78)
    print("Ожидаемый эффект:")
    for k, v in rec.expected_effect.items():
        print(f"    {k}: {v:.4g}")
    print("-" * 78)
    print("Проверка ограничений (Guard):")
    for c in rec.constraints_checked:
        print(f"    [{c.verdict.value.upper():5s}] {c.check_name}: {c.detail}")
    print("-" * 78)
    print(f"Уверенность: {rec.confidence.value}")
    for w in rec.confidence_warnings:
        print(f"    ! {w}")
    print("-" * 78)
    print(f"Объяснение: {rec.explanation}")
    if rec.alternatives:
        print("-" * 78)
        print(f"Альтернативы (Парето-фронт, {len(rec.alternatives)}):")
        for alt in rec.alternatives:
            a = alt.actions[0]
            print(f"    [{alt.candidate_id}] {a.variable_name}: {a.current_value:.4g} -> {a.recommended_value:.4g} {a.unit} (score={alt.score:.3g})")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True, help='напр. "2023-03-15 10:00:00"')
    args = parser.parse_args()

    decision_at = datetime.fromisoformat(args.timestamp)

    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    pak = load_pak(data_dir() / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx")

    orchestrator = build_orchestrator()
    rec = orchestrator.run_cycle(decision_at, avt, ht, lims, pak)
    print_recommendation_card(rec)


if __name__ == "__main__":
    main()
