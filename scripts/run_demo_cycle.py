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

С LLM Monitor (OpenAI-совместимый сервер, напр. Ollama на кластере через
ssh-туннель `ssh -N -L 11434:slurm-comp3:11434 hpc`):
    python scripts/run_demo_cycle.py --timestamp "2025-12-01 10:00:00" \
        --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neftekod_mas.blending.blending_agent import BlendingAgent  # noqa: E402
from neftekod_mas.economics.economics_agent import EconomicsAgent  # noqa: E402
from neftekod_mas.data.loaders import data_dir, load_kip, load_lims, load_pak  # noqa: E402
from neftekod_mas.optimization.optimization_agent import OptimizationAgent  # noqa: E402
from neftekod_mas.optimization.joint_envelope import JointEnvelopeChecker  # noqa: E402
from neftekod_mas.orchestrator.guard import Guard  # noqa: E402
from neftekod_mas.orchestrator.llm_monitor import LLMMonitor, OpenAICompatClient  # noqa: E402
from neftekod_mas.orchestrator.llm_orchestrator import LLMOrchestrator, OpenAIToolClient  # noqa: E402
from neftekod_mas.orchestrator.orchestrator import Orchestrator  # noqa: E402
from neftekod_mas.quality.quality_agent import QualityAgent  # noqa: E402
from neftekod_mas.quality.soft_sensors import SoftSensorService  # noqa: E402
from neftekod_mas.reliability.reliability_agent import ReliabilityAgent  # noqa: E402
from neftekod_mas.tags.pid_graph import TagGraph  # noqa: E402
from neftekod_mas.utils.logging_run import RunLogger  # noqa: E402
from neftekod_mas.utils.config import (  # noqa: E402
    CONFIG_DIR,
    load_blend_model,
    load_economics,
    load_control_bounds,
    load_control_variables,
    load_hard_constraints,
    load_kip_bounds,
    load_objective_weights,
    load_reliability_bounds,
    load_vak_formula_accuracy,
    load_astm_accuracy,
    load_freshness,
    load_soft_sensor_selection,
    load_sulfur_temp_response,
)


def build_orchestrator(
    enable_run_logging: bool = True,
    soft_sensors: SoftSensorService | None = None,
    llm_monitor: LLMMonitor | None = None,
) -> Orchestrator:
    hc = load_hard_constraints()
    cv = load_control_variables()
    cb = load_control_bounds()
    rb = load_reliability_bounds()
    ow = load_objective_weights()
    graph = TagGraph.from_json(CONFIG_DIR / "tag_ontology.json")
    formula_accuracy = load_vak_formula_accuracy()
    astm_accuracy = load_astm_accuracy()
    freshness = load_freshness()

    quality_agent = QualityAgent(
        hc,
        stale_lims_minutes=freshness["lims"]["fresh_minutes"],
        usable_lims_minutes=freshness["lims"]["usable_minutes"],
        usable_pak_minutes=freshness["pak"]["usable_minutes"],
        formula_accuracy=formula_accuracy,
        astm_accuracy=astm_accuracy,
        soft_sensors=soft_sensors,
    )
    reliability_agent = ReliabilityAgent(rb)
    # Агент блендинга подключается, только если посчитана модель
    # (config/blend_model.yaml). Без неё система работает как раньше --
    # просто без рычага блендинга (ARCHITECTURE.md §6.6).
    blend_model = {} if os.environ.get("NEFTEKOD_DISABLE_BLENDING") else load_blend_model()
    blending_agent = BlendingAgent(blend_model) if blend_model.get("components") else None
    economics_cfg = load_economics()
    economics_agent = EconomicsAgent(economics_cfg) if economics_cfg else None
    optimization_agent = OptimizationAgent(
        cv, cb, hc, ow, quality_agent, reliability_agent,
        blending_agent=blending_agent, economics_agent=economics_agent,
        sulfur_temp_response=load_sulfur_temp_response(),
    )
    joint_envelope_path = CONFIG_DIR / "joint_envelope.npz"
    joint_envelope = (
        JointEnvelopeChecker.from_npz(joint_envelope_path, cb) if joint_envelope_path.exists() else None
    )
    guard = Guard(graph, cv, cb, hard_constraints=hc, joint_envelope=joint_envelope)
    try:
        kip_bounds = load_kip_bounds()
    except FileNotFoundError:
        kip_bounds = None
    run_logger = RunLogger(REPO_ROOT / "runs") if enable_run_logging else None
    return Orchestrator(
        quality_agent, reliability_agent, optimization_agent, guard,
        kip_bounds=kip_bounds,
        run_logger=run_logger,
        llm_monitor=llm_monitor,
        stale_lims_minutes=freshness["lims"]["fresh_minutes"],
        stale_pak_minutes=freshness["pak"]["fresh_minutes"],
    )


def _rub(value: float) -> str:
    """Рубли с неразрывным разделением тысяч. Форматирование делается
    только над самим числом -- подстановка пробелов во всю строку
    портила запятые в поясняющем тексте."""
    return f"{value:+,.0f}".replace(",", "\u00a0")


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

    if rec.economic_effect:
        print("-" * 78)
        print("Экономический эффект (натуральные единицы -- измерены, рубли -- по ценам-допущениям):")
        _ECON_RU = {
            "product_rate_t_h": ("выпуск ДТ", "т/ч"),
            "furnace_p3_duty_gcal_h": ("тепло печи П-3", "Гкал/ч"),
            "ht_feed_heating_gcal_h": ("нагрев сырья гидроочистки", "Гкал/ч"),
            "recycle_compressor_kw": ("мощность ЦК-201", "кВт"),
            "makeup_hydrogen_nm3_h": ("свежий ВСГ", "нм3/ч"),
        }
        base = rec.economic_effect.get("baseline_physical") or {}
        for key, value in base.items():
            name, unit = _ECON_RU.get(key, (key, ""))
            print(f"    сейчас: {name}: {value:g} {unit}")
        for key, value in (rec.economic_effect.get("delta_physical") or {}).items():
            name, unit = _ECON_RU.get(key, (key, ""))
            print(f"    изменение: {name}: {value:+g} {unit}")
        money = rec.economic_effect.get("delta_rub_per_day") or {}
        for key, value in money.items():
            print(f"    рубли/сут ({key}): {_rub(value)}")
        net = rec.economic_effect.get("net_rub_per_day")
        if net is not None:
            print(f"    ИТОГО: {_rub(net)} руб/сут (цены -- допущение, config/economics.yaml)")
        for note in rec.economic_effect.get("notes") or []:
            print(f"    ! {note}")

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
    if rec.llm_status is not None:
        print("-" * 78)
        # Две принципиально разные роли LLM не должны выглядеть одинаково
        # в карточке: Monitor только пересказывает готовое решение (§9),
        # а Оркестратор вёл цикл и выбрал вариант из проверенных (§9.1).
        if rec.orchestration_mode == "llm":
            print(f"Цикл провёл LLM-оркестратор ({rec.llm_status}); "
                  "все расчёты и Guard -- детерминированные")
            print(f"    Обоснование выбора моделью: {rec.llm_commentary}"
                  if rec.llm_commentary else "    (модель не дала пояснения)")
        else:
            print(f"Комментарий LLM Monitor ({rec.llm_status}):")
            print(f"    {rec.llm_commentary}" if rec.llm_commentary else "    (не показан)")
    if rec.alternatives:
        print("-" * 78)
        print(f"Альтернативы (Парето-фронт, {len(rec.alternatives)}):")
        for alt in rec.alternatives:
            a = alt.actions[0]
            print(f"    [{alt.candidate_id}] {a.variable_name}: {a.current_value:.4g} -> {a.recommended_value:.4g} {a.unit} (score={alt.score:.3g})")
    print("=" * 78)


def print_trace(rec) -> None:
    print("Трасса обмена агентов:")
    for m in rec.trace:
        print(f"  {m.seq:2d}. {m.sender} -> {m.recipient} [{m.topic}] {m.summary}")


def load_tag_descriptions() -> dict[str, str]:
    """Смысл тегов из справочника КИП -- для фактов карточки LLM Monitor."""
    nodes = json.loads((CONFIG_DIR / "tag_ontology.json").read_text(encoding="utf-8"))["nodes"]
    return {k: v["description"] for k, v in nodes.items() if v.get("description")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True, help='напр. "2023-03-15 10:00:00"')
    parser.add_argument("--no-soft-sensors", action="store_true", help="прежний приоритет ЛИМС -> ПАК -> формула")
    parser.add_argument("--llm-url", default=None, help="OpenAI-совместимый endpoint, напр. http://localhost:11434/v1")
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--trace", action="store_true", help="напечатать трассу сообщений агентов")
    parser.add_argument(
        "--llm-orchestrator", action="store_true",
        help="цикл ведёт локальная LLM через tool-calling (ARCHITECTURE.md §9.1); "
             "Guard и все расчёты остаются детерминированными, при любой проблеме "
             "отрабатывает обычный оркестратор",
    )
    args = parser.parse_args()

    decision_at = datetime.fromisoformat(args.timestamp)

    avt = load_kip(data_dir() / "avt_tags.csv")
    ht = load_kip(data_dir() / "242000_tags.csv")
    lims = load_lims(data_dir() / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx")
    pak = load_pak(data_dir() / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx")

    selection = {} if args.no_soft_sensors else load_soft_sensor_selection()
    soft_sensors = SoftSensorService.from_history(selection, avt, ht, lims) if selection else None
    monitor = (
        LLMMonitor(OpenAICompatClient(args.llm_url, args.llm_model), tag_descriptions=load_tag_descriptions())
        if args.llm_url else None
    )

    orchestrator = build_orchestrator(soft_sensors=soft_sensors, llm_monitor=monitor)
    if args.llm_orchestrator:
        if not args.llm_url:
            parser.error("--llm-orchestrator требует --llm-url")
        orchestrator = LLMOrchestrator(orchestrator, OpenAIToolClient(args.llm_url, args.llm_model))
    rec = orchestrator.run_cycle(decision_at, avt, ht, lims, pak)
    print_recommendation_card(rec)
    if args.trace:
        print_trace(rec)


if __name__ == "__main__":
    main()
