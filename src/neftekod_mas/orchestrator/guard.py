"""
Guard -- независимая финальная проверка рекомендации перед публикацией
(ARCHITECTURE.md §6.5, §9), архитектурная калька Verification Agent из
Schall (2026), §3.1.4: "shares no model weights, no memory... except the
consolidated plan received via MQTT". Здесь роль "MQTT" играет тот факт,
что Guard принимает только уже сериализованный `ControlCandidate` и
`config/control_variables.yaml` -- не внутренние структуры данных
Агента оптимизации напрямую.

Проверки:
  1. tag_exists         -- тег присутствует в цифровом P&ID
  2. actuatable          -- переменная есть в кандидатном реестре
                             control_variables.yaml (авторитетный список,
                             см. ARCHITECTURE.md §4 -- НЕ сырой
                             TagGraph.actuatable, т.к. для 24-2000 P&ID
                             не выдан и actuatable там всегда False по
                             построению, что не должно блокировать
                             явно обоснованные кандидаты-допущения)
  3. within_bounds        -- рекомендуемое значение в пределах
                             валидированного диапазона (control_bounds.yaml)
  4. blend_sum_100        -- нет активных переменных блендинга в v1,
                             проверка присутствует "на будущее" и сейчас
                             всегда PASS
  5. downstream_impact     -- информационная проверка через TagGraph BFS,
                             прикладывается к объяснению, не блокирует
"""

from __future__ import annotations

from datetime import datetime

from neftekod_mas.schemas import ControlCandidate, GuardCheck, GuardReport, GuardVerdict
from neftekod_mas.tags.pid_graph import TagGraph


class Guard:
    def __init__(self, tag_graph: TagGraph, control_variables_cfg: dict, control_bounds: dict):
        self.tag_graph = tag_graph
        self.known_variables = {
            var["name"]: var
            for block in control_variables_cfg.get("installations", {}).values()
            for var in block.get("variables", [])
        }
        self.bounds = {k: v for k, v in control_bounds.items() if k != "_meta"}

    def review(self, candidate: ControlCandidate | None, decision_at: datetime) -> GuardReport:
        checks: list[GuardCheck] = []

        if candidate is None:
            checks.append(GuardCheck(check_name="candidate_present", verdict=GuardVerdict.BLOCK, detail="Нет кандидата для проверки"))
            return GuardReport(decision_at=decision_at, candidate_id=None, checks=checks, final_verdict=GuardVerdict.BLOCK)

        for action in candidate.actions:
            var_cfg = self.known_variables.get(action.variable_name)
            if var_cfg is None:
                checks.append(GuardCheck(
                    check_name="actuatable", verdict=GuardVerdict.BLOCK,
                    detail=f"'{action.variable_name}' отсутствует в config/control_variables.yaml",
                ))
                continue

            checks.append(GuardCheck(
                check_name="actuatable", verdict=GuardVerdict.PASS,
                detail=f"'{action.variable_name}' в реестре кандидатных управляющих воздействий "
                       f"(confidence={var_cfg.get('confidence')})",
            ))

            if action.tag and self.tag_graph.tag_exists(action.tag):
                checks.append(GuardCheck(check_name="tag_exists", verdict=GuardVerdict.PASS, detail=action.tag))
            elif action.tag:
                checks.append(GuardCheck(
                    check_name="tag_exists", verdict=GuardVerdict.WARN,
                    detail=f"Тег {action.tag} отсутствует в цифровом P&ID (нормально для 24-2000, P&ID не выдан)",
                ))

            b = self.bounds.get(action.tag or "")
            if b is not None:
                if b["p05"] <= action.recommended_value <= b["p95"]:
                    checks.append(GuardCheck(check_name="within_bounds", verdict=GuardVerdict.PASS, detail=f"{action.recommended_value} in [{b['p05']}, {b['p95']}]"))
                else:
                    checks.append(GuardCheck(
                        check_name="within_bounds", verdict=GuardVerdict.BLOCK,
                        detail=f"{action.recommended_value} вне валидированного диапазона [{b['p05']}, {b['p95']}]",
                    ))

            if action.tag and self.tag_graph.tag_exists(action.tag):
                downstream = self.tag_graph.downstream_impact(action.tag)
                checks.append(GuardCheck(
                    check_name="downstream_impact", verdict=GuardVerdict.PASS,
                    detail=f"Затронуто (грубая топология стадий): {len(downstream)} тегов ниже по потоку",
                ))

        checks.append(GuardCheck(check_name="blend_sum_100", verdict=GuardVerdict.PASS, detail="Переменные блендинга не активны в v1"))

        if any(c.verdict == GuardVerdict.BLOCK for c in checks):
            final = GuardVerdict.BLOCK
        elif any(c.verdict == GuardVerdict.WARN for c in checks):
            final = GuardVerdict.WARN
        else:
            final = GuardVerdict.PASS

        return GuardReport(
            decision_at=decision_at,
            candidate_id=candidate.candidate_id,
            checks=checks,
            final_verdict=final,
        )
