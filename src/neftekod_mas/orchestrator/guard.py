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
  6. joint_envelope        -- (опционально, если передан JointEnvelopeChecker)
                             совместное состояние всех активных управляющих
                             переменных после применения кандидата сверяется
                             с историческим облаком точек (nearest-neighbor),
                             не только маржинальный диапазон самой изменяемой
                             переменной (ARCHITECTURE.md §5.3, по методике
                             Ta & Liu 2027 §3.3.3). Даёт WARN, не BLOCK --
                             эвристика на даунсемплированном облаке, не
                             жёсткая гарантия.
"""

from __future__ import annotations

from datetime import datetime

from neftekod_mas.optimization.joint_envelope import JointEnvelopeChecker
from neftekod_mas.schemas import ControlCandidate, GuardCheck, GuardReport, GuardVerdict, ProcessState, RiskClass
from neftekod_mas.tags.pid_graph import TagGraph

JOINT_ENVELOPE_WARN_DISTANCE = 0.5  # нормализованные единицы -- явное допущение, не откалибровано


class Guard:
    def __init__(
        self,
        tag_graph: TagGraph,
        control_variables_cfg: dict,
        control_bounds: dict,
        hard_constraints: dict | None = None,
        joint_envelope: JointEnvelopeChecker | None = None,
    ):
        self.tag_graph = tag_graph
        self.known_variables = {
            var["name"]: var
            for block in control_variables_cfg.get("installations", {}).values()
            for var in block.get("variables", [])
        }
        self.bounds = {k: v for k, v in control_bounds.items() if k != "_meta"}
        self.hard_constraints = (hard_constraints or {}).get("product_diesel", {})
        self.joint_envelope = joint_envelope

    def review(
        self, candidate: ControlCandidate | None, decision_at: datetime, state: ProcessState | None = None
    ) -> GuardReport:
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
                    check_name="tag_exists", verdict=GuardVerdict.BLOCK,
                    detail=f"Тег {action.tag} отсутствует в цифровом P&ID",
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
            else:
                checks.append(GuardCheck(check_name="within_bounds", verdict=GuardVerdict.BLOCK,
                    detail=f"Для {action.tag} не задан диапазон"))

            if action.tag and self.tag_graph.tag_exists(action.tag):
                downstream = self.tag_graph.downstream_impact(action.tag)
                checks.append(GuardCheck(
                    check_name="downstream_impact", verdict=GuardVerdict.PASS,
                    detail=f"Затронуто (грубая топология стадий): {len(downstream)} тегов ниже по потоку",
                ))

        checks.append(GuardCheck(check_name="blend_sum_100", verdict=GuardVerdict.NOT_APPLICABLE,
            detail="Блендинг не участвует в кандидате"))
        by_metric = {estimate.metric: estimate for estimate in candidate.predicted_quality}
        for metric, cfg in self.hard_constraints.items():
            if cfg.get("limit") is None:
                continue
            estimate = by_metric.get(metric)
            if estimate is None:
                checks.append(GuardCheck(check_name=f"quality_{metric}", verdict=GuardVerdict.BLOCK,
                    detail=f"Нет прогноза обязательного показателя {metric} (UNKNOWN)"))
                continue
            error = estimate.typical_error or 0.0
            conservative = estimate.value + error if cfg["op"] == "<=" else estimate.value - error
            limit = float(cfg["limit"])
            passed = conservative <= limit if cfg["op"] == "<=" else conservative >= limit
            checks.append(GuardCheck(check_name=f"quality_{metric}", verdict=GuardVerdict.PASS if passed else GuardVerdict.BLOCK,
                detail=f"консервативный прогноз {conservative:.3g} {cfg['op']} {limit}"))
        if candidate.predicted_risk.risk_class == RiskClass.UNKNOWN:
            checks.append(GuardCheck(check_name="equipment_risk_coverage", verdict=GuardVerdict.BLOCK,
                detail="Недостаточно признаков для оценки риска оборудования"))

        if self.joint_envelope is not None and state is not None:
            current_values = {tag: r.value for tag, r in state.kip.items()}
            for action in candidate.actions:
                if action.tag:
                    current_values[action.tag] = action.recommended_value
            dist = self.joint_envelope.nearest_distance(current_values)
            if dist is None:
                pass  # не все координаты доступны -- проверка не применима, не блокирует
            elif dist > JOINT_ENVELOPE_WARN_DISTANCE:
                checks.append(GuardCheck(
                    check_name="joint_envelope", verdict=GuardVerdict.WARN,
                    detail=f"Совместное состояние активных переменных на расстоянии {dist:.2f} "
                    f"(норм. ед.) от ближайшей исторической точки > порога {JOINT_ENVELOPE_WARN_DISTANCE} -- "
                    "такая КОМБИНАЦИЯ значений в истории не встречалась, хотя каждая переменная "
                    "по отдельности в допустимых пределах (ARCHITECTURE.md §5.3)",
                ))
            else:
                checks.append(GuardCheck(
                    check_name="joint_envelope", verdict=GuardVerdict.PASS,
                    detail=f"Расстояние до ближайшей исторической точки: {dist:.2f}",
                ))

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
