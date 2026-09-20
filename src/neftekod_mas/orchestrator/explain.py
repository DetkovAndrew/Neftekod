"""
Шаблонная генерация объяснения для карточки рекомендации
(ARCHITECTURE.md §9, §10). Это ОСНОВНОЙ, обязательный путь объяснения --
работает без каких-либо внешних зависимостей и без LLM, демонстрация
обязана проходить именно на нём. Опциональный локальный LLM-Monitor
(см. llm_monitor.py, не подключён по умолчанию) может ПОВЕРХ этого текста
добавить `llm_commentary`, но не заменяет и не проверяет его.
"""

from __future__ import annotations

from neftekod_mas.schemas import ControlCandidate, EquipmentRiskAssessment, QualityAssessment, metric_name


def explain_no_action(quality: QualityAssessment, risk: EquipmentRiskAssessment) -> str:
    equipment = f"индекс тяжести режима оборудования {risk.severity_index:.2f} ({risk.risk_class.value})"
    watch = [v for v in quality.violations if v.risk_class.value == "medium"]
    if watch:
        return (
            "Управляющих действий не требуется: ни один показатель не достиг порога действия, "
            f"{equipment}. Под наблюдением: "
            + "; ".join(describe_violation(quality, v) for v in watch) + "."
        )
    return (
        "Режим стабилен: все оценённые показатели качества в норме, ни один не достиг "
        f"порога наблюдения (меньше всего запас у показателя: {_worst_violation(quality)}), {equipment}. "
        "Лишних управляющих действий не требуется."
    )


def explain_refusal(reason: str) -> str:
    return f"Надёжной рекомендации нет: {reason}"


def explain_recommendation(
    candidate: ControlCandidate,
    quality: QualityAssessment,
    risk_before: EquipmentRiskAssessment,
    feasible: list[ControlCandidate] | None = None,
) -> str:
    action = candidate.actions[0]
    direction = "увеличить" if action.recommended_value > action.current_value else "снизить"
    parts = [
        f"Обнаружен риск: {_worst_violation(quality)}.",
        f"Предлагается {direction} {action.variable_name} ({action.tag}) с "
        f"{action.current_value:.3g} до {action.recommended_value:.3g} {action.unit}.",
    ]
    if candidate.predicted_quality:
        eff = "; ".join(f"{metric_name(e.metric)} = {e.value:.3g} {e.unit}".rstrip() for e in candidate.predicted_quality)
        parts.append(f"Ожидаемый эффект на прогнозируемые показатели: {eff}.")
    parts.append(
        f"Риск оборудования: {risk_before.severity_index:.2f} -> "
        f"{candidate.predicted_risk.severity_index:.2f} ({candidate.predicted_risk.risk_class.value})."
    )
    parts.append(_why_chosen(candidate, feasible or []))
    if candidate.caveats:
        parts.append("Ограничения прогноза: " + " ".join(candidate.caveats))
    return " ".join(parts)


def _why_chosen(candidate: ControlCandidate, feasible: list[ControlCandidate]) -> str:
    """ТЗ п.5, блок "Объяснение": чем выбранный вариант лучше допустимых альтернатив."""
    others = [c for c in feasible if c.candidate_id != candidate.candidate_id]
    if not others:
        return "Других допустимых вариантов нет."
    runner = max(others, key=lambda c: c.score if c.score is not None else float("-inf"))
    a = runner.actions[0]
    diffs = []
    mine = {e.metric: e.value for e in candidate.predicted_quality}
    for e in runner.predicted_quality:
        if e.metric in mine and abs(e.value - mine[e.metric]) > 1e-6:
            diffs.append(f"{metric_name(e.metric)} {e.value:.3g} против {mine[e.metric]:.3g}")
    if runner.predicted_risk.severity_index != candidate.predicted_risk.severity_index:
        diffs.append(
            f"индекс тяжести {runner.predicted_risk.severity_index:.2f} против "
            f"{candidate.predicted_risk.severity_index:.2f}"
        )
    step_r, step_c = abs(a.recommended_value - a.current_value), abs(
        candidate.actions[0].recommended_value - candidate.actions[0].current_value)
    same_var = a.tag == candidate.actions[0].tag
    if same_var and abs(step_r - step_c) > 1e-9:
        diffs.append(f"сдвиг уставки {step_r:.3g} против {step_c:.3g} {a.unit}".rstrip())
    return (
        f"Выбран лучший по взвешенному критерию (запас по качеству, риск оборудования, "
        f"величина изменения) из {len(feasible)} допустимых вариантов. Ближайшая альтернатива -- "
        f"{a.variable_name} ({a.tag}) до {a.recommended_value:.4g} {a.unit}"
        + (f": {'; '.join(diffs)}." if diffs else ".")
    )


def _worst_violation(quality: QualityAssessment) -> str:
    if not quality.violations:
        return "нарушений не обнаружено"
    return describe_violation(quality, worst_violation(quality))


_RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def worst_violation(quality: QualityAssessment):
    """Сначала по классу риска, затем по запасу в долях полосы "порог действия ->
    норматив" (запасы в мг/кг, °C, ед. напрямую несравнимы, а доля от самого
    норматива занижает T95: 3.7 °C из 360 -- это 1%, но больше половины полосы)."""
    def scale(v) -> float:
        band = abs(v.limit - v.act_at) if v.act_at is not None else 0.0
        return band if band > 1e-9 else max(abs(v.limit), 1e-9)

    return min(quality.violations, key=lambda v: (_RISK_ORDER[v.risk_class.value], v.margin / scale(v)))


RISK_WORDS = {
    "low": "в норме",
    "medium": "близко к пределу, под наблюдением",
    "high": "вплотную к пределу",
    "critical": "за пределом",
}


def describe_violation(quality: QualityAssessment, v) -> str:
    """"сера = 8.9 мг/кг -- вплотную к пределу (норматив <= 10, запас 1.1; порог
    действия системы 8.5)" -- значение, норматив, запас и порог в единицах
    показателя, без процентов (вероятности превышения на этих данных плохо
    откалиброваны, MODEL_AUDIT.md). Норматив и внутренние пороги системы
    названы явно по-разному, чтобы их не путали ни оператор, ни LLM Monitor."""
    est = next((e for e in quality.current if e.metric == v.metric), None)
    name = metric_name(v.metric)
    value = f"{name} = {est.value:.4g} {est.unit}".rstrip() if est is not None else name
    if v.margin >= 0:
        gap = f"запас {v.margin:.3g}"
    else:  # для нижней границы (цетановое число >= 51) это не "превышение"
        gap = f"{'выше' if v.op == '<=' else 'ниже'} норматива на {-v.margin:.3g}"
    threshold = ""
    if v.margin >= 0 and v.act_at is not None:
        if v.risk_class.value == "low":
            threshold = f"; пороги системы: наблюдения {v.watch_at:.4g}, действия {v.act_at:.4g}"
        else:
            threshold = f"; порог действия системы {v.act_at:.4g}"
    return f"{value} -- {RISK_WORDS[v.risk_class.value]} (норматив {v.op} {v.limit:.4g}, {gap}{threshold})"
