"""
Шаблонная генерация объяснения для карточки рекомендации
(ARCHITECTURE.md §9, §10). Это ОСНОВНОЙ, обязательный путь объяснения --
работает без каких-либо внешних зависимостей и без LLM, демонстрация
обязана проходить именно на нём. Опциональный локальный LLM-Monitor
(см. llm_monitor.py, не подключён по умолчанию) может ПОВЕРХ этого текста
добавить `llm_commentary`, но не заменяет и не проверяет его.
"""

from __future__ import annotations

from neftekod_mas.schemas import ControlCandidate, EquipmentRiskAssessment, QualityAssessment


def explain_no_action(quality: QualityAssessment, risk: EquipmentRiskAssessment) -> str:
    text = (
        "Режим стабилен: все оценённые показатели качества укладываются в допуски "
        f"с запасом (наихудший риск нарушения -- {_worst_violation(quality)}), индекс тяжести "
        f"режима {risk.severity_index:.2f} ({risk.risk_class.value}). Лишних управляющих "
        "действий не требуется."
    )
    watch = [v for v in quality.violations if v.risk_class.value == "medium"]
    if watch:
        text += " Под наблюдением (работа близко к пределу, действие пока не нужно): " + "; ".join(
            describe_violation(quality, v) for v in watch
        ) + "."
    return text


def explain_refusal(reason: str) -> str:
    return f"Надёжной рекомендации нет: {reason}"


def explain_recommendation(
    candidate: ControlCandidate,
    quality: QualityAssessment,
    risk_before: EquipmentRiskAssessment,
) -> str:
    action = candidate.actions[0]
    direction = "увеличить" if action.recommended_value > action.current_value else "снизить"
    parts = [
        f"Обнаружен риск: {_worst_violation(quality)}.",
        f"Предлагается {direction} {action.variable_name} ({action.tag}) с "
        f"{action.current_value:.3g} до {action.recommended_value:.3g} {action.unit}.",
    ]
    if candidate.predicted_quality:
        eff = "; ".join(f"{e.metric}={e.value:.3g}{e.unit}" for e in candidate.predicted_quality)
        parts.append(f"Ожидаемый эффект на прогнозируемые показатели: {eff}.")
    parts.append(
        f"Риск оборудования: {risk_before.severity_index:.2f} -> "
        f"{candidate.predicted_risk.severity_index:.2f} ({candidate.predicted_risk.risk_class.value})."
    )
    if candidate.caveats:
        parts.append("Ограничения прогноза: " + " ".join(candidate.caveats))
    return " ".join(parts)


def _worst_violation(quality: QualityAssessment) -> str:
    if not quality.violations:
        return "нарушений не обнаружено"
    return describe_violation(quality, min(quality.violations, key=lambda v: v.margin))


def describe_violation(quality: QualityAssessment, v) -> str:
    """"sulfur_mg_kg = 9.97 при пределе <= 10 (запас 0.03, high)" -- значение,
    предел и запас явно, без внутреннего термина margin со знаком."""
    est = next((e for e in quality.current if e.metric == v.metric), None)
    value = f"{v.metric} = {est.value:.4g}" if est is not None else v.metric
    gap = f"превышение на {-v.margin:.3g}" if v.margin < 0 else f"запас {v.margin:.3g}"
    prob = "" if v.exceed_probability is None else f", вероятность превышения {v.exceed_probability:.0%}"
    return f"{value} при пределе {v.op} {v.limit:.4g} ({gap}{prob}, риск {v.risk_class.value})"
