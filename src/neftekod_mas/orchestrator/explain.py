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
    equipment = f"индекс тяжести режима оборудования {risk.severity_index:.2f} ({risk.risk_class.value})"
    watch = [v for v in quality.violations if v.risk_class.value == "medium"]
    if watch:
        return (
            "Управляющих действий не требуется: ни один показатель не достиг порога действия, "
            f"{equipment}. Под наблюдением: "
            + "; ".join(describe_violation(quality, v) for v in watch) + "."
        )
    return (
        "Режим стабилен: все оценённые показатели качества в норме "
        f"(наиболее близкий к пределу -- {_worst_violation(quality)}), {equipment}. "
        "Лишних управляющих действий не требуется."
    )


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
    return describe_violation(quality, worst_violation(quality))


_RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def worst_violation(quality: QualityAssessment):
    """Сначала по классу риска, затем по запасу в долях предела -- запасы
    разных показателей в своих единицах (мг/кг, °C, ед.) напрямую несравнимы."""
    return min(
        quality.violations,
        key=lambda v: (_RISK_ORDER[v.risk_class.value], v.margin / max(abs(v.limit), 1e-9)),
    )


RISK_WORDS = {
    "low": "в норме",
    "medium": "близко к пределу, под наблюдением",
    "high": "вплотную к пределу",
    "critical": "за пределом",
}


def describe_violation(quality: QualityAssessment, v) -> str:
    """"sulfur_mg_kg = 8.9 -- вплотную к пределу <= 10 (запас 1.1; порог действия 8.5)" --
    значение, предел, запас и порог в единицах показателя, без процентов
    (вероятности превышения на этих данных плохо откалиброваны, MODEL_AUDIT.md)."""
    est = next((e for e in quality.current if e.metric == v.metric), None)
    value = f"{v.metric} = {est.value:.4g}" if est is not None else v.metric
    gap = f"превышение на {-v.margin:.3g}" if v.margin < 0 else f"запас {v.margin:.3g}"
    threshold = ""
    if v.margin >= 0 and v.act_at is not None:
        threshold = f"; порог действия {v.act_at:.4g}"
        if v.risk_class.value == "low":
            threshold += f", наблюдения {v.watch_at:.4g}"
    return f"{value} -- {RISK_WORDS[v.risk_class.value]} {v.op} {v.limit:.4g} ({gap}{threshold})"
