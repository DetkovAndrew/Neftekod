from neftekod_mas.orchestrator.llm_monitor import ungrounded_items

FACTS = (
    "Текущее состояние: sulfur_mg_kg=9.534; t95_c=341.2\n"
    "Действие: ht_inlet_temp_c [242000:T5] 365 -> 370.5 °C\n"
    "Проверка within_bounds: pass (370.5 в [350, 385])"
)


def test_rounded_numbers_from_card_are_grounded():
    assert ungrounded_items("Сера 9.5 мг/кг, T95 около 341 °C; поднять T5 с 365 до 370,5.", FACTS) == []


def test_invented_number_is_rejected():
    assert ungrounded_items("Сера снизится до 7.2 мг/кг.", FACTS) == ["7.2"]


def test_rounding_must_match_precision_used():
    # 9.6 -- не округление 9.534 до одного знака
    assert ungrounded_items("сера 9.6", FACTS) == ["9.6"]


def test_unknown_tag_is_rejected_but_quality_names_allowed():
    assert ungrounded_items("Смотрите на P13 и T95.", FACTS) == ["P13"]
    assert ungrounded_items("Температура avt:T5 в норме", FACTS) == []


def test_negative_number_must_keep_sign():
    facts = "Проблема: сера = 19.37 при пределе <= 10 (превышение на 9.37); cfpp = -8"
    assert ungrounded_items("превышение на 9.37, CFPP -8 °C", facts) == []
    assert ungrounded_items("критерий -19.37", facts) == ["-19.37"]


def test_card_facts_translate_flag_codes_and_add_tag_meaning_from_catalog():
    from datetime import datetime
    from neftekod_mas.orchestrator.llm_monitor import card_facts
    from neftekod_mas.schemas import ConfidenceLevel, Recommendation

    rec = Recommendation(
        decision_at=datetime(2024, 1, 1), key_state={}, problem_or_risk="Риска не обнаружено.",
        proposed_actions=[], expected_effect={}, constraints_checked=[], confidence=ConfidenceLevel.LOW,
        confidence_warnings=["stuck_sensor: avt:T39 -- Последние 6 значений идентичны (99.57)"],
        explanation="-", is_refusal=False,
    )
    facts = card_facts(rec, {"avt:T39": "Температура в К-10 над секцией насадки 1а"})
    assert "stuck_sensor" not in facts
    assert "возможно, завис" in facts
    assert "avt:T39 (Температура в К-10 над секцией насадки 1а)" in facts


def test_monitor_rejects_answer_that_switches_to_another_language():
    from datetime import datetime
    from neftekod_mas.orchestrator.llm_monitor import LLMMonitor
    from neftekod_mas.schemas import ConfidenceLevel, Recommendation

    rec = Recommendation(
        decision_at=datetime(2024, 1, 1), key_state={}, problem_or_risk="Риска не обнаружено.",
        proposed_actions=[], expected_effect={}, constraints_checked=[], confidence=ConfidenceLevel.LOW,
        confidence_warnings=[], explanation="-", is_refusal=False,
    )
    mixed = "Управляющих действий не требуется, показатели в норме. Давление в некоторых设备无法继续生成俄语内容"
    good = "Управляющих действий не требуется: все показатели качества в норме, риск оборудования низкий."
    answers = iter([mixed, mixed])
    out = LLMMonitor(lambda s, u: next(answers)).annotate(rec)
    assert out.llm_commentary is None and "не на русском" in out.llm_status
    answers = iter([mixed, good])
    assert LLMMonitor(lambda s, u: next(answers)).annotate(rec).llm_commentary == good
