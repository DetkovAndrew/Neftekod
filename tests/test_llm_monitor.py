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
