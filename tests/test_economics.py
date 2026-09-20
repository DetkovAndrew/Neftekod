"""Тесты экономики в реальных единицах (ARCHITECTURE.md §6.7).

Главное, что здесь проверяется, -- что физика отделена от цен и что
"нечем измерить" никогда не подменяется нулём.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from neftekod_mas.economics.economics_agent import EconomicsAgent
from neftekod_mas.economics.thermo import (
    compressor_power_kw,
    heating_duty_gcal_h,
    liquid_heat_capacity_kj_kg_k,
    specific_gravity,
    watson_k,
)
from neftekod_mas.schemas import (
    ControlAction,
    ControlCandidate,
    DataQualityReport,
    DataSource,
    EquipmentRiskAssessment,
    ProcessState,
    RiskClass,
    TagReading,
)

CFG = {
    "prices": {
        "diesel_rub_per_t": {"value": 60000},
        "fuel_gas_rub_per_gcal": {"value": 1250},
        "electricity_rub_per_kwh": {"value": 6.5},
        "hydrogen_rub_per_nm3": {"value": 20.0},
    },
    "physical_constants": {
        "furnace_efficiency": {"value": 0.9},
        "compressor_polytropic_efficiency": {"value": 0.75},
        "recycle_gas_k_ratio": {"value": 1.38},
        "compressor_suction_temp_c": {"value": 40.0},
    },
    "streams": {
        "product_diesel": {"mass_flow_tag": "242000:F17"},
        "ht_feed": {"mass_flow_tag": "242000:F9", "typical_density_kg_m3": 845.0,
                    "typical_mean_boiling_point_c": 280.0},
        "ht_makeup_hydrogen": {"flow_tag": "242000:F25"},
        "ht_recycle_gas": {"flow_tag": "242000:F2", "suction_pressure_tag": "242000:P3",
                           "discharge_pressure_tag": "242000:P13"},
        "avt_vacuum_furnace_p3": {"mass_flow_tag": "avt:F31", "inlet_temp_tag": "avt:T33",
                                  "outlet_temp_tag": "avt:T55", "typical_density_kg_m3": 930.0,
                                  "typical_mean_boiling_point_c": 450.0},
    },
}

AT = datetime(2025, 12, 1, 20, 0, 0)
BASE_TAGS = {
    "242000:F17": 245.4, "242000:F9": 207.8, "242000:F25": 13210.9,
    "242000:F2": 92112.0, "242000:P3": 3.46, "242000:P13": 3.75, "242000:T5": 376.3,
    "avt:F31": 524.7, "avt:T33": 338.3, "avt:T55": 381.5,
}


def _state(**overrides) -> ProcessState:
    tags = {**BASE_TAGS, **overrides}
    kip = {k: TagReading(tag_id=k, value=v, unit="", timestamp=AT, source=DataSource.KIP)
           for k, v in tags.items() if v is not None}
    return ProcessState(decision_at=AT, kip=kip, lab_points={},
                        quality_report=DataQualityReport(decision_at=AT, sync_ok=True))


def _candidate(actions) -> ControlCandidate:
    return ControlCandidate(
        candidate_id="c1", actions=actions, predicted_quality=[],
        predicted_risk=EquipmentRiskAssessment(decision_at=AT, severity_index=0.0,
                                               risk_class=RiskClass.LOW, factors=[], hard_stop=False),
        feasible=True,
    )


# --- теплофизика ------------------------------------------------------

def test_specific_gravity_and_watson_k_are_typical_for_diesel():
    assert specific_gravity(845.0) == pytest.approx(0.846, abs=0.002)
    # Kw 11-12 -- парафинистое дизельное топливо; выход за этот диапазон
    # означал бы ошибку в единицах измерения температуры кипения.
    assert 11.0 < watson_k(280.0, 845.0) < 12.5


def test_heat_capacity_grows_with_temperature_and_is_physical():
    cold = liquid_heat_capacity_kj_kg_k(25.0, 845.0, 280.0)
    hot = liquid_heat_capacity_kj_kg_k(350.0, 845.0, 280.0)
    assert 1.8 < cold < 2.2, "Cp дизтоплива при 25 °C ~2 кДж/(кг·К)"
    assert hot > cold


def test_heating_duty_is_linear_in_flow_and_delta_t():
    a = heating_duty_gcal_h(100.0, 10.0, 845.0, 280.0, 370.0)
    assert heating_duty_gcal_h(200.0, 10.0, 845.0, 280.0, 370.0) == pytest.approx(2 * a)
    assert heating_duty_gcal_h(100.0, 20.0, 845.0, 280.0, 370.0) == pytest.approx(2 * a)


def test_compressor_power_is_zero_without_compression():
    assert compressor_power_kw(92112.0, 3.75, 3.75) == 0.0
    assert compressor_power_kw(92112.0, 3.46, 3.75) > 0.0


# --- агент ------------------------------------------------------------

def test_baseline_is_in_real_units_and_physically_sane():
    b = EconomicsAgent(CFG).energy_breakdown(_state())
    assert b.product_rate_t_h == pytest.approx(245.4)
    assert 5.0 < b.furnace_p3_duty_gcal_h < 60.0, "Гкал/ч печи АВТ -- десятки, не тысячи"
    assert 50.0 < b.recycle_compressor_kw < 5000.0


def test_missing_tag_is_reported_not_silently_zeroed():
    b = EconomicsAgent(CFG).energy_breakdown(_state(**{"242000:F17": None}))
    assert b.product_rate_t_h is None
    assert any("F17" in n for n in b.notes)


def test_raising_reactor_temperature_costs_fuel_gas():
    agent = EconomicsAgent(CFG)
    state = _state()
    candidate = _candidate([ControlAction(variable_name="ht_inlet_temp_c", tag="242000:T5",
                                          current_value=376.3, recommended_value=380.2, unit="°C")])
    effect = agent.candidate_effect(state, state, candidate)
    assert effect["delta_physical"]["ht_feed_heating_gcal_h"] > 0
    assert effect["delta_rub_per_day"]["fuel_gas"] < 0, "нагрев -- это затраты, а не доход"
    assert effect["net_rub_per_day"] < 0
    assert effect["prices_are_assumptions"] is True


def test_lever_without_energy_model_reports_nothing_rather_than_zero():
    """Для рычага, у которого нет модели энергозатрат, приращение должно
    ОТСУТСТВОВАТЬ. Ноль означал бы "посчитали, эффекта нет" -- это
    другое утверждение, и оно было бы неправдой."""
    agent = EconomicsAgent(CFG)
    state = _state()
    candidate = _candidate([ControlAction(variable_name="ht_pressure_mpa", tag="242000:P13",
                                          current_value=3.75, recommended_value=3.8, unit="МПа")])
    effect = agent.candidate_effect(state, state, candidate)
    assert "ht_feed_heating_gcal_h" not in effect["delta_physical"]


def test_physics_does_not_depend_on_prices():
    """Замена всех цен на нули не должна менять ни одной натуральной
    величины -- это и есть отделение измерения от допущения."""
    zero_prices = {**CFG, "prices": {k: {"value": 0.0} for k in CFG["prices"]}}
    state = _state()
    candidate = _candidate([ControlAction(variable_name="ht_inlet_temp_c", tag="242000:T5",
                                          current_value=376.3, recommended_value=380.2, unit="°C")])
    a = EconomicsAgent(CFG).candidate_effect(state, state, candidate)["delta_physical"]
    b = EconomicsAgent(zero_prices).candidate_effect(state, state, candidate)["delta_physical"]
    assert a == b


def test_agent_is_unavailable_without_config():
    assert EconomicsAgent({}).available is False
