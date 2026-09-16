"""
Контракты обмена данными между агентами МАС.

Это единственное место, где определён формат сообщений между слоями
(Data & Sync -> Quality/Reliability -> Optimization -> Orchestrator).
Ни один агент не должен принимать/отдавать "сырой словарь" вместо этих
моделей -- это то самое "явное разделение ролей и обмен информацией",
которое требует ТЗ (см. ARCHITECTURE.md §6).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Protocol

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Общие перечисления
# ---------------------------------------------------------------------------


class DataSource(str, Enum):
    LIMS = "lims"
    PAK = "pak"
    VAK_FORMULA = "vak_formula"
    ML_MODEL = "ml_model"
    KIP = "kip"
    LITERATURE_PROXY = "literature_proxy"  # см. quality/literature_proxies.py -- не производственная формула
    PUBLISHED_CORRELATION = "published_correlation"  # см. quality/astm_correlations.py -- валидированный отраслевой стандарт (ASTM)


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REFUSE = "refuse"  # недостаточно данных для рекомендации вообще


class RiskClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardVerdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Data & Sync Agent
# ---------------------------------------------------------------------------


class TagReading(BaseModel):
    """Единичное значение тега, приведённое к каноническим единицам."""

    tag_id: str
    value: float
    unit: str
    timestamp: datetime
    source: DataSource


class LabPointReading(BaseModel):
    """Последнее известное значение лабораторной/поточной точки на момент
    принятия решения -- со ЯВНЫМ возрастом (ARCHITECTURE.md §2, правило 3)."""

    point_id: str  # напр. "AVT.2.D15" -- установка.точка_отбора.показатель
    value: float
    unit: str
    measured_at: datetime
    decision_at: datetime
    source: DataSource

    @property
    def age_minutes(self) -> float:
        return (self.decision_at - self.measured_at).total_seconds() / 60.0


class DataQualityFlag(BaseModel):
    code: str  # stale_lims | stale_pak | missing_kip_tag | stuck_sensor | out_of_range
    tag_or_point: str
    detail: str
    severity: RiskClass


class DataQualityReport(BaseModel):
    decision_at: datetime
    flags: list[DataQualityFlag] = Field(default_factory=list)
    sync_ok: bool

    @property
    def has_blocking_issues(self) -> bool:
        return any(f.severity in (RiskClass.HIGH, RiskClass.CRITICAL) for f in self.flags)


class ProcessState(BaseModel):
    """Синхронизированный снимок состояния процесса на момент decision_at."""

    decision_at: datetime
    kip: dict[str, TagReading]
    lab_points: dict[str, LabPointReading]
    quality_report: DataQualityReport


# ---------------------------------------------------------------------------
# Агент качества
# ---------------------------------------------------------------------------


class QualityMetricEstimate(BaseModel):
    metric: str  # sulfur_mg_kg | t95_c | cetane_number | cfpp_c | density_kg_m3
    value: float
    unit: str
    source: DataSource
    age_minutes: Optional[float] = None
    confidence: ConfidenceLevel
    typical_error: Optional[float] = None  # MAE формулы по бэктесту против ЛИМС, см. vak_formula_accuracy.yaml


class SpecViolationRisk(BaseModel):
    metric: str
    limit: float
    margin: float  # limit - value (в единицах метрики); отрицательное = нарушение
    risk_class: RiskClass


class QualityAssessment(BaseModel):
    decision_at: datetime
    current: list[QualityMetricEstimate]
    violations: list[SpecViolationRisk] = Field(default_factory=list)
    overall_confidence: ConfidenceLevel


# ---------------------------------------------------------------------------
# Агент надёжности
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    tag_id: str
    description: str
    contribution: float  # 0..1, вклад в итоговый индекс тяжести режима
    is_assumption: bool
    assumption_note: Optional[str] = None


class EquipmentRiskAssessment(BaseModel):
    decision_at: datetime
    severity_index: float  # 0..1
    risk_class: RiskClass
    factors: list[RiskFactor]
    hard_stop: bool  # True -> Агент оптимизации обязан исключить варианты, ухудшающие режим


# ---------------------------------------------------------------------------
# Агент оптимизации
# ---------------------------------------------------------------------------


class ControlAction(BaseModel):
    variable_name: str  # из config/control_variables.yaml
    tag: Optional[str] = None
    current_value: float
    recommended_value: float
    unit: str


class ControlCandidate(BaseModel):
    candidate_id: str
    actions: list[ControlAction]
    predicted_quality: list[QualityMetricEstimate]
    predicted_risk: EquipmentRiskAssessment
    throughput_proxy: Optional[float] = None
    energy_cost_proxy: Optional[float] = None
    feasible: bool
    rejection_reason: Optional[str] = None
    caveats: list[str] = Field(default_factory=list)
    score: Optional[float] = None


class OptimizationResult(BaseModel):
    decision_at: datetime
    candidates_evaluated: int
    feasible_candidates: list[ControlCandidate]
    pareto_front_ids: list[str] = Field(default_factory=list)
    recommended_candidate_id: Optional[str] = None
    no_feasible_solution: bool = False
    no_feasible_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Guard (P&ID-grounded независимая проверка, см. ARCHITECTURE.md §6.5, §9)
# ---------------------------------------------------------------------------


class GuardCheck(BaseModel):
    check_name: str  # tag_exists | actuatable | within_bounds | blend_sum_100 | downstream_impact
    verdict: GuardVerdict
    detail: str


class GuardReport(BaseModel):
    decision_at: datetime
    candidate_id: Optional[str]
    checks: list[GuardCheck]
    final_verdict: GuardVerdict


# ---------------------------------------------------------------------------
# Итоговая рекомендация оператору (формат ровно по ТЗ п.5)
# ---------------------------------------------------------------------------


class Recommendation(BaseModel):
    decision_at: datetime
    key_state: dict[str, float]
    problem_or_risk: str
    proposed_actions: list[ControlAction]
    expected_effect: dict[str, float]
    constraints_checked: list[GuardCheck]
    confidence: ConfidenceLevel
    confidence_warnings: list[str]
    explanation: str
    llm_commentary: Optional[str] = None  # см. §9 ARCHITECTURE.md: никогда не влияет на проверки
    is_refusal: bool = False
    alternatives: list[ControlCandidate] = Field(default_factory=list)  # Парето-фронт минус выбранный (ТЗ п.3: "альтернативы")


# ---------------------------------------------------------------------------
# Протокол предиктора качества -- ВАК-формулы и будущая ML-модель
# взаимозаменяемы за этим интерфейсом (ARCHITECTURE.md §5.2)
# ---------------------------------------------------------------------------


class QualityPredictor(Protocol):
    def predict(self, state: ProcessState) -> list[QualityMetricEstimate]: ...
