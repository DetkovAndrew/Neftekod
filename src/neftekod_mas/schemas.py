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
    SOFT_SENSOR = "soft_sensor"  # см. quality/soft_sensors.py -- якорь (анализатор/ВАК) + поправка по прошлым ЛИМС


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REFUSE = "refuse"  # недостаточно данных для рекомендации вообще


class RiskClass(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardVerdict(str, Enum):
    PASS = "pass"
    NOT_APPLICABLE = "not_applicable"
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
    steady_regime: Optional[bool] = None  # data/regime.py; None -- режим не определён (нет T11)


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
    margin: float  # запас до предела в единицах метрики (с учётом op); отрицательное = нарушение
    op: str = "<="  # направление предела из hard_constraints.yaml
    act_at: Optional[float] = None  # действующий порог действия (с учётом сдвига за точность оценки)
    watch_at: Optional[float] = None  # действующий порог наблюдения
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
    # False -- фактор показывается оператору, но НЕ входит в индекс тяжести
    # режима. Так помечен ресурс катализатора: исчерпание запаса по
    # температуре -- повод планировать перегрузку, а не признак того, что
    # режим прямо сейчас тяжёлый. Смешивать эти две вещи в одном числе
    # значило бы останавливать оптимизацию из-за планового события.
    affects_index: bool = True


class EquipmentRiskAssessment(BaseModel):
    decision_at: datetime
    severity_index: float  # 0..1.5; >= 1.0 -- критический режим (reliability_agent)
    risk_class: RiskClass
    factors: list[RiskFactor]
    hard_stop: bool  # True -> Агент оптимизации обязан исключить варианты, ухудшающие режим
    coverage: float = 1.0
    missing_inputs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Агент блендинга (ARCHITECTURE.md §6.6)
# ---------------------------------------------------------------------------


class BlendComponentState(BaseModel):
    """Один компонент дизельного пула на момент решения."""

    key: str
    description: str
    flow_tag: str
    flow_t_h: float
    mass_share_pct: float
    volume_share_pct: float
    density_kg_m3: float
    sulfur_pct_mass: Optional[float] = None  # идентифицирована по истории, см. compute_blend_model.py


class BlendAssessment(BaseModel):
    """Состояние пула: компоненты, доли, качество смеси.

    `share_sum_pct` -- та самая сумма долей из жёстких ограничений ТЗ п.4.
    Она равна 100 по построению (доли считаются от суммарного расхода),
    но выносится в контракт явно, чтобы Guard проверял её независимо,
    а не доверял агенту на слово.
    """

    decision_at: datetime
    components: list[BlendComponentState]
    total_flow_t_h: float
    share_sum_pct: float
    blend_quality: dict[str, float] = Field(default_factory=dict)
    supported_metrics: list[str] = Field(default_factory=list)
    available: bool = True
    unavailable_reason: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


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
    # Экономический эффект в РЕАЛЬНЫХ единицах (т/сут, Гкал/ч, кВт, нм3/ч)
    # и в рублях по ценам-допущениям из config/economics.yaml.
    # None -- Агент экономики не подключён или не смог посчитать
    # (это "нечем измерить", а не "эффекта нет"), см. §6.7.
    economics: Optional[dict] = None


class OptimizationResult(BaseModel):
    decision_at: datetime
    candidates_evaluated: int
    feasible_candidates: list[ControlCandidate]
    pareto_front_ids: list[str] = Field(default_factory=list)
    recommended_candidate_id: Optional[str] = None
    no_feasible_solution: bool = False
    no_feasible_reason: Optional[str] = None
    # Отклонённые варианты с причиной отбраковки. Нужны, чтобы объяснить
    # оператору (и LLM-оркестратору, §9.1) не только что предложено, но и
    # ПОЧЕМУ остальное не подошло -- это прямое требование ТЗ п.5
    # ("чем выбранный вариант лучше допустимых альтернатив").
    rejected_candidates: list[ControlCandidate] = Field(default_factory=list)


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


class AgentMessage(BaseModel):
    """Одно сообщение между агентами за цикл (orchestrator/bus.py) --
    явный протокол обмена, воспроизводимый по журналу."""

    seq: int
    sender: str  # data_sync | quality | reliability | optimization | guard | orchestrator | llm_monitor
    recipient: str
    topic: str  # тип полезной нагрузки (имя pydantic-модели) или событие
    summary: str  # короткое содержание для журнала/карточки


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
    # Экономический эффект в реальных единицах и рублях (ARCHITECTURE.md §6.7).
    # Отделён от expected_effect намеренно: там -- показатели качества и
    # риска, здесь -- натуральные величины и деньги, часть которых опирается
    # на цены-допущения. Оператор должен видеть эту границу.
    economic_effect: Optional[dict] = None
    # Наблюдения о состоянии оборудования, которые НЕ влияют на индекс
    # тяжести режима, но которые оператор должен видеть -- прежде всего
    # остаток ресурса катализатора (ARCHITECTURE.md §6.3.1).
    equipment_notes: list[str] = Field(default_factory=list)
    llm_commentary: Optional[str] = None  # см. §9 ARCHITECTURE.md: никогда не влияет на проверки
    is_refusal: bool = False
    alternatives: list[ControlCandidate] = Field(default_factory=list)  # Парето-фронт минус выбранный (ТЗ п.3: "альтернативы")
    llm_status: Optional[str] = None  # ok | rejected: <причина> | unavailable: <причина> | None (Monitor выключен)
    # Кто вёл цикл: "deterministic" -- обычный Оркестратор; "llm" -- цикл
    # провела локальная модель через tool-calling (ARCHITECTURE.md §9.1).
    # Даже при "llm" все числа, проверки и Guard остаются
    # детерминированными: модель только выбирает из проверенных вариантов.
    orchestration_mode: str = "deterministic"
    trace: list[AgentMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Протокол предиктора качества -- ВАК-формулы и будущая ML-модель
# взаимозаменяемы за этим интерфейсом (ARCHITECTURE.md §5.2)
# ---------------------------------------------------------------------------


class QualityPredictor(Protocol):
    def predict(self, state: ProcessState) -> list[QualityMetricEstimate]: ...


# Имена показателей для оператора -- в карточке не должно быть программных ключей.
METRIC_NAMES = {
    "sulfur_mg_kg": "сера",
    "t95_c": "T95",
    "cetane_number": "цетановое число",
    "cfpp_c": "CFPP",
    "density_kg_m3": "плотность",
}


def metric_name(metric: str) -> str:
    return METRIC_NAMES.get(metric, metric)
