"""
Синхронизация КИП/ЛИМС/ПАК по времени в единый ProcessState
(ARCHITECTURE.md §2, §6.1). Реализует "Data & Sync Agent".

Никакого перемешивания строк, никакого сопоставления по номеру строки --
только backward as-of join по `measured_at <= decision_at` и явное
хранение возраста (ТЗ, "Правила работы с источниками качества").
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from neftekod_mas.data.regime import regime_at
from neftekod_mas.schemas import (
    DataQualityFlag,
    DataQualityReport,
    DataSource,
    LabPointReading,
    ProcessState,
    RiskClass,
    TagReading,
)
from neftekod_mas.tags.pid_graph import qualify

# Пороги свежести -- явные допущения (ARCHITECTURE.md §2, правило 6).
# Паспортных регламентов на периодичность отбора проб нам не передали,
# поэтому берётся консервативная эвристика: если следующий ожидаемый
# анализ пропущен больше чем в 2 раза относительно типичного интервала
# отбора для данной точки, значение считается устаревшим.
DEFAULT_STALE_LIMS_MINUTES = 24 * 60  # 24 часа
DEFAULT_STALE_PAK_MINUTES = 60  # 1 час -- ПАК поточный, должен обновляться часто
STUCK_WINDOW_SAMPLES = 6  # 6 x 10 мин = 1 час подряд одинаковых значений -> подозрение на "залипший" датчик


def _kip_snapshot(
    kip_df: pd.DataFrame,
    installation: str,
    decision_at: datetime,
    tolerance: timedelta,
    kip_bounds: dict[str, dict] | None = None,
) -> tuple[dict[str, TagReading], list[DataQualityFlag]]:
    flags: list[DataQualityFlag] = []
    readings: dict[str, TagReading] = {}

    idx = kip_df.index
    pos = idx.searchsorted(decision_at, side="right") - 1
    if pos < 0 or (decision_at - idx[pos]) > tolerance:
        flags.append(
            DataQualityFlag(
                code="missing_kip_snapshot",
                tag_or_point=installation,
                detail=f"Нет точки КИП для {installation} в пределах допуска {tolerance} от {decision_at}",
                severity=RiskClass.CRITICAL,
            )
        )
        return readings, flags

    row = kip_df.iloc[pos]
    row_ts = idx[pos]

    for tag_id, value in row.items():
        qid = qualify(installation, tag_id)
        if pd.isna(value):
            flags.append(
                DataQualityFlag(
                    code="missing_kip_tag",
                    tag_or_point=qid,
                    detail=f"NaN на {row_ts}",
                    severity=RiskClass.MEDIUM,
                )
            )
            continue
        readings[qid] = TagReading(
            tag_id=qid, value=float(value), unit="", timestamp=row_ts, source=DataSource.KIP
        )
        if kip_bounds is not None and qid in kip_bounds:
            b = kip_bounds[qid]
            if not (b["low"] <= value <= b["high"]):
                flags.append(
                    DataQualityFlag(
                        code="out_of_range",
                        tag_or_point=qid,
                        detail=f"{value:.4g} вне правдоподобного диапазона [{b['low']}, {b['high']}] "
                        "(0.5-99.5 перцентиль истории + запас, config/kip_bounds.json)",
                        severity=RiskClass.HIGH,
                    )
                )

    # Детектор "залипшего" датчика -- см. AAE (Schall 2026) TrendAnalyzer:
    # если последние STUCK_WINDOW_SAMPLES значений тега идентичны, это
    # подозрительно (прямой физический сигнал редко стоит на месте час подряд).
    window = kip_df.iloc[max(0, pos - STUCK_WINDOW_SAMPLES + 1) : pos + 1]
    if len(window) == STUCK_WINDOW_SAMPLES:
        for tag_id in window.columns:
            series = window[tag_id]
            if series.notna().all() and series.nunique() == 1:
                flags.append(
                    DataQualityFlag(
                        code="stuck_sensor",
                        tag_or_point=qualify(installation, tag_id),
                        detail=f"Последние {STUCK_WINDOW_SAMPLES} значений идентичны ({series.iloc[0]:.4g})",
                        severity=RiskClass.MEDIUM,
                    )
                )

    return readings, flags


def _latest_lab_point(
    df: pd.DataFrame,
    decision_at: datetime,
    stale_minutes: int,
    source: DataSource,
    id_cols: list[str],
) -> tuple[dict[str, LabPointReading], list[DataQualityFlag]]:
    readings: dict[str, LabPointReading] = {}
    flags: list[DataQualityFlag] = []

    past = df[df["measured_at"] <= decision_at]
    if past.empty:
        return readings, flags

    latest = past.sort_values("measured_at").groupby(id_cols, as_index=False).tail(1)

    for _, row in latest.iterrows():
        point_id = "|".join(str(row[c]) for c in id_cols)
        measured_at = row["measured_at"].to_pydatetime()
        reading = LabPointReading(
            point_id=point_id,
            value=float(row["value"]),
            unit=str(row.get("unit", "")),
            measured_at=measured_at,
            decision_at=decision_at,
            source=source,
        )
        readings[point_id] = reading
        if reading.age_minutes > stale_minutes:
            flags.append(
                DataQualityFlag(
                    code=f"stale_{source.value}",
                    tag_or_point=point_id,
                    detail=f"Возраст {reading.age_minutes:.0f} мин > порога {stale_minutes} мин",
                    severity=RiskClass.HIGH if source == DataSource.LIMS else RiskClass.MEDIUM,
                )
            )

    return readings, flags


def build_process_state(
    decision_at: datetime,
    avt_kip: pd.DataFrame,
    ht_kip: pd.DataFrame,
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
    kip_tolerance: timedelta = timedelta(minutes=15),
    stale_lims_minutes: int = DEFAULT_STALE_LIMS_MINUTES,
    stale_pak_minutes: int = DEFAULT_STALE_PAK_MINUTES,
    kip_bounds: dict[str, dict] | None = None,
) -> ProcessState:
    kip: dict[str, TagReading] = {}
    flags: list[DataQualityFlag] = []

    avt_readings, avt_flags = _kip_snapshot(avt_kip, "avt", decision_at, kip_tolerance, kip_bounds)
    ht_readings, ht_flags = _kip_snapshot(ht_kip, "242000", decision_at, kip_tolerance, kip_bounds)
    kip.update(avt_readings)
    kip.update(ht_readings)
    flags.extend(avt_flags)
    flags.extend(ht_flags)

    lims_readings, lims_flags = _latest_lab_point(
        lims_long, decision_at, stale_lims_minutes, DataSource.LIMS, ["point_label", "param"]
    )
    pak_readings, pak_flags = _latest_lab_point(
        pak_long, decision_at, stale_pak_minutes, DataSource.PAK, ["param"]
    )
    flags.extend(lims_flags)
    flags.extend(pak_flags)

    lab_points: dict[str, LabPointReading] = {**lims_readings, **pak_readings}

    # Переходный режим реактора (пуск/останов/разгон) -- вне области
    # применимости soft-sensor'ов и оптимизации (data/regime.py); флаг
    # блокирующий, Оркестратор по нему отказывается от рекомендации.
    regime = regime_at(ht_kip, decision_at, tolerance=pd.Timedelta(kip_tolerance))
    if regime.steady is False:
        flags.append(
            DataQualityFlag(
                code="transient_regime",
                tag_or_point=qualify("242000", "T11"),
                detail="Реактор не в стационарном режиме: " + regime.detail,
                severity=RiskClass.HIGH,
            )
        )

    sync_ok = not any(f.code == "missing_kip_snapshot" for f in flags)

    report = DataQualityReport(decision_at=decision_at, flags=flags, sync_ok=sync_ok)
    return ProcessState(
        decision_at=decision_at, kip=kip, lab_points=lab_points, quality_report=report,
        steady_regime=regime.steady,
    )
