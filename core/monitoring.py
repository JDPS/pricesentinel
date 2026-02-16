# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Monitoring and health summary utilities for daily operations."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core.data_manager import CountryDataManager

logger = logging.getLogger(__name__)

DEFAULT_MONITORING_CONFIG: dict[str, Any] = {
    "status_thresholds": {
        "mae_warn": 20.0,
        "mae_critical": 30.0,
        "drift_warn_pct": 0.20,
        "drift_critical_pct": 0.35,
        "min_coverage_7d": 0.70,
        "min_coverage_30d": 0.80,
    },
    "freshness_hours": {
        "electricity": 48,
        "weather": 72,
        "gas": 96,
    },
}


def _load_monitoring_config(country_code: str) -> dict[str, Any]:
    path = Path("config") / "monitoring" / f"{country_code.upper()}.yaml"
    if not path.exists():
        return dict(DEFAULT_MONITORING_CONFIG)

    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    config = dict(DEFAULT_MONITORING_CONFIG)
    status = dict(DEFAULT_MONITORING_CONFIG["status_thresholds"])
    status.update(loaded.get("status_thresholds", {}))
    config["status_thresholds"] = status

    freshness = dict(DEFAULT_MONITORING_CONFIG["freshness_hours"])
    freshness.update(loaded.get("freshness_hours", {}))
    config["freshness_hours"] = freshness
    return config


def _scorecard_path(manager: CountryDataManager) -> Path:
    return manager.get_processed_path() / "scorecards" / "daily_scorecard.csv"


def _read_scorecard(manager: CountryDataManager) -> pd.DataFrame:
    path = _scorecard_path(manager)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "target_date" in df.columns:
        df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    for col in ("mae", "rmse", "mape", "directional_accuracy", "peak_hour_abs_error"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _window(df: pd.DataFrame, as_of: date, days: int) -> pd.DataFrame:
    if df.empty or "target_date" not in df.columns:
        return pd.DataFrame()
    start = as_of - timedelta(days=days - 1)
    return df[(df["target_date"] >= start) & (df["target_date"] <= as_of)].copy()


def _rolling_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "records": 0,
            "ok_records": 0,
            "coverage": 0.0,
            "mae": None,
            "rmse": None,
            "mape": None,
        }

    ok_df = df[df.get("status") == "ok"] if "status" in df.columns else df
    records = int(len(df))
    ok_records = int(len(ok_df))
    coverage = float(ok_records / records) if records else 0.0

    return {
        "records": records,
        "ok_records": ok_records,
        "coverage": round(coverage, 4),
        "mae": float(ok_df["mae"].mean()) if "mae" in ok_df.columns and not ok_df.empty else None,
        "rmse": (
            float(ok_df["rmse"].mean()) if "rmse" in ok_df.columns and not ok_df.empty else None
        ),
        "mape": (
            float(ok_df["mape"].mean()) if "mape" in ok_df.columns and not ok_df.empty else None
        ),
    }


def _latest_data_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None

    files = [p for p in path.glob("*.csv") if p.name != ".gitkeep"]
    if not files:
        return None

    latest = max(files, key=lambda p: p.stat().st_mtime)
    age_seconds = datetime.now(tz=UTC).timestamp() - latest.stat().st_mtime
    return round(age_seconds / 3600.0, 2)


def _freshness_summary(manager: CountryDataManager, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = dict(config.get("freshness_hours", {}))
    result: dict[str, Any] = {}

    for source, threshold in thresholds.items():
        source_path = manager.get_raw_path(source)
        age_hours = _latest_data_age_hours(source_path)

        if age_hours is None:
            status = "critical"
            reason = "no_data"
        elif age_hours > float(threshold):
            status = "warn"
            reason = "stale"
        else:
            status = "ok"
            reason = "fresh"

        result[source] = {
            "age_hours": age_hours,
            "threshold_hours": float(threshold),
            "status": status,
            "reason": reason,
        }

    return result


def _expected_forecast_exists(manager: CountryDataManager, country_code: str, as_of: date) -> bool:
    next_day = as_of + timedelta(days=1)
    compact = next_day.strftime("%Y%m%d")
    forecast_dir = manager.get_processed_path() / "forecasts"
    if not forecast_dir.exists():
        return False
    return any(forecast_dir.glob(f"{country_code}_forecast_{compact}_*.csv"))


def _derive_alert_status(
    as_of: date,
    summary_7d: dict[str, Any],
    summary_30d: dict[str, Any],
    drift_pct: float | None,
    freshness: dict[str, Any],
    latest_row: pd.Series | None,
    expected_forecast_exists: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = dict(config.get("status_thresholds", {}))
    conditions: list[dict[str, str]] = []

    def add(severity: str, code: str, message: str) -> None:
        conditions.append({"severity": severity, "code": code, "message": message})

    mae_7d = summary_7d.get("mae")
    if isinstance(mae_7d, float):
        if mae_7d > float(thresholds["mae_critical"]):
            add("critical", "mae_spike", f"7d MAE {mae_7d:.3f} > critical threshold")
        elif mae_7d > float(thresholds["mae_warn"]):
            add("warn", "mae_spike", f"7d MAE {mae_7d:.3f} > warn threshold")

    coverage_7d = float(summary_7d.get("coverage") or 0.0)
    if coverage_7d < float(thresholds["min_coverage_7d"]):
        add("warn", "coverage_7d", "7d successful evaluation coverage below threshold")

    coverage_30d = float(summary_30d.get("coverage") or 0.0)
    if coverage_30d < float(thresholds["min_coverage_30d"]):
        add("warn", "coverage_30d", "30d successful evaluation coverage below threshold")

    if drift_pct is not None:
        if drift_pct > float(thresholds["drift_critical_pct"]):
            add("critical", "drift", f"MAE drift {drift_pct:.3f} > critical threshold")
        elif drift_pct > float(thresholds["drift_warn_pct"]):
            add("warn", "drift", f"MAE drift {drift_pct:.3f} > warn threshold")

    for source, source_summary in freshness.items():
        status = source_summary.get("status")
        if status == "critical":
            add("critical", "freshness", f"{source}: no recent data found")
        elif status == "warn":
            add("warn", "freshness", f"{source}: stale raw data")

    if latest_row is not None:
        latest_status = str(latest_row.get("status", ""))
        latest_reason = str(latest_row.get("reason", ""))
        if latest_status != "ok":
            severity = "critical" if latest_reason == "forecast_missing" else "warn"
            add(severity, "latest_eval", f"Latest evaluated day status is '{latest_status}'")

    if not expected_forecast_exists:
        add(
            "warn",
            "next_forecast_missing",
            f"No forecast file found for {(as_of + timedelta(days=1)).isoformat()}",
        )

    if any(c["severity"] == "critical" for c in conditions):
        overall = "critical"
    elif any(c["severity"] == "warn" for c in conditions):
        overall = "warn"
    else:
        overall = "ok"

    return {"overall_status": overall, "conditions": conditions}


def _summary_paths(manager: CountryDataManager, as_of: date) -> tuple[Path, Path, Path]:
    monitor_dir = manager.get_processed_path() / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    stamp = as_of.strftime("%Y%m%d")
    return (
        monitor_dir / f"health_summary_{stamp}.json",
        monitor_dir / f"health_summary_{stamp}.md",
        monitor_dir / f"alerts_{stamp}.log",
    )


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    alerts = summary["alerts"]
    rolling = summary["rolling"]
    freshness = summary["freshness"]

    lines = [
        f"# Health Summary - {summary['country_code']}",
        "",
        f"- As of: {summary['as_of_date']}",
        f"- Generated: {summary['generated_at_utc']}",
        f"- Status: **{alerts['overall_status']}**",
        "",
        "## Rolling Metrics",
        "",
        f"- 7d coverage: {rolling['7d']['coverage']}",
        f"- 7d MAE: {rolling['7d']['mae']}",
        f"- 30d coverage: {rolling['30d']['coverage']}",
        f"- 30d MAE: {rolling['30d']['mae']}",
        f"- MAE drift pct: {summary['drift']['mae_drift_pct']}",
        "",
        "## Freshness",
        "",
    ]

    for source, payload in freshness.items():
        lines.append(
            f"- {source}: status={payload['status']} age_hours={payload['age_hours']} "
            f"threshold={payload['threshold_hours']}"
        )

    lines.extend(["", "## Alert Conditions", ""])
    if not alerts["conditions"]:
        lines.append("- none")
    else:
        for cond in alerts["conditions"]:
            lines.append(f"- [{cond['severity']}] {cond['code']}: {cond['message']}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_health_summary(
    country_code: str,
    as_of_date: str | None = None,
    data_root: str = "data",
) -> dict[str, Any]:
    """
    Generate and persist daily health summary artifacts.

    Returns a dictionary containing summary payload and artifact paths.
    """
    code = country_code.upper()
    as_of = date.fromisoformat(as_of_date) if as_of_date else datetime.now(UTC).date()

    manager = CountryDataManager(code, base_path=data_root)
    config = _load_monitoring_config(code)
    scorecard = _read_scorecard(manager)

    hist = _window(scorecard, as_of, days=3650)
    last_7d = _window(scorecard, as_of, days=7)
    last_30d = _window(scorecard, as_of, days=30)

    summary_7d = _rolling_summary(last_7d)
    summary_30d = _rolling_summary(last_30d)

    mae_7d = summary_7d.get("mae")
    mae_30d = summary_30d.get("mae")
    drift_pct = None
    if isinstance(mae_7d, float) and isinstance(mae_30d, float) and mae_30d > 0:
        drift_pct = round(abs(mae_7d - mae_30d) / mae_30d, 4)

    freshness = _freshness_summary(manager, config)

    latest_row = None
    if not hist.empty:
        latest_idx = hist["target_date"].idxmax()
        latest_raw = hist.loc[latest_idx]
        if isinstance(latest_raw, pd.DataFrame):
            latest_row = latest_raw.iloc[-1]
        elif isinstance(latest_raw, pd.Series):
            latest_row = latest_raw
        else:
            latest_row = None

    expected_forecast = _expected_forecast_exists(manager, code, as_of)
    alerts = _derive_alert_status(
        as_of=as_of,
        summary_7d=summary_7d,
        summary_30d=summary_30d,
        drift_pct=drift_pct,
        freshness=freshness,
        latest_row=latest_row,
        expected_forecast_exists=expected_forecast,
        config=config,
    )

    payload = {
        "country_code": code,
        "as_of_date": as_of.isoformat(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rolling": {"7d": summary_7d, "30d": summary_30d},
        "drift": {"mae_drift_pct": drift_pct},
        "freshness": freshness,
        "alerts": alerts,
        "expected_next_day_forecast_present": expected_forecast,
        "monitoring_config": config,
    }

    json_path, md_path, alerts_path = _summary_paths(manager, as_of)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    _write_markdown(md_path, payload)

    if alerts["overall_status"] != "ok":
        line = json.dumps(
            {
                "country_code": code,
                "as_of_date": as_of.isoformat(),
                "status": alerts["overall_status"],
                "conditions": alerts["conditions"],
                "generated_at_utc": payload["generated_at_utc"],
            },
            sort_keys=True,
        )
        with open(alerts_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.warning("Health summary status is %s for %s", alerts["overall_status"], code)

    return {
        "summary": payload,
        "json_path": str(json_path.as_posix()),
        "markdown_path": str(md_path.as_posix()),
        "alerts_path": str(alerts_path.as_posix()),
    }
