# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Country qualification workflow for onboarding gates (A/B/C phases)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config.validation import validate_country_config
from core.logging_config import setup_logging
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

REQUIRED_SCHEMAS: dict[str, set[str]] = {
    "electricity_prices_clean": {"timestamp", "price_eur_mwh", "quality_flag"},
    "electricity_load_clean": {"timestamp", "load_mw", "quality_flag"},
    "weather_clean": {
        "timestamp",
        "location_name",
        "temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
        "precipitation_mm",
        "quality_flag",
    },
    "gas_prices_clean": {"timestamp", "price_eur_mwh", "hub_name", "quality_flag"},
    "holidays_clean": {"timestamp", "event_type", "description"},
}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _check_schema(path: Path, expected: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "fail",
            "reason": "file_missing",
            "path": str(path.as_posix()),
            "missing_columns": sorted(expected),
            "rows": 0,
        }

    df = pd.read_csv(path)
    cols = set(df.columns)
    missing = sorted(expected - cols)
    return {
        "status": "pass" if not missing else "fail",
        "reason": "ok" if not missing else "columns_missing",
        "path": str(path.as_posix()),
        "missing_columns": missing,
        "rows": int(len(df)),
    }


def _feature_completeness(features_path: Path) -> dict[str, Any]:
    if not features_path.exists():
        return {
            "status": "fail",
            "reason": "file_missing",
            "path": str(features_path.as_posix()),
            "rows": 0,
            "missing_ratio": None,
        }

    df = pd.read_csv(features_path)
    if df.empty:
        return {
            "status": "fail",
            "reason": "empty_features",
            "path": str(features_path.as_posix()),
            "rows": 0,
            "missing_ratio": None,
        }

    required_cols = ["target_price", "price_lag_1", "hour", "day_of_week"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {
            "status": "fail",
            "reason": "required_columns_missing",
            "path": str(features_path.as_posix()),
            "rows": int(len(df)),
            "missing_ratio": None,
            "missing_columns": missing_cols,
        }

    missing_ratio = float(df[required_cols].isna().mean().mean())
    status = "pass" if missing_ratio <= 0.1 else "fail"
    reason = "ok" if status == "pass" else "missingness_above_threshold"
    return {
        "status": status,
        "reason": reason,
        "path": str(features_path.as_posix()),
        "rows": int(len(df)),
        "missing_ratio": round(missing_ratio, 6),
        "missing_columns": [],
    }


def _phase_status(
    schema_checks: dict[str, dict[str, Any]],
    feature_check: dict[str, Any],
    smoke_ok: bool,
) -> dict[str, Any]:
    phase_a = all(v["status"] == "pass" for v in schema_checks.values())
    phase_b = phase_a and feature_check["status"] == "pass"
    phase_c = phase_b and smoke_ok

    if phase_c:
        ready = "C"
    elif phase_b:
        ready = "B"
    elif phase_a:
        ready = "A"
    else:
        ready = "not_ready"

    return {
        "phase_a_fetch_clean": phase_a,
        "phase_b_features_cv": phase_b,
        "phase_c_smoke_forecast": phase_c,
        "ready_phase": ready,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Country Qualification - {report['country_code']}",
        "",
        f"- Window: {report['window']['start']} to {report['window']['end']}",
        f"- Generated: {report['generated_at_utc']}",
        f"- Ready phase: **{report['phases']['ready_phase']}**",
        "",
        "## Schema Checks",
        "",
    ]

    for name, payload in report["schema_checks"].items():
        lines.append(
            f"- {name}: status={payload['status']} rows={payload['rows']} "
            f"missing={payload.get('missing_columns', [])}"
        )

    lines.extend(
        [
            "",
            "## Feature Completeness",
            "",
            f"- status={report['feature_check']['status']}",
            f"- rows={report['feature_check']['rows']}",
            f"- missing_ratio={report['feature_check']['missing_ratio']}",
            "",
            "## Smoke Forecast",
            "",
            f"- status={report['smoke_forecast']['status']}",
            f"- forecast_file={report['smoke_forecast']['forecast_file']}",
            "",
            "## Phases",
            "",
            f"- Phase A (fetch+clean): {report['phases']['phase_a_fetch_clean']}",
            f"- Phase B (features+offline checks): {report['phases']['phase_b_features_cv']}",
            f"- Phase C (smoke forecast): {report['phases']['phase_c_smoke_forecast']}",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run country qualification gates")
    parser.add_argument("--country", required=True, help="Country code (e.g. PT)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model-name",
        default="baseline",
        help="Model to use for smoke forecast (default: baseline)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetch stage and use local raw data",
    )
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if start > end:
        raise ValueError("start must be <= end")

    setup_logging(level="INFO")
    auto_register_countries()

    code = args.country.upper()

    config_path = Path("config") / "countries" / f"{code}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Country config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_payload = yaml.safe_load(f)
    validate_country_config(config_payload)

    pipeline = PipelineBuilder.create_pipeline(code)

    if not args.skip_fetch:
        await pipeline.fetch_data(args.start, args.end)

    pipeline.clean_and_verify(args.start, args.end)
    pipeline.engineer_features(args.start, args.end)

    schema_checks: dict[str, dict[str, Any]] = {}
    for dataset, expected_cols in REQUIRED_SCHEMAS.items():
        dataset_path = pipeline.data_manager.get_processed_file_path(dataset, args.start, args.end)
        schema_checks[dataset] = _check_schema(dataset_path, expected_cols)

    features_path = pipeline.data_manager.get_processed_file_path(
        "electricity_features", args.start, args.end
    )
    feature_check = _feature_completeness(features_path)

    smoke_status = "fail"
    forecast_file = ""
    try:
        pipeline.train_model(args.start, args.end, model_name=args.model_name)
        pipeline.generate_forecast(args.end, model_name=args.model_name)
        forecast_file = (
            f"{code}_forecast_{args.end.replace('-', '')}_"
            f"{pipeline.model_registry.resolve_model_name(code, args.model_name)}.csv"
        )
        forecast_path = pipeline.data_manager.get_processed_path() / "forecasts" / forecast_file
        if forecast_path.exists():
            forecast_df = pd.read_csv(forecast_path)
            smoke_status = "pass" if not forecast_df.empty else "fail"
    except Exception:
        smoke_status = "fail"

    phases = _phase_status(schema_checks, feature_check, smoke_status == "pass")

    report = {
        "country_code": code,
        "window": {"start": args.start, "end": args.end},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config_file": str(config_path.as_posix()),
        "schema_checks": schema_checks,
        "feature_check": feature_check,
        "smoke_forecast": {"status": smoke_status, "forecast_file": forecast_file},
        "phases": phases,
    }

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.end.replace("-", "")
    json_path = out_dir / f"qualification_{code}_{stamp}.json"
    md_path = out_dir / f"qualification_{code}_{stamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    _write_markdown(md_path, report)

    sys.stdout.write(
        json.dumps(
            {
                "country_code": code,
                "ready_phase": phases["ready_phase"],
                "report_json": str(json_path.as_posix()),
                "report_markdown": str(md_path.as_posix()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
