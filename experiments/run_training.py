# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible multi-year training workflow with quality gates and manifest output."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from core.logging_config import setup_logging
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries
from models import get_trainer

logger = logging.getLogger("run_training")

FEATURE_SCHEMA_VERSION = "v1"
DEFAULT_THRESHOLDS: dict[str, float] = {
    "core": 0.10,
    "weather": 0.65,
    "gas": 0.80,
    "events": 0.05,
    "fourier": 0.05,
    "volatility": 0.10,
    "momentum": 0.10,
}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _date_inclusive_mask(series: pd.Series, start: str, end: str) -> pd.Series:
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return (series >= start_dt) & (series <= end_dt)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _family_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(df.columns)
    return {
        "core": [
            c
            for c in columns
            if c in {"price_eur_mwh", "hour", "day_of_week", "load_mw", "target_price"}
            or c.startswith("price_lag_")
            or c.startswith("price_rolling_")
        ],
        "weather": [
            c
            for c in columns
            if c in {"temperature_c", "wind_speed_ms", "solar_radiation_wm2", "precipitation_mm"}
        ],
        "gas": [c for c in columns if c == "gas_price_eur_mwh"],
        "events": [c for c in columns if c in {"is_holiday", "is_event"}],
        "fourier": [c for c in columns if c.startswith("sin_") or c.startswith("cos_")],
        "volatility": [
            c for c in columns if c.startswith("price_volatility_") or c.startswith("price_range_")
        ],
        "momentum": [c for c in columns if c.startswith("price_roc_")],
    }


def _enabled_families(features_config: dict[str, Any]) -> dict[str, bool]:
    return {
        "core": True,
        "weather": bool(features_config.get("use_weather_features", True)),
        "gas": bool(features_config.get("use_gas_features", True)),
        "events": bool(features_config.get("use_event_features", True)),
        "fourier": bool(features_config.get("use_fourier_features", False)),
        "volatility": bool(features_config.get("use_price_volatility", False)),
        "momentum": bool(features_config.get("use_price_momentum", False)),
    }


def _missingness_report(
    features_df: pd.DataFrame,
    features_config: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    family_cols = _family_columns(features_df)
    enabled = _enabled_families(features_config)

    family_reports: dict[str, dict[str, Any]] = {}
    passed = True

    for family, cols in family_cols.items():
        threshold = float(thresholds.get(family, 1.0))
        enabled_for_family = enabled.get(family, True)

        if not cols:
            miss_ratio = 1.0 if enabled_for_family else 0.0
            family_pass = not enabled_for_family
            reason = (
                "expected columns missing"
                if enabled_for_family
                else "feature family disabled and columns absent"
            )
        else:
            miss_ratio = float(features_df[cols].isna().mean().mean())
            family_pass = miss_ratio <= threshold
            reason = "ok" if family_pass else "missingness above threshold"

        if not family_pass:
            passed = False

        family_reports[family] = {
            "enabled": enabled_for_family,
            "columns": cols,
            "missing_ratio": round(miss_ratio, 6),
            "threshold": threshold,
            "passed": family_pass,
            "reason": reason,
        }

    return {
        "passed": passed,
        "criteria": "missing_ratio <= threshold for each enabled feature family",
        "families": family_reports,
    }


def _leakage_report(features_df: pd.DataFrame, lags: list[int]) -> dict[str, Any]:
    failures: list[str] = []
    df = features_df.sort_values("timestamp").reset_index(drop=True).copy()

    if not df["timestamp"].is_monotonic_increasing:
        failures.append("timestamps are not monotonically increasing")

    if df["timestamp"].duplicated().any():
        failures.append("duplicate timestamps found in feature matrix")

    if "price_eur_mwh" in df.columns:
        for lag in lags:
            col = f"price_lag_{lag}"
            if col not in df.columns:
                failures.append(f"missing expected lag column: {col}")
                continue

            expected = df["price_eur_mwh"].shift(lag)
            mask = df[col].notna() & expected.notna()
            if mask.any() and not df.loc[mask, col].equals(expected.loc[mask]):
                failures.append(f"{col} is not consistent with historical shift({lag})")

        if "target_price" in df.columns:
            expected_target = df["price_eur_mwh"].shift(-1)
            mask_target = df["target_price"].notna() & expected_target.notna()
            if mask_target.any() and not df.loc[mask_target, "target_price"].equals(
                expected_target.loc[mask_target]
            ):
                failures.append("target_price is not consistent with shift(-1)")

    return {
        "passed": len(failures) == 0,
        "criteria": "lags must match historical shifts and target_price must match next-step shift",
        "failures": failures,
    }


def _drop_report(prices_df: pd.DataFrame, lags: list[int]) -> dict[str, Any]:
    if prices_df.empty:
        return {
            "raw_price_rows": 0,
            "rows_after_required_lag_target_filter": 0,
            "dropped_rows": 0,
            "drop_reasons": {"target_or_lag_unavailable": 0},
        }

    working = prices_df.sort_values("timestamp").reset_index(drop=True).copy()
    working["target_price"] = working["price_eur_mwh"].shift(-1)

    for lag in lags:
        working[f"price_lag_{lag}"] = working["price_eur_mwh"].shift(lag)

    required = ["target_price"] + [f"price_lag_{lag}" for lag in lags]
    rows_after = int(len(working.dropna(subset=required)))
    raw_rows = int(len(working))

    return {
        "raw_price_rows": raw_rows,
        "rows_after_required_lag_target_filter": rows_after,
        "dropped_rows": raw_rows - rows_after,
        "drop_reasons": {
            "target_or_lag_unavailable": raw_rows - rows_after,
        },
    }


def _collect_data_sources(
    pipeline: Any,
    data_start: str,
    data_end: str,
) -> list[dict[str, Any]]:
    names = [
        "electricity_prices_clean",
        "electricity_load_clean",
        "weather_clean",
        "gas_prices_clean",
        "holidays_clean",
        "manual_events_clean",
        "electricity_features",
    ]

    rows: list[dict[str, Any]] = []
    for name in names:
        path = pipeline.data_manager.get_processed_file_path(name, data_start, data_end)
        if path.exists():
            frame = pd.read_csv(path)
            rows.append(
                {
                    "dataset": name,
                    "path": str(path.as_posix()),
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                }
            )
        else:
            rows.append(
                {
                    "dataset": name,
                    "path": str(path.as_posix()),
                    "rows": None,
                    "columns": [],
                }
            )

    return rows


def _quality_and_train(
    pipeline: Any,
    country_code: str,
    model_name: str,
    train_start: str,
    train_end: str,
    holdout_start: str | None,
    holdout_end: str | None,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], Path]:
    data_end = holdout_end or train_end

    features_path = pipeline.data_manager.get_processed_file_path(
        "electricity_features",
        train_start,
        data_end,
    )
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    features_df = pd.read_csv(features_path, parse_dates=["timestamp"])
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
    features_df = features_df.sort_values("timestamp").reset_index(drop=True)

    prices_path = pipeline.data_manager.get_processed_file_path(
        "electricity_prices_clean",
        train_start,
        data_end,
    )
    if not prices_path.exists():
        raise FileNotFoundError(f"Clean prices file not found: {prices_path}")

    prices_df = pd.read_csv(prices_path, parse_dates=["timestamp"])
    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"], utc=True)

    windows_cfg = dict(pipeline.config.features_config.get("feature_windows", {}))
    lags = list(windows_cfg.get("lags", [1, 2, 24]))

    missingness = _missingness_report(
        features_df, dict(pipeline.config.features_config), thresholds
    )
    leakage = _leakage_report(features_df, lags)
    dropped = _drop_report(prices_df, lags)

    if not missingness["passed"]:
        raise ValueError(
            "Data quality gate failed: missingness thresholds exceeded. "
            f"Details: {json.dumps(missingness['families'], sort_keys=True)}"
        )

    if not leakage["passed"]:
        raise ValueError(f"Leakage gate failed. Failures: {', '.join(leakage['failures'])}")

    train_mask = _date_inclusive_mask(features_df["timestamp"], train_start, train_end)
    train_df = features_df.loc[train_mask].copy()
    if train_df.empty:
        raise ValueError("No training rows found for selected train window")

    holdout_df = pd.DataFrame()
    if holdout_start and holdout_end:
        holdout_mask = _date_inclusive_mask(features_df["timestamp"], holdout_start, holdout_end)
        holdout_df = features_df.loc[holdout_mask].copy()
        if holdout_df.empty:
            raise ValueError("Holdout window provided but no holdout rows found")

    target_col = "target_price"
    feature_cols = [c for c in features_df.columns if c not in ("timestamp", target_col)]

    x_train = train_df[feature_cols].select_dtypes(include="number")
    y_train = train_df[target_col]

    x_holdout = None
    y_holdout = None
    if not holdout_df.empty:
        x_holdout = holdout_df[feature_cols].select_dtypes(include="number")
        y_holdout = holdout_df[target_col]

    trainer = get_trainer(country_code, model_name=model_name)
    metrics = trainer.train(x_train, y_train, x_holdout, y_holdout)
    metrics["train_start_date"] = train_start
    metrics["train_end_date"] = train_end
    if holdout_start and holdout_end:
        metrics["holdout_start_date"] = holdout_start
        metrics["holdout_end_date"] = holdout_end

    trainer.save(country_code, pipeline.run_id, metrics=metrics)

    run_dir = Path("models") / country_code / model_name / pipeline.run_id

    manifest = {
        "country_code": country_code,
        "model_name": model_name,
        "train_window": {"start": train_start, "end": train_end},
        "holdout_window": (
            {"start": holdout_start, "end": holdout_end} if holdout_start and holdout_end else None
        ),
        "feature_window": {"start": train_start, "end": data_end},
        "timezone": pipeline.config.timezone,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_toggles": pipeline.config.features_config,
        "data_sources": _collect_data_sources(pipeline, train_start, data_end),
        "row_counts": {
            "features_total": int(len(features_df)),
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "dropped_rows_report": dropped,
        },
        "feature_columns": sorted([c for c in feature_cols if c in x_train.columns]),
        "quality_gates": {
            "missingness": missingness,
            "leakage": leakage,
        },
        "metrics": metrics,
        "code_version": {
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest, manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible training with strict quality gates"
    )
    parser.add_argument("--country", required=True, help="Country code (e.g. PT)")
    parser.add_argument("--model-name", default="baseline", help="Model name (trainer key)")
    parser.add_argument(
        "--train-start", required=True, help="Training window start date (YYYY-MM-DD)"
    )
    parser.add_argument("--train-end", required=True, help="Training window end date (YYYY-MM-DD)")
    parser.add_argument("--holdout-start", help="Optional holdout window start date (YYYY-MM-DD)")
    parser.add_argument("--holdout-end", help="Optional holdout window end date (YYYY-MM-DD)")
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch raw data before clean/features/train",
    )
    return parser


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)

    if train_start > train_end:
        raise ValueError("train-start must be <= train-end")

    if bool(args.holdout_start) ^ bool(args.holdout_end):
        raise ValueError("holdout-start and holdout-end must be provided together")

    holdout_start = args.holdout_start
    holdout_end = args.holdout_end

    if holdout_start and holdout_end:
        holdout_start_d = _parse_date(holdout_start)
        holdout_end_d = _parse_date(holdout_end)
        if holdout_start_d > holdout_end_d:
            raise ValueError("holdout-start must be <= holdout-end")
        if holdout_start_d <= train_end:
            raise ValueError("holdout window must start strictly after train-end")

    setup_logging(level="INFO")
    auto_register_countries()

    country = args.country.upper()
    effective_end = holdout_end or args.train_end

    pipeline = PipelineBuilder.create_pipeline(country)

    if args.fetch_data:
        await pipeline.fetch_data(args.train_start, effective_end)

    pipeline.clean_and_verify(args.train_start, effective_end)
    pipeline.engineer_features(args.train_start, effective_end)

    manifest, manifest_path = _quality_and_train(
        pipeline=pipeline,
        country_code=country,
        model_name=args.model_name,
        train_start=args.train_start,
        train_end=args.train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        thresholds=DEFAULT_THRESHOLDS,
    )

    logger.info(
        "Training completed: country=%s model=%s train_window=%s..%s holdout=%s",
        country,
        args.model_name,
        args.train_start,
        args.train_end,
        {"start": holdout_start, "end": holdout_end} if holdout_start and holdout_end else None,
    )
    logger.info("Manifest saved to %s", manifest_path)

    sys.stdout.write(
        json.dumps(
            {
                "manifest_path": str(manifest_path.as_posix()),
                "run_id": pipeline.run_id,
                "metrics": manifest.get("metrics", {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
