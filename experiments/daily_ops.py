# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Daily production ops loop: forecast-next-day and evaluate-yesterday."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.logging_config import setup_logging
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

logger = logging.getLogger("daily_ops")


@dataclass
class DailyScoreRecord:
    country_code: str
    target_date: str
    status: str
    reason: str
    forecast_file: str
    model_name: str
    rows_merged: int
    mae: float | None
    rmse: float | None
    mape: float | None
    directional_accuracy: float | None
    peak_hour_abs_error: float | None
    generated_at_utc: str


def _iso_today_utc() -> date:
    return datetime.now(UTC).date()


def _scorecard_paths(data_manager: Any) -> tuple[Path, Path]:
    root = data_manager.get_processed_path() / "scorecards"
    root.mkdir(parents=True, exist_ok=True)
    return root / "daily_scorecard.csv", root / "daily_scorecard.jsonl"


def _upsert_score_record(data_manager: Any, record: DailyScoreRecord) -> tuple[Path, Path]:
    csv_path, jsonl_path = _scorecard_paths(data_manager)

    row = pd.DataFrame([asdict(record)])
    if csv_path.exists():
        current = pd.read_csv(csv_path)
        current = current[
            ~(
                (current["country_code"] == record.country_code)
                & (current["target_date"] == record.target_date)
            )
        ]
        merged = pd.concat([current, row], ignore_index=True)
    else:
        merged = row

    merged = merged.sort_values(["country_code", "target_date"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for payload in merged.to_dict(orient="records"):
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    return csv_path, jsonl_path


def _load_forecast_for_date(
    data_manager: Any, country_code: str, target_date: str
) -> tuple[pd.DataFrame | None, Path | None]:
    compact = target_date.replace("-", "")
    forecast_dir = data_manager.get_processed_path() / "forecasts"
    if not forecast_dir.exists():
        return None, None

    files = sorted(
        forecast_dir.glob(f"{country_code}_forecast_{compact}_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None, None

    chosen = files[0]
    df = pd.read_csv(chosen, parse_dates=["forecast_timestamp"])
    if "forecast_timestamp" in df.columns:
        df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"], utc=True)
    return df, chosen


def _load_actuals_for_date(data_manager: Any, country_code: str, target_date: str) -> pd.DataFrame:
    processed = data_manager.get_processed_path()
    files = sorted(processed.glob(f"{country_code}_electricity_prices_clean_*.csv"))

    if not files:
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            logger.warning("Skipping unreadable actuals file %s: %s", path, exc)
            continue
        if "timestamp" not in df.columns or "price_eur_mwh" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df[["timestamp", "price_eur_mwh"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])

    all_actuals = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    all_actuals = all_actuals.sort_values("timestamp").reset_index(drop=True)

    start = pd.Timestamp(target_date, tz="UTC")
    end = start + pd.Timedelta(days=1)
    mask = (all_actuals["timestamp"] >= start) & (all_actuals["timestamp"] < end)
    return all_actuals.loc[mask].copy()


def _directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    true_delta = y_true.diff()
    pred_delta = y_pred.diff()
    mask = true_delta.notna() & pred_delta.notna()
    if not mask.any():
        return None

    agreement = np.sign(true_delta[mask]) == np.sign(pred_delta[mask])
    return float(np.mean(agreement))


def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    denom = y_true.replace(0, np.nan)
    ratio = ((y_true - y_pred).abs() / denom.abs()).replace([np.inf, -np.inf], np.nan)
    if ratio.notna().sum() == 0:
        return None
    return float(ratio.mean() * 100.0)


def _evaluate_daily(
    data_manager: Any,
    country_code: str,
    target_date: str,
) -> DailyScoreRecord:
    forecast_df, forecast_path = _load_forecast_for_date(data_manager, country_code, target_date)
    if forecast_df is None or forecast_path is None:
        return DailyScoreRecord(
            country_code=country_code,
            target_date=target_date,
            status="deferred",
            reason="forecast_missing",
            forecast_file="",
            model_name="",
            rows_merged=0,
            mae=None,
            rmse=None,
            mape=None,
            directional_accuracy=None,
            peak_hour_abs_error=None,
            generated_at_utc=datetime.now(UTC).isoformat(),
        )

    actuals_df = _load_actuals_for_date(data_manager, country_code, target_date)
    if len(actuals_df) < 23:
        model_name = (
            str(forecast_df["model_name"].iloc[0])
            if not forecast_df.empty and "model_name" in forecast_df.columns
            else ""
        )
        return DailyScoreRecord(
            country_code=country_code,
            target_date=target_date,
            status="deferred",
            reason="actuals_incomplete",
            forecast_file=forecast_path.name,
            model_name=model_name,
            rows_merged=0,
            mae=None,
            rmse=None,
            mape=None,
            directional_accuracy=None,
            peak_hour_abs_error=None,
            generated_at_utc=datetime.now(UTC).isoformat(),
        )

    merged = pd.merge(
        actuals_df,
        forecast_df,
        left_on="timestamp",
        right_on="forecast_timestamp",
        how="inner",
    )

    model_name = ""
    if "model_name" in merged.columns and not merged.empty:
        model_name = str(merged["model_name"].iloc[0])

    if merged.empty:
        return DailyScoreRecord(
            country_code=country_code,
            target_date=target_date,
            status="deferred",
            reason="no_timestamp_overlap",
            forecast_file=forecast_path.name,
            model_name=model_name,
            rows_merged=0,
            mae=None,
            rmse=None,
            mape=None,
            directional_accuracy=None,
            peak_hour_abs_error=None,
            generated_at_utc=datetime.now(UTC).isoformat(),
        )

    y_true = merged["price_eur_mwh"]
    y_pred = merged["forecast_price_eur_mwh"]

    peak_idx = y_true.idxmax()
    peak_hour_abs_error = float(abs(y_true.loc[peak_idx] - y_pred.loc[peak_idx]))

    return DailyScoreRecord(
        country_code=country_code,
        target_date=target_date,
        status="ok",
        reason="evaluation_complete",
        forecast_file=forecast_path.name,
        model_name=model_name,
        rows_merged=int(len(merged)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mape=_safe_mape(y_true, y_pred),
        directional_accuracy=_directional_accuracy(y_true, y_pred),
        peak_hour_abs_error=peak_hour_abs_error,
        generated_at_utc=datetime.now(UTC).isoformat(),
    )


async def _run_forecast_mode(args: argparse.Namespace) -> dict[str, Any]:
    country_code = args.country.upper()
    target_date = args.target_date or (_iso_today_utc() + timedelta(days=1)).isoformat()

    start_date = (date.fromisoformat(target_date) - timedelta(days=args.history_days)).isoformat()
    end_date = target_date

    auto_register_countries()
    pipeline = PipelineBuilder.create_pipeline(country_code)

    if not args.skip_fetch:
        await pipeline.fetch_data(start_date, end_date)

    pipeline.clean_and_verify(start_date, end_date)
    pipeline.engineer_features(start_date, end_date)

    requested_model = args.model_name or "champion"
    resolved_model = pipeline.model_registry.resolve_model_name(country_code, requested_model)
    pipeline.generate_forecast(target_date, model_name=requested_model)

    forecast_name = f"{country_code}_forecast_{target_date.replace('-', '')}_{resolved_model}.csv"
    forecast_path = pipeline.data_manager.get_processed_path() / "forecasts" / forecast_name

    return {
        "mode": "forecast",
        "country_code": country_code,
        "target_date": target_date,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "forecast_path": str(forecast_path.as_posix()),
        "exists": forecast_path.exists(),
    }


def _run_evaluate_mode(args: argparse.Namespace) -> dict[str, Any]:
    country_code = args.country.upper()
    target_date = args.target_date or (_iso_today_utc() - timedelta(days=1)).isoformat()

    auto_register_countries()
    pipeline = PipelineBuilder.create_pipeline(country_code)

    record = _evaluate_daily(pipeline.data_manager, country_code, target_date)
    csv_path, jsonl_path = _upsert_score_record(pipeline.data_manager, record)

    return {
        "mode": "evaluate",
        "country_code": country_code,
        "target_date": target_date,
        "status": record.status,
        "reason": record.reason,
        "forecast_file": record.forecast_file,
        "model_name": record.model_name,
        "rows_merged": record.rows_merged,
        "metrics": {
            "mae": record.mae,
            "rmse": record.rmse,
            "mape": record.mape,
            "directional_accuracy": record.directional_accuracy,
            "peak_hour_abs_error": record.peak_hour_abs_error,
        },
        "scorecard_csv": str(csv_path.as_posix()),
        "scorecard_jsonl": str(jsonl_path.as_posix()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily PriceSentinel operations loop")
    sub = parser.add_subparsers(dest="command", required=True)

    forecast = sub.add_parser("forecast", help="Generate D+1 forecast")
    forecast.add_argument("--country", required=True, help="Country code (e.g. PT)")
    forecast.add_argument("--target-date", help="Forecast date (YYYY-MM-DD). Default: tomorrow UTC")
    forecast.add_argument(
        "--model-name",
        default="champion",
        help="Model name to forecast with (default: champion)",
    )
    forecast.add_argument(
        "--history-days",
        type=int,
        default=45,
        help="History days to rebuild context features (default: 45)",
    )
    forecast.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetch stage and use locally available data",
    )

    evaluate = sub.add_parser("evaluate", help="Evaluate yesterday (or specified day) forecast")
    evaluate.add_argument("--country", required=True, help="Country code (e.g. PT)")
    evaluate.add_argument(
        "--target-date",
        help="Date to evaluate (YYYY-MM-DD). Default: yesterday UTC",
    )

    return parser


async def main() -> None:
    setup_logging(level="INFO")
    args = _build_parser().parse_args()

    if args.command == "forecast":
        result = await _run_forecast_mode(args)
    else:
        result = _run_evaluate_mode(args)

    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
