#!/usr/bin/env python
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Ensemble Walk-Forward Validation for PriceSentinel (Portugal).
Compares standalone XGBoost against Weighted and Stacking Ensembles.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime

from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ensemble_walk_forward")


def run_experiment(step_size: int, mode: str, start_date: str, end_date: str) -> None:
    logger.info(
        "Initializing Ensemble Walk-Forward Validation (step_size=%d, mode=%s, %s to %s)",
        step_size,
        mode,
        start_date,
        end_date,
    )

    auto_register_countries()

    # Apply fully-aware configuration
    stage_config = {
        "use_fourier_features": True,
        "use_price_volatility": True,
        "use_price_momentum": True,
        "use_weather_features": True,
        "use_gas_features": False,
        "use_event_features": True,
    }

    pipeline = PipelineBuilder.create_pipeline("PT", features_config_override=stage_config)

    initial_train_size = 4320

    pipeline.clean_and_verify(start_date, end_date)

    from core.types import ModelConfig

    # We will test two models (stacking was discarded):
    models_to_test: list[tuple[str, ModelConfig | None]] = [
        ("xgboost", None),
        (
            "ensemble_weighted",
            {"hyperparameters": {"sub_models": ["xgboost", "lightgbm", "baseline"]}},
        ),
    ]

    all_results = {}

    for model_name, model_config in models_to_test:
        logger.info(f"\n\n{'=' * 60}\nEvaluating Model: {model_name}\n{'=' * 60}\n")

        results = pipeline.run_walk_forward_validation(
            start_date=start_date,
            end_date=end_date,
            initial_train_size=initial_train_size,
            step_size=step_size,
            model_name=model_name,
            model_config=model_config,
            mode=mode,
        )
        all_results[model_name] = results

    # Generate Markdown Report
    report = [
        "# Ensemble Walk-Forward Validation Report (Portugal)",
        f"**Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"**Initial Train Size**: {initial_train_size} hours",
        f"**Step Size**: {step_size} hours",
        f"**Mode**: {mode}",
        "",
        "## Overall Metrics Comparison",
        "| Model | Mean MAE (€/MWh) | Mean RMSE (€/MWh) |",
        "|---|---|---|",
    ]

    for model_name, results in all_results.items():
        mean_mae = results["mae"].mean()
        mean_rmse = results["rmse"].mean()
        report.append(f"| {model_name} | {mean_mae:.4f} | {mean_rmse:.4f} |")

    report.extend(
        [
            "",
            "## Detailed Step-by-Step Results Comparison (MAE)",
            "| Step | Test Date | XGBoost | Weighted |",
            "|---|---|---|---|",
        ]
    )

    # Merge results on test_date
    df_xgb = all_results["xgboost"][["step_idx", "test_date", "mae"]].rename(
        columns={"mae": "xgb_mae"}
    )
    df_weighted = all_results["ensemble_weighted"][["step_idx", "mae"]].rename(
        columns={"mae": "weighted_mae"}
    )

    merged = df_xgb.merge(df_weighted, on="step_idx")

    for _, row in merged.iterrows():
        date_str = str(row["test_date"])
        report.append(
            f"| {int(row['step_idx'])} |"
            f" {date_str} |"
            f" {row['xgb_mae']:.4f} |"
            f" {row['weighted_mae']:.4f} |"
        )

    report_path = (
        f"outputs/reports/ensemble_walk_forward_PT_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md"
    )
    os.makedirs("outputs/reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    logger.info("Ensemble Walk-Forward Report generated at: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ensemble Walk-Forward Validation")
    parser.add_argument(
        "--step-size", type=int, default=168, help="Step size in hours (default 168 for weekly)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="expanding",
        choices=["expanding", "sliding"],
        help="Window mode",
    )
    parser.add_argument("--start-date", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end-date", type=str, default="2026-01-31", help="End date")
    args = parser.parse_args()

    run_experiment(args.step_size, args.mode, args.start_date, args.end_date)
