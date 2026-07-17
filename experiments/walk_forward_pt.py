#!/usr/bin/env python
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Walk-Forward Validation for PriceSentinel (Portugal).
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
logger = logging.getLogger("walk_forward")


def run_experiment(step_size: int, mode: str) -> None:
    logger.info("Initializing Walk-Forward Validation (step_size=%d, mode=%s)", step_size, mode)

    # Register adapters
    auto_register_countries()

    stage_config = {
        "use_fourier_features": True,
        "use_price_volatility": True,
        "use_price_momentum": True,
        "use_weather_features": True,
        "use_gas_features": False,
        "use_event_features": True,
    }
    pipeline = PipelineBuilder.create_pipeline("PT", features_config_override=stage_config)

    # 6 months of initial training data (approx)
    initial_train_size = 4320

    start_date = "2023-01-01"
    end_date = "2024-01-31"

    # Clean data across the whole range
    pipeline.clean_and_verify(start_date, end_date)

    # Using xgboost as the best performing standalone model
    model_name = "xgboost"
    model_config = None  # Using default/tuned config internal to the trainer

    results = pipeline.run_walk_forward_validation(
        start_date=start_date,
        end_date=end_date,
        initial_train_size=initial_train_size,
        step_size=step_size,
        model_name=model_name,
        model_config=model_config,
        mode=mode,
    )

    # Generate Markdown Report
    report = [
        "# Walk-Forward Validation Report (Portugal)",
        f"**Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"**Model**: {model_name}",
        f"**Initial Train Size**: {initial_train_size} hours",
        f"**Step Size**: {step_size} hours",
        f"**Mode**: {mode}",
        "",
        "## Overall Metrics",
        f"- **Mean MAE**: {results['mae'].mean():.4f} €/MWh",
        f"- **Mean RMSE**: {results['rmse'].mean():.4f} €/MWh",
        "",
        "## Step-by-Step Results",
        results.to_markdown(index=False),
        "",
    ]

    report_path = (
        f"outputs/reports/walk_forward_PT_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md"
    )
    os.makedirs("outputs/reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    logger.info("Walk-Forward Report generated at: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Walk-Forward Validation")
    parser.add_argument(
        "--step-size", type=int, default=168, help="Step size in hours (default 168 for weekly)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="expanding",
        choices=["expanding", "sliding"],
        help="Window mode (expanding or sliding)",
    )
    args = parser.parse_args()

    run_experiment(args.step_size, args.mode)
