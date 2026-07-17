#!/usr/bin/env python
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Ablation Experiment for PriceSentinel (Portugal).

Goal: Provide verifiable evidence of how each feature subset improves model accuracy.
Methodology:
1. Define multiple feature configurations (ablation stages).
2. For each stage, re-engineer features, train on 2023, and forecast Jan 2024.
3. Generate a comparative performance report.
"""

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation")

ablation_stages = [
    {
        "name": "01_Strict_Baseline",
        "config": {
            "use_fourier_features": False,
            "use_price_volatility": False,
            "use_price_momentum": False,
            "use_weather_features": False,
            "use_gas_features": False,
            "use_event_features": False,
        },
    },
    {
        "name": "02_Temporal_Aware",
        "config": {
            "use_fourier_features": True,
            "use_price_volatility": True,
            "use_price_momentum": True,
            "use_weather_features": False,
            "use_gas_features": False,
            "use_event_features": False,
        },
    },
    {
        "name": "03_Weather_Aware",
        "config": {
            "use_fourier_features": True,
            "use_price_volatility": True,
            "use_price_momentum": True,
            "use_weather_features": True,
            "use_gas_features": False,
            "use_event_features": False,
        },
    },
    {
        "name": "04_Fully_Aware",
        "config": {
            "use_fourier_features": True,
            "use_price_volatility": True,
            "use_price_momentum": True,
            "use_weather_features": True,
            "use_gas_features": True,
            "use_event_features": True,
        },
    },
]


async def run_ablation() -> None:
    country = "PT"
    train_start = "2023-01-01"
    train_end = "2023-12-31"
    test_start = "2024-01-01"
    test_end = "2024-01-31"

    logger.info(f"Starting Ablation Testing for {country}")

    auto_register_countries()

    # We create a dummy pipeline just to use its cleaner and repository to load actuals
    dummy_pipeline = PipelineBuilder.create_pipeline(country)
    dummy_pipeline.clean_and_verify(train_start, test_end)

    actuals_df = dummy_pipeline.repository.load_data(
        "electricity_prices_clean", train_start, test_end, source="processed"
    )

    if actuals_df is not None:
        actuals_df["timestamp"] = pd.to_datetime(actuals_df["timestamp"], utc=True)
        mask = (actuals_df["timestamp"] >= pd.Timestamp(test_start, tz="UTC")) & (
            actuals_df["timestamp"] <= pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
        )
        actuals_df = actuals_df.loc[mask]
    else:
        logger.error("Failed to load actuals data.")
        return

    results = []

    for stage in ablation_stages:
        stage_name = stage["name"]
        logger.info(f"\n{'=' * 50}\nRunning Ablation Stage: {stage_name}\n{'=' * 50}")

        # 1. Setup Pipeline with override
        stage_config = cast(dict[str, Any], stage["config"])
        pipeline = PipelineBuilder.create_pipeline(country, features_config_override=stage_config)

        # 2. Clean and Verify to ensure exact date-range processed files exist
        pipeline.clean_and_verify(train_start, train_end)
        pipeline.clean_and_verify(train_start, test_end)

        # 3. Re-engineer features (CRITICAL for ablation)
        logger.info(f"[{stage_name}] Engineering features...")
        pipeline.engineer_features(train_start, train_end)  # For training
        pipeline.engineer_features(train_start, test_end)  # For testing/forecast

        # 4. Train
        logger.info(f"[{stage_name}] Training model...")
        tune = stage_name in ("03_Weather_Aware", "04_Fully_Aware")
        pipeline.train_model(
            start_date=train_start, end_date=train_end, model_name="xgboost", tune=tune
        )

        # 4. Forecast
        logger.info(f"[{stage_name}] Generating forecasts...")
        pipeline.generate_forecast_range(test_start, test_end, model_name="xgboost")

        # 5. Evaluate
        forecasts = []
        current = datetime.strptime(test_start, "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(test_end, "%Y-%m-%d").replace(tzinfo=UTC)

        date_cursor = current
        while date_cursor <= end:
            d_compact = date_cursor.strftime("%Y%m%d")
            f_name = f"{country}_forecast_{d_compact}_xgboost.csv"
            try:
                f_df = pipeline.repository.load_forecast(f_name)
                forecasts.append(f_df)
            except FileNotFoundError:
                pass
            date_cursor += timedelta(days=1)

        if not forecasts:
            logger.error(f"No forecasts found for {stage_name}!")
            continue

        forecast_combined = pd.concat(forecasts)
        merged = pd.merge(
            actuals_df,
            forecast_combined,
            left_on="timestamp",
            right_on="forecast_timestamp",
            how="inner",
        )

        if merged.empty:
            logger.error(f"No overlapping timestamps for {stage_name}.")
            continue

        y_true = merged["price_eur_mwh"]
        y_pred = merged["forecast_price_eur_mwh"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        results.append({"Stage": stage_name, "MAE": mae, "RMSE": rmse})
        logger.info(f"[{stage_name}] Result - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Generate Report
    report = f"""# Ablation Report: PriceSentinel ({country})

**Date**: {datetime.now(UTC).strftime("%Y-%m-%d")}
**Test Period**: {test_start} to {test_end}

## Results
| Stage | MAE (€/MWh) | RMSE (€/MWh) | Improvement (MAE) |
|-------|-------------|--------------|-------------------|
"""

    if results:
        base_mae = results[0]["MAE"]
        for res in results:
            improvement = ((base_mae - res["MAE"]) / base_mae) * 100 if base_mae > 0 else 0
            report += (
                f"| {res['Stage']} | {res['MAE']:.2f} | {res['RMSE']:.2f} | {improvement:+.1f}% |\n"
            )

    logger.info("\n" + report)

    os.makedirs("outputs/reports", exist_ok=True)
    report_path = f"outputs/reports/ablation_{country}_{datetime.now(UTC).strftime('%Y%m%d')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Ablation complete. Report saved to {report_path}.")


if __name__ == "__main__":
    asyncio.run(run_ablation())
