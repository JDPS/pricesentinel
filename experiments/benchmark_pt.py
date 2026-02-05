#!/usr/bin/env python
# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark Experiment for PriceSentinel (Portugal).

Goal: Provide verifiable evidence of model accuracy and efficiency.
Methodology:
1. Train on historical data (2023).
2. Forecast on unseen data (Jan 2024).
3. Compare Model vs Naive Baseline (Persistence).
4. Generate a performance report.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark")


async def run_benchmark() -> None:
    country = "PT"

    # Define experiment period
    # Train: Full year 2023 to capture seasonality
    train_start = "2023-01-01"
    train_end = "2023-12-31"

    # Test: Jan 2024 (Winter peak) - challenging for models
    test_start = "2024-01-01"
    test_end = "2024-01-31"

    logger.info(f"Starting Benchmark for {country}")
    logger.info(f"Train Period: {train_start} to {train_end}")
    logger.info(f"Test Period:  {test_start} to {test_end}")

    # 0. Register Countries directly
    auto_register_countries()

    # 1. Setup
    # ConfigLoader().load_country_config(country)
    pipeline = PipelineBuilder.create_pipeline(country)

    # 2. Data Preparation (Fetch + Clean + Feature)
    # We fetch the superset of dates needed
    logger.info("--- Phase 1: Data Preparation ---")
    t0 = time.time()
    # await pipeline.fetch_data(train_start, test_end)  # Skip fetch, use local data

    # 1. Prepare Training Data (Exact match for safe loading)
    logger.info("Cleaning & Engineering features for Training set...")
    pipeline.clean_and_verify(train_start, train_end)

    # Engineer Features for TRAINING set (exact match for training)
    logger.info("Engineering features for Training set...")
    pipeline.engineer_features(train_start, train_end)
    logger.info(f"Data Prep Time: {time.time() - t0:.2f}s")

    # 3. Train Model
    logger.info("--- Phase 2: Model Training ---")
    t0 = time.time()
    pipeline.train_model(start_date=train_start, end_date=train_end, model_name="benchmark_rf")
    train_time = time.time() - t0
    logger.info(f"Training Time: {train_time:.2f}s")

    # Engineer Features for TESTING set (exact match for inference)
    # We use the full range (Train + Test) to ensure lags/context are available for the test period
    logger.info("Cleaning & Engineering features for Test set (with context)...")
    pipeline.clean_and_verify(train_start, test_end)
    pipeline.engineer_features(train_start, test_end)

    # 4. Generate Forecasts (Inference)
    logger.info("--- Phase 3: Inference on Test Set ---")
    # efficient: generate for the whole range day by day
    t0 = time.time()
    # We can use generate_forecast_range if we added it, otherwise loop
    pipeline.generate_forecast_range(test_start, test_end, model_name="benchmark_rf")
    inference_time = time.time() - t0
    logger.info(f"Inference Time: {inference_time:.2f}s")

    # 5. Evaluation & Comparison
    logger.info("--- Phase 4: Evaluation ---")

    # Load Actuals (Ground Truth) - load superset as we cleaned the full range
    actuals_df = pipeline.repository.load_data(
        "electricity_prices_clean", train_start, test_end, source="processed"
    )
    if actuals_df is not None:
        actuals_df["timestamp"] = pd.to_datetime(actuals_df["timestamp"], utc=True)
        mask = (actuals_df["timestamp"] >= pd.Timestamp(test_start, tz="UTC")) & (
            actuals_df["timestamp"] <= pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
        )
        actuals_df = actuals_df.loc[mask]

    # Load Forecasts
    forecasts = []
    # Use timezone-aware dates for comparison logic if needed, but file pattern is just date
    current = datetime.strptime(test_start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(test_end, "%Y-%m-%d").replace(tzinfo=UTC)

    from datetime import timedelta

    date_cursor = current
    while date_cursor <= end:
        d_str = date_cursor.strftime("%Y-%m-%d")
        d_compact = date_cursor.strftime("%Y%m%d")
        f_name = f"{country}_forecast_{d_compact}_benchmark_rf.csv"
        try:
            f_df = pipeline.repository.load_forecast(f_name)
            forecasts.append(f_df)
        except FileNotFoundError:
            logger.warning(f"Missing forecast for {d_str}")
        date_cursor += timedelta(days=1)

    if not forecasts:
        logger.error("No forecasts found! Benchmark failed.")
        return

    forecast_combined = pd.concat(forecasts)

    # Merge on timestamp
    # Actuals: timestamp, price_eur_mwh
    # Forecast: forecast_timestamp, forecast_price_eur_mwh

    if actuals_df is None:
        logger.error("Failed to load actuals data.")
        return

    merged = pd.merge(
        actuals_df,
        forecast_combined,
        left_on="timestamp",
        right_on="forecast_timestamp",
        how="inner",
    )

    if merged.empty:
        logger.error("No overlapping timestamps between actuals and forecasts.")
        return

    y_true = merged["price_eur_mwh"]
    y_pred = merged["forecast_price_eur_mwh"]

    # Calculate Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Calculate Naive Baseline (Persistence: 24h lag)
    # We need specific lag features, or we can approximate by shift(24) on y_true
    # But y_true is just the test window. We need context for the first 24h.
    # Simpler: Use 'price_lag_24' from features if we can load them,
    # or just accept we lose first 24h of evaluation.
    # Let's lose first 24h for baseline comparison validity.

    y_true_base = y_true.iloc[24:]
    y_naive = y_true.shift(24).iloc[24:]  # Predict t = t-24

    mae_naive = mean_absolute_error(y_true_base, y_naive)

    improvement = ((mae_naive - mae) / mae_naive) * 100

    # 6. Report
    report = f"""
    # Benchmark Report: PriceSentinel ({country})

    **Date**: {datetime.now(UTC).strftime("%Y-%m-%d")}
    **Model**: Random Forest (benchmark_rf)
    **Test Period**: {test_start} to {test_end}

    ## Performance
    | Metric | Model | Naive Baseline | Improvement |
    |--------|-------|----------------|-------------|
    | **MAE**  | **{mae:.2f} €** | {mae_naive:.2f} € | **{improvement:+.1f}%** |
    | RMSE   | {rmse:.2f} € | N/A | - |
    | R²     | {r2:.2f}    | N/A | - |

    ## Efficiency
    - Training Time: {train_time:.2f}s
    - Inference Time: {inference_time:.2f}s ({inference_time / 31:.4f}s/day)

    ## Conclusion
    The model {"outperforms" if mae < mae_naive else "underperforms"} the naive baseline by
    around {abs(improvement):.1f}%.
    """

    logger.info("\n" + "=" * 40)
    logger.info(report)
    logger.info("=" * 40 + "\n")

    # Save report
    os.makedirs("outputs/reports", exist_ok=True)
    with open(
        f"outputs/reports/benchmark_{country}_{datetime.now(UTC).strftime('%Y%m%d')}.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    logger.info("Benchmark complete. Report saved.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
