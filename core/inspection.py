"""Utilities to inspect trained runs and forecasts.

This module provides helpers to quickly inspect model runs by looking at
`metrics.json` under `models/{country}/{model_name}/{run_id}` and any
forecast CSVs under `data/{country}/processed/forecasts`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastSummary:
    """Summary for a single forecast CSV."""

    path: Path
    rows: int
    min_timestamp: str | None
    max_timestamp: str | None
    min_price: float | None
    max_price: float | None
    mean_price: float | None


@dataclass
class RunInspection:
    """Aggregated information for a single run."""

    country_code: str
    model_name: str
    run_id: str | None
    metrics_path: Path | None
    metrics: dict[str, Any] | None
    forecast_summaries: list[ForecastSummary]


def _find_run_dir(
    country_code: str,
    model_name: str,
    run_id: str | None,
    models_root: Path,
) -> Path | None:
    """Return the directory for a run, or None if not found."""
    country_dir = models_root / country_code / model_name

    if run_id is not None:
        run_dir = country_dir / run_id
        if run_dir.exists() and run_dir.is_dir():
            return run_dir
        logger.warning("Run directory not found: %s", run_dir)
        return None

    if not country_dir.exists():
        logger.warning("No model directory found for %s/%s", country_code, model_name)
        return None

    run_dirs = [d for d in country_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        logger.warning("No runs found under %s", country_dir)
        return None

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def inspect_run(
    country_code: str,
    model_name: str = "baseline",
    run_id: str | None = None,
    models_root: Path | str = "models",
    data_root: Path | str = "data",
) -> RunInspection:
    """Inspect a trained run and associated forecasts.

    This function is safe to call even if some artefacts are missing; it will
    log warnings and return a best-effort summary.
    """
    models_root_path = Path(models_root)
    data_root_path = Path(data_root)

    run_dir = _find_run_dir(country_code, model_name, run_id, models_root_path)
    metrics_path: Path | None = None
    metrics: dict[str, Any] | None = None

    if run_dir is not None:
        metrics_candidate = run_dir / "metrics.json"
        if metrics_candidate.exists():
            metrics_path = metrics_candidate
            try:
                metrics = json.loads(metrics_candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse metrics.json at %s: %s", metrics_candidate, exc)
        else:
            logger.warning("metrics.json not found under %s", run_dir)

    # Find forecast files for this country/model; run_id is stored inside the CSV
    forecast_dir = data_root_path / country_code / "processed" / "forecasts"
    forecast_summaries: list[ForecastSummary] = []

    if forecast_dir.exists():
        pattern = f"{country_code}_forecast_*_{model_name}.csv"
        for path in sorted(forecast_dir.glob(pattern)):
            try:
                df = pd.read_csv(path, parse_dates=["forecast_timestamp"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read forecast CSV %s: %s", path, exc)
                continue

            if df.empty:
                summary = ForecastSummary(
                    path=path,
                    rows=0,
                    min_timestamp=None,
                    max_timestamp=None,
                    min_price=None,
                    max_price=None,
                    mean_price=None,
                )
            else:
                min_ts = df["forecast_timestamp"].min()
                max_ts = df["forecast_timestamp"].max()
                prices = df["forecast_price_eur_mwh"]
                summary = ForecastSummary(
                    path=path,
                    rows=len(df),
                    min_timestamp=min_ts.isoformat(),
                    max_timestamp=max_ts.isoformat(),
                    min_price=float(prices.min()),
                    max_price=float(prices.max()),
                    mean_price=float(prices.mean()),
                )

            forecast_summaries.append(summary)

    selected_run_id = run_dir.name if run_dir is not None else None

    return RunInspection(
        country_code=country_code,
        model_name=model_name,
        run_id=selected_run_id,
        metrics_path=metrics_path,
        metrics=metrics,
        forecast_summaries=forecast_summaries,
    )
