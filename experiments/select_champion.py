# Copyright (c) 2026 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-driven model selection and champion registration workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import yaml

from core.logging_config import setup_logging
from core.pipeline_builder import PipelineBuilder
from data_fetchers import auto_register_countries
from experiments.model_comparison import ModelComparison
from models import get_trainer
from models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from core.types import ModelConfig

logger = logging.getLogger("select_champion")

DEFAULT_POLICY: dict[str, Any] = {
    "allowed_models": ["baseline"],
    "cv_method": "walk_forward",
    "initial_train_size": 720,
    "step_size": 24,
    "n_splits": 5,
    "metrics_priority": ["mae", "rmse", "stability", "model_name"],
    "hpo": {
        "enabled": False,
        "shortlist_size": 1,
        "n_trials": 10,
        "cv_splits": 3,
    },
}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _policy_path(country_code: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path("config") / "selection_policies" / f"{country_code}.yaml"


def _load_policy(country_code: str, explicit: str | None) -> tuple[dict[str, Any], Path | None]:
    path = _policy_path(country_code, explicit)
    if not path.exists():
        logger.warning("Policy file not found at %s. Using default policy.", path)
        return dict(DEFAULT_POLICY), None

    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    policy = dict(DEFAULT_POLICY)
    policy.update({k: v for k, v in loaded.items() if k != "hpo"})
    hpo_policy = dict(DEFAULT_POLICY["hpo"])
    hpo_policy.update(loaded.get("hpo", {}))
    policy["hpo"] = hpo_policy

    allowed_models = policy.get("allowed_models", [])
    if not isinstance(allowed_models, list) or not allowed_models:
        raise ValueError("Selection policy requires a non-empty 'allowed_models' list")

    return policy, path


def _load_matrix(
    pipeline: Any, start_date: str, end_date: str
) -> tuple[pd.DataFrame, pd.Series, Path]:
    path = pipeline.data_manager.get_processed_file_path(
        "electricity_features", start_date, end_date
    )
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "target_price" not in df.columns:
        raise ValueError("Feature file missing required column: target_price")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df_clean = df.dropna(subset=["target_price"]).reset_index(drop=True)

    feature_cols = [c for c in df_clean.columns if c not in ("timestamp", "target_price")]
    x = df_clean[feature_cols].select_dtypes(include="number")
    y = df_clean["target_price"]

    if len(x) < 48:
        raise ValueError("Not enough feature rows for robust selection (minimum: 48)")

    return x, y, path


def _ranking_key(row: pd.Series) -> tuple[float, float, float, str]:
    return (
        float(row["mean_mae"]),
        float(row["mean_rmse"]),
        float(row["std_mae"]),
        str(row["model"]),
    )


def _rank_results(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked = ranked.sort_values(
        by=["mean_mae", "mean_rmse", "std_mae", "model"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def _map_model_to_hpo_algorithm(model_name: str) -> str | None:
    mapping = {
        "baseline": "random_forest",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
    }
    return mapping.get(model_name)


def _train_selected_model(
    country_code: str,
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    trainer_config: dict[str, Any] | None,
    train_start: str,
    train_end: str,
) -> tuple[str, dict[str, Any]]:
    run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    split_idx = int(len(x) * 0.8)
    if split_idx < 1:
        raise ValueError("Insufficient data to train selected champion")

    x_train, x_val = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    trainer = get_trainer(
        country_code,
        model_name=model_name,
        config=cast("ModelConfig | None", trainer_config),
    )
    metrics = trainer.train(x_train, y_train, x_val, y_val)
    metrics["train_start_date"] = train_start
    metrics["train_end_date"] = train_end
    trainer.save(country_code, run_id, metrics=metrics)

    return run_id, metrics


async def main() -> None:
    parser = argparse.ArgumentParser(description="Select and register champion model")
    parser.add_argument("--country", required=True, help="Country code (e.g. PT)")
    parser.add_argument(
        "--start", required=True, help="Selection/training window start (YYYY-MM-DD)"
    )
    parser.add_argument("--end", required=True, help="Selection/training window end (YYYY-MM-DD)")
    parser.add_argument("--policy-file", help="Path to selection policy YAML")
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch raw data before clean/features/selection",
    )
    args = parser.parse_args()

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    if start_date > end_date:
        raise ValueError("start must be <= end")

    country_code = args.country.upper()

    setup_logging(level="INFO")
    auto_register_countries()

    policy, policy_path = _load_policy(country_code, args.policy_file)
    logger.info("Selection policy loaded for %s: %s", country_code, policy)

    pipeline = PipelineBuilder.create_pipeline(country_code)
    if args.fetch_data:
        await pipeline.fetch_data(args.start, args.end)

    pipeline.clean_and_verify(args.start, args.end)
    pipeline.engineer_features(args.start, args.end)

    x, y, features_path = _load_matrix(pipeline, args.start, args.end)

    comparison = ModelComparison(country_code=country_code, model_names=policy["allowed_models"])
    base_results = comparison.run(
        x=x,
        y=y,
        cv_method=policy["cv_method"],
        initial_train_size=int(policy["initial_train_size"]),
        step_size=int(policy["step_size"]),
        n_splits=int(policy["n_splits"]),
    )

    tuned_configs: dict[str, dict[str, Any]] = {}
    hpo_policy = dict(policy.get("hpo", {}))
    if bool(hpo_policy.get("enabled", False)):
        shortlisted = _rank_results(base_results).head(int(hpo_policy.get("shortlist_size", 1)))

        for _, row in shortlisted.iterrows():
            model_name = str(row["model"])
            algorithm = _map_model_to_hpo_algorithm(model_name)
            if algorithm is None:
                logger.info("Skipping HPO for unsupported model '%s'", model_name)
                continue

            try:
                from models.hpo import OptunaHPO
            except ImportError as exc:
                raise RuntimeError(
                    "HPO is enabled in policy but optional ML dependencies are missing. "
                    "Install with `pip install pricesentinel[ml]`."
                ) from exc

            logger.info("Running HPO for model '%s'", model_name)
            hpo = OptunaHPO(
                algorithm=algorithm,
                n_trials=int(hpo_policy.get("n_trials", 10)),
                cv_splits=int(hpo_policy.get("cv_splits", 3)),
                metric="mae",
            )
            best_params = hpo.optimize(x, y)
            tuned_configs[model_name] = {"hyperparameters": best_params}

            tuned_cmp = ModelComparison(
                country_code=country_code,
                model_names=[model_name],
                model_configs={model_name: tuned_configs[model_name]},
            )
            tuned_df = tuned_cmp.run(
                x=x,
                y=y,
                cv_method=policy["cv_method"],
                initial_train_size=int(policy["initial_train_size"]),
                step_size=int(policy["step_size"]),
                n_splits=int(policy["n_splits"]),
            )

            tuned_row = tuned_df.iloc[0]
            base_idx = base_results.index[base_results["model"] == model_name][0]
            base_row = base_results.loc[base_idx]

            if _ranking_key(tuned_row) < _ranking_key(base_row):
                logger.info("HPO improved model '%s'; using tuned metrics", model_name)
                for col in [
                    "mean_mae",
                    "std_mae",
                    "mean_rmse",
                    "std_rmse",
                    "train_time_s",
                    "inference_time_s",
                    "n_steps",
                    "status",
                ]:
                    base_results.loc[base_idx, col] = tuned_row[col]
            else:
                logger.info("HPO did not improve model '%s'; keeping base metrics", model_name)
                tuned_configs.pop(model_name, None)

    ranked = _rank_results(base_results)
    champion_row = ranked.iloc[0]
    champion_model = str(champion_row["model"])

    trainer_config = tuned_configs.get(champion_model)
    run_id, train_metrics = _train_selected_model(
        country_code=country_code,
        model_name=champion_model,
        x=x,
        y=y,
        trainer_config=trainer_config,
        train_start=args.start,
        train_end=args.end,
    )

    registry = ModelRegistry("models")
    champion_path = registry.set_champion(
        country_code=country_code,
        model_name=champion_model,
        run_id=run_id,
        trained_window={"start": args.start, "end": args.end},
        selection_metadata={
            "cv_method": policy["cv_method"],
            "initial_train_size": int(policy["initial_train_size"]),
            "step_size": int(policy["step_size"]),
            "n_splits": int(policy["n_splits"]),
            "policy_file": str(policy_path.as_posix()) if policy_path else None,
            "feature_file": str(features_path.as_posix()),
            "ranking": ranked.to_dict(orient="records"),
            "used_tuned_hyperparameters": bool(trainer_config),
        },
    )

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = args.end.replace("-", "")
    ranking_path = out_dir / f"champion_selection_{country_code}_{stamp}.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "country_code": country_code,
                "window": {"start": args.start, "end": args.end},
                "policy": policy,
                "ranked_results": ranked.to_dict(orient="records"),
                "champion": {
                    "model_name": champion_model,
                    "run_id": run_id,
                    "champion_path": str(champion_path.as_posix()),
                },
                "train_metrics": train_metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    sys.stdout.write(
        json.dumps(
            {
                "country_code": country_code,
                "champion_model": champion_model,
                "champion_run_id": run_id,
                "champion_file": str(champion_path.as_posix()),
                "ranking_file": str(ranking_path.as_posix()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
