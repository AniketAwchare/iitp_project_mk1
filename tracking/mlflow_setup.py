"""
tracking/mlflow_setup.py — MLflow experiment tracking helpers.
Wraps mlflow so modules only call log_metrics() / log_experiment().
"""
from __future__ import annotations

import mlflow
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def init_tracking(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking: uri=%s  experiment=%s", tracking_uri, experiment_name)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None,
                run_name: Optional[str] = None) -> str:
    """Start (or continue) a run and log a dict of scalar metrics. Returns run_id."""
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        flat = {k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None}
        mlflow.log_metrics(flat, step=step)
        return run.info.run_id


def log_experiment(
    name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts_dir: Optional[str] = None,
) -> str:
    """Log a full experiment run with params, metrics, and optional artifacts."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        flat = {k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None}
        mlflow.log_metrics(flat)
        if artifacts_dir:
            mlflow.log_artifacts(artifacts_dir)
        logger.info("MLflow run logged: %s  id=%s", name, run.info.run_id)
        return run.info.run_id
