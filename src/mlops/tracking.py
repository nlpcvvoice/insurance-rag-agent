from dataclasses import dataclass, field
from typing import Dict, Optional, List
import mlflow
from src.mlops.logging_setup import get_logger

def _sanitize_name(name: str) -> str:
    """Make a key MLflow-safe: replace disallowed chars with '_'."""
    return "".join(
        c if (c.isalnum() or c in "_-./ :") else "_"
        for c in name
    )


@dataclass
class ExperimentTracker:
    """Thin wrapper around MLflow for experiment tracking (local backend)."""
    tracking_uri: str = "./mlruns"
    experiment_name: str = "insurance-knowledge-assistant"
    _active: bool = False

    def __post_init__(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self._active = True
        except Exception as e:
            get_logger().warning(f"MLflow disabled: {e}")
            self._active = False

    def ensure_experiment(self) -> None:
        if not self._active:
            return
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            get_logger().warning(f"Failed to set experiment: {e}")

    def start_run(self, run_name: str = "", tags: Optional[Dict] = None) -> None:
        if not self._active:
            return
        self.ensure_experiment()
        try:
            mlflow.start_run(run_name=run_name or None)
            if tags:
                mlflow.set_tags(tags)
        except Exception as e:
            get_logger().warning(f"Failed to start mlflow run: {e}")

    def log_params(self, params: Dict) -> None:
        if not self._active:
            return
        try:
            mlflow.log_params(params)
        except Exception as e:
            get_logger().warning(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict) -> None:
        if not self._active:
            return
        try:
            mlflow.log_metrics(
                {_sanitize_name(k): v for k, v in metrics.items()}
            )
        except Exception as e:
            get_logger().warning(f"Failed to log metrics: {e}")

    def log_model_info(self, model_name: str, model_type: str) -> None:
        if not self._active:
            return
        try:
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_type", model_type)
        except Exception as e:
            get_logger().warning(f"Failed to log model info: {e}")

    def end_run(self) -> None:
        if not self._active:
            return
        try:
            mlflow.end_run()
        except Exception as e:
            get_logger().warning(f"Failed to end mlflow run: {e}")
