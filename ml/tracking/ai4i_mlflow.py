"""Local MLflow retrospective tracking utilities for AI4I history."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

TRACKING_CONFIG_RELATIVE_PATH = Path("ml") / "config" / "ai4i_mlflow_tracking.json"
MANIFEST_RELATIVE_PATH = Path("ml") / "config" / "ai4i_mlflow_manifest.json"
TRACKING_SUMMARY_RELATIVE_PATH = Path("reports") / "ai4i" / "mlflow_tracking_summary.json"
PROJECT_NAME = "industrial-fleet-intelligence"
EXPERIMENT_NAME = "industrial-fleet-ai4i"
DATASET_TAG = "AI4I-2020"
DATA_SOURCE_TAG = "UCI"
PORTFOLIO_DATA_TAG = "public_synthetic"
TRACKING_PROVENANCE = "retrospective_import"
TRACKING_MODE = "local_sqlite"
BACKEND_RELATIVE_PATH = ".mlflow/mlflow.db"
ARTIFACT_ROOT_RELATIVE_PATH = ".mlflow/artifacts"
MLFLOW_UI_HOST = "127.0.0.1"
MLFLOW_UI_PORT = 5000
DISALLOWED_ARTIFACT_PARTS = {
    ("data", "raw"),
    ("data", "processed"),
}
DISALLOWED_ARTIFACT_SUFFIXES = {".joblib", ".pkl", ".onnx"}
EXPECTED_RUN_KEYS = (
    "baseline_logistic_regression",
    "logistic_imbalance_strategy",
    "model_comparison_logistic",
    "model_comparison_random_forest",
    "model_comparison_xgboost",
    "random_forest_targeted_tuning",
    "final_holdout_evaluation",
    "final_model_packaging",
)


@dataclass(frozen=True)
class RunSpec:
    """A deterministic manifest entry for one retrospective MLflow run."""

    run_key: str
    run_name: str
    development_stage: str
    source_report: str
    source_artifacts: tuple[str, ...]
    test_data_used: bool
    tracking_provenance: str


@dataclass(frozen=True)
class PreparedRun:
    """A manifest run plus extracted source metadata."""

    spec: RunSpec
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]
    source_artifacts: tuple[Path, ...]


@dataclass(frozen=True)
class ImportResult:
    """Result from an idempotent retrospective import."""

    experiment_name: str
    expected_run_count: int
    imported_run_keys: tuple[str, ...]
    existing_run_keys: tuple[str, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracking_config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / TRACKING_CONFIG_RELATIVE_PATH


def manifest_path(root: Path | None = None) -> Path:
    return (root or project_root()) / MANIFEST_RELATIVE_PATH


def tracking_summary_path(root: Path | None = None) -> Path:
    return (root or project_root()) / TRACKING_SUMMARY_RELATIVE_PATH


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return data


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def canonical_hash(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tracking_config(root: Path | None = None) -> dict[str, Any]:
    return load_json(tracking_config_path(root))


def validate_relative_path(value: str, field_name: str) -> None:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"`{field_name}` must be project-relative.")
    if ".." in path.parts:
        raise ValueError(f"`{field_name}` must not contain parent traversal.")


def validate_tracking_config(config: Mapping[str, Any]) -> None:
    expected = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_mode": TRACKING_MODE,
        "backend_relative_path": BACKEND_RELATIVE_PATH,
        "artifact_root_relative_path": ARTIFACT_ROOT_RELATIVE_PATH,
        "historical_import_policy": TRACKING_PROVENANCE,
        "project": PROJECT_NAME,
    }
    errors = [
        f"`{key}` must be {expected_value!r}."
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    ]
    for field_name in ["backend_relative_path", "artifact_root_relative_path"]:
        value = config.get(field_name)
        if not isinstance(value, str):
            errors.append(f"`{field_name}` must be a string.")
            continue
        try:
            validate_relative_path(value, field_name)
        except ValueError as exc:
            errors.append(str(exc))
    serialized = json.dumps(config).lower()
    if any(token in serialized for token in ["password", "secret", "token", "username"]):
        errors.append("Tracking config must not contain secrets or user-specific fields.")
    if errors:
        raise ValueError("Invalid AI4I MLflow tracking config: " + " ".join(errors))


def backend_path(root: Path | None = None, config: Mapping[str, Any] | None = None) -> Path:
    config = config or load_tracking_config(root)
    return (root or project_root()) / str(config["backend_relative_path"])


def artifact_root_path(root: Path | None = None, config: Mapping[str, Any] | None = None) -> Path:
    config = config or load_tracking_config(root)
    return (root or project_root()) / str(config["artifact_root_relative_path"])


def tracking_uri(root: Path | None = None, config: Mapping[str, Any] | None = None) -> str:
    return f"sqlite:///{backend_path(root, config).as_posix()}"


def configure_tracking(root: Path | None = None) -> tuple[MlflowClient, str, dict[str, Any]]:
    root_path = root or project_root()
    config = load_tracking_config(root_path)
    validate_tracking_config(config)
    backend = backend_path(root_path, config)
    artifacts = artifact_root_path(root_path, config)
    backend.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri(root_path, config)
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri), uri, config


def get_or_create_experiment(client: MlflowClient, config: Mapping[str, Any], root: Path) -> str:
    experiment = client.get_experiment_by_name(str(config["experiment_name"]))
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(
        str(config["experiment_name"]),
        artifact_location=artifact_root_path(root, config).as_uri(),
    )


def load_manifest(root: Path | None = None) -> dict[str, Any]:
    return load_json(manifest_path(root))


def run_spec_from_mapping(mapping: Mapping[str, Any]) -> RunSpec:
    source_artifacts = mapping.get("source_artifacts", [])
    if not isinstance(source_artifacts, list) or not all(
        isinstance(item, str) for item in source_artifacts
    ):
        raise ValueError("Manifest `source_artifacts` must be a list of strings.")
    return RunSpec(
        run_key=str(mapping["run_key"]),
        run_name=str(mapping["run_name"]),
        development_stage=str(mapping["development_stage"]),
        source_report=str(mapping["source_report"]),
        source_artifacts=tuple(source_artifacts),
        test_data_used=bool(mapping["test_data_used"]),
        tracking_provenance=str(mapping["tracking_provenance"]),
    )


def validate_manifest(manifest: Mapping[str, Any]) -> list[RunSpec]:
    if manifest.get("experiment_name") != EXPERIMENT_NAME:
        raise ValueError("Manifest experiment_name is not expected.")
    if manifest.get("historical_import_policy") != TRACKING_PROVENANCE:
        raise ValueError("Manifest historical import policy must be retrospective_import.")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Manifest `runs` must be a list.")
    specs = [run_spec_from_mapping(item) for item in runs]
    run_keys = [spec.run_key for spec in specs]
    if tuple(run_keys) != EXPECTED_RUN_KEYS:
        raise ValueError("Manifest run keys do not match the expected AI4I history.")
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("Manifest run keys must be unique.")
    for raw_run in runs:
        if "metrics" in raw_run or "params" in raw_run:
            raise ValueError("Manifest must not contain manually copied metrics or params.")
    for spec in specs:
        if spec.tracking_provenance != TRACKING_PROVENANCE:
            raise ValueError(f"Run `{spec.run_key}` must use retrospective_import provenance.")
        if spec.run_key == "final_holdout_evaluation" and not spec.test_data_used:
            raise ValueError("Final holdout run must declare test_data_used=true.")
        if spec.run_key != "final_holdout_evaluation" and spec.test_data_used:
            raise ValueError(f"Run `{spec.run_key}` must declare test_data_used=false.")
    return specs


def safe_project_relative_path(root: Path, relative_path: str) -> Path:
    validate_relative_path(relative_path, "source_artifact")
    path = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in path.parents and path != root_resolved:
        raise ValueError(f"Source artifact escapes project root: {relative_path}")
    return path


def is_disallowed_artifact_path(relative_path: str) -> bool:
    path = Path(relative_path)
    normalized_parts = tuple(part.lower() for part in path.parts)
    if any(normalized_parts[: len(parts)] == parts for parts in DISALLOWED_ARTIFACT_PARTS):
        return True
    return path.suffix.lower() in DISALLOWED_ARTIFACT_SUFFIXES


def validate_source_artifacts(root: Path, spec: RunSpec) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    for relative_path in spec.source_artifacts:
        if is_disallowed_artifact_path(relative_path):
            raise ValueError(f"Disallowed MLflow source artifact: {relative_path}")
        path = safe_project_relative_path(root, relative_path)
        if not path.exists():
            raise FileNotFoundError(f"MLflow source artifact is missing: {relative_path}")
        artifacts.append(path)
    if spec.source_report not in spec.source_artifacts:
        raise ValueError(f"Source report must be listed as a source artifact: {spec.run_key}")
    return tuple(artifacts)


def _scalar_param_value(value: Any) -> str | None:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        text = str(value)
        return text if len(text) <= 250 else None
    return None


def add_param(params: dict[str, str], name: str, value: Any) -> None:
    scalar = _scalar_param_value(value)
    if scalar is not None:
        params[name] = scalar


def add_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
        metrics[name] = float(value)


def threshold_metrics(prefix: str, source: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2"]:
        add_metric(metrics, f"{prefix}_{key}", source.get(key))
    return metrics


def baseline_run_payload(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    logistic = report["logistic_regression"]
    params: dict[str, str] = {
        "model_family": "LogisticRegression",
        "evaluation_split": "validation",
    }
    for key, value in logistic.get("configuration", {}).items():
        add_param(params, key, value)
    add_param(params, "random_seed", report.get("random_seed"))
    metrics: dict[str, float] = {}
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        add_metric(metrics, f"validation_{key}", logistic.get(key))
    return params, metrics


def imbalance_run_payload(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    selected_model = report["selected_model"]
    selected_threshold = report["selected_threshold"]
    selected_oof = report["train_oof_results"][selected_model]
    selected_candidate = report["threshold_candidates"][selected_model]["max_f2"]
    validation = report["validation_results"]["selected_threshold"]
    params: dict[str, str] = {
        "model_family": "LogisticRegression",
        "selected_model": str(selected_model),
        "threshold_source": "train_oof_max_f2",
    }
    add_param(params, "selected_threshold", selected_threshold)
    add_param(params, "cv_folds", report["cv_configuration"].get("n_splits"))
    metrics: dict[str, float] = {}
    add_metric(metrics, "oof_average_precision", selected_oof.get("average_precision"))
    add_metric(metrics, "oof_roc_auc", selected_oof.get("roc_auc"))
    metrics.update(threshold_metrics("oof_max_f2_threshold", selected_candidate))
    metrics.update(threshold_metrics("validation", validation))
    return params, metrics


def model_comparison_key(run_key: str) -> str:
    return {
        "model_comparison_logistic": "standard_logistic",
        "model_comparison_random_forest": "random_forest",
        "model_comparison_xgboost": "xgboost",
    }[run_key]


def model_comparison_payload(
    run_key: str,
    report: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, float]]:
    model_key = model_comparison_key(run_key)
    oof = report["train_oof_results"][model_key]
    candidate = report["threshold_candidates"][model_key]["max_f2"]
    model_config = report["model_configurations"][model_key]
    params: dict[str, str] = {
        "model_key": model_key,
        "comparison_primary_metric": str(report["candidate_selection_policy"]["primary_metric"]),
    }
    for key, value in model_config.items():
        add_param(params, key, value)
    add_param(params, "cv_folds", report["cv_configuration"].get("n_splits"))
    add_param(params, "selected_model", report.get("selected_model"))
    add_param(params, "selected_threshold", report.get("selected_threshold"))
    metrics: dict[str, float] = {}
    add_metric(metrics, "oof_average_precision", oof.get("average_precision"))
    add_metric(metrics, "oof_roc_auc", oof.get("roc_auc"))
    metrics.update(threshold_metrics("oof_max_f2_threshold", candidate))
    return params, metrics


def tuning_run_payload(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    promotion = report["promotion_policy"]
    params: dict[str, str] = {
        "model_family": "RandomForestClassifier",
        "tuning_strategy": "targeted_nested_cv",
        "selected_candidate": str(promotion["selected_candidate"]),
        "threshold_source": str(promotion["threshold_source"]),
    }
    add_param(params, "outer_cv_folds", report["nested_cv_configuration"]["outer"].get("n_splits"))
    add_param(params, "inner_cv_folds", report["nested_cv_configuration"]["inner"].get("n_splits"))
    add_param(params, "candidate_count", report["parameter_grid"].get("candidate_count"))
    add_param(params, "selected_threshold", promotion.get("selected_threshold"))
    add_param(params, "promotion_delta_required", promotion.get("promotion_delta_required"))
    metrics: dict[str, float] = {}
    tuned = report["tuned_nested_oof_results"]
    add_metric(metrics, "tuned_nested_oof_average_precision", tuned.get("average_precision"))
    add_metric(metrics, "tuned_nested_oof_roc_auc", tuned.get("roc_auc"))
    add_metric(metrics, "fixed_oof_average_precision", promotion.get("fixed_average_precision"))
    add_metric(metrics, "average_precision_delta", promotion.get("average_precision_delta"))
    metrics.update(
        threshold_metrics(
            "tuned_nested_oof_max_f2_threshold", report["threshold_candidates"]["max_f2"]
        )
    )
    metrics.update(
        threshold_metrics("validation", report["validation_results"]["selected_threshold"])
    )
    return params, metrics


def final_holdout_payload(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    params: dict[str, str] = {
        "model_family": str(report["model_family"]),
        "final_config_hash": str(report["final_model_configuration_hash"]),
        "training_data_policy": str(report["training_data_policy"]),
    }
    for key, value in report["hyperparameters"].items():
        add_param(params, key, value)
    add_param(params, "frozen_threshold", report.get("frozen_threshold"))
    add_param(
        params, "development_training_row_count", report.get("development_training_row_count")
    )
    add_param(params, "development_positive_count", report.get("development_positive_count"))
    add_param(params, "test_row_count", report.get("test_row_count"))
    add_param(params, "test_positive_count", report.get("test_positive_count"))
    metrics: dict[str, float] = {}
    test_metrics = report["test_metrics"]
    for key in ["average_precision", "roc_auc"]:
        add_metric(metrics, f"test_{key}", test_metrics["threshold_independent"].get(key))
    metrics.update(threshold_metrics("test_threshold_0_14", test_metrics["threshold_0_14"]))
    metrics.update(threshold_metrics("test_threshold_0_5", test_metrics["threshold_0_5"]))
    return params, metrics


def packaging_payload(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, float]]:
    params: dict[str, str] = {}
    for key in [
        "model_name",
        "model_version",
        "final_config_hash",
        "serialization_format",
        "python_version",
        "scikit_learn_version",
        "joblib_version",
        "relative_local_artifact_path",
    ]:
        add_param(params, key, report.get(key))
    add_param(params, "frozen_threshold", report.get("frozen_threshold"))
    metrics: dict[str, float] = {}
    add_metric(metrics, "packaging_training_row_count", report.get("training_row_count"))
    add_metric(metrics, "packaging_training_positive_count", report.get("training_positive_count"))
    return params, metrics


def extract_run_payload(
    spec: RunSpec, report: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, float]]:
    if spec.run_key == "baseline_logistic_regression":
        return baseline_run_payload(report)
    if spec.run_key == "logistic_imbalance_strategy":
        return imbalance_run_payload(report)
    if spec.run_key.startswith("model_comparison_"):
        return model_comparison_payload(spec.run_key, report)
    if spec.run_key == "random_forest_targeted_tuning":
        return tuning_run_payload(report)
    if spec.run_key == "final_holdout_evaluation":
        return final_holdout_payload(report)
    if spec.run_key == "final_model_packaging":
        return packaging_payload(report)
    raise ValueError(f"Unsupported AI4I MLflow run key: {spec.run_key}")


def common_tags(
    spec: RunSpec,
    source_report_sha256: str,
    run_spec_hash: str,
) -> dict[str, str]:
    tags = {
        "project": PROJECT_NAME,
        "dataset": DATASET_TAG,
        "data_source": DATA_SOURCE_TAG,
        "tracking_provenance": TRACKING_PROVENANCE,
        "portfolio_data": PORTFOLIO_DATA_TAG,
        "run_key": spec.run_key,
        "development_stage": spec.development_stage,
        "test_data_used": str(spec.test_data_used).lower(),
        "source_report": spec.source_report,
        "source_report_sha256": source_report_sha256,
        "run_spec_hash": run_spec_hash,
    }
    if spec.run_key == "final_holdout_evaluation":
        tags["model_frozen_before_evaluation"] = "true"
        tags["adaptive_test_selection"] = "false"
    return tags


def prepare_run(root: Path, spec: RunSpec) -> PreparedRun:
    source_artifacts = validate_source_artifacts(root, spec)
    report = load_json(safe_project_relative_path(root, spec.source_report))
    params, metrics = extract_run_payload(spec, report)
    source_report_sha = file_sha256(safe_project_relative_path(root, spec.source_report))
    run_spec_hash = canonical_hash(
        {
            "run_key": spec.run_key,
            "run_name": spec.run_name,
            "development_stage": spec.development_stage,
            "source_report": spec.source_report,
            "source_artifacts": list(spec.source_artifacts),
            "test_data_used": spec.test_data_used,
            "tracking_provenance": spec.tracking_provenance,
            "source_report_sha256": source_report_sha,
        }
    )
    return PreparedRun(
        spec=spec,
        params=params,
        metrics=metrics,
        tags=common_tags(spec, source_report_sha, run_spec_hash),
        source_artifacts=source_artifacts,
    )


def load_prepared_runs(root: Path | None = None) -> list[PreparedRun]:
    root_path = root or project_root()
    manifest = load_manifest(root_path)
    specs = validate_manifest(manifest)
    return [prepare_run(root_path, spec) for spec in specs]


def search_runs_by_key(client: MlflowClient, experiment_id: str, run_key: str) -> list[Run]:
    return client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_key = '{run_key}'",
        max_results=1000,
    )


def existing_run_conflicts(run: Run, prepared: PreparedRun) -> list[str]:
    conflicts: list[str] = []
    data = run.data
    for key in [
        "run_key",
        "tracking_provenance",
        "test_data_used",
        "source_report_sha256",
        "run_spec_hash",
    ]:
        if data.tags.get(key) != prepared.tags[key]:
            conflicts.append(key)
    return conflicts


def log_prepared_run(experiment_id: str, prepared: PreparedRun) -> None:
    with mlflow.start_run(experiment_id=experiment_id, run_name=prepared.spec.run_name):
        mlflow.set_tags(prepared.tags)
        mlflow.log_params(prepared.params)
        mlflow.log_metrics(prepared.metrics)
        for artifact in prepared.source_artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="source_artifacts")


def deterministic_tracking_summary(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    specs = validate_manifest(manifest)
    return {
        "experiment_name": str(config["experiment_name"]),
        "expected_run_count": len(specs),
        "expected_run_keys": [spec.run_key for spec in specs],
        "historical_import_policy": str(config["historical_import_policy"]),
        "project": PROJECT_NAME,
        "runtime_id_policy": (
            "MLflow experiment IDs and run IDs are environment-specific and intentionally "
            "excluded from tracked summaries."
        ),
        "tracking_architecture": {
            "artifact_root_relative_path": str(config["artifact_root_relative_path"]),
            "backend": "SQLite",
            "backend_relative_path": str(config["backend_relative_path"]),
            "mode": str(config["tracking_mode"]),
            "zero_cost": True,
        },
        "runs": [
            {
                "development_stage": spec.development_stage,
                "run_key": spec.run_key,
                "run_name": spec.run_name,
                "source_report": spec.source_report,
                "test_data_used": spec.test_data_used,
                "tracking_provenance": spec.tracking_provenance,
            }
            for spec in specs
        ],
    }


def write_tracking_summary(root: Path | None = None) -> dict[str, Any]:
    root_path = root or project_root()
    config = load_tracking_config(root_path)
    validate_tracking_config(config)
    manifest = load_manifest(root_path)
    summary = deterministic_tracking_summary(config, manifest)
    write_json(summary, tracking_summary_path(root_path))
    return summary


def import_historical_runs(root: Path | None = None) -> ImportResult:
    root_path = root or project_root()
    client, _uri, config = configure_tracking(root_path)
    experiment_id = get_or_create_experiment(client, config, root_path)
    prepared_runs = load_prepared_runs(root_path)
    imported: list[str] = []
    existing: list[str] = []
    for prepared in prepared_runs:
        matches = search_runs_by_key(client, experiment_id, prepared.spec.run_key)
        if len(matches) > 1:
            raise ValueError(f"Duplicate MLflow runs found for run_key `{prepared.spec.run_key}`.")
        if len(matches) == 1:
            conflicts = existing_run_conflicts(matches[0], prepared)
            if conflicts:
                raise ValueError(
                    f"Existing MLflow run for `{prepared.spec.run_key}` conflicts on: "
                    + ", ".join(conflicts)
                )
            existing.append(prepared.spec.run_key)
            continue
        log_prepared_run(experiment_id, prepared)
        imported.append(prepared.spec.run_key)
    write_tracking_summary(root_path)
    return ImportResult(
        experiment_name=str(config["experiment_name"]),
        expected_run_count=len(prepared_runs),
        imported_run_keys=tuple(imported),
        existing_run_keys=tuple(existing),
    )


def mlflow_ui_command() -> str:
    return (
        ".\\.venv\\Scripts\\mlflow.exe ui "
        f"--backend-store-uri sqlite:///{BACKEND_RELATIVE_PATH} "
        f"--default-artifact-root {ARTIFACT_ROOT_RELATIVE_PATH} "
        f"--host {MLFLOW_UI_HOST} --port {MLFLOW_UI_PORT}"
    )
