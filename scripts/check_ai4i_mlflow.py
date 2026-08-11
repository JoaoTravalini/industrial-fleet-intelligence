"""Read-only validator for local AI4I MLflow tracking state."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from mlflow.entities import Run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.tracking import ai4i_mlflow  # noqa: E402


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    mandatory: bool = True


@dataclass(frozen=True)
class ValidationReport:
    results: list[CheckResult]

    @property
    def is_valid(self) -> bool:
        return not any(result.status is Status.FAIL and result.mandatory for result in self.results)


def result(name: str, passed: bool, pass_message: str, fail_message: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=Status.PASS if passed else Status.FAIL,
        message=pass_message if passed else fail_message,
    )


def has_runtime_ids(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"experiment_id", "run_id", "tracking_uri"}:
                return True
            if has_runtime_ids(child):
                return True
    if isinstance(value, list):
        return any(has_runtime_ids(item) for item in value)
    return False


def recursively_list_artifacts(client: Any, run_id: str, path: str = "") -> list[str]:
    artifacts: list[str] = []
    for artifact in client.list_artifacts(run_id, path):
        artifacts.append(artifact.path)
        if artifact.is_dir:
            artifacts.extend(recursively_list_artifacts(client, run_id, artifact.path))
    return artifacts


def artifact_paths_are_safe(artifact_paths: list[str]) -> bool:
    for path in artifact_paths:
        normalized = Path(path)
        parts = tuple(part.lower() for part in normalized.parts)
        if ("data", "raw") in zip(parts, parts[1:], strict=False):
            return False
        if ("data", "processed") in zip(parts, parts[1:], strict=False):
            return False
        if normalized.suffix.lower() in ai4i_mlflow.DISALLOWED_ARTIFACT_SUFFIXES:
            return False
    return True


def metrics_match(run: Run, expected: dict[str, float]) -> bool:
    actual = run.data.metrics
    for key, expected_value in expected.items():
        if key not in actual or abs(float(actual[key]) - float(expected_value)) > 1e-9:
            return False
    return True


def params_match(run: Run, expected: dict[str, str]) -> bool:
    actual = run.data.params
    return all(actual.get(key) == str(value) for key, value in expected.items())


def expected_summary() -> dict[str, Any]:
    config = ai4i_mlflow.load_tracking_config(PROJECT_ROOT)
    manifest = ai4i_mlflow.load_manifest(PROJECT_ROOT)
    return ai4i_mlflow.deterministic_tracking_summary(config, manifest)


def validate_mlflow_state() -> ValidationReport:
    results: list[CheckResult] = []
    try:
        config = ai4i_mlflow.load_tracking_config(PROJECT_ROOT)
        ai4i_mlflow.validate_tracking_config(config)
        manifest = ai4i_mlflow.load_manifest(PROJECT_ROOT)
        specs = ai4i_mlflow.validate_manifest(manifest)
        prepared_runs = ai4i_mlflow.load_prepared_runs(PROJECT_ROOT)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ValidationReport([CheckResult("Tracking Configuration", Status.FAIL, str(exc))])

    backend = ai4i_mlflow.backend_path(PROJECT_ROOT, config)
    artifact_root = ai4i_mlflow.artifact_root_path(PROJECT_ROOT, config)
    results.extend(
        [
            result(
                "Tracking Configuration",
                True,
                "Local MLflow tracking config is valid.",
                "Local MLflow tracking config is invalid.",
            ),
            result(
                "SQLite Backend",
                backend.exists(),
                "SQLite MLflow database exists.",
                "SQLite MLflow database does not exist.",
            ),
            result(
                "Artifact Directory",
                artifact_root.exists() and artifact_root.is_dir(),
                "Local MLflow artifact directory exists.",
                "Local MLflow artifact directory does not exist.",
            ),
        ]
    )
    if not backend.exists():
        return ValidationReport(results)

    client, _uri, _config = ai4i_mlflow.configure_tracking(PROJECT_ROOT)
    experiment = client.get_experiment_by_name(ai4i_mlflow.EXPERIMENT_NAME)
    results.append(
        result(
            "Expected Experiment",
            experiment is not None,
            "Expected MLflow experiment exists.",
            "Expected MLflow experiment does not exist.",
        )
    )
    if experiment is None:
        return ValidationReport(results)

    expected_by_key = {prepared.spec.run_key: prepared for prepared in prepared_runs}
    runs_by_key: dict[str, list[Run]] = {
        spec.run_key: ai4i_mlflow.search_runs_by_key(
            client,
            experiment.experiment_id,
            spec.run_key,
        )
        for spec in specs
    }
    missing = [key for key, runs in runs_by_key.items() if not runs]
    duplicate = [key for key, runs in runs_by_key.items() if len(runs) > 1]
    results.extend(
        [
            result(
                "Expected Run Keys",
                not missing,
                "All expected semantic run keys exist.",
                "Missing run key(s): " + ", ".join(missing),
            ),
            result(
                "Duplicate Run Keys",
                not duplicate,
                "No duplicate semantic run keys exist.",
                "Duplicate run key(s): " + ", ".join(duplicate),
            ),
        ]
    )

    all_artifact_paths: list[str] = []
    for run_key, prepared in expected_by_key.items():
        runs = runs_by_key.get(run_key, [])
        if len(runs) != 1:
            continue
        run = runs[0]
        tags = run.data.tags
        expected_tags = prepared.tags
        results.extend(
            [
                result(
                    f"{run_key} Provenance Tag",
                    tags.get("tracking_provenance") == ai4i_mlflow.TRACKING_PROVENANCE,
                    "Run is tagged as retrospective_import.",
                    "Run is not tagged as retrospective_import.",
                ),
                result(
                    f"{run_key} Dataset Tags",
                    tags.get("dataset") == ai4i_mlflow.DATASET_TAG
                    and tags.get("data_source") == ai4i_mlflow.DATA_SOURCE_TAG
                    and tags.get("portfolio_data") == ai4i_mlflow.PORTFOLIO_DATA_TAG,
                    "Run dataset/source tags are correct.",
                    "Run dataset/source tags are incorrect.",
                ),
                result(
                    f"{run_key} test_data_used Tag",
                    tags.get("test_data_used") == expected_tags["test_data_used"],
                    "Run test_data_used tag is correct.",
                    "Run test_data_used tag is incorrect.",
                ),
                result(
                    f"{run_key} Parameters",
                    params_match(run, prepared.params),
                    "Logged parameters match tracked source/config.",
                    "Logged parameters do not match tracked source/config.",
                ),
                result(
                    f"{run_key} Metrics",
                    metrics_match(run, prepared.metrics),
                    "Logged key metrics match tracked source report.",
                    "Logged key metrics do not match tracked source report.",
                ),
            ]
        )
        all_artifact_paths.extend(recursively_list_artifacts(client, run.info.run_id))

    final_run = runs_by_key.get("final_holdout_evaluation", [])
    final_tags = final_run[0].data.tags if len(final_run) == 1 else {}
    results.append(
        result(
            "Final Holdout Frozen Tags",
            final_tags.get("model_frozen_before_evaluation") == "true"
            and final_tags.get("adaptive_test_selection") == "false",
            "Final holdout run is marked frozen-before-evaluation and non-adaptive.",
            "Final holdout run is missing frozen-before-evaluation safeguards.",
        )
    )

    packaging_run = runs_by_key.get("final_model_packaging", [])
    if len(packaging_run) == 1:
        summary = ai4i_mlflow.load_json(
            PROJECT_ROOT / "reports" / "ai4i" / "model_packaging_summary.json"
        )
        params = packaging_run[0].data.params
        packaging_matches = (
            params.get("model_name") == summary.get("model_name")
            and params.get("model_version") == summary.get("model_version")
            and params.get("final_config_hash") == summary.get("final_config_hash")
            and params.get("serialization_format") == summary.get("serialization_format")
            and params.get("joblib_version") == summary.get("joblib_version")
        )
    else:
        packaging_matches = False
    results.append(
        result(
            "Packaging Provenance",
            packaging_matches,
            "Packaging run metadata matches model_packaging_summary.json.",
            "Packaging run metadata does not match model_packaging_summary.json.",
        )
    )

    results.append(
        result(
            "Logged Artifact Scope",
            artifact_paths_are_safe(all_artifact_paths),
            "MLflow artifacts exclude raw/processed datasets and model binaries.",
            "MLflow artifacts include a raw/processed dataset or model binary.",
        )
    )

    try:
        registered_models = client.search_registered_models()
    except Exception:  # pragma: no cover - defensive for MLflow backend variations
        registered_models = []
    results.append(
        result(
            "Model Registry",
            len(registered_models) == 0,
            "No MLflow registered model exists.",
            "An MLflow registered model exists.",
        )
    )

    summary_path = ai4i_mlflow.tracking_summary_path(PROJECT_ROOT)
    try:
        tracked_summary = ai4i_mlflow.load_json(summary_path)
        summary_matches = tracked_summary == expected_summary() and not has_runtime_ids(
            tracked_summary
        )
    except (OSError, json.JSONDecodeError, ValueError):
        summary_matches = False
    results.append(
        result(
            "Tracked Summary",
            summary_matches,
            "Tracked MLflow summary matches manifest semantics and excludes runtime IDs.",
            "Tracked MLflow summary does not match manifest semantics or includes runtime IDs.",
        )
    )
    return ValidationReport(results)


def print_report(report: ValidationReport) -> None:
    for item in report.results:
        print(f"{item.status} {item.name}: {item.message}")
    pass_count = sum(1 for item in report.results if item.status is Status.PASS)
    warn_count = sum(1 for item in report.results if item.status is Status.WARN)
    fail_count = sum(1 for item in report.results if item.status is Status.FAIL)
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")


def main() -> int:
    report = validate_mlflow_state()
    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
