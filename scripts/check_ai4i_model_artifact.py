"""Read-only validator for the packaged AI4I final model artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation import ai4i_final_evaluation  # noqa: E402
from ml.inference import ai4i_predictor  # noqa: E402


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return data


def is_git_ignored(path: Path) -> bool:
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
        timeout=10,
    )
    return completed.returncode == 0


def source_has_restricted_split_reference(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "test.csv" in text or "TEST_RELATIVE_PATH" in text


def validate_artifacts() -> ValidationReport:
    results: list[CheckResult] = []
    model_path = ai4i_predictor.artifact_path(PROJECT_ROOT)
    metadata_path = ai4i_predictor.artifact_metadata_path(PROJECT_ROOT)
    summary_path = ai4i_predictor.packaging_summary_path(PROJECT_ROOT)
    sample_path = ai4i_predictor.sample_input_path(PROJECT_ROOT)

    for name, path in [
        ("Model Artifact", model_path),
        ("Artifact Metadata", metadata_path),
        ("Tracked Packaging Summary", summary_path),
        ("Sample Inference Payload", sample_path),
    ]:
        results.append(
            result(
                name,
                path.exists() and path.stat().st_size > 0,
                f"{path.name} exists.",
                f"{path.name} is missing or empty.",
            )
        )
    if any(item.status is Status.FAIL for item in results):
        return ValidationReport(results)

    try:
        final_config = ai4i_predictor.load_final_config(PROJECT_ROOT)
        final_config_hash = ai4i_predictor.current_final_config_hash(final_config)
        metrics = ai4i_final_evaluation.load_json(
            ai4i_final_evaluation.final_test_metrics_path(PROJECT_ROOT)
        )
        decision = ai4i_final_evaluation.load_json(
            ai4i_final_evaluation.final_model_decision_path(PROJECT_ROOT)
        )
        metadata = ai4i_predictor.load_artifact_metadata(metadata_path)
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        results.append(CheckResult("Readable Metadata", Status.FAIL, str(exc)))
        return ValidationReport(results)

    artifact_hash = ai4i_predictor.file_sha256(model_path)
    results.extend(
        [
            result(
                "Artifact Binary SHA-256",
                metadata.get("model_artifact_sha256") == artifact_hash,
                "Artifact binary hash matches local metadata.",
                "Artifact binary hash does not match local metadata.",
            ),
            result(
                "Current Final Config Hash",
                metadata.get("final_config_hash") == final_config_hash,
                "Metadata config hash matches the current frozen config.",
                "Metadata config hash does not match the current frozen config.",
            ),
            result(
                "Final Evaluation Provenance Hash",
                metrics.get("final_model_configuration_hash") == final_config_hash
                and decision.get("final_decision", {}).get("configuration_hash")
                == final_config_hash,
                "Final config hash matches previous final evaluation provenance.",
                "Final config hash does not match previous final evaluation provenance.",
            ),
            result(
                "Model Name",
                metadata.get("model_name") == ai4i_predictor.MODEL_NAME,
                "Metadata model name is expected.",
                "Metadata model name is not expected.",
            ),
            result(
                "Model Version",
                metadata.get("model_version") == ai4i_predictor.MODEL_VERSION,
                "Metadata model version is expected.",
                "Metadata model version is not expected.",
            ),
            result(
                "Frozen Threshold",
                metadata.get("decision_threshold")
                == ai4i_final_evaluation.FROZEN_DECISION_THRESHOLD,
                "Metadata threshold is exactly 0.14.",
                "Metadata threshold is not 0.14.",
            ),
            result(
                "Tracked Summary Consistency",
                summary.get("model_name") == metadata.get("model_name")
                and summary.get("model_version") == metadata.get("model_version")
                and summary.get("final_config_hash") == metadata.get("final_config_hash")
                and summary.get("serialization_format") == metadata.get("serialization_format")
                and summary.get("training_row_count") == metadata.get("training_row_count")
                and summary.get("training_positive_count")
                == metadata.get("training_positive_count")
                and summary.get("test_data_used_for_packaging") is False,
                "Tracked packaging summary is consistent with local metadata.",
                "Tracked packaging summary is not consistent with local metadata.",
            ),
            result(
                "Git Ignore Model Artifact",
                is_git_ignored(model_path),
                "Model binary is protected by Git ignore rules.",
                "Model binary is not protected by Git ignore rules.",
            ),
            result(
                "Git Ignore Artifact Metadata",
                is_git_ignored(metadata_path),
                "Local artifact metadata is protected by Git ignore rules.",
                "Local artifact metadata is not protected by Git ignore rules.",
            ),
            result(
                "Metadata Excludes Test Metrics",
                not ai4i_predictor.metadata_contains_test_metrics(metadata),
                "Local metadata contains no test metrics.",
                "Local metadata contains test metrics.",
            ),
            result(
                "Packaging Source Guard",
                not source_has_restricted_split_reference(
                    PROJECT_ROOT / "scripts" / "package_ai4i_final_model.py"
                )
                and not source_has_restricted_split_reference(
                    PROJECT_ROOT / "ml" / "inference" / "ai4i_predictor.py"
                ),
                "Packaging and inference sources do not reference the restricted split file.",
                "Packaging or inference source references the restricted split file.",
            ),
        ]
    )

    try:
        predictor = ai4i_predictor.load_predictor(PROJECT_ROOT)
        ai4i_predictor.validate_pipeline_structure(predictor.pipeline, final_config)
        sample_payload = ai4i_predictor.load_inference_payload(sample_path)
        sample_records = sample_payload if isinstance(sample_payload, list) else [sample_payload]
        predictions = predictor.predict_batch(sample_records)
        loaded_ok = True
        load_error = ""
    except (OSError, ValueError) as exc:
        loaded_ok = False
        load_error = str(exc)
        predictions = []

    results.append(
        result(
            "Model Loads",
            loaded_ok,
            "Packaged model loads successfully through the trusted loader.",
            load_error,
        )
    )
    if not loaded_ok:
        return ValidationReport(results)

    classifier = predictor.pipeline.named_steps["classifier"]
    classifier_params = classifier.get_params()
    hyperparameters_match = all(
        classifier_params[key] == expected
        for key, expected in final_config["hyperparameters"].items()
    )
    results.extend(
        [
            result(
                "Pipeline Structure",
                True,
                "Loaded object is the expected scikit-learn pipeline structure.",
                "Loaded object is not the expected pipeline structure.",
            ),
            result(
                "Random Forest Hyperparameters",
                hyperparameters_match,
                "Loaded Random Forest hyperparameters match the frozen config.",
                "Loaded Random Forest hyperparameters do not match the frozen config.",
            ),
            result(
                "Preprocessing Policy",
                ai4i_predictor.validate_pipeline_structure(predictor.pipeline, final_config)
                is predictor.pipeline,
                "Loaded preprocessing policy matches the frozen config.",
                "Loaded preprocessing policy does not match the frozen config.",
            ),
            result(
                "Sample Prediction Count",
                len(predictions) == 3,
                "Sample inference returns exactly three predictions.",
                "Sample inference does not return exactly three predictions.",
            ),
            result(
                "Sample Probability Bounds",
                all(0 <= item["failure_probability"] <= 1 for item in predictions),
                "Sample inference probabilities are within [0, 1].",
                "Sample inference probabilities contain values outside [0, 1].",
            ),
            result(
                "Sample Prediction Values",
                {item["failure_prediction"] for item in predictions}.issubset({0, 1}),
                "Sample inference predictions are binary.",
                "Sample inference predictions contain non-binary values.",
            ),
            result(
                "Sample Threshold",
                all(item["decision_threshold"] == 0.14 for item in predictions),
                "Sample inference outputs use threshold 0.14.",
                "Sample inference outputs do not use threshold 0.14.",
            ),
            result(
                "Sample Model Identity",
                all(
                    item["model_name"] == ai4i_predictor.MODEL_NAME
                    and item["model_version"] == ai4i_predictor.MODEL_VERSION
                    and item["final_config_hash"] == final_config_hash
                    for item in predictions
                ),
                "Sample inference outputs include expected model identity.",
                "Sample inference outputs contain unexpected model identity.",
            ),
        ]
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
    report = validate_artifacts()
    print_report(report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
