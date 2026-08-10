"""Read-only structural validator for the AI4I 2020 dataset."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DATASET_DIRECTORY = Path("data") / "raw" / "ai4i"
DATASET_FILENAME = "ai4i2020.csv"
EXPECTED_ROW_COUNT = 10_000
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
EXPECTED_COLUMN_COUNT = len(EXPECTED_COLUMNS)
EXPECTED_TYPES = {"L", "M", "H"}
BINARY_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
PRIMARY_NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
INTEGER_COLUMNS = ["Rotational speed [rpm]", "Tool wear [min]"]


class Status(StrEnum):
    """Validation status values printed by the AI4I checker."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """A single AI4I validation result."""

    name: str
    status: Status
    message: str
    mandatory: bool = True


@dataclass(frozen=True)
class NumericRange:
    """Minimum and maximum values observed for a numeric column."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class DatasetProfile:
    """Concise factual profile for a structurally valid AI4I CSV file."""

    row_count: int
    column_count: int
    type_distribution: dict[str, int]
    machine_failure_distribution: dict[str, int]
    failure_mode_counts: dict[str, int]
    numeric_ranges: dict[str, NumericRange]


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation output for the AI4I dataset."""

    results: list[CheckResult]
    profile: DatasetProfile | None
    csv_sha256: str | None

    @property
    def is_valid(self) -> bool:
        return not any(result.status is Status.FAIL and result.mandatory for result in self.results)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_path(root: Path | None = None) -> Path:
    return (root or project_root()) / DATASET_DIRECTORY / DATASET_FILENAME


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_integer(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    signless = text[1:] if text.startswith("-") else text
    if not signless.isdigit():
        return None
    return int(text)


def parse_numeric(value: str) -> float | None:
    try:
        number = float(value.strip())
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def is_binary_value(value: str) -> bool:
    return value.strip() in {"0", "1"}


def format_number(value: float) -> str:
    return f"{value:g}"


def evaluate_header(header: Sequence[str]) -> list[CheckResult]:
    results = [
        CheckResult(
            "Column Count",
            Status.PASS if len(header) == EXPECTED_COLUMN_COUNT else Status.FAIL,
            (
                f"Found expected column count: {EXPECTED_COLUMN_COUNT}."
                if len(header) == EXPECTED_COLUMN_COUNT
                else f"Expected {EXPECTED_COLUMN_COUNT} columns, found {len(header)}."
            ),
        )
    ]

    header_set = set(header)
    expected_set = set(EXPECTED_COLUMNS)
    missing = expected_set - header_set
    unexpected = header_set - expected_set
    if not missing and not unexpected:
        results.append(
            CheckResult("Column Names", Status.PASS, "All expected columns are present.")
        )
    else:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        results.append(CheckResult("Column Names", Status.FAIL, "; ".join(details)))
    return results


def evaluate_row_count(row_count: int, expected: int = EXPECTED_ROW_COUNT) -> CheckResult:
    if row_count == expected:
        return CheckResult("Row Count", Status.PASS, f"Found expected row count: {expected}.")
    return CheckResult("Row Count", Status.FAIL, f"Expected {expected} rows, found {row_count}.")


def evaluate_identifier_values(values: Sequence[int], invalid_count: int) -> CheckResult:
    if invalid_count:
        return CheckResult("UDI", Status.FAIL, f"Found {invalid_count} non-integer UDI value(s).")

    expected_values = set(range(1, EXPECTED_ROW_COUNT + 1))
    actual_values = set(values)
    duplicate_count = len(values) - len(actual_values)
    missing = expected_values - actual_values
    unexpected = actual_values - expected_values
    if not duplicate_count and not missing and not unexpected:
        return CheckResult("UDI", Status.PASS, "UDI values are unique and cover 1 through 10000.")

    details: list[str] = []
    if duplicate_count:
        details.append(f"{duplicate_count} duplicate value(s)")
    if missing:
        details.append("missing " + ", ".join(str(value) for value in sorted(missing)[:5]))
    if unexpected:
        details.append("unexpected " + ", ".join(str(value) for value in sorted(unexpected)[:5]))
    return CheckResult("UDI", Status.FAIL, "; ".join(details))


def evaluate_invalid_count(name: str, invalid_count: int, pass_message: str) -> CheckResult:
    if invalid_count == 0:
        return CheckResult(name, Status.PASS, pass_message)
    return CheckResult(name, Status.FAIL, f"Found {invalid_count} invalid value(s).")


def evaluate_type_values(values: set[str]) -> CheckResult:
    unexpected = values - EXPECTED_TYPES
    if not unexpected:
        return CheckResult("Type", Status.PASS, "Type contains only L, M, and H.")
    return CheckResult(
        "Type", Status.FAIL, "Unexpected Type value(s): " + ", ".join(sorted(unexpected))
    )


def update_numeric_range(ranges: dict[str, NumericRange], column: str, value: float) -> None:
    existing = ranges.get(column)
    if existing is None:
        ranges[column] = NumericRange(value, value)
        return
    ranges[column] = NumericRange(min(existing.minimum, value), max(existing.maximum, value))


def validate_dataset_file(path: Path) -> ValidationReport:
    results: list[CheckResult] = []
    if not path.exists():
        return ValidationReport(
            [CheckResult("File Exists", Status.FAIL, f"Dataset file was not found: {path}")],
            None,
            None,
        )
    if not path.is_file():
        return ValidationReport(
            [CheckResult("File Exists", Status.FAIL, f"Dataset path is not a file: {path}")],
            None,
            None,
        )

    results.append(CheckResult("File Exists", Status.PASS, "Dataset CSV file exists."))

    try:
        csv_sha256 = calculate_sha256(path)
    except OSError as exc:
        return ValidationReport(
            [*results, CheckResult("CSV SHA-256", Status.FAIL, f"Could not hash CSV: {exc}")],
            None,
            None,
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            header = reader.fieldnames or []
            results.append(
                CheckResult("Readable CSV", Status.PASS, "CSV can be opened and parsed.")
            )
            results.extend(evaluate_header(header))

            has_expected_columns = set(EXPECTED_COLUMNS).issubset(set(header))
            row_count = 0
            empty_field_count = 0
            product_id_empty_count = 0
            udi_invalid_count = 0
            numeric_invalid_count = 0
            integer_invalid_count = 0
            binary_invalid_count = 0
            udi_values: list[int] = []
            type_values: set[str] = set()
            type_counter: Counter[str] = Counter()
            machine_failure_counter: Counter[str] = Counter()
            failure_mode_counts: Counter[str] = Counter()
            numeric_ranges: dict[str, NumericRange] = {}

            for row in reader:
                row_count += 1
                if not has_expected_columns:
                    continue

                for column in EXPECTED_COLUMNS:
                    raw_value = row.get(column)
                    if raw_value is None or raw_value.strip() == "":
                        empty_field_count += 1

                product_id = row.get("Product ID", "")
                if product_id.strip() == "":
                    product_id_empty_count += 1

                udi = parse_integer(row.get("UDI", ""))
                if udi is None:
                    udi_invalid_count += 1
                else:
                    udi_values.append(udi)

                machine_type = row.get("Type", "").strip()
                type_values.add(machine_type)
                type_counter[machine_type] += 1

                for column in PRIMARY_NUMERIC_COLUMNS:
                    number = parse_numeric(row.get(column, ""))
                    if number is None:
                        numeric_invalid_count += 1
                    else:
                        update_numeric_range(numeric_ranges, column, number)

                for column in INTEGER_COLUMNS:
                    if parse_integer(row.get(column, "")) is None:
                        integer_invalid_count += 1

                for column in BINARY_COLUMNS:
                    value = row.get(column, "").strip()
                    if not is_binary_value(value):
                        binary_invalid_count += 1
                        continue
                    if column == "Machine failure":
                        machine_failure_counter[value] += 1
                    elif value == "1":
                        failure_mode_counts[column] += 1

    except (csv.Error, OSError, UnicodeDecodeError) as exc:
        return ValidationReport(
            [*results, CheckResult("Readable CSV", Status.FAIL, f"Could not parse CSV: {exc}")],
            None,
            csv_sha256,
        )

    results.append(evaluate_row_count(row_count))

    if not has_expected_columns:
        return ValidationReport(results, None, csv_sha256)

    results.extend(
        [
            evaluate_invalid_count(
                "Empty Fields", empty_field_count, "No empty fields were found."
            ),
            evaluate_identifier_values(udi_values, udi_invalid_count),
            evaluate_invalid_count(
                "Product ID", product_id_empty_count, "Product ID values are non-empty."
            ),
            evaluate_type_values(type_values),
            evaluate_invalid_count(
                "Numeric Values", numeric_invalid_count, "Primary numeric columns are numeric."
            ),
            evaluate_invalid_count(
                "Integer Values", integer_invalid_count, "Integer columns contain integers."
            ),
            evaluate_invalid_count(
                "Binary Values", binary_invalid_count, "Binary columns contain only 0 and 1."
            ),
        ]
    )

    profile = DatasetProfile(
        row_count=row_count,
        column_count=len(header),
        type_distribution={key: type_counter[key] for key in sorted(type_counter)},
        machine_failure_distribution={
            key: machine_failure_counter[key] for key in sorted(machine_failure_counter)
        },
        failure_mode_counts={
            column: failure_mode_counts[column] for column in FAILURE_MODE_COLUMNS
        },
        numeric_ranges={column: numeric_ranges[column] for column in PRIMARY_NUMERIC_COLUMNS},
    )
    return ValidationReport(results, profile, csv_sha256)


def print_profile(profile: DatasetProfile, csv_sha256: str) -> None:
    print()
    print("Profile:")
    print(f"CSV SHA-256: {csv_sha256}")
    print(f"Rows: {profile.row_count}")
    print(f"Columns: {profile.column_count}")
    print("Type distribution:")
    for key in ["H", "L", "M"]:
        if key in profile.type_distribution:
            print(f"  {key}: {profile.type_distribution[key]}")
    print("Machine failure distribution:")
    for key in ["0", "1"]:
        if key in profile.machine_failure_distribution:
            print(f"  {key}: {profile.machine_failure_distribution[key]}")
    print("Failure mode positive counts:")
    for column in FAILURE_MODE_COLUMNS:
        print(f"  {column}: {profile.failure_mode_counts[column]}")
    print("Numeric min/max:")
    for column in PRIMARY_NUMERIC_COLUMNS:
        value_range = profile.numeric_ranges[column]
        minimum = format_number(value_range.minimum)
        maximum = format_number(value_range.maximum)
        print(f"  {column}: {minimum} / {maximum}")


def print_report(report: ValidationReport) -> None:
    print("Industrial Fleet Intelligence Platform AI4I dataset validation")
    print()

    name_width = max(len(result.name) for result in report.results)
    for result in report.results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")

    pass_count = sum(1 for result in report.results if result.status is Status.PASS)
    warn_count = sum(1 for result in report.results if result.status is Status.WARN)
    fail_count = sum(1 for result in report.results if result.status is Status.FAIL)

    print()
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

    if report.profile is not None and report.csv_sha256 is not None:
        print_profile(report.profile, report.csv_sha256)


def exit_code_for(report: ValidationReport) -> int:
    return (
        1
        if any(result.status is Status.FAIL and result.mandatory for result in report.results)
        else 0
    )


def main() -> int:
    try:
        report = validate_dataset_file(dataset_path())
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        print("Industrial Fleet Intelligence Platform AI4I dataset validation")
        print()
        print(f"FAIL Validator encountered an unexpected error: {exc}")
        return 2

    print_report(report)
    return exit_code_for(report)


if __name__ == "__main__":
    raise SystemExit(main())
