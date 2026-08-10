"""Download the public AI4I 2020 dataset from the official UCI source."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

try:  # Works when imported by pytest from the repository root.
    from scripts import check_ai4i
except ImportError:  # pragma: no cover - works when executed as scripts/download_ai4i.py.
    import check_ai4i  # type: ignore[no-redef]

DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaint"
DATASET_ID = 601
DOI = "10.24432/C5HS5C"
LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
CSV_FILENAME = check_ai4i.DATASET_FILENAME
METADATA_FILENAME = "download_metadata.json"
NETWORK_TIMEOUT_SECONDS = 60


class Status(StrEnum):
    """Download status values."""

    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DownloadResult:
    """A single download/acquisition result."""

    name: str
    status: Status
    message: str


class DatasetValidator(Protocol):
    def __call__(self, path: Path) -> check_ai4i.ValidationReport: ...


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_directory(root: Path | None = None) -> Path:
    return (root or project_root()) / check_ai4i.DATASET_DIRECTORY


def dataset_path(root: Path | None = None) -> Path:
    return dataset_directory(root) / CSV_FILENAME


def metadata_path(root: Path | None = None) -> Path:
    return dataset_directory(root) / METADATA_FILENAME


def is_official_https_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    official = urllib.parse.urlparse(SOURCE_URL)
    return parsed.scheme == "https" and parsed.netloc == official.netloc and url == SOURCE_URL


def is_safe_zip_member_name(member_name: str) -> bool:
    if not member_name or member_name.endswith(("/", "\\")):
        return False
    if PureWindowsPath(member_name).is_absolute():
        return False
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    parts = PurePosixPath(normalized).parts
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)


def find_expected_csv_member(member_names: list[str]) -> str:
    candidates = [
        member_name
        for member_name in member_names
        if is_safe_zip_member_name(member_name)
        and PurePosixPath(member_name.replace("\\", "/")).name == CSV_FILENAME
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"Archive does not contain {CSV_FILENAME}.")
    raise ValueError(f"Archive contains multiple {CSV_FILENAME} members.")


def validate_zip_archive(zip_path: Path) -> str:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Downloaded file is not a valid ZIP archive.")
    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP archive failed integrity check at member: {bad_member}")
        return find_expected_csv_member(archive.namelist())


def download_source_to_file(
    url: str, destination: Path, timeout: int = NETWORK_TIMEOUT_SECONDS
) -> None:
    if not is_official_https_url(url):
        raise ValueError("Refusing to download from a non-official or non-HTTPS source URL.")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "industrial-fleet-intelligence/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status >= 400:
            raise OSError(f"Download failed with HTTP status {status}.")
        with destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)


def extract_csv_member(zip_path: Path, member_name: str, destination: Path) -> None:
    if not is_safe_zip_member_name(member_name):
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def local_dataset_is_valid(
    path: Path,
    validator: DatasetValidator = check_ai4i.validate_dataset_file,
) -> tuple[bool, check_ai4i.ValidationReport]:
    report = validator(path)
    return report.is_valid, report


def read_metadata(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as metadata_file:
            data = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def write_metadata(path: Path, zip_sha256: str, csv_sha256: str) -> None:
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_id": str(DATASET_ID),
        "source_url": SOURCE_URL,
        "dataset_page": DATASET_PAGE,
        "doi": DOI,
        "license": LICENSE,
        "downloaded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "zip_sha256": zip_sha256,
        "csv_sha256": csv_sha256,
        "dataset_filename": CSV_FILENAME,
    }
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


def run_download(force: bool = False) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    target_csv = dataset_path()
    target_metadata = metadata_path()

    if target_csv.exists() and not force:
        is_valid, report = local_dataset_is_valid(target_csv)
        if is_valid:
            metadata = read_metadata(target_metadata)
            zip_sha = metadata.get("zip_sha256", "not recorded")
            results.extend(
                [
                    DownloadResult(
                        "Local Dataset",
                        Status.PASS,
                        "Existing AI4I CSV is structurally valid; skipping download.",
                    ),
                    DownloadResult("ZIP SHA-256", Status.PASS, zip_sha),
                    DownloadResult("CSV SHA-256", Status.PASS, report.csv_sha256 or "not recorded"),
                ]
            )
            return results
        results.append(
            DownloadResult(
                "Local Dataset",
                Status.FAIL,
                "Existing AI4I CSV is not structurally valid; use --force to replace it.",
            )
        )
        return results

    try:
        with tempfile.TemporaryDirectory(prefix="ai4i_download_") as temporary_directory:
            temp_dir = Path(temporary_directory)
            zip_path = temp_dir / "ai4i.zip"
            temp_csv = temp_dir / CSV_FILENAME

            download_source_to_file(SOURCE_URL, zip_path)
            zip_sha256 = check_ai4i.calculate_sha256(zip_path)

            results.append(DownloadResult("Download", Status.PASS, "Downloaded official UCI ZIP."))
            results.append(DownloadResult("ZIP SHA-256", Status.PASS, zip_sha256))

            member_name = validate_zip_archive(zip_path)
            results.append(DownloadResult("ZIP Validation", Status.PASS, f"Found {member_name}."))

            extract_csv_member(zip_path, member_name, temp_csv)
            validation_report = check_ai4i.validate_dataset_file(temp_csv)
            if not validation_report.is_valid or validation_report.csv_sha256 is None:
                failed = [
                    result.message
                    for result in validation_report.results
                    if result.status is check_ai4i.Status.FAIL
                ]
                message = (
                    "; ".join(failed) if failed else "Extracted CSV failed structural validation."
                )
                results.append(DownloadResult("CSV Validation", Status.FAIL, message))
                return results

            target_csv.parent.mkdir(parents=True, exist_ok=True)
            temp_final = target_csv.with_name(target_csv.name + ".tmp")
            if temp_final.exists():
                temp_final.unlink()
            shutil.move(str(temp_csv), str(temp_final))
            temp_final.replace(target_csv)
            write_metadata(target_metadata, zip_sha256, validation_report.csv_sha256)

            results.append(DownloadResult("Extraction", Status.PASS, f"Wrote {target_csv}."))
            results.append(DownloadResult("CSV SHA-256", Status.PASS, validation_report.csv_sha256))
            results.append(
                DownloadResult("Metadata", Status.PASS, f"Wrote {target_metadata.name}.")
            )
            return results
    except Exception as exc:
        results.append(DownloadResult("Download", Status.FAIL, str(exc)))
        return results


def print_report(results: Sequence[DownloadResult]) -> None:
    print("Industrial Fleet Intelligence Platform AI4I dataset download")
    print()

    name_width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.status.value:<4} {result.name:<{name_width}} {result.message}")

    pass_count = sum(1 for result in results if result.status is Status.PASS)
    fail_count = sum(1 for result in results if result.status is Status.FAIL)

    print()
    print(f"Summary: {pass_count} PASS, {fail_count} FAIL")


def exit_code_for(results: Sequence[DownloadResult]) -> int:
    return 1 if any(result.status is Status.FAIL for result in results) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public AI4I 2020 dataset from UCI.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the official UCI source again and replace the local CSV if valid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_download(force=args.force)
    print_report(results)
    return exit_code_for(results)


if __name__ == "__main__":
    raise SystemExit(main())
