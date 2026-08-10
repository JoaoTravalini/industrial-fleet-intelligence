from scripts import download_ai4i as download


class FakeReport:
    def __init__(self, is_valid: bool, csv_sha256: str | None = "abc"):
        self.is_valid = is_valid
        self.csv_sha256 = csv_sha256


def test_is_official_https_url_accepts_only_exact_source_url():
    assert download.is_official_https_url(download.SOURCE_URL)
    assert not download.is_official_https_url(download.SOURCE_URL.replace("https://", "http://"))
    assert not download.is_official_https_url("https://example.com/ai4i.zip")


def test_is_safe_zip_member_name_rejects_path_traversal():
    assert download.is_safe_zip_member_name("ai4i2020.csv")
    assert download.is_safe_zip_member_name("folder/ai4i2020.csv")
    assert not download.is_safe_zip_member_name("../ai4i2020.csv")
    assert not download.is_safe_zip_member_name("/ai4i2020.csv")
    assert not download.is_safe_zip_member_name(r"C:\temp\ai4i2020.csv")
    assert not download.is_safe_zip_member_name("folder/")


def test_find_expected_csv_member_accepts_single_safe_match():
    member = download.find_expected_csv_member(["readme.txt", "folder/ai4i2020.csv"])

    assert member == "folder/ai4i2020.csv"


def test_find_expected_csv_member_rejects_missing_file():
    try:
        download.find_expected_csv_member(["readme.txt"])
    except ValueError as exc:
        assert "does not contain" in str(exc)
    else:
        raise AssertionError("Expected missing ai4i2020.csv to raise ValueError.")


def test_find_expected_csv_member_rejects_multiple_matches():
    try:
        download.find_expected_csv_member(["ai4i2020.csv", "nested/ai4i2020.csv"])
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("Expected duplicate ai4i2020.csv members to raise ValueError.")


def test_calculate_sha256_uses_validator_helper(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("abc", encoding="utf-8")

    assert download.check_ai4i.calculate_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_local_dataset_is_valid_uses_injected_validator(tmp_path):
    path = tmp_path / "ai4i2020.csv"

    is_valid, report = download.local_dataset_is_valid(
        path, lambda candidate: FakeReport(True, "hash")
    )

    assert is_valid
    assert report.csv_sha256 == "hash"


def test_local_valid_dataset_skips_download(monkeypatch, tmp_path):
    csv_path = tmp_path / "data" / "raw" / "ai4i" / "ai4i2020.csv"
    metadata_path = csv_path.with_name("download_metadata.json")
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("placeholder", encoding="utf-8")
    metadata_path.write_text('{"zip_sha256":"ziphash"}', encoding="utf-8")

    monkeypatch.setattr(download, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        download,
        "local_dataset_is_valid",
        lambda path: (True, FakeReport(True, "csvhash")),
    )

    results = download.run_download(force=False)

    assert results[0].name == "Local Dataset"
    assert results[0].status is download.Status.PASS
    assert any(result.message == "ziphash" for result in results)
    assert any(result.message == "csvhash" for result in results)
