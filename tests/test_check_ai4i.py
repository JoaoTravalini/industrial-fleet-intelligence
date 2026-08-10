import csv

from scripts import check_ai4i as ai4i


def write_csv(path, rows, header=None):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header or ai4i.EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(udi=1):
    return {
        "UDI": str(udi),
        "Product ID": f"L{udi:05d}",
        "Type": "L",
        "Air temperature [K]": "298.1",
        "Process temperature [K]": "308.6",
        "Rotational speed [rpm]": "1551",
        "Torque [Nm]": "42.8",
        "Tool wear [min]": "0",
        "Machine failure": "0",
        "TWF": "0",
        "HDF": "0",
        "PWF": "0",
        "OSF": "0",
        "RNF": "0",
    }


def test_parse_integer_rejects_decimal_values():
    assert ai4i.parse_integer("1551") == 1551
    assert ai4i.parse_integer("1551.0") is None
    assert ai4i.parse_integer("") is None


def test_parse_numeric_rejects_non_finite_values():
    assert ai4i.parse_numeric("298.1") == 298.1
    assert ai4i.parse_numeric("nan") is None
    assert ai4i.parse_numeric("not-number") is None


def test_is_binary_value_accepts_only_zero_or_one():
    assert ai4i.is_binary_value("0")
    assert ai4i.is_binary_value("1")
    assert not ai4i.is_binary_value("2")


def test_evaluate_header_accepts_expected_columns():
    results = ai4i.evaluate_header(ai4i.EXPECTED_COLUMNS)

    assert all(result.status is ai4i.Status.PASS for result in results)


def test_evaluate_header_rejects_missing_column():
    header = [column for column in ai4i.EXPECTED_COLUMNS if column != "RNF"]

    results = ai4i.evaluate_header(header)

    assert any(result.status is ai4i.Status.FAIL for result in results)


def test_evaluate_row_count_requires_expected_count():
    assert ai4i.evaluate_row_count(10_000).status is ai4i.Status.PASS
    assert ai4i.evaluate_row_count(9999).status is ai4i.Status.FAIL


def test_evaluate_identifier_values_requires_unique_range():
    assert ai4i.evaluate_identifier_values(list(range(1, 10_001)), 0).status is ai4i.Status.PASS
    assert ai4i.evaluate_identifier_values([1, 1], 0).status is ai4i.Status.FAIL
    assert ai4i.evaluate_identifier_values([], 1).status is ai4i.Status.FAIL


def test_validate_dataset_file_reports_missing_file(tmp_path):
    report = ai4i.validate_dataset_file(tmp_path / "missing.csv")

    assert not report.is_valid
    assert report.results[0].status is ai4i.Status.FAIL


def test_validate_dataset_file_detects_invalid_binary_value(tmp_path):
    row = valid_row()
    row["Machine failure"] = "2"
    path = tmp_path / "ai4i2020.csv"
    write_csv(path, [row])

    report = ai4i.validate_dataset_file(path)

    assert not report.is_valid
    assert any(
        result.name == "Binary Values" and result.status is ai4i.Status.FAIL
        for result in report.results
    )


def test_validate_dataset_file_detects_invalid_type_value(tmp_path):
    row = valid_row()
    row["Type"] = "X"
    path = tmp_path / "ai4i2020.csv"
    write_csv(path, [row])

    report = ai4i.validate_dataset_file(path)

    assert not report.is_valid
    assert any(
        result.name == "Type" and result.status is ai4i.Status.FAIL for result in report.results
    )


def test_validate_dataset_file_detects_empty_fields(tmp_path):
    row = valid_row()
    row["Product ID"] = ""
    path = tmp_path / "ai4i2020.csv"
    write_csv(path, [row])

    report = ai4i.validate_dataset_file(path)

    assert not report.is_valid
    assert any(
        result.name == "Empty Fields" and result.status is ai4i.Status.FAIL
        for result in report.results
    )
    assert any(
        result.name == "Product ID" and result.status is ai4i.Status.FAIL
        for result in report.results
    )
