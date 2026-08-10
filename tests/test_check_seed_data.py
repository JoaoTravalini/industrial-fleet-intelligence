from scripts import check_seed_data as seed_data


def test_expected_machine_identifiers_cover_1_to_100():
    identifiers = seed_data.expected_machine_identifiers()

    assert len(identifiers) == 100
    assert identifiers[0] == "MCH-0001"
    assert identifiers[-1] == "MCH-0100"


def test_parse_lines_strips_blank_lines_and_nul_characters():
    output = "\x00MCH-0001\n\nMCH-0002\n"

    assert seed_data.parse_lines(output) == ["MCH-0001", "MCH-0002"]


def test_parse_count_requires_single_integer_line():
    assert seed_data.parse_count("100\n") == 100
    assert seed_data.parse_count("100\n101\n") is None
    assert seed_data.parse_count("not-a-count\n") is None


def test_parse_key_counts_returns_mapping():
    output = "active|85\nmaintenance|10\ninactive|5\n"

    assert seed_data.parse_key_counts(output) == {
        "active": 85,
        "maintenance": 10,
        "inactive": 5,
    }


def test_parse_key_counts_rejects_invalid_rows():
    try:
        seed_data.parse_key_counts("active\n")
    except ValueError as exc:
        assert "Could not parse key-count row" in str(exc)
    else:
        raise AssertionError("Expected invalid key-count row to raise ValueError.")


def test_evaluate_exact_count_passes_and_fails():
    passing = seed_data.evaluate_exact_count("Machine Count", 100, 100)
    failing = seed_data.evaluate_exact_count("Machine Count", 99, 100)

    assert passing.status is seed_data.Status.PASS
    assert failing.status is seed_data.Status.FAIL


def test_evaluate_identifier_range_accepts_exact_expected_set():
    result = seed_data.evaluate_identifier_range(seed_data.expected_machine_identifiers())

    assert result.status is seed_data.Status.PASS


def test_evaluate_identifier_range_reports_missing_identifier():
    identifiers = seed_data.expected_machine_identifiers()
    identifiers.remove("MCH-0050")

    result = seed_data.evaluate_identifier_range(identifiers)

    assert result.status is seed_data.Status.FAIL
    assert "MCH-0050" in result.message


def test_evaluate_unique_identifiers_rejects_duplicates():
    result = seed_data.evaluate_unique_identifiers(["MCH-0001", "MCH-0001"])

    assert result.status is seed_data.Status.FAIL
    assert "MCH-0001" in result.message


def test_evaluate_key_counts_requires_exact_distribution():
    passing = seed_data.evaluate_key_counts(
        "Status Counts",
        {"active": 85, "maintenance": 10, "inactive": 5},
        seed_data.EXPECTED_STATUS_COUNTS,
    )
    failing = seed_data.evaluate_key_counts(
        "Status Counts",
        {"active": 100},
        seed_data.EXPECTED_STATUS_COUNTS,
    )

    assert passing.status is seed_data.Status.PASS
    assert failing.status is seed_data.Status.FAIL


def test_evaluate_allowed_values_rejects_unexpected_values():
    passing = seed_data.evaluate_allowed_values(
        "Machine Categories",
        {"excavator", "generator"},
        seed_data.EXPECTED_CATEGORIES,
    )
    failing = seed_data.evaluate_allowed_values(
        "Machine Categories",
        {"excavator", "real_product_line"},
        seed_data.EXPECTED_CATEGORIES,
    )

    assert passing.status is seed_data.Status.PASS
    assert failing.status is seed_data.Status.FAIL
    assert "real_product_line" in failing.message


def test_evaluate_empty_tables_requires_all_other_tables_empty():
    passing = seed_data.evaluate_empty_tables(seed_data.EXPECTED_EMPTY_TABLE_COUNTS)
    failing_counts = dict(seed_data.EXPECTED_EMPTY_TABLE_COUNTS)
    failing_counts["alerts"] = 1
    failing = seed_data.evaluate_empty_tables(failing_counts)

    assert passing.status is seed_data.Status.PASS
    assert failing.status is seed_data.Status.FAIL


def test_exit_code_uses_mandatory_failures():
    passing = [seed_data.CheckResult("A", seed_data.Status.PASS, "ok")]
    warning = [seed_data.CheckResult("A", seed_data.Status.WARN, "check")]
    failing = [seed_data.CheckResult("A", seed_data.Status.FAIL, "bad")]

    assert seed_data.exit_code_for(passing) == 0
    assert seed_data.exit_code_for(warning) == 0
    assert seed_data.exit_code_for(failing) == 1
