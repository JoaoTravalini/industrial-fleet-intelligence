from scripts import check_schema as schema


def test_parse_lines_strips_blank_lines_and_nul_characters():
    output = "\x00machines\n\nalerts\n"

    assert schema.parse_lines(output) == ["machines", "alerts"]


def test_parse_bool_accepts_common_psql_values():
    assert schema.parse_bool("t\n") is True
    assert schema.parse_bool("true\n") is True
    assert schema.parse_bool("1\n") is True
    assert schema.parse_bool("f\n") is False
    assert schema.parse_bool("false\n") is False
    assert schema.parse_bool("0\n") is False


def test_parse_bool_returns_none_for_unexpected_output():
    assert schema.parse_bool("maybe\n") is None
    assert schema.parse_bool("") is None


def test_missing_items_returns_expected_difference():
    assert schema.missing_items({"a", "b"}, {"b", "c"}) == {"a"}


def test_evaluate_expected_items_passes_when_complete():
    result = schema.evaluate_expected_items("Tables", {"machines"}, {"machines", "alerts"})

    assert result.status is schema.Status.PASS


def test_evaluate_expected_items_fails_with_missing_items():
    result = schema.evaluate_expected_items("Tables", {"machines", "alerts"}, {"machines"})

    assert result.status is schema.Status.FAIL
    assert "alerts" in result.message


def test_evaluate_forbidden_tables_passes_when_raw_telemetry_absent():
    result = schema.evaluate_forbidden_tables({"machines", "alerts"})

    assert result.status is schema.Status.PASS


def test_evaluate_forbidden_tables_fails_when_raw_telemetry_present():
    result = schema.evaluate_forbidden_tables({"machines", "raw_telemetry"})

    assert result.status is schema.Status.FAIL
    assert "raw_telemetry" in result.message


def test_exit_code_uses_mandatory_failures():
    passing = [schema.CheckResult("A", schema.Status.PASS, "ok")]
    warning = [schema.CheckResult("A", schema.Status.WARN, "check")]
    failing = [schema.CheckResult("A", schema.Status.FAIL, "bad")]

    assert schema.exit_code_for(passing) == 0
    assert schema.exit_code_for(warning) == 0
    assert schema.exit_code_for(failing) == 1
