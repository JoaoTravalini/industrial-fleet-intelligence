from pathlib import Path

from scripts import seed_database as seed


def write_seed(path: Path, content: str = "SELECT 1;\n") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_validate_seed_filename_accepts_expected_pattern():
    assert seed.validate_seed_filename("001_development_fleet.sql")


def test_validate_seed_filename_rejects_unversioned_name():
    assert not seed.validate_seed_filename("development_fleet.sql")


def test_discover_seed_files_orders_by_filename(tmp_path):
    second = write_seed(tmp_path / "002_more_data.sql")
    first = write_seed(tmp_path / "001_development_fleet.sql")

    discovered = seed.discover_seed_files(tmp_path)

    assert [seed_file.path for seed_file in discovered] == [first, second]
    assert [seed_file.filename for seed_file in discovered] == [
        "001_development_fleet.sql",
        "002_more_data.sql",
    ]


def test_discover_seed_files_rejects_invalid_filename(tmp_path):
    write_seed(tmp_path / "bad-name.sql")

    try:
        seed.discover_seed_files(tmp_path)
    except ValueError as exc:
        assert "Invalid seed filename" in str(exc)
    else:
        raise AssertionError("Expected invalid seed filename to raise ValueError.")


def test_parse_bool_accepts_psql_values():
    assert seed.parse_bool("t\n") is True
    assert seed.parse_bool("false\n") is False
    assert seed.parse_bool("unexpected\n") is None


def test_parse_count_accepts_integer_output():
    assert seed.parse_count("100\n") == 100
    assert seed.parse_count("not-a-count\n") is None


def test_sql_literal_escapes_single_quotes():
    assert seed.sql_literal("owner's_seed.sql") == "'owner''s_seed.sql'"


def test_build_seed_transaction_wraps_sql(tmp_path):
    path = write_seed(
        tmp_path / "001_development_fleet.sql", "INSERT INTO machines DEFAULT VALUES;\n"
    )
    seed_file = seed.SeedFile("001_development_fleet.sql", path)

    sql = seed.build_seed_transaction(seed_file)

    assert "BEGIN;" in sql
    assert "INSERT INTO machines" in sql
    assert "COMMIT;" in sql
    assert sql.index("INSERT INTO machines") < sql.index("COMMIT;")


def test_exit_code_uses_failures():
    passing = [seed.CheckResult("A", seed.Status.PASS, "ok")]
    failing = [seed.CheckResult("A", seed.Status.FAIL, "bad")]

    assert seed.exit_code_for(passing) == 0
    assert seed.exit_code_for(failing) == 1
