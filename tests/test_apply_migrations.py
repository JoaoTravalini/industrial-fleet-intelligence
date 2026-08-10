from pathlib import Path

from scripts import apply_migrations as migrations


def write_migration(path: Path, content: str = "SELECT 1;\n") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_validate_migration_filename_accepts_expected_pattern():
    assert migrations.validate_migration_filename("001_initial_operational_schema.sql")


def test_validate_migration_filename_rejects_unversioned_name():
    assert not migrations.validate_migration_filename("initial_schema.sql")


def test_discover_migrations_orders_by_filename(tmp_path):
    second = write_migration(tmp_path / "002_second_change.sql")
    first = write_migration(tmp_path / "001_initial_operational_schema.sql")

    discovered = migrations.discover_migrations(tmp_path)

    assert [migration.path for migration in discovered] == [first, second]
    assert [migration.filename for migration in discovered] == [
        "001_initial_operational_schema.sql",
        "002_second_change.sql",
    ]


def test_discover_migrations_rejects_invalid_filename(tmp_path):
    write_migration(tmp_path / "bad-name.sql")

    try:
        migrations.discover_migrations(tmp_path)
    except ValueError as exc:
        assert "Invalid migration filename" in str(exc)
    else:
        raise AssertionError("Expected invalid migration filename to raise ValueError.")


def test_parse_applied_migrations_returns_filename_checksum_mapping():
    output = "001_initial_operational_schema.sql|abc123\n"

    assert migrations.parse_applied_migrations(output) == {
        "001_initial_operational_schema.sql": "abc123"
    }


def test_parse_applied_migrations_rejects_invalid_rows():
    try:
        migrations.parse_applied_migrations("not-a-valid-row\n")
    except ValueError as exc:
        assert "Could not parse applied migration row" in str(exc)
    else:
        raise AssertionError("Expected invalid applied migration output to raise ValueError.")


def test_pending_migrations_excludes_already_applied(tmp_path):
    first = migrations.Migration(
        "001_initial_operational_schema.sql",
        tmp_path / "001_initial_operational_schema.sql",
        "abc",
    )
    second = migrations.Migration(
        "002_second_change.sql", tmp_path / "002_second_change.sql", "def"
    )

    pending = migrations.pending_migrations(
        [first, second],
        {"001_initial_operational_schema.sql": "abc"},
    )

    assert pending == [second]


def test_detect_checksum_mismatches_reports_changed_applied_file(tmp_path):
    migration = migrations.Migration(
        "001_initial_operational_schema.sql",
        tmp_path / "001_initial_operational_schema.sql",
        "current",
    )

    assert migrations.detect_checksum_mismatches(
        [migration],
        {"001_initial_operational_schema.sql": "old"},
    ) == ["001_initial_operational_schema.sql"]


def test_sql_literal_escapes_single_quotes():
    assert migrations.sql_literal("owner's_migration.sql") == "'owner''s_migration.sql'"


def test_build_migration_transaction_records_after_sql(tmp_path):
    path = write_migration(
        tmp_path / "001_initial_operational_schema.sql", "CREATE TABLE x (id int);\n"
    )
    migration = migrations.Migration("001_initial_operational_schema.sql", path, "abc123")

    sql = migrations.build_migration_transaction(migration)

    assert sql.index("CREATE TABLE x") < sql.index("INSERT INTO schema_migrations")
    assert "BEGIN;" in sql
    assert "COMMIT;" in sql
    assert "'001_initial_operational_schema.sql'" in sql
