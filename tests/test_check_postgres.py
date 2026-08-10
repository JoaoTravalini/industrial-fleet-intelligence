from scripts import check_postgres as postgres


def test_parse_compose_services_strips_blank_lines():
    output = "\npostgres\n\n"

    assert postgres.parse_compose_services(output) == ["postgres"]


def test_compose_has_postgres_service():
    output = "postgres\n"

    assert postgres.compose_has_service(output)


def test_compose_has_service_rejects_missing_postgres():
    output = "api\nweb\n"

    assert not postgres.compose_has_service(output)


def test_parse_container_id_uses_first_non_empty_line():
    output = "\nabc123\ndef456\n"

    assert postgres.parse_container_id(output) == "abc123"


def test_parse_container_id_returns_none_for_empty_output():
    assert postgres.parse_container_id("\n") is None


def test_container_running_detection():
    assert postgres.is_container_running("running\n")
    assert not postgres.is_container_running("exited\n")


def test_health_state_parsing():
    assert postgres.parse_health_state("healthy\n") == "healthy"
    assert postgres.parse_health_state("") is None


def test_container_healthy_detection():
    assert postgres.is_container_healthy("healthy")
    assert not postgres.is_container_healthy("starting")
    assert not postgres.is_container_healthy("unhealthy")


def test_pg_isready_accepting_connections_passes():
    output = "/var/run/postgresql:5432 - accepting connections"

    assert postgres.pg_isready_passed(0, output)


def test_pg_isready_nonzero_fails():
    output = "/var/run/postgresql:5432 - no response"

    assert not postgres.pg_isready_passed(2, output)


def test_select_one_success_detection():
    assert postgres.select_one_passed(0, "1\n")
    assert postgres.select_one_passed(0, "\n1\n")


def test_select_one_rejects_wrong_output():
    assert not postgres.select_one_passed(0, "2\n")
    assert not postgres.select_one_passed(1, "1\n")


def test_exit_code_uses_mandatory_failures():
    passing = [postgres.CheckResult("A", postgres.Status.PASS, "ok")]
    warning = [postgres.CheckResult("A", postgres.Status.WARN, "check")]
    failing = [postgres.CheckResult("A", postgres.Status.FAIL, "bad")]

    assert postgres.exit_code_for(passing) == 0
    assert postgres.exit_code_for(warning) == 0
    assert postgres.exit_code_for(failing) == 1
