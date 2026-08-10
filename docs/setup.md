# Development Environment Setup

This project is currently in its repository-initialization stage. No project application dependencies have been installed yet.

## Required Tools

Use a local Windows development environment with the following tools:

- Windows as the host operating system.
- Python 3.12.x.
- Node.js 24.x LTS.
- npm available on the Windows PATH and compatible with Node.js 24.
- Java JDK 17 or newer, including both `java` and `javac`.
- Git 2.x or newer.
- WSL2 available on Windows.
- Docker Desktop installed.
- Docker Engine running with Linux containers.
- Docker Compose v2 through the `docker compose` command.

WSL2 is required because Docker Desktop uses it internally for Linux containers on Windows. Developers do not need to work inside Ubuntu or another Linux distribution for this project, and the environment validator does not require any manually installed Linux distribution.

## Python Development Environment

Create the project virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
```

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the declared development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Run the Python test suite with pytest:

```powershell
python -m pytest
```

Run Ruff lint checks:

```powershell
python -m ruff check .
```

Run the Ruff formatter check:

```powershell
python -m ruff format --check .
```

Apply Ruff formatting:

```powershell
python -m ruff format .
```

The `.venv` directory must never be committed. Developers should activate `.venv` before running project Python commands. No application or runtime dependencies have been introduced yet.

## PostgreSQL Local Infrastructure

Docker Desktop provides PostgreSQL for local development, so developers do not need to install PostgreSQL directly on Windows.

Ensure Docker Desktop is running, then create local environment configuration from the example:

```powershell
Copy-Item .env.example .env
```

The local `.env` values may then be edited for development. The `.env` file must never be committed.

Start only PostgreSQL:

```powershell
docker compose up -d postgres
```

Check service status:

```powershell
docker compose ps
```

Run infrastructure validation:

```powershell
python scripts/check_postgres.py
```

View PostgreSQL logs when troubleshooting:

```powershell
docker compose logs postgres
```

Stop the container without deleting data:

```powershell
docker compose stop postgres
```

Start it again:

```powershell
docker compose start postgres
```

Stop and remove containers while preserving the named volume:

```powershell
docker compose down
```

Use `docker compose down -v` only intentionally. It deletes the PostgreSQL development volume and all database data stored in that volume.

## Validate The Environment

Run the read-only environment validator from the repository root:

```powershell
py -3.12 scripts/check_environment.py
```

The validator uses only the Python standard library. It checks whether each required tool is available, reads installed versions where possible, reports `PASS`, `WARN`, or `FAIL`, and exits with a non-zero status when a mandatory requirement fails.

## Run Unit Tests

The project uses pytest for Python tests:

```powershell
python -m pytest
```

The current tests cover pure version-parsing and requirement-checking logic. They do not depend on the developer machine's installed tools and do not mock or execute system software checks.