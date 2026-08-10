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

## Validate The Environment

Run the read-only environment validator from the repository root:

```powershell
py -3.12 scripts/check_environment.py
```

The validator uses only the Python standard library. It checks whether each required tool is available, reads installed versions where possible, reports `PASS`, `WARN`, or `FAIL`, and exits with a non-zero status when a mandatory requirement fails.

## Run Unit Tests

Run the unit tests with Python 3.12 and the standard library `unittest` module:

```powershell
py -3.12 -m unittest discover -s tests -p "test_*.py"
```

These tests cover pure version-parsing and requirement-checking logic. They do not depend on the developer machine's installed tools and do not mock or execute system software checks.