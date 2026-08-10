# Engineering Instructions

These instructions apply to the entire repository.

## Project Principles

### 1. Zero-cost architecture

- Never introduce a paid API or mandatory paid service.
- Prefer open-source software and local execution.
- No cloud service requiring billing information may be required.
- Databricks may only be used through Free Edition.
- The core application must remain functional without Databricks.
- Generative AI inference must use a local model through Ollama.
- Any optional external service must have a completely free alternative.

### 2. Incremental development

- Implement only the scope explicitly requested in each task.
- Never proactively implement future phases.
- Inspect the existing repository before editing.
- Do not refactor unrelated working code.
- Prefer small, reviewable changes.
- Stop after completing the requested phase.

### 3. Code quality

- New functionality must include appropriate tests.
- Run relevant tests after changes.
- Use Python type hints.
- Keep functions and modules focused.
- Avoid duplicated business logic.
- Prefer explicit configuration over hidden behavior.
- Fail clearly instead of silently swallowing exceptions.
- Follow established project formatting and linting rules once configured.

### 4. Security

- Never commit credentials, API keys, passwords, tokens, or personal secrets.
- Use environment variables for runtime configuration.
- Keep `.env` ignored by Git.
- Maintain `.env.example` when configuration changes.
- Validate external inputs.
- Apply least-privilege principles.
- Any database access exposed to the AI copilot must be read-only and validated.

### 5. Data engineering

- Never represent synthetic data as real industrial data.
- Clearly distinguish public datasets, generated telemetry, and derived data.
- Preserve dataset attribution and licensing information.
- Make random generation reproducible through explicit seeds where appropriate.
- Define schemas explicitly.
- Validate data before moving between Bronze, Silver, and Gold layers.
- Do not silently discard invalid records.

### 6. Machine learning

- Prevent data leakage.
- Establish a simple baseline before complex models.
- Use reproducible train/validation/test procedures.
- Record model parameters and metrics.
- Do not select a model solely because it is more complex.
- Document assumptions and limitations.
- Separate model predictions from business rules.
- Keep inference code consistent with training preprocessing.

### 7. Generative AI

- The AI copilot must never fabricate telemetry, maintenance records, predictions, or machine state.
- Distinguish deterministic system data, ML predictions, and LLM-generated explanations.
- Never execute arbitrary SQL generated directly by an LLM.
- Structured database access must be read-only, parameterized, and validated.
- Prefer retrieval and structured tools over unsupported generation.
- Local inference through Ollama is mandatory for the core project.

### 8. Documentation

- All source code, comments, documentation, variables, commit messages, and UI text must be in professional English.
- README must describe only functionality that actually exists.
- Planned features must be explicitly labeled as planned.
- Architecture documentation must remain synchronized with implementation.
- Significant architectural decisions should be documented.

### 9. Git

- Keep changes logically scoped.
- Do not modify generated or unrelated files unnecessarily.
- Never commit local environments, caches, model binaries, large datasets, secrets, or runtime artifacts.
- Do not rewrite Git history unless explicitly instructed.
- Use conventional-style commit messages where practical.

### 10. Project identity

- This is an independent portfolio project.
- Do not use Liebherr logos, branding, proprietary datasets, confidential information, or copyrighted internal material.
- Do not imply affiliation with or endorsement by Liebherr.
- It is acceptable to describe the project as inspired by industrial Data and AI challenges.