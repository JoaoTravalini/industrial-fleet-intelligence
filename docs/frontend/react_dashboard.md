# React Operational Dashboard

## Purpose

The React dashboard provides a polished, responsive, read-only operational view over the local Industrial Fleet Intelligence platform. It visualizes already-materialized FastAPI state and performs no browser-side model inference, anomaly scoring, drift calculation, SHAP processing, or direct database access.

## Architecture

The dashboard runs as a local Vite application and consumes the FastAPI API through HTTP. The data path is PostgreSQL materialized state -> FastAPI `/api/v1` -> typed fetch client -> TanStack Query hooks -> React routes and components.

## Stack

- React with TypeScript.
- Vite for development and build tooling.
- React Router for SPA navigation.
- TanStack Query for read-only server state.
- Browser `fetch` through a centralized typed API client.
- Custom CSS for the visual system.
- Vitest, jsdom, and Testing Library for unit tests.

## Local Development

Install frontend dependencies from the frontend workspace:

```powershell
cd apps\web
npm install
```

Run the dashboard locally:

```powershell
npm run dev
```

The Vite development server defaults to `http://localhost:5173`.

## API Integration

The dashboard consumes the existing local FastAPI service at `/health` and `/api/v1`. It expects the backend to expose read-only endpoints for fleet overview, machines, prediction history, anomaly history, drift monitoring, and alerts.

All operational data is loaded from the API at runtime. The frontend does not contain production fallback data, seeded machine lists, hard-coded fleet counts, or direct PostgreSQL credentials.

## Environment Configuration

The API base URL is configured with `VITE_API_BASE_URL` in `apps/web/.env` for local overrides. The tracked example is:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

`apps/web/.env` must not be committed.

## Routing

Implemented routes:

- `/` for the fleet overview dashboard.
- `/machines` for the server-paginated machine list.
- `/machines/:machineCode` for machine detail, recent predictions, and recent anomalies.
- `/alerts` for server-paginated operational alerts with supported filters.
- `/drift` for latest drift monitoring.
- Unknown routes render a frontend Not Found page.

## Overview

The overview page fetches `GET /api/v1/fleet/overview` and surfaces machine count, prediction projection coverage, prediction history count, model-positive prediction count, flagged anomaly count, open alert count, failure-probability summary, and latest drift statuses.

AI4I classifier outputs are labeled as model probabilities and decisions. They are not presented as observed failures, health scores, or factual future failures.

## Machines

The machine list fetches `GET /api/v1/machines` with server-side `limit` and `offset`. Pagination state is kept in URL search parameters, and machine codes link to their detail pages.

The page displays only fields supported by the backend contract: machine code, machine type, model family, commissioned date, operational status, latest failure probability, latest model decision, and latest prediction event time.

## Machine Detail

The machine detail page fetches the machine record plus small recent slices from the prediction and anomaly history endpoints. It shows identity, metadata, latest AI4I projection, history counts, recent prediction rows, and recent anomaly rows.

Anomaly score is displayed as a detector score, not a probability.

## Alerts

The alerts page fetches `GET /api/v1/alerts` with supported server-side filters for status, severity, alert type, and machine code. It displays severity, alert type, linked machine code, message, status, and source/event time.

The page is read-only and does not expose acknowledge, resolve, create, or delete controls.

## Drift Monitoring

The drift page fetches `GET /api/v1/drift/latest` and keeps the AI4I model-input scope separate from the operational anomaly-input scope. It displays overall status, feature name, feature type, PSI, feature status, reference/current counts, and concise diagnostics.

Distribution shift is described as input monitoring, not model performance measurement.

## Loading / Error / Empty States

Data-backed pages use reusable loading, API error, and empty-state components. Error states avoid stack traces and expose a safe Retry action for read-only requests.

## Responsive Design

The application shell uses a desktop sidebar and a mobile-friendly header/navigation layout. KPI grids, filter forms, detail sections, and tables reflow for tablet and mobile widths. Tables become label/value rows on narrow screens.

## Accessibility

The dashboard uses semantic navigation, headings, forms, tables with headers, accessible button labels, and text labels inside status badges. Color is not the only channel used for drift, severity, alert status, model decision, or anomaly flag semantics.

## Testing

Frontend unit tests run with Vitest in jsdom and Testing Library. They mock browser `fetch` at the API boundary and do not require FastAPI, PostgreSQL, Docker, Kafka, Spark, internet access, or any paid service.

Run tests from `apps/web`:

```powershell
npm run test
```

## Current Limitations

This first frontend phase intentionally omits advanced telemetry charts, SHAP visualizations, AI copilot features, authentication, frontend mutations, Dockerized frontend services, and deployment configuration.

## Future Visual Analytics

A later phase can add deliberate charting for prediction history, anomaly history, drift trends, and richer fleet analysis after the core API integration and dashboard architecture are stable.

## Future AI Copilot

A later phase can add a local Ollama-backed copilot. It must use read-only validated data access and must not fabricate telemetry, maintenance records, predictions, or machine state.
