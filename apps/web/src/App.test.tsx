import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  alertListFixture,
  anomalyListFixture,
  driftFixture,
  emptyAlertListFixture,
  fleetOverviewFixture,
  healthFixture,
  machineDetailFixture,
  machineListFixture,
  predictionListFixture,
} from './test/fixtures'
import { renderWithProviders } from './test/render'

type FetchMode = 'default' | 'overview-error' | 'machine-missing' | 'alerts-empty' | 'overview-loading'

let fetchMode: FetchMode = 'default'

beforeEach(() => {
  fetchMode = 'default'
  vi.stubGlobal('fetch', vi.fn(handleFetch))
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('dashboard routing and data states', () => {
  it('renders the drift route through application routing', async () => {
    renderWithProviders(<App />, { route: '/drift' })

    expect(await screen.findByRole('heading', { name: 'Drift Monitoring' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toBeInTheDocument()
  })

  it('renders the overview with real API response fields', async () => {
    renderWithProviders(<App />)

    expect(await screen.findByRole('heading', { name: 'Fleet Overview' })).toBeInTheDocument()
    expect(screen.getByText('Machine count')).toBeInTheDocument()
    expect(screen.getByText('Model-positive predictions')).toBeInTheDocument()
    expect(screen.getByText('3.33%')).toBeInTheDocument()
    expect(screen.getByText('AI4I input drift status')).toBeInTheDocument()
  })

  it('shows an overview loading state', () => {
    fetchMode = 'overview-loading'

    renderWithProviders(<App />)

    expect(screen.getByText('Loading fleet overview')).toBeInTheDocument()
  })

  it('shows an overview API error state', async () => {
    fetchMode = 'overview-error'

    renderWithProviders(<App />)

    expect(await screen.findByRole('heading', { name: 'Unable to load fleet overview' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('renders a paginated machine list', async () => {
    renderWithProviders(<App />, { route: '/machines' })

    expect(await screen.findByRole('heading', { name: 'Machines' })).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'MCH-0001' })).toBeInTheDocument()
    expect(screen.getByText('Showing 1-20 of 45')).toBeInTheDocument()
  })

  it('updates machine pagination through URL-backed controls', async () => {
    const user = userEvent.setup()
    renderWithProviders(<App />, { route: '/machines' })

    await screen.findByText('Showing 1-20 of 45')
    await user.click(screen.getByRole('button', { name: 'Next page' }))

    expect(await screen.findByText('Showing 21-40 of 45')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'MCH-0021' })).toBeInTheDocument()
  })

  it('renders machine detail with prediction and anomaly history slices', async () => {
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByRole('heading', { name: 'MCH-0001' })).toBeInTheDocument()
    expect(screen.getByText('Recent Predictions')).toBeInTheDocument()
    expect(screen.getByText('Score is a detector score, not a probability.')).toBeInTheDocument()
    expect(screen.getByText('0.840')).toBeInTheDocument()
  })

  it('renders a clear not-found state for an unknown machine', async () => {
    fetchMode = 'machine-missing'

    renderWithProviders(<App />, { route: '/machines/MCH-404' })

    expect(await screen.findByRole('heading', { name: 'Machine not found' })).toBeInTheDocument()
    expect(screen.getByText('The requested resource was not found.')).toBeInTheDocument()
  })

  it('renders alert list rows', async () => {
    renderWithProviders(<App />, { route: '/alerts' })

    expect(await screen.findByRole('heading', { name: 'Alerts' })).toBeInTheDocument()
    expect(await screen.findByText('Bearing anomaly watch')).toBeInTheDocument()
    const alertRow = screen.getByRole('row', { name: /Bearing anomaly watch/ })
    expect(within(alertRow).getByText('Critical')).toBeInTheDocument()
  })

  it('renders drift scopes separately', async () => {
    renderWithProviders(<App />, { route: '/drift' })

    expect(await screen.findByText('AI4I Model Inputs')).toBeInTheDocument()
    expect(screen.getByText('Operational Anomaly Inputs')).toBeInTheDocument()
    expect(screen.getByText('Distribution shift does not directly measure model accuracy.')).toBeInTheDocument()
  })

  it('renders valid empty states', async () => {
    fetchMode = 'alerts-empty'

    renderWithProviders(<App />, { route: '/alerts' })

    expect(await screen.findByRole('heading', { name: 'No alerts found' })).toBeInTheDocument()
  })

  it('renders the frontend Not Found page for unknown routes', async () => {
    renderWithProviders(<App />, { route: '/not-a-dashboard-route' })

    expect(await screen.findByRole('heading', { name: 'Page not found' })).toBeInTheDocument()
  })
})

async function handleFetch(input: RequestInfo | URL): Promise<Response> {
  const url = getUrl(input)
  const path = url.pathname

  if (path === '/health') {
    return jsonResponse(healthFixture)
  }

  if (path === '/api/v1/fleet/overview') {
    if (fetchMode === 'overview-error') {
      return jsonResponse({ detail: 'database unavailable' }, 503)
    }

    if (fetchMode === 'overview-loading') {
      return new Promise<Response>(() => undefined)
    }

    return jsonResponse(fleetOverviewFixture)
  }

  if (path === '/api/v1/machines') {
    const offset = Number(url.searchParams.get('offset') ?? '0')
    return jsonResponse(machineListFixture(offset))
  }

  if (path === '/api/v1/machines/MCH-0001') {
    return jsonResponse(machineDetailFixture)
  }

  if (path === '/api/v1/machines/MCH-0001/predictions') {
    return jsonResponse(predictionListFixture)
  }

  if (path === '/api/v1/machines/MCH-0001/anomalies') {
    return jsonResponse(anomalyListFixture)
  }

  if (path === '/api/v1/machines/MCH-404') {
    return jsonResponse({ detail: 'machine not found' }, 404)
  }

  if (path === '/api/v1/alerts') {
    return jsonResponse(fetchMode === 'alerts-empty' ? emptyAlertListFixture : alertListFixture)
  }

  if (path === '/api/v1/drift/latest') {
    return jsonResponse(driftFixture)
  }

  return jsonResponse({ detail: `Unhandled test URL: ${url.toString()}` }, 404)
}

function getUrl(input: RequestInfo | URL): URL {
  if (typeof input === 'string') {
    return new URL(input)
  }

  if (input instanceof URL) {
    return input
  }

  return new URL(input.url)
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
    },
  })
}



