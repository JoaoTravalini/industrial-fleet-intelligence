import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  alertListFixture,
  anomalyListFixture,
  copilotChatFixture,
  copilotHealthFixture,
  copilotUnavailableHealthFixture,
  driftFixture,
  emptyAlertListFixture,
  eventTwo,
  explanationFixture,
  fleetOverviewFixture,
  healthFixture,
  machineDetailFixture,
  machineListFixture,
  predictionListFixture,
} from './test/fixtures'
import { renderWithProviders } from './test/render'

type FetchMode =
  | 'default'
  | 'overview-error'
  | 'machine-missing'
  | 'alerts-empty'
  | 'overview-loading'
  | 'explanation-missing'
  | 'explanation-error'
  | 'explanation-loading'
  | 'copilot-unavailable'
  | 'copilot-error'
  | 'copilot-timeout'
  | 'copilot-loading'

let fetchMode: FetchMode = 'default'
let requestedPaths: string[] = []

beforeEach(() => {
  fetchMode = 'default'
  requestedPaths = []
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

  it('renders the copilot route with local and read-only status', async () => {
    renderWithProviders(<App />, { route: '/copilot' })

    expect(await screen.findByRole('heading', { name: 'AI Copilot' })).toBeInTheDocument()
    expect(await screen.findByText('Local AI available')).toBeInTheDocument()
    expect(screen.getByText('Runs locally with Ollama')).toBeInTheDocument()
    expect(screen.getByText('Model loaded')).toBeInTheDocument()
    expect(screen.getByText('Copilot is read-only and cannot change platform state.')).toBeInTheDocument()
  })

  it('shows copilot unavailable guidance', async () => {
    fetchMode = 'copilot-unavailable'

    renderWithProviders(<App />, { route: '/copilot' })

    expect(await screen.findByText('Local AI unavailable')).toBeInTheDocument()
    expect(screen.getByText('Start Ollama and ensure qwen3:4b-instruct is installed.')).toBeInTheDocument()
  })

  it('submits a suggested copilot question and shows sources', async () => {
    const user = userEvent.setup()
    renderWithProviders(<App />, { route: '/copilot' })

    await user.click(await screen.findByRole('button', { name: 'Fleet overview' }))

    expect(await screen.findByText('The fleet has 100 fictional machines and 5 open alerts.')).toBeInTheDocument()
    expect(screen.getByText('Sources')).toBeInTheDocument()
    expect(screen.getAllByText('Fleet overview').length).toBeGreaterThan(1)
    expect(screen.getByText('Model: qwen3:4b-instruct')).toBeInTheDocument()
  })

  it('submits typed copilot messages and can clear the conversation', async () => {
    const user = userEvent.setup()
    renderWithProviders(<App />, { route: '/copilot' })

    await screen.findByText('Local AI available')
    await user.type(screen.getByLabelText('Copilot message'), 'What does anomaly score mean?')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('The fleet has 100 fictional machines and 5 open alerts.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Clear conversation' }))
    expect(screen.queryByText('The fleet has 100 fictional machines and 5 open alerts.')).not.toBeInTheDocument()
  })

  it('shows copilot loading and error states', async () => {
    fetchMode = 'copilot-loading'
    const loadingUser = userEvent.setup()
    const loadingView = renderWithProviders(<App />, { route: '/copilot' })

    await loadingUser.click(await screen.findByRole('button', { name: 'Fleet overview' }))
    expect(await screen.findByText(/Local model is processing/)).toBeInTheDocument()
    expect(screen.getByText(/Local inference can take longer on the first request/)).toBeInTheDocument()
    loadingView.unmount()

    fetchMode = 'copilot-error'
    const errorUser = userEvent.setup()
    const errorView = renderWithProviders(<App />, { route: '/copilot' })
    await errorUser.click(await screen.findByRole('button', { name: 'Fleet overview' }))

    expect(await screen.findByRole('heading', { name: 'Copilot unavailable' })).toBeInTheDocument()
    errorView.unmount()

    fetchMode = 'copilot-timeout'
    const timeoutUser = userEvent.setup()
    renderWithProviders(<App />, { route: '/copilot' })
    await timeoutUser.click(await screen.findByRole('button', { name: 'Fleet overview' }))

    expect(await screen.findByRole('heading', { name: 'Local model response timed out' })).toBeInTheDocument()
    expect(screen.getByText('Verify Ollama is running, then try again after the model is warm.')).toBeInTheDocument()
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

  it('renders failure probability history with the model threshold label', async () => {
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByText('Failure Probability History')).toBeInTheDocument()
    expect(screen.getByText('Model decision threshold 14.00%')).toBeInTheDocument()
  })

  it('renders anomaly monitoring charts without formatting score as a percent', async () => {
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByText('Operational Sensor Monitoring')).toBeInTheDocument()
    expect(screen.getByText('Vibration History')).toBeInTheDocument()
    expect(screen.getByText('Pressure History')).toBeInTheDocument()
    expect(screen.getByText('Anomaly Score History')).toBeInTheDocument()
    expect(screen.getByText('0.840')).toBeInTheDocument()
    expect(screen.queryByText('84.00%')).not.toBeInTheDocument()
  })

  it('renders persisted SHAP explanation details for all six semantic features', async () => {
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByText('SHAP Contribution')).toBeInTheDocument()
    expect(screen.getByText('SHAP values are signed decimal model attributions; they are not probabilities.')).toBeInTheDocument()
    expect(screen.getByText('Positive SHAP: toward higher model failure-risk output. Negative SHAP: toward lower model failure-risk output.')).toBeInTheDocument()
    for (const label of ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0)
    }
    expect(screen.getByText('-0.0180')).toBeInTheDocument()
    expect(screen.queryByText('-1.80%')).not.toBeInTheDocument()
  })

  it('selects another prediction and requests its persisted explanation', async () => {
    const user = userEvent.setup()
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    await screen.findByText('SHAP Contribution')
    await user.click(screen.getByRole('button', { name: '502' }))

    await waitFor(() => {
      expect(requestedPaths.some((path) => path.includes(eventTwo))).toBe(true)
    })
    expect((await screen.findAllByText('20.00%')).length).toBeGreaterThan(0)
  })

  it('shows a stable empty state when an explanation is not materialized', async () => {
    fetchMode = 'explanation-missing'

    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByText('Explanation not materialized for this prediction.')).toBeInTheDocument()
  })

  it('shows explanation loading and error states', async () => {
    fetchMode = 'explanation-loading'
    const loadingView = renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByText('Loading materialized explanation')).toBeInTheDocument()
    loadingView.unmount()

    fetchMode = 'explanation-error'
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByRole('heading', { name: 'Unable to load prediction explanation' })).toBeInTheDocument()
  })

  it('keeps chart layouts mobile-stackable through structural classes', async () => {
    renderWithProviders(<App />, { route: '/machines/MCH-0001' })

    expect(await screen.findByTestId('machine-monitoring-chart-grid')).toHaveClass('chart-grid--three')
    expect(document.querySelector('.chart-frame')).toBeInTheDocument()
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
    expect(screen.getByText('AI4I Model Inputs PSI')).toBeInTheDocument()
    expect(screen.getAllByText('Heuristic monitoring bands: 0.10 watch, 0.25 drift.').length).toBeGreaterThan(0)
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
  requestedPaths.push(path)

  if (path === '/health') {
    return jsonResponse(healthFixture)
  }

  if (path === '/api/v1/copilot/health') {
    return jsonResponse(fetchMode === 'copilot-unavailable' ? copilotUnavailableHealthFixture : copilotHealthFixture)
  }

  if (path === '/api/v1/copilot/chat') {
    if (fetchMode === 'copilot-loading') {
      return new Promise<Response>(() => undefined)
    }

    if (fetchMode === 'copilot-error') {
      return jsonResponse({ detail: 'Local AI Copilot is unavailable. Start Ollama and try again.' }, 503)
    }

    if (fetchMode === 'copilot-timeout') {
      return jsonResponse({ detail: 'Local model response timed out.' }, 504)
    }

    return jsonResponse(copilotChatFixture)
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

  const explanationMatch = path.match(
    /^\/api\/v1\/machines\/MCH-0001\/predictions\/([^/]+)\/explanation$/,
  )
  if (explanationMatch) {
    if (fetchMode === 'explanation-loading') {
      return new Promise<Response>(() => undefined)
    }

    if (fetchMode === 'explanation-error') {
      return jsonResponse({ detail: 'database unavailable' }, 503)
    }

    if (fetchMode === 'explanation-missing') {
      return jsonResponse({ detail: 'Prediction explanation not found' }, 404)
    }

    return jsonResponse(explanationFixture(decodeURIComponent(explanationMatch[1])))
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



