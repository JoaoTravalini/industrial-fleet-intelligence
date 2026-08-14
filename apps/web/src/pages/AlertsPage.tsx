import { Link, useSearchParams } from 'react-router'

import type { AlertSeverityFilter, AlertStatusFilter } from '../api/queries'
import { useAlerts } from '../api/queries'
import { EmptyState, ErrorState, LoadingState, PageHeader, Pagination, StatusBadge } from '../components'
import { formatTimestamp, humanizeToken } from '../utils/format'
import { getOffset, parsePositivePage } from '../utils/pagination'

const PAGE_SIZE = 20
const alertStatuses: AlertStatusFilter[] = ['open', 'acknowledged', 'resolved']
const alertSeverities: AlertSeverityFilter[] = ['info', 'warning', 'critical']

export function AlertsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const page = parsePositivePage(searchParams.get('page'))
  const offset = getOffset(page, PAGE_SIZE)
  const status = parseAlertStatus(searchParams.get('status'))
  const severity = parseAlertSeverity(searchParams.get('severity'))
  const alertType = searchParams.get('alert_type')?.trim() || undefined
  const machineCode = searchParams.get('machine_code')?.trim() || undefined
  const alerts = useAlerts({
    limit: PAGE_SIZE,
    offset,
    status,
    severity,
    alert_type: alertType,
    machine_code: machineCode,
  })

  function updateFilter(key: string, value: string) {
    const next = new URLSearchParams(searchParams)
    if (value.trim().length > 0) {
      next.set(key, value.trim())
    } else {
      next.delete(key)
    }
    next.delete('page')
    setSearchParams(next)
  }

  function setPage(nextPage: number) {
    const next = new URLSearchParams(searchParams)
    next.set('page', String(Math.max(1, nextPage)))
    setSearchParams(next)
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Operational Monitoring"
        title="Alerts"
        description="Read-only alerts materialized from persisted model and anomaly state."
      />

      <form className="filter-grid" onSubmit={(event) => event.preventDefault()}>
        <label className="field-control">
          <span>Status</span>
          <select value={status ?? ''} onChange={(event) => updateFilter('status', event.target.value)}>
            <option value="">All statuses</option>
            {alertStatuses.map((value) => (
              <option key={value} value={value}>
                {humanizeToken(value)}
              </option>
            ))}
          </select>
        </label>
        <label className="field-control">
          <span>Severity</span>
          <select value={severity ?? ''} onChange={(event) => updateFilter('severity', event.target.value)}>
            <option value="">All severities</option>
            {alertSeverities.map((value) => (
              <option key={value} value={value}>
                {humanizeToken(value)}
              </option>
            ))}
          </select>
        </label>
        <label className="field-control">
          <span>Alert type</span>
          <input
            value={alertType ?? ''}
            onChange={(event) => updateFilter('alert_type', event.target.value)}
            placeholder="telemetry_anomaly"
          />
        </label>
        <label className="field-control">
          <span>Machine code</span>
          <input
            value={machineCode ?? ''}
            onChange={(event) => updateFilter('machine_code', event.target.value)}
            placeholder="MCH-0001"
          />
        </label>
      </form>

      {alerts.isLoading ? <LoadingState message="Loading alerts" /> : null}
      {alerts.isError ? (
        <ErrorState title="Unable to load alerts" error={alerts.error} onRetry={() => void alerts.refetch()} />
      ) : null}
      {alerts.data && alerts.data.total === 0 ? (
        <EmptyState
          title="No alerts found"
          message="No operational alerts match the current server-side filters."
        />
      ) : null}
      {alerts.data && alerts.data.total > 0 ? (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Severity</th>
                  <th scope="col">Type</th>
                  <th scope="col">Machine</th>
                  <th scope="col">Message</th>
                  <th scope="col">Status</th>
                  <th scope="col">Source time</th>
                </tr>
              </thead>
              <tbody>
                {alerts.data.items.map((alert) => (
                  <tr key={alert.alert_id}>
                    <td data-label="Severity">
                      <StatusBadge kind="severity" value={alert.severity} />
                    </td>
                    <td data-label="Type">{humanizeToken(alert.alert_type)}</td>
                    <td data-label="Machine">
                      <Link className="table-link" to={`/machines/${alert.machine_code}`}>
                        {alert.machine_code}
                      </Link>
                    </td>
                    <td data-label="Message">
                      <strong>{alert.title}</strong>
                      {alert.description ? <p className="table-detail">{alert.description}</p> : null}
                    </td>
                    <td data-label="Status">
                      <StatusBadge value={alert.status} />
                    </td>
                    <td data-label="Source time">{formatTimestamp(alert.source_observed_at ?? alert.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Pagination
            count={alerts.data.count}
            limit={alerts.data.limit}
            offset={alerts.data.offset}
            total={alerts.data.total}
            onPageChange={setPage}
          />
        </>
      ) : null}
    </div>
  )
}

function parseAlertStatus(value: string | null): AlertStatusFilter | undefined {
  if (value === 'open' || value === 'acknowledged' || value === 'resolved') {
    return value
  }

  return undefined
}

function parseAlertSeverity(value: string | null): AlertSeverityFilter | undefined {
  if (value === 'info' || value === 'warning' || value === 'critical') {
    return value
  }

  return undefined
}
