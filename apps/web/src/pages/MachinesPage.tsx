import { Link, useSearchParams } from 'react-router'

import type { MachineStatusFilter } from '../api/queries'
import { useMachines } from '../api/queries'
import { EmptyState, ErrorState, LoadingState, PageHeader, Pagination, StatusBadge } from '../components'
import { formatDate, formatDecision, formatProbability, formatTimestamp } from '../utils/format'
import { getOffset, parsePositivePage } from '../utils/pagination'

const PAGE_SIZE = 20
const machineStatuses: MachineStatusFilter[] = ['active', 'maintenance', 'inactive']

export function MachinesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const page = parsePositivePage(searchParams.get('page'))
  const selectedStatus = parseMachineStatus(searchParams.get('status'))
  const offset = getOffset(page, PAGE_SIZE)
  const machines = useMachines({ limit: PAGE_SIZE, offset, status: selectedStatus })

  function setPage(nextPage: number) {
    const next = new URLSearchParams(searchParams)
    next.set('page', String(Math.max(1, nextPage)))
    setSearchParams(next)
  }

  function setStatus(nextStatus: string) {
    const next = new URLSearchParams(searchParams)
    if (nextStatus.length > 0) {
      next.set('status', nextStatus)
    } else {
      next.delete('status')
    }
    next.delete('page')
    setSearchParams(next)
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Fleet Inventory"
        title="Machines"
        description="Paginated asset list sourced from the FastAPI machine endpoint."
      />

      <div className="toolbar">
        <label className="field-control">
          <span>Operational status</span>
          <select value={selectedStatus ?? ''} onChange={(event) => setStatus(event.target.value)}>
            <option value="">All statuses</option>
            {machineStatuses.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </label>
      </div>

      {machines.isLoading ? <LoadingState message="Loading machines" /> : null}

      {machines.isError ? (
        <ErrorState title="Unable to load machines" error={machines.error} onRetry={() => void machines.refetch()} />
      ) : null}

      {machines.data && machines.data.total === 0 ? (
        <EmptyState
          title="No machines found"
          message="The current server-side filters returned no matching machines."
        />
      ) : null}

      {machines.data && machines.data.total > 0 ? (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Machine</th>
                  <th scope="col">Type</th>
                  <th scope="col">Model family</th>
                  <th scope="col">Commissioned</th>
                  <th scope="col">Status</th>
                  <th scope="col">Failure probability</th>
                  <th scope="col">Model decision</th>
                  <th scope="col">Prediction time</th>
                </tr>
              </thead>
              <tbody>
                {machines.data.items.map((machine) => (
                  <tr key={machine.machine_code}>
                    <td data-label="Machine">
                      <Link className="table-link" to={`/machines/${machine.machine_code}`}>
                        {machine.machine_code}
                      </Link>
                    </td>
                    <td data-label="Type">{machine.machine_type}</td>
                    <td data-label="Model family">{machine.model_family}</td>
                    <td data-label="Commissioned">{formatDate(machine.commissioned_on)}</td>
                    <td data-label="Status">
                      <StatusBadge value={machine.operational_status} />
                    </td>
                    <td data-label="Failure probability">
                      {formatProbability(machine.latest_prediction?.failure_probability)}
                    </td>
                    <td data-label="Model decision">
                      <StatusBadge kind="decision" value={machine.latest_prediction?.failure_prediction} />
                      <span className="sr-only">
                        {formatDecision(machine.latest_prediction?.failure_prediction)} model decision
                      </span>
                    </td>
                    <td data-label="Prediction time">
                      {formatTimestamp(machine.latest_prediction?.event_time)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Pagination
            count={machines.data.count}
            limit={machines.data.limit}
            offset={machines.data.offset}
            total={machines.data.total}
            onPageChange={setPage}
          />
        </>
      ) : null}
    </div>
  )
}

function parseMachineStatus(value: string | null): MachineStatusFilter | undefined {
  if (value === 'active' || value === 'maintenance' || value === 'inactive') {
    return value
  }

  return undefined
}
