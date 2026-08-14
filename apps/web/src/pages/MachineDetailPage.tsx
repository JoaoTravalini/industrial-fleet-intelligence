import { Link, useParams } from 'react-router'

import { ApiError } from '../api/client'
import { useMachineAnomalies, useMachineDetail, useMachinePredictions } from '../api/queries'
import { EmptyState, ErrorState, KpiCard, LoadingState, PageHeader, Section, StatusBadge } from '../components'
import {
  formatAnomalyScore,
  formatDate,
  formatInteger,
  formatProbability,
  formatPsi,
  formatTimestamp,
} from '../utils/format'

const HISTORY_LIMIT = 5

export function MachineDetailPage() {
  const { machineCode } = useParams()
  const machine = useMachineDetail(machineCode)
  const predictions = useMachinePredictions(machineCode, HISTORY_LIMIT)
  const anomalies = useMachineAnomalies(machineCode, HISTORY_LIMIT)

  if (machine.isLoading) {
    return <LoadingState message="Loading machine detail" />
  }

  if (machine.isError) {
    const title = machine.error instanceof ApiError && machine.error.status === 404 ? 'Machine not found' : 'Unable to load machine'
    return <ErrorState title={title} error={machine.error} onRetry={() => void machine.refetch()} />
  }

  if (!machine.data) {
    return <EmptyState title="Machine not available" message="No machine detail was returned by the API." />
  }

  const latest = machine.data.latest_prediction

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Machine Detail"
        title={machine.data.machine_code}
        description="Read-only machine identity, latest AI4I projection, and recent monitoring history."
        actions={
          <Link className="button button--secondary" to="/machines">
            Back to Machines
          </Link>
        }
      />

      <section className="kpi-grid" aria-label="Machine summary cards">
        <KpiCard label="Machine type" value={machine.data.machine_type} detail={machine.data.model_family} />
        <KpiCard
          label="Operational status"
          value={<StatusBadge value={machine.data.operational_status} />}
          detail={`Commissioned ${formatDate(machine.data.commissioned_on)}`}
        />
        <KpiCard
          label="Prediction-history count"
          value={formatInteger(machine.data.prediction_history_count)}
          detail="Persisted AI4I model outputs"
          tone="info"
        />
        <KpiCard
          label="Anomaly-history count"
          value={formatInteger(machine.data.anomaly_audit_count)}
          detail="Independent detector audit rows"
          tone="info"
        />
      </section>

      <Section
        title="Latest AI4I Model Projection"
        description="Failure probability and model decision are classifier outputs, not observed failure state."
      >
        {latest ? (
          <div className="detail-grid">
            <div className="metric-row">
              <span>Failure probability</span>
              <strong>{formatProbability(latest.failure_probability)}</strong>
            </div>
            <div className="metric-row">
              <span>Model decision</span>
              <StatusBadge kind="decision" value={latest.failure_prediction} />
            </div>
            <div className="metric-row">
              <span>Frozen threshold</span>
              <strong>{formatPsi(latest.frozen_threshold)}</strong>
            </div>
            <div className="metric-row">
              <span>Prediction event time</span>
              <strong>{formatTimestamp(latest.event_time)}</strong>
            </div>
            <div className="metric-row">
              <span>Model</span>
              <strong>{latest.model_name ?? 'No model'} {latest.model_version ?? ''}</strong>
            </div>
            <div className="metric-row">
              <span>Config hash</span>
              <strong className="mono-text">{latest.final_config_hash ?? 'No hash'}</strong>
            </div>
          </div>
        ) : (
          <EmptyState
            title="No latest projection"
            message="This machine does not currently have a latest AI4I projection in materialized state."
          />
        )}
      </Section>

      <Section title="Recent Predictions" description={`Latest ${HISTORY_LIMIT} persisted model outputs.`}>
        {predictions.isLoading ? <LoadingState message="Loading recent predictions" /> : null}
        {predictions.isError ? (
          <ErrorState
            title="Unable to load recent predictions"
            error={predictions.error}
            onRetry={() => void predictions.refetch()}
          />
        ) : null}
        {predictions.data && predictions.data.total === 0 ? (
          <EmptyState title="No predictions found" message="No model prediction rows exist for this machine." />
        ) : null}
        {predictions.data && predictions.data.total > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">ID</th>
                  <th scope="col">Event time</th>
                  <th scope="col">Failure probability</th>
                  <th scope="col">Model decision</th>
                  <th scope="col">Threshold</th>
                  <th scope="col">Model</th>
                </tr>
              </thead>
              <tbody>
                {predictions.data.items.map((prediction) => (
                  <tr key={prediction.model_prediction_id}>
                    <td data-label="ID">{prediction.model_prediction_id}</td>
                    <td data-label="Event time">{formatTimestamp(prediction.event_time)}</td>
                    <td data-label="Failure probability">{formatProbability(prediction.failure_probability)}</td>
                    <td data-label="Model decision">
                      <StatusBadge kind="decision" value={prediction.failure_prediction} />
                    </td>
                    <td data-label="Threshold">{formatPsi(prediction.frozen_threshold)}</td>
                    <td data-label="Model">
                      {prediction.model_name} {prediction.model_version}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Section>

      <Section
        title="Recent Anomalies"
        description="Score is a detector score, not a probability."
      >
        {anomalies.isLoading ? <LoadingState message="Loading recent anomalies" /> : null}
        {anomalies.isError ? (
          <ErrorState
            title="Unable to load recent anomalies"
            error={anomalies.error}
            onRetry={() => void anomalies.refetch()}
          />
        ) : null}
        {anomalies.data && anomalies.data.total === 0 ? (
          <EmptyState title="No anomalies found" message="No anomaly audit rows exist for this machine." />
        ) : null}
        {anomalies.data && anomalies.data.total > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">ID</th>
                  <th scope="col">Event time</th>
                  <th scope="col">Anomaly score</th>
                  <th scope="col">Anomaly flag</th>
                  <th scope="col">Vibration</th>
                  <th scope="col">Pressure</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.data.items.map((anomaly) => (
                  <tr key={anomaly.anomaly_id}>
                    <td data-label="ID">{anomaly.anomaly_id}</td>
                    <td data-label="Event time">{formatTimestamp(anomaly.event_time)}</td>
                    <td data-label="Anomaly score">{formatAnomalyScore(anomaly.anomaly_score)}</td>
                    <td data-label="Anomaly flag">
                      <StatusBadge kind="flag" value={anomaly.anomaly_flag} />
                    </td>
                    <td data-label="Vibration">{formatPsi(anomaly.vibration_mm_s)} mm/s</td>
                    <td data-label="Pressure">{formatPsi(anomaly.pressure_bar)} bar</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Section>
    </div>
  )
}
