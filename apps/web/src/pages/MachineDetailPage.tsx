import { Link, useParams } from 'react-router'
import { useEffect, useMemo, useState } from 'react'

import { ApiError } from '../api/client'
import {
  useMachineAnomalies,
  useMachineDetail,
  useMachinePredictions,
  usePredictionExplanation,
} from '../api/queries'
import {
  AnomalyMonitoringCharts,
  EmptyState,
  ErrorState,
  KpiCard,
  LoadingState,
  PageHeader,
  PredictionProbabilityChart,
  Section,
  ShapContributionChart,
  StatusBadge,
} from '../components'
import {
  formatAnomalyScore,
  formatDate,
  formatDecision,
  formatInteger,
  formatProbability,
  formatPsi,
  formatShapValue,
  formatTimestamp,
} from '../utils/format'

const CHART_HISTORY_LIMIT = 100
const TABLE_HISTORY_LIMIT = 5

export function MachineDetailPage() {
  const { machineCode } = useParams()
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null)
  const machine = useMachineDetail(machineCode)
  const predictions = useMachinePredictions(machineCode, CHART_HISTORY_LIMIT)
  const anomalies = useMachineAnomalies(machineCode, CHART_HISTORY_LIMIT)
  const latestPredictionEventId = predictions.data?.items.find((prediction) => prediction.event_id)?.event_id
  const explanation = usePredictionExplanation(machineCode, selectedEventId)
  const recentPredictionRows = useMemo(
    () => predictions.data?.items.slice(0, TABLE_HISTORY_LIMIT) ?? [],
    [predictions.data?.items],
  )
  const recentAnomalyRows = useMemo(
    () => anomalies.data?.items.slice(0, TABLE_HISTORY_LIMIT) ?? [],
    [anomalies.data?.items],
  )

  useEffect(() => {
    setSelectedEventId(null)
  }, [machineCode])

  useEffect(() => {
    if (!selectedEventId && latestPredictionEventId) {
      setSelectedEventId(latestPredictionEventId)
    }
  }, [latestPredictionEventId, selectedEventId])

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

      <Section
        title="Prediction History Analytics"
        description={`Latest ${CHART_HISTORY_LIMIT} persisted model outputs requested for charting.`}
      >
        {predictions.isLoading ? <LoadingState message="Loading prediction history chart" /> : null}
        {predictions.isError ? (
          <ErrorState
            title="Unable to load prediction history chart"
            error={predictions.error}
            onRetry={() => void predictions.refetch()}
          />
        ) : null}
        {predictions.data && predictions.data.total === 0 ? (
          <EmptyState title="No prediction history" message="No model prediction rows exist for this machine." />
        ) : null}
        {predictions.data && predictions.data.total > 0 ? (
          <PredictionProbabilityChart predictions={predictions.data.items} />
        ) : null}
      </Section>

      <Section
        title="Model Explanation"
        description="Persisted SHAP attribution for the selected prediction; values are not probabilities."
      >
        {selectedEventId === null ? (
          <EmptyState title="No prediction selected" message="Select a prediction row to inspect its explanation." />
        ) : null}
        {selectedEventId !== null && explanation.isLoading ? (
          <LoadingState message="Loading materialized explanation" />
        ) : null}
        {selectedEventId !== null && explanation.isError ? (
          explanation.error instanceof ApiError && explanation.error.status === 404 ? (
            <EmptyState
              title="Explanation not materialized"
              message="Explanation not materialized for this prediction."
            />
          ) : (
            <ErrorState
              title="Unable to load prediction explanation"
              error={explanation.error}
              onRetry={() => void explanation.refetch()}
            />
          )
        ) : null}
        {explanation.data ? (
          <div className="section-stack">
            <div className="detail-grid">
              <div className="metric-row">
                <span>Selected event</span>
                <strong className="mono-text">{explanation.data.event_id}</strong>
              </div>
              <div className="metric-row">
                <span>Failure probability</span>
                <strong>{formatProbability(explanation.data.failure_probability)}</strong>
              </div>
              <div className="metric-row">
                <span>Model decision</span>
                <strong>{formatDecision(explanation.data.failure_prediction)}</strong>
              </div>
              <div className="metric-row">
                <span>Base value</span>
                <strong>{formatShapValue(explanation.data.base_value)}</strong>
              </div>
              <div className="metric-row">
                <span>Model output value</span>
                <strong>{formatProbability(explanation.data.model_output_value)}</strong>
              </div>
              <div className="metric-row">
                <span>Additivity error</span>
                <strong>{formatShapValue(explanation.data.additivity_error)}</strong>
              </div>
            </div>
            <ShapContributionChart contributions={explanation.data.feature_contributions} />
          </div>
        ) : null}
      </Section>

      <Section
        title="Recent Predictions"
        description={`Latest ${TABLE_HISTORY_LIMIT} persisted model outputs. Select a row to update the explanation panel.`}
      >
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
                {recentPredictionRows.map((prediction) => (
                  <tr
                    className={prediction.event_id === selectedEventId ? 'is-selected-row' : undefined}
                    key={prediction.model_prediction_id}
                  >
                    <td data-label="ID">
                      <button
                        className="table-action"
                        type="button"
                        aria-pressed={prediction.event_id === selectedEventId}
                        disabled={!prediction.event_id}
                        onClick={() => setSelectedEventId(prediction.event_id)}
                      >
                        {prediction.model_prediction_id}
                      </button>
                    </td>
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
        title="Operational Sensor Monitoring"
        description={`Latest ${CHART_HISTORY_LIMIT} anomaly audit rows requested for sensor and detector-score charting.`}
      >
        {anomalies.isLoading ? <LoadingState message="Loading anomaly monitoring charts" /> : null}
        {anomalies.isError ? (
          <ErrorState
            title="Unable to load anomaly monitoring charts"
            error={anomalies.error}
            onRetry={() => void anomalies.refetch()}
          />
        ) : null}
        {anomalies.data && anomalies.data.total === 0 ? (
          <EmptyState title="No anomaly audit rows" message="No anomaly audit rows exist for this machine." />
        ) : null}
        {anomalies.data && anomalies.data.total > 0 ? (
          <AnomalyMonitoringCharts anomalies={anomalies.data.items} />
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
                {recentAnomalyRows.map((anomaly) => (
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
