import { useLatestDrift } from '../api/queries'
import type { DriftFeatureMetric } from '../api/types'
import { EmptyState, ErrorState, LoadingState, PageHeader, Section, StatusBadge } from '../components'
import { formatInteger, formatPsi, formatTimestamp, humanizeToken } from '../utils/format'

const AI4I_SCOPE = 'ai4i_model_input'
const ANOMALY_SCOPE = 'operational_anomaly_inputs'

export function DriftPage() {
  const drift = useLatestDrift()

  if (drift.isLoading) {
    return <LoadingState message="Loading drift monitoring snapshot" />
  }

  if (drift.isError) {
    return (
      <ErrorState
        title="Unable to load drift monitoring"
        error={drift.error}
        onRetry={() => void drift.refetch()}
      />
    )
  }

  if (!drift.data || drift.data.drift_snapshot_id === null) {
    return (
      <EmptyState
        title="No drift snapshot"
        message="No persisted drift monitoring snapshot is available yet."
      />
    )
  }

  const ai4iFeatures = drift.data.features_by_scope[AI4I_SCOPE] ?? []
  const anomalyFeatures = drift.data.features_by_scope[ANOMALY_SCOPE] ?? []

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Input Distribution Monitoring"
        title="Drift Monitoring"
        description="Distribution shift does not directly measure model accuracy."
      />

      <div className="snapshot-strip">
        <div className="metric-row">
          <span>Snapshot</span>
          <strong>{drift.data.drift_snapshot_id}</strong>
        </div>
        <div className="metric-row">
          <span>Created</span>
          <strong>{formatTimestamp(drift.data.created_at)}</strong>
        </div>
        <div className="metric-row">
          <span>Monitor version</span>
          <strong>{drift.data.monitor_version ?? 'No version'}</strong>
        </div>
      </div>

      <DriftScope
        title="AI4I Model Inputs"
        status={drift.data.ai4i_overall_status}
        currentCount={drift.data.ai4i_current_count}
        features={ai4iFeatures}
      />

      <DriftScope
        title="Operational Anomaly Inputs"
        status={drift.data.anomaly_overall_status}
        currentCount={drift.data.anomaly_current_count}
        features={anomalyFeatures}
      />
    </div>
  )
}

interface DriftScopeProps {
  title: string
  status: string | null
  currentCount: number | null
  features: DriftFeatureMetric[]
}

function DriftScope({ title, status, currentCount, features }: DriftScopeProps) {
  return (
    <Section
      title={title}
      description={`${formatInteger(currentCount)} current records evaluated for this monitoring scope.`}
    >
      <div className="section-status-row">
        <span>Overall status</span>
        <StatusBadge kind="drift" value={status} />
      </div>

      {features.length === 0 ? (
        <EmptyState
          title="No feature metrics"
          message="This scope has no feature-level drift metrics in the latest snapshot."
        />
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Feature</th>
                <th scope="col">Type</th>
                <th scope="col">PSI</th>
                <th scope="col">Status</th>
                <th scope="col">Reference</th>
                <th scope="col">Current</th>
                <th scope="col">Diagnostics</th>
              </tr>
            </thead>
            <tbody>
              {features.map((feature) => (
                <tr key={feature.feature_name}>
                  <td data-label="Feature">{feature.feature_name}</td>
                  <td data-label="Type">{humanizeToken(feature.feature_type)}</td>
                  <td data-label="PSI">{formatPsi(feature.psi)}</td>
                  <td data-label="Status">
                    <StatusBadge kind="drift" value={feature.status} />
                  </td>
                  <td data-label="Reference">{formatInteger(feature.reference_count)}</td>
                  <td data-label="Current">{formatInteger(feature.current_count)}</td>
                  <td data-label="Diagnostics">{formatDiagnostics(feature)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}

function formatDiagnostics(feature: DriftFeatureMetric): string {
  const details = [
    feature.standardized_mean_shift === null
      ? null
      : `mean shift ${formatPsi(feature.standardized_mean_shift)}`,
    feature.outside_reference_range_rate === null
      ? null
      : `outside range ${formatPsi(feature.outside_reference_range_rate)}`,
  ].filter((value): value is string => value !== null)

  if (details.length > 0) {
    return details.join('; ')
  }

  const diagnosticKeys = Object.keys(feature.diagnostics)
  if (diagnosticKeys.length > 0) {
    return diagnosticKeys.slice(0, 2).map(humanizeToken).join(', ')
  }

  return 'No additional diagnostics'
}
