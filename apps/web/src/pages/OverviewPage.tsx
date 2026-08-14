import { Link } from 'react-router'

import { useFleetOverview } from '../api/queries'
import { EmptyState, ErrorState, KpiCard, LoadingState, PageHeader, Section, StatusBadge } from '../components'
import { formatInteger, formatProbability } from '../utils/format'

export function OverviewPage() {
  const overview = useFleetOverview()

  if (overview.isLoading) {
    return <LoadingState message="Loading fleet overview" />
  }

  if (overview.isError) {
    return (
      <ErrorState
        title="Unable to load fleet overview"
        error={overview.error}
        onRetry={() => void overview.refetch()}
      />
    )
  }

  if (!overview.data) {
    return (
      <EmptyState
        title="No fleet overview available"
        message="The API returned no overview payload for the current materialized state."
      />
    )
  }

  const data = overview.data

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Operational Dashboard"
        title="Fleet Overview"
        description="Read-only view of materialized PostgreSQL state exposed through FastAPI."
        actions={
          <Link className="button button--primary" to="/machines">
            View Machines
          </Link>
        }
      />

      <section className="kpi-grid" aria-label="Fleet key performance indicators">
        <KpiCard label="Machine count" value={formatInteger(data.machine_count)} detail="Registered assets" />
        <KpiCard
          label="Machines with prediction projection"
          value={formatInteger(data.machines_with_prediction_projection)}
          detail="Latest AI4I model output present"
          tone="info"
        />
        <KpiCard
          label="Prediction history count"
          value={formatInteger(data.prediction_history_count)}
          detail="Persisted model outputs"
        />
        <KpiCard
          label="Model-positive predictions"
          value={formatInteger(data.positive_prediction_count)}
          detail={`${formatInteger(data.negative_prediction_count)} negative model decisions`}
          tone="warning"
        />
        <KpiCard
          label="Flagged anomaly count"
          value={formatInteger(data.flagged_anomaly_count)}
          detail={`${formatInteger(data.anomaly_audit_count)} anomaly audit rows`}
          tone="warning"
        />
        <KpiCard
          label="Open alert count"
          value={formatInteger(data.open_alert_count)}
          detail="Operational monitoring alerts"
          tone={data.open_alert_count > 0 ? 'warning' : 'success'}
        />
      </section>

      <div className="two-column-grid">
        <Section
          title="Failure-Risk Summary"
          description="AI4I classifier probabilities are model outputs, not observed machine failures."
        >
          <div className="metric-list">
            <div className="metric-row">
              <span>Mean model failure probability</span>
              <strong>{formatProbability(data.mean_failure_probability)}</strong>
            </div>
            <div className="metric-row">
              <span>Maximum model failure probability</span>
              <strong>{formatProbability(data.max_failure_probability)}</strong>
            </div>
          </div>
        </Section>

        <Section
          title="Monitoring Status"
          description="Drift status tracks input-distribution shift, separate from alert severity and model decisions."
        >
          <div className="metric-list">
            <div className="metric-row">
              <span>AI4I input drift status</span>
              <StatusBadge kind="drift" value={data.latest_ai4i_drift_status} />
            </div>
            <div className="metric-row">
              <span>Anomaly-input drift status</span>
              <StatusBadge kind="drift" value={data.latest_anomaly_drift_status} />
            </div>
          </div>
        </Section>
      </div>
    </div>
  )
}
