import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { FleetOverviewResponse } from '../../api/types'
import { formatInteger } from '../../utils/format'
import { ChartPanel } from './ChartPanel'

interface OverviewStateChartProps {
  overview: FleetOverviewResponse
}

const BAR_COLORS = ['var(--warning)', 'var(--info)', 'var(--danger)', 'var(--success)']

export function OverviewStateChart({ overview }: OverviewStateChartProps) {
  const points = [
    { label: 'Model positive', value: overview.positive_prediction_count },
    { label: 'Model negative', value: overview.negative_prediction_count },
    { label: 'Anomaly flagged', value: overview.flagged_anomaly_count },
    { label: 'Anomaly not flagged', value: overview.non_flagged_anomaly_count },
  ]

  return (
    <ChartPanel
      title="Monitoring State Split"
      description="Counts compare model decisions and anomaly flags from materialized API state."
      summary={`${formatInteger(overview.prediction_history_count)} prediction rows and ${formatInteger(
        overview.anomaly_audit_count,
      )} anomaly audit rows are represented.`}
    >
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={points} margin={{ top: 12, right: 18, bottom: 20, left: 12 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="label" tickMargin={10} interval={0} />
          <YAxis tickFormatter={(value) => formatInteger(Number(value))} width={72} />
          <Tooltip formatter={(value) => [formatInteger(Number(value)), 'Count']} />
          <Bar dataKey="value" name="Count" radius={[4, 4, 0, 0]}>
            {points.map((point, index) => (
              <Cell key={point.label} fill={BAR_COLORS[index] ?? 'var(--info)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartPanel>
  )
}
