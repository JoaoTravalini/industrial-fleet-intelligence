import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { DriftFeatureMetric } from '../../api/types'
import { formatPsi, humanizeToken } from '../../utils/format'
import { ChartPanel } from './ChartPanel'

interface DriftPsiChartProps {
  title: string
  features: DriftFeatureMetric[]
}

interface DriftPoint {
  featureName: string
  label: string
  psi: number
  status: string
}

const STATUS_COLORS: Record<string, string> = {
  stable: 'var(--success)',
  watch: 'var(--warning)',
  drift: 'var(--danger)',
}

export function DriftPsiChart({ title, features }: DriftPsiChartProps) {
  const points = features
    .map<DriftPoint>((feature) => ({
      featureName: feature.feature_name,
      label: humanizeToken(feature.feature_name),
      psi: feature.psi,
      status: feature.status,
    }))
    .sort((left, right) => right.psi - left.psi || left.label.localeCompare(right.label))
  const highest = points[0]

  return (
    <ChartPanel
      title={`${title} PSI`}
      description="Population Stability Index values use heuristic monitoring bands, not statistical guarantees."
      summary={
        highest
          ? `Highest PSI is ${formatPsi(highest.psi)} for ${highest.label} with ${humanizeToken(
              highest.status,
            )} status.`
          : 'No PSI feature metrics are available for this scope.'
      }
    >
      <div className="chart-legend-note">Heuristic monitoring bands: 0.10 watch, 0.25 drift.</div>
      <ResponsiveContainer width="100%" height={Math.max(240, points.length * 42)}>
        <BarChart
          data={points}
          layout="vertical"
          margin={{ top: 12, right: 28, bottom: 18, left: 116 }}
        >
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis type="number" tickFormatter={(value) => formatPsi(Number(value))} />
          <YAxis dataKey="label" type="category" width={128} />
          <Tooltip
            formatter={(value) => [formatPsi(Number(value)), 'PSI']}
            labelFormatter={(_label, payload) =>
              payload?.[0]?.payload
                ? `${payload[0].payload.featureName} (${humanizeToken(payload[0].payload.status)})`
                : ''
            }
          />
          <ReferenceLine x={0.1} stroke="var(--warning)" strokeDasharray="4 4" />
          <ReferenceLine x={0.25} stroke="var(--danger)" strokeDasharray="4 4" />
          <Bar dataKey="psi" name="PSI" radius={[0, 4, 4, 0]}>
            {points.map((point) => (
              <Cell key={point.featureName} fill={STATUS_COLORS[point.status] ?? 'var(--info)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartPanel>
  )
}
