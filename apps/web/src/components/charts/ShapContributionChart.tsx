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

import type { ExplanationFeatureContribution } from '../../api/types'
import { formatSensorValue, formatShapValue } from '../../utils/format'
import { ChartPanel } from './ChartPanel'

interface ShapContributionChartProps {
  contributions: ExplanationFeatureContribution[]
}

interface ShapPoint {
  featureName: string
  label: string
  featureValue: string
  shapValue: number
}

const FEATURE_LABELS: Record<string, string> = {
  'Air temperature [K]': 'Air temperature',
  'Process temperature [K]': 'Process temperature',
  'Rotational speed [rpm]': 'Rotational speed',
  'Torque [Nm]': 'Torque',
  'Tool wear [min]': 'Tool wear',
  Type: 'Type',
}

export function ShapContributionChart({ contributions }: ShapContributionChartProps) {
  const points = contributions.map<ShapPoint>((contribution) => ({
    featureName: contribution.feature_name,
    label: FEATURE_LABELS[contribution.feature_name] ?? contribution.feature_name,
    featureValue: formatFeatureValue(contribution),
    shapValue: contribution.shap_value,
  }))
  const strongest = [...points].sort(
    (left, right) => Math.abs(right.shapValue) - Math.abs(left.shapValue),
  )[0]

  return (
    <ChartPanel
      title="SHAP Contribution"
      description="SHAP values are signed decimal model attributions; they are not probabilities."
      summary={
        strongest
          ? `Largest absolute attribution is ${strongest.label} at ${formatShapValue(
              strongest.shapValue,
            )}.`
          : 'No SHAP contributions are available.'
      }
    >
      <div className="chart-legend-note">
        Positive SHAP: toward higher model failure-risk output. Negative SHAP: toward lower model
        failure-risk output.
      </div>
      <ResponsiveContainer width="100%" height={Math.max(260, points.length * 44)}>
        <BarChart
          data={points}
          layout="vertical"
          margin={{ top: 12, right: 28, bottom: 18, left: 118 }}
        >
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis type="number" tickFormatter={(value) => formatShapValue(Number(value))} />
          <YAxis dataKey="label" type="category" width={128} />
          <Tooltip
            formatter={(value) => [formatShapValue(Number(value)), 'SHAP attribution']}
            labelFormatter={(_label, payload) =>
              payload?.[0]?.payload
                ? `${payload[0].payload.featureName}: ${payload[0].payload.featureValue}`
                : ''
            }
          />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Bar dataKey="shapValue" name="SHAP attribution" radius={[0, 4, 4, 0]}>
            {points.map((point) => (
              <Cell
                key={point.featureName}
                fill={point.shapValue >= 0 ? 'var(--warning)' : 'var(--info)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="contribution-list" aria-label="SHAP feature contribution details">
        {points.map((point) => (
          <div className="contribution-row" key={point.featureName}>
            <span>
              <strong>{point.label}</strong>
              <small>{point.featureValue}</small>
            </span>
            <strong>{formatShapValue(point.shapValue)}</strong>
          </div>
        ))}
      </div>
    </ChartPanel>
  )
}

function formatFeatureValue(contribution: ExplanationFeatureContribution): string {
  if (contribution.feature_name === 'Type') {
    return String(contribution.feature_value)
  }
  if (contribution.feature_name === 'Air temperature [K]') {
    return formatSensorValue(Number(contribution.feature_value), 'K', 3)
  }
  if (contribution.feature_name === 'Process temperature [K]') {
    return formatSensorValue(Number(contribution.feature_value), 'K', 3)
  }
  if (contribution.feature_name === 'Rotational speed [rpm]') {
    return formatSensorValue(Number(contribution.feature_value), 'rpm', 0)
  }
  if (contribution.feature_name === 'Torque [Nm]') {
    return formatSensorValue(Number(contribution.feature_value), 'Nm', 3)
  }
  if (contribution.feature_name === 'Tool wear [min]') {
    return formatSensorValue(Number(contribution.feature_value), 'min', 0)
  }
  return String(contribution.feature_value)
}
