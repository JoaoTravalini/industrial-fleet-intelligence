import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { PredictionResponse } from '../../api/types'
import { formatProbability, formatShortTimestamp, formatTimestamp } from '../../utils/format'
import { ChartPanel } from './ChartPanel'

interface ProbabilityPoint {
  eventId: string
  label: string
  fullTime: string
  timeMs: number
  failureProbability: number
  threshold: number | null
}

interface PredictionProbabilityChartProps {
  predictions: PredictionResponse[]
}

export function PredictionProbabilityChart({ predictions }: PredictionProbabilityChartProps) {
  const points = predictions
    .filter((prediction) => prediction.failure_probability !== null && prediction.event_time)
    .map<ProbabilityPoint>((prediction) => ({
      eventId: prediction.event_id ?? String(prediction.model_prediction_id),
      label: formatShortTimestamp(prediction.event_time),
      fullTime: formatTimestamp(prediction.event_time),
      timeMs: new Date(prediction.event_time ?? '').getTime(),
      failureProbability: prediction.failure_probability ?? 0,
      threshold: prediction.frozen_threshold,
    }))
    .sort((left, right) => left.timeMs - right.timeMs)
  const latest = points.at(-1)
  const threshold = points.find((point) => point.threshold !== null)?.threshold ?? null

  return (
    <ChartPanel
      title="Failure Probability History"
      description="Failure probability is the frozen AI4I classifier output displayed as a percentage."
      summary={
        latest
          ? `Latest ${formatProbability(latest.failureProbability)}; model decision threshold ${
              threshold === null ? 'not available' : formatProbability(threshold)
            }.`
          : 'No chartable probability points are available.'
      }
    >
      <div className="chart-legend-note">
        Model decision threshold {threshold === null ? 'not available' : formatProbability(threshold)}
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={points} margin={{ top: 12, right: 24, bottom: 16, left: 12 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="label" tickMargin={10} minTickGap={18} />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(value) => formatProbability(Number(value))}
            width={72}
          />
          <Tooltip
            formatter={(value) => [formatProbability(Number(value)), 'Failure probability']}
            labelFormatter={(_label, payload) => payload?.[0]?.payload?.fullTime ?? ''}
          />
          <Legend />
          {threshold !== null ? (
            <ReferenceLine
              y={threshold}
              stroke="var(--warning)"
              strokeDasharray="4 4"
              ifOverflow="extendDomain"
            />
          ) : null}
          <Line
            type="monotone"
            dataKey="failureProbability"
            name="Failure probability"
            stroke="var(--info)"
            strokeWidth={2.4}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartPanel>
  )
}
