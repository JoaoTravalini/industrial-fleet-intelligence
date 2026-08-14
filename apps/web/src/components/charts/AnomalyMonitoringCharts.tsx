import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { AnomalyResponse } from '../../api/types'
import {
  formatAnomalyScore,
  formatDecimal,
  formatSensorValue,
  formatShortTimestamp,
  formatTimestamp,
} from '../../utils/format'
import { ChartPanel } from './ChartPanel'

interface AnomalyPoint {
  eventId: string
  label: string
  fullTime: string
  timeMs: number
  vibration: number | null
  pressure: number | null
  anomalyScore: number
  flagged: boolean
}

interface AnomalyMonitoringChartsProps {
  anomalies: AnomalyResponse[]
}

export function AnomalyMonitoringCharts({ anomalies }: AnomalyMonitoringChartsProps) {
  const points = anomalies
    .filter((anomaly) => anomaly.event_time)
    .map<AnomalyPoint>((anomaly) => ({
      eventId: anomaly.event_id ?? String(anomaly.anomaly_id),
      label: formatShortTimestamp(anomaly.event_time),
      fullTime: formatTimestamp(anomaly.event_time),
      timeMs: new Date(anomaly.event_time ?? '').getTime(),
      vibration: anomaly.vibration_mm_s,
      pressure: anomaly.pressure_bar,
      anomalyScore: anomaly.anomaly_score,
      flagged: anomaly.anomaly_flag === true,
    }))
    .sort((left, right) => left.timeMs - right.timeMs)
  const flaggedCount = points.filter((point) => point.flagged).length

  return (
    <div className="chart-grid chart-grid--three" data-testid="machine-monitoring-chart-grid">
      <MetricChart
        title="Vibration History"
        description="Operational vibration values in millimeters per second."
        data={points}
        dataKey="vibration"
        unit="mm/s"
        color="var(--info)"
        flaggedCount={flaggedCount}
        formatter={(value) => formatSensorValue(value, 'mm/s')}
      />
      <MetricChart
        title="Pressure History"
        description="Operational pressure values in bar."
        data={points}
        dataKey="pressure"
        unit="bar"
        color="var(--success)"
        flaggedCount={flaggedCount}
        formatter={(value) => formatSensorValue(value, 'bar')}
      />
      <MetricChart
        title="Anomaly Score History"
        description="Detector score shown as a decimal score, not a probability."
        data={points}
        dataKey="anomalyScore"
        unit="score"
        color="var(--warning)"
        flaggedCount={flaggedCount}
        formatter={formatAnomalyScore}
      />
    </div>
  )
}

interface MetricChartProps {
  title: string
  description: string
  data: AnomalyPoint[]
  dataKey: 'vibration' | 'pressure' | 'anomalyScore'
  unit: string
  color: string
  flaggedCount: number
  formatter: (value: number | null | undefined) => string
}

function MetricChart({
  title,
  description,
  data,
  dataKey,
  unit,
  color,
  flaggedCount,
  formatter,
}: MetricChartProps) {
  const latest = data.at(-1)
  const flaggedPoints = data.filter((point) => point.flagged && point[dataKey] !== null)

  return (
    <ChartPanel
      title={title}
      description={description}
      summary={
        latest
          ? `Latest ${formatter(latest[dataKey])}; ${flaggedCount} flagged point(s) in this window.`
          : 'No chartable anomaly points are available.'
      }
    >
      <ResponsiveContainer width="100%" height={230}>
        <ComposedChart data={data} margin={{ top: 12, right: 18, bottom: 16, left: 12 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="label" tickMargin={10} minTickGap={18} />
          <YAxis
            tickFormatter={(value) =>
              unit === 'score' ? formatAnomalyScore(Number(value)) : formatDecimal(Number(value))
            }
            width={68}
          />
          <Tooltip
            formatter={(value) => [formatter(Number(value)), title]}
            labelFormatter={(_label, payload) => payload?.[0]?.payload?.fullTime ?? ''}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey={dataKey}
            name={title}
            stroke={color}
            strokeWidth={2.2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            connectNulls
          />
          <Scatter
            data={flaggedPoints}
            dataKey={dataKey}
            fill="var(--danger)"
            name="Flagged anomaly point"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartPanel>
  )
}
