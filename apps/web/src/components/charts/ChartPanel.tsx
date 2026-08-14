import type { ReactNode } from 'react'

interface ChartPanelProps {
  title: string
  description: string
  summary: string
  children: ReactNode
}

export function ChartPanel({ title, description, summary, children }: ChartPanelProps) {
  return (
    <div className="chart-panel">
      <div className="chart-heading">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      <div className="chart-frame">{children}</div>
      <p className="chart-summary">{summary}</p>
    </div>
  )
}
