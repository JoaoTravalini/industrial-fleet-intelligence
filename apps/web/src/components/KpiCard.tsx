import type { ReactNode } from 'react'

interface KpiCardProps {
  label: string
  value: ReactNode
  detail?: ReactNode
  tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'info'
}

export function KpiCard({ label, value, detail, tone = 'neutral' }: KpiCardProps) {
  return (
    <article className={`kpi-card kpi-card--${tone}`}>
      <p className="kpi-label">{label}</p>
      <p className="kpi-value">{value}</p>
      {detail ? <p className="kpi-detail">{detail}</p> : null}
    </article>
  )
}
