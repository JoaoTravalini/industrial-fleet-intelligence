import { formatDecision, formatFlag, humanizeToken } from '../utils/format'

export type BadgeKind = 'status' | 'drift' | 'severity' | 'decision' | 'flag'

type BadgeValue = string | boolean | null | undefined

type BadgeTone = 'neutral' | 'success' | 'warning' | 'danger' | 'info'

interface StatusBadgeProps {
  kind?: BadgeKind
  value: BadgeValue
}

export function StatusBadge({ kind = 'status', value }: StatusBadgeProps) {
  const tone = getTone(kind, value)
  const label = getLabel(kind, value)

  return <span className={`status-badge status-badge--${tone}`}>{label}</span>
}

function getLabel(kind: BadgeKind, value: BadgeValue): string {
  if (kind === 'decision') {
    return formatDecision(typeof value === 'boolean' ? value : null)
  }

  if (kind === 'flag') {
    return formatFlag(typeof value === 'boolean' ? value : null)
  }

  return humanizeToken(typeof value === 'string' ? value : null)
}

function getTone(kind: BadgeKind, value: BadgeValue): BadgeTone {
  if (kind === 'decision') {
    if (value === true) {
      return 'warning'
    }

    if (value === false) {
      return 'success'
    }

    return 'neutral'
  }

  if (kind === 'flag') {
    if (value === true) {
      return 'warning'
    }

    if (value === false) {
      return 'success'
    }

    return 'neutral'
  }

  const normalized = typeof value === 'string' ? value.toLowerCase() : ''

  if (kind === 'drift') {
    if (normalized === 'stable') {
      return 'success'
    }

    if (normalized === 'watch') {
      return 'warning'
    }

    if (normalized === 'drift') {
      return 'danger'
    }
  }

  if (kind === 'severity') {
    if (normalized === 'critical') {
      return 'danger'
    }

    if (normalized === 'warning') {
      return 'warning'
    }

    if (normalized === 'info') {
      return 'info'
    }
  }

  if (normalized === 'active' || normalized === 'resolved' || normalized === 'stable') {
    return 'success'
  }

  if (normalized === 'maintenance' || normalized === 'open' || normalized === 'watch') {
    return 'warning'
  }

  if (normalized === 'critical' || normalized === 'drift') {
    return 'danger'
  }

  if (normalized === 'acknowledged' || normalized === 'info') {
    return 'info'
  }

  return 'neutral'
}
