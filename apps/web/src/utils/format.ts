const integerFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const decimalFormatter = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
})
const probabilityFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})
const utcFormatter = new Intl.DateTimeFormat('en-US', {
  timeZone: 'UTC',
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
})

export function formatInteger(value: number | null | undefined): string {
  return value === null || value === undefined ? 'No data' : integerFormatter.format(value)
}

export function formatProbability(value: number | null | undefined): string {
  return value === null || value === undefined ? 'No projection' : probabilityFormatter.format(value)
}

export function formatDecimal(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) {
    return 'No data'
  }

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)
}

export function formatPsi(value: number | null | undefined): string {
  return formatDecimal(value, 4)
}

export function formatAnomalyScore(value: number | null | undefined): string {
  return value === null || value === undefined ? 'No score' : decimalFormatter.format(value)
}

export function formatSignedDecimal(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) {
    return 'No data'
  }

  const formatted = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    signDisplay: 'always',
  }).format(value)

  return Object.is(value, -0) ? formatted.replace('-', '+') : formatted
}

export function formatShapValue(value: number | null | undefined): string {
  return formatSignedDecimal(value, 4)
}

export function formatSensorValue(
  value: number | null | undefined,
  unit: string,
  digits = 3,
): string {
  return value === null || value === undefined ? 'No data' : `${formatDecimal(value, digits)} ${unit}`
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) {
    return 'No timestamp'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return 'Invalid timestamp'
  }

  return `${utcFormatter.format(parsed)} UTC`
}

export function formatShortTimestamp(value: string | null | undefined): string {
  if (!value) {
    return 'No timestamp'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return 'Invalid timestamp'
  }

  return utcFormatter.format(parsed)
}

export function formatDate(value: string | null | undefined): string {
  if (!value) {
    return 'No date'
  }

  const parsed = new Date(`${value}T00:00:00Z`)
  if (Number.isNaN(parsed.getTime())) {
    return 'Invalid date'
  }

  return parsed.toISOString().slice(0, 10)
}

export function formatDecision(value: boolean | null | undefined): string {
  if (value === true) {
    return 'Positive'
  }

  if (value === false) {
    return 'Negative'
  }

  return 'No decision'
}

export function formatFlag(value: boolean | null | undefined): string {
  if (value === true) {
    return 'Flagged'
  }

  if (value === false) {
    return 'Not flagged'
  }

  return 'No flag'
}

export function humanizeToken(value: string | null | undefined): string {
  if (!value) {
    return 'No data'
  }

  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ')
}
