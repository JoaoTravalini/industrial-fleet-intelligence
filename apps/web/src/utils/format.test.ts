import { describe, expect, it } from 'vitest'

import {
  formatAnomalyScore,
  formatProbability,
  formatSensorValue,
  formatShapValue,
  formatShortTimestamp,
  formatSignedDecimal,
  formatTimestamp,
} from './format'

describe('format helpers', () => {
  it('formats failure probability as a human-readable percentage', () => {
    expect(formatProbability(0.033333)).toBe('3.33%')
  })

  it('formats anomaly score as a decimal score instead of a percentage', () => {
    expect(formatAnomalyScore(0.84)).toBe('0.840')
  })

  it('formats timestamps with explicit UTC semantics', () => {
    expect(formatTimestamp('2026-08-14T12:05:00Z')).toContain('UTC')
  })

  it('formats chart domains without changing their units', () => {
    expect(formatSignedDecimal(0.12345)).toBe('+0.1235')
    expect(formatShapValue(-0.12)).toBe('-0.1200')
    expect(formatSensorValue(4.2, 'mm/s')).toBe('4.200 mm/s')
    expect(formatShortTimestamp('2026-08-14T12:05:00Z')).toContain('Aug')
  })
})
