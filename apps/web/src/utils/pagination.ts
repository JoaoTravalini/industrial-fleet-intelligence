export function parsePositivePage(value: string | null): number {
  const parsed = Number(value ?? '1')
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 1
}

export function getOffset(page: number, pageSize: number): number {
  return Math.max(0, page - 1) * pageSize
}
