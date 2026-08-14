const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'

type QueryValue = string | number | boolean | null | undefined
export type QueryParams = Record<string, QueryValue>

interface ApiErrorPayload {
  detail?: unknown
}

export class ApiError extends Error {
  readonly status: number
  readonly payload: unknown

  constructor(status: number, message: string, payload: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim()
  return configured && configured.length > 0 ? configured : DEFAULT_API_BASE_URL
}

export function buildApiUrl(path: string, params: QueryParams = {}): string {
  const url = new URL(path, getApiBaseUrl())

  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') {
      continue
    }

    url.searchParams.set(key, String(value))
  }

  return url.toString()
}

export async function fetchJson<T>(
  path: string,
  params: QueryParams = {},
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set('Accept', 'application/json')

  const response = await fetch(buildApiUrl(path, params), {
    ...init,
    headers,
  })

  if (!response.ok) {
    const payload = await readErrorPayload(response)
    throw new ApiError(response.status, getApiErrorMessage(response.status, payload), payload)
  }

  return (await response.json()) as T
}

async function readErrorPayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? ''

  if (contentType.includes('application/json')) {
    try {
      return await response.json()
    } catch {
      return null
    }
  }

  try {
    const text = await response.text()
    return text.length > 0 ? text : null
  } catch {
    return null
  }
}

function getApiErrorMessage(status: number, payload: unknown): string {
  const detail = getPayloadDetail(payload)

  if (detail) {
    return detail
  }

  if (status === 404) {
    return 'The requested API resource was not found.'
  }

  if (status === 503) {
    return 'The API or database is currently unavailable.'
  }

  return `The API returned HTTP ${status}.`
}

function getPayloadDetail(payload: unknown): string | null {
  if (typeof payload === 'string') {
    return payload
  }

  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as ApiErrorPayload).detail
    if (typeof detail === 'string') {
      return detail
    }
  }

  return null
}

