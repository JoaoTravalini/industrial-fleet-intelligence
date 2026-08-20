import { useQuery } from '@tanstack/react-query'

import { fetchJson } from './client'
import type {
  AlertListResponse,
  AnomalyListResponse,
  CopilotChatRequest,
  CopilotChatResponse,
  CopilotHealthResponse,
  DriftLatestResponse,
  FleetOverviewResponse,
  HealthResponse,
  MachineDetailResponse,
  MachineListResponse,
  PredictionExplanationResponse,
  PredictionListResponse,
} from './types'

const DEFAULT_STALE_TIME_MS = 60_000
const HEALTH_REFETCH_INTERVAL_MS = 60_000

export type MachineStatusFilter = 'active' | 'maintenance' | 'inactive'
export type AlertStatusFilter = 'open' | 'acknowledged' | 'resolved'
export type AlertSeverityFilter = 'info' | 'warning' | 'critical'

export interface MachineListParams {
  limit: number
  offset: number
  status?: MachineStatusFilter
}

export interface AlertListParams {
  limit: number
  offset: number
  status?: AlertStatusFilter
  severity?: AlertSeverityFilter
  alert_type?: string
  machine_code?: string
}

export const queryKeys = {
  health: () => ['health'] as const,
  fleetOverview: () => ['fleet', 'overview'] as const,
  machines: (params: MachineListParams) => ['machines', params] as const,
  machineDetail: (machineCode: string) => ['machines', machineCode, 'detail'] as const,
  machinePredictions: (machineCode: string, limit: number, offset: number) =>
    ['machines', machineCode, 'predictions', { limit, offset }] as const,
  predictionExplanation: (machineCode: string, eventId: string) =>
    ['machines', machineCode, 'predictions', eventId, 'explanation'] as const,
  machineAnomalies: (
    machineCode: string,
    limit: number,
    offset: number,
    flaggedOnly: boolean,
  ) => ['machines', machineCode, 'anomalies', { limit, offset, flaggedOnly }] as const,
  latestDrift: () => ['drift', 'latest'] as const,
  alerts: (params: AlertListParams) => ['alerts', params] as const,
  copilotHealth: () => ['copilot', 'health'] as const,
}

export function useHealthStatus() {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: () => fetchJson<HealthResponse>('/health'),
    staleTime: 30_000,
    refetchInterval: HEALTH_REFETCH_INTERVAL_MS,
    retry: false,
  })
}

export function useFleetOverview() {
  return useQuery({
    queryKey: queryKeys.fleetOverview(),
    queryFn: () => fetchJson<FleetOverviewResponse>('/api/v1/fleet/overview'),
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useMachines(params: MachineListParams) {
  return useQuery({
    queryKey: queryKeys.machines(params),
    queryFn: () => fetchJson<MachineListResponse>('/api/v1/machines', { ...params }),
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useMachineDetail(machineCode: string | undefined) {
  const code = machineCode ?? ''

  return useQuery({
    queryKey: queryKeys.machineDetail(code),
    queryFn: () => fetchJson<MachineDetailResponse>(`/api/v1/machines/${encodeURIComponent(code)}`),
    enabled: code.length > 0,
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useMachinePredictions(
  machineCode: string | undefined,
  limit = 5,
  offset = 0,
) {
  const code = machineCode ?? ''

  return useQuery({
    queryKey: queryKeys.machinePredictions(code, limit, offset),
    queryFn: () =>
      fetchJson<PredictionListResponse>(`/api/v1/machines/${encodeURIComponent(code)}/predictions`, {
        limit,
        offset,
      }),
    enabled: code.length > 0,
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function usePredictionExplanation(
  machineCode: string | undefined,
  eventId: string | null | undefined,
) {
  const code = machineCode ?? ''
  const selectedEventId = eventId ?? ''

  return useQuery({
    queryKey: queryKeys.predictionExplanation(code, selectedEventId),
    queryFn: () =>
      fetchJson<PredictionExplanationResponse>(
        `/api/v1/machines/${encodeURIComponent(code)}/predictions/${encodeURIComponent(
          selectedEventId,
        )}/explanation`,
      ),
    enabled: code.length > 0 && selectedEventId.length > 0,
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useMachineAnomalies(
  machineCode: string | undefined,
  limit = 5,
  offset = 0,
  flaggedOnly = false,
) {
  const code = machineCode ?? ''

  return useQuery({
    queryKey: queryKeys.machineAnomalies(code, limit, offset, flaggedOnly),
    queryFn: () =>
      fetchJson<AnomalyListResponse>(`/api/v1/machines/${encodeURIComponent(code)}/anomalies`, {
        limit,
        offset,
        flagged_only: flaggedOnly,
      }),
    enabled: code.length > 0,
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useLatestDrift() {
  return useQuery({
    queryKey: queryKeys.latestDrift(),
    queryFn: () => fetchJson<DriftLatestResponse>('/api/v1/drift/latest'),
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useAlerts(params: AlertListParams) {
  return useQuery({
    queryKey: queryKeys.alerts(params),
    queryFn: () => fetchJson<AlertListResponse>('/api/v1/alerts', { ...params }),
    staleTime: DEFAULT_STALE_TIME_MS,
    retry: false,
  })
}

export function useCopilotHealth() {
  return useQuery({
    queryKey: queryKeys.copilotHealth(),
    queryFn: () => fetchJson<CopilotHealthResponse>('/api/v1/copilot/health'),
    staleTime: 30_000,
    refetchInterval: HEALTH_REFETCH_INTERVAL_MS,
    retry: false,
  })
}

export function sendCopilotMessage(request: CopilotChatRequest) {
  return fetchJson<CopilotChatResponse>('/api/v1/copilot/chat', {}, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
}

