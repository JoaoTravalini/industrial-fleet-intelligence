import type {
  AlertListResponse,
  AlertResponse,
  AnomalyListResponse,
  AnomalyResponse,
  DriftLatestResponse,
  FleetOverviewResponse,
  HealthResponse,
  MachineDetailResponse,
  MachineListResponse,
  MachineSummary,
  PredictionListResponse,
  PredictionResponse,
  SourceLineage,
} from '../api/types'

const lineage: SourceLineage = {
  source_kafka_topic: 'industrial.telemetry.v1',
  source_kafka_partition: 0,
  source_kafka_offset: 42,
  source_kafka_timestamp: '2026-08-14T12:00:00Z',
  source_kafka_key: 'MCH-0001',
  payload_sha256: 'test-only-payload-hash',
}

export const healthFixture: HealthResponse = {
  status: 'ok',
  database: 'connected',
}

export const fleetOverviewFixture: FleetOverviewResponse = {
  machine_count: 100,
  machines_with_prediction_projection: 96,
  prediction_history_count: 106,
  positive_prediction_count: 7,
  negative_prediction_count: 99,
  mean_failure_probability: 0.033333,
  max_failure_probability: 0.74,
  anomaly_audit_count: 106,
  flagged_anomaly_count: 4,
  non_flagged_anomaly_count: 102,
  latest_ai4i_drift_status: 'watch',
  latest_anomaly_drift_status: 'stable',
  open_alert_count: 5,
}

export function machineSummaryFixture(machineCode = 'MCH-0001'): MachineSummary {
  return {
    machine_code: machineCode,
    machine_type: 'Hydraulic Press',
    model_family: 'HX-200',
    commissioned_on: '2024-01-15',
    operational_status: 'active',
    latest_prediction: {
      event_time: '2026-08-14T12:05:00Z',
      failure_probability: 0.033333,
      failure_prediction: false,
      frozen_threshold: 0.42,
      model_name: 'ai4i_random_forest',
      model_version: '1.0.0',
      final_config_hash: 'test-only-config-hash',
    },
  }
}

export const machineDetailFixture: MachineDetailResponse = {
  ...machineSummaryFixture('MCH-0001'),
  prediction_history_count: 12,
  anomaly_audit_count: 8,
}

export const predictionFixture: PredictionResponse = {
  model_prediction_id: 501,
  event_id: 'evt-test-001',
  event_time: '2026-08-14T12:05:00Z',
  failure_probability: 0.033333,
  failure_prediction: false,
  decision_semantics: 'model_decision_not_observed_failure',
  frozen_threshold: 0.42,
  model_name: 'ai4i_random_forest',
  model_version: '1.0.0',
  final_config_hash: 'test-only-config-hash',
  adapter_version: '1.0.0',
  model_input_sha256: 'test-only-input-hash',
  lineage,
}

export const anomalyFixture: AnomalyResponse = {
  anomaly_id: 901,
  event_id: 'evt-test-001',
  event_time: '2026-08-14T12:05:00Z',
  vibration_mm_s: 4.2,
  pressure_bar: 118.5,
  anomaly_score: 0.84,
  anomaly_flag: true,
  score_semantics: 'anomaly_score_not_probability',
  model_name: 'telemetry_isolation_forest',
  model_version: '1.0.0',
  model_config_hash: 'test-only-anomaly-config-hash',
  baseline_event_id_sha256: 'test-only-baseline-events',
  baseline_feature_data_sha256: 'test-only-baseline-features',
  lineage,
}

export const alertFixture: AlertResponse = {
  alert_id: 301,
  machine_code: 'MCH-0001',
  severity: 'critical',
  alert_type: 'telemetry_anomaly',
  title: 'Bearing anomaly watch',
  description: 'Test-only alert materialized from anomaly state.',
  status: 'open',
  source_kind: 'anomaly',
  model_prediction_id: null,
  anomaly_id: 901,
  source_event_id: 'evt-test-001',
  source_observed_at: '2026-08-14T12:05:00Z',
  created_at: '2026-08-14T12:06:00Z',
}

export function machineListFixture(offset = 0): MachineListResponse {
  const count = offset >= 40 ? 5 : 20
  const items = Array.from({ length: count }, (_, index) => {
    const sequence = String(offset + index + 1).padStart(4, '0')
    return machineSummaryFixture(`MCH-${sequence}`)
  })

  return {
    items,
    limit: 20,
    offset,
    count,
    total: 45,
  }
}

export const predictionListFixture: PredictionListResponse = {
  machine_code: 'MCH-0001',
  items: [predictionFixture],
  limit: 5,
  offset: 0,
  count: 1,
  total: 1,
}

export const anomalyListFixture: AnomalyListResponse = {
  machine_code: 'MCH-0001',
  flagged_only: false,
  items: [anomalyFixture],
  limit: 5,
  offset: 0,
  count: 1,
  total: 1,
}

export const alertListFixture: AlertListResponse = {
  items: [alertFixture],
  limit: 20,
  offset: 0,
  count: 1,
  total: 1,
}

export const emptyAlertListFixture: AlertListResponse = {
  items: [],
  limit: 20,
  offset: 0,
  count: 0,
  total: 0,
}

export const driftFixture: DriftLatestResponse = {
  drift_snapshot_id: 17,
  monitor_version: '1.0.0',
  reference_profile_sha256: 'test-only-reference-hash',
  ai4i_reference_identity: { source: 'test-only-ai4i-reference' },
  anomaly_reference_identity: { source: 'test-only-anomaly-reference' },
  ai4i_current_data_hash: 'test-only-ai4i-current-hash',
  anomaly_current_data_hash: 'test-only-anomaly-current-hash',
  ai4i_overall_status: 'watch',
  anomaly_overall_status: 'stable',
  ai4i_current_count: 106,
  anomaly_current_count: 106,
  created_at: '2026-08-14T12:10:00Z',
  features_by_scope: {
    ai4i_model_input: [
      {
        feature_name: 'air_temperature_k',
        feature_type: 'numeric',
        psi: 0.12,
        status: 'watch',
        reference_count: 6000,
        current_count: 106,
        reference_mean: 300.0,
        current_mean: 301.4,
        reference_std: 2.0,
        current_std: 1.8,
        reference_min: 295.0,
        reference_max: 305.0,
        current_min: 296.2,
        current_max: 306.1,
        standardized_mean_shift: 0.7,
        outside_reference_range_count: 1,
        outside_reference_range_rate: 0.0094,
        reference_proportions: {},
        current_proportions: {},
        bin_edges: [295, 300, 305],
        diagnostics: { test_scope: 'ai4i' },
      },
    ],
    operational_anomaly_inputs: [
      {
        feature_name: 'vibration_mm_s',
        feature_type: 'numeric',
        psi: 0.03,
        status: 'stable',
        reference_count: 5000,
        current_count: 106,
        reference_mean: 3.2,
        current_mean: 3.3,
        reference_std: 0.4,
        current_std: 0.5,
        reference_min: 2.1,
        reference_max: 5.0,
        current_min: 2.3,
        current_max: 4.9,
        standardized_mean_shift: 0.25,
        outside_reference_range_count: 0,
        outside_reference_range_rate: 0,
        reference_proportions: {},
        current_proportions: {},
        bin_edges: [2, 3, 4, 5],
        diagnostics: { test_scope: 'anomaly' },
      },
    ],
  },
}
