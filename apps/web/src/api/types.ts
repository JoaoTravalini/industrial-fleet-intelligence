export type ISODate = string
export type ISODateTime = string

export interface HealthResponse {
  status: 'ok'
  database: 'connected'
}

export interface LatestPredictionProjection {
  event_time: ISODateTime | null
  failure_probability: number | null
  failure_prediction: boolean | null
  frozen_threshold: number | null
  model_name: string | null
  model_version: string | null
  final_config_hash: string | null
}

export interface MachineSummary {
  machine_code: string
  machine_type: string
  model_family: string
  commissioned_on: ISODate | null
  operational_status: string
  latest_prediction: LatestPredictionProjection | null
}

export interface MachineListResponse {
  items: MachineSummary[]
  limit: number
  offset: number
  count: number
  total: number
}

export interface MachineDetailResponse extends MachineSummary {
  prediction_history_count: number
  anomaly_audit_count: number
}

export interface SourceLineage {
  source_kafka_topic: string | null
  source_kafka_partition: number | null
  source_kafka_offset: number | null
  source_kafka_timestamp: ISODateTime | null
  source_kafka_key: string | null
  payload_sha256: string | null
}

export interface PredictionResponse {
  model_prediction_id: number
  event_id: string | null
  event_time: ISODateTime | null
  failure_probability: number | null
  failure_prediction: boolean | null
  decision_semantics: 'model_decision_not_observed_failure'
  frozen_threshold: number | null
  model_name: string
  model_version: string
  final_config_hash: string | null
  adapter_version: string | null
  model_input_sha256: string | null
  lineage: SourceLineage
}

export interface PredictionListResponse {
  machine_code: string
  items: PredictionResponse[]
  limit: number
  offset: number
  count: number
  total: number
}

export interface ExplanationFeatureContribution {
  feature_name: string
  feature_value: string | number
  shap_value: number
}

export interface PredictionExplanationResponse {
  prediction_explanation_id: number
  model_prediction_id: number
  event_id: string
  machine_code: string
  event_time: ISODateTime
  failure_probability: number
  failure_prediction: boolean
  decision_semantics: 'model_decision_not_observed_failure'
  frozen_threshold: number
  model_name: string
  model_version: string
  final_config_hash: string
  model_input_sha256: string
  explainer_name: string
  explainer_version: string
  explanation_config_hash: string
  output_semantics: 'positive_class_failure_risk_model_output'
  attribution_semantics: 'shap_model_attribution_not_causality'
  positive_contribution_semantics: 'positive_shap_pushes_model_output_toward_higher_failure_risk'
  negative_contribution_semantics: 'negative_shap_pushes_model_output_toward_lower_failure_risk'
  base_value: number
  model_output_value: number
  contribution_sum: number
  additivity_error: number
  feature_contributions: ExplanationFeatureContribution[]
  lineage: SourceLineage
}

export interface AnomalyResponse {
  anomaly_id: number
  event_id: string | null
  event_time: ISODateTime | null
  vibration_mm_s: number | null
  pressure_bar: number | null
  anomaly_score: number
  anomaly_flag: boolean | null
  score_semantics: 'anomaly_score_not_probability'
  model_name: string | null
  model_version: string | null
  model_config_hash: string | null
  baseline_event_id_sha256: string | null
  baseline_feature_data_sha256: string | null
  lineage: SourceLineage
}

export interface AnomalyListResponse {
  machine_code: string
  flagged_only: boolean
  items: AnomalyResponse[]
  limit: number
  offset: number
  count: number
  total: number
}

export interface FleetOverviewResponse {
  machine_count: number
  machines_with_prediction_projection: number
  prediction_history_count: number
  positive_prediction_count: number
  negative_prediction_count: number
  mean_failure_probability: number | null
  max_failure_probability: number | null
  anomaly_audit_count: number
  flagged_anomaly_count: number
  non_flagged_anomaly_count: number
  latest_ai4i_drift_status: string | null
  latest_anomaly_drift_status: string | null
  open_alert_count: number
}

export interface DriftFeatureMetric {
  feature_name: string
  feature_type: string
  psi: number
  status: string
  reference_count: number
  current_count: number
  reference_mean: number | null
  current_mean: number | null
  reference_std: number | null
  current_std: number | null
  reference_min: number | null
  reference_max: number | null
  current_min: number | null
  current_max: number | null
  standardized_mean_shift: number | null
  outside_reference_range_count: number | null
  outside_reference_range_rate: number | null
  reference_proportions: unknown
  current_proportions: unknown
  bin_edges: unknown | null
  diagnostics: Record<string, unknown>
}

export interface DriftLatestResponse {
  drift_snapshot_id: number | null
  monitor_version: string | null
  reference_profile_sha256: string | null
  ai4i_reference_identity: Record<string, unknown> | null
  anomaly_reference_identity: Record<string, unknown> | null
  ai4i_current_data_hash: string | null
  anomaly_current_data_hash: string | null
  ai4i_overall_status: string | null
  anomaly_overall_status: string | null
  ai4i_current_count: number | null
  anomaly_current_count: number | null
  created_at: ISODateTime | null
  features_by_scope: Record<string, DriftFeatureMetric[]>
}

export interface AlertResponse {
  alert_id: number
  machine_code: string
  severity: string
  alert_type: string
  title: string
  description: string | null
  status: string
  source_kind: 'model_prediction' | 'anomaly' | 'unknown'
  model_prediction_id: number | null
  anomaly_id: number | null
  source_event_id: string | null
  source_observed_at: ISODateTime | null
  created_at: ISODateTime
}

export interface AlertListResponse {
  items: AlertResponse[]
  limit: number
  offset: number
  count: number
  total: number
}
