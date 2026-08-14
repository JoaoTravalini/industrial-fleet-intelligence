-- Add deterministic operational alert identity indexes.
-- This migration is additive and does not create alerts or alter existing alert lifecycle data.

CREATE UNIQUE INDEX uq_alerts_model_prediction_policy_identity
    ON alerts (alert_type, model_prediction_id)
    WHERE model_prediction_id IS NOT NULL;

CREATE UNIQUE INDEX uq_alerts_anomaly_policy_identity
    ON alerts (alert_type, anomaly_id)
    WHERE anomaly_id IS NOT NULL;