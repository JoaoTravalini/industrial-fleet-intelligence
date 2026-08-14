-- Add auditable data drift monitoring history.
-- This migration is additive and does not create alerts or modify predictions,
-- anomaly audit rows, machine_health, or telemetry lakehouse data.

CREATE TABLE drift_snapshots (
    drift_snapshot_id BIGSERIAL,
    monitor_version TEXT NOT NULL,
    reference_profile_sha256 TEXT NOT NULL,
    ai4i_reference_identity JSONB NOT NULL,
    anomaly_reference_identity JSONB NOT NULL,
    ai4i_current_data_hash TEXT NOT NULL,
    anomaly_current_data_hash TEXT NOT NULL,
    ai4i_overall_status TEXT NOT NULL,
    anomaly_overall_status TEXT NOT NULL,
    ai4i_current_count INTEGER NOT NULL,
    anomaly_current_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_drift_snapshots PRIMARY KEY (drift_snapshot_id),
    CONSTRAINT ck_drift_snapshots_monitor_version_not_blank
        CHECK (btrim(monitor_version) <> ''),
    CONSTRAINT ck_drift_snapshots_reference_profile_hash_format
        CHECK (reference_profile_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_drift_snapshots_current_hash_format
        CHECK (
            ai4i_current_data_hash ~ '^[0-9a-f]{64}$'
            AND anomaly_current_data_hash ~ '^[0-9a-f]{64}$'
        ),
    CONSTRAINT ck_drift_snapshots_status
        CHECK (
            ai4i_overall_status IN ('stable', 'watch', 'drift')
            AND anomaly_overall_status IN ('stable', 'watch', 'drift')
        ),
    CONSTRAINT ck_drift_snapshots_counts
        CHECK (ai4i_current_count > 0 AND anomaly_current_count > 0)
);

CREATE UNIQUE INDEX uq_drift_snapshots_business_identity
    ON drift_snapshots (
        monitor_version,
        reference_profile_sha256,
        ai4i_current_data_hash,
        anomaly_current_data_hash
    );

CREATE INDEX idx_drift_snapshots_reference
    ON drift_snapshots (monitor_version, reference_profile_sha256, drift_snapshot_id DESC);

CREATE TABLE drift_feature_metrics (
    drift_feature_metric_id BIGSERIAL,
    drift_snapshot_id BIGINT NOT NULL,
    monitor_scope TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    psi NUMERIC NOT NULL,
    status TEXT NOT NULL,
    reference_count INTEGER NOT NULL,
    current_count INTEGER NOT NULL,
    reference_mean NUMERIC,
    current_mean NUMERIC,
    reference_std NUMERIC,
    current_std NUMERIC,
    reference_min NUMERIC,
    reference_max NUMERIC,
    current_min NUMERIC,
    current_max NUMERIC,
    standardized_mean_shift NUMERIC,
    outside_reference_range_count INTEGER,
    outside_reference_range_rate NUMERIC,
    reference_proportions JSONB NOT NULL,
    current_proportions JSONB NOT NULL,
    bin_edges JSONB,
    diagnostics JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_drift_feature_metrics PRIMARY KEY (drift_feature_metric_id),
    CONSTRAINT fk_drift_feature_metrics_snapshot
        FOREIGN KEY (drift_snapshot_id)
        REFERENCES drift_snapshots (drift_snapshot_id)
        ON DELETE CASCADE,
    CONSTRAINT ck_drift_feature_metrics_scope
        CHECK (monitor_scope IN ('ai4i_model_input', 'operational_anomaly_inputs')),
    CONSTRAINT ck_drift_feature_metrics_type
        CHECK (feature_type IN ('numeric', 'categorical')),
    CONSTRAINT ck_drift_feature_metrics_status
        CHECK (status IN ('stable', 'watch', 'drift')),
    CONSTRAINT ck_drift_feature_metrics_psi
        CHECK (psi >= 0),
    CONSTRAINT ck_drift_feature_metrics_counts
        CHECK (reference_count > 0 AND current_count > 0),
    CONSTRAINT ck_drift_feature_metrics_range_rate
        CHECK (
            outside_reference_range_rate IS NULL
            OR (
                outside_reference_range_rate >= 0
                AND outside_reference_range_rate <= 1
            )
        ),
    CONSTRAINT ck_drift_feature_metrics_numeric_fields
        CHECK (
            (
                feature_type = 'numeric'
                AND reference_mean IS NOT NULL
                AND current_mean IS NOT NULL
                AND reference_std IS NOT NULL
                AND current_std IS NOT NULL
                AND reference_min IS NOT NULL
                AND reference_max IS NOT NULL
                AND current_min IS NOT NULL
                AND current_max IS NOT NULL
                AND outside_reference_range_count IS NOT NULL
                AND outside_reference_range_rate IS NOT NULL
                AND bin_edges IS NOT NULL
            )
            OR
            (
                feature_type = 'categorical'
                AND reference_mean IS NULL
                AND current_mean IS NULL
                AND reference_std IS NULL
                AND current_std IS NULL
                AND reference_min IS NULL
                AND reference_max IS NULL
                AND current_min IS NULL
                AND current_max IS NULL
                AND standardized_mean_shift IS NULL
                AND outside_reference_range_count IS NULL
                AND outside_reference_range_rate IS NULL
                AND bin_edges IS NULL
            )
        )
);

CREATE UNIQUE INDEX uq_drift_feature_metrics_identity
    ON drift_feature_metrics (drift_snapshot_id, monitor_scope, feature_name);

CREATE INDEX idx_drift_feature_metrics_status
    ON drift_feature_metrics (monitor_scope, status, psi DESC);
