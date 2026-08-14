-- Add auditable telemetry anomaly detector provenance to the existing anomalies table.
-- This migration is additive and does not create alerts, update machine_health, or modify AI4I.

ALTER TABLE anomalies
    ADD COLUMN event_id UUID,
    ADD COLUMN event_time TIMESTAMPTZ,
    ADD COLUMN anomaly_flag BOOLEAN,
    ADD COLUMN model_name TEXT,
    ADD COLUMN model_version TEXT,
    ADD COLUMN model_config_hash TEXT,
    ADD COLUMN baseline_event_id_sha256 TEXT,
    ADD COLUMN baseline_feature_data_sha256 TEXT,
    ADD COLUMN vibration_mm_s NUMERIC(10, 6),
    ADD COLUMN pressure_bar NUMERIC(10, 6),
    ADD COLUMN source_kafka_topic TEXT,
    ADD COLUMN source_kafka_partition INTEGER,
    ADD COLUMN source_kafka_offset BIGINT,
    ADD COLUMN source_kafka_timestamp TIMESTAMPTZ,
    ADD COLUMN source_kafka_key TEXT,
    ADD COLUMN payload_sha256 TEXT;

ALTER TABLE anomalies
    ADD CONSTRAINT ck_anomalies_telemetry_required_fields
        CHECK (
            anomaly_type <> 'telemetry_isolation_forest_score'
            OR (
                event_id IS NOT NULL
                AND event_time IS NOT NULL
                AND anomaly_flag IS NOT NULL
                AND model_name IS NOT NULL
                AND btrim(model_name) <> ''
                AND model_version IS NOT NULL
                AND btrim(model_version) <> ''
                AND model_config_hash IS NOT NULL
                AND baseline_event_id_sha256 IS NOT NULL
                AND baseline_feature_data_sha256 IS NOT NULL
                AND vibration_mm_s IS NOT NULL
                AND pressure_bar IS NOT NULL
                AND source_kafka_topic IS NOT NULL
                AND btrim(source_kafka_topic) <> ''
                AND source_kafka_partition IS NOT NULL
                AND source_kafka_partition >= 0
                AND source_kafka_offset IS NOT NULL
                AND source_kafka_offset >= 0
                AND source_kafka_timestamp IS NOT NULL
                AND source_kafka_key IS NOT NULL
                AND btrim(source_kafka_key) <> ''
                AND payload_sha256 IS NOT NULL
            )
        ),
    ADD CONSTRAINT ck_anomalies_telemetry_model_config_hash_format
        CHECK (model_config_hash IS NULL OR model_config_hash ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ck_anomalies_telemetry_baseline_event_hash_format
        CHECK (baseline_event_id_sha256 IS NULL OR baseline_event_id_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ck_anomalies_telemetry_baseline_feature_hash_format
        CHECK (
            baseline_feature_data_sha256 IS NULL
            OR baseline_feature_data_sha256 ~ '^[0-9a-f]{64}$'
        ),
    ADD CONSTRAINT ck_anomalies_telemetry_payload_hash_format
        CHECK (payload_sha256 IS NULL OR payload_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ck_anomalies_telemetry_feature_values
        CHECK (
            (vibration_mm_s IS NULL OR vibration_mm_s >= 0)
            AND (pressure_bar IS NULL OR pressure_bar >= 0)
        );

CREATE UNIQUE INDEX uq_anomalies_telemetry_business_identity
    ON anomalies (event_id, model_name, model_version, model_config_hash)
    WHERE anomaly_type = 'telemetry_isolation_forest_score';

CREATE INDEX idx_anomalies_telemetry_model_scope
    ON anomalies (
        model_name,
        model_version,
        model_config_hash,
        anomaly_flag,
        detected_at DESC
    )
    WHERE anomaly_type = 'telemetry_isolation_forest_score';
