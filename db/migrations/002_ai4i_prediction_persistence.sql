-- Add auditable AI4I telemetry prediction persistence fields and latest projection fields.
-- This migration is additive and does not create alerts, anomalies, raw telemetry history,
-- or model outputs.

ALTER TABLE model_predictions
    ADD COLUMN event_id UUID,
    ADD COLUMN event_time TIMESTAMPTZ,
    ADD COLUMN failure_probability NUMERIC(10, 6),
    ADD COLUMN failure_prediction BOOLEAN,
    ADD COLUMN frozen_threshold NUMERIC(10, 6),
    ADD COLUMN final_config_hash TEXT,
    ADD COLUMN adapter_version TEXT,
    ADD COLUMN model_input_sha256 TEXT,
    ADD COLUMN source_kafka_topic TEXT,
    ADD COLUMN source_kafka_partition INTEGER,
    ADD COLUMN source_kafka_offset BIGINT,
    ADD COLUMN source_kafka_timestamp TIMESTAMPTZ,
    ADD COLUMN source_kafka_key TEXT,
    ADD COLUMN payload_sha256 TEXT;

ALTER TABLE model_predictions
    ADD CONSTRAINT ck_model_predictions_ai4i_required_fields
        CHECK (
            prediction_type <> 'ai4i_failure_risk'
            OR (
                event_id IS NOT NULL
                AND event_time IS NOT NULL
                AND failure_probability IS NOT NULL
                AND failure_prediction IS NOT NULL
                AND frozen_threshold IS NOT NULL
                AND final_config_hash IS NOT NULL
                AND adapter_version IS NOT NULL
                AND model_input_sha256 IS NOT NULL
                AND source_kafka_topic IS NOT NULL
                AND source_kafka_partition IS NOT NULL
                AND source_kafka_offset IS NOT NULL
                AND source_kafka_timestamp IS NOT NULL
                AND source_kafka_key IS NOT NULL
                AND payload_sha256 IS NOT NULL
            )
        ),
    ADD CONSTRAINT ck_model_predictions_failure_probability
        CHECK (failure_probability IS NULL OR failure_probability BETWEEN 0 AND 1),
    ADD CONSTRAINT ck_model_predictions_failure_prediction_consistency
        CHECK (
            failure_probability IS NULL
            OR failure_prediction IS NULL
            OR frozen_threshold IS NULL
            OR failure_prediction = (failure_probability >= frozen_threshold)
        ),
    ADD CONSTRAINT ck_model_predictions_frozen_threshold
        CHECK (frozen_threshold IS NULL OR frozen_threshold BETWEEN 0 AND 1),
    ADD CONSTRAINT ck_model_predictions_final_config_hash_format
        CHECK (final_config_hash IS NULL OR final_config_hash ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ck_model_predictions_adapter_version_not_blank
        CHECK (adapter_version IS NULL OR btrim(adapter_version) <> ''),
    ADD CONSTRAINT ck_model_predictions_model_input_sha256_format
        CHECK (model_input_sha256 IS NULL OR model_input_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT ck_model_predictions_source_kafka_topic_not_blank
        CHECK (source_kafka_topic IS NULL OR btrim(source_kafka_topic) <> ''),
    ADD CONSTRAINT ck_model_predictions_source_kafka_partition
        CHECK (source_kafka_partition IS NULL OR source_kafka_partition >= 0),
    ADD CONSTRAINT ck_model_predictions_source_kafka_offset
        CHECK (source_kafka_offset IS NULL OR source_kafka_offset >= 0),
    ADD CONSTRAINT ck_model_predictions_source_kafka_key_not_blank
        CHECK (source_kafka_key IS NULL OR btrim(source_kafka_key) <> ''),
    ADD CONSTRAINT ck_model_predictions_payload_sha256_format
        CHECK (payload_sha256 IS NULL OR payload_sha256 ~ '^[0-9a-f]{64}$');

CREATE UNIQUE INDEX uq_model_predictions_ai4i_business_identity
    ON model_predictions (event_id, model_name, model_version, final_config_hash)
    WHERE prediction_type = 'ai4i_failure_risk';

CREATE INDEX idx_model_predictions_ai4i_latest
    ON model_predictions (
        machine_id,
        model_name,
        model_version,
        final_config_hash,
        event_time DESC,
        source_kafka_timestamp DESC,
        source_kafka_topic DESC,
        source_kafka_partition DESC,
        source_kafka_offset DESC,
        event_id DESC
    )
    WHERE prediction_type = 'ai4i_failure_risk';

ALTER TABLE machine_health
    ALTER COLUMN health_score DROP NOT NULL,
    ADD COLUMN latest_model_prediction_id BIGINT,
    ADD COLUMN latest_prediction_event_id UUID,
    ADD COLUMN latest_prediction_at TIMESTAMPTZ,
    ADD COLUMN latest_failure_probability NUMERIC(10, 6),
    ADD COLUMN latest_failure_prediction BOOLEAN,
    ADD COLUMN latest_frozen_threshold NUMERIC(10, 6),
    ADD COLUMN latest_model_name TEXT,
    ADD COLUMN latest_model_version TEXT,
    ADD COLUMN latest_final_config_hash TEXT,
    ADD COLUMN latest_model_input_sha256 TEXT,
    ADD COLUMN latest_source_kafka_topic TEXT,
    ADD COLUMN latest_source_kafka_partition INTEGER,
    ADD COLUMN latest_source_kafka_offset BIGINT,
    ADD COLUMN latest_source_kafka_timestamp TIMESTAMPTZ,
    ADD COLUMN latest_source_kafka_key TEXT,
    ADD COLUMN latest_payload_sha256 TEXT;

ALTER TABLE machine_health
    ADD CONSTRAINT fk_machine_health_latest_model_prediction
        FOREIGN KEY (latest_model_prediction_id)
        REFERENCES model_predictions (model_prediction_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,
    ADD CONSTRAINT ck_machine_health_latest_prediction_required_fields
        CHECK (
            latest_model_prediction_id IS NULL
            OR (
                latest_prediction_event_id IS NOT NULL
                AND latest_prediction_at IS NOT NULL
                AND latest_failure_probability IS NOT NULL
                AND latest_failure_prediction IS NOT NULL
                AND latest_frozen_threshold IS NOT NULL
                AND latest_model_name IS NOT NULL
                AND latest_model_version IS NOT NULL
                AND latest_final_config_hash IS NOT NULL
                AND latest_model_input_sha256 IS NOT NULL
                AND latest_source_kafka_topic IS NOT NULL
                AND latest_source_kafka_partition IS NOT NULL
                AND latest_source_kafka_offset IS NOT NULL
                AND latest_source_kafka_timestamp IS NOT NULL
                AND latest_source_kafka_key IS NOT NULL
                AND latest_payload_sha256 IS NOT NULL
            )
        ),
    ADD CONSTRAINT ck_machine_health_latest_failure_probability
        CHECK (
            latest_failure_probability IS NULL
            OR latest_failure_probability BETWEEN 0 AND 1
        ),
    ADD CONSTRAINT ck_machine_health_latest_failure_prediction_consistency
        CHECK (
            latest_failure_probability IS NULL
            OR latest_failure_prediction IS NULL
            OR latest_frozen_threshold IS NULL
            OR latest_failure_prediction = (latest_failure_probability >= latest_frozen_threshold)
        ),
    ADD CONSTRAINT ck_machine_health_latest_frozen_threshold
        CHECK (
            latest_frozen_threshold IS NULL
            OR latest_frozen_threshold BETWEEN 0 AND 1
        ),
    ADD CONSTRAINT ck_machine_health_latest_model_name_not_blank
        CHECK (latest_model_name IS NULL OR btrim(latest_model_name) <> ''),
    ADD CONSTRAINT ck_machine_health_latest_model_version_not_blank
        CHECK (latest_model_version IS NULL OR btrim(latest_model_version) <> ''),
    ADD CONSTRAINT ck_machine_health_latest_final_config_hash_format
        CHECK (
            latest_final_config_hash IS NULL
            OR latest_final_config_hash ~ '^[0-9a-f]{64}$'
        ),
    ADD CONSTRAINT ck_machine_health_latest_model_input_sha256_format
        CHECK (
            latest_model_input_sha256 IS NULL
            OR latest_model_input_sha256 ~ '^[0-9a-f]{64}$'
        ),
    ADD CONSTRAINT ck_machine_health_latest_source_kafka_topic_not_blank
        CHECK (
            latest_source_kafka_topic IS NULL
            OR btrim(latest_source_kafka_topic) <> ''
        ),
    ADD CONSTRAINT ck_machine_health_latest_source_kafka_partition
        CHECK (
            latest_source_kafka_partition IS NULL
            OR latest_source_kafka_partition >= 0
        ),
    ADD CONSTRAINT ck_machine_health_latest_source_kafka_offset
        CHECK (
            latest_source_kafka_offset IS NULL
            OR latest_source_kafka_offset >= 0
        ),
    ADD CONSTRAINT ck_machine_health_latest_source_kafka_key_not_blank
        CHECK (
            latest_source_kafka_key IS NULL
            OR btrim(latest_source_kafka_key) <> ''
        ),
    ADD CONSTRAINT ck_machine_health_latest_payload_sha256_format
        CHECK (
            latest_payload_sha256 IS NULL
            OR latest_payload_sha256 ~ '^[0-9a-f]{64}$'
        );

CREATE INDEX idx_machine_health_latest_prediction
    ON machine_health (
        latest_model_name,
        latest_model_version,
        latest_final_config_hash,
        latest_prediction_at DESC,
        latest_source_kafka_timestamp DESC,
        latest_source_kafka_topic DESC,
        latest_source_kafka_partition DESC,
        latest_source_kafka_offset DESC,
        latest_prediction_event_id DESC
    );
