-- Initial operational PostgreSQL schema for the Industrial Fleet Intelligence Platform.
-- This schema stores structured operational entities only; raw telemetry history belongs
-- in the planned Kafka, Spark, and local lake layers instead of PostgreSQL.

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TABLE machines (
    machine_id BIGINT GENERATED ALWAYS AS IDENTITY,
    machine_identifier TEXT NOT NULL,
    machine_type TEXT NOT NULL,
    model_family TEXT NOT NULL,
    commissioned_on DATE,
    operational_status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_machines PRIMARY KEY (machine_id),
    CONSTRAINT uq_machines_machine_identifier UNIQUE (machine_identifier),
    CONSTRAINT ck_machines_machine_identifier_format
        CHECK (machine_identifier ~ '^MCH-[0-9]{4,}$'),
    CONSTRAINT ck_machines_machine_type_not_blank
        CHECK (btrim(machine_type) <> ''),
    CONSTRAINT ck_machines_model_family_not_blank
        CHECK (btrim(model_family) <> ''),
    CONSTRAINT ck_machines_operational_status
        CHECK (operational_status IN ('active', 'maintenance', 'inactive'))
);

CREATE TRIGGER trg_machines_set_updated_at
BEFORE UPDATE ON machines
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE maintenance_records (
    maintenance_record_id BIGINT GENERATED ALWAYS AS IDENTITY,
    machine_id BIGINT NOT NULL,
    maintenance_at TIMESTAMPTZ NOT NULL,
    maintenance_type TEXT NOT NULL,
    description TEXT NOT NULL,
    is_scheduled BOOLEAN NOT NULL,
    downtime_minutes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_maintenance_records PRIMARY KEY (maintenance_record_id),
    CONSTRAINT fk_maintenance_records_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines (machine_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    CONSTRAINT ck_maintenance_records_type_not_blank
        CHECK (btrim(maintenance_type) <> ''),
    CONSTRAINT ck_maintenance_records_description_not_blank
        CHECK (btrim(description) <> ''),
    CONSTRAINT ck_maintenance_records_downtime_minutes
        CHECK (downtime_minutes IS NULL OR downtime_minutes >= 0)
);

CREATE INDEX idx_maintenance_records_machine_timestamp
    ON maintenance_records (machine_id, maintenance_at DESC);

CREATE TABLE model_predictions (
    model_prediction_id BIGINT GENERATED ALWAYS AS IDENTITY,
    machine_id BIGINT NOT NULL,
    prediction_at TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    predicted_value NUMERIC(14, 4) NOT NULL,
    confidence NUMERIC(6, 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_model_predictions PRIMARY KEY (model_prediction_id),
    CONSTRAINT fk_model_predictions_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines (machine_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    CONSTRAINT ck_model_predictions_model_name_not_blank
        CHECK (btrim(model_name) <> ''),
    CONSTRAINT ck_model_predictions_model_version_not_blank
        CHECK (btrim(model_version) <> ''),
    CONSTRAINT ck_model_predictions_type_not_blank
        CHECK (btrim(prediction_type) <> ''),
    CONSTRAINT ck_model_predictions_confidence
        CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX idx_model_predictions_machine_timestamp
    ON model_predictions (machine_id, prediction_at DESC);

CREATE TABLE anomalies (
    anomaly_id BIGINT GENERATED ALWAYS AS IDENTITY,
    machine_id BIGINT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    anomaly_score NUMERIC(6, 5) NOT NULL,
    anomaly_type TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_anomalies PRIMARY KEY (anomaly_id),
    CONSTRAINT fk_anomalies_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines (machine_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    CONSTRAINT ck_anomalies_score
        CHECK (anomaly_score >= 0 AND anomaly_score <= 1),
    CONSTRAINT ck_anomalies_type_not_blank
        CHECK (btrim(anomaly_type) <> '')
);

CREATE INDEX idx_anomalies_machine_timestamp
    ON anomalies (machine_id, detected_at DESC);

CREATE TABLE alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY,
    machine_id BIGINT NOT NULL,
    model_prediction_id BIGINT,
    anomaly_id BIGINT,
    severity TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    CONSTRAINT pk_alerts PRIMARY KEY (alert_id),
    CONSTRAINT fk_alerts_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines (machine_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    CONSTRAINT fk_alerts_model_prediction
        FOREIGN KEY (model_prediction_id)
        REFERENCES model_predictions (model_prediction_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,
    CONSTRAINT fk_alerts_anomaly
        FOREIGN KEY (anomaly_id)
        REFERENCES anomalies (anomaly_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,
    CONSTRAINT ck_alerts_severity
        CHECK (severity IN ('info', 'warning', 'critical')),
    CONSTRAINT ck_alerts_type_not_blank
        CHECK (btrim(alert_type) <> ''),
    CONSTRAINT ck_alerts_title_not_blank
        CHECK (btrim(title) <> ''),
    CONSTRAINT ck_alerts_status
        CHECK (status IN ('open', 'acknowledged', 'resolved')),
    CONSTRAINT ck_alerts_acknowledged_after_created
        CHECK (acknowledged_at IS NULL OR acknowledged_at >= created_at),
    CONSTRAINT ck_alerts_resolved_after_created
        CHECK (resolved_at IS NULL OR resolved_at >= created_at),
    CONSTRAINT ck_alerts_resolved_after_acknowledged
        CHECK (
            acknowledged_at IS NULL
            OR resolved_at IS NULL
            OR resolved_at >= acknowledged_at
        ),
    CONSTRAINT ck_alerts_status_timestamp_consistency
        CHECK (
            (status = 'open' AND acknowledged_at IS NULL AND resolved_at IS NULL)
            OR (status = 'acknowledged' AND acknowledged_at IS NOT NULL AND resolved_at IS NULL)
            OR (status = 'resolved' AND resolved_at IS NOT NULL)
        )
);

CREATE INDEX idx_alerts_machine_status_severity
    ON alerts (machine_id, status, severity, created_at DESC);

CREATE TABLE machine_health (
    machine_id BIGINT NOT NULL,
    health_score NUMERIC(5, 2) NOT NULL,
    failure_risk NUMERIC(6, 5),
    anomaly_score NUMERIC(6, 5),
    health_classification TEXT NOT NULL DEFAULT 'unknown',
    last_telemetry_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_machine_health PRIMARY KEY (machine_id),
    CONSTRAINT fk_machine_health_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines (machine_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT ck_machine_health_health_score
        CHECK (health_score >= 0 AND health_score <= 100),
    CONSTRAINT ck_machine_health_failure_risk
        CHECK (failure_risk IS NULL OR (failure_risk >= 0 AND failure_risk <= 1)),
    CONSTRAINT ck_machine_health_anomaly_score
        CHECK (anomaly_score IS NULL OR (anomaly_score >= 0 AND anomaly_score <= 1)),
    CONSTRAINT ck_machine_health_classification
        CHECK (health_classification IN ('healthy', 'attention', 'critical', 'unknown'))
);

CREATE TRIGGER trg_machine_health_set_updated_at
BEFORE UPDATE ON machine_health
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
