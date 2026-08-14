-- Add persisted operational AI4I prediction explanations.
-- This migration is additive and does not calculate SHAP, run model inference,
-- alter model predictions, or modify alert/anomaly/drift semantics.

CREATE TABLE prediction_explanations (
    prediction_explanation_id BIGSERIAL,
    model_prediction_id BIGINT NOT NULL,
    machine_id BIGINT NOT NULL,
    event_id UUID NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    model_input_sha256 TEXT NOT NULL,
    explainer_name TEXT NOT NULL,
    explainer_version TEXT NOT NULL,
    explanation_config_hash TEXT NOT NULL,
    output_semantics TEXT NOT NULL,
    attribution_semantics TEXT NOT NULL,
    base_value NUMERIC NOT NULL,
    model_output_value NUMERIC NOT NULL,
    contribution_sum NUMERIC NOT NULL,
    additivity_error NUMERIC NOT NULL,
    feature_contributions JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_prediction_explanations PRIMARY KEY (prediction_explanation_id),
    CONSTRAINT fk_prediction_explanations_prediction
        FOREIGN KEY (model_prediction_id)
        REFERENCES model_predictions (model_prediction_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT fk_prediction_explanations_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines (machine_id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT,
    CONSTRAINT ck_prediction_explanations_model_input_hash_format
        CHECK (model_input_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_prediction_explanations_config_hash_format
        CHECK (explanation_config_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_prediction_explanations_identity_not_blank
        CHECK (
            btrim(explainer_name) <> ''
            AND btrim(explainer_version) <> ''
            AND btrim(output_semantics) <> ''
            AND btrim(attribution_semantics) <> ''
        ),
    CONSTRAINT ck_prediction_explanations_model_output
        CHECK (model_output_value >= 0 AND model_output_value <= 1),
    CONSTRAINT ck_prediction_explanations_additivity_error
        CHECK (additivity_error >= 0),
    CONSTRAINT ck_prediction_explanations_feature_contributions_shape
        CHECK (
            jsonb_typeof(feature_contributions) = 'array'
            AND jsonb_array_length(feature_contributions) = 6
            AND feature_contributions @> '[{"feature_name": "Type"}]'::jsonb
            AND feature_contributions @> '[{"feature_name": "Air temperature [K]"}]'::jsonb
            AND feature_contributions @> '[{"feature_name": "Process temperature [K]"}]'::jsonb
            AND feature_contributions @> '[{"feature_name": "Rotational speed [rpm]"}]'::jsonb
            AND feature_contributions @> '[{"feature_name": "Torque [Nm]"}]'::jsonb
            AND feature_contributions @> '[{"feature_name": "Tool wear [min]"}]'::jsonb
        )
);

CREATE UNIQUE INDEX uq_prediction_explanations_stable_identity
    ON prediction_explanations (
        model_prediction_id,
        explainer_name,
        explainer_version,
        explanation_config_hash
    );

CREATE INDEX idx_prediction_explanations_event
    ON prediction_explanations (
        machine_id,
        event_time DESC,
        event_id,
        explanation_config_hash
    );
