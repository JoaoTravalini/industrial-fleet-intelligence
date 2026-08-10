-- Deterministic fictional development fleet seed.
-- This data is synthetic portfolio development data only. It is not real industrial,
-- manufacturer, or proprietary equipment data, and it intentionally inserts no
-- telemetry, maintenance, prediction, anomaly, alert, or health-summary records.

WITH seed_rows AS (
    SELECT
        machine_number,
        'MCH-' || lpad(machine_number::text, 4, '0') AS machine_identifier,
        CASE ((machine_number - 1) % 8)
            WHEN 0 THEN 'excavator'
            WHEN 1 THEN 'wheel_loader'
            WHEN 2 THEN 'crawler_crane'
            WHEN 3 THEN 'mobile_crane'
            WHEN 4 THEN 'mining_truck'
            WHEN 5 THEN 'bulldozer'
            WHEN 6 THEN 'industrial_pump'
            ELSE 'generator'
        END AS machine_type,
        CASE ((machine_number - 1) % 8)
            WHEN 0 THEN 'EX-Series'
            WHEN 1 THEN 'WL-Series'
            WHEN 2 THEN 'CC-Series'
            WHEN 3 THEN 'MC-Series'
            WHEN 4 THEN 'MT-Series'
            WHEN 5 THEN 'BD-Series'
            WHEN 6 THEN 'IP-Series'
            ELSE 'GN-Series'
        END AS model_family,
        DATE '2016-01-15' + ((machine_number - 1) * 37) AS commissioned_on,
        CASE
            WHEN machine_number <= 85 THEN 'active'
            WHEN machine_number <= 95 THEN 'maintenance'
            ELSE 'inactive'
        END AS operational_status,
        TIMESTAMPTZ '2026-01-01 00:00:00+00' AS created_at,
        TIMESTAMPTZ '2026-01-01 00:00:00+00' AS updated_at
    FROM generate_series(1, 100) AS series(machine_number)
)
INSERT INTO machines (
    machine_identifier,
    machine_type,
    model_family,
    commissioned_on,
    operational_status,
    created_at,
    updated_at
)
SELECT
    seed_rows.machine_identifier,
    seed_rows.machine_type,
    seed_rows.model_family,
    seed_rows.commissioned_on,
    seed_rows.operational_status,
    seed_rows.created_at,
    seed_rows.updated_at
FROM seed_rows
WHERE NOT EXISTS (
    SELECT 1
    FROM machines
    WHERE machines.machine_identifier = seed_rows.machine_identifier
)
ON CONFLICT (machine_identifier) DO NOTHING;
