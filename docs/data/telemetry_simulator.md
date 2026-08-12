# Synthetic Industrial Telemetry Simulator

## Purpose
The simulator creates deterministic synthetic industrial telemetry events for local data-engineering and streaming preparation. It now provides complete validated JSON events for the local Kafka telemetry producer and downstream Spark Bronze/Silver processing, while Gold data layers, model-inference integration, anomaly detection, and dashboard work remain planned.

## Synthetic Data Disclaimer
The generated telemetry is synthetic generated telemetry. It is not real industrial telemetry, not manufacturer telemetry, not UCI AI4I data, and not a replay of the AI4I CSV. The simulator is realistic enough for software architecture demonstrations, but it is not a physically validated equipment model.

## Relationship to Operational Fleet
The simulator uses machine codes from the same deterministic identifier range as the fictional PostgreSQL development fleet: `MCH-0001` through `MCH-0100`. This phase does not query PostgreSQL and does not write telemetry into PostgreSQL. A future integration phase may validate simulator machine IDs against the operational fleet table.

## Telemetry Event Contract
The tracked contract is `services/simulator/telemetry_contract.json` with schema version `1.0`. Each JSONL event contains exactly these fields:

- `schema_version`
- `event_id`
- `machine_code`
- `sequence_number`
- `event_time`
- `source`
- `product_quality_type`
- `air_temperature_k`
- `process_temperature_k`
- `rotational_speed_rpm`
- `torque_nm`
- `tool_wear_min`
- `vibration_mm_s`
- `pressure_bar`

Telemetry events are observations only. They do not include `Machine failure`, failure-mode flags, prediction outputs, SHAP values, or anomaly labels.

## Machine Identity
Machine codes use the `MCH-XXXX` format and are valid only from `MCH-0001` through `MCH-0100`. Batch simulation currently selects machines from `MCH-0001` upward.

## Sensor Fields
The simulator emits five AI4I-compatible synthetic measurements and two additional telemetry sensors. Broad guardrail bounds keep values finite and consistent:

- `air_temperature_k`: 294 to 306 K.
- `process_temperature_k`: 304 to 315 K, above air temperature by simulator invariant.
- `rotational_speed_rpm`: 1000 to 3000 RPM.
- `torque_nm`: 0 to 80 Nm.
- `tool_wear_min`: 0 to 300 minutes and non-decreasing per machine.
- `vibration_mm_s`: 0 to 15 mm/s.
- `pressure_bar`: 1 to 12 bar.

These bounds are simulator guardrails, not real equipment specifications.

## AI4I-Compatible Fields
A future adapter may map simulator fields into the existing AI4I model input contract:

- `product_quality_type` -> `Type`
- `air_temperature_k` -> `Air temperature [K]`
- `process_temperature_k` -> `Process temperature [K]`
- `rotational_speed_rpm` -> `Rotational speed [rpm]`
- `torque_nm` -> `Torque [Nm]`
- `tool_wear_min` -> `Tool wear [min]`

This phase does not call the AI4I predictor and does not perform streaming inference.

## AI4I Type vs Operational Machine Type
`product_quality_type` is a simulator-only synthetic event-level field with values `L`, `M`, and `H`. The same `machine_code` may produce events with different `product_quality_type` values over time. It is not the operational PostgreSQL `machine_type`, which contains generic equipment categories such as `excavator`, `wheel_loader`, and `crawler_crane`. Do not map operational `machine_type` to AI4I `Type`.

## Additional Sensors
`vibration_mm_s` and `pressure_bar` are intentionally outside the frozen AI4I classifier contract. They are included for future streaming analytics, fleet monitoring, and anomaly detection. They must not be forced into the existing packaged AI4I model.

## Deterministic Generation
The default seed is `42`, the default start time is `2026-01-01T00:00:00Z`, and the default interval is five seconds. Machine baselines are derived from the simulation seed plus machine code, and event IDs use deterministic UUID5 identity.

## Temporal Evolution
Each machine keeps local state across the simulation. Sensor values evolve with bounded deterministic variation, so consecutive observations from the same machine are related. This temporal behavior supports data-pipeline demonstrations without claiming physical fidelity.

## Event Ordering
Batch output is ordered by timestamp, then `machine_code`. Sequence numbers are per-machine and start at `1`.

## Reproducibility
The same seed, machine count, events per machine, start time, and interval produce byte-identical JSONL output. The canonical sample summary stores the SHA-256 hash of `data/sample/telemetry_events.jsonl`.

## Sample Dataset
Generate the tracked canonical sample with:

```powershell
.\.venv\Scripts\python.exe scripts/generate_telemetry_sample.py
```

Validate it with:

```powershell
.\.venv\Scripts\python.exe scripts/check_telemetry_simulator.py
```

The canonical sample contains 100 events: 10 machines and 10 timestamps per machine.

## Limitations
This simulator is a local portfolio data source. It does not represent a specific manufacturer, fleet, site, machine model, physics system, maintenance policy, failure process, or production telemetry feed.

## Implemented Kafka Integration
The deterministic simulator is now wrapped by `scripts/produce_telemetry_kafka.py` for local Kafka delivery. Kafka payloads preserve the exact telemetry schema version `1.0`; message keys are UTF-8 encoded `machine_code` values. Spark Bronze and Silver processing are implemented downstream. Gold processing, model inference on streaming data, anomaly detection, and PostgreSQL telemetry writes are not implemented here.
