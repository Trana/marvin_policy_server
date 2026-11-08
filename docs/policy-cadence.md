# Per-policy inference cadence

The policy runtime reads timing from the exact environment bytes loaded for the
policy. `sim.dt` defines the physics interval. An explicit `policy_interval_ticks`
pattern takes precedence over `decimation`. Legacy policies use an explicit
`decimation` as a one-element pattern. Missing or invalid timing fails startup.

For example, `sim.dt: 0.005` and `policy_interval_ticks: [4, 5]` select alternating
20/25 ms intervals (44.44 Hz). `decimation: 6` without a pattern selects 30 ms.
The runtime does not infer timing from the policy filename or use a default rate.
Existing export `training.policy_cadence` and `policy_rate_hz` fields must agree
with the environment when present. Deployment physics timing must agree too.

`policy_cadence.py` owns parsing, validation and the canonical projection.
`InferenceSchedule` owns Python deadlines and skips expired slots without bursts.
Clock rewind and existing activation/input-invalid transitions reset its phase.
Async admission advances the schedule once. Action release keeps its independent
modelled delay. Runtime callback intervals must represent every policy interval.
The single pending release must fit within the shortest interval.

The native C++ frontend still owns a fixed 20 ms schedule. It rejects a policy
that requires other intervals before starting its helper. This change does not
retime native hardware execution. Policy reload retains the exact environment
hash and semantic metadata checks. A different cadence requires a new runtime.

## Existing state API

`/policy/state` has an additive optional `timing` object, schema
`marvin.policy-timing.v1`. It contains:

- `cadence`: physics nanoseconds, interval ticks, interval nanoseconds, mean Hz,
  and a SHA-256 digest of the canonical physics/tick pair.
- `environment_sha256` and `source`: artifact identity and the selected config key.
- `scheduler`: `python-cyclic` or `native-fixed`.
- `observed`: the most recent 101 admitted inference timestamps summarized as
  sample count, mean Hz, last observation timestamp and skipped deadline slots.

The digest describes cadence, not policy identity. Observed statistics reset
with the schedule. Zero samples or null rate do not establish active inference.
The mean rate alone does not prove the interval pattern or action-release delay.
Use simulation trace evidence to check those details. Older services omit timing.
The Rust API preserves this typed field. robotctl requires matching timing before
operator activation and checks the declaration again during the session.

Decision and simulation evidence: Trana/marvin_wiki#50. This is a development
branch change, not a hardware deployment or readiness approval.
