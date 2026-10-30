# Read-only simulation asset inspection

Diagnostic generator correction (2026-10-11): root-stage metadata must be copied,
not assumed to inherit from the reference sublayer. Old `delay-probe.pPbPlA`
wrappers used centimeters/Y-up/24 time codes per second instead of meters/Z-up/60.
Their behavior/performance is not a valid original-scene baseline. Regenerate into
a new path with the corrected tool. Preserve old captures with this qualification.
Run `test/test_sim_probe_stage.py` using Isaac's Python to test real USD metadata
preservation. Passing it does not replace a live scene comparison.

New probe windows include start/end simulation and monotonic times, plus the
oldest buffered input's simulation time. For a steady-state phase, include only
windows whose output times and buffered input times are fully inside that phase.
Do not map monotonic time across machines or combine simulation clock resets.
Missing simulation timestamps remain null. Existing generated scenes embed the
old wrapper: generate a new scene to obtain these fields. Old logs cannot be
retroactively assigned exact phase boundaries from window counts alone.

Run `scripts/inspect_sim_asset.py ASSET.usda` with Python that provides the existing
USD `pxr` package. The script does not start Isaac physics, ROS or robot services.
It prints JSON to standard output. It does not modify the asset.

On the current workstation, from `/home/trana/Development`:

```bash
LD_LIBRARY_PATH=/home/trana/Development/isaac/isaac-sim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311/bin \
PYTHONPATH=/home/trana/Development/isaac/isaac-sim/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311 \
isaac/isaac-sim/python.sh \
ros2/robotdog_ws/src/marvin_policy_server/scripts/inspect_sim_asset.py \
ros2/robotdog_ws/src/marvin/description/models/usd/marvin/marvin_isaacsim_imu_backlash.usda
```

The versioned USD package path can change after an Isaac upgrade. No new dependency
was installed for this tool. Run `test/test_sim_asset_inspection.py` with the same
Python environment to test mass units, missing masses and hash changes.

The report identifies current default-variant composition. It lists used file
layers with SHA-256 hashes, rigid bodies, enabled flags, explicit masses and units.
Review USD warnings for unresolved references. Do not edit source assets while
the report runs. Missing explicit mass does not mean zero physical mass: PhysX
can infer mass from density and collision geometry. The sum includes disabled
bodies, so inspect their flags before interpreting a total.

This is not a full training snapshot. The report excludes external texture/mesh
files, runtime randomization and importer overrides. Hashes detect differences
but do not preserve old bytes. Archive source assets separately for reproducibility.

October 11 check: backlash asset had 25 enabled rigid bodies, no missing explicit
masses, five file layers and a 17.2 kg sum. Both USD tests passed. A Python
environment without USD skips those tests explicitly rather than claiming a pass.

## Separate delay diagnostic scene

`scripts/prepare_sim_delay_probe.py REFERENCE_SCENE NEW_OUTPUT.usda` uses the same
USD Python environment. The output directory must exist. Existing output is refused.
The new layer references the original scene and overrides only the delay script.
It records reference scene/script hashes in layer metadata. Referenced files remain
external dependencies, not immutable archived copies.

The wrapper calls the original target calculation and checks its output against a
parallel timestamp queue. It replaces the existing per-callback buffer-length print
with a summary every 30 nonempty callbacks. `SIM_DELAY_PROBE` reports monotonic
buffer age, callback rate, startup pass-through and output mismatches. These are
wall-time measurements, not simulation-clock measurements. Stop the experiment
if any output mismatch occurs. The diagnostic itself adds overhead.

Use this scene only for an isolated simulation test. Do not select it as the normal
UI scene. Restore the reference scene selection after testing. The tool does not
start simulation, publish commands or control the physical robot.

The October 11 probe logged 1020 checked callbacks without mismatches. Six startup
commands passed through. At approximately 49 incoming callbacks/s, mean buffered
age was 120-123 ms per complete window. Thus six callback entries do not mean six
5 ms physics steps. Partial final windows may be absent if cleanup does not run.
See the wiki alignment workflow for rates, provenance and measurement limits.

The current generator additionally connects `inputs:probeSimulationTime` to the
scene's `IsaacReadSimulationTime` output. New summaries report simulation-clock
buffer ages, callback frequency and real-time factor separately. Missing or invalid
simulation ages are counted and excluded, not converted to zero. This leaves the
original target calculation unchanged. The earlier wall-time-only logs retain their
original meaning.

The follow-up live check measured exactly 30 ms of simulation time in steady
high-rate windows, with approximately 0.5 real-time factor. Do not interpret the
earlier 60 ms wall-time measurement as a 60 ms simulated actuator delay. At lower
delivery rates the simulated buffer delay increased. Normal policy runtime delivery
rate still requires validation.
