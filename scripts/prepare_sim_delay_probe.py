#!/usr/bin/env python3
"""Create a separate USD diagnostic layer. Never edit the reference scene."""

import argparse
import hashlib
from pathlib import Path

NODE = '/World/marvin_isaacsim_ros/Graph/ROS_Joint_States/script_node'
SIM_TIME = ('/World/marvin_isaacsim_ros/Graph/ROS_Joint_States/'
            'isaac_read_simulation_time.outputs:simulationTime')
WRAPPER = r"""

# Diagnostic wrapper: original target calculation remains authoritative.
import time as _probe_time
import json as _probe_json
import math as _probe_math
_original_setup = setup
_original_compute = compute

def setup(db):
    _original_setup(db)
    db.internal_state.probe_queue = collections.deque()
    db.internal_state.probe_rows = []
    db.internal_state.probe_count = 0

def _probe_emit(db):
    rows = db.internal_state.probe_rows
    if not rows:
        return
    ages = [r[1] for r in rows if not r[2]]
    sim_ages = [r[5] for r in rows if not r[2] and r[5] is not None]
    span = rows[-1][0] - rows[0][0]
    sim_span = rows[-1][4] - rows[0][4] if all(r[4] is not None for r in rows) else None
    print('SIM_DELAY_PROBE ' + _probe_json.dumps(dict(
        callbacks_total=db.internal_state.probe_count,
        window_count=len(rows),
        window_start_monotonic_s=rows[0][0],
        window_end_monotonic_s=rows[-1][0],
        window_start_sim_s=rows[0][4],
        window_end_sim_s=rows[-1][4],
        oldest_buffered_input_sim_s=min(
            (r[4]-r[5] for r in rows if not r[2] and r[5] is not None), default=None),
        callback_hz=(len(rows)-1)/span if span > 0 else None,
        buffered_count=len(ages),
        passthrough_count=sum(r[2] for r in rows),
        output_mismatches=sum(not r[3] for r in rows),
        buffer_age_mean_ms=1000*sum(ages)/len(ages) if ages else None,
        buffer_age_min_ms=1000*min(ages) if ages else None,
        buffer_age_max_ms=1000*max(ages) if ages else None,
        sim_age_samples=len(sim_ages),
        invalid_sim_age_samples=len(ages)-len(sim_ages),
        buffer_sim_age_mean_ms=1000*sum(sim_ages)/len(sim_ages) if sim_ages else None,
        buffer_sim_age_min_ms=1000*min(sim_ages) if sim_ages else None,
        buffer_sim_age_max_ms=1000*max(sim_ages) if sim_ages else None,
        callback_sim_hz=(len(rows)-1)/sim_span if sim_span is not None and sim_span > 0 else None,
        real_time_factor=(sim_span/span
                          if sim_span is not None and sim_span >= 0 and span > 0 else None),
        clock='monotonic_at_script_entry_and_graph_simulation_time',
    )), flush=True)
    rows.clear()

def compute(db):
    current = db.inputs.positionCommand
    if current is None or len(current) == 0:
        return _original_compute(db)
    now = _probe_time.monotonic()
    sim_now = getattr(db.inputs, 'probeSimulationTime', None)
    if sim_now is not None:
        sim_now = float(sim_now)
        if not _probe_math.isfinite(sim_now) or sim_now < 0:
            sim_now = None
    values = list(current)
    queue = db.internal_state.probe_queue
    queue.append((now, values, sim_now))
    passthrough = len(queue) <= 6
    expected_time, expected, expected_sim_time = (
        (now, values, sim_now) if passthrough else queue.popleft())
    sim_age = (sim_now - expected_sim_time
               if sim_now is not None and expected_sim_time is not None
               and sim_now >= expected_sim_time else None)
    result = _original_compute(db)
    actual = list(db.outputs.delayed_position_command)
    matches = len(actual) == len(expected) and all(a == b for a,b in zip(actual, expected))
    db.internal_state.probe_count += 1
    db.internal_state.probe_rows.append(
        (now, now-expected_time, passthrough, matches, sim_now, sim_age))
    if len(db.internal_state.probe_rows) >= 30:
        _probe_emit(db)
    return result

def cleanup(db):
    _probe_emit(db)
"""


def instrument(source):
    marker = '        print(len(db.internal_state.buffer))'
    if source.count(marker) != 1 or source.count('delay_steps = 6') != 1:
        raise ValueError('Unexpected source buffer; review before instrumenting')
    # Replace per-callback debug output with a bounded diagnostic summary.
    return (
        source.replace(
            marker, '        # Per-callback debug print suppressed in diagnostic layer.'
        )
        + WRAPPER
    )


def prepare(scene, output):
    from pxr import Sdf
    from pxr import Usd

    scene = scene.resolve(strict=True)
    if output.exists():
        raise ValueError('Diagnostic output already exists; do not overwrite evidence')
    original = Usd.Stage.Open(str(scene))
    source = original.GetPrimAtPath(NODE).GetAttribute('inputs:script').Get()
    if not original.GetAttributeAtPath(SIM_TIME):
        raise ValueError('Scene has no expected simulation clock output')
    code = instrument(source)
    layer = Sdf.Layer.CreateNew(str(output))
    # Stage metadata is not inherited from a sublayer. Without this copy,
    # USD falls back to centimeters/Y-up even for a meters/Z-up source.
    for key in original.GetRootLayer().pseudoRoot.ListInfoKeys():
        if key not in ('subLayers', 'subLayerOffsets', 'customLayerData'):
            layer.pseudoRoot.SetInfo(key, original.GetRootLayer().pseudoRoot.GetInfo(key))
    layer.subLayerPaths = [str(scene)]
    layer.customLayerData = dict(
        original.GetRootLayer().customLayerData,
        **{
            'reference_scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
            'reference_script_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'diagnostic': 'buffer timing only; not the reference scene',
        },
    )
    stage = Usd.Stage.Open(layer)
    stage.OverridePrim(NODE).CreateAttribute('inputs:script', Sdf.ValueTypeNames.String).Set(code)
    stage.OverridePrim(NODE).CreateAttribute(
        'inputs:probeSimulationTime', Sdf.ValueTypeNames.Double
    ).SetConnections([SIM_TIME])
    layer.Save()
    return str(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(prepare(args.scene, args.output))
