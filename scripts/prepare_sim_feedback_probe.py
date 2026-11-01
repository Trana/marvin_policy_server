#!/usr/bin/env python3
"""Prepare a separate simulation layer with read-only articulation diagnostics."""
import argparse
from pathlib import Path

from prepare_sim_delay_probe import NODE, prepare


FEEDBACK_WRAPPER = r'''

# Inspect the articulation API without changing targets or physics settings.
_feedback_original_compute = compute
_feedback_articulation = None
_feedback_next_time = 0.0

def compute(db):
    global _feedback_articulation, _feedback_next_time
    result = _feedback_original_compute(db)
    now = float(db.inputs.probeSimulationTime)
    if now < _feedback_next_time:
        return result
    _feedback_next_time = now + 0.1
    try:
        if _feedback_articulation is None:
            from isaacsim.core.prims import SingleArticulation
            candidate = SingleArticulation(
                '/World/marvin_isaacsim_ros', reset_xform_properties=False)
            candidate.initialize()
            _feedback_articulation = candidate
        print('SIM_FEEDBACK_PROBE ' + _probe_json.dumps(dict(
            simulation_time=now,
            names=list(_feedback_articulation.dof_names),
            position=_feedback_articulation.get_joint_positions().tolist(),
            velocity=_feedback_articulation.get_joint_velocities().tolist(),
            source='SingleArticulation getters; command graph callback',
        )), flush=True)
    except Exception as error:
        print('SIM_FEEDBACK_PROBE_ERROR ' + str(error), flush=True)
    return result
'''


def prepare_feedback(scene, output):
    from pxr import Usd
    result = prepare(scene, output)
    stage = Usd.Stage.Open(result)
    attribute = stage.GetPrimAtPath(NODE).GetAttribute('inputs:script')
    code = attribute.Get() + FEEDBACK_WRAPPER
    compile(code, '<simulation-feedback-probe>', 'exec')
    attribute.Set(code)
    layer = stage.GetRootLayer()
    metadata = dict(layer.customLayerData)
    metadata['diagnostic'] = 'read-only articulation feedback plus delay timing; not reference scene'
    layer.customLayerData = metadata
    layer.Save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(prepare_feedback(args.scene, args.output))
