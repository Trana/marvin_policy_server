"""
Instrument the fixed-base script in a diagnostic layer, never the source USD.

Records script acceptance and target-setter return, not physical
actuation. Diagnostic failures must not suppress original target calls.
"""

SETTER = """dc.set_dof_position_target(
                    db.per_instance_state.dofs[name], float(position)
                )"""

HELPERS = r"""

import time as _target_probe_time
import json as _target_probe_json

def _target_probe_error(db, phase, error):
    # At most one error per phase per activation; never disrupt target delivery.
    try:
        state = db.per_instance_state
        seen = getattr(state, '_target_probe_errors', set())
        if phase not in seen:
            seen.add(phase)
            state._target_probe_errors = seen
            print('SIM_TARGET_PROBE_ERROR ' + phase + ': ' + repr(error), flush=True)
    except Exception:
        pass

def _target_probe_reset(db):
    try:
        state = db.per_instance_state
        state._target_probe_queue = collections.deque()
        state._target_probe_rows = []
        state._target_probe_sequence = 0
        state._target_probe_expected = None
        state._target_probe_errors = set()
    except Exception as error:
        _target_probe_error(db, 'reset', error)

def _target_probe_accept(db, names, positions):
    try:
        state = db.per_instance_state
        if not hasattr(state, '_target_probe_queue'):
            state._target_probe_queue = collections.deque()
            state._target_probe_rows = []
            state._target_probe_sequence = 0
        state._target_probe_sequence += 1
        item = (state._target_probe_sequence, _target_probe_time.monotonic_ns(),
                _target_probe_time.time_ns(), dict(zip(names, positions)))
        state._target_probe_queue.append(item)
        state._target_probe_expected = (state._target_probe_queue.popleft()
            if len(state._target_probe_queue) > DELAY_STEPS else item)
    except Exception as error:
        # Probe failure must not prevent the original queue operation.
        _target_probe_error(db, 'accept', error)
        try:
            db.per_instance_state._target_probe_expected = None
        except Exception:
            pass

def _target_probe_emit(db):
    state = db.per_instance_state
    rows = getattr(state, '_target_probe_rows', [])
    state._target_probe_rows = []
    if rows:
        print('SIM_TARGET_PROBE ' + _target_probe_json.dumps(rows), flush=True)

def _target_probe_set(db, dc, handle, target, name):
    # The authoritative setter runs first; propagate its errors unchanged.
    result = dc.set_dof_position_target(handle, target)
    try:
        if name == 'FL_calf_joint':
            state = db.per_instance_state
            expected = getattr(state, '_target_probe_expected', None)
            if expected is not None:
                sequence, mono, wall, values = expected
                state._target_probe_rows.append(dict(
                    input_sequence=sequence, accepted_monotonic_ns=mono,
                    accepted_wall_ns=wall,
                    setter_return_monotonic_ns=_target_probe_time.monotonic_ns(),
                    setter_return_wall_ns=_target_probe_time.time_ns(),
                    callback_sequence=state._target_probe_sequence,
                    joint=name, target_rad=target,
                    setter_result=(result if isinstance(
                        result, (bool, int, float, str, type(None)))
                                   else repr(result)),
                    shadow_matches=bool(values.get(name) == target)))
                if len(state._target_probe_rows) >= 100:
                    _target_probe_emit(db)
    except Exception as error:
        _target_probe_error(db, 'setter_record', error)
    return result

_target_probe_original_cleanup = globals().get('cleanup')
def cleanup(db):
    try:
        _target_probe_emit(db)
    except Exception as error:
        _target_probe_error(db, 'cleanup', error)
    if _target_probe_original_cleanup is not None:
        return _target_probe_original_cleanup(db)
"""


def instrument(source, joint='FL_calf_joint'):
    allowed = {
        f'{leg}_{part}_joint'
        for leg in ('FL', 'FR', 'RL', 'RR')
        for part in ('hip', 'thigh', 'calf')
    }
    if joint not in allowed:
        raise ValueError('Probe joint must name one of the 12 motor joints')
    append = '        db.per_instance_state.buffer.append((names, positions))'
    reset = '    db.per_instance_state.buffer = collections.deque()'
    for marker in (append, reset, SETTER, 'DELAY_STEPS = 6'):
        if source.count(marker) != 1:
            raise ValueError('Unexpected fixed-base controller; review instrumentation')
    if '_target_probe_' in source:
        raise ValueError('Source is already instrumented')
    code = source.replace(append, '        _target_probe_accept(db, names, positions)\n' + append)
    code = code.replace(reset, reset + '\n    _target_probe_reset(db)')
    code = code.replace(
        SETTER,
        '_target_probe_set(db, dc, db.per_instance_state.dofs[name], float(position), name)',
    )
    code += HELPERS.replace("name == 'FL_calf_joint'", f'name == {joint!r}')
    compile(code, '<fixed-command-probe>', 'exec')
    return code
