"""Simulation-only inference availability. Physical inference never uses this gate."""
import json
import random
from collections import deque
import numpy as np


class SimActionRelease:
    def __init__(self, profile, initial, seed=42):
        if profile.get('revision') != 'marvin-timing-v1' or profile.get('physics_dt') != .005:
            raise ValueError('Unsupported simulation timing profile')
        bounds = profile['inference_delay_ms']
        if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= 20 or any(v % 5 for v in bounds):
            raise ValueError('Simulation inference delay requires 5 ms increments within 0–20 ms')
        self.profile = json.dumps(profile, sort_keys=True)
        self.bounds = tuple(int(v / 5) for v in bounds)
        self.rng = random.Random(seed + 272)
        self.last_raw = np.zeros_like(initial)
        self.last_target = np.array(initial, copy=True)
        self.pending = None
        self.session = None
        self.identity_ns = None
        self.last_clock = None
        self.permission = None
        self.error = 'Waiting for the matching simulation timing profile'
        self.last_release = None
        self.observation_times = deque(maxlen=101)
        self.last_cadence_warning = None

    def identity(self, payload):
        data = json.loads(payload)
        if json.dumps(data['profile'], sort_keys=True) != self.profile:
            self.error = 'Scene and policy timing profiles do not match'
            self.pending = None
            return False
        if data.get('imu_acceleration_convention') != 'specific_force':
            self.error = 'Timed scene must declare specific-force IMU acceleration; export the scene again'
            self.pending = None
            return False
        session, stamp = data['session_id'], data['sim_time_ns']
        if not isinstance(session, str) or not session or not isinstance(stamp, int) or stamp < 0:
            raise ValueError('Invalid simulation timing identity')
        changed = session != self.session
        if changed:
            self.pending = None
            self.last_clock = None
        self.session, self.identity_ns, self.error = session, stamp, None
        return changed

    def readiness(self, now):
        if self.error:
            return False, self.error
        if self.identity_ns is None or not -100_000_000 <= now - self.identity_ns <= 300_000_000:
            return False, 'Simulation timing identity is stale'
        return True, ''

    def clear(self):
        self.pending = None

    def reset(self, initial):
        self.clear()
        self.observation_times.clear()
        self.last_cadence_warning = None
        self.last_raw = np.zeros_like(initial)
        self.last_target = np.array(initial, copy=True)

    def observe(self, now, permission):
        changed = ((self.last_clock is not None and now < self.last_clock) or
                   (self.permission is not None and permission != self.permission))
        self.last_clock, self.permission = now, permission
        if changed:
            self.clear()
        return changed

    def submit(self, raw, target, observation_ns, completion_ns, trace_id):
        if self.pending is not None:
            raise RuntimeError('A simulation inference result is still pending')
        due = observation_ns + self.rng.randint(*self.bounds) * 5_000_000
        self.pending = (np.array(raw, copy=True), np.array(target, copy=True),
                        observation_ns, completion_ns, due, trace_id)

    def take(self, now):
        if self.pending is None or now < max(self.pending[3], self.pending[4]):
            return None
        raw, target, observation, completion, due, trace_id = self.pending
        self.pending = None
        self.last_raw, self.last_target = raw, target
        self.observation_times.append(observation)
        self.last_release = dict(observation_ns=observation, completion_ns=completion,
                                 due_ns=due, release_ns=now, trace_id=trace_id)
        return raw, target, trace_id

    def cadence_warning(self, now):
        """Measure simulated-time cadence, separately from per-result delay."""
        stamps = self.observation_times
        if len(stamps) < 101 or stamps[-1] <= stamps[0]:
            return None
        hz = (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0])
        if hz >= 48 or (self.last_cadence_warning is not None and
                        now - self.last_cadence_warning < 5_000_000_000):
            return None
        self.last_cadence_warning = now
        return f'Simulation policy cadence is {hz:.1f} Hz; expected 50 Hz per simulated second'


def simulation_gate(profile, initial, *, use_sim_time, native_io):
    if not profile or not use_sim_time:
        return None
    if native_io:
        raise ValueError('Timed simulation requires the Python policy gateway backend')
    return SimActionRelease(profile, initial)
