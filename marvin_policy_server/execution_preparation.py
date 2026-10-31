"""Validated preparation for the Python policy command gateway."""
import json
import math
import time


class PolicyPreparation:
    def __init__(self, joint_names, clock=time.monotonic):
        self.joint_names, self.clock = tuple(joint_names), clock
        self.generation = None
        self.targets = None
        self.expires = 0.
        self.started = None
        self.sequence = 0

    def prepare(self, generation, encoded, *, released):
        if not released:
            raise ValueError('Policy must be released before preparation')
        state = json.loads(encoded)
        if not generation or len(generation) > 64:
            raise ValueError('Invalid command generation')
        if tuple(state['joint_names']) != self.joint_names or state['position_unit'] != 'rad' or state['model'] != 'marvin':
            raise ValueError('Robot state schema mismatch')
        if not 0 <= self.clock()-float(state['sampled_at']) <= .5:
            raise ValueError('Preparation state expired or clock domain mismatch')
        if not all(0 <= float(state[k]) <= .25 for k in ('joint_age', 'imu_age')):
            raise ValueError('Feedback is stale')
        targets = state['last_targets'] or state['positions']
        if len(targets) != 12 or not all(math.isfinite(v) for v in targets):
            raise ValueError('Invalid initial joint targets')
        self.generation, self.targets = generation, tuple(targets)
        self.expires, self.started, self.sequence = self.clock()+5., None, 0

    def start(self):
        if self.generation is None or self.clock() > self.expires:
            raise ValueError('A fresh preparation is required')
        self.started = self.clock()

    def command(self, targets):
        if self.started is None:
            return None
        t = min(1., max(0., self.clock()-self.started)/2.)
        self.sequence += 1
        return tuple(a+(b-a)*t for a, b in zip(self.targets, targets))
