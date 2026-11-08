"""Artifact-derived inference timing, shared by runtime and deployment consumers."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math


def seconds_ns(value):
    if isinstance(value, bool):
        raise ValueError('Physics dt must be positive integer nanoseconds')
    try:
        ns = Decimal(str(value)) * 1_000_000_000
        if not ns.is_finite() or ns <= 0 or ns != ns.to_integral_value():
            raise ValueError('Physics dt must be positive integer nanoseconds')
        return int(ns)
    except InvalidOperation as exc:
        raise ValueError('Invalid physics dt') from exc


def positive_ticks(values):
    if (not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 1024 or
            any(type(v) is not int or not 0 < v <= 1_000_000 for v in values)):
        raise ValueError('Cadence requires 1–1024 positive integer intervals')
    return tuple(values)


@dataclass(frozen=True)
class PolicyCadence:
    physics_dt_ns: int
    pattern_ticks: tuple[int, ...]
    source: str

    @classmethod
    def resolve(cls, env, training=None):
        dt = seconds_ns(env.get('sim', {}).get('dt'))
        if 'policy_interval_ticks' in env:
            ticks, source = positive_ticks(env['policy_interval_ticks']), 'policy_interval_ticks'
        else:
            ticks, source = positive_ticks([env.get('decimation')]), 'decimation'
        result = cls(dt, ticks, source)
        profile = env.get('actions', {}).get('joint_pos', {}).get('deployment_timing')
        if profile and seconds_ns(profile.get('physics_dt')) != dt:
            raise ValueError('Deployment timing physics dt differs from the environment')
        if training is not None:
            declared = training.get('policy_cadence')
            if declared is not None:
                if (declared.get('schema') != 'marvin.policy-cadence.v1' or
                        seconds_ns(declared.get('physics_dt_s')) != dt or
                        positive_ticks(declared.get('pattern_ticks')) != ticks):
                    raise ValueError('Export cadence differs from the environment')
                result._check_rate(declared.get('mean_policy_hz'))
            if 'policy_rate_hz' in training:
                result._check_rate(training['policy_rate_hz'])
        return result

    def _check_rate(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(value, self.mean_hz, rel_tol=1e-9):
            raise ValueError('Export policy rate differs from the environment')

    @property
    def periods_ns(self):
        return tuple(self.physics_dt_ns * tick for tick in self.pattern_ticks)

    @property
    def mean_hz(self):
        return 1e9 * len(self.pattern_ticks) / sum(self.periods_ns)

    def validate_backend(self, callback_ns, *, native=False, delay_max_ns=0):
        if type(callback_ns) is not int or callback_ns <= 0 or any(p % callback_ns for p in self.periods_ns):
            raise ValueError('Policy cadence cannot be represented by the runtime callback period')
        if native and any(p != 20_000_000 for p in self.periods_ns):
            raise ValueError('Native policy I/O supports only 20 ms inference intervals; use the Python backend')
        if delay_max_ns > min(self.periods_ns):
            raise ValueError('Inference delay exceeds the shortest policy interval')

    def projection(self):
        canonical = dict(physics_dt_ns=self.physics_dt_ns, pattern_ticks=list(self.pattern_ticks))
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return dict(**canonical, periods_ns=list(self.periods_ns), mean_hz=self.mean_hz, digest=digest)
