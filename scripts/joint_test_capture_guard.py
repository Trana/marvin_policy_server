"""No-ROS recording gate for guarded joint tests. Never enables hardware."""
import math


class CaptureGuard:
    """Authorize renewal only after recent samples were persisted successfully.

    The caller must flush its capture before calling committed(). A fault latches
    for this capture. This gate cannot verify a caller's filesystem operations.
    """

    def __init__(self, instance, activation, joints, max_age=.15):
        if not instance or activation <= 0 or not joints or len(set(joints)) != len(joints):
            raise ValueError('Invalid capture identity')
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('Invalid freshness limit')
        self.identity = (instance, activation)
        self.joints = tuple(joints)
        self.max_age = max_age
        self.last_sequence = None
        self.positions = None
        self.feedback_positions = None
        self.last_source = None
        self.last_target = self.last_feedback = self.last_commit = None
        self.target_count = self.feedback_count = 0
        self.missing = 0
        self.fault = None

    def reject(self, reason):
        self.fault = self.fault or reason
        return False

    def target(self, sample, now):
        if (sample.controller_instance_id, sample.activation_generation) != self.identity:
            return self.reject('TARGET_IDENTITY_CHANGED')
        if tuple(sample.joint_names) != self.joints or len(sample.positions) != len(self.joints):
            return self.reject('TARGET_JOINTS_CHANGED')
        source = sample.sampled_at_monotonic_s
        if not all(math.isfinite(v) for v in [now, source, *sample.positions]):
            return self.reject('INVALID_TARGET')
        if not 0 <= now-source <= self.max_age:
            return self.reject('STALE_TARGET')
        if self.last_sequence is not None:
            gap = sample.update_sequence-self.last_sequence-1
            if gap != 0:
                self.missing += max(0, gap)
                return self.reject('TARGET_SEQUENCE_GAP')
            if source <= self.last_source:
                return self.reject('TARGET_CLOCK_REGRESSION')
        if sample.update_sequence <= 0:
            return self.reject('INVALID_SEQUENCE')
        self.last_sequence, self.last_source = sample.update_sequence, source
        self.positions = tuple(sample.positions)
        self.last_target = now
        self.target_count += 1
        return not self.fault

    def feedback(self, names, positions, now):
        if len(names) != len(set(names)) or len(names) != len(positions):
            return self.reject('INVALID_FEEDBACK')
        values = dict(zip(names, positions))
        if not math.isfinite(now) or any(j not in values or not math.isfinite(values[j]) for j in self.joints):
            return self.reject('INVALID_FEEDBACK')
        self.last_feedback = now
        self.feedback_positions = tuple(values[j] for j in self.joints)
        self.feedback_count += 1
        return not self.fault

    def committed(self, now):
        if not math.isfinite(now):
            return self.reject('INVALID_COMMIT_TIME')
        self.last_commit = now
        return not self.fault

    def live_ready(self, now):
        if self.fault or self.target_count < 2 or self.feedback_count < 2 or not math.isfinite(now):
            return False
        times = (self.last_source, self.last_target, self.last_feedback)
        return all(t is not None and 0 <= now-t <= self.max_age for t in times)

    def ready(self, now):
        return (self.live_ready(now) and self.last_commit is not None and
                0 <= now-self.last_commit <= self.max_age)
