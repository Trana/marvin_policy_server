"""Read-only input timing diagnostics; never used to accept/reject control data."""
import time
import math


class InputTiming:
    def __init__(self, timing, *, clock=time.monotonic_ns):
        self.timing = timing
        self.clock = clock
        self.receipts = {}
        self.used = {}
        self.previous_inference = None
        self.encoder_snapshot = None
        self.encoder_used = {}

    def received_encoders(self, message, receipts):
        self.encoder_snapshot = (message, dict(receipts))

    def encoder_inference(self, message):
        if self.timing.mode == 'off':
            return
        snapshot = self.encoder_snapshot
        if snapshot is None or snapshot[0] is not message:
            self.timing.count('encoder_snapshot_mismatch')
            return
        now = self.clock() / 1e9
        ages = []
        for name, receipt in snapshot[1].items():
            metric = 'encoder_' + name
            if not math.isfinite(receipt) or receipt <= 0 or receipt > now:
                self.timing.count(metric + '_receipt_invalid')
                continue
            previous = self.encoder_used.get(name)
            if previous is not None and receipt < previous:
                self.timing.count(metric + '_receipt_regressed')
                continue
            age = (now - receipt) * 1000
            self.timing.observe_ms(metric + '_age_at_inference', age)
            if previous == receipt:
                self.timing.count(metric + '_sample_reused')
            self.encoder_used[name] = receipt
            ages.append(age)
        if ages and len(ages) == len(snapshot[1]):
            self.timing.observe_ms('encoder_oldest_age_at_inference', max(ages))
            self.timing.observe_ms('encoder_receipt_spread_at_inference', max(ages) - min(ages))
        else:
            self.timing.count('encoder_inference_incomplete')

    def received(self, topic):
        if self.timing.mode == 'off':
            return
        previous = self.receipts.get(topic)
        self.receipts[topic] = (self.clock(), 1 if previous is None else previous[1] + 1)

    def reset_interval(self):
        """Exclude inactive time; active stalls remain visible in interval metrics."""
        self.previous_inference = None
        self.used.clear()
        self.encoder_used.clear()

    def inference(self, ros_now_ns, messages, *, overridden=()):
        if self.timing.mode == 'off':
            return
        now = self.clock()
        if self.previous_inference is not None:
            self.timing.observe_ms('inference_interval', (now - self.previous_inference) / 1e6)
        self.previous_inference = now
        for topic, message in messages.items():
            if topic in overridden:
                self.timing.count(topic + '_input_overridden')
                continue
            stamp = message.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if stamp_ns <= 0:
                self.timing.count(topic + '_header_age_missing')
            else:
                self.timing.observe_ms(topic + '_header_age', (ros_now_ns - stamp_ns) / 1e6)
            receipt = self.receipts.get(topic)
            if receipt is None:
                self.timing.count(topic + '_receipt_missing')
                continue
            received_at, sequence = receipt
            self.timing.observe_ms(topic + '_callback_age', (now - received_at) / 1e6)
            if self.used.get(topic) == sequence:
                self.timing.count(topic + '_callback_sample_reused')
            self.used[topic] = sequence
