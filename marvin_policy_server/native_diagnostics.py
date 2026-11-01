"""Bounded native inference evidence. No file I/O in the control callback."""
import json
import os
import time


class NativeDiagnostics:
    def __init__(self, path, joint_names, max_records=6000):
        if not 1 <= max_records <= 60000:
            raise ValueError('Diagnostic max_records must be between 1 and 60000')
        self.rows = []
        self.max_records = max_records
        self.dropped = 0
        self.closed = False
        self.error = None
        self._stream = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w')
        self._stream.write(json.dumps(dict(
            schema='native-inference-capture.v1', event='metadata', joint_names=list(joint_names),
            started_unix_ns=time.time_ns(), started_monotonic_ns=time.monotonic_ns(),
            max_records=max_records,
            limits='Buffered until orderly shutdown. Kill/power loss loses buffered rows. '
                   'Result submission is not publication. COMMIT means first publication, not motor receipt.'
        )) + '\n')
        self._stream.flush()

    def record(self, event, packet, *, estimated_velocity=None, sensor_values=None,
               effective_velocity_command=None):
        if self.closed or len(self.rows) >= self.max_records:
            self.dropped += 1
            return
        self.rows.append(dict(
            event=event, received_monotonic_ns=time.monotonic_ns(),
            sequence=packet.sequence, generation=packet.generation, trace_id=str(packet.trace_id),
            requested_ns=packet.requested_ns, ros_ns=packet.ros_ns,
            joint_ns=packet.joint_ns, imu_ns=packet.imu_ns, flags=int(packet.flags),
            values=list(packet.values),
            estimated_velocity=None if estimated_velocity is None else [float(v) for v in estimated_velocity],
            effective_velocity_command=None if effective_velocity_command is None else
            [float(v) for v in effective_velocity_command],
            sensor_values=None if sensor_values is None else list(sensor_values)))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for row in self.rows:
                self._stream.write(json.dumps(row, allow_nan=False) + '\n')
            self._stream.write(json.dumps(dict(event='summary', records=len(self.rows),
                                              dropped=self.dropped)) + '\n')
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            try:
                self._stream.close()
            except OSError as exc:
                self.error = str(exc)
