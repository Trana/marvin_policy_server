#!/usr/bin/env python3
"""Synthetic recorder microbenchmark. No ROS, policy loading or hardware access."""

import json
from pathlib import Path
import resource
import statistics
import tempfile
import time

from marvin_policy_server.native_diagnostics import NativeDiagnostics
from marvin_policy_server.native_protocol import Kind
from marvin_policy_server.native_protocol import Packet
import numpy as np


def main():
    directory = Path(tempfile.mkdtemp(prefix='native-diagnostics-benchmark-'))
    report = []
    estimate = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    effective_command = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    sensors = tuple(float(i) / 100 for i in range(34))
    for trial in range(3):
        path = directory / f'trial-{trial}.jsonl'
        recorder = NativeDiagnostics(path, [f'joint_{i}' for i in range(12)])
        durations = []
        for sequence in range(3000):
            result = Packet(
                Kind.RESULT,
                sequence=sequence,
                generation=1,
                trace_id=sequence + 1,
                values=(0.1,) * 24,
            )
            commit = Packet(
                Kind.COMMIT,
                sequence=sequence,
                generation=1,
                trace_id=sequence + 1,
                values=(0.2,) * 24,
            )
            started = time.perf_counter_ns()
            recorder.record(
                'result_submitted',
                result,
                estimated_velocity=estimate,
                sensor_values=sensors,
                effective_velocity_command=effective_command,
            )
            recorder.record('publication_commit', commit)
            durations.append((time.perf_counter_ns() - started) / 1000)
        recorder.record('overflow', result)
        started = time.perf_counter_ns()
        recorder.close()
        export_ms = (time.perf_counter_ns() - started) / 1e6
        assert recorder.error is None, recorder.error
        # Verify actual export, not just successful return from close().
        with path.open() as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == 6002
        assert rows[-1] == {'event': 'summary', 'records': 6000, 'dropped': 1}
        for result_row, commit_row in zip(rows[1:-1:2], rows[2:-1:2]):
            assert result_row['trace_id'] == commit_row['trace_id']
            assert result_row['event'] == 'result_submitted'
            assert commit_row['event'] == 'publication_commit'
            assert result_row['effective_velocity_command'] == effective_command.tolist()
            assert commit_row['effective_velocity_command'] is None
        report.append({
            'trial': trial,
            'pairs': 3000,
            'pair_mean_us': statistics.mean(durations),
            'pair_p95_us': sorted(durations)[2849],
            'pair_max_us': max(durations),
            'export_ms': export_ms,
            'bytes': path.stat().st_size,
        })
    print(
        json.dumps(
            {
                'trials': report,
                'output': str(directory),
                'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'limits': (
                    'Synthetic back-to-back calls, not loaded runtime timing. '
                    'RSS includes Python, NumPy and export verification objects. '
                    'Packet identities are synthetic, not publication evidence.'
                ),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
