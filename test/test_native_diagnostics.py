import json

import pytest

from marvin_policy_server.native_diagnostics import NativeDiagnostics
from marvin_policy_server.native_protocol import Kind, Packet


def test_bounded_records_preserve_identity_and_null_estimate(tmp_path):
    path = tmp_path / 'trace.jsonl'
    recorder = NativeDiagnostics(path, ['joint'], max_records=2)
    packet = Packet(Kind.RESULT, sequence=8, generation=3, trace_id=2**63, values=(1.,))
    recorder.record('result_submitted', packet, sensor_values=(2.,))
    recorder.record('publication_commit', packet)
    recorder.record('result_submitted', packet)
    # No callback writes: only initial metadata exists until close.
    assert len(path.read_text().splitlines()) == 1
    recorder.close()
    recorder.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]['trace_id'] == str(2**63)
    assert rows[1]['estimated_velocity'] is None
    assert rows[1]['effective_velocity_command'] is None
    assert rows[1]['sensor_values'] == [2.]
    assert rows[2]['event'] == 'publication_commit'
    assert rows[-1] == {'event': 'summary', 'records': 2, 'dropped': 1}


def test_refuses_overwrite(tmp_path):
    path = tmp_path / 'existing'
    path.write_text('original')
    with pytest.raises(FileExistsError):
        NativeDiagnostics(path, [])
    assert path.read_text() == 'original'


def test_invalid_limit_creates_no_file(tmp_path):
    path = tmp_path / 'trace'
    with pytest.raises(ValueError):
        NativeDiagnostics(path, [], max_records=0)
    assert not path.exists()


def test_effective_command_is_copied_with_result_identity(tmp_path):
    path = tmp_path / 'trace.jsonl'
    recorder = NativeDiagnostics(path, ['joint'])
    command = [.2, -.1, .3]
    recorder.record('result_submitted', Packet(Kind.RESULT, sequence=9),
                    effective_velocity_command=command)
    command[0] = 99.
    recorder.close()
    row = json.loads(path.read_text().splitlines()[1])
    assert row['sequence'] == 9
    assert row['effective_velocity_command'] == [.2, -.1, .3]
