import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'analysis', Path(__file__).parents[1] / 'scripts/analyze_native_capture.py'
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def commit():
    return {
        'event': 'publication_commit',
        'trace_id': '42',
        'values': [0.0] * 12 + [1.0] * 12,
        'ros_ns': 10_000_000,
        'joint_ns': 5_000_000,
        'imu_ns': 8_000_000,
    }


def test_matching_command_values_and_stamps():
    result = analysis.summarize(
        [commit(), {'event': 'summary'}], {'42': [(20_000_000, [1.0] * 12)]}
    )
    assert result['matched'] == 1
    assert result['output_mismatches'] == 0
    assert result['timing']['joint_age_at_publication_ms']['mean'] == 15
    assert result['complete_export']


def test_missing_ambiguous_and_nonfinite_are_not_matches():
    assert analysis.summarize([commit()], {})['unmatched'] == 1
    assert analysis.summarize([commit(), commit()], {'42': [(20, [])]})['ambiguous'] == 2
    result = analysis.summarize([commit()], {'42': [(1, [float('nan')] * 12)]})
    assert result['output_mismatches'] == 1
    assert result['invalid_time_pairs'] == 3
    assert not result['complete_export']


def test_percentiles_and_empty_distribution():
    assert analysis.distribution([])['p95'] is None
    assert analysis.distribution([7])['p99'] == 7
    result = analysis.distribution([0, 10, 20])
    assert result['p50'] == 10
    assert result['p95'] == 19
    assert result['p99'] == 19.8


def test_summary_presence_is_not_lossless_evidence():
    result = analysis.summarize([commit(), {'event': 'summary', 'records': 1, 'dropped': 2}], {})
    assert result['complete_export']
    assert result['export_quality']['counts_consistent']
    assert result['export_quality']['dropped_records'] == 2
    assert not result['export_quality']['lossless_export']
    assert not analysis.summarize([commit(), {'event': 'summary'}], {})['export_quality'][
        'lossless_export'
    ]


def test_export_counts_reject_mismatch_and_invalid_types():
    for count in (0, 2, True, '1', None):
        result = analysis.summarize(
            [commit(), {'event': 'summary', 'records': count, 'dropped': 0}], {}
        )
        assert not result['export_quality']['lossless_export']
    result = analysis.summarize(
        [{'event': 'metadata'}, commit(), {'event': 'summary', 'records': 1, 'dropped': 0}], {}
    )
    assert result['export_quality']['lossless_export']
    assert result['unmatched'] == 1  # Lossless diagnostics do not prove bag coverage.


def limit_fixture():
    names = [f'joint_{i}' for i in range(12)]
    rows = [{'event': 'metadata', 'joint_names': names}, commit()]
    limits = {name: {'enabled': True, 'min_position': -1.0, 'max_position': 1.0} for name in names}
    return rows, limits


def test_limit_counts_use_only_verified_first_publications():
    rows, limits = limit_fixture()
    report = analysis.command_limit_occupancy(rows, {'42': [(20, [1.0] * 12)]}, limits)
    assert report['accepted_commits'] == 1
    assert report['joints']['joint_0']['near_upper'] == 1
    assert report['joints']['joint_0']['near_lower'] == 0
    assert report['joints']['joint_0']['outside_limits'] == 0
    for commands in (
        {},
        {'42': [(20, [0.0] * 12)]},
        {'42': [(20, [float('nan')] * 12)]},
        {'42': [(20, [1.0] * 12), (21, [1.0] * 12)]},
    ):
        rejected = analysis.command_limit_occupancy(rows, commands, limits)
        assert rejected['accepted_commits'] == 0
        assert rejected['joints']['joint_0']['min_rad'] is None


def test_limit_counts_report_outside_and_validate_metadata():
    rows, limits = limit_fixture()
    limits['joint_0']['max_position'] = 0.9
    report = analysis.command_limit_occupancy(rows, {'42': [(20, [1.0] * 12)]}, limits)
    assert report['joints']['joint_0']['outside_limits'] == 1
    assert report['joints']['joint_0']['near_upper'] == 0
    with pytest.raises(ValueError):
        analysis.command_limit_occupancy(rows[1:], {}, limits)
    with pytest.raises(ValueError):
        analysis.command_limit_occupancy(rows, {}, limits, float('nan'))
    limits['joint_0']['enabled'] = False
    with pytest.raises(ValueError):
        analysis.command_limit_occupancy(rows, {}, limits)


def test_limit_window_uses_publication_time_and_exclusive_end():
    rows, limits = limit_fixture()
    commands = {'42': [(20_000_000, [1.0] * 12)]}
    inside = analysis.command_limit_occupancy(
        rows, commands, limits, start_ros_ns=20_000_000, end_ros_ns=21_000_000
    )
    assert inside['accepted_commits'] == 1
    outside = analysis.command_limit_occupancy(rows, commands, limits, end_ros_ns=20_000_000)
    assert outside['accepted_commits'] == 0
    assert outside['outside_window'] == 1
    assert outside['joints']['joint_0']['max_rad'] is None
    with pytest.raises(ValueError):
        analysis.command_limit_occupancy(rows, commands, limits, start_ros_ns=20, end_ros_ns=10)
