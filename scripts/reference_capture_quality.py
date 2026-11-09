#!/usr/bin/env python3
"""Read-only quality report for completed, uncompressed SQLite reference bags."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def summarize_timestamps(values):
    values = sorted(values)
    gaps = [b - a for a, b in zip(values, values[1:])]
    span = values[-1] - values[0] if values else 0
    return {
        'count': len(values),
        'first_ns': values[0] if values else None,
        'last_ns': values[-1] if values else None,
        'mean_recorded_hz': (len(values) - 1) * 1e9 / span if span else None,
        'max_gap_ms': max(gaps) / 1e6 if gaps else None,
        'duplicate_timestamps': sum(gap == 0 for gap in gaps),
    }


def report(output):
    metadata = json.loads((output / 'metadata.json').read_text())
    files = sorted((output / 'topics').glob('*.db3'))
    if not files:
        raise ValueError(
            'No SQLite bags found; MCAP/compressed bags are not supported by this report'
        )
    timestamps = defaultdict(list)
    for path in files:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            for topic, stamp in db.execute(
                'SELECT topics.name, messages.timestamp FROM messages '
                'JOIN topics ON topics.id = messages.topic_id'
            ):
                timestamps[topic].append(stamp)
    topics = {
        name: summarize_timestamps(timestamps[name])
        for name in sorted(set(metadata['topics']) | set(timestamps))
    }
    status_path = output / 'status.jsonl'
    rows, malformed = [], 0
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected object')
                rows.append(row)
            except (ValueError, TypeError):
                malformed += 1
    return {
        'schema': 'reference-quality.v1',
        'topics': topics,
        'empty_topics': [name for name, value in topics.items() if not value['count']],
        'status': {
            'records': len(rows),
            'malformed': malformed,
            'missing_file': not status_path.exists(),
            'successful': sum(row.get('success') is True for row in rows),
            'typed_snapshots': sum(isinstance(row.get('state'), dict) for row in rows),
            'unsuccessful': sum(
                row.get('success') is not True and not isinstance(row.get('state'), dict)
                for row in rows
            ),
        },
        'limits': (
            'Bag receive timestamps, not sensor age or inference latency. '
            'Gaps do not distinguish publisher pauses from recording loss. '
            'Empty optional topics are not automatically faults. '
            'Typed snapshot receipt does not prove a healthy or moving controller. '
            'Run only after capture stops. This is not a locomotion acceptance verdict.'
        ),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.output), indent=2, allow_nan=False))
