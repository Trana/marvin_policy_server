#!/usr/bin/env python3
"""Read recorder status or request a clean stop. Never open its SQLite files."""
import argparse
import json
from pathlib import Path
import time


def control(root, stop=False):
    progress = json.loads((root / 'progress.json').read_text())
    age = (time.time_ns() - progress['observed_unix_ns']) / 1e9
    progress['status_age_s'] = age
    if stop:
        if (root / 'complete.json').exists() or progress['state'] != 'recording':
            raise ValueError('Recorder is not recording; no stop request made')
        if not 0 <= age <= 10:
            raise ValueError('Recorder status is stale; inspect process and log before recovery')
        # This is a request, not an assertion that recording has stopped.
        with (root / 'stop.request').open('a'):
            pass
        progress['stop_requested'] = True
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--stop', action='store_true', help='Request graceful finalization; does not stop motors')
    args = parser.parse_args()
    print(json.dumps(control(args.capture, args.stop), indent=2))


if __name__ == '__main__':
    main()
