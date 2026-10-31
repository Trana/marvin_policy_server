#!/usr/bin/env python3
"""Record an operator event beside a reference capture. Does not use ROS."""
import argparse
import json
from pathlib import Path

from capture_reference_run import mark_event


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('label', help='For example: grounded, forward begins, lifted')
    args = parser.parse_args()
    print(json.dumps(mark_event(args.output, args.label)))
