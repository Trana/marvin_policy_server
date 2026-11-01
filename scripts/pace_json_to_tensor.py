#!/usr/bin/env python3
"""Run in the PACE/PyTorch environment; no ROS or robot access."""
import argparse
import json
from pathlib import Path


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    raw = json.loads(args.input.read_text())
    data = {key: torch.tensor(raw[key], dtype=torch.float32)
            for key in ('time', 'des_dof_pos', 'dof_pos')}
    t, c, q = data['time'], data['des_dof_pos'], data['dof_pos']
    if (t.ndim != 1 or len(t) < 2 or c.ndim != 2 or c.shape != q.shape
            or c.shape[0] != len(t) or c.shape[1] == 0
            or not all(torch.isfinite(v).all() for v in data.values())
            or not torch.all(t[1:] > t[:-1])
            or not torch.allclose(t[1:]-t[:-1], (t[1]-t[0]).expand(len(t)-1), atol=1e-5, rtol=1e-4)):
        raise ValueError('Invalid PACE tensor shapes, values or time grid')
    with args.output.open('xb') as stream:
        torch.save(data, stream)


if __name__ == '__main__':
    main()
