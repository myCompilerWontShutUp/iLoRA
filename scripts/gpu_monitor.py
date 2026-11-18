"""GPU resource log for a run folder: appends one nvidia-smi sample every --interval seconds to
<run_dir>/gpu.csv. Runs as its own process next to training and stops when the run's meta.json gets
an end time, when --pid exits, or on Ctrl-C.

    python scripts/gpu_monitor.py <run_dir> --interval 60 [--pid <training pid>]
    python scripts/gpu_monitor.py --latest                     # newest folder under runs/

Fields: nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw
--format=csv,noheader,nounits, written with the header of measurement.schema.GPU.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement.schema import GPU, names  # noqa: E402

QUERY = ['nvidia-smi', '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw',
         '--format=csv,noheader,nounits']


def ended(run_dir):
    try:
        with open(os.path.join(run_dir, 'meta.json'), encoding='utf-8') as f:
            return bool(json.load(f).get('end_time_utc'))
    except (OSError, ValueError):
        return False


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', nargs='?')
    p.add_argument('--latest', action='store_true')
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--pid', type=int, default=None)
    a = p.parse_args()
    run_dir = a.run_dir
    if a.latest:
        candidates = sorted(glob.glob(os.path.join('runs', '*', 'meta.json')), key=os.path.getmtime)
        if not candidates:
            raise SystemExit('no run folder under runs/')
        run_dir = os.path.dirname(candidates[-1])
    if not run_dir or not os.path.isdir(run_dir):
        raise SystemExit('give a run folder or --latest')
    path = os.path.join(run_dir, 'gpu.csv')
    new = not os.path.exists(path)
    print(f'[gpu_monitor] {path}, every {a.interval:g} s')
    with open(path, 'a', encoding='utf-8') as f:
        if new:
            f.write(','.join(names(GPU)) + '\n')
            f.flush()
        try:
            while True:
                out = subprocess.run(QUERY, capture_output=True, text=True)
                for line in out.stdout.strip().splitlines():
                    f.write(','.join(x.strip() for x in line.split(',')) + '\n')
                f.flush()
                if ended(run_dir) or (a.pid is not None and not alive(a.pid)):
                    print('[gpu_monitor] run ended; stopping')
                    break
                time.sleep(a.interval)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
