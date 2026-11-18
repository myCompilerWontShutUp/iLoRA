"""Live view of a running (or finished) run folder: redraws <run_dir>/figures/live.png from the
flushed CSVs (loss, gradient norm, loss scale, m_t with the fp16 limit) and warns about non-finite
losses and critical anomaly events.

    python scripts/plot_live.py <run_dir>                # once
    python scripts/plot_live.py <run_dir> --watch 300    # every 5 minutes until Ctrl-C or the run ends
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement import plots  # noqa: E402


def once(run_dir):
    run = plots.Run(run_dir)
    plots.fig_live(run, os.path.join(run_dir, 'figures'))
    warnings = []
    if run.micro is not None and 'loss_finite' in run.micro:
        bad = run.micro[~run.micro['loss_finite'].astype(str).str.lower().isin(['true', '1'])]
        if len(bad):
            warnings.append(f"{len(bad)} micro-batch(es) with a non-finite loss, first at micro-batch "
                            f"{int(bad['micro_batch'].iloc[0])}")
    if run.anomalies is not None:
        crit = run.anomalies[run.anomalies['severity'] == 'critical']
        if len(crit):
            warnings.append(f"{len(crit)} critical anomaly event(s), first: event {int(crit['event_id'].iloc[0])} "
                            f"{crit['module'].iloc[0]} value {crit['value'].iloc[0]}")
    n = 0 if run.micro is None else len(run.micro)
    last = '' if run.micro is None or not n else f", m_t {run.micro['m_t'].iloc[-1]:.6g}"
    print(f"[live] {time.strftime('%H:%M:%S')} {n} micro-batches{last}")
    for w in warnings:
        print(f'[live] WARNING: {w}', flush=True)
    return bool(run.meta.get('end_time_utc'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--watch', type=float, default=0, help='Redraw every N seconds (0 = once).')
    a = p.parse_args()
    while True:
        done = once(a.run_dir)
        if not a.watch or done:
            break
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
