"""Print one anomaly event of a run folder: its anomalies.csv row, the dump (top-32 values and their
(sample, token, feature) positions, the token vector, the other modules of the layer) and the prompt.

    python scripts/inspect_anomaly.py <run_dir> <event_id>
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement.io import read_rows  # noqa: E402
from measurement.schema import short_name  # noqa: E402


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run_dir, event_id = sys.argv[1], sys.argv[2]
    rows = [r for r in read_rows(os.path.join(run_dir, 'anomalies.csv')) if r['event_id'] == str(event_id)]
    if not rows:
        raise SystemExit(f'event {event_id} not in {run_dir}/anomalies.csv')
    event = rows[0]
    print(f'=== event {event_id} ===')
    for k, v in event.items():
        print(f'{k:>22}: {v}')
    if not event.get('dump_file'):
        print('\n(no dump for this event: the dump limit was reached and its severity is below high)')
        return
    d = np.load(os.path.join(run_dir, event['dump_file']), allow_pickle=False)
    print(f"\noutput shape {d['output_shape'].tolist()}")
    print('\ntop |values| of the same output tensor:')
    print(f"{'rank':>4} {'value':>12} {'sample':>6} {'token':>6} {'feature':>8}")
    for i, (v, (s, t, f)) in enumerate(zip(d['top_values'], d['top_coords'])):
        print(f'{i + 1:>4} {v:>12.6g} {s:>6} {t:>6} {f:>8}')
    vec = d['token_vector']
    order = np.argsort(-np.abs(vec))[:10]
    print(f"\ntoken vector at the event (features {len(vec)}): mean |v| {np.abs(vec).mean():.4g}, "
          f"median |v| {np.median(np.abs(vec)):.4g}")
    print('largest features: ' + ', '.join(f'{int(i)}:{vec[i]:.5g}' for i in order))
    if len(d['same_layer_modules']):
        print('\nsame layer, same micro-batch:')
        for name, v in sorted(zip(d['same_layer_modules'], d['same_layer_max_abs']), key=lambda x: -x[1]):
            print(f'  {short_name(str(name)):<45} max |v| {v:.6g}')
    txt = os.path.join(run_dir, event['dump_file'][:-4] + '.txt')
    if os.path.exists(txt):
        with open(txt, encoding='utf-8') as f:
            text = f.read()
        print('\n' + text[text.find('prompt'):])


if __name__ == '__main__':
    main()
