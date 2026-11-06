"""CPU tests of the measurement package with synthetic tensors (no model, no GPU, no Lightning).

1. anomaly tracker: a value planted at a known (sample, token, feature) is reported at exactly that
   place, with the right severity and rule (abs / high / critical / ratio / record), and dumped;
2. sampled quantiles agree with the exact quantiles (exact when the output is small);
3. the hooks change nothing: loss, gradients and the global RNG state are bit-identical with and
   without the monitor (float32 and bf16 autocast);
4. a writer killed mid-run leaves every row up to its last flush readable;
5. end to end: a toy run folder gets every file, and plot_measurement.py, plot_live.py,
   inspect_anomaly.py, finalize_run.py and compare_base_vs_trained.py work from the folder alone;
6. main.py's measurement flags are split off without touching any main.py argument;
7. values written to the CSV files read back bit-identical (pandas' default float parser changes
   the last bit of many values; measurement.io.read_frame uses round_trip).

    python scripts/test_measurement.py
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement.activation import ActivationMonitor  # noqa: E402
from measurement.anomaly import AnomalyTracker, TokenInfo  # noqa: E402
from measurement.config import MeasureSettings, measurement_parser  # noqa: E402
from measurement.io import CsvWriter, read_frame, read_rows  # noqa: E402
from measurement.recorder import Recorder, batch_fields  # noqa: E402
from measurement.rundir import RunMeta, make_run_dir  # noqa: E402
from measurement.schema import ANOMALIES, names  # noqa: E402

failures = []


def check(ok, msg):
    print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


class Enc(dict):
    def __getattr__(self, k):
        return self[k]


class Tok:
    bos_token_id, pad_token_id = 1, 0

    def convert_ids_to_tokens(self, i):
        return '<s>' if i == 1 else f'tok{i}'

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(self.convert_ids_to_tokens(int(i)) for i in ids)


class Emit(nn.Module):
    def forward(self, x):
        return x


def toy_llama():
    """Modules named like the real model: llama_model.layers.<n>.mlp.down_proj / .self_attn.q_proj."""
    root = nn.Module()
    root.llama_model = nn.Module()
    layer = nn.Module()
    layer.mlp, layer.self_attn = nn.Module(), nn.Module()
    layer.mlp.down_proj, layer.self_attn.q_proj = Emit(), Emit()
    root.llama_model.layers = nn.ModuleDict({'30': layer})
    return root


def batch_for(n, length, sample_ids, special=None):
    ids = torch.full((n, length), 5, dtype=torch.long)
    ids[:, 0] = 1
    for (s, t), v in (special or {}).items():
        ids[s, t] = v
    tt = torch.zeros(n, length + 1, dtype=torch.long)
    tt[:, -3:] = 1
    return {'tokens': Enc(input_ids=ids, attention_mask=torch.ones(n, length, dtype=torch.long), token_type_ids=tt),
            'sample_idx': torch.tensor(sample_ids), 'flag': False}


def run_toy_batch(model, monitor, x, collect_full=False):
    monitor.begin(collect_full)
    layer = model.llama_model.layers['30']
    layer.self_attn.q_proj(x * 0.5)
    layer.mlp.down_proj(x)
    return monitor.end()


def test_tracker():
    print('1. anomaly tracker coordinates and rules')
    tmp = tempfile.mkdtemp()
    s = MeasureSettings(record_warmup=5, ratio_min_history=5, print_level='critical')
    model = toy_llama()
    monitor = ActivationMonitor(s).attach(model)
    writer = CsvWriter(os.path.join(tmp, 'anomalies.csv'), names(ANOMALIES))
    os.makedirs(os.path.join(tmp, 'anomaly_dumps'))
    tracker = AnomalyTracker(s, tmp, 'bf16', writer)
    tok = Tok()
    torch.manual_seed(0)
    events = []
    for step in range(12):  # history: small values
        x = torch.randn(3, 7, 4096).to(torch.bfloat16)
        batch = batch_for(3, 7, [10, 11, 12])
        events += tracker.process({'micro_batch': step + 1, 'optimizer_step': 1}, run_toy_batch(model, monitor, x),
                                  TokenInfo.from_batch(batch, tok))
    check(not [e for e in events if e['severity'] != 'info'], f'no warn+ events on N(0,1) data ({len(events)} info)')

    x = torch.randn(3, 7, 4096).to(torch.bfloat16)
    x[1, 4, 2533] = 40960.0                     # exactly representable in bf16 and fp16
    batch = batch_for(3, 7, [10, 11, 12], {(1, 4): 777})
    ev = tracker.process({'micro_batch': 13, 'optimizer_step': 1}, run_toy_batch(model, monitor, x),
                         TokenInfo.from_batch(batch, tok))
    down = [e for e in ev if e['module'].endswith('down_proj')]
    e = down[0] if down else {}
    check(len(down) == 1 and e['value'] == 40960.0, f"one down_proj event with value 40960 (got {e.get('value')})")
    check((e.get('sample_idx'), e.get('token_pos'), e.get('feature_dim')) == (1, 4, 2533),
          f"position (sample 1, token 4, dim 2533): got {(e.get('sample_idx'), e.get('token_pos'), e.get('feature_dim'))}")
    check(e.get('sample_id') == 11 and e.get('token_id') == 777 and e.get('token_str') == 'tok777',
          'sample id 11, token id 777, token string')
    check(e.get('severity') == 'high' and set(e.get('rule', '').split(',')) >= {'high', 'abs', 'ratio'},
          f"severity high, rules high+abs+ratio (got {e.get('severity')}, {e.get('rule')})")
    check(e.get('is_known_massive_dim') is True and e.get('is_first_token') is False, 'dim 2533 flagged, not first token')
    check(e.get('layer') == 30 and e.get('module_type') == 'down_proj', 'layer 30, type down_proj')
    d = np.load(os.path.join(tmp, e['dump_file']))
    check(tuple(d['top_coords'][0]) == (1, 4, 2533) and float(d['top_values'][0]) == 40960.0, 'dump: top-1 coordinate and value')
    check(np.array_equal(d['token_vector'], x[1, 4].float().numpy()), 'dump: token vector is the output row (sample 1, token 4)')
    check(list(d['input_ids']) == batch['tokens'].input_ids[1].tolist(), 'dump: input ids of the sample')
    check(set(map(str, d['same_layer_modules'])) == {'llama_model.layers.30.mlp.down_proj',
                                                    'llama_model.layers.30.self_attn.q_proj'}, 'dump: same-layer modules')
    with open(os.path.join(tmp, e['dump_file'][:-4] + '.txt'), encoding='utf-8') as f:
        check('tok777' in f.read(), 'dump: decoded prompt in the .txt')

    x = torch.randn(3, 7, 4096).to(torch.bfloat16)
    x[2, 0, 7] = float('inf')
    ev = tracker.process({'micro_batch': 14, 'optimizer_step': 1}, run_toy_batch(model, monitor, x),
                         TokenInfo.from_batch(batch_for(3, 7, [10, 11, 12]), tok))
    crit = [e for e in ev if e['severity'] == 'critical' and e['module'].endswith('down_proj')]
    check(len(crit) == 1 and crit[0]['rule'] == 'nonfinite' and (crit[0]['sample_idx'], crit[0]['token_pos'],
          crit[0]['feature_dim']) == (2, 0, 7) and crit[0]['is_first_token'] is True,
          'inf at (2, 0, 7): critical, rule nonfinite, located, first token')

    x = torch.randn(3, 7, 4096).to(torch.bfloat16)
    x[0, 3, 100] = 60.0                         # ~13x the window median (about 4.5), far below 16384
    ev = tracker.process({'micro_batch': 15, 'optimizer_step': 1}, run_toy_batch(model, monitor, x),
                         TokenInfo.from_batch(batch_for(3, 7, [10, 11, 12]), tok))
    r = [e for e in ev if e['module'].endswith('down_proj')]
    check(r and r[0]['severity'] == 'warn' and 'ratio' in r[0]['rule'] and 'abs' not in r[0]['rule'],
          f"relative rule alone: warn/ratio (got {[(e['severity'], e['rule']) for e in r]})")
    writer.close()
    rows = read_rows(os.path.join(tmp, 'anomalies.csv'))
    check(len(rows) == tracker.next_id - 1, f'anomalies.csv has every event ({len(rows)})')


def test_quantiles():
    print('2. sampled quantiles')
    torch.manual_seed(1)
    model = toy_llama()
    for n_tokens, sample, tol in ((20, 100_000, 0.0), (2000, 100_000, 0.03)):
        monitor = ActivationMonitor(MeasureSettings(quantile_sample=sample)).attach(model)
        x = torch.randn(4, n_tokens, 1024).to(torch.bfloat16)
        mb = run_toy_batch(model, monitor, x, collect_full=True)
        i = [k for k, r in enumerate(mb.records) if r['name'].endswith('down_proj')][0]
        got = mb.full[i][4:]
        exact = torch.quantile(x.abs().float().reshape(-1), torch.tensor([0.5, 0.9, 0.99, 0.999])).numpy()
        rel = np.abs(got - exact) / exact
        n = x.numel()
        ok = np.array_equal(got.astype(np.float32), exact) if n <= sample else bool((rel[:3] < tol).all() and rel[3] < 3 * tol)
        check(ok, f'{n} elements, sample {sample}: rel. error p50/p90/p99/p999 = {np.round(rel, 4).tolist()}'
                  f"{' (exact expected)' if n <= sample else ''}")
        mean_abs, std = mb.full[i][0], mb.full[i][1]
        check(abs(mean_abs - x.abs().float().mean().item()) < 1e-4 and abs(std - x.float().std().item()) < 1e-4,
              'mean |a| and std over all elements')


class Cast(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


def test_noop():
    print('3. hooks do not change the computation')
    # Matmuls in float32 (CPU oneDNN has no bf16 backward on every CPU); with dtype bf16 the cast,
    # GELU and dropout outputs in the middle are bf16, which is what the hooks read.
    torch.backends.mkldnn.enabled = False  # native CPU kernels; some CPUs have no oneDNN bf16 backward
    for dtype in (torch.float32, torch.bfloat16):
        results = []
        for hooked in (False, True):
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(64, 128), Cast(dtype), nn.GELU(), nn.Dropout(0.1), Cast(torch.float32),
                                  nn.Linear(128, 64))
            monitor = ActivationMonitor(MeasureSettings(quantile_sample=50)).attach(model) if hooked else None
            x = torch.randn(4, 9, 64)
            torch.manual_seed(123)
            if monitor:
                monitor.begin(True)
            loss = model(x).float().pow(2).mean()
            loss.backward()
            if monitor:
                mb = monitor.end()
            results.append((loss.detach().clone(), [p.grad.clone() for p in model.parameters()], torch.get_rng_state()))
        (l0, g0, r0), (l1, g1, r1) = results
        check(torch.equal(l0, l1) and all(torch.equal(a, b) for a, b in zip(g0, g1)) and torch.equal(r0, r1),
              f'{dtype}: loss, {len(g0)} gradients and the global RNG state bit-identical '
              f'(monitor saw {len(mb)} lowp outputs)')
    torch.backends.mkldnn.enabled = True


WRITER_CHILD = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
from measurement.io import CsvWriter
w = CsvWriter(sys.argv[2], ['i', 'value'])
for i in range(200):
    w.writerow({'i': i, 'value': i * 0.5})
w.flush()
print('flushed', flush=True)
for i in range(200, 100000):
    w.writerow({'i': i, 'value': i * 0.5})
time.sleep(60)
'''


def test_kill():
    print('4. killed writer')
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'rows.csv.gz')
    proc = subprocess.Popen([sys.executable, '-c', WRITER_CHILD, REPO_ROOT, path], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    time.sleep(1.0)
    proc.kill()
    proc.wait()
    rows = read_rows(path)
    ok = line.strip() == 'flushed' and len(rows) >= 200 and all(int(r['i']) == k for k, r in enumerate(rows))
    check(ok, f'killed after the flush: {len(rows)} intact rows readable (>= 200 expected)')


def test_end_to_end():
    print('5. end to end (toy run folder)')
    root = tempfile.mkdtemp()
    s = MeasureSettings(record_warmup=10, ratio_min_history=5, print_level='critical', stats_every=2)
    model = toy_llama()
    run_dirs = []
    for name, spike in (('toy_trained', True), ('toy_base', False)):
        run_dir = make_run_dir(root, name)
        meta = RunMeta(run_dir)
        meta.start(kind='test', config={'warmup_steps': 2}, settings=s.as_dict(), precision='16', lr=1e-4, seed=0)
        rec = Recorder(run_dir, s, '16')
        rec.attach(model, Tok())
        torch.manual_seed(2)
        scale = 65536.0
        for mb in range(1, 41):
            batch = batch_for(2, 6, [2 * mb, 2 * mb + 1])
            rec.begin_micro()
            layer = model.llama_model.layers['30']
            x = torch.randn(2, 6, 4096).to(torch.bfloat16)   # bf16: fully supported on CPU in torch 2.0
            if spike and mb == 30:
                x[0, 0, 1415] = 20480.0                      # exact in bf16
            layer.self_attn.q_proj(x * 0.5)
            layer.mlp.down_proj(x)
            row = batch_fields(batch, 0)
            row.update(epoch=0, in_optimizer_step=(mb - 1) // 4 + 1, steps_completed=mb // 4, lr=1e-4,
                       loss=2.0 / mb, loss_finite=True, amp_scale=scale, seconds=0.01)
            rec.end_micro(row, batch, {'optimizer_step': row['in_optimizer_step'], 'lr': 1e-4, 'amp_scale': scale, 'loss': row['loss']})
            if mb % 4 == 0:
                skipped = mb == 8
                rec.write_step({'epoch': 0, 'optimizer_step': mb // 4, 'lr': 1e-4, 'mean_loss': 2.0 / mb,
                                'grad_norm': 1.0 / mb, 'skipped_by_grad_scaler': skipped, 'scale_before': scale,
                                'scale_after': scale / 2 if skipped else scale})
                scale = scale / 2 if skipped else scale
        rec.close()
        meta.update(end_time_utc='done', summary=rec.summary)
        run_dirs.append(run_dir)
    trained, base = run_dirs
    with open(os.path.join(trained, 'gpu.csv'), 'w') as f:  # synthetic GPU log, test only
        f.write('timestamp,memory_used_mib,memory_total_mib,utilization_gpu_pct,power_draw_w\n')
        for i in range(5):
            f.write(f'2026/10/23 10:0{i}:00.000,{60000 + i},81920,{90 + i},{300 + i}\n')
    for fname in ('meta.json', 'micro_batches.csv.gz', 'optimizer_steps.csv.gz', 'activation_stats.csv.gz',
                  'anomalies.csv', 'README.txt', 'anomaly_report.md'):
        check(os.path.exists(os.path.join(trained, fname)), f'{fname} written')
    micro = read_frame(os.path.join(trained, 'micro_batches.csv.gz'))
    spike = micro[micro['micro_batch'] == 30].iloc[0]
    check(spike['m_t'] == 20480.0 and spike['m_t_token_pos'] == 0 and spike['m_t_feature_dim'] == 1415
          and bool(spike['m_t_is_first_token']) and bool(spike['m_t_is_known_massive_dim']),
          'micro-batch 30: m_t 20480 at token 0, dim 1415 (first token, known massive dim)')
    stats = read_frame(os.path.join(trained, 'activation_stats.csv.gz'))
    check(sorted(stats['micro_batch'].unique().tolist()) == list(range(1, 41, 2)), 'activation_stats every 2nd micro-batch')
    ev = read_frame(os.path.join(trained, 'anomalies.csv'))
    hit = ev[(ev['micro_batch'] == 30) & ev['module'].str.endswith('down_proj')]
    check(len(hit) == 1 and 'abs' in hit.iloc[0]['rule'], 'spike event recorded (rule abs)')

    sys.argv = ['plot_measurement.py']
    sys.path.insert(0, os.path.join(REPO_ROOT, 'analysis'))
    import plot_measurement
    written = plot_measurement.main([trained, '--base', base, '--compare', base])
    expected = {'m_t.png', 'module_max_abs.png', 'layer_module_heatmap.png', 'loss_grad_norm.png', 'loss_scale.png',
                'anomaly_positions.png', 'gpu.png', 'base_vs_trained.png'}
    got = {os.path.basename(w) for w in written if w}
    check(expected <= got and any(g.startswith('quantile_trend_') for g in got), f'figures: {sorted(got)}')
    for script, args in (('plot_live.py', [trained]), ('inspect_anomaly.py', [trained, str(int(hit.iloc[0]['event_id']))]),
                         ('finalize_run.py', [trained]),
                         ('compare_base_vs_trained.py', ['--base', base, '--trained', trained,
                                                         '--out', os.path.join(root, 'cmp.md')])):
        out = subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'scripts', script)] + args,
                             capture_output=True, text=True)
        check(out.returncode == 0, f'{script} runs from the folder alone' + ('' if out.returncode == 0 else f': {out.stderr[-300:]}'))
    with open(os.path.join(root, 'cmp.md'), encoding='utf-8') as f:
        text = f.read()
    check('layers.30.mlp.down_proj' in text and 'same' in text, 'compare_base_vs_trained.md: focus module and batch order')


def test_main_flags():
    print('6. main.py measurement flags')
    argv = ['--mode', 'train', '--lr', '1e-4', '--max_epochs', '5', '--no_augment', '--save', 'part', '--seed', '1']
    m, rest = measurement_parser().parse_known_args(argv)
    check(rest == argv and m.measure is False, 'without --measure every argument is passed through unchanged')
    m, rest = measurement_parser().parse_known_args(argv + ['--measure', '--stats_every', '4', '--measure_name', 'x'])
    check(rest == argv and m.measure and m.stats_every == 4 and m.measure_name == 'x',
          'with --measure the measurement flags are removed, the rest unchanged')


def test_exact_round_trip():
    print('7. raw values read back exactly')
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.standard_normal(5000) * 10.0 ** rng.integers(-12, 8, 5000),
                           [22400.0, 65504.0, 0.1 + 0.2, 1e-30, 3.4e38, float('inf'), -float('inf')]])
    path = os.path.join(tempfile.mkdtemp(), 'values.csv.gz')
    w = CsvWriter(path, ['value'])
    for v in vals:
        w.writerow({'value': float(v)})
    w.close()
    frame = read_frame(path)['value'].to_numpy(dtype=np.float64)
    rows = np.array([float(r['value']) for r in read_rows(path)])
    check(np.array_equal(frame.view(np.int64), vals.view(np.int64)) and np.array_equal(rows.view(np.int64), vals.view(np.int64)),
          f'{len(vals)} floats bit-identical through read_frame and read_rows')


if __name__ == '__main__':
    for test in (test_tracker, test_quantiles, test_noop, test_kill, test_end_to_end, test_main_flags,
                 test_exact_round_trip):
        test()
    if failures:
        print(f'{len(failures)} check(s) failed', file=sys.stderr)
        sys.exit(1)
    print('All measurement checks passed.')
