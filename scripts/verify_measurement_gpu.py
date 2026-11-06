"""GPU checks of the measurement package on the real model (VESSL A100). Section 9 of the task:

1. bit-identity: one fixed batch, forward + backward with the hooks off, off again (to see whether
   the GPU run itself is deterministic) and on (full statistics); loss and every trainable gradient
   compared bit for bit, for fp16 and bf16 autocast;
2. overhead: micro-batch/s for forward + backward over --timing_batches batches with the hooks off,
   on with m_t / anomalies only, and on with full statistics (including the CPU-side recording);
   bytes written per micro-batch -> run-folder size for one epoch (8,265 micro-batches);
3. short real run: diagnose_nan.py with --max_micro_batches 50 in a subprocess, then every file is
   checked and analysis/plot_measurement.py redraws the figures from the folder alone;
4. hard kill: diagnose_nan.py killed with SIGKILL after 20 micro-batches; the flushed CSV rows are
   read back and scripts/finalize_run.py writes README / report.

    python scripts/verify_measurement_gpu.py --llm_path "$LLM_PATH"            # all, about 25-40 min
    python scripts/verify_measurement_gpu.py --llm_path "$LLM_PATH" --skip_runs  # 1 and 2 only
Results: printed and written to runs/verify_measurement_<time>/verify_report.json.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import torch
import pytorch_lightning as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, 'scripts')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model.model_interface import MInterface
from data.data_interface import DInterface
# rec_model/movielens.pt was pickled as __main__.SASRec: same imports as main.py.
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU  # noqa: F401
from SASRecModules_ori import *  # noqa: F401,F403
from benchmark_baseline import build_args as build_baseline_args
from measurement.config import MeasureSettings
from measurement.io import find, read_rows
from measurement.recorder import Recorder, batch_fields
from measurement.rundir import make_run_dir

EPOCH_MICRO_BATCHES = 8265
DTYPES = {'16': torch.float16, 'bf16': torch.bfloat16}


def to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, 'to') and not isinstance(v, (str, bytes)) else v) for k, v in batch.items()}


def step(model, batch, dtype, seed):
    """One forward + backward; returns (loss, {name: grad copy})."""
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.autocast(device_type='cuda', dtype=dtype):
        out = model(batch)
    out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.detach().clone(), grads


def compare(a, b):
    (la, ga), (lb, gb) = a, b
    same_loss = torch.equal(la, lb)
    same = sum(torch.equal(ga[k], gb[k]) for k in ga if k in gb)
    max_diff = max(((ga[k].float() - gb[k].float()).abs().max().item() for k in ga if k in gb), default=0.0)
    return {'loss_identical': bool(same_loss), 'grads_identical': int(same), 'grads_total': len(ga),
            'all_identical': bool(same_loss and same == len(ga) and set(ga) == set(gb)), 'max_abs_grad_diff': max_diff}


def bitwise(model, batch, precisions, out_dir):
    results = {}
    for prec in precisions:
        dtype = DTYPES[prec]
        off1 = step(model, batch, dtype, 1234)
        off2 = step(model, batch, dtype, 1234)
        rec = Recorder(make_run_dir(out_dir, f'bitwise_p{prec}'), MeasureSettings(), prec)
        rec.attach(model, model.llama_tokenizer)
        rec.begin_micro()
        on = step(model, batch, dtype, 1234)
        rec.end_micro(batch_fields(batch, model.llama_tokenizer.pad_token_id), batch, {})
        rec.close()
        results[prec] = {'off_vs_off': compare(off1, off2), 'off_vs_on': compare(off1, on)}
        r = results[prec]
        print(f"[bitwise] {prec}: hooks off vs off identical={r['off_vs_off']['all_identical']}; "
              f"off vs on identical={r['off_vs_on']['all_identical']} "
              f"(loss {r['off_vs_on']['loss_identical']}, grads {r['off_vs_on']['grads_identical']}/"
              f"{r['off_vs_on']['grads_total']}, max |diff| {r['off_vs_on']['max_abs_grad_diff']:.3g})", flush=True)
    return results


def timing(model, batches, precision, out_dir):
    dtype = DTYPES[precision]
    results = {}
    for mode in ('off', 'light', 'full'):
        rec = None
        if mode != 'off':
            settings = MeasureSettings(stats_every=1 if mode == 'full' else 0)
            rec = Recorder(make_run_dir(out_dir, f'timing_{mode}'), settings, precision)
            rec.attach(model, model.llama_tokenizer)
        step(model, batches[0], dtype, 0)  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        for i, batch in enumerate(batches):
            if rec:
                rec.begin_micro()
            loss, _ = step(model, batch, dtype, i)
            if rec:
                row = batch_fields(batch, model.llama_tokenizer.pad_token_id)
                row.update(loss=float(loss), loss_finite=True)
                rec.end_micro(row, batch, {'optimizer_step': 1, 'loss': float(loss)})
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        entry = {'micro_batches_per_s': round(len(batches) / elapsed, 3)}
        if rec:
            rec.close()
            sizes = {name: os.path.getsize(os.path.join(rec.run_dir, name)) for name in os.listdir(rec.run_dir)
                     if os.path.isfile(os.path.join(rec.run_dir, name))}
            per_mb = sum(sizes.values()) / len(batches)
            entry.update(bytes_per_micro_batch=round(per_mb), epoch_estimate_mib=round(per_mb * EPOCH_MICRO_BATCHES / 2 ** 20, 1),
                         hooked_modules=rec.monitor.hooked_modules if rec.monitor else 0, files=sizes)
        results[mode] = entry
        print(f"[timing] {precision} hooks {mode}: {entry['micro_batches_per_s']} micro-batch/s"
              + (f", ~{entry['epoch_estimate_mib']} MiB per epoch" if rec else ''), flush=True)
    base = results['off']['micro_batches_per_s']
    for mode in ('light', 'full'):
        results[mode]['relative_to_off'] = round(results[mode]['micro_batches_per_s'] / base, 3)
    return results


def diag_cmd(llm_path, out_dir, micro_batches):
    return [sys.executable, os.path.join(REPO_ROOT, 'scripts', 'diagnose_nan.py'), '--llm_path', llm_path,
            '--lr', '1e-4', '--precision', '16', '--max_micro_batches', str(micro_batches), '--out_dir', out_dir]


def short_run(llm_path, out_root):
    run_dir = os.path.join(out_root, 'short_run_50')
    proc = subprocess.run(diag_cmd(llm_path, run_dir, 50), cwd=REPO_ROOT)
    files = sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []
    needed = ['meta.json', 'micro_batches.csv.gz', 'optimizer_steps.csv.gz', 'activation_stats.csv.gz',
              'anomalies.csv', 'README.txt', 'anomaly_report.md', 'summary.json']
    rows = len(read_rows(find(run_dir, 'micro_batches')))
    plot = subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'analysis', 'plot_measurement.py'), run_dir],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    figures = sorted(os.listdir(os.path.join(run_dir, 'figures'))) if os.path.isdir(os.path.join(run_dir, 'figures')) else []
    result = {'returncode': proc.returncode, 'missing_files': [f for f in needed if f not in files],
              'micro_batch_rows': rows, 'plot_returncode': plot.returncode, 'figures': figures, 'run_dir': run_dir}
    print(f"[short run] exit {proc.returncode}, {rows} micro-batch rows, missing {result['missing_files']}, "
          f"figures {figures}", flush=True)
    return result


def kill_run(llm_path, out_root, kill_after=20):
    run_dir = os.path.join(out_root, 'killed_run')
    proc = subprocess.Popen(diag_cmd(llm_path, run_dir, 400), cwd=REPO_ROOT)
    rows = 0
    deadline = time.time() + 1800
    while time.time() < deadline and proc.poll() is None:
        time.sleep(10)
        rows = len(read_rows(find(run_dir, 'micro_batches'))) if os.path.isdir(run_dir) else 0
        if rows >= kill_after:
            break
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    time.sleep(2)
    rows = len(read_rows(find(run_dir, 'micro_batches')))
    steps = len(read_rows(find(run_dir, 'optimizer_steps')))
    stats = len(read_rows(find(run_dir, 'activation_stats')))
    fin = subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'scripts', 'finalize_run.py'), run_dir],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    result = {'micro_batch_rows_after_kill': rows, 'optimizer_step_rows': steps, 'activation_stats_rows': stats,
              'finalize_returncode': fin.returncode, 'readme': os.path.exists(os.path.join(run_dir, 'README.txt')),
              'run_dir': run_dir}
    print(f"[kill] SIGKILL after >= {kill_after} rows: {rows} micro-batch rows, {steps} step rows, {stats} stats "
          f"rows readable; finalize exit {fin.returncode}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llm_path', required=True)
    p.add_argument('--precisions', default='16,bf16')
    p.add_argument('--timing_batches', type=int, default=20)
    p.add_argument('--skip_runs', action='store_true', help='Skip the 50-micro-batch run and the kill test.')
    a = p.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    out_root = make_run_dir('runs', 'verify_measurement')
    args = build_baseline_args(['--llm_path', a.llm_path, '--num_workers', '2',
                                '--benchmark_dir', os.path.join(out_root, 'trainer_dirs')])
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))
    model = model.cuda().train()
    batches = []
    for batch in data_module.train_dataloader():
        batches.append(to_device(batch, 'cuda'))
        if len(batches) >= a.timing_batches:
            break
    report = {'bitwise': bitwise(model, batches[0], [x for x in a.precisions.split(',') if x], out_root),
              'timing': timing(model, batches, '16', out_root)}
    del model, batches
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if not a.skip_runs:
        report['short_run'] = short_run(a.llm_path, out_root)
        report['kill_test'] = kill_run(a.llm_path, out_root)
    with open(os.path.join(out_root, 'verify_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[verify] report: {os.path.join(out_root, "verify_report.json")}')


if __name__ == '__main__':
    main()
