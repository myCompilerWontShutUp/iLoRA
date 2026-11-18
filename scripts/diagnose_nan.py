"""Short diagnostic training run: where and why does the MovieLens baseline loss become NaN?

Same model, data, seed and hyperparameters as run_baseline_movielens.sh (taken from
scripts/benchmark_baseline.py:build_args: MovieLens, seed 1234, num_moe 4, lora_r 8, batch 8 x
accumulation 16, the MoE-LoRA memory fix). The LR schedule is built for the full 5-epoch run
(main.py's max_steps formula), so warmup ends at the same optimizer step as in the real run; --lr
only changes the peak, the warmup start and the floor stay at the run scripts' 8e-6. Training stops
--after_nan_batches micro-batches after the first non-finite loss, or after --max_micro_batches /
--max_optimizer_steps (default: the first epoch). No validation, no checkpoint callback.

Everything goes to a new run folder, runs/nan_diag_<precision>_<lr>_<YYYYmmdd-HHMMSS>/ (never
overwritten; --out_dir must be new or empty), written by the measurement package (see the folder's
README.txt and experiments/IMPLEMENTATION_NOTES.md section 11j): meta.json, micro_batches.csv.gz,
optimizer_steps.csv.gz, activation_stats.csv.gz, anomalies.csv, anomaly_dumps/, anomaly_report.md.
This script adds to micro_batches.csv.gz the first module with a non-finite output and the
"origin module" (the first module, in execution order, whose inputs were finite but whose output
was not: the innermost module whose own code created the inf/NaN), and writes summary.json plus,
at the first non-finite loss, first_nan_batch.pt / first_nan_trainable.pt.

Runs that differ only in --precision/--lr see the same batches in the same order: model init, data
shuffling and the collator's prompt sampling are seeded before training and do not depend on either
flag, and the measurement draws its samples from its own generator. --compare checks it.

    python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision 16     # A
    python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision bf16   # B
    python scripts/diagnose_nan.py --compare <run_dir_A> <run_dir_B>
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

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
from measurement.callback import MeasurementCallback, float_tensors
from measurement.config import add_measurement_args, settings_from_args
from measurement.io import find, read_json, read_rows
from measurement.rundir import claim_dir, make_run_dir
from measurement.schema import FP16_MAX

DIAG_FIELDS = ('nonfinite_modules', 'first_nonfinite_module', 'origin_module')
STEP_KEYS_FOR_FIRST_NAN = ('optimizer_step', 'grad_norm', 'nonfinite_grad_params', 'skipped_by_grad_scaler',
                           'scale_before', 'scale_after', 'params_finite_before_step', 'params_finite_after_step')
COMPARE_KEYS = ('lr', 'lr_warmup_start_lr', 'lr_decay_min_lr', 'gradient_clip_val', 'max_optimizer_steps',
                'max_micro_batches', 'batch_size', 'accumulate_grad_batches', 'num_moe', 'lora_r', 'seed',
                'max_steps', 'warmup_steps')


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llm_path')
    p.add_argument('--precision', default='16', choices=['16', 'bf16'],
                   help='16: fp16 AMP with GradScaler, what main.py uses (default). bf16: bf16 autocast, '
                        'no GradScaler, LLaRA\'s default.')
    p.add_argument('--lr', type=float, default=8e-4,
                   help='Peak lr; 8e-4 is the released train_movielens.sh value, 1e-4 the paper\'s (Appendix E).')
    p.add_argument('--gradient_clip_val', type=float, default=None, help='Default: no clipping, as main.py.')
    p.add_argument('--max_optimizer_steps', type=int, default=0,
                   help='Stop after this many optimizer steps; 0 = the whole first epoch.')
    p.add_argument('--max_micro_batches', type=int, default=0,
                   help='Stop after this many micro-batches (overrides --max_optimizer_steps); 0 = off.')
    p.add_argument('--after_nan_batches', type=int, default=48,
                   help='Micro-batches to keep training after the first non-finite loss.')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--out_dir', default=None,
                   help='Run folder; must not exist or be empty. Default: a new runs/nan_diag_<tag>_<time>/.')
    p.add_argument('--compare', nargs=2, metavar='RUN_DIR',
                   help='Compare two finished runs (no training): an fp16 and a bf16 run with the same lr.')
    add_measurement_args(p)
    ours = p.parse_args(argv)
    if ours.compare:
        return ours, None
    if not ours.llm_path:
        p.error('--llm_path is required unless --compare is given')

    tag = f"p{ours.precision}_lr{ours.lr:g}"
    if ours.gradient_clip_val:
        tag += f"_clip{ours.gradient_clip_val:g}"
    if ours.max_micro_batches:
        tag += f"_mb{ours.max_micro_batches}"
    elif ours.max_optimizer_steps:
        tag += f"_steps{ours.max_optimizer_steps}"
    ours.tag = tag
    ours.run_dir = (claim_dir(ours.out_dir) if ours.out_dir
                    else make_run_dir(ours.measure_root, ours.measure_name or f'nan_diag_{tag}'))
    args = build_baseline_args(['--llm_path', ours.llm_path, '--lr', repr(ours.lr),
                                '--num_workers', str(ours.num_workers),
                                '--benchmark_dir', os.path.join(ours.run_dir, 'trainer_dirs')])
    args.log_dir = 'nan_diag_logs'
    return ours, args


class NanDiagnostics(MeasurementCallback):
    """MeasurementCallback plus the NaN-specific parts: origin module, first-NaN events and dumps,
    early stop. Read-only like its base class."""
    kind = 'diagnose_nan'
    keep_history = True

    def __init__(self, run_dir, settings, precision, config, lr, seed, after_nan_batches, warmup_steps, max_steps,
                 activation_monitor=True):
        super().__init__(run_dir, settings, precision, config, lr, seed, activation_monitor=activation_monitor,
                         extra_micro_fields=DIAG_FIELDS, warmup_steps=warmup_steps, max_steps=max_steps)
        self.after_nan_batches = after_nan_batches
        self.hooks_see_kwargs = True
        self.recording = False
        self.events = {}
        self.order = hashlib.sha1()
        self._reset_origin()

    def _reset_origin(self):
        self.names, self.out_bad, self.in_bad = [], [], []

    def _record(self, name, args, kwargs, output):
        if not self.recording:
            return
        outs = float_tensors(output)
        if not outs:
            return
        ins = float_tensors(tuple(args) + tuple(kwargs.values()))
        device = outs[0].device
        with torch.no_grad():
            out_bad = torch.stack([(~torch.isfinite(t).all()).to(device) for t in outs]).any()
            in_bad = (torch.stack([(~torch.isfinite(t).all()).to(device) for t in ins]).any() if ins
                      else torch.zeros((), dtype=torch.bool, device=device))
        self.names.append(name)
        self.out_bad.append(out_bad)
        self.in_bad.append(in_bad)

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        for name, module in pl_module.named_modules():
            if not name:
                continue
            try:
                module.register_forward_hook(
                    lambda m, a, k, o, _n=name: self._record(_n, a, k, o), with_kwargs=True)
            except TypeError:  # torch without with_kwargs: keyword inputs are not seen
                self.hooks_see_kwargs = False
                module.register_forward_hook(lambda m, a, o, _n=name: self._record(_n, a, {}, o))
        self.meta.update(origin_hooks_see_kwargs=self.hooks_see_kwargs)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, *args):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx, *args)
        self._reset_origin()
        self.recording = True

    def extra_micro(self, row, batch):
        self.recording = False
        row.update(nonfinite_modules=0, first_nonfinite_module='', origin_module='')
        if self.names:
            dev = self.out_bad[0].device
            out_bad = torch.stack([t.to(dev) for t in self.out_bad]).cpu()
            in_bad = torch.stack([t.to(dev) for t in self.in_bad]).cpu()
            bad_idx = out_bad.nonzero().flatten().tolist()
            origin_idx = (out_bad & ~in_bad).nonzero().flatten().tolist()
            row.update(nonfinite_modules=len(bad_idx),
                       first_nonfinite_module=self.names[bad_idx[0]] if bad_idx else '',
                       origin_module=self.names[origin_idx[0]] if origin_idx else '')

    def after_micro(self, trainer, row, batch, events):
        self.order.update(str(row.get('sample_ids', '')).encode() + b';')
        where = ('epoch', 'micro_batch', 'in_optimizer_step', 'lr', 'loss', 'seq_len', 'mean_tokens')
        if row['nonfinite_modules'] and 'first_nonfinite_output' not in self.events:
            self.events['first_nonfinite_output'] = {k: row[k] for k in where + (
                'first_nonfinite_module', 'origin_module')}
        if row.get('gate_finite') is False and 'first_nonfinite_gate' not in self.events:
            self.events['first_nonfinite_gate'] = {k: row[k] for k in where}
        if not row['loss_finite'] and 'first_nonfinite_loss' not in self.events:
            self._on_first_nonfinite_loss(row, batch)
        first = self.events.get('first_nonfinite_loss')
        if first and row['micro_batch'] - first['micro_batch'] >= self.after_nan_batches:
            trainer.should_stop = True

    def on_step(self, trainer, row):
        step = super().on_step(trainer, row)
        for key, hit in (('first_skipped_step', step['skipped_by_grad_scaler']),
                         ('first_nonfinite_grad_step', step['nonfinite_grad_params'] > 0),
                         ('first_nonfinite_param_step', step['nonfinite_params'] > 0)):
            if hit and key not in self.events:
                self.events[key] = {'epoch': step['epoch'], 'optimizer_step': step['optimizer_step'],
                                    'lr': step['lr'], 'grad_norm': step['grad_norm'],
                                    'param': step['first_nonfinite_param'] or step['first_nonfinite_grad_param']}
        first = self.events.get('first_nonfinite_loss')
        if first and 'step' not in first and step['optimizer_step'] == first['in_optimizer_step']:
            first['step'] = {k: step[k] for k in STEP_KEYS_FOR_FIRST_NAN}
        return step

    def _on_first_nonfinite_loss(self, row, batch):
        before = self.history[:-1]
        recent = before[-64:]
        first = {k: row.get(k, '') for k in (
            'epoch', 'micro_batch', 'in_optimizer_step', 'lr', 'loss', 'seq_len', 'mean_tokens', 'target_tokens',
            'min_target_tokens_per_sample', 'placeholder_prompt', 'gate_finite', 'gate_min', 'gate_max',
            'first_nonfinite_module', 'origin_module', 'amp_scale', 'sample_ids', 'm_t', 'm_t_module')}
        first.update(
            params_finite=self._params_finite(),
            max_abs_lowp_first_16=max((r['max_abs_lowp_out'] for r in before[:16]), default=None),
            max_abs_lowp_last_64=max((r['max_abs_lowp_out'] for r in recent), default=None),
            max_abs_lowp_last_64_module=(max(recent, key=lambda r: r['max_abs_lowp_out'])['max_abs_lowp_module']
                                         if recent else ''),
        )
        saved = {}
        for key, value in batch.items():
            if key == 'tokens':
                saved.update({f'tokens.{k}': v.detach().cpu() for k, v in value.items()})
            elif torch.is_tensor(value):
                saved[key] = value.detach().cpu()
            else:
                saved[key] = value
        torch.save(saved, os.path.join(self.run_dir, 'first_nan_batch.pt'))
        torch.save({n: p.detach().cpu() for n, p in self.params},
                   os.path.join(self.run_dir, 'first_nan_trainable.pt'))
        self.events['first_nonfinite_loss'] = first
        print(f"[diag] FIRST NON-FINITE LOSS at epoch {first['epoch']} micro-batch {first['micro_batch']} "
              f"(optimizer step {first['in_optimizer_step']}, lr {first['lr']:.3g}); origin module: "
              f"{first['origin_module'] or '-'}; first non-finite output: {first['first_nonfinite_module'] or '-'}; "
              f"parameters finite: {first['params_finite']}; gate finite: {first['gate_finite']}; "
              f"tokens {first['seq_len']} (mean {first['mean_tokens']}), targets {first['target_tokens']}", flush=True)


def reading(diag):
    """Heuristic interpretation of the recorded events; every line says what it is based on."""
    first = diag.events.get('first_nonfinite_loss')
    if first is None:
        return [f"No non-finite loss in {len(diag.history)} micro-batches ({len(diag.step_rows)} optimizer steps; "
                f"warmup ends at step {diag.warmup_steps})."]
    lines = []
    if not first['params_finite']:
        lines.append("The trainable parameters were already non-finite when the loss first became NaN: an "
                     "optimizer update wrote inf/NaN into them (first_nonfinite_param_step in this summary).")
    elif first['gate_finite'] is False:
        lines.append("The router gate was already non-finite for that batch.")
    elif first['target_tokens'] == 0:
        lines.append("That batch has no target tokens, so the mean cross-entropy is 0/0.")
    elif first['origin_module']:
        lines.append(f"Parameters and gate finite, {first['target_tokens']} target tokens; the inf/NaN was first "
                     f"created inside {first['origin_module']} (its inputs were finite).")
        if diag.precision == '16':
            lines.append("Under fp16 autocast with finite parameters this is the signature of an fp16 overflow "
                         f"(|x| > {FP16_MAX:g}).")
    else:
        lines.append("Not classified automatically; reproduce the forward offline from first_nan_batch.pt and "
                     "first_nan_trainable.pt.")

    later = [r for r in diag.history if r['micro_batch'] > first['micro_batch']]
    later_steps = [s for s in diag.step_rows if s['optimizer_step'] > first['in_optimizer_step']]
    if later and not any(r['loss_finite'] for r in later):
        if (later_steps and all(s['skipped_by_grad_scaler'] for s in later_steps)
                and not any(s['nonfinite_params'] for s in later_steps)):
            lines.append(f"All {len(later)} later micro-batches are NaN too, all {len(later_steps)} later optimizer "
                         "steps were skipped by GradScaler and the parameters stayed finite: training is stuck at "
                         "parameters whose forward overflows, so the loss can never recover.")
        else:
            lines.append(f"All {len(later)} later micro-batches are non-finite too.")
    elif later:
        lines.append(f"{sum(not r['loss_finite'] for r in later)}/{len(later)} later micro-batches are non-finite; "
                     "the loss is finite in between.")
    return lines


# ---- --compare ---------------------------------------------------------------------------------
VERDICTS = {
    (True, False): "fp16 NaN / bf16 stable -> the precision is very likely the cause.",
    (False, False): "both stable -> NaN needs the previous lr 8e-4, alone or together with fp16 "
                    "(optional runs C: --lr 8e-4 --precision 16, D: --lr 8e-4 --precision bf16).",
    (True, True): "both NaN -> not a precision problem alone.",
    (False, True): "fp16 stable / bf16 NaN -> unexpected; re-check the implementation.",
}


def _fmt(x, spec='.4g'):
    return 'n/a' if x is None or x == '' else format(float(x), spec)


def _load_run(run_dir):
    """summary.json + micro-batch rows of a run folder (new .csv.gz or the earlier plain .csv)."""
    summary = read_json(os.path.join(run_dir, 'summary.json'))
    if not summary:
        raise FileNotFoundError(f'{run_dir}/summary.json missing')
    return summary, read_rows(find(run_dir, 'micro_batches'))


def compare(dir_a, dir_b):
    runs = {}
    for d in (dir_a, dir_b):
        summary, rows = _load_run(d)
        runs[d] = (summary, rows)
    precisions = sorted(s['config']['precision'] for s, _ in runs.values())
    problems = []
    if precisions != ['16', 'bf16']:
        problems.append(f"expected one fp16 and one bf16 run, got precisions {precisions}")
    (c1, _), (c2, _) = [(s['config'], r) for s, r in runs.values()]
    shared = [k for k in COMPARE_KEYS if k in c1 and k in c2]
    diff = sorted(k for k in shared if c1.get(k) != c2.get(k))
    if diff:
        problems.append(f"configs differ in more than the precision: {diff}")

    (s1, r1), (s2, r2) = runs.values()
    n = min(len(r1), len(r2))
    mismatch = next((i for i in range(n) if r1[i]['sample_ids'] != r2[i]['sample_ids']), None)
    if mismatch is not None:
        problems.append(f"data order differs from micro-batch {mismatch + 1}")

    print("=== diagnose_nan.py --compare ===")
    for d, (s, rows) in runs.items():
        c = s['config']
        first = s['events'].get('first_nonfinite_loss')
        print(f"\n[{'fp16' if c['precision'] == '16' else 'bf16'}] {d}")
        print(f"  lr {c['lr']:g}, status: {s['status']}, micro-batches {s['micro_batches']}, optimizer steps "
              f"{s['optimizer_steps']} (warmup ends {c['warmup_steps']}), skipped by GradScaler {s['skipped_steps']}")
        if first is None:
            print("  no non-finite loss")
            continue
        step = first.get('step', {})
        print(f"  first non-finite loss: epoch {first['epoch']}, micro-batch {first['micro_batch']}, optimizer step "
              f"{first['in_optimizer_step']}, lr {first['lr']:.3g}, loss {first['loss']}")
        print(f"  origin module: {first['origin_module'] or '-'}; first non-finite output: "
              f"{first['first_nonfinite_module'] or '-'}")
        print(f"  tokens {first['seq_len']} (mean {first['mean_tokens']}), targets {first['target_tokens']}, "
              f"gate finite {first['gate_finite']}, trainable params finite {first['params_finite']}")
        print(f"  that optimizer step: grad norm {_fmt(step.get('grad_norm'))}, scale {_fmt(step.get('scale_before'), 'g')}"
              f" -> {_fmt(step.get('scale_after'), 'g')}, skipped {step.get('skipped_by_grad_scaler')}, params finite "
              f"before/after {step.get('params_finite_before_step')}/{step.get('params_finite_after_step')}")
        for key in ('first_nonfinite_gate', 'first_nonfinite_grad_step', 'first_nonfinite_param_step'):
            if key in s['events']:
                print(f"  {key}: {json.dumps(s['events'][key])}")

    print(f"\ndata order: same for the {n} micro-batches both runs trained" if mismatch is None
          else f"\ndata order: DIFFERS from micro-batch {mismatch + 1}")
    by_precision = {s['config']['precision']: s for s, _ in runs.values()}
    if set(by_precision) == {'16', 'bf16'}:
        fp16_nan = 'first_nonfinite_loss' in by_precision['16']['events']
        bf16_nan = 'first_nonfinite_loss' in by_precision['bf16']['events']
        print(f"verdict: {VERDICTS[(fp16_nan, bf16_nan)]}")
        if fp16_nan != bf16_nan:
            stable, nan = (by_precision['bf16'], by_precision['16']) if fp16_nan else (by_precision['16'], by_precision['bf16'])
            reached = stable['micro_batches']
            at = nan['events']['first_nonfinite_loss']['micro_batch']
            if reached < at:
                problems.append(f"the stable run stopped at micro-batch {reached}, before the other run's first NaN "
                                f"at {at}: it did not cover that point")
    for p in problems:
        print(f"WARNING: {p}")


def main():
    torch.multiprocessing.set_start_method('spawn')
    ours, args = parse_args(sys.argv[1:])
    if ours.compare:
        compare(*ours.compare)
        return
    for d in (args.ckpt_dir, args.output_dir, args.results_dir):
        os.makedirs(d, exist_ok=True)

    print("=== diagnose_nan.py ===")
    print(f"precision={ours.precision} lr={ours.lr:g} gradient_clip_val={ours.gradient_clip_val} "
          f"max_micro_batches={ours.max_micro_batches or '-'} max_optimizer_steps="
          f"{ours.max_optimizer_steps or 'one epoch'} (run scripts: 16 / 1e-4 / None) -> {ours.run_dir}")

    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))
    n_train = len(data_module.trainset)
    max_steps = n_train * args.max_epochs // (args.accumulate_grad_batches * args.batch_size)  # main.py
    warmup_steps = max_steps // 20  # MInterface.configure_optimizers
    print(f"train samples {n_train}; schedule max_steps {max_steps}, warmup {warmup_steps} optimizer steps "
          f"({warmup_steps * args.accumulate_grad_batches} micro-batches)")

    config = {'precision': ours.precision, 'lr': ours.lr, 'lr_warmup_start_lr': args.lr_warmup_start_lr,
              'lr_decay_min_lr': args.lr_decay_min_lr, 'gradient_clip_val': ours.gradient_clip_val,
              'max_optimizer_steps': ours.max_optimizer_steps, 'max_micro_batches': ours.max_micro_batches,
              'after_nan_batches': ours.after_nan_batches, 'batch_size': args.batch_size,
              'accumulate_grad_batches': args.accumulate_grad_batches, 'num_moe': args.num_moe,
              'lora_r': args.lora_r, 'seed': args.seed, 'max_steps': max_steps, 'warmup_steps': warmup_steps}
    settings = settings_from_args(ours)
    diag = NanDiagnostics(ours.run_dir, settings, ours.precision, dict(vars(args), **config), ours.lr, args.seed,
                          ours.after_nan_batches, warmup_steps, max_steps,
                          activation_monitor=not ours.no_activation_monitor)
    if ours.max_micro_batches > 0:
        limit = ours.max_micro_batches
    elif ours.max_optimizer_steps > 0:
        limit = ours.max_optimizer_steps * args.accumulate_grad_batches
    else:
        limit = 1.0
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision=16 if ours.precision == '16' else 'bf16',
        amp_backend='native',
        max_epochs=1,
        max_steps=max_steps,
        limit_train_batches=limit,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[diag],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=ours.gradient_clip_val,
    )

    t0 = time.time()
    error = None
    try:
        trainer.fit(model=model, datamodule=data_module)
    except BaseException as e:  # still write the summary, then re-raise
        error = e

    if error is not None:
        status = f"error: {type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
    elif trainer.interrupted:
        status = 'interrupted'
    elif 'first_nonfinite_loss' in diag.events:
        status = 'stopped after the first non-finite loss'
    else:
        status = 'finished without a non-finite loss'
    diag.finalize(status)
    elapsed = time.time() - t0
    summary = {
        'status': status,
        'config': config,
        'run_dir': ours.run_dir,
        'hooks_see_kwargs': diag.hooks_see_kwargs,
        'micro_batches': len(diag.history),
        'optimizer_steps': len(diag.step_rows),
        'skipped_steps': sum(s['skipped_by_grad_scaler'] for s in diag.step_rows),
        'nonfinite_losses': sum(not r['loss_finite'] for r in diag.history),
        'zero_target_batches': sum(r['target_tokens'] == 0 for r in diag.history),
        'sample_order_sha1': diag.order.hexdigest(),
        'elapsed_seconds': round(elapsed, 1),
        'micro_batches_per_second': round(len(diag.history) / elapsed, 3) if elapsed > 0 else None,
        'max_m_t': diag.recorder.summary['max_m_t'],
        'max_m_t_module': diag.recorder.summary['max_m_t_module'],
        'anomalies': dict(diag.recorder.tracker.severity_counts) if diag.recorder.tracker else None,
        'events': diag.events,
        'reading': reading(diag),
    }
    with open(os.path.join(ours.run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    diag.meta.update(status=status, diag_summary_file='summary.json')

    print("\n=== Summary ===")
    print(f"status: {status}")
    print(f"micro-batches {summary['micro_batches']} ({summary['micro_batches_per_second']} /s), optimizer steps "
          f"{summary['optimizer_steps']}, skipped by GradScaler {summary['skipped_steps']}, non-finite losses "
          f"{summary['nonfinite_losses']}, zero-target batches {summary['zero_target_batches']}")
    print(f"max m_t {summary['max_m_t']} ({summary['max_m_t_module']}); anomalies {summary['anomalies']}")
    for key, event in diag.events.items():
        print(f"{key}: {json.dumps(event)}")
    print("reading (heuristic):")
    for line in summary['reading']:
        print(f"  - {line}")
    print(f"run folder: {ours.run_dir} (README.txt lists every file)")

    if error is not None:
        raise error


if __name__ == '__main__':
    main()
