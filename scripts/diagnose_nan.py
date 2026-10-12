"""Short diagnostic training run: where and why does the MovieLens baseline loss become NaN?

Same model, data, seed and hyperparameters as run_baseline_movielens.sh (taken from
scripts/benchmark_baseline.py:build_args, which mirrors the released train_movielens.sh: MovieLens,
seed 1234, num_moe 4, lora_r 8, batch 8 x accumulation 16, the MoE-LoRA memory fix). The LR schedule
is built for the full 5-epoch run (main.py's max_steps formula), so warmup ends at the same optimizer
step as in the real run; --lr only changes the peak, the warmup start and the floor stay at the run
scripts' 8e-6. Training stops --after_nan_batches micro-batches after the first non-finite loss, or
after --max_optimizer_steps (default: the first epoch). No validation, no checkpoint callback;
everything is written to --out_dir (under benchmark_runs/, which is git-ignored).

Runs that differ only in --precision/--lr see the same batches in the same order: model init, data
shuffling and the collator's prompt sampling are seeded before training and do not depend on either
flag. micro_batches.csv records the sample ids of every micro-batch, and --compare checks it. The
order may differ from a main.py run (main.py runs the sanity validation first, which consumes RNG).

micro_batches.csv    one row per micro-batch: epoch, optimizer step it belongs to, loss, lr, AMP loss
                     scale, padded / mean token length, target tokens, gate finite + min/max, the
                     largest |output| among fp16/bf16 module outputs, the first module with a
                     non-finite output and the origin module (see below), sample ids
optimizer_steps.csv  one row per optimizer step: gradient norm after unscaling, parameters with
                     non-finite gradients, GradScaler scale before/after and whether it skipped the
                     step, trainable parameters finite before and after the step
first_nan_batch.pt / first_nan_trainable.pt
                     the batch that first gave a non-finite loss and the trainable parameters at
                     that moment, to reproduce the forward offline
summary.json         first non-finite loss / output / gate / gradient / parameter events, the
                     optimizer step containing the first non-finite loss, and a heuristic reading

Every module gets a forward hook. The "origin module" is the first module, in execution order,
whose inputs were finite but whose output was not: the innermost module whose own code created the
inf/NaN (e.g. a LlamaAttention whose submodules are all fine points at the attention scores).

Defaults are what main.py and the run scripts use (fp16 AMP, lr 8e-4, no clipping).

    python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision 16     # A
    python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision bf16   # B
    python scripts/diagnose_nan.py --compare benchmark_runs/nan_diag/p16_lr0.0001 \\
                                             benchmark_runs/nan_diag/pbf16_lr0.0001
"""
import argparse
import csv
import hashlib
import json
import math
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

FP16_MAX = 65504.0
LOWP = (torch.float16, torch.bfloat16)

MICRO_FIELDS = [
    'epoch', 'micro_batch', 'in_optimizer_step', 'steps_completed', 'lr', 'loss', 'loss_finite',
    'placeholder_prompt', 'seq_len', 'mean_tokens', 'target_tokens', 'min_target_tokens_per_sample',
    'gate_finite', 'gate_min', 'gate_max', 'max_abs_lowp_out', 'max_abs_lowp_module', 'nonfinite_modules',
    'first_nonfinite_module', 'origin_module', 'amp_scale', 'seconds', 'sample_ids',
]
STEP_FIELDS = [
    'epoch', 'optimizer_step', 'lr', 'mean_loss', 'finite_losses', 'micro_batches', 'grad_norm',
    'nonfinite_grad_params', 'first_nonfinite_grad_param', 'skipped_by_grad_scaler', 'scale_before',
    'scale_after', 'params_finite_before_step', 'params_finite_after_step', 'nonfinite_params',
    'first_nonfinite_param', 'max_abs_param',
]
STEP_KEYS_FOR_FIRST_NAN = ('optimizer_step', 'grad_norm', 'nonfinite_grad_params', 'skipped_by_grad_scaler',
                           'scale_before', 'scale_after', 'params_finite_before_step', 'params_finite_after_step')


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llm_path')
    p.add_argument('--precision', default='16', choices=['16', 'bf16'],
                   help='16: fp16 AMP with GradScaler, what main.py uses (default). bf16: bf16 autocast, '
                        'no GradScaler, LLaRA\'s default.')
    p.add_argument('--lr', type=float, default=8e-4,
                   help='Peak lr; 8e-4 is the run scripts\' value, 1e-4 the paper\'s (Appendix E).')
    p.add_argument('--gradient_clip_val', type=float, default=None, help='Default: no clipping, as main.py.')
    p.add_argument('--max_optimizer_steps', type=int, default=0,
                   help='Stop after this many optimizer steps; 0 = the whole first epoch.')
    p.add_argument('--after_nan_batches', type=int, default=48,
                   help='Micro-batches to keep training after the first non-finite loss.')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--out_dir', default=None, help='Default: benchmark_runs/nan_diag/p<precision>_lr<lr>[_clip]')
    p.add_argument('--compare', nargs=2, metavar='RUN_DIR',
                   help='Compare two finished runs (no training): an fp16 and a bf16 run with the same lr.')
    ours = p.parse_args(argv)
    if ours.compare:
        return ours, None
    if not ours.llm_path:
        p.error('--llm_path is required unless --compare is given')

    tag = f"p{ours.precision}_lr{ours.lr:g}"
    if ours.gradient_clip_val:
        tag += f"_clip{ours.gradient_clip_val:g}"
    if ours.max_optimizer_steps:
        tag += f"_steps{ours.max_optimizer_steps}"
    ours.out_dir = ours.out_dir or os.path.join('benchmark_runs', 'nan_diag', tag)
    args = build_baseline_args(['--llm_path', ours.llm_path, '--lr', repr(ours.lr),
                                '--num_workers', str(ours.num_workers), '--benchmark_dir', ours.out_dir])
    args.log_dir = 'nan_diag_logs'
    return ours, args


def _float_tensors(obj):
    if torch.is_tensor(obj):
        items = [obj]
    elif isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (tuple, list)):
        items = list(obj)
    else:
        return []
    return [t for t in items if torch.is_tensor(t) and t.is_floating_point() and t.numel() > 0]


def _any_nonfinite(tensors, device):
    return torch.stack([(~torch.isfinite(t).all()).to(device) for t in tensors]).any()


def _absmax(t):
    return t.detach().abs().amax().float()


def _target_counts(batch, pad_id):
    """Same target masking as MInterface.forward; returns (total, min per sample)."""
    tokens = batch['tokens']
    try:
        targets = tokens.input_ids.masked_fill(tokens.input_ids == pad_id, -100)
        targets = targets.masked_fill((tokens.token_type_ids == 0)[:, 1:], -100)
    except RuntimeError:
        return -1, -1
    per_sample = (targets != -100).sum(dim=1)
    return int(per_sample.sum()), int(per_sample.min())


def _fmt(x, spec='.4g'):
    return 'n/a' if x is None or x == '' else format(float(x), spec)


class NanDiagnostics(pl.Callback):
    """Read-only: hooks and checks never modify tensors, gradients or the optimizer."""

    def __init__(self, out_dir, after_nan_batches, precision, warmup_steps, max_steps):
        self.out_dir = out_dir
        self.after_nan_batches = after_nan_batches
        self.precision = precision
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.hooks_see_kwargs = True
        self.recording = False
        self.params = []
        self.history = []
        self.step_rows = []
        self.events = {}
        self.n_micro = 0
        self.last_global_step = 0
        self.step_losses = []
        self.step_stats = {}
        self.order = hashlib.sha1()
        self._reset_batch()

    def _reset_batch(self):
        self.names, self.out_bad, self.in_bad, self.lowp_max = [], [], [], []
        self.gates = []
        self.loss = None

    def _params_finite(self):
        with torch.no_grad():
            return not bool(torch.stack([~torch.isfinite(p).all() for _, p in self.params]).any())

    # ---- hooks -------------------------------------------------------------------------------
    def _record(self, name, args, kwargs, output):
        if not self.recording:
            return
        outs = _float_tensors(output)
        if not outs:
            return
        ins = _float_tensors(tuple(args) + tuple(kwargs.values()))
        device = outs[0].device
        with torch.no_grad():
            out_bad = _any_nonfinite(outs, device)
            in_bad = _any_nonfinite(ins, device) if ins else torch.zeros((), dtype=torch.bool, device=device)
            lowp = [t for t in outs if t.dtype in LOWP]
            lowp_max = (torch.stack([_absmax(t).to(device) for t in lowp]).max() if lowp
                        else torch.zeros((), device=device))
        self.names.append(name)
        self.out_bad.append(out_bad)
        self.in_bad.append(in_bad)
        self.lowp_max.append(lowp_max)

    def _install_hooks(self, pl_module):
        for name, module in pl_module.named_modules():
            if not name:
                continue
            try:
                module.register_forward_hook(
                    lambda m, a, k, o, _n=name: self._record(_n, a, k, o), with_kwargs=True)
            except TypeError:  # torch without with_kwargs: keyword inputs are not seen
                self.hooks_see_kwargs = False
                module.register_forward_hook(lambda m, a, o, _n=name: self._record(_n, a, {}, o))

        def loss_hook(module, args, output):
            if self.recording:
                loss = getattr(output, 'loss', None)
                if loss is not None:
                    self.loss = loss.detach().float()
        pl_module.llama_model.register_forward_hook(loss_hook)

        def gate_hook(module, args, output):  # the shared router's gate fed into every MoE-LoRA layer
            if self.recording:
                gates = _float_tensors(output)
                if gates:
                    self.gates.append(gates[0].detach())
        if getattr(pl_module, 'router', None) is not None:
            pl_module.router.register_forward_hook(gate_hook)

    # ---- callback ----------------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.out_dir, exist_ok=True)
        self.params = [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]
        self._install_hooks(pl_module)
        self._micro_file = open(os.path.join(self.out_dir, 'micro_batches.csv'), 'w', newline='')
        self._micro_csv = csv.DictWriter(self._micro_file, fieldnames=MICRO_FIELDS)
        self._micro_csv.writeheader()
        self._step_file = open(os.path.join(self.out_dir, 'optimizer_steps.csv'), 'w', newline='')
        self._step_csv = csv.DictWriter(self._step_file, fieldnames=STEP_FIELDS)
        self._step_csv.writeheader()
        print(f"[diag] {len(self.params)} trainable tensors "
              f"({sum(p.numel() for _, p in self.params) / 1e6:.2f}M parameters); hooks see keyword inputs: "
              f"{self.hooks_see_kwargs}; gate hook: {getattr(pl_module, 'router', None) is not None}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, *args):
        self._reset_batch()
        self.recording = True
        self.t0 = time.time()
        self.step_at_start = trainer.global_step
        self.batch_targets = _target_counts(batch, pl_module.llama_tokenizer.pad_token_id)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, *args):
        # For fp16 AMP, Lightning unscales the gradients before this hook.
        scaler = getattr(trainer.precision_plugin, 'scaler', None)
        grads = [(n, p.grad) for n, p in self.params if p.grad is not None]
        stats = {'scale_before': scaler.get_scale() if scaler is not None else None,
                 'grad_norm': None, 'nonfinite_grad_params': 0, 'first_nonfinite_grad_param': '',
                 'params_finite_before_step': self._params_finite()}
        if grads:
            with torch.no_grad():
                norms = torch.stack([g.detach().float().norm() for _, g in grads])
                bad = (~torch.isfinite(norms)).nonzero().flatten().tolist()
                stats.update(grad_norm=float(norms.norm()), nonfinite_grad_params=len(bad),
                             first_nonfinite_grad_param=grads[bad[0]][0] if bad else '')
        self.step_stats = stats

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        self.recording = False
        self.n_micro += 1
        loss = float(self.loss) if self.loss is not None else float('nan')
        scaler = getattr(trainer.precision_plugin, 'scaler', None)
        tokens = batch['tokens']
        sample_ids = ' '.join(str(i) for i in batch['sample_idx'].tolist()) if 'sample_idx' in batch else ''
        self.order.update(sample_ids.encode() + b';')
        row = {
            'epoch': trainer.current_epoch,
            'micro_batch': self.n_micro,
            'in_optimizer_step': self.step_at_start + 1,
            'steps_completed': trainer.global_step,
            'lr': trainer.optimizers[0].param_groups[0]['lr'],
            'loss': loss,
            'loss_finite': math.isfinite(loss),
            'placeholder_prompt': bool(batch.get('flag', False)),
            'seq_len': int(tokens.input_ids.shape[1]),
            'mean_tokens': round(float(tokens.attention_mask.sum(dim=1).float().mean()), 1),
            'target_tokens': self.batch_targets[0],
            'min_target_tokens_per_sample': self.batch_targets[1],
            'gate_finite': '', 'gate_min': '', 'gate_max': '',
            'max_abs_lowp_out': 0.0, 'max_abs_lowp_module': '',
            'nonfinite_modules': 0, 'first_nonfinite_module': '', 'origin_module': '',
            'amp_scale': scaler.get_scale() if scaler is not None else None,
            'sample_ids': sample_ids,
        }
        if self.gates:
            g = torch.cat([t.float().flatten() for t in self.gates]).cpu()
            finite = torch.isfinite(g)
            row['gate_finite'] = bool(finite.all())
            if finite.any():
                row.update(gate_min=float(g[finite].min()), gate_max=float(g[finite].max()))
        if self.names:
            dev = self.out_bad[0].device
            out_bad = torch.stack([t.to(dev) for t in self.out_bad]).cpu()
            in_bad = torch.stack([t.to(dev) for t in self.in_bad]).cpu()
            lowp = torch.stack([t.to(dev) for t in self.lowp_max]).cpu()
            lowp = torch.where(torch.isfinite(lowp), lowp, torch.zeros_like(lowp))
            i_max = int(lowp.argmax())
            if float(lowp[i_max]) > 0:
                row.update(max_abs_lowp_out=float(lowp[i_max]), max_abs_lowp_module=self.names[i_max])
            bad_idx = out_bad.nonzero().flatten().tolist()
            origin_idx = (out_bad & ~in_bad).nonzero().flatten().tolist()
            row.update(nonfinite_modules=len(bad_idx),
                       first_nonfinite_module=self.names[bad_idx[0]] if bad_idx else '',
                       origin_module=self.names[origin_idx[0]] if origin_idx else '')
        row['seconds'] = round(time.time() - self.t0, 3)
        self.history.append(row)
        self.step_losses.append(loss)
        self._micro_csv.writerow(row)
        self._micro_file.flush()

        where = ('epoch', 'micro_batch', 'in_optimizer_step', 'lr', 'loss', 'seq_len', 'mean_tokens')
        if row['nonfinite_modules'] and 'first_nonfinite_output' not in self.events:
            self.events['first_nonfinite_output'] = {k: row[k] for k in where + (
                'first_nonfinite_module', 'origin_module')}
        if row['gate_finite'] is False and 'first_nonfinite_gate' not in self.events:
            self.events['first_nonfinite_gate'] = {k: row[k] for k in where}
        if not row['loss_finite'] and 'first_nonfinite_loss' not in self.events:
            self._on_first_nonfinite_loss(row, batch)

        if trainer.global_step != self.last_global_step:
            self.last_global_step = trainer.global_step
            self._on_optimizer_step(trainer, row)

        first = self.events.get('first_nonfinite_loss')
        if first and self.n_micro - first['micro_batch'] >= self.after_nan_batches:
            trainer.should_stop = True

    def _on_optimizer_step(self, trainer, row):
        stats = self.step_stats
        scale_after = row['amp_scale']
        skipped = (stats.get('scale_before') is not None and scale_after is not None
                   and scale_after < stats['scale_before'])
        with torch.no_grad():
            bad = torch.stack([~torch.isfinite(p).all() for _, p in self.params]).cpu()
            pmax = torch.stack([_absmax(p) for _, p in self.params]).cpu()
        bad_idx = bad.nonzero().flatten().tolist()
        finite_pmax = pmax[torch.isfinite(pmax)]
        losses = self.step_losses
        finite = [x for x in losses if math.isfinite(x)]
        step = {
            'epoch': row['epoch'], 'optimizer_step': trainer.global_step, 'lr': row['lr'],
            'mean_loss': sum(finite) / len(finite) if finite else float('nan'),
            'finite_losses': len(finite), 'micro_batches': len(losses),
            'grad_norm': stats.get('grad_norm'),
            'nonfinite_grad_params': stats.get('nonfinite_grad_params', 0),
            'first_nonfinite_grad_param': stats.get('first_nonfinite_grad_param', ''),
            'skipped_by_grad_scaler': skipped,
            'scale_before': stats.get('scale_before'), 'scale_after': scale_after,
            'params_finite_before_step': stats.get('params_finite_before_step'),
            'params_finite_after_step': not bad_idx,
            'nonfinite_params': len(bad_idx),
            'first_nonfinite_param': self.params[bad_idx[0]][0] if bad_idx else '',
            'max_abs_param': float(finite_pmax.max()) if finite_pmax.numel() else float('nan'),
        }
        self.step_rows.append(step)
        self.step_losses = []
        self.step_stats = {}
        self._step_csv.writerow(step)
        self._step_file.flush()
        for key, hit in (('first_skipped_step', skipped),
                         ('first_nonfinite_grad_step', step['nonfinite_grad_params'] > 0),
                         ('first_nonfinite_param_step', step['nonfinite_params'] > 0)):
            if hit and key not in self.events:
                self.events[key] = {'epoch': step['epoch'], 'optimizer_step': step['optimizer_step'],
                                    'lr': step['lr'], 'grad_norm': step['grad_norm'],
                                    'param': step['first_nonfinite_param'] or step['first_nonfinite_grad_param']}
        first = self.events.get('first_nonfinite_loss')
        if first and 'step' not in first and step['optimizer_step'] == first['in_optimizer_step']:
            first['step'] = {k: step[k] for k in STEP_KEYS_FOR_FIRST_NAN}

        print(f"[diag] epoch {step['epoch']} step {step['optimizer_step']}/{self.max_steps} "
              f"(warmup ends {self.warmup_steps}) lr {step['lr']:.3g} | loss {_fmt(step['mean_loss'])} "
              f"({len(finite)}/{len(losses)} finite) | max|lowp out| {row['max_abs_lowp_out']:.4g} "
              f"{row['max_abs_lowp_module']} | grad {_fmt(step['grad_norm'])} | scale {_fmt(scale_after, 'g')}"
              f"{' SKIPPED' if skipped else ''}"
              f"{' | NON-FINITE PARAMS: ' + str(len(bad_idx)) if bad_idx else ''}", flush=True)

    def _on_first_nonfinite_loss(self, row, batch):
        before = self.history[:-1]
        recent = before[-64:]
        first = {k: row[k] for k in (
            'epoch', 'micro_batch', 'in_optimizer_step', 'lr', 'loss', 'seq_len', 'mean_tokens', 'target_tokens',
            'min_target_tokens_per_sample', 'placeholder_prompt', 'gate_finite', 'gate_min', 'gate_max',
            'first_nonfinite_module', 'origin_module', 'amp_scale', 'sample_ids')}
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
        torch.save(saved, os.path.join(self.out_dir, 'first_nan_batch.pt'))
        torch.save({n: p.detach().cpu() for n, p in self.params},
                   os.path.join(self.out_dir, 'first_nan_trainable.pt'))
        self.events['first_nonfinite_loss'] = first
        print(f"[diag] FIRST NON-FINITE LOSS at epoch {first['epoch']} micro-batch {first['micro_batch']} "
              f"(optimizer step {first['in_optimizer_step']}, lr {first['lr']:.3g}); origin module: "
              f"{first['origin_module'] or '-'}; first non-finite output: {first['first_nonfinite_module'] or '-'}; "
              f"parameters finite: {first['params_finite']}; gate finite: {first['gate_finite']}; "
              f"tokens {first['seq_len']} (mean {first['mean_tokens']}), targets {first['target_tokens']}", flush=True)

    def close(self):
        for f in (getattr(self, '_micro_file', None), getattr(self, '_step_file', None)):
            if f is not None and not f.closed:
                f.close()


def reading(diag):
    """Heuristic interpretation of the recorded events; every line says what it is based on."""
    first = diag.events.get('first_nonfinite_loss')
    if first is None:
        return [f"No non-finite loss in {diag.n_micro} micro-batches ({len(diag.step_rows)} optimizer steps; "
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


def _load_run(run_dir):
    with open(os.path.join(run_dir, 'summary.json')) as f:
        summary = json.load(f)
    with open(os.path.join(run_dir, 'micro_batches.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    return summary, rows


def compare(dir_a, dir_b):
    runs = {}
    for d in (dir_a, dir_b):
        summary, rows = _load_run(d)
        runs[d] = (summary, rows)
    precisions = sorted(s['config']['precision'] for s, _ in runs.values())
    problems = []
    if precisions != ['16', 'bf16']:
        problems.append(f"expected one fp16 and one bf16 run, got precisions {precisions}")
    configs = [{k: v for k, v in s['config'].items() if k != 'precision'} for s, _ in runs.values()]
    diff = sorted(k for k in set(configs[0]) | set(configs[1]) if configs[0].get(k) != configs[1].get(k))
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
    for d in (ours.out_dir, args.ckpt_dir, args.output_dir, args.results_dir):
        os.makedirs(d, exist_ok=True)

    print("=== diagnose_nan.py ===")
    print(f"precision={ours.precision} lr={ours.lr:g} gradient_clip_val={ours.gradient_clip_val} "
          f"max_optimizer_steps={ours.max_optimizer_steps or 'one epoch'} "
          f"(run scripts: 16 / 8e-4 / None) -> {ours.out_dir}")

    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))
    n_train = len(data_module.trainset)
    max_steps = n_train * args.max_epochs // (args.accumulate_grad_batches * args.batch_size)  # main.py
    warmup_steps = max_steps // 20  # MInterface.configure_optimizers
    print(f"train samples {n_train}; schedule max_steps {max_steps}, warmup {warmup_steps} optimizer steps "
          f"({warmup_steps * args.accumulate_grad_batches} micro-batches)")

    diag = NanDiagnostics(ours.out_dir, ours.after_nan_batches, ours.precision, warmup_steps, max_steps)
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision=16 if ours.precision == '16' else 'bf16',
        amp_backend='native',
        max_epochs=1,
        max_steps=max_steps,
        limit_train_batches=(ours.max_optimizer_steps * args.accumulate_grad_batches
                             if ours.max_optimizer_steps > 0 else 1.0),
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
    finally:
        diag.close()

    if error is not None:
        status = f"error: {type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
    elif trainer.interrupted:
        status = 'interrupted'
    elif 'first_nonfinite_loss' in diag.events:
        status = 'stopped after the first non-finite loss'
    else:
        status = 'finished without a non-finite loss'
    summary = {
        'status': status,
        'config': {'precision': ours.precision, 'lr': ours.lr, 'lr_warmup_start_lr': args.lr_warmup_start_lr,
                   'lr_decay_min_lr': args.lr_decay_min_lr, 'gradient_clip_val': ours.gradient_clip_val,
                   'max_optimizer_steps': ours.max_optimizer_steps, 'after_nan_batches': ours.after_nan_batches,
                   'batch_size': args.batch_size, 'accumulate_grad_batches': args.accumulate_grad_batches,
                   'num_moe': args.num_moe, 'lora_r': args.lora_r, 'seed': args.seed,
                   'max_steps': max_steps, 'warmup_steps': warmup_steps},
        'hooks_see_kwargs': diag.hooks_see_kwargs,
        'micro_batches': diag.n_micro,
        'optimizer_steps': len(diag.step_rows),
        'skipped_steps': sum(s['skipped_by_grad_scaler'] for s in diag.step_rows),
        'nonfinite_losses': sum(not r['loss_finite'] for r in diag.history),
        'zero_target_batches': sum(r['target_tokens'] == 0 for r in diag.history),
        'sample_order_sha1': diag.order.hexdigest(),
        'elapsed_seconds': round(time.time() - t0, 1),
        'events': diag.events,
        'reading': reading(diag),
    }
    with open(os.path.join(ours.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"status: {status}")
    print(f"micro-batches {summary['micro_batches']}, optimizer steps {summary['optimizer_steps']}, "
          f"skipped by GradScaler {summary['skipped_steps']}, non-finite losses {summary['nonfinite_losses']}, "
          f"zero-target batches {summary['zero_target_batches']}")
    for key, event in diag.events.items():
        print(f"{key}: {json.dumps(event)}")
    print("reading (heuristic):")
    for line in summary['reading']:
        print(f"  - {line}")
    print(f"files: {ours.out_dir}/summary.json, micro_batches.csv, optimizer_steps.csv"
          f"{', first_nan_batch.pt, first_nan_trainable.pt' if 'first_nonfinite_loss' in diag.events else ''}")

    if error is not None:
        raise error


if __name__ == '__main__':
    main()
