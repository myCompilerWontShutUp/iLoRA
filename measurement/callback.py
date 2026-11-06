"""Lightning callback that writes a run folder during training (main.py --measure, diagnose_nan.py).

Read-only: hooks and checks never modify tensors, gradients or the optimizer, and the measurement's
random sampling uses its own generator. With the callback absent, training is unchanged.
"""
import atexit
import math
import os
import time

import torch
import pytorch_lightning as pl

from .recorder import Recorder, batch_fields
from .rundir import RunMeta, training_artifacts


def float_tensors(obj):
    if torch.is_tensor(obj):
        items = [obj]
    elif isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (tuple, list)):
        items = list(obj)
    else:
        return []
    return [t for t in items if torch.is_tensor(t) and t.is_floating_point() and t.numel() > 0]


def _fmt(x, spec='.4g'):
    return 'n/a' if x is None or x == '' else format(float(x), spec)


class MeasurementCallback(pl.Callback):
    kind = 'train'
    keep_history = False  # micro-batch rows kept in memory (diagnose_nan.py needs them, long runs do not)

    def __init__(self, run_dir, settings, precision, config, lr, seed, activation_monitor=True,
                 extra_micro_fields=(), meta_extra=None, warmup_steps=None, max_steps=None):
        self.run_dir = run_dir
        self.settings = settings
        self.precision = str(precision)
        self.config = config
        self.lr, self.seed = lr, seed
        self.meta_extra = dict(meta_extra or {})
        self.warmup_steps, self.max_steps = warmup_steps, max_steps
        self.recorder = Recorder(run_dir, settings, self.precision, extra_micro_fields, activation_monitor)
        self.meta = RunMeta(run_dir)
        self.params = []
        self.history = []
        self.step_rows = []
        self.last_global_step = 0
        self.step_losses = []
        self.step_stats = {}
        self.loss = None
        self.gates = []
        self.tensorboard_dir = None
        self._started = self._finalized = False
        atexit.register(self.finalize, 'process exit without a normal end of training')

    # ---- meta ----------------------------------------------------------------------------------
    def _start_meta(self, trainer):
        if self._started:
            return
        self._started = True
        logger = getattr(trainer, 'logger', None)
        self.tensorboard_dir = getattr(logger, 'log_dir', None) if logger else None
        self.meta.start(kind=self.kind, config=self.config, settings=self.settings.as_dict(),
                        precision=self.precision, lr=self.lr, seed=self.seed,
                        extra=dict(self.meta_extra, tensorboard_dir=self.tensorboard_dir,
                                   activation_monitor=self.recorder.monitor is not None,
                                   gpu_monitor_command=f'python scripts/gpu_monitor.py {self.run_dir} --interval 60'))
        print(f'[measure] run folder: {self.run_dir}')
        print(f'[measure] GPU log (separate process): python scripts/gpu_monitor.py {self.run_dir} --interval 60')

    def setup(self, trainer, pl_module, stage=None):
        self._start_meta(trainer)

    def finalize(self, status='finished'):
        if self._finalized:
            return
        self._finalized = True
        try:
            self.recorder.close()
        finally:
            files = {}
            for name in sorted(os.listdir(self.run_dir)):
                p = os.path.join(self.run_dir, name)
                files[name] = os.path.getsize(p) if os.path.isfile(p) else 'dir'
            self.meta.update(end_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status=status,
                             summary=self.recorder.summary, files=files,
                             anomaly_counts=dict(self.recorder.tracker.severity_counts) if self.recorder.tracker else {},
                             artifacts=training_artifacts(self.config, self.tensorboard_dir))

    def on_exception(self, trainer, pl_module, exception):
        self.finalize(f'exception: {type(exception).__name__}: {str(exception).splitlines()[0] if str(exception) else ""}')

    def teardown(self, trainer, pl_module, stage=None):
        self.finalize('interrupted' if getattr(trainer, 'interrupted', False) else 'finished')

    # ---- hooks ---------------------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        self._start_meta(trainer)
        self.params = [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]
        hooked = self.recorder.attach(pl_module, tokenizer=getattr(pl_module, 'llama_tokenizer', None))

        def loss_hook(module, args, output):
            loss = getattr(output, 'loss', None)
            if loss is not None:
                self.loss = loss.detach().float()
        pl_module.llama_model.register_forward_hook(loss_hook)

        def gate_hook(module, args, output):  # the shared router's gate fed into every MoE-LoRA layer
            gates = float_tensors(output)
            if gates:
                self.gates.append(gates[0].detach())
        if getattr(pl_module, 'router', None) is not None:
            pl_module.router.register_forward_hook(gate_hook)
        self.meta.update(hooked_modules=hooked, trainable_tensors=len(self.params),
                         trainable_parameters=sum(p.numel() for _, p in self.params))
        print(f'[measure] {len(self.params)} trainable tensors; activation hooks on {hooked} modules; '
              f'stats every {self.settings.stats_every} micro-batch(es)')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, *args):
        self.loss, self.gates = None, []
        self.t0 = time.time()
        self.step_at_start = trainer.global_step
        self.batch_info = batch_fields(batch, pl_module.llama_tokenizer.pad_token_id)
        self.recorder.begin_micro()

    def _params_finite(self):
        with torch.no_grad():
            return not bool(torch.stack([~torch.isfinite(p).all() for _, p in self.params]).any())

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

    def extra_micro(self, row, batch):
        """Subclass hook: add extra micro-batch fields before the row is written."""

    def after_micro(self, trainer, row, batch, events):
        """Subclass hook: runs after the row is written."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        loss = float(self.loss) if self.loss is not None else float('nan')
        scaler = getattr(trainer.precision_plugin, 'scaler', None)
        row = dict(self.batch_info)
        row.update({
            'epoch': trainer.current_epoch, 'in_optimizer_step': self.step_at_start + 1,
            'steps_completed': trainer.global_step, 'lr': trainer.optimizers[0].param_groups[0]['lr'],
            'loss': loss, 'loss_finite': math.isfinite(loss),
            'amp_scale': scaler.get_scale() if scaler is not None else '',
        })
        if self.gates:
            g = torch.cat([t.float().flatten() for t in self.gates]).cpu()
            finite = torch.isfinite(g)
            row['gate_finite'] = bool(finite.all())
            if finite.any():
                row.update(gate_min=float(g[finite].min()), gate_max=float(g[finite].max()))
        self.extra_micro(row, batch)
        row['seconds'] = round(time.time() - self.t0, 3)
        ctx = {'optimizer_step': row['in_optimizer_step'], 'lr': row['lr'], 'amp_scale': row['amp_scale'],
               'loss': loss}
        row, events = self.recorder.end_micro(row, batch, ctx)
        if self.keep_history:
            self.history.append(row)
        self.step_losses.append(loss)
        self.after_micro(trainer, row, batch, events)
        if trainer.global_step != self.last_global_step:
            self.last_global_step = trainer.global_step
            self.on_step(trainer, row)

    def on_step(self, trainer, row):
        stats = self.step_stats
        scale_after = row['amp_scale'] if row['amp_scale'] != '' else None
        skipped = (stats.get('scale_before') is not None and scale_after is not None
                   and scale_after < stats['scale_before'])
        with torch.no_grad():
            bad = torch.stack([~torch.isfinite(p).all() for _, p in self.params]).cpu()
            pmax = torch.stack([p.detach().abs().amax().float() for _, p in self.params]).cpu()
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
        self.step_losses, self.step_stats = [], {}
        self.recorder.write_step(step)
        total = f'/{self.max_steps}' if self.max_steps else ''
        warm = f' (warmup ends {self.warmup_steps})' if self.warmup_steps else ''
        print(f"[measure] epoch {step['epoch']} step {step['optimizer_step']}{total}{warm} lr {step['lr']:.3g} | "
              f"loss {_fmt(step['mean_loss'])} ({len(finite)}/{len(losses)} finite) | m_t {_fmt(row.get('m_t'))} "
              f"{row.get('m_t_module', '')} | grad {_fmt(step['grad_norm'])} | scale {_fmt(scale_after, 'g')}"
              f"{' SKIPPED' if skipped else ''}{' | NON-FINITE PARAMS: ' + str(len(bad_idx)) if bad_idx else ''}",
              flush=True)
        return step
