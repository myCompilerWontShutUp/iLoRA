"""Activation baseline without training: forward passes only, same data, order and configuration as
diagnose_nan.py, written in the same run-folder format (measurement package).

Tells whether a large activation (e.g. 22,400 in layers.30.mlp.down_proj during diagnostic run A) is
already there in the pretrained model or grows during training.

Conditions (one run folder each, runs/base_<condition>_p<precision>_<time>/):
  base  LoRA disabled (MoeLoraModel.disable_adapter_layers): the original Llama-2-7B weights
  lora  the MoE-LoRA model as it starts training (step 0). lora_B is initialised to zero
        (MoeLoraLayer.reset_lora_parameters), so the LoRA branch adds exactly 0 and the module
        outputs on the base path equal condition "base"; only the lora_A / lora_B submodules are new.
        --ckpt_path measures a trained checkpoint instead (loaded as in main.py).
Both use the same N micro-batches, drawn once from the seeded training dataloader. The prompt
embeddings still come from the (randomly initialised, untrained) projector in both conditions.
Dropout follows --dropout_mode (train = as during training, default).

    python scripts/measure_base_activation.py --llm_path "$LLM_PATH" --precision 16 --conditions base,lora \
        --num_micro_batches 200 --compare_with runs/nan_diag_p16_lr0.0001_<time>
"""
import argparse
import math
import os
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
from measurement.callback import float_tensors
from measurement.config import add_measurement_args, settings_from_args
from measurement.recorder import Recorder, batch_fields
from measurement.report import compare_base_vs_trained
from measurement.rundir import RunMeta, make_run_dir


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llm_path', required=True)
    p.add_argument('--precision', default='16', choices=['16', 'bf16'])
    p.add_argument('--conditions', default='base,lora', help='Comma list of base, lora.')
    p.add_argument('--num_micro_batches', type=int, default=200)
    p.add_argument('--ckpt_path', default=None, help='Measure this checkpoint for condition "lora".')
    p.add_argument('--dropout_mode', default='train', choices=['train', 'eval'])
    p.add_argument('--compare_with', default=None, help='Trained run folder (e.g. a diagnose_nan.py run).')
    p.add_argument('--num_workers', type=int, default=8)
    add_measurement_args(p)
    ours = p.parse_args(argv)
    ours.conditions = [c.strip() for c in ours.conditions.split(',') if c.strip()]
    bad = [c for c in ours.conditions if c not in ('base', 'lora')]
    if bad:
        p.error(f'unknown condition(s) {bad}')
    return ours


def to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, 'to') and not isinstance(v, (str, bytes)) else v) for k, v in batch.items()}


def set_adapters(model, enabled):
    target = getattr(model.llama_model, 'base_model', None)
    method = getattr(target, 'enable_adapter_layers' if enabled else 'disable_adapter_layers', None)
    if method is None:
        raise RuntimeError('MoE-LoRA model without enable/disable_adapter_layers')
    method()
    return sum(1 for m in model.llama_model.modules() if getattr(m, 'disable_adapters', None) is (not enabled))


def measure(ours, args, model, batches, condition, autocast_dtype):
    name = ours.measure_name or f'base_{condition}_p{ours.precision}'
    run_dir = make_run_dir(ours.measure_root, name)
    settings = settings_from_args(ours)
    recorder = Recorder(run_dir, settings, ours.precision, activation_monitor=not ours.no_activation_monitor)
    meta = RunMeta(run_dir)
    toggled = set_adapters(model, enabled=(condition == 'lora'))
    meta.start(kind='base_activation', config=vars(args), settings=settings.as_dict(), precision=ours.precision,
               lr=None, seed=args.seed, extra={'condition': condition, 'adapters_enabled': condition == 'lora',
                                               'moelora_layers_toggled': toggled, 'ckpt_path': ours.ckpt_path,
                                               'dropout_mode': ours.dropout_mode,
                                               'num_micro_batches': len(batches)})
    hooked = recorder.attach(model, tokenizer=model.llama_tokenizer)
    captured = {}

    def loss_hook(module, a, output):
        loss = getattr(output, 'loss', None)
        if loss is not None:
            captured['loss'] = loss.detach().float()

    def gate_hook(module, a, output):
        gates = float_tensors(output)
        if gates:
            captured.setdefault('gates', []).append(gates[0].detach())
    handles = [model.llama_model.register_forward_hook(loss_hook)]
    if getattr(model, 'router', None) is not None:
        handles.append(model.router.register_forward_hook(gate_hook))
    model.train(ours.dropout_mode == 'train')
    print(f'[base] {condition}: {run_dir}; hooks on {hooked} modules; adapters '
          f"{'on' if condition == 'lora' else 'off'} ({toggled} MoE-LoRA layers set)")
    t_start = time.time()
    status = 'finished'
    try:
        for batch in batches:
            batch = to_device(batch, 'cuda')
            captured.clear()
            row = batch_fields(batch, model.llama_tokenizer.pad_token_id)
            recorder.begin_micro()
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=autocast_dtype):
                model(batch)
            loss = float(captured['loss']) if 'loss' in captured else float('nan')
            row.update(epoch=0, in_optimizer_step=0, steps_completed=0, lr='', loss=loss,
                       loss_finite=math.isfinite(loss), amp_scale='')
            if captured.get('gates'):
                g = torch.cat([t.float().flatten() for t in captured['gates']]).cpu()
                fin = torch.isfinite(g)
                row['gate_finite'] = bool(fin.all())
                if fin.any():
                    row.update(gate_min=float(g[fin].min()), gate_max=float(g[fin].max()))
            row['seconds'] = round(time.time() - t0, 3)
            row, _ = recorder.end_micro(row, batch, {'optimizer_step': 0, 'lr': '', 'amp_scale': '', 'loss': loss})
            if recorder.n_micro % 50 == 0:
                print(f"[base] {condition} micro-batch {recorder.n_micro}/{len(batches)} m_t {row.get('m_t')} "
                      f"{row.get('m_t_module')}", flush=True)
    except BaseException as e:
        status = f'error: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}'
        raise
    finally:
        for h in handles:
            h.remove()
        recorder.close()
        set_adapters(model, enabled=True)
        meta.update(end_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status=status,
                    summary=recorder.summary, elapsed_seconds=round(time.time() - t_start, 1),
                    anomaly_counts=dict(recorder.tracker.severity_counts) if recorder.tracker else {})
    print(f"[base] {condition}: max m_t {recorder.summary['max_m_t']} ({recorder.summary['max_m_t_module']})")
    return run_dir


def main():
    torch.multiprocessing.set_start_method('spawn')
    ours = parse_args(sys.argv[1:])
    args = build_baseline_args(['--llm_path', ours.llm_path, '--num_workers', str(ours.num_workers),
                                '--benchmark_dir', os.path.join(ours.measure_root, 'base_activation_trainer_dirs')])
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))
    if ours.ckpt_path:
        ckpt = torch.load(ours.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'[base] loaded {ours.ckpt_path}')
    model = model.cuda()
    batches = []
    for batch in data_module.train_dataloader():  # same seeded order as diagnose_nan.py (checked in the report)
        batches.append(batch)
        if len(batches) >= ours.num_micro_batches:
            break
    dtype = torch.float16 if ours.precision == '16' else torch.bfloat16
    run_dirs = [measure(ours, args, model, batches, c, dtype) for c in ours.conditions]
    if ours.compare_with:
        out = compare_base_vs_trained(run_dirs, ours.compare_with,
                                      os.path.join(run_dirs[0], 'compare_base_vs_trained.md'))
        print(f'[base] comparison: {out}')
    print('[base] run folders: ' + ', '.join(run_dirs))


if __name__ == '__main__':
    main()
