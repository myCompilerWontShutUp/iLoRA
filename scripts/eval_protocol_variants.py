"""Protocol variants of the test-only re-evaluation in eval_prompt_semantics.py. Nothing is trained.

Same path as eval_prompt_semantics.py (run_baseline_movielens.sh's `main.py --mode test` command run in
this process, outputs in a new folder, checkpoint and reference files hashed before and after, code
identical to the checkpoint's commit), with these switches:

  --split val            evaluate the validation set: the test loader is DInterface.val_dataloader's
                         loader (the one the in-process validation of the training run used). The
                         result is compared with the validation counts the training run logged, which
                         were computed in the training process, i.e. with the router BatchNorm running
                         statistics that MInterface.on_save_checkpoint never writes to the checkpoint.
  --batch_size N         test batch size (the run used 8)
  --drop_last            drop the last incomplete batch, as upstream's test_dataloader does
                         (batch 4 keeps 92 of the 95 test sequences, the N iLoRA Table 1 implies)
  --bn_stats recomputed  re-estimate the router BatchNorm running statistics before evaluating: one pass
                         over the training sequences in train mode, batches of 8 as in fine-tuning
                         (router on batch statistics, SASRec dropout on, fp16 autocast on CUDA),
                         cumulative average. All RNG states are restored afterwards, so candidate
                         sampling is the same as without this switch.
  --prompt_mode          current_hybrid or aug2024_text_only, as in eval_prompt_semantics.py

Candidates are drawn by MovielensData.__getitem__ in seeded DataLoader workers. A different batch size
or split draws different candidates; the test split with batch 8 and no drop_last draws the reference
ones, which reference_check verifies.

    python scripts/eval_protocol_variants.py --split val --ckpt "checkpoints/baseline/epoch=01-metric=0.521.ckpt"
    python scripts/eval_protocol_variants.py --split val --bn_stats recomputed --ckpt "checkpoints/baseline/epoch=01-metric=0.521.ckpt"
    python scripts/eval_protocol_variants.py --batch_size 4 --drop_last --ckpt "checkpoints/baseline/epoch=01-metric=0.521.ckpt"
"""
import argparse
import contextlib
import datetime
import importlib
import json
import os
import random
import sys
import time
from math import erf, sqrt

import eval_prompt_semantics as base  # also puts the repository root first on sys.path

REPO = base.REPO
NEW_FILES = ('scripts/eval_protocol_variants.py', 'scripts/router_bn_sensitivity.py',
             'scripts/build_historical_snapshot.sh', 'scripts/historical_preflight.py')
PROTECTED = ('runs/semantics_eval_v2',)  # in addition to eval_prompt_semantics.PROTECTED
DEFAULT_OUT_ROOT = 'runs/protocol_variants'
# In-process validation of the 6c7ebac baseline run, (total, valid, correct) per epoch, from
# log/movielens_baseline_logs/version_0 (val_prediction_valid / val_hr at global steps 516, 1033, 1550,
# 2067). Epoch 4 has none: max_steps=2583 was reached at micro-batch 8240/8277, before its validation,
# so the metric in epoch=04-metric=0.479.ckpt and last.ckpt is epoch 3's.
IN_PROCESS_VALIDATION = {0: (94, 93, 43), 1: (94, 92, 49), 2: (94, 90, 48), 3: (94, 93, 45)}
LOADER_CHECK, BN_REPORT = {}, {}


def recompute_router_bn(router, rec_model, seqs, lens, batch_size=8, seed=0, autocast=False):
    """Re-estimate the running statistics of every BatchNorm in `router` from the training sequences.

    Train mode as during fine-tuning (PyTorch Lightning puts the whole module, SASRec included, in train
    mode), drop_last batches of `batch_size` in a fixed permutation, momentum=None (cumulative average).
    The original statistics were an exponential average over the last training batches; this is an
    approximation of them for the same weights. RNG states are saved and restored around the pass."""
    import numpy as np
    import torch
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    norms = [m for m in router.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    modes = (router.training, rec_model.training)
    device = next(router.parameters()).device
    order = np.random.RandomState(seed).permutation(len(seqs))
    seqs, lens = torch.as_tensor(seqs), torch.as_tensor(lens)
    amp = torch.autocast(device_type='cuda', dtype=torch.float16) if autocast and device.type == 'cuda' \
        else contextlib.nullcontext()
    batches = 0
    try:
        for m in norms:
            m.reset_running_stats()
            m.momentum = None
        router.train()
        rec_model.train()
        # Lightning runs test hooks under inference mode; the running-statistics update needs normal tensors
        with torch.inference_mode(False), torch.no_grad(), amp:
            for start in range(0, len(order) - batch_size + 1, batch_size):
                idx = torch.as_tensor(order[start:start + batch_size])
                router(rec_model.cacul_h(seqs[idx].to(device), lens[idx].to(device)))
                batches += 1
    finally:
        for m, momentum in zip(norms, momenta):
            m.momentum = momentum
        router.train(modes[0])
        rec_model.train(modes[1])
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    return {'batchnorm_layers': len(norms), 'batches': batches, 'batch_size': batch_size, 'sequences': len(order),
            'permutation_seed': seed, 'fp16_autocast': autocast and device.type == 'cuda',
            'first_layer_running_mean_head': [round(float(v), 6) for v in norms[0].running_mean[:4]],
            'first_layer_running_var_head': [round(float(v), 6) for v in norms[0].running_var[:4]]}


def use_bn_recompute():
    """Run recompute_router_bn in MInterface.on_test_start, after Lightning has moved the model to the GPU
    and switched it to eval mode, before the test loader draws its worker seeds."""
    from model.model_interface import MInterface

    def on_test_start(self):
        data = self.trainer.datamodule.trainset.session_data
        BN_REPORT.update(recompute_router_bn(self.router, self.rec_model, list(data['seq']), list(data['len_seq']),
                                             autocast=True))
        self.router.eval()
        self.rec_model.eval()

    MInterface.on_test_start = on_test_start


def use_loader(split, drop_last, historical_folder):
    """DInterface.test_dataloader replaced by a loader over `split` with the requested drop_last and test
    collater; every other DataLoader setting is checked against the run's own loader for that split."""
    from torch.utils.data import DataLoader
    import data.data_interface as current
    collater_class = current.TrainCollater
    if historical_folder:
        sys.path.insert(0, historical_folder)  # spawned workers inherit sys.path
        collater_class = importlib.import_module(base.HIST_MODULE).TrainCollater
    own_loader = current.DInterface.val_dataloader if split == 'val' else current.DInterface.test_dataloader

    def test_dataloader(self):
        reference = own_loader(self)
        loader = DataLoader(self.valset if split == 'val' else self.testset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            drop_last=drop_last,
                            collate_fn=collater_class(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer,
                                                      train=False))
        for name in ('dataset', 'batch_size', 'num_workers', 'pin_memory', 'timeout', 'prefetch_factor',
                     'persistent_workers', 'generator'):
            if getattr(loader, name) is not getattr(reference, name) and getattr(loader, name) != getattr(reference, name):
                raise RuntimeError(f'variant loader differs from the run\'s {split} loader in {name}')
        if type(loader.sampler) is not type(reference.sampler):
            raise RuntimeError('variant loader uses a different sampler')
        LOADER_CHECK.update(split=split, dataset_samples=len(loader.dataset), batch_size=loader.batch_size,
                            num_workers=loader.num_workers, drop_last=loader.drop_last,
                            reference_drop_last=reference.drop_last, sampler=type(loader.sampler).__name__,
                            collater=f'{collater_class.__module__}.{collater_class.__name__}')
        return loader

    current.DInterface.test_dataloader = test_dataloader


def two_proportion(c1, n1, c2, n2):
    if not n1 or not n2:
        return None
    p1, p2, pooled = c1 / n1, c2 / n2, (c1 + c2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    return {'difference': p1 - p2, 'z': z, 'two_sided_p': 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))}


def code_identity():
    identity = base.code_identity()
    changed = [p for p in identity['files_differing_from_checkpoint_commit'] if p not in NEW_FILES]
    return dict(identity, files_differing_from_checkpoint_commit=changed,
                baseline_code_identical=not changed and not identity['untracked_py_in_import_paths'])


def evaluate(args):
    ckpt = base.repo_path(args.ckpt)
    name = (f"{os.path.splitext(os.path.basename(ckpt))[0]}__{args.prompt_mode}__{args.split}__bs{args.batch_size}"
            f"{'_droplast' if args.drop_last else ''}__bn-{args.bn_stats}")
    out_dir = base.repo_path(args.out_dir) if args.out_dir else os.path.join(base.repo_path(args.out_root), name)
    blocked = base.protected_by(out_dir) or next(
        (p for p in PROTECTED if os.path.commonpath([out_dir, base.repo_path(p)]) == base.repo_path(p)), None)
    if blocked:
        raise SystemExit(f'{out_dir} is inside {blocked}/, which this script never writes to')
    if os.path.exists(out_dir):
        raise SystemExit(f'{out_dir} exists; evaluation folders are never overwritten')
    if not os.path.isfile(ckpt):
        raise SystemExit(f'{ckpt} not found')
    identity = code_identity()
    if not identity['baseline_code_identical'] and not args.allow_code_difference:
        raise SystemExit(f'code differs from {base.CHECKPOINT_COMMIT}: {identity}')

    argv = base.baseline_test_argv()
    base.set_arg(argv, '--ckpt_path', ckpt)
    base.set_arg(argv, '--output_dir', os.path.join(out_dir, 'output') + '/')
    base.set_arg(argv, '--results_dir', os.path.join(out_dir, 'results'))
    base.set_arg(argv, '--log_dir', 'protocol_variants_logs')
    base.set_arg(argv, '--batch_size', str(args.batch_size))
    notes = []
    if '--export_gates' in argv and (args.prompt_mode == 'aug2024_text_only' or args.split == 'val'):
        argv.remove('--export_gates')
        notes.append(base.EXPORT_GATES_NOTE if args.prompt_mode == 'aug2024_text_only' else
                     '--export_gates removed: gates of validation samples would be written under the test label')
    if args.dry_run:
        print(json.dumps({'out_dir': out_dir, 'code': identity, 'main_py_argv': argv, 'notes': notes}, indent=2))
        return 0

    watched = [ckpt] + [os.path.join(REPO, p) for p in (base.REFERENCE_METRICS, base.REFERENCE_TEST_CSV)]
    before = {p: base.sha256(p) for p in watched if os.path.exists(p)}
    info = base.checkpoint_info(ckpt)
    os.makedirs(out_dir)
    historical = base.materialize_historical(out_dir) if args.prompt_mode == 'aug2024_text_only' else None
    use_loader(args.split, args.drop_last, historical)
    if args.bn_stats == 'recomputed':
        use_bn_recompute()

    start = time.time()
    base.run_main(argv)
    wall = time.time() - start

    content = base.read_test_csv(os.path.join(out_dir, 'output', 'test.csv'))
    metrics = base.score(content)
    stored = json.load(open(os.path.join(out_dir, 'results', 'metrics.json')))
    if [metrics['total'], metrics['valid'], metrics['correct']] != \
            [stored['test_total_num'], stored['test_valid_num'], stored['test_correct_num']]:
        raise SystemExit(f'test.csv rescoring {metrics} disagrees with the in-process metrics.json {stored}')
    after = {p: base.sha256(p) for p in before}

    comparison = None
    if args.split == 'val':
        epoch = info['epoch_from_filename'] if info['file'].startswith('epoch=') else None
        logged = IN_PROCESS_VALIDATION.get(epoch)
        comparison = {'in_process_validation': None if logged is None else dict(zip(('total', 'valid', 'correct'), logged)),
                      'note': 'candidates are drawn again here, so this is not a paired comparison' if logged else
                              'no in-process validation exists for this checkpoint'}
        if logged:
            comparison['hr_difference_out_minus_in'] = two_proportion(metrics['correct'], metrics['valid'], logged[2], logged[1])
            comparison['correct_over_total_difference_out_minus_in'] = two_proportion(
                metrics['correct'], metrics['total'], logged[2], logged[0])
    elif os.path.exists(os.path.join(REPO, base.REFERENCE_TEST_CSV)):
        reference = base.read_test_csv(os.path.join(REPO, base.REFERENCE_TEST_CSV))
        same_rows = len(reference['real']) >= len(content['real'])
        n = len(content['real'])
        comparison = {'reference_test_csv': base.REFERENCE_TEST_CSV,
                      'candidates_identical_to_reference_prefix': same_rows and reference['real'][:n] == content['real']
                      and reference['cans'][:n] == content['cans'],
                      'reference_prefix_metrics': base.score({k: v[:n] for k, v in reference.items()})}

    result = {
        'created_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'training_performed': False,
        'code': dict(identity, installed_patches_match_debug=base.transformers_patches()),
        'checkpoint': dict(info, unchanged_after_eval=after[ckpt] == before[ckpt] == info['sha256']),
        'variant': {'split': args.split, 'batch_size': args.batch_size, 'drop_last': args.drop_last,
                    'bn_stats': args.bn_stats, 'prompt_mode': args.prompt_mode,
                    'prompt_mode_description': base.PROMPT_MODES[args.prompt_mode],
                    'test_collater': dict(base.HISTORICAL) if historical else 'data/data_interface.py TrainCollater',
                    'notes': notes},
        'loader_check': LOADER_CHECK,
        'bn_recompute': BN_REPORT or None,
        'main_py_argv': argv,
        'metrics': metrics,
        'comparison': comparison,
        'reference_files_unchanged': all(before[p] == after[p] for p in before),
        'test_loop_seconds': stored.get('elapsed_seconds'),
        'wall_seconds': wall,
    }
    with open(os.path.join(out_dir, 'eval_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    m = metrics
    print(f"\n[{name}] total={m['total']} valid={m['valid']} correct={m['correct']} ValidRatio={m['valid_ratio']:.4f} "
          f"HR@1={m['hr_correct_over_valid']:.4f} correct/total={m['correct_over_total']:.4f}")
    if comparison and comparison.get('in_process_validation'):
        ip = comparison['in_process_validation']
        d = comparison['hr_difference_out_minus_in']
        print(f"  in-process validation of this epoch: {ip['total']}/{ip['valid']}/{ip['correct']} "
              f"HR={ip['correct'] / ip['valid']:.4f}; out-of-process minus in-process HR {d['difference']:+.4f} "
              f"(z={d['z']:.2f}, p={d['two_sided_p']:.3f}, unpaired)")
    elif comparison and 'candidates_identical_to_reference_prefix' in comparison:
        print(f"  candidates identical to the reference test.csv (first {m['total']} rows): "
              f"{comparison['candidates_identical_to_reference_prefix']}")
    print(f"  wrote {os.path.join(out_dir, 'eval_result.json')} ({wall:.0f}s)")
    if not result['reference_files_unchanged'] or not result['checkpoint']['unchanged_after_eval']:
        print('FAIL: a checkpoint or reference file changed during evaluation', file=sys.stderr)
        return 4
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt_mode', choices=sorted(base.PROMPT_MODES), default='current_hybrid')
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--bn_stats', choices=['checkpoint', 'recomputed'], default='checkpoint',
                        help='checkpoint: the statistics a --mode test run has (initial values, they are not saved)')
    parser.add_argument('--out_root', default=DEFAULT_OUT_ROOT, help='default %(default)s')
    parser.add_argument('--out_dir', help='new folder for this evaluation, overriding --out_root')
    parser.add_argument('--allow_code_difference', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    sys.exit(evaluate(parser.parse_args()))


if __name__ == '__main__':
    main()
