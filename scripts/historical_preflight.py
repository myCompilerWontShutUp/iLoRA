"""Checks on a historical snapshot (scripts/build_historical_snapshot.sh) before any training. CPU only;
no LLaMA weights are loaded and nothing is trained.

  1. provenance   snapshot HEAD and the two replaced files match HISTORICAL_SNAPSHOT_MANIFEST.json, and
                  nothing else in the snapshot is modified
  2. imports      data.data_interface, data.movielens_data and optims are imported from the snapshot,
                  and its collater is the 2024-08 one (`if p > thresh_hold:`)
  3. arguments    the run-A training command parsed by the snapshot's own main.py parser
  4. schedule     learning rate per parameter group that the snapshot's configure_optimizers and
                  optims.LinearWarmupCosineLRScheduler produce for run A, the optimizer steps per epoch,
                  and where max_steps stops training
  5. curriculum   the snapshot's train/val/test loaders run through the real DataLoader settings (8
                  spawned workers, per-epoch worker restart): share of hybrid batches per epoch, kind of
                  the last batch of each epoch (it decides whether the projector is in that epoch's
                  checkpoint), and the validation/test prompts. A text stub replaces the tokenizer (the
                  branch taken does not depend on it); the random draws are this simulation's, not a
                  training run's.
Writes runs/historical_preflight/<UTC time>/preflight.json in this repository.

    python scripts/historical_preflight.py --snapshot ../ilora_hist_2024_08
"""
import argparse
import ast
import datetime
import json
import math
import os
import random
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_A = {  # paper/arXiv-v1 settings, see experiments notes; lr floor = warmup start = peak/100 is inferred
    'mode': 'train', 'router': 'share', 'gating': 'Dense', 'batch_size': '8', 'accumulate_grad_batches': '16',
    'dataset': 'movielens_data', 'data_dir': 'data/ref/movielens', 'cans_num': '20', 'prompt_path': './prompt/movie.txt',
    'rec_embed': 'SASRec', 'llm_tuning': 'moelora', 'llm_path': '$LLM_PATH', 'rec_model_path': './rec_model/movielens.pt',
    'ckpt_dir': './checkpoints/hist_A/', 'output_dir': './output/hist_A/', 'log_dir': 'hist_A_logs',
    'lr_warmup_start_lr': '1e-6', 'lr': '1e-4', 'lr_decay_min_lr': '1e-6', 'max_epochs': '5', 'seed': '1234',
}
EXPECTED = {'lr': 1e-4, 'lr_warmup_start_lr': 1e-6, 'lr_decay_min_lr': 1e-6, 'max_epochs': 5, 'batch_size': 8,
            'accumulate_grad_batches': 16, 'precision': 16, 'seed': 1234, 'num_workers': 8, 'weight_decay': 1e-5,
            'lora_r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1, 'num_moe': 4, 'save': 'part', 'lr_scheduler': 'cosine'}


class TextTokenizer:
    """Stands in for LlamaTokenizer in the simulation: returns the prompt texts."""
    def __call__(self, texts, **kwargs):
        return {'text': [t[0] if isinstance(t, (list, tuple)) else t for t in texts]}


def command(llm_path):
    args = ' '.join(f'--{k} {v}' for k, v in RUN_A.items() if k != 'llm_path')
    return f'python main.py {args} --llm_path {llm_path}'


def check_provenance(snapshot):
    manifest = json.load(open(os.path.join(snapshot, 'HISTORICAL_SNAPSHOT_MANIFEST.json')))
    git = lambda *a: subprocess.check_output(['git', '-C', snapshot, *a]).decode()
    files = {f['path']: f['snapshot_blob'] for f in manifest['replaced_files']}
    # porcelain lines start with a meaningful space (' M path'), so the output is not stripped
    status = [line for line in git('status', '--porcelain', '--untracked-files=all').splitlines()
              if line and '__pycache__/' not in line and not line.endswith('.pyc')]
    expected = sorted([f' M {p}' for p in files] + ['?? HISTORICAL_SNAPSHOT_MANIFEST.json'])
    report = {'head': git('rev-parse', 'HEAD').strip(), 'base_commit': manifest['base_commit'],
              'replaced_files': {p: git('hash-object', p).strip() == blob for p, blob in files.items()},
              'other_changes': sorted(set(status) - set(expected))}
    report['ok'] = report['head'] == manifest['base_commit'] and all(report['replaced_files'].values()) \
        and not report['other_changes'] and sorted(status) == expected
    return report, manifest


def check_imports(snapshot):
    import data.data_interface as di
    import data.movielens_data as md
    import optims
    origin = {m.__name__: os.path.abspath(m.__file__) for m in (di, md, optims)}
    inside = all(path.startswith(os.path.abspath(snapshot) + os.sep) for path in origin.values())
    source = open(di.__file__, encoding='utf-8').read()
    return {'modules': origin, 'all_from_snapshot': inside,
            'collater_condition_2024_08': 'if p > thresh_hold:' in source and 'not self.train' not in source,
            'ok': inside and 'if p > thresh_hold:' in source}


def parse_run_a(snapshot):
    """Parse the run-A command with the snapshot main.py's own parser (only its __main__ block is run, with
    `main` replaced by a function that records the arguments)."""
    source = open(os.path.join(snapshot, 'main.py'), encoding='utf-8').read()
    block = source[source.index("if __name__ == '__main__':"):]
    captured = {}
    torch_stub = types.SimpleNamespace(multiprocessing=types.SimpleNamespace(set_start_method=lambda *a, **k: None))
    saved = sys.argv
    sys.argv = ['main.py'] + [x for k, v in RUN_A.items() for x in (f'--{k}', v)]
    try:
        exec(compile(block, 'main.py', 'exec'), {'__name__': '__main__', 'ArgumentParser': argparse.ArgumentParser,
                                                 'torch': torch_stub, 'main': lambda a: captured.update(vars(a))})
    finally:
        sys.argv = saved
    mismatches = {k: captured.get(k) for k, v in EXPECTED.items() if captured.get(k) != v}
    return captured, {'expected': EXPECTED, 'mismatches': mismatches, 'ok': not mismatches}


def schedule(snapshot, parsed, n_train):
    """configure_optimizers of the snapshot's MInterface, compiled from its source and run on a stand-in
    module whose projector/router/LLM are tiny layers; then the scheduler is stepped the way
    MInterface.training_step steps it (global step, current epoch, max_steps)."""
    import torch
    import optims
    path = os.path.join(snapshot, 'model', 'model_interface.py')
    tree = ast.parse(open(path, encoding='utf-8').read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MInterface')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_optimizers')
    namespace = {'torch': torch, 'LinearWarmupCosineLRScheduler': optims.LinearWarmupCosineLRScheduler}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), namespace)
    max_steps = n_train * parsed['max_epochs'] // (parsed['accumulate_grad_batches'] * parsed['batch_size'])  # main.py
    micro_batches = n_train // parsed['batch_size']                       # train loader: drop_last=True
    steps_per_epoch = math.ceil(micro_batches / parsed['accumulate_grad_batches'])  # Lightning steps on the last partial window
    module = types.SimpleNamespace(hparams=types.SimpleNamespace(**parsed), trainer=types.SimpleNamespace(max_steps=max_steps),
                                   projector=torch.nn.Linear(2, 2), router=torch.nn.Linear(2, 2), llama_model=torch.nn.Linear(2, 2))
    optimizer = namespace['configure_optimizers'](module)
    groups = [{'lr_at_build': g['lr'], 'weight_decay': g.get('weight_decay', 0)} for g in optimizer.param_groups]
    steps = sorted({0, 1, module.scheduler.warmup_steps // 2, module.scheduler.warmup_steps - 1, module.scheduler.warmup_steps,
                    steps_per_epoch, 2 * steps_per_epoch, max_steps // 2, max_steps - 1})
    table = []
    for step in steps:
        module.scheduler.step(step, step // steps_per_epoch, max_steps)
        table.append({'global_step': step, 'epoch': step // steps_per_epoch,
                      'lr_per_group': [g['lr'] for g in optimizer.param_groups]})
    last_epoch = max_steps // steps_per_epoch
    trained_in_last_epoch = (max_steps - last_epoch * steps_per_epoch) * parsed['accumulate_grad_batches']
    return {'parameter_groups': ['projector', 'router', 'llama_model (LoRA; gating excluded)'], 'groups': groups,
            'warmup_steps': module.scheduler.warmup_steps, 'max_steps': max_steps, 'micro_batches_per_epoch': micro_batches,
            'optimizer_steps_per_epoch': steps_per_epoch, 'lr_table': table,
            'stop': {'epoch': last_epoch, 'micro_batches_trained_in_it': trained_in_last_epoch, 'of': micro_batches,
                     'validation_of_that_epoch_runs': trained_in_last_epoch >= micro_batches}}


def curriculum(snapshot, epochs, seed):
    import torch
    from data.data_interface import DInterface
    random.seed(seed)
    torch.manual_seed(seed)
    dm = DInterface(llm_tokenizer=TextTokenizer(), num_workers=8, dataset='movielens_data', batch_size=8, max_epochs=5,
                    prompt_path=os.path.join(snapshot, 'prompt', 'movie.txt'),
                    data_dir=os.path.join(snapshot, 'data', 'ref', 'movielens'), cans_num=20)
    loader = dm.train_dataloader()
    report = {'collater_max_step': dm.max_steps, 'simulation_seed': seed, 'epochs': [], 'examples': {}}
    for epoch in range(epochs):
        flags = []
        for batch in loader:
            flags.append(bool(batch['flag']))
            kind = 'text_only_train' if batch['flag'] else 'hybrid_train'
            report['examples'].setdefault(kind, batch['tokens']['text'][0][:300])
        n = len(flags)
        report['epochs'].append({'batches': n, 'hybrid_share': 1 - sum(flags) / n,
                                 'hybrid_share_first_10pct': 1 - sum(flags[:n // 10]) / (n // 10),
                                 'hybrid_share_last_10pct': 1 - sum(flags[-(n // 10):]) / (n // 10),
                                 'last_batch_hybrid_so_projector_saved': not flags[-1]})
    for split, make in (('val', dm.val_dataloader), ('test', dm.test_dataloader)):
        texts = [t for batch in make() for t in batch['tokens']['text']]
        report[split] = {'samples': len(texts), 'all_text_only': all('[PH]' in t and '[HistoryEmb]' not in t
                                                                     and '[CansEmb]' not in t for t in texts),
                         'drop_last': make().drop_last}
        report['examples'][f'{split}_prompt'] = texts[0][:300]
    shares = [e['hybrid_share'] for e in report['epochs']]
    report['ok'] = all(0.85 < s < 0.95 for s in shares) and report['val']['all_text_only'] and report['test']['all_text_only']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--epochs', type=int, default=5, help='simulated training epochs')
    parser.add_argument('--seed', type=int, default=1234, help='seed of the simulation')
    args = parser.parse_args()
    snapshot = os.path.abspath(args.snapshot)
    os.chdir(snapshot)
    sys.path[0:1] = [snapshot]  # the snapshot replaces this script's folder; spawned workers inherit sys.path

    provenance, manifest = check_provenance(snapshot)
    imports = check_imports(snapshot)
    parsed, arguments = parse_run_a(snapshot)
    from data.movielens_data import MovielensData
    n_train = len(MovielensData(data_dir=os.path.join(snapshot, 'data', 'ref', 'movielens'), stage='train', cans_num=20))
    lr = schedule(snapshot, parsed, n_train)
    sim = curriculum(snapshot, args.epochs, args.seed)
    llm_path = os.environ.get('LLM_PATH', '$LLM_PATH')
    report = {'created_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), 'training_performed': False,
              'snapshot': snapshot, 'manifest': manifest, 'provenance': provenance, 'imports': imports,
              'arguments': dict(arguments, parsed=parsed), 'schedule': lr, 'curriculum': sim, 'train_samples': n_train,
              'run_A_command_not_executed': f'cd {snapshot} && {command(llm_path)}'}
    report['ok'] = all(part['ok'] for part in (provenance, imports, arguments, sim))
    out_dir = os.path.join(REPO, 'runs', 'historical_preflight', datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'))
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'preflight.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"provenance  {'OK' if provenance['ok'] else 'FAIL'}  head {provenance['head'][:7]}, replaced files "
          f"{provenance['replaced_files']}, other changes {provenance['other_changes']}")
    print(f"imports     {'OK' if imports['ok'] else 'FAIL'}  from snapshot: {imports['all_from_snapshot']}, "
          f"2024-08 collater: {imports['collater_condition_2024_08']}")
    print(f"arguments   {'OK' if arguments['ok'] else 'FAIL'}  mismatches {arguments['mismatches']}")
    print(f"schedule    warmup {lr['warmup_steps']} steps, max_steps {lr['max_steps']}, {lr['optimizer_steps_per_epoch']} "
          f"steps/epoch; training stops in epoch {lr['stop']['epoch']} after {lr['stop']['micro_batches_trained_in_it']}/"
          f"{lr['stop']['of']} micro-batches (validation of that epoch runs: {lr['stop']['validation_of_that_epoch_runs']})")
    for row in lr['lr_table']:
        print(f"            step {row['global_step']:5d} (epoch {row['epoch']}): " + ', '.join(f'{v:.3g}' for v in row['lr_per_group']))
    print(f"curriculum  {'OK' if sim['ok'] else 'FAIL'}  hybrid share per epoch "
          f"{[round(e['hybrid_share'], 3) for e in sim['epochs']]}, last batch hybrid (projector saved) "
          f"{[e['last_batch_hybrid_so_projector_saved'] for e in sim['epochs']]}; val {sim['val']['samples']} text-only "
          f"{sim['val']['all_text_only']}; test {sim['test']['samples']} text-only {sim['test']['all_text_only']} "
          f"(drop_last {sim['test']['drop_last']})")
    print(f"run A (not executed): cd {snapshot} && {command(llm_path)}")
    print(f"wrote {os.path.join(out_dir, 'preflight.json')}  overall {'OK' if report['ok'] else 'FAIL'}")
    sys.exit(0 if report['ok'] else 1)


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')  # as main.py
    main()
