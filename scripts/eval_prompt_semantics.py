"""Re-evaluate an already trained checkpoint under two test-prompt semantics. Nothing is trained.

The evaluation is main.py's own `--mode test` path, run in this process with runpy: the exact
`python main.py --mode test ...` command of run_baseline_movielens.sh, with only --ckpt_path and the
output locations (--output_dir, --results_dir, --log_dir) replaced. Seeding, model construction,
DataLoader worker seeding, candidate sampling, generation, parsing and calculate_hr1 are therefore
the ones that produced results/baseline/metrics.json. `--mode test` only calls trainer.test
(main.py:79-82): no optimizer is built, nothing is back-propagated and no checkpoint is written.

--prompt_mode
  current_hybrid      data/data_interface.py of this checkout (upstream 65e2fcd, 2024-11-13, and
                      later): validation/test batches always get the hybrid prompt, whose
                      [HistoryEmb]/[CansEmb] positions are filled with projector outputs.
  aug2024_text_only   the test collate_fn is upstream's TrainCollater from blob b69479a0, the
                      data/data_interface.py introduced by 9f6c078 (2024-08-10) and replaced by
                      65e2fcd, written byte for byte from git history and imported unchanged. Its
                      branch condition is `p > cur_step / max_step` with max_step=1 for test, which
                      is never true, so every test batch gets the text-only prompt ([PH]
                      placeholders; the projector output is computed but never inserted). It draws
                      the same random numbers as the current collater (one random.choice and one
                      random.random per batch), so candidate sets and prompt templates are the same
                      in both modes.
  Everything else (dataset, candidate sampling, DataLoader settings, model, generation, parsing,
  metric) is the current code in both modes. aug2024_text_only drops --export_gates: the
  historical batch has no sample_idx, and gates do not depend on the prompt anyway.

Safety: the output folder must not exist; the checkpoint and the reference results are hashed
before and after; the code must be identical to the commit that trained the checkpoint.

Relative paths given to this script are resolved against the repository root, whatever the
current directory is.

    python scripts/eval_prompt_semantics.py --prompt_mode current_hybrid \\
        --ckpt checkpoints/baseline/last.ckpt --expect_reference
    python scripts/eval_prompt_semantics.py --prompt_mode aug2024_text_only --ckpt checkpoints/baseline/last.ckpt
    python scripts/eval_prompt_semantics.py --summarize runs/semantics_eval_v2
"""
import argparse
import ast
import datetime
import hashlib
import importlib
import inspect
import json
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `python scripts/eval_prompt_semantics.py` puts scripts/ first on sys.path. The repository packages
# (imported before main.py runs, e.g. data.data_interface for aug2024_text_only) and main.py itself
# need the repository root there instead, exactly as for `python main.py`.
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path[0] = REPO
elif REPO not in sys.path:
    sys.path.insert(0, REPO)
CHECKPOINT_COMMIT = '6c7ebaceac99a97fa868ac7590a786d489b88c37'  # trained and tested results/baseline
UPSTREAM_BASE = '28e47371192121b6ced44401d9e149fedfae13de'      # AkaliKong/iLoRA main the fork is based on
HISTORICAL = {
    'repository': 'https://github.com/AkaliKong/iLoRA',
    'path': 'data/data_interface.py',
    'blob': 'b69479a07433489ac583b79c40930500be802960',
    'introduced_by': '9f6c0781bf65c9ec6a148471169fbceb9b6ecb24',  # 2024-08-10
    'replaced_by': '65e2fcdee09b3a748d272615aad552a2a01c26f4',    # 2024-11-13
    'class': 'TrainCollater',
}
HIST_MODULE = 'ilora_upstream_b69479a_data_interface'
PROMPT_MODES = {
    'current_hybrid': 'test prompt of data/data_interface.py at this checkout (upstream 65e2fcd and later): '
                      'always hybrid, projector outputs at [HistoryEmb]/[CansEmb]',
    'aug2024_text_only': 'test prompt of upstream data/data_interface.py blob b69479a0 (9f6c078, 2024-08-10, '
                         'before 65e2fcd): always text-only, [PH] placeholders, projector output not inserted',
}
REFERENCE_METRICS = 'results/baseline/metrics.json'
REFERENCE_TEST_CSV = 'outputs/baseline/test.csv'
REFERENCE_COUNTS = [95, 86, 30]  # total, valid, correct of the 6c7ebac baseline test (VR 0.9052631579, HR 0.3488372093)
EXPORT_GATES_NOTE = ('--export_gates removed: the historical collater builds the old batch schema without sample_idx, '
                     'which gate export reads; gates depend only on the SASRec user representation, not on the prompt')
DATALOADER_CHECK = {}
# read only for this script; runs/semantics_eval holds the results of the first version (11dc9e3)
PROTECTED = ('checkpoints', 'results/baseline', 'outputs/baseline', 'archives', 'runs/semantics_eval')
DEFAULT_OUT_ROOT = 'runs/semantics_eval_v2'
OWN_FILES = ('scripts/eval_prompt_semantics.py',)
IMPORT_DIRS = ('model', 'data', 'recommender', 'measurement')
UPSTREAM_TEST_N = 88  # upstream test_dataloader: drop_last=True with batch 8 keeps the first 88 of 95


def repo_path(path):
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(REPO, path))


def protected_by(path):
    for protected in PROTECTED:
        root = repo_path(protected)  # abspath normalizes separators, so the comparison also holds on Windows
        try:
            if os.path.commonpath([os.path.abspath(path), root]) == root:
                return protected
        except ValueError:  # different drives
            pass
    return None


def git(*args, binary=False):
    out = subprocess.check_output(['git', '-C', REPO, *args])
    return out if binary else out.decode().strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def affects_evaluation(path):
    """Sources, scripts and inputs of the run. Upstream tracks __pycache__/*.pyc files, which Python
    rewrites on import, and IDE/documentation files; neither changes what is computed."""
    if '__pycache__/' in path or path.endswith('.pyc'):
        return False
    return (path.endswith(('.py', '.sh')) or path == 'requirements.txt'
            or path.startswith(('prompt/', 'data/ref/', 'rec_model/', 'debug/')))


def code_identity():
    """The evaluated code and inputs must be byte-identical to the commit that trained the checkpoint."""
    changed = [p for p in git('diff', '--name-only', CHECKPOINT_COMMIT, '--').splitlines()
               if affects_evaluation(p) and p not in OWN_FILES]
    untracked = [p for p in git('ls-files', '--others', '--exclude-standard').splitlines()
                 if p.endswith('.py') and ('/' not in p or p.split('/')[0] in IMPORT_DIRS)]
    return {'eval_head': git('rev-parse', 'HEAD'), 'checkpoint_commit': CHECKPOINT_COMMIT,
            'upstream_base': UPSTREAM_BASE, 'files_differing_from_checkpoint_commit': changed,
            'untracked_py_in_import_paths': untracked, 'baseline_code_identical': not changed and not untracked}


def transformers_patches():
    """README's debug/*.py replacements, compared with the files of the installed transformers."""
    import transformers
    root = os.path.dirname(transformers.__file__)
    pairs = {'debug/utils.py': 'generation/utils.py', 'debug/modeling_llama.py': 'models/llama/modeling_llama.py'}
    return {'transformers_version': transformers.__version__,
            **{dst: sha256(os.path.join(REPO, src)) == sha256(os.path.join(root, dst)) for src, dst in pairs.items()}}


def baseline_test_argv():
    """main.py arguments of run_baseline_movielens.sh's evaluation step, with the script's own values."""
    text = open(os.path.join(REPO, 'run_baseline_movielens.sh'), encoding='utf-8').read()
    values = dict(re.findall(r'^([A-Z_]+)="([^"$]*)"$', text, flags=re.M))
    values['SEED'] = os.environ.get('SEED', '1234')
    if not os.environ.get('LLM_PATH'):
        raise SystemExit('LLM_PATH must be set, as for run_baseline_movielens.sh')
    values['LLM_PATH'] = os.environ['LLM_PATH']
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r'\s*python main\.py', line):
            parts = [line]
            while parts[-1].rstrip().endswith('\\'):
                parts.append(lines[i + len(parts)])
            command = ' '.join(p.rstrip().rstrip('\\') for p in parts)
            if '--mode test' in command:
                command = re.sub(r'\$\{?(\w+)\}?', lambda m: values[m.group(1)], command)
                return shlex.split(command)[2:]
    raise SystemExit('run_baseline_movielens.sh has no `python main.py --mode test` command')


def set_arg(argv, flag, value):
    argv[argv.index(flag) + 1] = value


def checkpoint_info(path):
    import torch
    state = torch.load(path, map_location='cpu')['state_dict']
    name = os.path.basename(path)
    epoch, metric = re.search(r'epoch=(\d+)', name), re.search(r'metric=(\d+\.\d+)', name)
    buffers = [k for k in state if k.startswith('router.') and ('running_' in k or 'num_batches' in k)]
    digest, folder = sha256(path), os.path.dirname(os.path.abspath(path))
    identical = [n for n in sorted(os.listdir(folder)) if n != name and n.endswith('.ckpt')
                 and os.path.getsize(os.path.join(folder, n)) == os.path.getsize(path)
                 and sha256(os.path.join(folder, n)) == digest]
    # last.ckpt carries no metric in its name; it is a byte copy of the last epoch's checkpoint
    named = [m for m in [metric] + [re.search(r'metric=(\d+\.\d+)', n) for n in identical] if m]
    info = {'path': os.path.relpath(os.path.abspath(path), REPO), 'file': name, 'sha256': digest,
            'identical_checkpoints': identical,
            'epoch_from_filename': int(epoch.group(1)) if epoch else None,
            'val_metric_from_filename': float(named[0].group(1)) if named else None,
            'state_dict_keys': len(state), 'lora_keys': sum('.lora_A.' in k or '.lora_B.' in k for k in state),
            'projector_keys': sum(k.startswith('projector.') for k in state),
            'router_parameter_keys': sum(k.startswith('router.') for k in state) - len(buffers),
            'router_batchnorm_buffers': len(buffers)}
    del state
    return info


def materialize_historical(out_dir):
    """Write upstream's pre-65e2fcd data/data_interface.py, byte for byte, as an importable module."""
    for rev in (HISTORICAL['introduced_by'], HISTORICAL['replaced_by'] + '^'):
        blob = git('rev-parse', f"{rev}:{HISTORICAL['path']}")
        if blob != HISTORICAL['blob']:
            raise SystemExit(f"{rev}:{HISTORICAL['path']} is blob {blob}, expected {HISTORICAL['blob']}")
    data = git('cat-file', 'blob', HISTORICAL['blob'], binary=True)
    if hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() != HISTORICAL['blob']:
        raise SystemExit('historical blob content does not hash to its id')
    folder = os.path.join(out_dir, 'historical')
    os.makedirs(folder)
    with open(os.path.join(folder, HIST_MODULE + '.py'), 'wb') as f:
        f.write(data)
    return folder


def use_historical_test_collater(folder):
    """DInterface.test_dataloader with the historical TrainCollater as collate_fn, nothing else changed.
    Spawned DataLoader workers inherit sys.path and import the module by name."""
    sys.path.insert(0, folder)
    historical = importlib.import_module(HIST_MODULE)
    from torch.utils.data import DataLoader
    import data.data_interface as current
    current_test_dataloader = current.DInterface.test_dataloader

    def test_dataloader(self):
        reference = current_test_dataloader(self)
        loader = DataLoader(self.testset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            collate_fn=historical.TrainCollater(prompt_list=self.prompt_list,
                                                                llm_tokenizer=self.llm_tokenizer, train=False))
        for name in ('dataset', 'batch_size', 'num_workers', 'drop_last', 'pin_memory', 'timeout',
                     'prefetch_factor', 'persistent_workers', 'generator'):
            if getattr(loader, name) is not getattr(reference, name) and getattr(loader, name) != getattr(reference, name):
                raise RuntimeError(f'historical test DataLoader differs from the current one in {name}')
        if type(loader.sampler) is not type(reference.sampler):
            raise RuntimeError('historical test DataLoader uses a different sampler')
        DATALOADER_CHECK.update(identical_to_current_except_collate_fn=True, sampler=type(loader.sampler).__name__,
                                batch_size=loader.batch_size, num_workers=loader.num_workers, drop_last=loader.drop_last)
        return loader

    current.DInterface.test_dataloader = test_dataloader


def run_main(argv):
    """Execute main.py as `python main.py <argv>` would (the repository root is first on sys.path, see top)."""
    os.chdir(REPO)
    saved = sys.argv
    sys.argv = ['main.py'] + argv
    try:
        runpy.run_path(os.path.join(REPO, 'main.py'), run_name='__main__')
    finally:
        sys.argv = saved


def read_test_csv(path):
    import pandas as pd
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    return {'generate': list(frame['generate']), 'real': list(frame['real']),
            'cans': [ast.literal_eval(c) for c in frame['cans']]}


def score(content):
    """Counts and ratios with MInterface.calculate_hr1 itself."""
    from model.model_interface import MInterface
    valid_ratio, hr = MInterface.calculate_hr1(None, content)
    total = len(content['generate'])
    valid = round(valid_ratio * total)
    correct = round(hr * valid)
    return {'total': total, 'valid': valid, 'correct': correct, 'valid_ratio': valid_ratio,
            'hr_correct_over_valid': hr, 'correct_over_total': correct / total}


def compare_with_reference(content, metrics):
    reference = read_test_csv(os.path.join(REPO, REFERENCE_TEST_CSV))
    stored = json.load(open(os.path.join(REPO, REFERENCE_METRICS)))
    same_rows = len(reference['real']) == len(content['real'])
    return {
        'reference_metrics': REFERENCE_METRICS, 'reference_test_csv': REFERENCE_TEST_CSV,
        'reference_counts': [stored['test_total_num'], stored['test_valid_num'], stored['test_correct_num']],
        'reference_counts_are_the_audited_baseline': [stored['test_total_num'], stored['test_valid_num'],
                                                      stored['test_correct_num']] == REFERENCE_COUNTS,
        'reference_valid_ratio': stored['test_prediction_valid'], 'reference_hr': stored['test_hr'],
        'metrics_identical': ([metrics['total'], metrics['valid'], metrics['correct']] ==
                              [stored['test_total_num'], stored['test_valid_num'], stored['test_correct_num']]
                              and abs(metrics['valid_ratio'] - stored['test_prediction_valid']) < 1e-12
                              and abs(metrics['hr_correct_over_valid'] - stored['test_hr']) < 1e-12),
        'candidates_identical': same_rows and reference['real'] == content['real'] and reference['cans'] == content['cans'],
        'generations_differing': [i for i, (a, b) in enumerate(zip(reference['generate'], content['generate'])) if a != b]
                                 if same_rows else 'row count differs',
    }


def evaluate(args):
    ckpt = repo_path(args.ckpt)
    out_dir = repo_path(args.out_dir) if args.out_dir else os.path.join(
        repo_path(args.out_root), f"{os.path.splitext(os.path.basename(ckpt))[0]}__{args.prompt_mode}")
    if protected_by(out_dir):
        raise SystemExit(f'{out_dir} is inside {protected_by(out_dir)}/, which this script never writes to')
    if os.path.exists(out_dir):
        raise SystemExit(f'{out_dir} exists; evaluation folders are never overwritten')
    if not os.path.isfile(ckpt):
        raise SystemExit(f'{ckpt} not found')

    identity = code_identity()
    if not identity['baseline_code_identical'] and not args.allow_code_difference:
        raise SystemExit(f'code differs from {CHECKPOINT_COMMIT}: {identity}')
    argv = baseline_test_argv()
    set_arg(argv, '--ckpt_path', ckpt)
    set_arg(argv, '--output_dir', os.path.join(out_dir, 'output') + '/')
    set_arg(argv, '--results_dir', os.path.join(out_dir, 'results'))
    set_arg(argv, '--log_dir', 'semantics_eval_logs')
    if args.prompt_mode == 'aug2024_text_only':
        argv.remove('--export_gates')
    if args.dry_run:
        report = {'out_dir': out_dir, 'code': identity, 'main_py_argv': argv}
        if args.prompt_mode == 'aug2024_text_only':
            # check the historical blob and its import in a throw-away folder; out_dir is not created
            scratch = tempfile.mkdtemp(prefix='eval_prompt_semantics_')
            try:
                use_historical_test_collater(materialize_historical(scratch))
                report['historical_collater'] = dict(HISTORICAL, blob_verified=True, imported=HIST_MODULE)
            finally:
                shutil.rmtree(scratch)
        print(json.dumps(report, indent=2))
        return 0

    watched = [ckpt] + [os.path.join(REPO, p) for p in (REFERENCE_METRICS, REFERENCE_TEST_CSV)]
    before = {p: sha256(p) for p in watched if os.path.exists(p)}
    info = checkpoint_info(ckpt)
    os.makedirs(out_dir)
    if args.prompt_mode == 'aug2024_text_only':
        use_historical_test_collater(materialize_historical(out_dir))

    start = time.time()
    run_main(argv)
    wall = time.time() - start

    from model.model_interface import MInterface
    stored = json.load(open(os.path.join(out_dir, 'results', 'metrics.json')))
    content = read_test_csv(os.path.join(out_dir, 'output', 'test.csv'))
    metrics = score(content)
    if [metrics['total'], metrics['valid'], metrics['correct']] != \
            [stored['test_total_num'], stored['test_valid_num'], stored['test_correct_num']]:
        raise SystemExit(f'test.csv rescoring {metrics} disagrees with the in-process metrics.json {stored}')
    after = {p: sha256(p) for p in before}
    reference = compare_with_reference(content, metrics) if os.path.exists(os.path.join(REPO, REFERENCE_TEST_CSV)) else None
    generation = {k: v.default for k, v in inspect.signature(MInterface.generate).parameters.items()
                  if v.default is not inspect.Parameter.empty and k != 'split'}
    result = {
        'created_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'training_performed': False,
        'code': dict(identity, installed_patches_match_debug=transformers_patches()),
        'checkpoint': dict(info, unchanged_after_eval=after[ckpt] == before[ckpt] == info['sha256']),
        'prompt_mode': {'name': args.prompt_mode, 'description': PROMPT_MODES[args.prompt_mode],
                        'test_collater': dict(HISTORICAL) if args.prompt_mode == 'aug2024_text_only'
                        else 'data/data_interface.py TrainCollater of this checkout',
                        'argv_difference_from_current_hybrid': [EXPORT_GATES_NOTE]
                        if args.prompt_mode == 'aug2024_text_only' else [],
                        'test_dataloader_check': DATALOADER_CHECK or 'current DInterface.test_dataloader, unchanged'},
        'main_py_argv': argv,
        'generation': dict(generation, decoding='greedy' if not generation['do_sample'] and generation['num_beams'] == 1
                           else 'see fields', tokenizer_padding_side='right', parsed_output='first line of the generation'),
        'evaluation': {'valid': 'exactly one candidate title is a substring of the stripped, lower-cased output',
                       'correct': 'that single candidate equals the target title',
                       'hr': 'correct/valid (calculate_hr1)', 'test_dataloader': 'batch_size from argv, shuffle=False, '
                       'drop_last=False (fork babe9ec; upstream drops the last 7 of 95)',
                       'candidates': 'resampled by MovielensData.__getitem__ from seeded DataLoader workers'},
        'known_caveats': [
            'router BatchNorm running statistics are never saved by MInterface.on_save_checkpoint, so the router '
            'runs with initial statistics in every --mode test evaluation, including the original one',
            'if projector_keys == 0 the hybrid prompt is filled by an untrained projector; the text-only prompt '
            'does not use the projector'],
        'metrics': {'all': metrics, f'first_{UPSTREAM_TEST_N}': score({k: v[:UPSTREAM_TEST_N] for k, v in content.items()})},
        'reference_check': reference,
        'reference_files_unchanged': all(before[p] == after[p] for p in before),
        'test_loop_seconds': stored.get('elapsed_seconds'),
        'wall_seconds': wall,
    }
    with open(os.path.join(out_dir, 'eval_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    m = metrics
    print(f"\n[{args.prompt_mode}] {info['file']}: total={m['total']} valid={m['valid']} correct={m['correct']} "
          f"ValidRatio={m['valid_ratio']:.6f} HR@1={m['hr_correct_over_valid']:.6f} correct/total={m['correct_over_total']:.6f}")
    print(f"  wrote {os.path.join(out_dir, 'eval_result.json')} ({wall:.0f}s)")
    if not result['reference_files_unchanged'] or not result['checkpoint']['unchanged_after_eval']:
        print('FAIL: a checkpoint or reference file changed during evaluation', file=sys.stderr)
        return 4
    if args.expect_reference:
        if not (reference and reference['metrics_identical'] and reference['candidates_identical']
                and reference['reference_counts_are_the_audited_baseline']):
            print(f'SANITY FAIL: the reference result is not reproduced: {reference}', file=sys.stderr)
            return 3
        print(f"  SANITY PASS: {REFERENCE_METRICS} reproduced (counts, ratios and all candidate sets); "
              f"generations differing: {reference['generations_differing']}")
    return 0


def validation_metrics(ckpt_dir):
    """Validation metric in the file name of every epoch checkpoint the training run saved."""
    found = {}
    for name in sorted(os.listdir(ckpt_dir)):
        match = re.fullmatch(r'epoch=(\d+)-metric=(\d+\.\d+)\.ckpt', name)
        if match:
            found[name] = float(match.group(2))
    return found


def summarize(root, ckpt_dir):
    root, ckpt_dir = repo_path(root), repo_path(ckpt_dir)
    results = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, 'eval_result.json')
        if os.path.exists(path):
            results.append(json.load(open(path)))
    table = {}
    for r in results:
        table.setdefault(r['checkpoint']['file'], {'checkpoint': r['checkpoint']})[r['prompt_mode']['name']] = r
    metric_of = {n: row['checkpoint']['val_metric_from_filename'] for n, row in table.items()}
    # best over all saved epoch checkpoints, evaluated here or not; ties are all marked
    saved = validation_metrics(ckpt_dir)
    top = max(saved.values(), default=None)
    best = {n for n, v in saved.items() if v == top}

    def cell(row, mode, subset='all'):
        if mode not in row:
            return 'not run'
        m = row[mode]['metrics'][subset]
        return (f"{m['total']}/{m['valid']}/{m['correct']} VR={m['valid_ratio']:.4f} "
                f"HR={m['hr_correct_over_valid']:.4f} c/N={m['correct_over_total']:.4f}")

    def status(name):
        if name in table:
            return 'evaluated'
        aliases = [n for n, row in table.items() if name in row['checkpoint']['identical_checkpoints']]
        return f"evaluated as {', '.join(aliases)} (same bytes)" if aliases else 'not evaluated in this folder'

    lines = ['best validation checkpoint: '
             + ('; '.join(f'{n} (val metric {saved[n]}, {status(n)})' for n in sorted(best)) or 'none found'),
             f'chosen by the validation metric in the file names of all epoch=*-metric=*.ckpt in '
             f'{os.path.relpath(ckpt_dir, REPO) if ckpt_dir.startswith(REPO + os.sep) else ckpt_dir}/, '
             'never by a test result:',
             ', '.join(f'{n} {v}' for n, v in saved.items()) or 'none found', '',
             '| checkpoint | val metric (filename) | projector / LoRA / router BN-buffer keys | '
             'current-hybrid total/valid/correct | 2024-08-text-only total/valid/correct |', '|---|---|---|---|---|']
    for name in sorted(table, key=lambda n: (metric_of[n] is None, n)):
        row, ck = table[name], table[name]['checkpoint']
        same = ck['identical_checkpoints']
        # only an epoch file named in `best` is marked; a byte copy such as last.ckpt is shown as "(= ...)"
        label = name + (' (best val)' if name in best else '') + (f" (= {', '.join(same)})" if same else '')
        lines.append(f"| {label} | {ck['val_metric_from_filename']} | "
                     f"{ck['projector_keys']} / {ck['lora_keys']} / {ck['router_batchnorm_buffers']} | "
                     f"{cell(row, 'current_hybrid')} | {cell(row, 'aug2024_text_only')} |")
    lines += ['', f'first {UPSTREAM_TEST_N} test sequences (upstream drop_last=True, batch 8):', '',
              '| checkpoint | current-hybrid | 2024-08-text-only |', '|---|---|---|']
    for name in sorted(table, key=lambda n: (metric_of[n] is None, n)):
        lines.append(f"| {name} | {cell(table[name], 'current_hybrid', f'first_{UPSTREAM_TEST_N}')} | "
                     f"{cell(table[name], 'aug2024_text_only', f'first_{UPSTREAM_TEST_N}')} |")
    text = '\n'.join(lines)
    print(text)
    if protected_by(root):
        print(f'(summary.md not written: {root} is inside {protected_by(root)}/, which this script never writes to)')
    else:
        with open(os.path.join(root, 'summary.md'), 'w') as f:
            f.write(text + '\n')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt_mode', choices=sorted(PROMPT_MODES))
    parser.add_argument('--ckpt')
    parser.add_argument('--out_root', default=DEFAULT_OUT_ROOT,
                        help='folder that receives <checkpoint>__<prompt_mode>/ (default %(default)s)')
    parser.add_argument('--out_dir', help='new folder for this evaluation, overriding --out_root')
    parser.add_argument('--expect_reference', action='store_true',
                        help=f'fail unless {REFERENCE_METRICS} and the candidate sets of {REFERENCE_TEST_CSV} are reproduced')
    parser.add_argument('--allow_code_difference', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='print the main.py arguments and code check only')
    parser.add_argument('--summarize', metavar='ROOT')
    parser.add_argument('--ckpt_dir', default='checkpoints/baseline',
                        help='--summarize: folder whose epoch=*-metric=*.ckpt file names define the best validation epoch')
    args = parser.parse_args()
    if args.summarize:
        sys.exit(summarize(args.summarize, args.ckpt_dir))
    if not args.prompt_mode or not args.ckpt:
        parser.error('--prompt_mode and --ckpt are required')
    sys.exit(evaluate(args))


if __name__ == '__main__':
    main()
