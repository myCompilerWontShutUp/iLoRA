"""Regression check: main.py parses every main.py command of the run scripts into exactly the same
args as before the measurement flags were added (commit 6d20c21), with and without --measure.

Only main.py's `if __name__ == '__main__':` block is executed, with `main` replaced by a function
that records its arguments, so neither transformers nor a GPU is needed.

    python scripts/test_main_args_unchanged.py
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = '6d20c21'   # last main.py before the --measure flags
SCRIPTS = ('run_baseline_movielens.sh', 'run_cluster_hard_movielens.sh', 'bootstrap_vessl.sh',
           'train_movielens.sh', 'test_movielens.sh')
SHELL_VALUES = {'SEED': '1234', 'N_CLUSTERS': '4'}
sys.path.insert(0, REPO)


def parse_with(source, argv):
    block = source[source.index("if __name__ == '__main__':"):]
    captured = {}

    def fake_main(args, measure_args=None):
        captured.update(args=vars(args).copy(), measure=measure_args)
    torch_stub = types.SimpleNamespace(multiprocessing=types.SimpleNamespace(set_start_method=lambda *a, **k: None))
    saved = sys.argv
    sys.argv = ['main.py'] + argv
    try:
        exec(compile(block, 'main.py', 'exec'),
             {'__name__': '__main__', 'ArgumentParser': argparse.ArgumentParser, 'torch': torch_stub, 'main': fake_main})
    finally:
        sys.argv = saved
    return captured


def main_commands(path):
    lines = open(os.path.join(REPO, path), encoding='utf-8').read().splitlines()
    out = []
    for i, line in enumerate(lines):
        if re.match(r'\s*python main\.py', line):
            parts = [line]
            while parts[-1].rstrip().endswith('\\'):
                parts.append(lines[i + len(parts)])
            cmd = ' '.join(p.rstrip().rstrip('\\') for p in parts)
            cmd = re.sub(r'"?\$\{?(\w+)\}?"?', lambda m: SHELL_VALUES.get(m.group(1), f'VAL_{m.group(1)}'), cmd)
            out.append(shlex.split(cmd)[2:])
    return out


def main():
    old_src = subprocess.check_output(['git', '-C', REPO, 'show', f'{REFERENCE}:main.py']).decode('utf-8')
    new_src = open(os.path.join(REPO, 'main.py'), encoding='utf-8').read()
    failures, total = 0, 0
    for script in SCRIPTS:
        for argv in main_commands(script):
            total += 1
            old, new = parse_with(old_src, argv), parse_with(new_src, argv)
            measured = parse_with(new_src, argv + ['--measure', '--stats_every', '4'])
            ok = (old['args'] == new['args'] and not new['measure'].measure
                  and measured['args'] == old['args'] and measured['measure'].measure)
            failures += not ok
            print(f"  [{'OK' if ok else 'FAIL'}] {script}: --mode {new['args']['mode']} --routing_mode "
                  f"{new['args']['routing_mode']} --lr {new['args']['lr']}: {len(new['args'])} args")
    if failures or not total:
        print(f'{failures} of {total} command(s) differ', file=sys.stderr)
        sys.exit(1)
    print(f'{total} main.py commands parse to the same args as {REFERENCE}, with and without --measure.')


if __name__ == '__main__':
    main()
