"""Environment check right before training on VESSL. Prints one line per check (OK / WARN / FAIL)
and exits 1 if any check fails, so a run script calling it never starts training on a broken
environment. The report is also saved as logs/env_check_<UTC time>.json.

Checks: git commit / branch, Python executable and version inside the `ilora` conda env, packages
imported from the env (not ~/.local), torch 2.0.0 / torch.version.cuda 11.7 / CUDA available, GPU
name and VRAM, the libcudart torch loaded, the bitsandbytes binary actually loaded
(scripts/verify_bnb_cuda.py), transformers 4.28.0 and its source patches, pytorch_lightning 1.8.6,
LLM_PATH weights, MovieLens data, free disk, writable output folders, tmux (runs, and does not load
conda's libtinfo), and whether the repository / weights / env sit on the container filesystem that
a workspace Pause deletes.

    python scripts/env_check.py                    # everything (VESSL, env activated)
    python scripts/env_check.py --storage-only     # storage warning only; stdlib, any Python
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = {'python': '3.10', 'env': 'ilora', 'torch': '2.0.0', 'torch_cuda': '11.7', 'transformers': '4.28.0',
            'pytorch_lightning': '1.8.6', 'branch': 'exp/gate-cluster'}
DATASET_FILES = ['data/ref/movielens/train_data.df', 'data/ref/movielens/Val_data.df',
                 'data/ref/movielens/Test_data.df', 'data/ref/movielens/u.item', 'rec_model/movielens.pt']
WRITABLE_DIRS = ['logs', 'log', 'checkpoints', 'outputs', 'results', 'runs', 'benchmark_runs', 'archives']
EPHEMERAL_TYPES = {'overlay', 'overlayfs', 'tmpfs', 'aufs'}
PSEUDO_TYPES = {'proc', 'sysfs', 'devtmpfs', 'devpts', 'mqueue', 'cgroup', 'cgroup2', 'securityfs', 'debugfs',
                'pstore', 'bpf', 'tracefs', 'configfs', 'fusectl', 'hugetlbfs', 'autofs', 'binfmt_misc', 'nsfs',
                'shm', 'tmpfs'}
GiB = 1024 ** 3


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail):
        self.rows.append({'status': status, 'check': name, 'detail': detail})
        print(f'  [{status:<4}] {name}: {detail}', flush=True)

    def ok(self, name, detail):
        self.add('OK', name, detail)

    def warn(self, name, detail):
        self.add('WARN', name, detail)

    def fail(self, name, detail):
        self.add('FAIL', name, detail)

    def count(self, status):
        return sum(r['status'] == status for r in self.rows)


def run(cmd, timeout=60, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, '', str(e)


# ---- pure helpers (unit-tested by scripts/test_env_scripts.sh) ----------------------------------
def parse_df(output):
    """`df -PT <path>` output -> dict(filesystem, type, avail_bytes, mount) of its last line."""
    lines = [l for l in output.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    parts = lines[-1].split()
    if len(parts) < 7:
        return None
    return {'filesystem': parts[0], 'type': parts[1], 'avail_bytes': int(parts[4]) * 1024,
            'mount': ' '.join(parts[6:])}


def is_ephemeral(info):
    """Container filesystem: overlay-type root, or simply mounted at '/'."""
    return info is not None and (info['type'].lower() in EPHEMERAL_TYPES or info['mount'] == '/')


def ncurses_dirs(ld_library_path):
    """LD_LIBRARY_PATH entries holding libtinfo / libncurses: system binaries such as tmux would load
    them instead of the system copies (the 'NCURSES6_TINFO_6.4.current not found' failure)."""
    hits = []
    for d in [p for p in ld_library_path.split(':') if p]:
        if glob.glob(os.path.join(d, 'libtinfo.so*')) or glob.glob(os.path.join(d, 'libncurses*.so*')):
            hits.append(d)
    return hits


def mount_of(path):
    code, out, _ = run(['df', '-PT', path], timeout=20)
    return parse_df(out) if code == 0 else None


# ---- checks -------------------------------------------------------------------------------------
def check_storage(r):
    paths = {'repository': REPO_ROOT, 'LLM_PATH': os.environ.get('LLM_PATH', ''),
             'conda env': os.environ.get('CONDA_PREFIX', '')}
    ephemeral = []
    for label, path in paths.items():
        if not path or not os.path.exists(path):
            continue
        info = mount_of(path)
        if info is None:
            r.warn(f'storage {label}', f'{path}: could not determine the filesystem (df -PT failed)')
        elif is_ephemeral(info):
            ephemeral.append(label)
            r.warn(f'storage {label}', f"{path} is on the container filesystem ({info['type']} at {info['mount']}): "
                                       'a workspace Pause/Stop deletes it')
        else:
            r.ok(f'storage {label}', f"{path} is on a separate mount ({info['type']} at {info['mount']}); "
                                     'confirm in VESSL that it is a persistent volume')
    code, out, _ = run(['df', '-PT'], timeout=20)
    if code == 0:
        mounts = []
        for line in out.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 7 and parts[1].lower() not in PSEUDO_TYPES | EPHEMERAL_TYPES and parts[6] != '/':
                mounts.append(f"{' '.join(parts[6:])} ({parts[1]}, {int(parts[4]) * 1024 / GiB:.0f} GiB free)")
        r.add('INFO', 'other mounts', ', '.join(mounts) if mounts else 'none besides the container root '
              '(/opt/vessl, if listed, is VESSL\'s own mount, not a data volume)')
    if ephemeral:
        print('\n  !!! Results, logs, checkpoints and run folders on the container filesystem are lost on Pause/Stop.\n'
              '  !!! Before pausing: bash scripts/archive_results.sh, then download archives/<file>.tgz\n'
              '  !!! (JupyterLab file browser or scp), or attach a persistent volume and work there.\n', flush=True)


def check_git(r):
    code, commit, _ = run(['git', '-C', REPO_ROOT, 'rev-parse', 'HEAD'])
    _, branch, _ = run(['git', '-C', REPO_ROOT, 'rev-parse', '--abbrev-ref', 'HEAD'])
    _, status, _ = run(['git', '-C', REPO_ROOT, 'status', '--porcelain', '--untracked-files=no'])
    if code != 0:
        return r.warn('git', 'not a git checkout')
    branch = branch.strip()
    detail = f'{commit.strip()[:12]} on {branch}' + (f', {len(status.splitlines())} modified tracked file(s)'
                                                     if status.strip() else ', no modified tracked files')
    (r.ok if branch == EXPECTED['branch'] and not status.strip() else r.warn)('git', detail)


def check_python(r):
    prefix = os.environ.get('CONDA_PREFIX', '')
    env = os.environ.get('CONDA_DEFAULT_ENV', '')
    version = '%d.%d.%d' % sys.version_info[:3]
    if env != EXPECTED['env'] or not prefix:
        r.fail('conda env', f"active env is '{env or 'none'}'; run: source \"$(conda info --base)/etc/profile.d/conda.sh\" "
                            f"&& conda activate {EXPECTED['env']}")
    else:
        r.ok('conda env', f'{env} at {prefix}')
    exe = os.path.realpath(sys.executable)
    in_env = bool(prefix) and exe.startswith(os.path.realpath(prefix) + os.sep)
    (r.ok if in_env else r.fail)('python executable', f"{sys.executable}{'' if in_env else ' is not inside CONDA_PREFIX'}")
    (r.ok if version.startswith(EXPECTED['python'] + '.') else r.fail)(
        'python version', f"{version} (required {EXPECTED['python']}; the image's base Python is not used)")


def module_file_ok(r, name, module, expected=None):
    version = getattr(module, '__version__', None)
    path = os.path.realpath(getattr(module, '__file__', '') or '')
    inside = path.startswith(os.path.realpath(sys.prefix) + os.sep)
    ok = inside and (expected is None or (version or '').split('+')[0] == expected)
    detail = f"{version} from {os.path.dirname(path)}"
    if not inside:
        detail += ' (outside the env: user site-packages? export PYTHONNOUSERSITE=1)'
    elif expected is not None and not ok:
        detail += f' (required {expected})'
    (r.ok if ok else r.fail)(name, detail)


def check_torch(r):
    try:
        import torch
    except Exception as e:
        return r.fail('torch', f'import failed: {e}')
    module_file_ok(r, 'torch', torch, EXPECTED['torch'])
    (r.ok if torch.version.cuda == EXPECTED['torch_cuda'] else r.fail)(
        'torch.version.cuda', f"{torch.version.cuda} (required {EXPECTED['torch_cuda']}; nvidia-smi's CUDA version "
                              'is the driver\'s and is not used)')
    available = torch.cuda.is_available()
    (r.ok if available else r.fail)('torch.cuda.is_available()', str(available))
    if not available:
        return
    count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / GiB
    (r.ok if count == 1 else r.fail)('GPU count', f'{count} (exactly 1 expected)')
    (r.ok if vram >= 75 else r.fail)('GPU', f'{props.name}, {vram:.1f} GiB (A100 80GB expected)')
    torch.zeros(1, device='cuda')
    try:
        with open('/proc/self/maps') as f:
            paths = sorted({l.split()[-1] for l in f if 'libcudart' in l and l.split()[-1].startswith('/')})
    except OSError:
        paths = []
    if not paths:
        r.warn('libcudart loaded by torch', 'not found in /proc/self/maps')
    else:
        bad = [p for p in paths if p.startswith('/usr/local/cuda')]
        (r.warn if bad else r.ok)('libcudart loaded by torch', ', '.join(paths))


def check_bnb(r):
    code, out, err = run([sys.executable, os.path.join(REPO_ROOT, 'scripts', 'verify_bnb_cuda.py')], timeout=300)
    keep = [l.strip() for l in out.splitlines() if any(k in l for k in ('loaded binary', 'picks', 'libcudart.so ->',
                                                                         'mapped', 'FAIL'))]
    if code == 0:
        r.ok('bitsandbytes', '; '.join(keep) or 'verify_bnb_cuda.py passed')
    else:
        r.fail('bitsandbytes', ('; '.join(keep) or err.strip()[-300:]) +
               ' -> re-run: bash scripts/setup_bnb_cuda_runtime.sh && conda deactivate && conda activate ilora')


def check_packages(r):
    try:
        import transformers
        module_file_ok(r, 'transformers', transformers, EXPECTED['transformers'])
        tdir = os.path.dirname(transformers.__file__)
        patched = []
        for src, dst in (('debug/utils.py', 'generation/utils.py'), ('debug/modeling_llama.py', 'models/llama/modeling_llama.py')):
            with open(os.path.join(REPO_ROOT, src), 'rb') as a, open(os.path.join(tdir, dst), 'rb') as b:
                patched.append(a.read() == b.read())
        (r.ok if all(patched) else r.fail)(
            'transformers patches', 'debug/utils.py and debug/modeling_llama.py applied' if all(patched)
            else 'not applied (generate() fails without them): run bash setup_vessl.sh')
    except Exception as e:
        r.fail('transformers', f'{type(e).__name__}: {e}')
    try:
        import pytorch_lightning
        module_file_ok(r, 'pytorch_lightning', pytorch_lightning, EXPECTED['pytorch_lightning'])
    except Exception as e:
        r.fail('pytorch_lightning', f'{type(e).__name__}: {e}')


def check_llm(r):
    path = os.environ.get('LLM_PATH', '')
    if not path or not os.path.isdir(path):
        return r.fail('LLM_PATH', f"'{path}' is not a directory (export LLM_PATH=/root/models/Llama-2-7b-hf)")
    missing = [f for f in ('config.json', 'tokenizer.model') if not os.path.exists(os.path.join(path, f))]
    index = next((os.path.join(path, f) for f in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
                  if os.path.exists(os.path.join(path, f))), None)
    shards = []
    if index is None:
        missing.append('*.index.json')
    else:
        with open(index) as f:
            shards = sorted(set(json.load(f)['weight_map'].values()))
        missing += [s for s in shards if not os.path.exists(os.path.join(path, s))]
    size = sum(os.path.getsize(os.path.join(path, s)) for s in shards if os.path.exists(os.path.join(path, s)))
    (r.ok if not missing else r.fail)('LLM_PATH', f'{path}: {len(shards)} shards, {size / GiB:.1f} GiB'
                                      + (f'; missing {missing} (re-run bootstrap with HF_TOKEN)' if missing else ''))


def check_dataset(r):
    missing = [f for f in DATASET_FILES if not os.path.exists(os.path.join(REPO_ROOT, f))]
    (r.ok if not missing else r.fail)('dataset', 'MovieLens files and rec_model/movielens.pt present' if not missing
                                      else f'missing {missing}')


def check_disk(r, fail_gib, warn_gib):
    info = mount_of(REPO_ROOT)
    if info is None:
        free = shutil.disk_usage(REPO_ROOT).free
    else:
        free = info['avail_bytes']
    gib = free / GiB
    status = r.fail if gib < fail_gib else (r.warn if gib < warn_gib else r.ok)
    status('free disk', f'{gib:.0f} GiB free for the repository (fail < {fail_gib}, warn < {warn_gib})')


def check_writable(r):
    bad = []
    for d in WRITABLE_DIRS:
        path = os.path.join(REPO_ROOT, d)
        try:
            os.makedirs(path, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path):
                pass
        except OSError as e:
            bad.append(f'{d} ({e})')
    (r.ok if not bad else r.fail)('writable folders', ', '.join(WRITABLE_DIRS) if not bad else f'not writable: {bad}')


def check_tmux(r):
    tmux = shutil.which('tmux')
    conflicts = ncurses_dirs(os.environ.get('LD_LIBRARY_PATH', ''))
    if conflicts:
        r.fail('LD_LIBRARY_PATH', f'contains libtinfo/libncurses directories {conflicts}: system tools such as tmux '
                                  "load them ('NCURSES6_TINFO_6.4.current not found'). Remove them; the ilora "
                                  'activation hook only adds the nvidia-* wheel directories')
    if not tmux:
        return r.warn('tmux', 'not installed (bootstrap_vessl.sh installs it with apt as root); use nohup instead')
    env = {k: v for k, v in os.environ.items() if k != 'TMUX'}
    code, out, err = run([tmux, '-V'], env=env)
    if code != 0:
        return r.fail('tmux', f'{tmux} -V failed: {err.strip()[-200:]}')
    socket = f'ilora-env-check-{os.getpid()}'
    c1, _, e1 = run([tmux, '-L', socket, 'new-session', '-d', '-s', 'check', 'sleep 10'], env=env)
    c2, _, _ = run([tmux, '-L', socket, 'has-session', '-t', 'check'], env=env)
    run([tmux, '-L', socket, 'kill-server'], env=env)
    tinfo = ''
    if shutil.which('ldd'):
        _, ldd, _ = run(['ldd', tmux])
        tinfo = next((l.split('=>')[-1].split('(')[0].strip() for l in ldd.splitlines() if 'libtinfo' in l), '')
    conda_base = os.path.dirname(os.path.dirname(os.environ['CONDA_EXE'])) if os.environ.get('CONDA_EXE') else ''
    from_conda = bool(tinfo) and any(tinfo.startswith(p + os.sep) for p in (os.environ.get('CONDA_PREFIX'), conda_base) if p)
    detail = f"{out.strip()} at {tmux}; detached session start/stop {'works' if c1 == 0 and c2 == 0 else 'FAILED: ' + e1.strip()}"
    if tinfo:
        detail += f'; libtinfo from {tinfo}'
    (r.ok if c1 == 0 and c2 == 0 and not from_conda else r.fail)('tmux', detail)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--storage-only', action='store_true')
    p.add_argument('--fail-free-gib', type=float, default=10)
    p.add_argument('--warn-free-gib', type=float, default=30)
    p.add_argument('--no-report', action='store_true', help='Do not write logs/env_check_<time>.json.')
    a = p.parse_args()
    r = Report()
    print('=== scripts/env_check.py ===')
    check_storage(r)
    if not a.storage_only:
        for check in (check_git, check_python, check_torch, check_bnb, check_packages, check_llm, check_dataset,
                      lambda rep: check_disk(rep, a.fail_free_gib, a.warn_free_gib), check_writable, check_tmux):
            try:
                check(r)
            except Exception as e:  # a crashing check is a failed check, never a silent pass
                r.fail(getattr(check, '__name__', 'check'), f'{type(e).__name__}: {e}')
        if not a.no_report:
            stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            try:
                os.makedirs(os.path.join(REPO_ROOT, 'logs'), exist_ok=True)
                with open(os.path.join(REPO_ROOT, 'logs', f'env_check_{stamp}.json'), 'w') as f:
                    json.dump({'time_utc': stamp, 'checks': r.rows}, f, indent=2)
            except OSError:
                pass
    fails, warns = r.count('FAIL'), r.count('WARN')
    print(f'env_check: {fails} FAIL, {warns} WARN' + (' -> do not start training' if fails else ''))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
