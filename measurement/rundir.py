"""Run folders (`runs/<name>_<YYYYmmdd-HHMMSS>/`, never reused) and their meta.json."""
import os
import platform
import re
import shlex
import subprocess
import sys
import time

from .io import read_json, write_json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_run_dir(root, name):
    stamp = time.strftime('%Y%m%d-%H%M%S')
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_') or 'run'
    base = os.path.join(root, f'{safe}_{stamp}')
    path, i = base, 1
    while True:
        try:
            os.makedirs(path)  # exist_ok=False: an existing folder is never written into
            break
        except FileExistsError:
            path, i = f'{base}-{i}', i + 1
    os.makedirs(os.path.join(path, 'anomaly_dumps'))
    return path


def claim_dir(path):
    """Use a user-given folder only if it does not exist yet or is empty."""
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f'{path} is not empty; run folders are never overwritten')
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'anomaly_dumps'), exist_ok=True)
    return path


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_info():
    status = _git('status', '--porcelain')
    lines = status.splitlines() if status else []
    tracked = [l for l in lines if not l.startswith('??')]
    return {'commit': _git('rev-parse', 'HEAD'), 'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
            'tracked_files_unmodified': status is not None and not tracked,
            'status_porcelain': lines}


def environment():
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'hostname': platform.node()}
    try:
        import torch
        env.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                   cuda_available=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:  # reading a run folder must not need torch
        env['torch'] = f'unavailable: {e}'
    return env


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


class RunMeta:
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'meta.json')
        self.data = read_json(self.path)

    def start(self, kind, config, settings, precision, lr, seed, extra=None):
        self.data.update({
            'kind': kind,
            'run_dir': os.path.abspath(os.path.dirname(self.path)),
            'command': ' '.join(shlex.quote(a) for a in [sys.executable] + sys.argv),
            'cwd': os.getcwd(),
            'git': git_info(),
            'environment': environment(),
            'config': _jsonable(config),
            'seed': seed,
            'precision': precision,
            'max_lr': lr,
            'settings': _jsonable(settings),
            'start_time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'end_time_utc': None,
        })
        if extra:
            self.data.update(_jsonable(extra))
        self.save()

    def update(self, **fields):
        self.data.update(_jsonable(fields))
        self.save()

    def save(self):
        write_json(self.path, self.data)


def artifact_entry(path):
    entry = {'path': os.path.abspath(path) if path else None, 'exists': bool(path) and os.path.exists(path)}
    if entry['exists'] and os.path.isfile(path):
        entry['bytes'] = os.path.getsize(path)
    return entry


def training_artifacts(config, tensorboard_dir=None):
    """Paths of the raw inputs of the later analyses (P3-P6), from a main.py / run-script config."""
    get = (config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d))
    results = get('results_dir') or ''
    clustering = get('cluster_assignment_dir') or ''
    out = {'tensorboard_dir': artifact_entry(tensorboard_dir),
           'metrics_json': artifact_entry(os.path.join(results, 'metrics.json') if results else None),
           'ckpt_dir': artifact_entry(get('ckpt_dir')),
           'output_dir': artifact_entry(get('output_dir'))}
    for split in ('test', 'validation'):
        out[f'gates_{split}_csv'] = artifact_entry(os.path.join(results, f'gates_{split}.csv') if results else None)
    for split in ('train', 'validation', 'test'):
        out[f'representations_{split}_npz'] = artifact_entry(
            os.path.join(results, f'representations_{split}.npz') if results else None)
        out[f'{split}_assignments_csv'] = artifact_entry(
            os.path.join(clustering, f'{split}_assignments.csv') if clustering else None)
    return out
