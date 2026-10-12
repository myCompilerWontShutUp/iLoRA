"""Check that the SASRec causal-mask change in SASRecModules_ori.py changes no value.

torch 2.0's CUDA torch.tril has no bfloat16 kernel, so MultiHeadAttention now builds the lower
triangle as bool and casts it to the attention scores' dtype. This compares it with the upstream
code (git show 28e4737:SASRecModules_ori.py), bit for bit:

1. the 0/1 mask itself, for float32 / float16 / bfloat16, on CPU and (if available) CUDA;
2. MultiHeadAttention.forward on random inputs with padded keys/queries, same dtypes and devices;
3. rec_model/movielens.pt's cacul_h (the path encode_users uses) on real MovieLens training
   sequences, under float32 and under autocast float16 / bfloat16, exactly as during training.

Where the upstream code cannot run (CUDA bfloat16, the reason for the change) the new result is
compared with the upstream result computed on CPU in bfloat16 for the mask, and checked to be
finite for the forward passes. Exits non-zero on any mismatch.

    python scripts/test_sasrec_mask_equivalence.py
"""
import os
import subprocess
import sys
import types

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# rec_model/movielens.pt was pickled as __main__.SASRec: same imports as main.py.
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU  # noqa: F401,E402
from SASRecModules_ori import *  # noqa: F401,F403,E402
import SASRecModules_ori as new_mod  # noqa: E402

UPSTREAM_COMMIT = '28e4737'
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
failures = []


def check(ok, message):
    print(f"  [{'OK' if ok else 'FAIL'}] {message}")
    if not ok:
        failures.append(message)


def same(a, b):
    """Bitwise equality, NaN positions included."""
    return (a.dtype == b.dtype and a.shape == b.shape and torch.equal(torch.isnan(a), torch.isnan(b))
            and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)))


def load_upstream():
    source = subprocess.check_output(['git', 'show', f'{UPSTREAM_COMMIT}:SASRecModules_ori.py'], cwd=REPO_ROOT)
    module = types.ModuleType('SASRecModules_upstream')
    exec(compile(source, f'{UPSTREAM_COMMIT}:SASRecModules_ori.py', 'exec'), module.__dict__)
    return module


def run_upstream(fn):
    try:
        return fn(), None
    except RuntimeError as e:
        return None, str(e).splitlines()[0]


def new_tril(like):
    return torch.tril(torch.ones(like.shape, dtype=torch.bool, device=like.device)).to(like.dtype)


old_mod = load_upstream()
devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
print(f"torch {torch.__version__}; devices: {devices}")

print("1. causal mask values")
for device in devices:
    for dtype in DTYPES:
        diag_vals = torch.ones(10, 10, dtype=dtype, device=device)
        new = new_tril(diag_vals)
        old, err = run_upstream(lambda: torch.tril(diag_vals))
        if old is None:
            reference = torch.tril(torch.ones(10, 10, dtype=dtype)).to(device)
            check(same(new, reference), f"{device} {dtype}: upstream fails ({err}); new == CPU torch.tril")
        else:
            check(same(new, old), f"{device} {dtype}: new == torch.tril")

print("2. MultiHeadAttention.forward, random inputs with padded keys/queries")
for device in devices:
    torch.manual_seed(0)
    mha = new_mod.MultiHeadAttention(64, 64, 1, 0.1).to(device).eval()
    x = torch.randn(6, 10, 64, device=device)
    x[:, 7:] = 0  # padded positions: key and query masks kick in
    x[0] = 0      # a fully padded sequence
    for dtype in DTYPES:
        m = mha.to(dtype)
        xs = x.to(dtype)
        new = m(xs, xs)
        old, err = run_upstream(lambda: old_mod.MultiHeadAttention.forward(m, xs, xs))
        if old is None:
            check(bool(torch.isfinite(new).all()), f"{device} {dtype}: upstream fails ({err}); new output finite")
        else:
            check(same(new, old), f"{device} {dtype}: new == upstream")

print("3. rec_model/movielens.pt cacul_h on MovieLens training sequences")
from data.movielens_data import MovielensData  # noqa: E402

data = MovielensData(data_dir='data/ref/movielens', stage='train', cans_num=20)
samples = [data[i] for i in range(0, len(data), max(len(data) // 64, 1))][:64]
seq = torch.tensor([s['seq'] for s in samples])
len_seq = torch.tensor([s['len_seq'] for s in samples])
rec = torch.load('rec_model/movielens.pt', map_location='cpu', weights_only=False)  # the repo's own checkpoint, as MInterface.load_rec_model
rec.eval()
for device in devices:
    rec.to(device)
    rec.device = device
    attn = rec.mh_attn
    for dtype in DTYPES:
        autocast = (torch.autocast(device_type=device, dtype=dtype) if dtype != torch.float32
                    else torch.autocast(device_type=device, enabled=False))
        with torch.no_grad(), autocast:
            try:
                new = rec.cacul_h(seq.to(device), len_seq.to(device))
            except RuntimeError as e:  # e.g. an op without a kernel for this dtype on this device
                check(False, f"{device} {dtype}: new cacul_h raised {str(e).splitlines()[0]}")
                continue
            attn.forward = types.MethodType(old_mod.MultiHeadAttention.forward, attn)
            try:
                old, err = run_upstream(lambda: rec.cacul_h(seq.to(device), len_seq.to(device)))
            finally:
                del attn.forward
        if old is None:
            check(bool(torch.isfinite(new).all()), f"{device} autocast {dtype}: upstream fails ({err}); "
                                                   f"new output finite, shape {tuple(new.shape)}")
        else:
            check(same(new, old), f"{device} autocast {dtype}: new == upstream, shape {tuple(new.shape)}")

if failures:
    print(f"{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("SASRec mask change: identical values wherever the upstream code runs.")
