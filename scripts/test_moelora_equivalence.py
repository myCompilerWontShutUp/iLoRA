"""Regression test: model/peft/tuners/moelora.py Linear vs the upstream iLoRA version.

The LoRA branch of Linear.forward was rewritten from
    B_out = einsum(A_out, B)            # [batch, seq, num_moe, out_features], kept for backward
    result += (B_out * Gate).sum(-2) * scaling
to the algebraically identical
    result += lora_B((A_out * Gate).reshape(batch, seq, r)) * scaling
which avoids materializing the per-expert output tensor (the cause of the A100 80GB OOM, see
experiments/IMPLEMENTATION_NOTES.md). This script loads the upstream file from git (commit
28e4737, the fork point of exp/gate-cluster) next to the working-tree file and checks, with
identical weights and inputs:
  1. outputs and gradients w.r.t. x, gate, lora_A, lora_B agree (float64 and float32),
     including lora_dropout=0.1 in train mode and a single-token (generation) step;
  2. the rewrite saves fewer bytes for backward, by exactly the upstream B_out tensor;
  3. a CUDA out-of-memory error inside the layer propagates instead of being turned into zeros,
     while any other exception keeps the upstream behavior (print + zeros).
Runs on CPU. transformers / huggingface_hub are stubbed only if they are not installed (they
are only needed by PEFT config/hub utilities, not by Linear.forward).
"""
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_COMMIT = "28e4737"
sys.path.insert(0, REPO)


def _install_stubs_if_missing():
    try:
        import transformers  # noqa: F401
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass
    tf = types.ModuleType("transformers"); tf.__path__ = []
    tf_utils = types.ModuleType("transformers.utils"); tf_utils.PushToHubMixin = type("PushToHubMixin", (), {})
    tf_pt = types.ModuleType("transformers.pytorch_utils"); tf_pt.Conv1D = type("Conv1D", (torch.nn.Module,), {})
    hf = types.ModuleType("huggingface_hub"); hf.__path__ = []
    for name in ("hf_hub_download", "get_hf_file_metadata", "hf_hub_url"):
        setattr(hf, name, lambda *a, **k: None)
    hf_utils = types.ModuleType("huggingface_hub.utils")
    hf_utils.EntryNotFoundError = type("EntryNotFoundError", (Exception,), {})
    sys.modules.update({"transformers": tf, "transformers.utils": tf_utils, "transformers.pytorch_utils": tf_pt,
                        "huggingface_hub": hf, "huggingface_hub.utils": hf_utils})


def _load_modules():
    _install_stubs_if_missing()
    # Skip the heavy package __init__ files (peft_model etc.); keep the real submodule files.
    for name, rel in (("model.peft", "model/peft"), ("model.peft.tuners", "model/peft/tuners")):
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [os.path.join(REPO, rel)]
            sys.modules[name] = m
    current = importlib.import_module("model.peft.tuners.moelora")
    src = subprocess.check_output(["git", "-C", REPO, "show", f"{UPSTREAM_COMMIT}:model/peft/tuners/moelora.py"])
    path = os.path.join(tempfile.mkdtemp(), "moelora_upstream.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("model.peft.tuners.moelora_upstream", path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    return current, upstream


def _make_pair(current, upstream, in_f, out_f, dropout, dtype):
    gate_lists = ([], [])
    layers = []
    for mod, gl in zip((upstream, current), gate_lists):
        torch.manual_seed(0)
        layer = mod.Linear("default", in_f, out_f, r=8, num_moe=4, gating="Dense", global_user_embeds=[],
                           gate_weights=gl, lora_alpha=32, lora_dropout=dropout)
        torch.manual_seed(1)
        torch.nn.init.normal_(layer.lora_B["default"].weight, std=0.02)  # upstream initializes B to zero
        layers.append(layer.to(dtype).train())
    return layers


def _run(layer, x, gate):
    x = x.clone().requires_grad_(True)
    g = gate.clone().requires_grad_(True)
    layer.gate_weights.clear(); layer.gate_weights.append(g)
    layer.zero_grad()
    torch.manual_seed(123)  # identical dropout masks for both implementations
    out = layer(x)
    out.square().sum().backward()
    return [out.detach(), x.grad, g.grad, layer.lora_A["default"].weight.grad, layer.lora_B["default"].weight.grad]


def _saved_bytes(layer, x, gate):
    # Parameters are often saved as views (e.g. weight.t()), so exclude them by storage address.
    param_storages = {p.untyped_storage().data_ptr() for p in layer.parameters()}
    seen, total = set(), [0]

    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key[0] not in param_storages and key not in seen:
            seen.add(key); total[0] += key[1]
        return t
    g = gate.clone().requires_grad_(True)
    layer.gate_weights.clear(); layer.gate_weights.append(g)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        layer(x.clone().requires_grad_(True))
    return total[0]


def main():
    current, upstream = _load_modules()
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    names = ["output", "grad_x", "grad_gate", "grad_lora_A", "grad_lora_B"]
    for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 1e-4)):
        for dropout in (0.0, 0.1):
            for (b, l) in ((3, 7), (2, 1)):
                up, cur = _make_pair(current, upstream, 64, 96, dropout, dtype)
                x = torch.randn(b, l, 64, dtype=dtype)
                gate = torch.softmax(torch.randn(b, 1, 4, dtype=dtype), dim=-1)
                ref, new = _run(up, x, gate), _run(cur, x, gate)
                rel = {n: ((r - c).abs().max() / (r.abs().max() + 1e-30)).item() for n, r, c in zip(names, ref, new)}
                check(f"equivalence {str(dtype).split('.')[-1]} dropout={dropout} batch={b} seq={l}",
                      max(rel.values()) < tol, ", ".join(f"{k}={v:.1e}" for k, v in rel.items()))

    for in_f, out_f in ((4096, 4096), (4096, 11008)):
        up, cur = _make_pair(current, upstream, in_f, out_f, 0.1, torch.float32)
        x = torch.randn(2, 6, in_f)
        gate = torch.softmax(torch.randn(2, 1, 4), dim=-1)
        b_up, b_cur = _saved_bytes(up, x, gate), _saved_bytes(cur, x, gate)
        b_out_bytes = 2 * 6 * 4 * out_f * 4  # upstream's fp32 [batch, seq, num_moe, out] tensor in this run
        # The rewrite additionally keeps the tiny [batch, seq, r] gated_A_out, hence 0.99.
        check(f"saved-for-backward bytes {in_f}->{out_f}", b_up - b_cur >= 0.99 * b_out_bytes,
              f"upstream={b_up/2**20:.2f} MiB, current={b_cur/2**20:.2f} MiB, B_out={b_out_bytes/2**20:.2f} MiB")
        del up, cur

    _, cur = _make_pair(current, upstream, 64, 96, 0.0, torch.float32)
    cur.gate_weights.clear(); cur.gate_weights.append(torch.softmax(torch.randn(2, 1, 4), -1))
    x = torch.randn(2, 5, 64)

    def boom(msg):
        def f(*a, **k):
            raise RuntimeError(msg)
        return f
    cur.lora_A["default"].forward = boom("CUDA out of memory. Tried to allocate 148.00 MiB")
    try:
        cur(x)
        check("CUDA OOM inside the layer propagates", False, "no exception raised")
    except RuntimeError as e:
        check("CUDA OOM inside the layer propagates", "out of memory" in str(e))
    cur.lora_A["default"].forward = boom("some unrelated error")
    out = cur(x)
    check("other exceptions keep upstream behavior (zeros)", torch.count_nonzero(out).item() == 0 and out.shape == x.shape)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("\nAll moelora equivalence checks passed.")


if __name__ == "__main__":
    main()
