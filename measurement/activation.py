"""Read-only forward hooks on the outputs of fp16/bf16-output modules.

Every hooked call reduces its output to a few GPU tensors (no .item(), no host sync): max |value|
with its position, the 32 largest |values|, the output row (sample, token) holding the maximum,
the non-finite count, and, on recorded micro-batches, mean/std/threshold fractions and sampled
quantiles. `end()` moves everything to the CPU in one transfer per device.

The hooks only read detached tensors and never return a value, so the forward and backward
computation is unchanged. Quantile sampling uses the monitor's own torch.Generator, so the global
CPU/CUDA RNG that dropout draws from is not consumed.
"""
import re

import numpy as np
import torch

from .schema import parse_module_name, short_name  # noqa: F401 (re-exported)

LOWP = (torch.float16, torch.bfloat16)
LIGHT = ('max_abs', 'signed', 'row', 'feature', 'nonfinite')
FULL = ('mean_abs', 'std', 'frac_over_1e3', 'frac_over_1e4', 'median_abs', 'p90_abs', 'p99_abs', 'p999_abs')
QUANTILE_INPUT_LIMIT = 16_000_000  # torch.quantile refuses larger inputs
def first_lowp_tensor(output):
    if torch.is_tensor(output):
        items = [output]
    elif isinstance(output, dict):
        items = list(output.values())
    elif isinstance(output, (tuple, list)):
        items = list(output)
    else:
        return None
    for t in items:
        if torch.is_tensor(t) and t.dtype in LOWP and t.dim() >= 1 and t.numel() > 0:
            return t
    return None


def row_coords(shape, row):
    """(sample, token) of a row of the output viewed as [-1, features]; token -1 if not 3-D."""
    row = int(row)
    if len(shape) == 3:
        return row // shape[1], row % shape[1]
    if len(shape) <= 2:
        return row, -1
    return int(np.unravel_index(row, shape[:-1])[0]), -1


def _stack_cpu(tensors):
    by_device = {}
    for i, t in enumerate(tensors):
        by_device.setdefault(t.device, []).append(i)
    out = [None] * len(tensors)
    for idxs in by_device.values():
        arr = torch.stack([tensors[i] for i in idxs]).cpu().numpy()
        for j, i in enumerate(idxs):
            out[i] = arr[j]
    return np.stack(out)


class ModuleBatch:
    """One micro-batch: `records` (per hooked call, GPU tensors kept for dumps), `light` [calls, 5]
    and `full` {call index: [8]} as float64 numpy."""

    def __init__(self, records, light, full):
        self.records, self.light, self.full = records, light, full

    def __len__(self):
        return len(self.records)


class ActivationMonitor:
    def __init__(self, settings):
        self.s = settings
        self.regex = re.compile(settings.module_regex) if settings.module_regex else None
        self.sample_size = max(1, min(int(settings.quantile_sample), QUANTILE_INPUT_LIMIT))
        self.handles = []
        self.enabled = False
        self.collect_full = False
        self.records = []
        self._calls = {}
        self._generators = {}
        self._qpoints = {}

    @property
    def hooked_modules(self):
        return len(self.handles)

    def attach(self, root, skip=()):
        for name, module in root.named_modules():
            if not name or (self.regex is not None and not self.regex.search(name)):
                continue
            if any(name == s or name.startswith(s + '.') for s in skip):
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def begin(self, collect_full):
        self.records, self._calls = [], {}
        self.collect_full = bool(collect_full)
        self.enabled = True

    def end(self):
        self.enabled = False
        records, self.records = self.records, []
        if not records:
            return ModuleBatch([], np.zeros((0, len(LIGHT))), {})
        light = _stack_cpu([r['light'] for r in records])
        idx = [i for i, r in enumerate(records) if r['full'] is not None]
        full = dict(zip(idx, _stack_cpu([records[i]['full'] for i in idx]))) if idx else {}
        return ModuleBatch(records, light, full)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            t = first_lowp_tensor(output)
            if t is None:
                return
            n = self._calls.get(name, 0) + 1
            self._calls[name] = n
            self._observe(name if n == 1 else f'{name}#{n}', t)
        return hook

    def _generator(self, device):
        g = self._generators.get(device)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.s.sample_seed)
            self._generators[device] = g
        return g

    def _quantile_points(self, device):
        q = self._qpoints.get(device)
        if q is None:
            q = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32, device=device)
            self._qpoints[device] = q
        return q

    @torch.no_grad()
    def _observe(self, name, t):
        x = t.detach()
        features = x.shape[-1]
        xr = x.reshape(-1, features)
        a = xr.abs()
        nonfinite = (~torch.isfinite(xr)).sum()
        rowmax = a.amax(dim=1).float()
        _, top_rows_idx = rowmax.topk(min(self.s.topk, rowmax.numel()))
        rows = a.index_select(0, top_rows_idx).float()          # every one of the k largest values lies in these rows
        top_abs, flat = rows.reshape(-1).topk(min(self.s.topk, rows.numel()))
        top_rows, top_feats = top_rows_idx[flat // features], flat % features
        max_row, max_feat = top_rows[0], top_feats[0]
        light = torch.stack([top_abs[0].double(), xr[max_row, max_feat].double(), max_row.double(),
                             max_feat.double(), nonfinite.double()])
        full = None
        if self.collect_full:
            n = a.numel()
            flat_a = a.reshape(-1)
            if n <= self.sample_size:
                sample = flat_a.float()
            else:
                idx = torch.randint(n, (self.sample_size,), device=a.device, generator=self._generator(a.device))
                sample = flat_a[idx].float()
            q = torch.quantile(sample, self._quantile_points(a.device))
            std = xr.float().std() if n > 1 else torch.zeros((), device=a.device)
            full = torch.cat([torch.stack([(a.sum(dtype=torch.float32) / n).double(), std.double(),
                                           (a > 1e3).sum().double() / n, (a > 1e4).sum().double() / n]),
                              q.double()])
        self.records.append({
            'name': name, 'shape': tuple(x.shape), 'dtype': str(x.dtype).replace('torch.', ''), 'numel': x.numel(),
            'light': light, 'full': full, 'top_abs': top_abs, 'top_signed': xr[top_rows, top_feats].float(),
            'top_rows': top_rows, 'top_feats': top_feats, 'token_vec': xr[max_row].float(),
        })


def fetch_dump_tensors(record):
    """CPU copies of the per-call tensors kept for an anomaly dump."""
    return {k: record[k].cpu().numpy() for k in ('top_abs', 'top_signed', 'top_rows', 'top_feats', 'token_vec')}
