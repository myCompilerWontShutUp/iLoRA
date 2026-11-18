"""M12 기록 (DESIGN.md §16.8 #1-#7, #12, #13; test names of §17.6.9).

``TrainRecorder``, ``ActivationMaxMonitor``, the gate and z contracts, ``RepresentationTable`` with its NPZ
and CSV files, ``write_eval``/``read_eval_records``, ``crosscheck_gates`` and the export ``Router`` are tested
with synthetic inputs (``seeding.np_rng(0, "test", <name>)``, ``seeding.torch_seed_scope(0, "test", <name>)``,
§17.6.2), the tiny CPU model (``tiny_model``) and the converted SASRec weights. #8-#11 and the end-to-end
export are in ``test_m12_analysis.py``; #14-#16 are VESSL tests in ``tests/vessl/``.

``EvalRecord`` is the M11 type of ``ilora/evaluate.py`` (§15.6.1). While that file does not exist, the
``eval_record_cls`` fixture registers a stand-in dataclass with the 3.1 fields as ``ilora.evaluate`` for one
test; once it exists the real class is used.
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import math
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest
import torch
from torch import nn

from ilora import records, seeding
from ilora.candidates import FLAG_NAMES
from ilora.data import HISTORY_LEN, PAD_ID, Sample, pad_histories
from ilora.records import (EPOCH_FIELDS, MICRO_FIELDS, ActivationMaxMonitor, RecordsError, RepresentationTable,
                           TrainRecorder, step_fields)

K = 4


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class StandInEvalRecord:
    """The 3.1 ``EvalRecord`` fields in the M11 §15.6.1 order (used only while ilora/evaluate.py is missing)."""
    sample_id: str
    split: str
    user_id: int
    template_id: int
    hybrid: bool
    cand_ids: tuple
    cand_titles: tuple
    target_id: int
    target_title: str
    raw_text: str
    first_line: str
    matched: tuple
    valid: bool
    correct: bool
    gates: tuple
    flags: Mapping
    gen_token_ids: tuple


@pytest.fixture
def eval_record_cls(monkeypatch):
    """M11 ``EvalRecord`` (real once evaluate.py exists, else the stand-in registered as ``ilora.evaluate``)."""
    if importlib.util.find_spec("ilora.evaluate") is not None:
        from ilora.evaluate import EvalRecord

        return EvalRecord
    module = types.ModuleType("ilora.evaluate")
    module.EvalRecord = StandInEvalRecord
    monkeypatch.setitem(sys.modules, "ilora.evaluate", module)
    return StandInEvalRecord


def recorder_cfg(flush_every: int = 20, activation_max: bool = False) -> SimpleNamespace:
    """The two ``records`` keys TrainRecorder reads (RecordsConfig attributes)."""
    return SimpleNamespace(flush_every_steps=flush_every, activation_max=activation_max)


def micro_row(tau: int, epoch: int, step: int, rng: np.random.Generator, **changes) -> dict:
    row = {"tau": tau, "epoch": epoch, "step": step, "hybrid": bool(tau % 3 == 0), "curriculum_p": tau / 100.0,
           "batch_size": 4, "seq_len": 200 + tau % 7, "n_label_tokens": 9, "loss": float(rng.random() * 3),
           "loss_finite": True, "act_max_abs": None, "act_max_module": None, "time_s": float(tau)}
    row.update(changes)
    return row


def step_row(step: int, epoch: int, first: int, last: int, rng: np.random.Generator, k: int = K, **changes) -> dict:
    omega = rng.dirichlet(np.ones(k))
    row = {"step": step, "epoch": epoch, "tau_first": first, "tau_last": last, "n_micro": last - first + 1,
           "n_hybrid": 1, "loss_mean": float(rng.random()), "loss_hybrid_mean": float(rng.random()),
           "loss_text_mean": None, "nonfinite_loss": 0, "lr_lora": 1e-4, "lr_projector": 1e-4,
           "lr_gate": 1e-4 if k > 1 else None, "grad_norm_lora": float(rng.random()), "grad_norm_projector": None,
           "grad_norm_gate": float(rng.random()) if k > 1 else None, "grad_norm_total": float(rng.random()),
           "projector_has_grad": False, "scale_before": 65536.0, "scale_after": 65536.0, "skipped": False,
           **{f"omega_mean_{j}": float(omega[j]) for j in range(k)},
           "omega_entropy_mean": 0.9 if k > 1 else None, "gate_calls": last - first + 1, "cluster_lookups": 0,
           "max_mem_gib": None, "max_mem_reserved_gib": None, "time_s": float(last) + 0.5}
    row.update(changes)
    return row


def epoch_row(epoch: int, last_step: int, tau_last: int, **changes) -> dict:
    row = {"epoch": epoch, "last_step": last_step, "tau_last": tau_last, "plan_digest": f"{epoch:064x}",
           "n_hybrid": 3, "train_loss_mean": 1.25, "n_skipped_steps": 0, "val_n_total": 94, "val_n_valid": 90,
           "val_n_correct": 40, "val_valid_ratio": 90 / 94, "val_hr_at_1": 40 / 90, "val_combined_metric": 40 / 94,
           "act_max_abs": None, "act_max_module": None, "max_mem_gib": None, "max_mem_reserved_gib": None,
           "train_s": 1.0, "val_s": 2.0, "save_s": 0.5, "time_s": float(tau_last) + 0.75}
    row.update(changes)
    return row


def drive(recorder: TrainRecorder, *, epochs: range, steps_per_epoch: int, n_micro: int, name: str,
          explicit_flush: bool = True) -> None:
    """Rows in the M10 order (§14.6.4): window micro rows, step row; at the epoch end flush() (save), log_epoch."""
    rng = seeding.np_rng(0, "test", name)
    for epoch in epochs:
        for s in range(steps_per_epoch):
            step = epoch * steps_per_epoch + s
            first = step * n_micro + 1
            for tau in range(first, first + n_micro):
                recorder.log_micro_batch(micro_row(tau, epoch, step, rng))
            recorder.log_optimizer_step(step_row(step, epoch, first, first + n_micro - 1, rng,
                                                 k=recorder.num_experts))
        if explicit_flush:
            recorder.flush()
        last_step = (epoch + 1) * steps_per_epoch - 1
        recorder.log_epoch(epoch_row(epoch, last_step, (last_step + 1) * n_micro))


def train_dir(run_dir: Path) -> Path:
    return run_dir / "records" / "train"


def raw_rows(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


# ----------------------------------------------------------------------------------------------- #1
DESIGN_MICRO = ("tau, epoch, step, hybrid, curriculum_p, batch_size, seq_len, n_label_tokens, loss, loss_finite, "
                "act_max_abs, act_max_module, time_s")
DESIGN_STEP_HEAD = ("step, epoch, tau_first, tau_last, n_micro, n_hybrid, loss_mean, loss_hybrid_mean, loss_text_mean, "
                    "nonfinite_loss, lr_lora, lr_projector, lr_gate, grad_norm_lora, grad_norm_projector, "
                    "grad_norm_gate, grad_norm_total, projector_has_grad, scale_before, scale_after, skipped")
DESIGN_STEP_TAIL = "omega_entropy_mean, gate_calls, cluster_lookups, max_mem_gib, max_mem_reserved_gib, time_s"
DESIGN_EPOCH = ("epoch, last_step, tau_last, plan_digest, n_hybrid, train_loss_mean, n_skipped_steps, val_n_total, "
                "val_n_valid, val_n_correct, val_valid_ratio, val_hr_at_1, val_combined_metric, act_max_abs, "
                "act_max_module, max_mem_gib, max_mem_reserved_gib, train_s, val_s, save_s, time_s")


def _names(text: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in text.split(","))


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


@pytest.mark.criterion("M12-1")
def test_columns_and_row_validation(tmp_path):
    """Column order and names of 16.6.2, step_fields(1)/(4), bad rows -> ValueError, floats back bit for bit."""
    assert MICRO_FIELDS == _names(DESIGN_MICRO)
    assert EPOCH_FIELDS == _names(DESIGN_EPOCH)
    assert step_fields(4) == (*_names(DESIGN_STEP_HEAD), "omega_mean_0", "omega_mean_1", "omega_mean_2",
                              "omega_mean_3", *_names(DESIGN_STEP_TAIL))
    assert step_fields(1) == (*_names(DESIGN_STEP_HEAD), "omega_mean_0", *_names(DESIGN_STEP_TAIL))
    rec = TrainRecorder(tmp_path, recorder_cfg(flush_every=1), num_experts=K, total_micro_batches=100, total_steps=20)
    files = {"micro_batches.csv": MICRO_FIELDS, "optimizer_steps.csv": step_fields(K), "epochs.csv": EPOCH_FIELDS}
    for name, fields in files.items():
        assert raw_rows(train_dir(tmp_path) / name) == [list(fields)]
    columns = json.loads((train_dir(tmp_path) / "columns.json").read_text(encoding="utf-8"))
    for name, fields in files.items():
        assert [c["name"] for c in columns[name]] == list(fields)
        assert all(c["unit"] and c["description"] and c["kind"] in ("int", "float", "bool", "str") for c in columns[name])

    rng = seeding.np_rng(0, "test", "m12_columns")
    good = micro_row(1, 0, 0, rng)
    bad_rows = {
        "missing": {k: v for k, v in good.items() if k != "loss"},
        "extra": {**good, "lr": 1e-4},
        "float_in_int": {**good, "tau": 1.0},
        "bool_in_int": {**good, "batch_size": True},
        "str_in_float": {**good, "loss": "0.5"},
        "none_not_nullable": {**good, "seq_len": None},
        "int_in_bool": {**good, "hybrid": 1},
        "empty_str": {**good, "act_max_abs": 3.0, "act_max_module": ""},
        "tensor_not_0dim": {**good, "act_max_abs": torch.ones(2)},
        "tau_gap": {**good, "tau": 2},
    }
    for case, row in bad_rows.items():
        with pytest.raises(ValueError):
            rec.log_micro_batch(row)
        assert rec._pending["micro"] == [], case
    # floats that a short format would change; NaN/inf losses are recorded as such
    tricky = [0.1, 1 / 3, 1e-300, 5e-324, -0.0, 65504.0, float(np.float32(0.1)), 2.0 ** 60 + 1.0, math.nan, math.inf]
    rows = [micro_row(i + 1, 0, 0, rng, loss=x, loss_finite=math.isfinite(x), curriculum_p=x if math.isfinite(x) else 0.5,
                      time_s=float(np.nextafter(i, i + 1))) for i, x in enumerate(tricky)]
    for row in rows:
        rec.log_micro_batch(row)
    rec.log_optimizer_step(step_row(0, 0, 1, len(rows), rng, scale_before=2.0 ** -24, grad_norm_total=1e308))
    back = records.read_rows(train_dir(tmp_path) / "micro_batches.csv", MICRO_FIELDS)     # step 0 flushed (0 % 1)
    assert len(back) == len(rows)
    for row, got in zip(rows, back):
        for name in ("loss", "curriculum_p", "time_s"):
            assert _bits(got[name]) == _bits(row[name]) or (math.isnan(got[name]) and math.isnan(row[name])), name
        assert got["hybrid"] is row["hybrid"] and got["act_max_abs"] is None and got["tau"] == row["tau"]
    steps = records.read_rows(train_dir(tmp_path) / "optimizer_steps.csv", step_fields(K))
    assert steps[0]["scale_before"] == 2.0 ** -24 and steps[0]["grad_norm_total"] == 1e308
    assert steps[0]["loss_text_mean"] is None and steps[0]["skipped"] is False
    with pytest.raises(RecordsError, match="already exist"):
        TrainRecorder(tmp_path, recorder_cfg(), num_experts=K, total_micro_batches=100, total_steps=20)

# ----------------------------------------------------------------------------------------------- #2
class AppendSpy:
    """Wraps ``records._append_rows`` (the only write of training rows) and notes where each write happens."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[tuple[str, int, str]] = []
        self.phase = "other"
        original = records._append_rows

        def spy(path, rows):
            self.calls.append((Path(path).name, len(rows), self.phase))
            return original(path, rows)

        monkeypatch.setattr(records, "_append_rows", spy)


@pytest.mark.criterion("M12-2")
def test_flush_schedule(tmp_path, monkeypatch):
    """64 steps: files are written only at flushes (step % 20 == 0, before the save, after the epoch row, close)."""
    spy = AppendSpy(monkeypatch)
    rec = TrainRecorder(tmp_path, recorder_cfg(flush_every=20), num_experts=K, total_micro_batches=256,
                        total_steps=64)
    rng = seeding.np_rng(0, "test", "m12_flush")
    flush_points = []
    for epoch in range(2):
        for s in range(32):
            step = epoch * 32 + s
            first = step * 4 + 1
            for tau in range(first, first + 4):
                spy.phase = "micro"
                rec.log_micro_batch(micro_row(tau, epoch, step, rng))
            spy.phase = f"step{step}"
            before = len(spy.calls)
            rec.log_optimizer_step(step_row(step, epoch, first, first + 3, rng))
            if len(spy.calls) > before:
                flush_points.append(step)
        spy.phase = f"save{epoch}"
        rec.flush()
        spy.phase = f"epoch{epoch}"
        rec.log_epoch(epoch_row(epoch, (epoch + 1) * 32 - 1, (epoch + 1) * 128))
    spy.phase = "close"
    rec.close("completed", {"best_val_epoch": 1})
    assert flush_points == [0, 20, 40, 60]                                  # step % flush_every_steps == 0
    assert not [c for c in spy.calls if c[2] == "micro"]                    # never per micro-batch
    phases = [c[2] for c in spy.calls]
    assert phases == ["step0"] * 2 + ["step20"] * 2 + ["save0"] * 2 + ["epoch0"] + ["step40"] * 2 + ["step60"] * 2 \
        + ["save1"] * 2 + ["epoch1"]                                        # nothing left for close
    assert rec.flush_count == 8
    written = {name: sum(n for f, n, _ in spy.calls if f == name) for name in
               ("micro_batches.csv", "optimizer_steps.csv", "epochs.csv")}
    assert written == {"micro_batches.csv": 256, "optimizer_steps.csv": 64, "epochs.csv": 2}
    summary = json.loads((train_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["micro_batches"] == 256 and summary["optimizer_steps"] == 64
    assert summary["gate_calls"] == 256 and summary["train_summary"] == {"best_val_epoch": 1}
    assert set(summary["files"]) == {"micro_batches.csv", "optimizer_steps.csv", "epochs.csv",
                                     "activation_module_max.csv", "columns.json"}


@pytest.mark.criterion("M12-2")
def test_close_failed_keeps_rows(tmp_path):
    """An exception in the middle of an epoch: close("failed") in finally still writes every row and the summary."""
    rec = TrainRecorder(tmp_path, recorder_cfg(flush_every=20), num_experts=K, total_micro_batches=400,
                        total_steps=100)
    rng = seeding.np_rng(0, "test", "m12_failed")
    with pytest.raises(RuntimeError, match="injected"):
        try:
            for step in range(7):
                first = step * 4 + 1
                for tau in range(first, first + 4):
                    rec.log_micro_batch(micro_row(tau, 0, step, rng))
                rec.log_optimizer_step(step_row(step, 0, first, first + 3, rng, skipped=step == 2))
            rec.log_micro_batch(micro_row(29, 0, 7, rng))                     # half a window, then the failure
            raise RuntimeError("injected failure")
        finally:
            rec.close("failed", {"epochs_done": 0})
    micro = records.read_rows(train_dir(tmp_path) / "micro_batches.csv", MICRO_FIELDS)
    steps = records.read_rows(train_dir(tmp_path) / "optimizer_steps.csv", step_fields(K))
    assert [r["tau"] for r in micro] == list(range(1, 30)) and [r["step"] for r in steps] == list(range(7))
    summary = json.loads((train_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and summary["skipped_steps"] == [2] and summary["micro_batches"] == 29
    assert summary["expected"] == {"micro_batches": 400, "optimizer_steps": 100}
    with pytest.raises(RecordsError, match="closed"):
        rec.log_micro_batch(micro_row(30, 0, 7, rng))
    with pytest.raises(RecordsError, match="already closed"):
        rec.close("completed", {})


# ----------------------------------------------------------------------------------------------- #3
class Batches:
    """Real ``TrainBatch`` of the smoke_cpu data from the public M2, M4 and M5 functions (pattern of test_m9_model)."""

    def __init__(self, cfg, data, bundle, templates) -> None:
        self.cfg, self.data, self.bundle, self.templates = cfg, data, bundle, templates

    def train(self, *, hybrid: bool, n: int = 4, start: int = 0):
        from ilora.batching import collate_train, encode_train
        from ilora.candidates import candidates_for
        from ilora.prompts import build_prompt, choose_template

        cfg, samples = self.cfg, list(self.data.train[start:start + n])
        cands, tids, encoded = [], [], []
        for s in samples:
            c = candidates_for(s, main_seed=cfg.run.seed, epoch=0, cfg=cfg.candidates)
            k = choose_template(cfg.prompt, cfg.run.seed, len(self.templates.templates), s, 0)
            prompt = build_prompt(s, c, self.data.catalog, self.templates, k, hybrid)
            cands.append(c)
            tids.append(k)
            encoded.append(encode_train(self.bundle, prompt, self.data.catalog.titles[s.target],
                                        cfg.batching.label_mode))
        return collate_train(samples, cands, tids, encoded, self.bundle, hybrid=hybrid, tau=1)


def _loss_grads_rng(model, batch, name: str) -> tuple[torch.Tensor, dict, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    with seeding.torch_seed_scope(0, "test", name):
        out = model.train_forward(batch)                  # train mode: LoRA dropout draws from the global CPU RNG
        out.loss.backward()
        state = torch.get_rng_state()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return out.loss.detach().clone(), grads, state


@pytest.mark.criterion("M12-3")
def test_activation_monitor(tiny_model, cpu_cfg, smoke_data, bundle, templates, tmp_path):
    """A planted 1234.5 is recorded with its module name; the hooks change nothing; detach leaves no hook."""
    from ilora.moe_lora import GateContext, MoELoRALinear

    model = tiny_model()
    model.train()
    batch = Batches(cpu_cfg, smoke_data, bundle, templates).train(hybrid=True)
    names = [n for n, m in model.named_modules() if isinstance(m, MoELoRALinear)]
    assert len(names) == model.n_wrapped == 14

    # hooks on and off: loss, every gradient and the RNG state are bit for bit the same
    loss_a, grads_a, rng_a = _loss_grads_rng(model, batch, "m12_monitor_dropout")
    monitor = ActivationMaxMonitor(model, MoELoRALinear)
    assert monitor.module_names == tuple(names)
    monitor.attach()
    loss_b, grads_b, rng_b = _loss_grads_rng(model, batch, "m12_monitor_dropout")
    value, index = monitor.end_micro_batch(1)
    assert value.dim() == 0 and index.dim() == 0 and value.dtype == torch.float32 and index.dtype == torch.int64
    assert torch.equal(loss_a, loss_b) and torch.equal(rng_a, rng_b)
    assert grads_a.keys() == grads_b.keys() and all(torch.equal(grads_a[n], grads_b[n]) for n in grads_a)
    assert math.isfinite(float(value)) and 0 < float(value) < 1234.5

    # a value planted in the output of the last wrapper (nothing wrapped runs after it) is recorded with its name
    last = dict(model.named_modules())[names[-1]]

    def plant(module, inputs, output):
        output = output.clone()
        output[0, 0, 0] = 1234.5
        return output

    monitor.detach()
    handle = last.register_forward_hook(plant)            # before the monitor's hook, so the monitor sees it
    monitor = ActivationMaxMonitor(model, MoELoRALinear)
    monitor.attach()
    rec = TrainRecorder(tmp_path, recorder_cfg(flush_every=1, activation_max=True), num_experts=K,
                        total_micro_batches=10, total_steps=10)
    rec.attach_monitor(monitor)
    with torch.no_grad():
        model.train_forward(batch)
    value, index = monitor.end_micro_batch(1)
    rng = seeding.np_rng(0, "test", "m12_monitor_rows")
    rec.log_micro_batch(micro_row(1, 0, 0, rng, act_max_abs=value, act_max_module=index))
    rec.log_optimizer_step(step_row(0, 0, 1, 1, rng))                     # flush: one CPU copy of the tensors
    row = records.read_rows(train_dir(tmp_path) / "micro_batches.csv", MICRO_FIELDS)[0]
    assert row["act_max_abs"] == 1234.5 and row["act_max_module"] == names[-1]
    monitor.pause()
    with torch.no_grad():
        model.train_forward(batch)                        # paused: nothing recorded
    assert float(monitor.end_micro_batch(2)[0]) == -math.inf
    handle.remove()
    monitor.detach()
    rec.close("completed", {})
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
    table = raw_rows(train_dir(tmp_path) / "activation_module_max.csv")
    assert table[0] == list(records.MODULE_MAX_FIELDS) and len(table) == 1 + len(names)
    assert table[-1] == [str(len(names) - 1), names[-1], "1234.5", "1"]
    summary = json.loads((train_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["activation_max"]["max_abs"] == 1234.5 and summary["activation_max"]["module"] == names[-1]
    assert summary["activation_max"]["hook_modules"] == 14

    # no wrapper -> error; a CPU fp16 wrapper -> the aminmax probe of attach fails (torch 2.0 has no CPU Half aminmax)
    with pytest.raises(RecordsError, match="no MoELoRALinear"):
        ActivationMaxMonitor(nn.Sequential(nn.Linear(2, 2)), MoELoRALinear).attach()
    half = MoELoRALinear(nn.Linear(4, 4).half(), r=4, num_experts=2, alpha=8, dropout=0.0, scaling_rule="expert_rank",
                         gate_ctx=GateContext(2), name="half")
    with pytest.raises(RecordsError, match="aminmax"):
        ActivationMaxMonitor(nn.Sequential(half), MoELoRALinear).attach()

# ----------------------------------------------------------------------------------------------- #4
def _softmax_omega(name: str, b: int = 6, k: int = K) -> torch.Tensor:
    with seeding.torch_seed_scope(0, "test", name):
        return torch.softmax(torch.randn(b, k) * 2.0, dim=-1)


def _gate_case(case: str) -> tuple[torch.Tensor, int]:
    omega = _softmax_omega("m12_gate_cases")
    if case == "batch_axis":
        return omega.unsqueeze(1), K                  # [B, 1, K] of the released router (U model/router/nlpr.py:80)
    if case == "k_mismatch":
        return omega, K - 1
    if case == "float16":
        return omega.half(), K
    omega = omega.clone()
    if case == "nan":
        omega[2, 1] = math.nan
    elif case == "inf":
        omega[2, 1] = math.inf
    elif case == "negative":
        omega[2] = torch.tensor([-0.1, 0.6, 0.3, 0.2])
    elif case == "rowsum":
        omega[2, 0] += 2e-5                           # row sum 1 + 2e-5 > 1e-5
    return omega, K


@pytest.mark.criterion("M12-4")
@pytest.mark.parametrize("case", ["batch_axis", "k_mismatch", "nan", "inf", "negative", "rowsum", "float16"])
def test_validate_gate_rejects(case):
    """[B, 1, K], K mismatch, NaN, Inf, negative, row sum 1 + 2e-5 and float16 are rejected; fp32 softmax passes."""
    records.validate_gate(_softmax_omega("m12_gate_cases"), batch=6, num_experts=K, rowsum_atol=1e-5)
    omega, k = _gate_case(case)
    ids = [f"val:{i}" for i in range(6)]
    with pytest.raises(ValueError) as info:
        records.validate_gate(omega, batch=6, num_experts=k, rowsum_atol=1e-5, sample_ids=ids)
    if case in ("nan", "inf", "negative", "rowsum"):
        assert "val:2" in str(info.value)             # offending sample_id and values in the message (16.6.8)


@pytest.mark.criterion("M12-4")
@pytest.mark.parametrize("case", ["batch_axis", "d_mismatch", "float16"])
def test_validate_z_rejects(case):
    """z [B, 1, d], a wrong d and float16 are rejected; float32 [B, d] passes."""
    with seeding.torch_seed_scope(0, "test", "m12_z_cases"):
        z = torch.randn(5, 8)
    records.validate_z(z, batch=5, dim=8)
    bad = {"batch_axis": (z.unsqueeze(1), 8), "d_mismatch": (z, 64), "float16": (z.half(), 8)}[case]
    with pytest.raises(ValueError):
        records.validate_z(bad[0], batch=5, dim=bad[1])


# ----------------------------------------------------------------------------------------------- #5
def synthetic_encoder(d: int):
    """A SASRecEncoder with hidden size ``d`` and seeded weights (state layout of M3 §7.6.2)."""
    from ilora import sasrec

    arch = sasrec.SASRecArch(hidden_size=d, item_num=PAD_ID, state_size=HISTORY_LEN, num_heads=1, layernorm_eps=1e-5)
    with seeding.torch_seed_scope(0, "test", f"m12_sasrec_{d}"):
        state = {key: torch.randn(shape) * 0.5 for key, shape in sasrec.expected_state_shapes(arch).items()}
    encoder = sasrec.SASRecEncoder(arch, sasrec.state_digest(state))
    encoder.load_state_dict(state, strict=True)
    return encoder


def synthetic_samples(split: str, n: int, name: str) -> list[Sample]:
    """``n`` samples with ascending rows (with gaps), 3-10 distinct history items and a target outside them."""
    rng = seeding.np_rng(0, "test", name)
    rows = np.sort(rng.choice(5 * n, size=n, replace=False))
    out = []
    for i, row in enumerate(rows.tolist()):
        items = rng.choice(PAD_ID, size=11, replace=False).tolist()
        length = int(rng.integers(3, HISTORY_LEN + 1))
        out.append(Sample(split=split, row=int(row), user_id=int(1000 + i // 3), history=tuple(items[:length]),
                          target=int(items[10])))
    return out


@pytest.mark.criterion("M12-5")
@pytest.mark.parametrize("d", [8, 64])
def test_compute_representations_synthetic(d, cpu_cfg, tmp_path, torch_threads1):
    """Synthetic SASRec (d = 8, 64) + M7 Router: row order, d from the encoder, NPZ and CSV round trips, CSV columns."""
    from ilora.router import Router

    encoder = synthetic_encoder(d)
    router = Router(cpu_cfg.router, num_experts=K, z_dim=d, main_seed=0, hard_gate=None)
    samples = synthetic_samples("val", 37, f"m12_samples_{d}")
    table = records.compute_representations(samples, encoder, router, num_experts=K, batch_size=8, num_threads=1,
                                            rowsum_atol=1e-5)
    assert table.sample_ids == tuple(s.sample_id for s in samples) and table.split == "val"
    assert table.rows.tolist() == [s.row for s in samples] and table.user_ids.tolist() == [s.user_id for s in samples]
    assert table.dim == d and table.z.shape == (37, d) and table.omega.shape == (37, K)
    assert table.meta["d"] == d and table.meta["num_threads"] == 1 and router.gate_calls == 5        # 4 x 8 + 5
    with torch.no_grad():                    # the same batches through the public M1/M3/M7 API, 1 thread as well
        z_parts, omega_parts = [], []
        for start in range(0, 37, 8):
            part = samples[start:start + 8]
            hist, lens = pad_histories([s.history for s in part], PAD_ID, HISTORY_LEN)
            z_parts.append(encoder(hist, lens))
            omega_parts.append(router(z_parts[-1], [s.sample_id for s in part]))
    assert np.array_equal(table.z, torch.cat(z_parts).numpy())
    assert np.array_equal(table.omega, torch.cat(omega_parts).numpy())
    assert np.array_equal(table.argmax, np.argmax(table.omega, axis=1))

    sha = records.save_npz(tmp_path / "a.npz", table)
    assert records.save_npz(tmp_path / "b.npz", table) == sha              # the same table gives the same bytes
    with np.load(tmp_path / "a.npz", allow_pickle=False) as npz:
        assert sorted(npz.files) == sorted(records.NPZ_KEYS)
        assert npz["sample_id"].dtype.kind == "U" and npz["meta_json"].ndim == 0 and npz["z"].dtype == np.float32
        assert npz["split"].tolist() == ["val"] * 37 and npz["row"].dtype == np.int64
    back = records.load_npz(tmp_path / "a.npz")
    assert back.sample_ids == table.sample_ids and back.split == "val" and back.meta == table.meta
    for name in ("rows", "user_ids", "z", "omega", "argmax"):
        assert np.array_equal(getattr(back, name).view(np.uint8), getattr(table, name).view(np.uint8)), name
    with pytest.raises(RecordsError, match="never overwritten"):
        records.save_npz(tmp_path / "a.npz", table)

    records.write_gate_csv(tmp_path / "gates_validation.csv", table)
    header = raw_rows(tmp_path / "gates_validation.csv")[0]
    assert header == ["sample_id", "user_id", "split", *(f"gate_{k}" for k in range(K)), "argmax_expert",
                      *(f"sasrec_{j}" for j in range(d))]                 # [SPEC §4] order, d from the shape
    csv_table = records.read_gate_csv(tmp_path / "gates_validation.csv")
    assert csv_table.sample_ids == table.sample_ids and np.array_equal(csv_table.rows, table.rows)
    assert np.array_equal(csv_table.omega.view(np.uint32), table.omega.view(np.uint32))
    assert np.array_equal(csv_table.z.view(np.uint32), table.z.view(np.uint32))
    assert np.array_equal(csv_table.argmax, table.argmax) and np.array_equal(csv_table.user_ids, table.user_ids)

    z_only = records.compute_representations(samples, encoder, None, num_experts=K, batch_size=8, num_threads=1,
                                             rowsum_atol=1e-5)
    assert z_only.omega is None and z_only.argmax is None and np.array_equal(z_only.z, table.z)
    records.save_npz(tmp_path / "z_only.npz", z_only)
    with np.load(tmp_path / "z_only.npz", allow_pickle=False) as npz:
        assert "omega" not in npz.files and "argmax_expert" not in npz.files
    assert records.load_npz(tmp_path / "z_only.npz").omega is None
    with pytest.raises(RecordsError, match="needs omega"):
        records.write_gate_csv(tmp_path / "z_only.csv", z_only)
    with pytest.raises(RecordsError, match="row order"):
        records.compute_representations(samples[::-1], encoder, None, num_experts=K, batch_size=8, num_threads=1,
                                        rowsum_atol=1e-5)


# ----------------------------------------------------------------------------------------------- #6
@pytest.mark.slow
@pytest.mark.criterion("M12-6")
def test_real_sasrec_z_export(sasrec_state_path, full_data, paper_cfg):
    """Converted SASRec, z only: 66,126 / 94 / 95 rows, d = 64 read from the weights, finite, two runs bit for bit."""
    from ilora.sasrec import SASRecEncoder

    encoder = SASRecEncoder.from_config(paper_cfg, full_data.catalog, HISTORY_LEN)
    rc = paper_cfg.records
    for split, n in (("train", 66_126), ("val", 94), ("test", 95)):
        tables = [records.compute_representations(full_data.samples(split), encoder, None, num_experts=K,
                                                  batch_size=rc.export_batch_size, num_threads=rc.export_num_threads,
                                                  rowsum_atol=rc.gate_rowsum_atol) for _ in range(2)]
        a, b = tables
        assert len(a) == n and a.dim == encoder.hidden_size == encoder.arch.hidden_size == 64
        assert a.omega is None and bool(np.isfinite(a.z).all())
        assert a.sample_ids == tuple(s.sample_id for s in full_data.samples(split))
        assert np.array_equal(a.z.view(np.uint32), b.z.view(np.uint32)), split

# ----------------------------------------------------------------------------------------------- #7
def synthetic_table(split: str, n: int, name: str, *, k: int = K, d: int = 8, meta=None) -> RepresentationTable:
    rng = seeding.np_rng(0, "test", name)
    logits = rng.normal(size=(n, k)) * 2.0
    omega = np.exp(logits - logits.max(axis=1, keepdims=True))
    omega = (omega / omega.sum(axis=1, keepdims=True)).astype(np.float32)
    rows = np.arange(0, 2 * n, 2, dtype=np.int64)
    return RepresentationTable(split=split, sample_ids=tuple(f"{split}:{r}" for r in rows.tolist()), rows=rows,
                               user_ids=np.arange(n, dtype=np.int64) + 754, z=rng.normal(size=(n, d)).astype(np.float32),
                               omega=omega, argmax=np.argmax(omega, axis=1).astype(np.int64),
                               meta=meta or {"routing_mode": "dynamic"})


def records_from_table(cls, table: RepresentationTable) -> list:
    """Evaluation records whose gates are the table's omega rows (float32 values as Python floats)."""
    rng = seeding.np_rng(0, "test", f"m12_records_{table.split}_{len(table)}")
    out = []
    for i, sid in enumerate(table.sample_ids):
        cands = tuple(int(c) for c in rng.choice(PAD_ID, size=20, replace=False))
        target = cands[int(rng.integers(20))]
        kind = i % 3                                     # 0 correct, 1 valid but wrong, 2 invalid
        matched = (target,) if kind == 0 else ((cands[0] if cands[0] != target else cands[1]),) if kind == 1 else ()
        out.append(cls(sample_id=sid, split=table.split, user_id=int(table.user_ids[i]), template_id=i % 3,
                       hybrid=True, cand_ids=cands, cand_titles=tuple(f"Movie {c}" for c in cands), target_id=target,
                       target_title=f"Movie {target}", raw_text=f" Movie {target}\nrest", first_line=f"Movie {target}",
                       matched=matched, valid=kind != 2, correct=kind == 0,
                       gates=tuple(float(x) for x in table.omega[i].tolist()),
                       flags={flag: False for flag in FLAG_NAMES}, gen_token_ids=(29871, 13, 2)))
    return out


def _replace_rows(table: RepresentationTable, keep: np.ndarray, **arrays) -> RepresentationTable:
    fields = {"rows": table.rows[keep], "user_ids": table.user_ids[keep], "z": table.z[keep],
              "omega": table.omega[keep], "argmax": table.argmax[keep]}
    fields.update(arrays)
    return dataclasses.replace(table, sample_ids=tuple(np.asarray(table.sample_ids)[keep].tolist()), **fields)


@pytest.mark.criterion("M12-7")
@pytest.mark.parametrize("case", ["same", "plus_2e-5", "missing_sample", "records_one_bit"])
def test_crosscheck_gates(case, eval_record_cls):
    """Export vs evaluation NPZ vs records.jsonl: equal passes (max |dz| recorded); +2e-5, a missing sample or one
    bit of a records.jsonl gate fails."""
    evaluation = synthetic_table("val", 12, "m12_cross", meta={"source": "evaluate"})
    recs = records_from_table(eval_record_cls, evaluation)
    with seeding.torch_seed_scope(0, "test", "m12_cross_dz"):
        dz = (torch.randn(12, 8) * 1e-3).numpy()
    export = dataclasses.replace(evaluation, z=(evaluation.z + dz).astype(np.float32), meta={"source": "export"})
    keep = np.arange(12)
    if case == "same":
        result = records.crosscheck_gates(export, evaluation, recs, atol=1e-5)
        assert result["passed"] and result["max_abs_diff_omega"] == 0.0 and result["argmax_mismatches"] == 0
        assert result["max_abs_diff_z"] == pytest.approx(float(np.abs(dz.astype(np.float64)).max()), rel=1e-3)
        assert result["records_equal_eval_npz_bitwise"] and result["n"] == 12
        return
    if case == "plus_2e-5":
        omega = export.omega.copy()
        omega[3, 0] += np.float32(2e-5)
        omega[3, 1] -= np.float32(2e-5)                  # the row still sums to 1
        export = _replace_rows(export, keep, omega=omega, argmax=np.argmax(omega, axis=1).astype(np.int64))
    elif case == "missing_sample":
        export = _replace_rows(export, keep[:-1])
    else:
        gates = list(recs[2].gates)
        gates[1] = float(np.nextafter(np.float32(gates[1]), np.float32(2.0)))
        recs[2] = dataclasses.replace(recs[2], gates=tuple(gates))
    with pytest.raises(RecordsError):
        records.crosscheck_gates(export, evaluation, recs, atol=1e-5)


@pytest.mark.criterion("M12-7")
def test_crosscheck_failure_is_diagnosable(eval_record_cls):
    """A tolerance failure (16.6.5 step 7, 16.7) is a ``GateCrosscheckError`` whose message names max |d omega| with
    its worst sample, max |d z| and the argmax mismatch count, and whose ``result`` holds those numbers with
    ``passed`` false (what run_export writes to crosscheck.json); a missing sample stays a plain RecordsError."""
    evaluation = synthetic_table("val", 12, "m12_cross", meta={"source": "evaluate"})
    recs = records_from_table(eval_record_cls, evaluation)
    with seeding.torch_seed_scope(0, "test", "m12_cross_dz"):
        dz = (torch.randn(12, 8) * 1e-3).numpy()
    omega = evaluation.omega.copy()
    order = np.argsort(omega[3])
    omega[3, [order[-1], order[-2]]] = omega[3, [order[-2], order[-1]]]       # another gate for one sample
    export = dataclasses.replace(evaluation, z=(evaluation.z + dz).astype(np.float32), omega=omega,
                                 argmax=np.argmax(omega, axis=1).astype(np.int64), meta={"source": "export"})
    max_omega = float(np.abs(omega.astype(np.float64) - evaluation.omega.astype(np.float64)).max())
    max_z = float(np.abs(export.z.astype(np.float64) - evaluation.z.astype(np.float64)).max())
    with pytest.raises(records.GateCrosscheckError) as info:
        records.crosscheck_gates(export, evaluation, recs, atol=1e-5)
    assert isinstance(info.value, RecordsError)
    message, result = str(info.value), info.value.result
    sample = evaluation.sample_ids[3]
    assert f"max |export omega - eval omega| = {max_omega!r} > 1e-05 (worst sample {sample})" in message
    assert f"max |export z - eval z| = {max_z!r}, argmax mismatches 1 of 12" in message
    assert result["passed"] is False and result["max_abs_diff_omega"] == max_omega > 1e-5
    assert (result["max_abs_diff_omega_sample_id"], result["argmax_mismatches"], result["max_abs_diff_z"]) == (
        sample, 1, max_z)
    assert result["argmax_mismatch_sample_ids"] == [sample] and result["n"] == 12 and result["atol"] == 1e-5
    missing = _replace_rows(export, np.arange(11))
    with pytest.raises(RecordsError) as info:
        records.crosscheck_gates(missing, evaluation, recs, atol=1e-5)
    assert not isinstance(info.value, records.GateCrosscheckError)


def _trainable_state(name: str, *, gate: bool, extra: Mapping = ()) -> dict:
    """A read_trainable_state-like dict (§14.6.6): gate key only for dynamic, other keys that export must ignore."""
    with seeding.torch_seed_scope(0, "test", name):
        state = {"projector.mlp_proj.0.weight": torch.randn(3, 3),
                 "llm.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 2),
                 "generated_frozen.embed_rows": torch.randn(5, 7)}
        if gate:
            state["omega_source.gate.proj.weight"] = torch.randn(K, 64)
    state.update(dict(extra))
    return state


@pytest.mark.criterion("M12-7")
@pytest.mark.parametrize("mode", ["dynamic", "cluster_hard"])
def test_export_router_loading(mode, profile_cfg, smoke_data, fake_hard_gate, request):
    """Export Router: dynamic loads only the ``omega_source.`` keys (strict); cluster_hard gives the one-hot omega."""
    ids = [s.sample_id for s in smoke_data.val[:6]]
    with seeding.torch_seed_scope(0, "test", "m12_router_z"):
        z = torch.randn(6, 64)
    if mode == "dynamic":
        cfg = profile_cfg("smoke_cpu")
        state = _trainable_state("m12_router_state", gate=True)
        router = records.export_router(cfg, state, z_dim=64)
        weight = state["omega_source.gate.proj.weight"]
        assert list(router.state_dict()) == ["gate.proj.weight"] and torch.equal(router.gate.proj.weight, weight)
        assert not router.training
        omega = router(z, ids)
        assert torch.equal(omega, torch.softmax(torch.nn.functional.linear(z, weight), dim=-1))
        assert router.gate_calls == 1
        for broken in ({"omega_source.gate.proj.bias": torch.zeros(K)},
                       {"omega_source.gate.proj.weight": torch.zeros(K, 63)}):
            with pytest.raises(RecordsError):
                records.export_router(cfg, {**state, **broken}, z_dim=64)
        with pytest.raises(RecordsError):
            records.export_router(cfg, _trainable_state("m12_router_state", gate=False), z_dim=64)
        return
    if importlib.util.find_spec("ilora.cluster") is not None:     # M15 present: the export's own preparation
        from ilora.sasrec import SASRecEncoder

        cluster_dir = request.getfixturevalue("smoke_cluster_dir")
        cfg = profile_cfg("smoke_cpu", "router.routing_mode=cluster_hard",
                          f"cluster.artifact_dir={json.dumps(Path(cluster_dir).as_posix())}")
        encoder = SASRecEncoder.from_config(cfg, smoke_data.catalog, HISTORY_LEN)
        gate, live = records._cluster_hard_gate(cfg, smoke_data, encoder, request.getfixturevalue("repo_root"))
        assert isinstance(live, dict)
    else:                                                          # M15 not built yet: the HardGate protocol stand-in
        cfg = profile_cfg("smoke_cpu", "router.routing_mode=cluster_hard", "cluster.artifact_dir=unused/cluster")
        gate = fake_hard_gate({s.sample_id: s.row % K for s in smoke_data.val}, K)
    router = records.export_router(cfg, _trainable_state("m12_router_state", gate=False), z_dim=64, hard_gate=gate)
    assert router.state_dict() == {}
    lookups = gate.lookups
    omega = router(z, ids)
    assert omega.dtype == torch.float32 and tuple(omega.shape) == (6, K)
    assert bool(((omega == 0) | (omega == 1)).all()) and torch.equal(omega.sum(dim=1), torch.ones(6))
    assert router.gate_calls == 0 and gate.lookups == lookups + 6
    with pytest.raises(RecordsError):
        records.export_router(cfg, _trainable_state("m12_router_state", gate=True), z_dim=64, hard_gate=gate)

# ----------------------------------------------------------------------------------------------- #12
def split_metrics(split: str, recs: list) -> dict:
    n, v, c = len(recs), sum(r.valid for r in recs), sum(r.correct for r in recs)
    return {"split": split, "n_total": n, "n_valid": v, "n_correct": c, "valid_ratio": v / n,
            "hr_at_1": c / v if v else 0.0, "combined_metric": c / n}


def make_metrics(split: str, recs: list, ckpt_tag: str) -> dict:
    """metrics.json content of M11 §15.6.9 for synthetic records (core keys, M11 blocks, meta)."""
    metrics = split_metrics(split, recs)
    metrics["epoch"] = None if ckpt_tag == "init" else int(ckpt_tag[5:])
    metrics["ckpt_tag"] = ckpt_tag
    metrics["aux_protocols"] = {} if split == "val" else {
        f"drop_last_bs{b}": split_metrics(split, recs[:(len(recs) // b) * b]) for b in records.AUX_BATCH_SIZES}
    metrics["supplementary"] = {"wilson95": {"valid_ratio": [0.1, 0.9]}}
    metrics["meta"] = {"seed": 1234, "dataset": "movielens", "routing_mode": "dynamic", "gate_calls": 12,
                       "n_batches": 12}
    return metrics


def _write_eval_case(case: str, cls) -> tuple[list, dict, list]:
    table = synthetic_table("test", 95, "m12_write_eval")
    recs = records_from_table(cls, table)
    expected = list(table.sample_ids)
    metrics = make_metrics("test", recs, "epoch1")
    if case == "n_94":
        recs = recs[:94]
        metrics = make_metrics("test", recs, "epoch1")
    elif case == "duplicate_id":
        recs[5] = dataclasses.replace(recs[5], sample_id=recs[4].sample_id)
    elif case == "correct_not_valid":
        recs[1] = dataclasses.replace(recs[1], correct=True, valid=False, matched=())
        metrics = make_metrics("test", recs, "epoch1")
    elif case == "metrics_counts":
        metrics["n_correct"] += 1
    elif case == "aux_protocols":
        metrics["aux_protocols"]["drop_last_bs4"] = split_metrics("test", recs)          # all 95, not the first 92
    return recs, metrics, expected


@pytest.mark.criterion("M12-12")
@pytest.mark.parametrize("case", ["n_94", "duplicate_id", "correct_not_valid", "metrics_counts", "aux_protocols"])
def test_write_eval_rejects(case, eval_record_cls, tmp_path):
    """94 of 95 samples, a duplicate id, correct without valid, counts that disagree with the records and an
    aux_protocols block not equal to the first 92 (and 88) records are rejected; nothing is written."""
    recs, metrics, expected = _write_eval_case(case, eval_record_cls)
    with pytest.raises(RecordsError):
        records.write_eval(tmp_path, recs, metrics, expected_ids=expected, num_experts=K)
    assert not (tmp_path / "records.jsonl").exists() and not (tmp_path / "metrics.json").exists()
    if case == "aux_protocols":
        recs, metrics, expected = _write_eval_case("valid", eval_record_cls)
        assert sum(r.correct for r in recs[88:]) > 0          # the 88-record block differs from the 95-record count
        metrics["aux_protocols"]["drop_last_bs8"] = split_metrics("test", recs)
        with pytest.raises(RecordsError, match="drop_last_bs8"):
            records.write_eval(tmp_path, recs, metrics, expected_ids=expected, num_experts=K)


@pytest.mark.criterion("M12-12")
def test_write_eval_deterministic(eval_record_cls, tmp_path):
    """The same records twice give the same records.jsonl sha256; init (epoch null) and the M11 blocks pass;
    read_eval_records gives the records back."""
    from ilora import run

    recs, metrics, expected = _write_eval_case("valid", eval_record_cls)
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        records.write_eval(tmp_path / name, recs, metrics, expected_ids=expected, num_experts=K)
    assert run.content_sha256(tmp_path / "a" / "records.jsonl") == run.content_sha256(tmp_path / "b" / "records.jsonl")
    first = (tmp_path / "a" / "records.jsonl").read_bytes().split(b"\n")[0]
    keys = list(json.loads(first))
    assert keys == [*records.EVAL_RECORD_FIELDS[:14], "gate_0", "gate_1", "gate_2", "gate_3", "flags", "gen_token_ids"]
    assert b", " not in first and b"\r" not in (tmp_path / "a" / "records.jsonl").read_bytes()
    written = json.loads((tmp_path / "a" / "metrics.json").read_text(encoding="utf-8"))
    assert list(written)[:8] == list(records.METRICS_CORE) and list(written)[-1] == "meta"
    back = records.read_eval_records(tmp_path / "a" / "records.jsonl")
    assert len(back) == 95
    for got, want in zip(back, recs):
        assert all(getattr(got, f) == getattr(want, f) for f in records.EVAL_RECORD_FIELDS)
    with pytest.raises(RecordsError, match="never overwritten"):
        records.write_eval(tmp_path / "a", recs, metrics, expected_ids=expected, num_experts=K)
    # aux blocks given as SplitMetrics-like dataclasses (M11 15.6.1) are written as their fields
    SplitMetricsLike = dataclasses.make_dataclass("SplitMetricsLike", list(records.SPLIT_METRICS_FIELDS), frozen=True)
    as_objects = {**metrics, "aux_protocols": {name: SplitMetricsLike(**block)
                                               for name, block in metrics["aux_protocols"].items()}}
    (tmp_path / "c").mkdir()
    records.write_eval(tmp_path / "c", recs, as_objects, expected_ids=expected, num_experts=K)
    assert json.loads((tmp_path / "c" / "metrics.json").read_text(encoding="utf-8")) == written
    # init on validation: epoch null, aux {}, M11 blocks
    table = synthetic_table("val", 10, "m12_write_eval_init")
    val_recs = records_from_table(eval_record_cls, table)
    (tmp_path / "init").mkdir()
    records.write_eval(tmp_path / "init", val_recs, make_metrics("val", val_recs, "init"),
                       expected_ids=list(table.sample_ids), num_experts=K)
    init = json.loads((tmp_path / "init" / "metrics.json").read_text(encoding="utf-8"))
    assert init["epoch"] is None and init["ckpt_tag"] == "init" and init["aux_protocols"] == {}
    (tmp_path / "bad").mkdir()
    with pytest.raises(RecordsError, match="does not match ckpt_tag"):
        records.write_eval(tmp_path / "bad", val_recs, {**make_metrics("val", val_recs, "init"), "epoch": 0},
                           expected_ids=list(table.sample_ids), num_experts=K)


# ----------------------------------------------------------------------------------------------- #13
@pytest.mark.criterion("M12-13")
def test_recorder_resume(tmp_path):
    """Rows beyond the loaded boundary move to *.incomplete-<UTC>.csv, the files keep (tau_done, global_step,
    epochs_complete) rows and logging continues without duplicates; header or row mismatches -> ValueError."""
    def run_two_epochs(run_dir: Path) -> TrainRecorder:
        rec = TrainRecorder(run_dir, recorder_cfg(flush_every=2), num_experts=K, total_micro_batches=24, total_steps=6)
        drive(rec, epochs=range(2), steps_per_epoch=3, n_micro=4, name="m12_resume")
        return rec

    reference = tmp_path / "uninterrupted"
    run_two_epochs(reference).close("completed", {})
    interrupted = tmp_path / "interrupted"
    first = run_two_epochs(interrupted)                     # epoch 1 rows were written, then the run "died"
    first.close("failed", {})
    resumed = TrainRecorder.resume(interrupted, recorder_cfg(flush_every=2), num_experts=K, total_micro_batches=24,
                                   total_steps=6, tau_done=12, global_step=3, epochs_complete=1)
    folder = train_dir(interrupted)
    assert len(records.read_rows(folder / "micro_batches.csv", MICRO_FIELDS)) == 12
    assert len(records.read_rows(folder / "optimizer_steps.csv", step_fields(K))) == 3
    assert len(records.read_rows(folder / "epochs.csv", EPOCH_FIELDS)) == 1
    moved = {p.name.split(".incomplete-")[0]: p for p in folder.glob("*.incomplete-*")}
    assert set(moved) == {"micro_batches", "optimizer_steps", "epochs", "activation_module_max", "run_summary"}
    assert [int(r[0]) for r in raw_rows(moved["micro_batches"])[1:]] == list(range(13, 25))
    assert [int(r[0]) for r in raw_rows(moved["optimizer_steps"])[1:]] == [3, 4, 5]
    with pytest.raises(RecordsError, match="tau must be 13"):
        resumed.log_micro_batch(micro_row(12, 1, 3, seeding.np_rng(0, "test", "x")))
    drive(resumed, epochs=range(1, 2), steps_per_epoch=3, n_micro=4, name="m12_resume_epoch1")
    resumed.close("completed", {})
    timeless = {"time_s", "loss", "loss_mean", "loss_hybrid_mean", "grad_norm_lora", "grad_norm_gate",
                "grad_norm_total", *(f"omega_mean_{k}" for k in range(K))}
    for name, fields in (("micro_batches.csv", MICRO_FIELDS), ("optimizer_steps.csv", step_fields(K)),
                         ("epochs.csv", EPOCH_FIELDS)):
        a = records.read_rows(train_dir(reference) / name, fields)
        b = records.read_rows(folder / name, fields)
        assert [{k: v for k, v in r.items() if k not in timeless} for r in a] == \
               [{k: v for k, v in r.items() if k not in timeless} for r in b], name
    summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["micro_batches"] == 24 and summary["activation_max"]["module_maxima_from_tau"] == 13

    def broken(change) -> Path:
        run_dir = tmp_path / f"broken_{change}"
        run_two_epochs(run_dir).close("failed", {})
        path = train_dir(run_dir) / ("epochs.csv" if change == "header" else "micro_batches.csv")
        rows = raw_rows(path)
        if change == "header":
            rows[0][-1] = "seconds"
        else:
            del rows[5]                                      # tau 5 is inside the boundary
        path.write_bytes(records._csv_bytes(rows))
        return run_dir

    for change in ("header", "missing_row"):
        with pytest.raises(ValueError):
            TrainRecorder.resume(broken(change), recorder_cfg(), num_experts=K, total_micro_batches=24, total_steps=6,
                                 tau_done=12, global_step=3, epochs_complete=1)
