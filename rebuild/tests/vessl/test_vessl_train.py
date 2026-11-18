"""M10 GPU 학습 시험 (DESIGN.md §14.8 #16, #19, §14.6.10, §17.6.12). VESSL_REQUIRED: skipped locally.

The small GPU training of §17.6.12: ``smoke_gpu_100`` with ``data.smoke_users=[1,10,10]``, ``train.epochs=1``,
``train.grad_accum=2`` and ``train.warmup_divisor=4`` gives 132 train samples (user 117), 16 micro-batches of 8,
8 optimizer steps and w = 2 (``warmup_divisor`` 20 would give w = 0, which M10 refuses). It runs the real fp16
autocast and ``torch.cuda.amp.GradScaler`` path of M10 (§14.6.3) through M0 ``run_command`` into a tmp folder.
``test_resume_restores_cuda_state`` runs the same data for 2 epochs (T 16, w 4) through a failure and
``train --resume``: the CUDA part of the resume path (enabled GradScaler state, CUDA RNG, Adam state on cuda:0)
that the CPU tests (M10-7, M10-8) cannot reach, before a real E2, E4 or E1 crash needs it.
"""
from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.vessl

SMALL_GPU_SETS = ("data.smoke_users=[1,10,10]", "train.epochs=1", "train.grad_accum=2", "train.warmup_divisor=4")
NAN_STEP = 5                         # optimizer step 5 = micro-batches tau 11 and 12 (grad_accum 2)
RESUME_SETS = ("data.smoke_users=[1,10,10]", "train.epochs=2", "train.grad_accum=2", "train.warmup_divisor=4",
               "eval.batch_invariance_check=false")


def _clone_state(value: Any) -> Any:
    """A copy of an optimizer ``state_dict()`` value: the live Adam moments change in place after the spy returns."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    return value


def fail_then_resume(cfg: Any, sets: tuple, root: Path, llm_path: Any, monkeypatch: Any, *, tokenizer_dir: Any = None
                     ) -> tuple[Path, dict, dict]:
    """M0 ``run_command('train')`` that fails when epoch 1's plan is built (``checkpoints/epoch0`` COMPLETE), then
    ``train --resume`` as main.py does it (``load_run_config``), with a spy that records right after M10
    ``load_checkpoint`` the GradScaler ``state_dict()``, the CPU and (CUDA) RNG states and a copy of the optimizer
    ``state_dict()``. Returns (run folder, epoch0/train_state.pt, the recorded values)."""
    import torch

    from ilora import commands, prompts, run
    from ilora import train as train_module

    original_plan = prompts.build_batch_plan

    def failing_plan(samples, epoch, plan_cfg):
        if epoch == 1:
            raise RuntimeError("injected failure at the start of epoch 1")
        return original_plan(samples, epoch, plan_cfg)

    monkeypatch.setattr(prompts, "build_batch_plan", failing_plan)
    with pytest.raises(RuntimeError, match="injected failure at the start of epoch 1"):
        commands.run_command("train", cfg, list(sets), profile="vessl_resume", root=root, llm_path=llm_path,
                             tokenizer_dir=tokenizer_dir, argv=["fail_then_resume"])
    monkeypatch.setattr(prompts, "build_batch_plan", original_plan)
    (run_dir,) = [p for p in Path(root).iterdir() if p.is_dir()]
    assert run.read_meta(run_dir)["status"] == "failed"
    assert (run_dir / "checkpoints" / "epoch0" / "COMPLETE").is_file()
    assert not (run_dir / "checkpoints" / "epoch1").exists()
    saved = torch.load(run_dir / "checkpoints" / "epoch0" / "train_state.pt", map_location="cpu", weights_only=True)
    original_load = train_module.load_checkpoint
    seen: dict = {}

    def load_spy(ckpt_dir, model, optimizer, scaler, load_cfg, **kwargs):
        info = original_load(ckpt_dir, model, optimizer, scaler, load_cfg, **kwargs)
        seen.update(device=model.device, scaler=dict(scaler.state_dict()), cpu_rng=torch.get_rng_state(),
                    optimizer=_clone_state(optimizer.state_dict()),
                    cuda_rng=torch.cuda.get_rng_state(model.device) if model.device.type == "cuda" else None)
        return info

    monkeypatch.setattr(train_module, "load_checkpoint", load_spy)
    resume_cfg, profile = commands.load_run_config(run_dir)
    commands.run_command("train", resume_cfg, [], profile=profile, root=run_dir.parent, resume=run_dir,
                         llm_path=llm_path, tokenizer_dir=tokenizer_dir, argv=["fail_then_resume", "--resume"])
    monkeypatch.setattr(train_module, "load_checkpoint", original_load)
    meta = run.read_meta(run_dir)
    assert meta["status"] == "completed" and [r["previous_status"] for r in meta["resumes"]] == ["failed"]
    return run_dir, saved, seen


def check_loaded_state(saved: dict, seen: dict) -> None:
    """The values right after ``load_checkpoint`` equal ``epoch0/train_state.pt``: GradScaler dict, CPU and CUDA RNG
    states, Adam ``step`` values and moments (bit for bit, on the model device) of the same parameters."""
    import torch

    assert seen["scaler"] == saved["scaler"]
    assert torch.equal(seen["cpu_rng"], saved["rng"]["cpu"])
    if seen["device"].type == "cuda":
        assert saved["rng"]["cuda"] is not None and torch.equal(seen["cuda_rng"], saved["rng"]["cuda"])
    state, want = seen["optimizer"]["state"], saved["optimizer"]["state"]
    assert state and sorted(state) == sorted(want)
    assert seen["optimizer"]["param_groups"] == saved["optimizer"]["param_groups"]
    for index, values in want.items():
        assert float(state[index]["step"]) == float(values["step"]), index
        for key in ("exp_avg", "exp_avg_sq"):
            got = state[index][key]
            assert got.device == seen["device"] and torch.equal(got.cpu(), values[key]), (index, key)


def check_resumed_records(run_dir: Path, cfg: Any) -> dict:
    """``records/train/`` of the resumed run: contiguous tau, step and epoch rows (no duplicate, no gap), and the
    deterministic columns (tau, epoch, step, hybrid, curriculum_p, n_micro, n_hybrid, tau range, lr per group) equal
    the M4 plan and M10 ``lr_at``. Losses, gradient norms and times are not compared (D16). Returns the counts."""
    from ilora import records, run
    from ilora import train as train_module
    from ilora.data import load_dataset
    from ilora.prompts import build_batch_plan

    folder = run_dir / "records" / "train"
    groups = [g for g in train_module.GROUP_ORDER if g != "gate" or cfg.router.routing_mode == "dynamic"]
    micro = records.read_rows(folder / "micro_batches.csv", records.MICRO_FIELDS)
    steps = records.read_rows(folder / "optimizer_steps.csv", records.step_fields(cfg.lora.num_experts))
    epochs = records.read_rows(folder / "epochs.csv", records.EPOCH_FIELDS)
    data = load_dataset(cfg, run.REPO_ROOT)
    sc = train_module.step_counts(len(data.train), cfg.train)
    assert [r["tau"] for r in micro] == list(range(1, sc.total_micro + 1))
    assert [r["step"] for r in steps] == list(range(sc.total_steps))
    assert [r["epoch"] for r in epochs] == list(range(sc.epochs))
    windows = list(sc.window_sizes()) * sc.epochs
    for epoch in range(sc.epochs):
        plan = build_batch_plan(data.train, epoch, cfg)
        rows = [r for r in micro if r["epoch"] == epoch]
        assert [(r["tau"], r["hybrid"]) for r in rows] == [(e.tau, e.hybrid) for e in plan.entries], epoch
        assert [r["curriculum_p"] for r in rows] == [
            train_module.curriculum_probability(cfg.prompt.train_curriculum, e.tau, sc.total_micro)
            for e in plan.entries]
        assert (epochs[epoch]["plan_digest"], epochs[epoch]["n_hybrid"]) == (plan.digest(), plan.hybrid_count())
    for s, row in enumerate(steps):
        window = [r for r in micro if r["step"] == s]
        lrs = train_module.group_lrs(s, sc, cfg.train, groups)
        assert row["epoch"] == s // sc.steps_per_epoch and row["n_micro"] == len(window) == windows[s], s
        assert row["n_hybrid"] == sum(r["hybrid"] for r in window), s
        assert (row["tau_first"], row["tau_last"]) == (window[0]["tau"], window[-1]["tau"]), s
        assert (row["lr_projector"], row["lr_lora"], row["lr_gate"]) == (lrs["projector"], lrs["lora"],
                                                                         lrs.get("gate")), s
    return {"micro": len(micro), "step": len(steps), "epoch": len(epochs), "steps_per_epoch": sc.steps_per_epoch,
            "scales": [[r["scale_before"], r["scale_after"]] for r in steps]}


@pytest.mark.criterion("M10-16")
def test_nan_step_skipped(llm_path, profile_cfg, tmp_path, monkeypatch, measurements):
    """A NaN multiplied into the loss of step 5 (``train_forward`` monkeypatched): that step is skipped (``skipped``
    1, scale halved, gradient norm not finite, two non-finite losses), step 6 has a finite gradient norm at the halved
    scale, and the training completes (§14.8 #16)."""
    from ilora import commands, run
    from ilora.model import ILoRAModel

    cfg = profile_cfg("smoke_gpu_100", *SMALL_GPU_SETS)
    original = ILoRAModel.train_forward
    nan_taus = (2 * NAN_STEP + 1, 2 * NAN_STEP + 2)

    def nan_forward(self, batch):
        out = original(self, batch)
        if batch.tau in nan_taus:
            return dataclasses.replace(out, loss=out.loss * float("nan"))
        return out

    monkeypatch.setattr(ILoRAModel, "train_forward", nan_forward)
    run_dir = commands.run_command("train", cfg, list(SMALL_GPU_SETS), profile="vessl_nan_step", root=tmp_path,
                                   llm_path=llm_path, argv=["test_nan_step_skipped"])
    assert run.read_meta(run_dir)["status"] == "completed"
    with open(run_dir / "records" / "train" / "optimizer_steps.csv", "r", encoding="utf-8", newline="") as f:
        steps = sorted(csv.DictReader(f), key=lambda row: int(row["step"]))
    assert [int(row["step"]) for row in steps] == list(range(8))
    step, after = steps[NAN_STEP], steps[NAN_STEP + 1]
    assert (int(step["tau_first"]), int(step["tau_last"])) == nan_taus
    assert step["skipped"] == "1" and int(step["nonfinite_loss"]) == 2
    assert float(step["scale_after"]) == float(step["scale_before"]) * 0.5
    assert not math.isfinite(float(step["grad_norm_total"]))
    assert after["skipped"] == "0" and math.isfinite(float(after["grad_norm_total"]))
    assert float(after["scale_before"]) == float(step["scale_after"])
    measurements["m10_nan_step_skipped"] = {
        "skipped_steps": [int(row["step"]) for row in steps if row["skipped"] == "1"],
        "scale": [[float(row["scale_before"]), float(row["scale_after"])] for row in steps],
        "grad_norm_total": [row["grad_norm_total"] for row in steps]}


@pytest.mark.criterion("M10-19")
def test_resume_restores_cuda_state(llm_path, profile_cfg, tmp_path, monkeypatch, measurements):
    """§14.8 #19, the CUDA resume path (§14.6.6, §14.6.10): ``smoke_gpu_100`` with ``RESUME_SETS`` (16 micro-batches
    and 8 steps per epoch, T 16, w 4) in fp16 fails when epoch 1's plan is built, then ``train --resume`` reloads
    epoch 0. Right after ``load_checkpoint`` the enabled GradScaler state, the CUDA and CPU RNG states and the Adam
    state (moments on cuda:0, bit for bit) equal ``epoch0/train_state.pt``; step 8 starts at the scale step 7 ended
    with; the records are 32 micro, 16 step and 2 epoch rows whose deterministic columns follow the plan and
    ``lr_at``. Losses, gradient norms and parameters are not compared across GPU runs (D16)."""
    cfg = profile_cfg("smoke_gpu_100", *RESUME_SETS)
    assert cfg.train.precision == "fp16_amp" and cfg.model.device == "cuda:0"
    run_dir, saved, seen = fail_then_resume(cfg, RESUME_SETS, tmp_path, llm_path, monkeypatch)
    assert seen["device"].type == "cuda" and set(saved["scaler"]) == {"scale", "growth_factor", "backoff_factor",
                                                                      "growth_interval", "_growth_tracker"}
    check_loaded_state(saved, seen)
    counts = check_resumed_records(run_dir, cfg)
    assert (counts["micro"], counts["step"], counts["epoch"], counts["steps_per_epoch"]) == (32, 16, 2, 8)
    scales = counts["scales"]
    assert scales[8][0] == scales[7][1] == saved["scaler"]["scale"]
    measurements["m10_resume_cuda_state"] = {"scaler_epoch0": saved["scaler"], "scales": scales}
