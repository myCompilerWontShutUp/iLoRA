"""M10 학습 루프 (DESIGN.md §14; 14.6.1 공개 API, 14.6.2-14.6.10, 결정 D3, D5, D6, D9, D11, D12, D16, D19, D24).

``train`` runs the whole ``train`` command inside the run folder that M0 ``run_command`` prepared:

- preparation in a fixed order (§14.6.4 step 0): data (M1), templates (M4), tokenizer (M5); for
  cluster_hard only the M15 artifacts -> expert mapping -> ``ClusterHardGate``; ``build_model`` (M9); for
  cluster_hard the live z check; the Adam optimizer of the D19 groups and the M0 GradScaler; the M12
  ``TrainRecorder`` and ``ActivationMaxMonitor``; the frozen-parameter checksum and the SASRec digest; then
  the one ``("torch_global",)`` seed of a new training (§14.6.5);
- per epoch the M4 ``BatchPlan`` consumed in accumulation windows of ``train.grad_accum`` micro-batches (the
  last window of an epoch keeps the rest and still takes one step, D24): ``loss / train.grad_accum`` backward
  per micro-batch inside the M0 autocast (the last window divides by ``grad_accum`` too, D24),
  ``scaler.unscale_``, group gradient norms, the gradient presence rule, ``scaler.step``/``update`` with the
  skip rule, and the lr of ``lr_at`` set once before each window (D5, D6, §14.6.2);
- at each epoch end the checks, the checkpoint (``trainable.pt``, ``train_state.pt``, ``STATE_SAVED``),
  the validation of M11, the epoch row of M12, ``val_metrics.json`` and ``COMPLETE`` (§14.6.6, §14.6.7);
- after the last epoch the selection by integer validation answers (D11), ``eval/selection.json`` and the
  test evaluations of best_val and last. ``train --resume`` continues at an epoch boundary (§14.6.10).

Checkpoints are read only through ``read_trainable_state``/``load_model_state`` (M11 eval command, M12
export) and ``load_checkpoint`` (resume). Errors of this module are ``TrainingError``; errors of M1-M9,
M11, M12 and M15 propagate unchanged (§14.6.1). Python ``random`` and the numpy global generator are not
used (§14.6.5).
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from . import run, seeding
from .config import config_hash
from .precision import autocast_for, make_grad_scaler

if TYPE_CHECKING:   # type names only: M4 plan entries, M5 batches, M11 metrics, M0 config sections
    from .batching import TrainBatch
    from .data import Sample
    from .evaluate import SplitMetrics
    from .prompts import PlanEntry
    from .sections import Config, TrainConfig

logger = logging.getLogger("ilora.train")


class TrainingError(RuntimeError):
    """A rule of the M10 training loop is broken (§14.6.1); the message names the step, epoch or file."""


# ----------------------------------------------------------------------------
# Constants (§14.6.1, §14.6.6, §14.6.7)
# ----------------------------------------------------------------------------
GROUP_ORDER: Final = ("projector", "gate", "lora")              # D19 groups = M9 param_groups keys (§14.6.3)
CHECKPOINTS_DIR: Final = "checkpoints"                          # checkpoints/epoch{e}/ (§3.1 실행 폴더)
TRAINABLE_FILE: Final = "trainable.pt"
TRAIN_STATE_FILE: Final = "train_state.pt"
STATE_SAVED_MARKER: Final = "STATE_SAVED"                       # both files written (their sha256)
VAL_METRICS_FILE: Final = "val_metrics.json"
COMPLETE_MARKER: Final = "COMPLETE"                             # the three sha256: the epoch is done
SELECTION_FILE: Final = "selection.json"                        # eval/selection.json (§14.6.7)
TRAIN_STATE_FORMAT: Final = "ilora-rebuild-train-state-v1"      # train_state.pt "format_version" [결정]
EPOCH_DIR_RE: Final = re.compile(r"epoch(0|[1-9][0-9]*)")
SUMMARY_PREFIX: Final = "train step"                            # run.log summary lines (§14.6.8)
SELECTION_RULE: Final = ("best_val = the epoch with the most correct validation answers (integer n_correct; "
                         "every epoch has the same N), the later epoch on ties; last = the final epoch; test is "
                         "evaluated for both, best_val is the primary result (D11)")
# Keys of train_state.pt that train() passes to save_checkpoint (§14.6.6); the rest comes from the objects.
# frozen_checksum is the start value (checked on reload); frozen_checksum_at_save is the value recomputed at the end
# of the saved epoch, which the epoch-end check compared with it (so the files show that the check held, 17.6.7 #2).
STATE_KEYS: Final = ("epoch", "global_step", "tau_done", "config_hash", "input_hashes", "cluster",
                     "sasrec_state_digest", "frozen_checksum", "frozen_checksum_at_save", "counters", "epoch_stats",
                     "step_counts")
# Fields of the provenance that a checkpoint must match on reload (§14.6.6 읽기는 strict).
PROVENANCE_KEYS: Final = ("config_hash", "input_hashes", "router_provenance", "sasrec_state_digest", "cluster")


def _is_int(value: object) -> bool:
    """Python or numpy integer, bool excluded."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def group_hparams(group: str, tc: "TrainConfig") -> tuple[float, float]:
    """``(lr_scale, weight_decay)`` of one D19 group from the scalar keys of §14.6.9."""
    table = {"projector": (tc.lr_scale_projector, tc.weight_decay_projector),
             "gate": (tc.lr_scale_gate, tc.weight_decay_gate),
             "lora": (tc.lr_scale_lora, tc.weight_decay_lora)}
    if group not in table:
        raise TrainingError(f"unknown parameter group {group!r}, expected one of {GROUP_ORDER}")
    scale, decay = table[group]
    return float(scale), float(decay)


# ----------------------------------------------------------------------------
# Training length and lr schedule (§14.6.2, D5, D6, D24)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StepCounts:
    """Lengths computed from the number of train samples (§14.6.2): paper_spec 8,265 / 9 / 517 / 2,585 / 129."""
    n_samples: int
    micro_batch: int
    accum: int
    epochs: int
    micro_per_epoch: int             # n_samples // micro_batch (drop_last)
    tail: int                        # micro_per_epoch % accum (0: no partial window)
    steps_per_epoch: int             # ceil(micro_per_epoch / accum)
    total_steps: int                 # T
    warmup_steps: int                # w = T // warmup_divisor
    total_micro: int                 # epochs x micro_per_epoch (the T of D3)

    def window_sizes(self) -> tuple[int, ...]:
        """Micro-batches per optimizer step of one epoch: ``accum`` each, the last one ``tail`` (D24)."""
        full = self.micro_per_epoch // self.accum
        return (self.accum,) * full + ((self.tail,) if self.tail else ())


def step_counts(n_samples: int, tc: "TrainConfig") -> StepCounts:
    """§14.6.2 from N train samples and the ``train`` section; a smoke misconfiguration is a ``TrainingError``."""
    values = {"n_samples": n_samples, "train.micro_batch_size": tc.micro_batch_size, "train.grad_accum": tc.grad_accum,
              "train.epochs": tc.epochs, "train.warmup_divisor": tc.warmup_divisor}
    for name, value in values.items():
        if not _is_int(value) or int(value) < 1:
            raise TrainingError(f"{name} must be an int >= 1, got {value!r}")
    n, b, a, epochs = int(n_samples), int(tc.micro_batch_size), int(tc.grad_accum), int(tc.epochs)
    if b > n:
        raise TrainingError(f"train.micro_batch_size {b} > {n} train samples: no full micro-batch (drop_last)")
    micro = n // b
    steps = -(-micro // a)
    if steps < 1:
        raise TrainingError(f"{micro} micro-batches per epoch give no optimizer step")
    total = epochs * steps
    warmup = total // int(tc.warmup_divisor)
    if warmup < 1:
        raise TrainingError(f"warmup w = T // train.warmup_divisor = {total} // {tc.warmup_divisor} = 0; choose a "
                            "smaller warmup_divisor for this data size (smoke_cpu uses 4)")
    return StepCounts(n_samples=n, micro_batch=b, accum=a, epochs=epochs, micro_per_epoch=micro, tail=micro % a,
                      steps_per_epoch=steps, total_steps=total, warmup_steps=warmup, total_micro=epochs * micro)


def lr_at(step: int, sc: StepCounts, tc: "TrainConfig") -> float:
    """lr of optimizer step ``step`` (from 0, skipped steps included) [논문 App E p.17] [v1 App E p.16] D5, D6.

    s < w: ``start + (peak - start) * s / w``; s >= w: ``floor + (peak - floor) * 0.5 * (1 + cos(pi (s - w) /
    (T - w)))``. The cosine continues from the end of the warmup ([코드] U optims.py:20-36, ``cur_step -
    warmup_steps`` :25); the ``cur_epoch == 0`` condition of :22 has no effect because w < steps per epoch.
    The operations follow U optims.py:28-36 in the same order, so the values are equal bit for bit.
    """
    if not _is_int(step) or not 0 <= int(step) < sc.total_steps:
        raise TrainingError(f"step must be an int in 0..{sc.total_steps - 1}, got {step!r}")
    s, w, total = int(step), sc.warmup_steps, sc.total_steps
    start, peak, floor = float(tc.lr_warmup_start), float(tc.lr_peak), float(tc.lr_floor)
    if s < w:
        return start + (peak - start) * s / w                                    # U optims.py:33-36
    return (peak - floor) * 0.5 * (1 + math.cos(math.pi * (s - w) / (total - w))) + floor   # U optims.py:28-31


def group_lrs(step: int, sc: StepCounts, tc: "TrainConfig", groups: Sequence[str]) -> dict[str, float]:
    """``lr_scale_g x lr_at(s)`` for each group (§14.6.2; linear in start/peak/floor, U :253-267)."""
    base = lr_at(step, sc, tc)
    return {g: group_hparams(g, tc)[0] * base for g in groups}


def curriculum_probability(mode: str, tau: int, total: int) -> float:
    """Hybrid probability of micro-batch ``tau`` (M12 ``curriculum_p``): linear tau/T (D3), all_hybrid 1,
    reverse 1 - tau/T (M4 §8.6.5)."""
    if mode == "linear":
        return tau / total
    if mode == "all_hybrid":
        return 1.0
    if mode == "reverse":
        return 1 - tau / total
    raise TrainingError(f"unknown prompt.train_curriculum {mode!r}")


# ----------------------------------------------------------------------------
# Record and summary values (§14.6.7, §14.6.8, 3.1 StepRecord / TrainSummary)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StepRecord:
    """One optimizer-step row; ``as_row()`` gives the M12 ``step_fields(K)`` columns in order (§16.6.2)."""
    step: int
    epoch: int
    tau_first: int
    tau_last: int
    n_micro: int
    n_hybrid: int
    loss_mean: float
    loss_hybrid_mean: Optional[float]
    loss_text_mean: Optional[float]
    nonfinite_loss: int
    lr_lora: float
    lr_projector: float
    lr_gate: Optional[float]
    grad_norm_lora: float
    grad_norm_projector: Optional[float]
    grad_norm_gate: Optional[float]
    grad_norm_total: float
    projector_has_grad: bool
    scale_before: float
    scale_after: float
    skipped: bool
    omega_mean: tuple[float, ...]            # omega_mean_0..omega_mean_{K-1}
    omega_entropy_mean: Optional[float]
    gate_calls: int
    cluster_lookups: int
    max_mem_gib: Optional[float]
    max_mem_reserved_gib: Optional[float]
    time_s: float

    def as_row(self) -> dict[str, object]:
        from .records import step_fields

        out: dict[str, object] = {}
        for name in step_fields(len(self.omega_mean)):
            out[name] = self.omega_mean[int(name.rsplit("_", 1)[1])] if name.startswith("omega_mean_") \
                else getattr(self, name)
        return out


@dataclass(frozen=True)
class TrainSummary:
    """Summary of a training run (§14.6.7): kept in M12 ``run_summary.json`` (``train_summary``) and read by M14.

    ``val``: epoch -> M11 ``SplitMetrics``; ``test``: epoch -> {tags, metrics, aux_protocols, eval_dir};
    ``trainable_params``: per group and ``total`` ([SPEC §12] TrainableParams); ``seconds``: training,
    validation, save and test time over every process of the run; ``epoch_max_mem_gib``: largest step value
    of each epoch (None on CPU); ``unused_experts``: experts no cluster maps to (cluster_hard, §19.6.7).
    """
    status: str
    routing_mode: str
    num_experts: int
    counts: Optional[StepCounts]
    epochs_complete: int
    best_val_epoch: Optional[int]
    last_epoch: Optional[int]
    val: Mapping[int, "SplitMetrics"]
    test: Mapping[int, Mapping[str, object]]
    trainable_params: Mapping[str, int]
    seconds: Mapping[str, float]
    skipped_steps: int
    nonfinite_loss: int
    gate_calls: int
    cluster_lookups: int
    epoch_max_mem_gib: Mapping[int, Optional[float]]
    unused_experts: tuple[int, ...]
    resumes: int


# ----------------------------------------------------------------------------
# Optimizer (§14.6.3, D19) [코드] U model/model_interface.py:227-236, L model/model_interface.py:153-156
# ----------------------------------------------------------------------------
def group_sizes(groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]]) -> dict[str, int]:
    """Parameter count per group and ``total`` ([SPEC §12] TrainableParams; paper_spec 17,047,552 / 256 /
    19,988,480, total 37,036,288)."""
    sizes = {g: int(sum(p.numel() for _, p in groups[g])) for g in GROUP_ORDER if g in groups}
    sizes["total"] = int(sum(sizes.values()))
    return sizes


def build_optimizer(groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]], tc: "TrainConfig", *,
                    model: Optional[nn.Module] = None) -> torch.optim.Adam:
    """``torch.optim.Adam`` with one group per D19 group in ``GROUP_ORDER`` (§14.6.3).

    Coupled L2 weight decay per group (Adam, not AdamW; [코드] U :227-232: projector and gate 1e-5, LoRA 0),
    ``betas``/``eps`` from the ``train`` keys, ``amsgrad=False``, torch's default ``foreach``. Each group dict
    carries ``"name"`` (kept in ``state_dict``). Checks (``TrainingError``): only the groups ``projector``,
    ``gate``, ``lora`` (projector and lora present), names as given by M9 (sorted, unique), every parameter in
    exactly one group, float32, ``requires_grad`` and on one device (GradScaler refuses to unscale fp16
    gradients, torch2.0 cuda/amp/grad_scaler.py:211-212). With ``model`` also every other parameter of the
    model must have ``requires_grad=False``. The initial lr is ``lr_scale x lr_warmup_start`` (= lr at s = 0);
    the loop sets the lr of every window (§14.6.2).
    """
    names = list(groups)
    unknown = [g for g in names if g not in GROUP_ORDER]
    if unknown or "projector" not in names or "lora" not in names:
        raise TrainingError(f"parameter groups {names} must be projector, lora and optionally gate (D19, §14.6.3)")
    seen: dict[int, str] = {}
    devices: set[torch.device] = set()
    problems: list[str] = []
    for group in GROUP_ORDER:
        if group not in groups:
            continue
        items = list(groups[group])
        if not items:
            problems.append(f"group {group} is empty")
        labels = [name for name, _ in items]
        if labels != sorted(labels) or len(set(labels)) != len(labels):
            problems.append(f"group {group}: names must be sorted and unique (M9 param_groups)")
        for name, p in items:
            if not isinstance(p, nn.Parameter):
                problems.append(f"{group}/{name}: not an nn.Parameter ({type(p).__name__})")
                continue
            if id(p) in seen:
                problems.append(f"{name} is in two groups ({seen[id(p)]} and {group})")
            seen[id(p)] = f"{group}/{name}"
            if p.dtype != torch.float32:
                problems.append(f"{group}/{name}: dtype {p.dtype}, trainable parameters are float32 (D9)")
            if not p.requires_grad:
                problems.append(f"{group}/{name}: requires_grad is False")
            devices.add(p.device)
    if len(devices) > 1:
        problems.append(f"trainable parameters on several devices {sorted(str(d) for d in devices)}")
    if model is not None:
        own = {id(p) for p in model.parameters()}
        outside = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
        foreign = [label for pid, label in seen.items() if pid not in own]
        if outside:
            problems.append(f"trainable parameters outside every group {outside[:5]} ({len(outside)}); every other "
                            "parameter must have requires_grad=False (§14.6.3)")
        if foreign:
            problems.append(f"group parameters that are not parameters of the model {foreign[:5]}")
    if problems:
        raise TrainingError("optimizer groups rejected: " + "; ".join(problems))
    param_groups = []
    for group in GROUP_ORDER:
        if group in groups:
            scale, decay = group_hparams(group, tc)
            param_groups.append({"params": [p for _, p in groups[group]], "name": group,
                                 "lr": scale * float(tc.lr_warmup_start), "weight_decay": decay})
    return torch.optim.Adam(param_groups, betas=(float(tc.adam_beta1), float(tc.adam_beta2)), eps=float(tc.adam_eps),
                            amsgrad=False)


def set_lrs(optimizer: torch.optim.Optimizer, lrs: Mapping[str, float]) -> None:
    """Set every group lr once before the first micro-batch of a window (§14.6.2)."""
    for pg in optimizer.param_groups:
        if pg.get("name") not in lrs:
            raise TrainingError(f"optimizer group {pg.get('name')!r} has no lr (groups {sorted(lrs)})")
        pg["lr"] = lrs[pg["name"]]


# ----------------------------------------------------------------------------
# Consecutive GradScaler skips (§14.6.8) [결정]
# ----------------------------------------------------------------------------
class SkipGuard:
    """Stops the training after ``limit`` consecutive skipped steps (``train.max_consecutive_skips`` = 16).

    16 = log2(init_scale): after 16 halvings the scale is 1/65,536 of the start, so a further overflow is a
    problem of the forward/backward values themselves, not of loss scaling. Pure logic, no tensors.
    """

    def __init__(self, limit: int) -> None:
        if not _is_int(limit) or int(limit) < 1:
            raise TrainingError(f"train.max_consecutive_skips must be an int >= 1, got {limit!r}")
        self.limit = int(limit)
        self.consecutive = 0
        self.total = 0
        self.run_start: Optional[int] = None
        self.run_scale: Optional[float] = None

    def update(self, step: int, skipped: bool, scale_before: float, scale_after: float, *,
               grad_norm: Optional[float] = None) -> None:
        """Count one step; ``TrainingError`` with the step range, the scale change and the last gradient norm at
        the ``limit``-th consecutive skip. A successful step resets the run."""
        if not skipped:
            self.consecutive, self.run_start, self.run_scale = 0, None, None
            return
        if self.consecutive == 0:
            self.run_start, self.run_scale = int(step), float(scale_before)
        self.consecutive += 1
        self.total += 1
        if self.consecutive >= self.limit:
            raise TrainingError(
                f"{self.consecutive} consecutive GradScaler skips at steps {self.run_start}-{int(step)} "
                f"(train.max_consecutive_skips = {self.limit}): scale {self.run_scale} -> {float(scale_after)}, "
                f"last grad norm {grad_norm}; the overflow is in the forward/backward values, not in loss "
                "scaling (§14.6.8)")


def enter_train_mode(model: Any) -> None:
    """``model.train()`` and then SASRec must still be in eval mode (D12; Lightning switched it to train)."""
    model.train()
    if not model.sasrec_is_eval():
        raise TrainingError("SASRec is in train mode after model.train(); it must stay in eval mode (D12, §14.6.4)")


# ----------------------------------------------------------------------------
# Plan and batch checks (§14.6.4 epoch loop steps 1 and 5.2)
# ----------------------------------------------------------------------------
def check_plan(plan: Any, epoch: int, sc: StepCounts, *, tau_done: int) -> None:
    """The epoch plan: ``micro_per_epoch`` entries, tau from ``tau_done + 1`` without gaps, B samples each, no
    repeated sample, bool ``hybrid`` (M4 ``BatchPlan`` checks the rest when it is built)."""
    entries = list(plan.entries)
    problems = []
    if plan.epoch != epoch:
        problems.append(f"plan epoch {plan.epoch} != {epoch}")
    if len(entries) != sc.micro_per_epoch:
        problems.append(f"{len(entries)} micro-batches, expected micro_per_epoch {sc.micro_per_epoch}")
    taus = [e.tau for e in entries]
    if taus != list(range(tau_done + 1, tau_done + 1 + len(entries))):
        problems.append(f"tau {taus[:1]}..{taus[-1:]} does not continue from tau_done {tau_done}")
    if any(len(e.sample_indices) != sc.micro_batch for e in entries):
        problems.append(f"a micro-batch does not hold train.micro_batch_size = {sc.micro_batch} samples")
    flat = [j for e in entries for j in e.sample_indices]
    if len(set(flat)) != len(flat):
        problems.append("a sample appears twice in the epoch")
    if any(not isinstance(e.hybrid, bool) for e in entries):
        problems.append("hybrid must be bool")
    if problems:
        raise TrainingError(f"batch plan of epoch {epoch}: " + "; ".join(problems))


def check_batch(batch: "TrainBatch", entry: "PlanEntry", train_samples: Sequence["Sample"]) -> None:
    """The loader output equals the plan entry: tau, hybrid, sample ids in ``sample_indices`` order, and every row
    has answer tokens (``(labels != -100).any(1)``, checked on the CPU batch)."""
    expected = [train_samples[j].sample_id for j in entry.sample_indices]
    problems = []
    if batch.tau != entry.tau:
        problems.append(f"batch tau {batch.tau} != plan tau {entry.tau}")
    if batch.hybrid is not entry.hybrid:
        problems.append(f"batch hybrid {batch.hybrid} != plan hybrid {entry.hybrid}")
    if list(batch.sample_ids) != expected:
        problems.append(f"sample ids {list(batch.sample_ids)[:4]}... != plan {expected[:4]}...")
    if batch.labels.device.type != "cpu":
        problems.append(f"labels on {batch.labels.device}, the loader gives CPU batches")
    else:
        has_answer = (batch.labels != -100).any(dim=1)
        if not bool(has_answer.all()):
            row = int((~has_answer).nonzero()[0, 0])
            problems.append(f"row {row} ({batch.sample_ids[row]}) has no answer token")
    if problems:
        raise TrainingError(f"micro-batch tau {entry.tau}: " + "; ".join(problems))


# ----------------------------------------------------------------------------
# Checkpoint files (§14.6.6) [결정]: temporary name + os.replace, STATE_SAVED after both files, COMPLETE last
# ----------------------------------------------------------------------------
def _plain(obj: object) -> object:
    """JSON values for markers (dataclasses such as SplitMetrics become dicts, paths strings)."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    raise TrainingError(f"cannot write {type(obj).__name__} into a JSON marker")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _torch_bytes(obj: object) -> bytes:
    """``torch.save`` into memory: the file bytes are known (and hashed) before they are written. The archive
    name is the fixed in-memory one, so the same state gives the same bytes (checked on reload)."""
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
        torch.save(obj, buffer)
    return buffer.getvalue()


def _torch_load(path: Path) -> Any:
    """``torch.load(path, map_location="cpu", weights_only=True)``; torch 2.0.0's own TypedStorage deprecation
    warning is silenced, read errors become ``TrainingError`` with the path."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with the path
        raise TrainingError(f"{path}: cannot be read with torch.load(weights_only=True): "
                            f"{type(exc).__name__}: {exc}") from exc


def _write_bytes_atomic(path: Path, data: bytes) -> str:
    """Write ``data`` under a temporary name, fsync, ``os.replace`` (no half-written file); returns its sha256."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_bytes(data)


def write_marker(path: Path, payload: Mapping[str, object]) -> str:
    """A JSON file of the checkpoint folder (``STATE_SAVED``, ``val_metrics.json``, ``COMPLETE``) or of ``eval/``
    (``selection.json``): UTF-8, LF, written once (never overwritten). Returns its sha256."""
    path = Path(path)
    if path.exists():
        raise TrainingError(f"{path} exists; checkpoint and selection files are never overwritten (4.6.3)")
    data = _plain(payload)
    hits = run.find_secrets(data)
    if hits:
        raise run.SecretLeakError(f"refusing to write token-like values at {hits} to {path}")
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return _write_bytes_atomic(path, text.encode("utf-8"))


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingError(f"{path}: cannot be read as JSON ({type(exc).__name__}: {exc})") from exc
    if not isinstance(data, dict):
        raise TrainingError(f"{path}: expected a JSON object")
    return data


def trainable_names(model: Any) -> dict[str, list[str]]:
    """Group -> parameter names in M9 ``param_groups`` order (``group_param_names`` of the checkpoint)."""
    return {group: [name for name, _ in items] for group, items in model.param_groups().items()}


def frozen_checksum(model: Any) -> dict[str, float]:
    """``torch.sum(p, dtype=float64)`` of every parameter outside the trainable groups (base LLaMA, embeddings,
    lm_head, SASRec), computed on the device and moved to the CPU once (§14.6.6)."""
    trainable = {id(p) for items in model.param_groups().values() for _, p in items}
    names, sums = [], []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if id(p) not in trainable:
                names.append(name)
                sums.append(torch.sum(p.detach(), dtype=torch.float64).reshape(()))
    if not names:
        return {}
    by_device: dict[torch.device, list[int]] = {}
    for i, value in enumerate(sums):
        by_device.setdefault(value.device, []).append(i)
    values: list[float] = [0.0] * len(sums)
    for idx in by_device.values():
        for i, v in zip(idx, torch.stack([sums[i] for i in idx]).cpu().tolist()):
            values[i] = float(v)
    return dict(zip(names, values))


def tensor_digest(state: Mapping[str, Tensor]) -> str:
    """sha256 over ``key|dtype|shape`` lines and the raw bytes of each tensor (sorted keys), any dtype (bf16
    included). meta.json records it for the new-token rows (§13.8 #14: same seed -> same value)."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().to("cpu").contiguous()
        h.update(f"{key}|{str(t.dtype).replace('torch.', '', 1)}|{','.join(str(s) for s in t.shape)}\n".encode("utf-8"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _trainable_bytes(model: Any) -> bytes:
    """``trainable.pt`` content: M9 ``trainable_state()`` (all trainable parameters, no ``requires_grad`` filter)
    plus ``generated_frozen_state()`` (the five seeded new-token rows, §13.6.8)."""
    return _torch_bytes({**model.trainable_state(), **model.generated_frozen_state()})


def current_provenance(cfg: "Config", model: Any, *, repo_root: Path = run.REPO_ROOT, llm_path: Optional[Path] = None,
                       tokenizer_dir: Optional[Path] = None) -> dict[str, object]:
    """Values a checkpoint must match on reload (§14.6.6): config hash (M0), input hashes (M0 ``meta.json``
    ``inputs`` + converted SASRec state, tokenizer model, LLaMA ``config.json`` and its dtype, cluster manifest),
    M7 ``provenance()``, the SASRec digest (M3) and, for cluster_hard, the cluster source (manifest sha256,
    ``expert_mapping``, unit, n_clusters, fit split; M7 §11.6.7, M15 §19.6.7)."""
    from .commands import asset_record

    assets = asset_record(cfg, repo_root=Path(repo_root), llm_path=llm_path, tokenizer_dir=tokenizer_dir)
    llama = assets.get("llama_config") or None
    input_hashes = {
        "inputs": run.input_hashes(Path(repo_root)),
        "sasrec_state": (assets.get("sasrec_state") or {}).get("sha256"),
        "tokenizer_model": (assets.get("tokenizer_model") or {}).get("sha256"),
        "llama_config": None if llama is None else {"sha256": llama.get("sha256"), "torch_dtype": llama.get("torch_dtype")},
        "cluster_manifest": (assets.get("cluster_manifest") or {}).get("sha256"),
    }
    router = model.omega_source
    cluster = None
    if router.routing_mode == "cluster_hard":
        gate = getattr(router, "hard_gate", None)
        artifacts = getattr(gate, "artifacts", None)
        manifest = dict(getattr(artifacts, "manifest", None) or {})
        cluster = {"manifest_sha256": input_hashes["cluster_manifest"], "expert_mapping": cfg.cluster.expert_mapping,
                   "mapping": [int(e) for e in getattr(gate, "mapping", ())],
                   "unit": getattr(artifacts, "unit", cfg.cluster.unit),
                   "n_clusters": int(getattr(artifacts, "n_clusters", cfg.cluster.n_clusters)),
                   "fit_split": manifest.get("fit_split")}
    return {"config_hash": config_hash(cfg), "input_hashes": input_hashes, "router_provenance": router.provenance(),
            "sasrec_state_digest": model.sasrec.state_digest, "cluster": cluster}


def save_checkpoint(ckpt_dir: Path, model: Any, optimizer: torch.optim.Optimizer, scaler: Any,
                    state: Mapping[str, object]) -> dict[str, str]:
    """Write ``trainable.pt``, ``train_state.pt`` and ``STATE_SAVED`` into an empty ``ckpt_dir`` (§14.6.6).

    ``state`` holds the run values of ``STATE_KEYS`` (epoch, global_step, tau_done, config hash, input hashes,
    cluster source, SASRec digest, frozen checksum, counters, epoch statistics, step counts); this function adds
    ``format_version``, the optimizer and GradScaler ``state_dict``, ``group_param_names``, the CPU and CUDA RNG
    states, M7 ``provenance()`` and the trainable sha256. Both files are loadable with ``weights_only=True``.
    Returns ``{file name: sha256}`` (also the content of ``STATE_SAVED`` with the save time).
    """
    started = time.perf_counter()
    ckpt_dir = Path(ckpt_dir)
    missing = [key for key in STATE_KEYS if key not in state]
    if missing:
        raise TrainingError(f"save_checkpoint: state misses {missing}")
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        raise TrainingError(f"{ckpt_dir} is not empty; checkpoints are never overwritten")
    names = trainable_names(model)
    if [pg.get("name") for pg in optimizer.param_groups] != list(names):
        raise TrainingError(f"optimizer groups {[pg.get('name') for pg in optimizer.param_groups]} != model groups "
                            f"{list(names)}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainable = _trainable_bytes(model)
    device = model.device
    payload = {"format_version": TRAIN_STATE_FORMAT, **{key: state[key] for key in STATE_KEYS},
               "optimizer": optimizer.state_dict(), "group_param_names": names, "scaler": scaler.state_dict(),
               "rng": {"cpu": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
               "router_provenance": model.omega_source.provenance(), "trainable_sha256": _sha256_bytes(trainable)}
    files = {TRAINABLE_FILE: _write_bytes_atomic(ckpt_dir / TRAINABLE_FILE, trainable),
             TRAIN_STATE_FILE: _write_bytes_atomic(ckpt_dir / TRAIN_STATE_FILE, _torch_bytes(payload))}
    write_marker(ckpt_dir / STATE_SAVED_MARKER, {"epoch": state["epoch"], "files": files,
                                                 "seconds": time.perf_counter() - started,
                                                 "utc": run.utc_now().isoformat()})
    return files


# ----------------------------------------------------------------------------
# Reading checkpoints (§14.6.6 읽기는 strict; the only readers for M10 resume, M11 eval and M12 export)
# ----------------------------------------------------------------------------
def checkpoint_files(ckpt_dir: Path, *, need: Sequence[str] = (TRAINABLE_FILE,)) -> dict[str, str]:
    """``{file: sha256}`` of the markers: ``COMPLETE`` (three files) when present, else ``STATE_SAVED`` (two). The
    files in ``need`` must exist with that sha256; markers that disagree are an error."""
    ckpt_dir = Path(ckpt_dir)
    saved_path, complete_path = ckpt_dir / STATE_SAVED_MARKER, ckpt_dir / COMPLETE_MARKER
    if not saved_path.is_file() and not complete_path.is_file():
        raise TrainingError(f"{ckpt_dir}: no {STATE_SAVED_MARKER} or {COMPLETE_MARKER} marker; the checkpoint was not "
                            "saved completely (14.6.6)")
    files: dict[str, str] = {}
    for path in (saved_path, complete_path):
        if not path.is_file():
            continue
        listed = _read_json(path).get("files")
        if not isinstance(listed, Mapping):
            raise TrainingError(f"{path}: no file list")
        for name, sha in listed.items():
            if name in files and files[name] != sha:
                raise TrainingError(f"{ckpt_dir}: {STATE_SAVED_MARKER} and {COMPLETE_MARKER} disagree on {name}")
            files[str(name)] = str(sha)
    for name in need:
        path = ckpt_dir / name
        if name not in files:
            raise TrainingError(f"{ckpt_dir}: the markers do not list {name}")
        if not path.is_file():
            raise TrainingError(f"{path} is missing")
        got = run.content_sha256(path)
        if got != files[name]:
            raise TrainingError(f"{path}: sha256 {got} != marker {files[name]} (file changed after it was saved)")
    return files


def read_trainable_state(ckpt_dir: Path) -> dict[str, Tensor]:
    """The whole ``trainable.pt`` ``{name: tensor}`` (``generated_frozen.*`` included, no optimizer) after the
    marker (``COMPLETE`` or ``STATE_SAVED``) and sha256 checks (§14.6.6). M12 export uses only the
    ``omega_source.`` keys (§13.6.8)."""
    ckpt_dir = Path(ckpt_dir)
    checkpoint_files(ckpt_dir, need=(TRAINABLE_FILE,))
    state = _torch_load(ckpt_dir / TRAINABLE_FILE)
    if not isinstance(state, Mapping) or not state or \
            not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()):
        raise TrainingError(f"{ckpt_dir / TRAINABLE_FILE}: expected a non-empty mapping name -> tensor")
    return dict(state)


def _same_bits(a: Tensor, b: Tensor) -> bool:
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def _apply_trainable(state: Mapping[str, Tensor], model: Any, where: str) -> None:
    """Strict application of a ``trainable.pt`` mapping (§14.6.6, §13.6.8): ``generated_frozen.*`` equal bit for bit
    to the rebuilt model's rows, the other names exactly the ``param_groups`` names (no missing, no extra) with the
    same shapes and dtypes; only then M9 ``load_trainable_state(strict=True)``."""
    from .model import GENERATED_FROZEN_PREFIX

    problems = []
    expected_frozen = model.generated_frozen_state()
    for key, value in expected_frozen.items():
        saved = state.get(key)
        if saved is None:
            problems.append(f"{key} is missing")
        elif not _same_bits(saved, value):
            problems.append(f"{key} differs bit for bit from the generated_frozen rows of the rebuilt model "
                            "(another seed, torch build or tokenizer, 13.6.8)")
    extra_frozen = sorted(k for k in state if k.startswith(GENERATED_FROZEN_PREFIX) and k not in expected_frozen)
    params = {name: p for items in model.param_groups().values() for name, p in items}
    rest = {k: v for k, v in state.items() if not k.startswith(GENERATED_FROZEN_PREFIX)}
    missing = sorted(set(params) - set(rest))
    extra = sorted(set(rest) - set(params)) + extra_frozen
    if missing:
        problems.append(f"missing keys {missing[:5]} ({len(missing)})")
    if extra:
        problems.append(f"unexpected keys {extra[:5]} ({len(extra)})")
    for name in sorted(set(params) & set(rest)):
        value, param = rest[name], params[name]
        if tuple(value.shape) != tuple(param.shape) or value.dtype != param.dtype:
            problems.append(f"{name}: {list(value.shape)} {value.dtype} != model {list(param.shape)} {param.dtype}")
    if problems:
        raise TrainingError(f"{where}: trainable state rejected: " + "; ".join(problems))
    model.load_trainable_state(rest, strict=True)


def load_model_state(ckpt_dir: Path, model: Any) -> str:
    """``read_trainable_state`` -> ``generated_frozen.*`` bit comparison -> M9 ``load_trainable_state(strict=True)``
    (§14.6.6; M11 eval command, M10 test). Returns the trainable sha256 recomputed from the loaded model, which
    must equal the file's."""
    ckpt_dir = Path(ckpt_dir)
    state = read_trainable_state(ckpt_dir)
    expected = checkpoint_files(ckpt_dir, need=())[TRAINABLE_FILE]
    _apply_trainable(state, model, str(ckpt_dir))
    got = _sha256_bytes(_trainable_bytes(model))
    if got != expected:
        raise TrainingError(f"{ckpt_dir}: trainable sha256 recomputed after loading {got} != {expected}")
    return got


def load_checkpoint(ckpt_dir: Path, model: Any, optimizer: torch.optim.Optimizer, scaler: Any, cfg: "Config", *,
                    provenance: Optional[Mapping[str, object]] = None) -> dict[str, object]:
    """Full strict reload for ``train --resume`` (§14.6.6, §14.6.10): markers and sha256; ``format_version``;
    config hash, input hashes, M7 provenance, SASRec digest and cluster source equal to ``provenance`` (default
    ``current_provenance(cfg, model)``); the frozen-parameter checksum; the trainable mapping (``_apply_trainable``);
    ``group_param_names`` equal to the current groups before ``optimizer.load_state_dict`` (torch matches by
    position); the GradScaler state; then the CPU and CUDA RNG states (no new seed). The recomputed trainable
    sha256 must equal the file's. Returns the run values (epoch, global_step, tau_done, counters, ...)."""
    ckpt_dir = Path(ckpt_dir)
    where = str(ckpt_dir)
    files = checkpoint_files(ckpt_dir, need=(TRAINABLE_FILE, TRAIN_STATE_FILE))
    state = _torch_load(ckpt_dir / TRAIN_STATE_FILE)
    if not isinstance(state, Mapping) or state.get("format_version") != TRAIN_STATE_FORMAT:
        got = state.get("format_version") if isinstance(state, Mapping) else type(state).__name__
        raise TrainingError(f"{where}: train_state format {got!r} != {TRAIN_STATE_FORMAT!r}")
    required = (*STATE_KEYS, "optimizer", "group_param_names", "scaler", "rng", "router_provenance", "trainable_sha256")
    absent = [key for key in required if key not in state]
    if absent:
        raise TrainingError(f"{where}: train_state misses {absent}")
    if state["trainable_sha256"] != files[TRAINABLE_FILE]:
        raise TrainingError(f"{where}: train_state trainable_sha256 {state['trainable_sha256']} != marker "
                            f"{files[TRAINABLE_FILE]}")
    current = dict(provenance) if provenance is not None else current_provenance(cfg, model)
    problems = [f"{key}: checkpoint {state.get(key)!r} != current {current.get(key)!r}"
                for key in PROVENANCE_KEYS if state.get(key) != current.get(key)]
    if current.get("config_hash") != config_hash(cfg):
        problems.append("the given provenance belongs to another config")
    names = trainable_names(model)
    saved_names = state["group_param_names"]
    if not isinstance(saved_names, Mapping) or list(saved_names.items()) != list(names.items()):
        problems.append("group_param_names (group order or names) differ from the current optimizer groups; torch "
                        "loads Adam states by position")
    if [pg.get("name") for pg in optimizer.param_groups] != list(names):
        problems.append(f"optimizer groups {[pg.get('name') for pg in optimizer.param_groups]} != {list(names)}")
    saved_scaler = state["scaler"]
    if bool(saved_scaler) != bool(scaler.is_enabled()):
        problems.append(f"GradScaler state {'present' if saved_scaler else 'empty'} but the scaler is "
                        f"{'enabled' if scaler.is_enabled() else 'disabled'}")
    if problems:
        raise TrainingError(f"{where}: checkpoint does not match this run: " + "; ".join(problems))
    frozen = frozen_checksum(model)
    if state["frozen_checksum"] != frozen:
        diff = sorted(k for k in set(frozen) | set(state["frozen_checksum"])
                      if frozen.get(k) != state["frozen_checksum"].get(k))
        raise TrainingError(f"{where}: frozen parameter checksum differs for {diff[:5]} ({len(diff)}): another base "
                            "model or SASRec")
    _apply_trainable(read_trainable_state(ckpt_dir), model, where)
    try:
        optimizer.load_state_dict(state["optimizer"])
    except (ValueError, KeyError, RuntimeError) as exc:
        raise TrainingError(f"{where}: optimizer state rejected: {type(exc).__name__}: {exc}") from exc
    if saved_scaler:
        scaler.load_state_dict(saved_scaler)
    rng = state["rng"]
    torch.set_rng_state(rng["cpu"])
    if model.device.type == "cuda":
        if rng.get("cuda") is None:
            raise TrainingError(f"{where}: no CUDA RNG state saved for a CUDA run")
        torch.cuda.set_rng_state(rng["cuda"], model.device)
    got = _sha256_bytes(_trainable_bytes(model))
    if got != files[TRAINABLE_FILE]:
        raise TrainingError(f"{where}: trainable sha256 recomputed after loading {got} != {files[TRAINABLE_FILE]}")
    out = {key: state[key] for key in STATE_KEYS}
    out.update({"files": files, "trainable_sha256": got, "scaler": dict(saved_scaler)})
    return out


# ----------------------------------------------------------------------------
# Selection (§14.6.7, D11)
# ----------------------------------------------------------------------------
def select_epochs(val: Mapping[int, "SplitMetrics"]) -> tuple[int, int]:
    """``(best, last)``: best = argmax over (n_correct, epoch), the later epoch on ties; last = the final epoch.

    CombinedMetric = n_correct / N with the same N every epoch, so the integer counts decide (not the float
    product ``hr x valid_ratio`` of U model/model_interface.py:181); a different N is a ``TrainingError``.
    """
    if not isinstance(val, Mapping) or not val:
        raise TrainingError("select_epochs needs the validation metrics of at least one epoch")
    epochs = sorted(val)
    if any(not _is_int(e) for e in epochs) or epochs != list(range(len(epochs))):
        raise TrainingError(f"validation epochs must be 0..E-1, got {epochs}")
    totals = sorted({int(val[e].n_total) for e in epochs})
    if len(totals) != 1:
        raise TrainingError(f"validation N differs between epochs {totals}; the counts cannot be compared (D17)")
    best = max(epochs, key=lambda e: (int(val[e].n_correct), e))
    return int(best), int(epochs[-1])


def epochs_to_test(best: int, last: int) -> tuple[int, ...]:
    """Test order: best_val and last ascending, once when they are the same epoch (§14.6.7)."""
    return tuple(sorted({int(best), int(last)}))


def selection_document(val: Mapping[int, "SplitMetrics"], best: int, last: int) -> dict[str, object]:
    """``eval/selection.json`` (keys fixed for M11 eval and M12 export, §14.6.7)."""
    fields = ("n_total", "n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")
    return {"rule": SELECTION_RULE,
            "val": {str(e): {name: getattr(val[e], name) for name in fields} for e in sorted(val)},
            "best_val_epoch": int(best), "last_epoch": int(last),
            "test_dirs": {"best_val": f"eval/epoch{best}/test", "last": f"eval/epoch{last}/test"},
            "primary": "best_val"}


# ----------------------------------------------------------------------------
# One accumulation window (§14.6.4 epoch loop step 5, D24)
# ----------------------------------------------------------------------------
@dataclass
class MicroResult:
    """One micro-batch of a window. ``loss`` and ``omega`` stay detached on the device until the window ends
    (one transfer per window); ``act`` is the 0-dim device pair of ``ActivationMaxMonitor.end_micro_batch``."""
    tau: int
    hybrid: bool
    batch_size: int
    seq_len: int
    n_label_tokens: int
    loss: Tensor
    omega: Tensor
    act: Optional[tuple[Tensor, Tensor]]
    time_s: float


@dataclass
class WindowResult:
    """CPU values of one optimizer step (§14.6.4 steps 5.1-5.7) from which the M12 rows are built."""
    step: int
    micros: list[MicroResult]
    losses: list[float]
    omega: np.ndarray                         # float64 [window samples, K]
    lrs: dict[str, float]
    norms: dict[str, Optional[float]]         # group -> L2 norm, None without gradient
    grad_norm_total: float
    scale_before: float
    scale_after: float
    skipped: bool
    gate_calls: int
    cluster_lookups: int
    max_mem_gib: Optional[float]
    max_mem_reserved_gib: Optional[float]


def accumulate_window(model: Any, batches: Iterator["TrainBatch"], entries: Sequence["PlanEntry"],
                      train_samples: Sequence["Sample"], *, precision: str, scaler: Any, divisor: int,
                      monitor: Any = None, clock: Optional[Any] = None) -> list[MicroResult]:
    """Forward and backward of the ``n = len(entries)`` micro-batches of one window (§14.6.4 step 5.2).

    Each batch must match its plan entry (``check_batch``); the forward runs inside ``autocast_for(precision)``
    (fp16_amp: CUDA float16, D9) and ``scaler.scale(loss / divisor).backward()`` runs outside it. ``train_window``
    passes ``train.grad_accum`` for every window, so the last window of an epoch (9 micro-batches in paper_spec)
    is divided by 16 as well and every micro-batch weighs 1/grad_accum in its step (D24) [코드] PL1.8.6
    loops/optimization/optimizer_loop.py:88, :416 (``closure_loss / accumulate_grad_batches``),
    loops/epoch/training_epoch_loop.py:333-337 (a step on the final batch), U main.py:57, U train_movielens.sh:6.
    """
    n = len(entries)
    if n < 1:
        raise TrainingError("an accumulation window needs at least one micro-batch")
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor < n:
        raise TrainingError(f"the loss divisor must be an int >= the window size {n} (train.grad_accum, D24), got "
                            f"{divisor!r}")
    clock = clock or time.perf_counter
    device = model.device
    out: list[MicroResult] = []
    for entry in entries:
        try:
            batch = next(batches)
        except StopIteration:
            raise TrainingError(f"the train loader ended before tau {entry.tau}; the plan has more micro-batches") \
                from None
        check_batch(batch, entry, train_samples)
        with autocast_for(precision, device):
            forward = model.train_forward(batch)
        loss = forward.loss
        if not isinstance(loss, Tensor) or loss.dtype != torch.float32 or loss.dim() != 0:
            raise TrainingError(f"tau {entry.tau}: train_forward loss must be a 0-dim float32 tensor (M9 13.6.5), got "
                                f"{getattr(loss, 'dtype', type(loss).__name__)} {list(getattr(loss, 'shape', []))}")
        scaler.scale(loss / divisor).backward()
        act = monitor.end_micro_batch(int(entry.tau)) if monitor is not None else None
        out.append(MicroResult(tau=int(entry.tau), hybrid=bool(entry.hybrid), batch_size=int(batch.input_ids.shape[0]),
                               seq_len=int(batch.input_ids.shape[1]), n_label_tokens=int(forward.n_label_tokens),
                               loss=loss.detach(), omega=forward.omega.detach(), act=act, time_s=float(clock())))
    return out


def gradient_norms(groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]]
                   ) -> tuple[dict[str, Optional[Tensor]], Optional[Tensor], dict[str, str]]:
    """After ``scaler.unscale_``: the fp32 L2 norm of each group and of every gradient (device tensors; parameters
    without a gradient are left out, a group without any is None) and per group ``all``/``some``/``none`` for the
    presence of gradients. No clipping ([코드] U main.py:57: Trainer default gradient_clip_val)."""
    norms: dict[str, Optional[Tensor]] = {}
    presence: dict[str, str] = {}
    parts: list[Tensor] = []
    with torch.no_grad():
        for group, items in groups.items():
            grads = [p.grad for _, p in items if p.grad is not None]
            presence[group] = "all" if len(grads) == len(items) else ("none" if not grads else "some")
            if grads:
                per = torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])
                norms[group] = torch.linalg.vector_norm(per)
                parts.append(per)
            else:
                norms[group] = None
        total = torch.linalg.vector_norm(torch.cat(parts)) if parts else None
    return norms, total, presence


def check_grad_presence(presence: Mapping[str, str], *, n_hybrid: int, where: str = "") -> None:
    """§14.6.4 step 5.4: projector gradient <=> a hybrid micro-batch in the window (the easy loss has no projector,
    [LLaRA 간접 §4.3 p.6 식 7]); gate (dynamic) and LoRA always get gradients. Otherwise ``TrainingError``."""
    problems = []
    projector = presence.get("projector")
    if n_hybrid > 0 and projector != "all":
        problems.append(f"projector gradient {projector!r} although {n_hybrid} micro-batches are hybrid")
    if n_hybrid == 0 and projector != "none":
        problems.append(f"projector gradient {projector!r} in a text-only window")
    if "gate" in presence and presence["gate"] != "all":
        problems.append(f"gate gradient {presence['gate']!r} (dynamic routing always trains W_g)")
    if presence.get("lora") != "all":
        problems.append(f"LoRA gradient {presence.get('lora')!r} (every expert A and B gets one)")
    if problems:
        raise TrainingError(f"{where}: gradient presence rule broken: " + "; ".join(problems))


def optimizer_step(optimizer: torch.optim.Optimizer, scaler: Any) -> tuple[float, float, bool]:
    """``scaler.step`` + ``update``; ``skipped = scale_after < scale_before`` (the fork measurement rule,
    F measurement/schema.py:75). A skipped step still counts for lr and tau (§14.6.2 [결정])."""
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    return before, after, after < before


def peak_memory(device: torch.device) -> tuple[Optional[float], Optional[float]]:
    """``max_memory_allocated``/``max_memory_reserved`` in GiB since the previous call, then reset (CUDA only)."""
    if device.type != "cuda":
        return None, None
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    return allocated, reserved


def check_parameters_finite(groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]], where: str) -> None:
    """Every trainable parameter finite (one synchronisation); else ``TrainingError`` (§14.6.8)."""
    items = [(name, p) for members in groups.values() for name, p in members]
    with torch.no_grad():
        flags = torch.stack([torch.isfinite(p).all() for _, p in items])
    if not bool(flags.all()):
        bad = [items[i][0] for i in (~flags).nonzero().flatten().tolist()]
        raise TrainingError(f"{where}: trainable parameters became non-finite {bad[:5]} ({len(bad)}) (§14.6.8)")


def train_window(model: Any, optimizer: torch.optim.Optimizer, scaler: Any, batches: Iterator["TrainBatch"],
                 entries: Sequence["PlanEntry"], train_samples: Sequence["Sample"], *, step: int, sc: StepCounts,
                 tc: "TrainConfig", monitor: Any = None, clock: Optional[Any] = None) -> WindowResult:
    """One optimizer step s (§14.6.4 step 5): lr of every group set once; gate-call and cluster-lookup counters
    read; ``accumulate_window`` with the divisor ``train.grad_accum`` (the last window of an epoch too, D24);
    ``scaler.unscale_``; group gradient norms; the gradient presence rule; one transfer of the window values to the
    CPU; without a GradScaler a non-finite norm is a ``TrainingError`` before the step; ``scaler.step``/``update``;
    ``zero_grad(set_to_none=True)`` (Adam then skips parameters without a gradient, torch2.0 optim/adam.py:77);
    peak memory read and reset (CUDA)."""
    groups = model.param_groups()
    lrs = group_lrs(step, sc, tc, list(groups))
    set_lrs(optimizer, lrs)
    router = model.omega_source
    hard_gate = getattr(router, "hard_gate", None)
    calls0 = int(router.gate_calls)
    lookups0 = int(hard_gate.lookups) if hard_gate is not None else 0
    micros = accumulate_window(model, batches, entries, train_samples, precision=tc.precision, scaler=scaler,
                               divisor=int(tc.grad_accum), monitor=monitor, clock=clock)
    scaler.unscale_(optimizer)
    norm_tensors, total_tensor, presence = gradient_norms(groups)
    n_hybrid = sum(1 for m in micros if m.hybrid)
    check_grad_presence(presence, n_hybrid=n_hybrid, where=f"step {step} (tau {micros[0].tau}-{micros[-1].tau})")
    with_norm = [g for g, t in norm_tensors.items() if t is not None]
    k = int(micros[0].omega.shape[1])
    vector = torch.cat([torch.stack([m.loss.float() for m in micros]),
                        torch.stack([norm_tensors[g] for g in with_norm] + [total_tensor]).float(),
                        torch.cat([m.omega.float() for m in micros]).reshape(-1)])
    values = vector.detach().cpu().double().numpy()            # the one synchronisation of the window values
    n, m = len(micros), len(with_norm)
    norm_values = values[n:n + m + 1]
    norms = {g: (float(norm_values[with_norm.index(g)]) if g in with_norm else None) for g in groups}
    if not scaler.is_enabled() and not bool(np.isfinite(norm_values).all()):
        raise TrainingError(f"step {step}: non-finite gradient norm {norms} with precision {tc.precision} (no "
                            "GradScaler to skip the step, §14.6.4 step 5.6)")
    before, after, skipped = optimizer_step(optimizer, scaler)
    optimizer.zero_grad(set_to_none=True)
    mem_alloc, mem_reserved = peak_memory(model.device)
    return WindowResult(step=int(step), micros=micros, losses=[float(v) for v in values[:n]],
                        omega=values[n + m + 1:].reshape(-1, k), lrs=lrs, norms=norms,
                        grad_norm_total=float(norm_values[-1]), scale_before=before, scale_after=after,
                        skipped=bool(skipped), gate_calls=int(router.gate_calls) - calls0,
                        cluster_lookups=(int(hard_gate.lookups) - lookups0) if hard_gate is not None else 0,
                        max_mem_gib=mem_alloc, max_mem_reserved_gib=mem_reserved)


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def window_rows(result: WindowResult, *, epoch: int, curriculum: str, total_micro: int,
                time_s: float) -> tuple[list[dict[str, object]], StepRecord]:
    """The M12 micro-batch rows (``MICRO_FIELDS``) and the ``StepRecord`` of one window (§14.6.8, §16.6.2)."""
    rows = []
    for micro, loss in zip(result.micros, result.losses):
        rows.append({"tau": micro.tau, "epoch": int(epoch), "step": result.step, "hybrid": micro.hybrid,
                     "curriculum_p": curriculum_probability(curriculum, micro.tau, total_micro),
                     "batch_size": micro.batch_size, "seq_len": micro.seq_len, "n_label_tokens": micro.n_label_tokens,
                     "loss": loss, "loss_finite": math.isfinite(loss),
                     "act_max_abs": micro.act[0] if micro.act is not None else None,
                     "act_max_module": micro.act[1] if micro.act is not None else None, "time_s": micro.time_s})
    hybrid = [loss for micro, loss in zip(result.micros, result.losses) if micro.hybrid]
    text = [loss for micro, loss in zip(result.micros, result.losses) if not micro.hybrid]
    omega = result.omega
    k = omega.shape[1]
    entropy = None
    if k > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(omega > 0, -omega * np.log(np.where(omega > 0, omega, 1.0)), 0.0)   # 0 log 0 = 0
        entropy = float((terms.sum(axis=1) / math.log(k)).mean())
    record = StepRecord(
        step=result.step, epoch=int(epoch), tau_first=result.micros[0].tau, tau_last=result.micros[-1].tau,
        n_micro=len(result.micros), n_hybrid=len(hybrid), loss_mean=_mean(result.losses),
        loss_hybrid_mean=_mean(hybrid), loss_text_mean=_mean(text),
        nonfinite_loss=sum(1 for loss in result.losses if not math.isfinite(loss)),
        lr_lora=result.lrs["lora"], lr_projector=result.lrs["projector"], lr_gate=result.lrs.get("gate"),
        grad_norm_lora=result.norms["lora"], grad_norm_projector=result.norms["projector"],
        grad_norm_gate=result.norms.get("gate"), grad_norm_total=result.grad_norm_total,
        projector_has_grad=result.norms["projector"] is not None, scale_before=result.scale_before,
        scale_after=result.scale_after, skipped=result.skipped,
        omega_mean=tuple(float(v) for v in omega.mean(axis=0)), omega_entropy_mean=entropy,
        gate_calls=result.gate_calls, cluster_lookups=result.cluster_lookups, max_mem_gib=result.max_mem_gib,
        max_mem_reserved_gib=result.max_mem_reserved_gib, time_s=float(time_s))
    return rows, record


# ----------------------------------------------------------------------------
# Resume point (§14.6.10) [결정]: epoch boundaries only
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ResumePoint:
    """Where ``train --resume`` continues: ``complete`` epochs 0..c-1 have ``COMPLETE``; ``load_epoch`` is the
    checkpoint to load (None: nothing saved, start again); ``state_saved_only``: ``load_epoch`` = c has only
    ``STATE_SAVED`` and resumes at its validation; ``moved``: folders without a marker kept as
    ``epoch{k}.incomplete-<UTC>``."""
    complete: tuple[int, ...]
    load_epoch: Optional[int]
    state_saved_only: bool
    moved: tuple[str, ...]

    def boundary(self, sc: StepCounts) -> tuple[int, int, int]:
        """``(tau_done, global_step, epochs_complete)`` of the loaded state (M12 ``TrainRecorder.resume``)."""
        if self.load_epoch is None:
            return 0, 0, 0
        done = self.load_epoch + 1
        return sc.micro_per_epoch * done, sc.steps_per_epoch * done, len(self.complete)


def find_resume_point(run_dir: Path, epochs: int) -> ResumePoint:
    """Scan ``checkpoints/``: the ``COMPLETE`` epochs must be 0..c-1; a ``STATE_SAVED``-only folder may only be
    epoch c; folders without a marker are renamed to ``epoch{k}.incomplete-<UTC>`` (kept, never deleted)."""
    root = Path(run_dir) / CHECKPOINTS_DIR
    complete, saved, unmarked = [], [], []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        match = EPOCH_DIR_RE.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        epoch = int(match.group(1))
        if (path / COMPLETE_MARKER).is_file():
            complete.append(epoch)
        elif (path / STATE_SAVED_MARKER).is_file():
            saved.append(epoch)
        else:
            unmarked.append((epoch, path))
    complete.sort()
    c = len(complete)
    problems = []
    if complete != list(range(c)) or c > epochs:
        problems.append(f"COMPLETE epochs {complete} are not 0..c-1 within {epochs} epochs")
    if saved and saved != [c]:
        problems.append(f"STATE_SAVED-only epochs {saved}; only epoch {c} (after the last COMPLETE) can resume")
    early = [epoch for epoch, _ in unmarked if epoch < c]
    if early:
        problems.append(f"epochs {early} have no marker although later epochs are COMPLETE")
    if problems:
        raise TrainingError(f"cannot resume {run_dir}: " + "; ".join(problems))
    stamp, moved = run.utc_stamp(), []
    for epoch, path in sorted(unmarked):
        target, n = path.with_name(f"epoch{epoch}.incomplete-{stamp}"), 1
        while target.exists():
            target, n = path.with_name(f"epoch{epoch}.incomplete-{stamp}-{n}"), n + 1
        path.rename(target)
        moved.append(target.name)
        logger.warning("checkpoint folder %s without marker kept as %s", path.name, target.name)
    if saved:
        return ResumePoint(tuple(complete), c, True, tuple(moved))
    return ResumePoint(tuple(complete), c - 1 if c else None, False, tuple(moved))


def _keep_stale_val_metrics(ckpt: Path, stamp: str) -> Optional[Path]:
    """Move the ``val_metrics.json`` that a resumed ``STATE_SAVED``-only epoch left behind (no ``COMPLETE`` refers
    to it) to ``val_metrics.incomplete-<UTC>.json``, adding ``-<n>`` when that name is taken, like the other
    ``.incomplete`` names (§14.6.10, §3.1) [결정]. Never overwrites: ``Path.rename`` replaces an existing target on
    Linux. Returns the new path, or None when there is nothing to keep."""
    stale = ckpt / VAL_METRICS_FILE
    if not stale.exists():
        return None
    stem, suffix = Path(VAL_METRICS_FILE).stem, Path(VAL_METRICS_FILE).suffix
    kept, n = ckpt / f"{stem}.incomplete-{stamp}{suffix}", 1
    while kept.exists():
        kept, n = ckpt / f"{stem}.incomplete-{stamp}-{n}{suffix}", n + 1
    stale.rename(kept)
    logger.warning("%s without COMPLETE kept as %s", stale, kept.name)
    return kept


def segment_problems(segments: Sequence[Mapping[str, object]], resumes: Sequence[Mapping[str, object]]) -> list[str]:
    """Problems of the ``meta.json`` ``train.segments`` of a finished run (one segment per process, §14.6.10).

    Every segment has ``started_utc``; a completed or failed one has ``ended_utc``; ``ended_utc`` stays null only in
    a segment that a later resume marked ``interrupted`` (its process was killed before closing it, so its end was
    never measured) with its ``detected_utc``; a ``running`` segment was never closed nor marked; there are at most
    as many interrupted segments as resumes of a ``running`` folder (M0 ``resumes``); the last segment is completed.
    The E2 run check (M10-15, tests/vessl/test_run_outputs.py) and its CPU test use this rule.
    """
    if not segments:
        return ["no train segments"]
    problems = []
    for i, segment in enumerate(segments):
        status = segment.get("status")
        if not segment.get("started_utc"):
            problems.append(f"segment {i} has no started_utc")
        if status in ("completed", "failed"):
            if not segment.get("ended_utc"):
                problems.append(f"segment {i} ({status}) has no ended_utc")
        elif status == "interrupted":
            if segment.get("ended_utc") is not None or not segment.get("detected_utc"):
                problems.append(f"segment {i} (interrupted) needs ended_utc null and a detected_utc")
        elif status == "running":
            problems.append(f"segment {i} is running: never closed and not marked interrupted by a resume")
        else:
            problems.append(f"segment {i} has status {status!r}")
    interrupted = sum(1 for segment in segments if segment.get("status") == "interrupted")
    killed = sum(1 for entry in resumes if entry.get("previous_status") == "running")
    if interrupted > killed:
        problems.append(f"{interrupted} interrupted segments but {killed} resumes of a running folder")
    last = segments[-1]
    if last.get("status") != "completed" or not last.get("ended_utc"):
        problems.append(f"the last segment is {last.get('status')!r}, not completed with ended_utc")
    return problems


# ----------------------------------------------------------------------------
# run.log summary of a step interval (§14.6.8)
# ----------------------------------------------------------------------------
class _Interval:
    """Values since the previous summary line: losses per format, hybrid count, skips, gradient norms, omega."""

    def __init__(self, now: float) -> None:
        self.reset(now)

    def reset(self, now: float) -> None:
        self.t0, self.micro, self.hybrid, self.skipped = now, 0, 0, 0
        self.loss_hybrid: list[float] = []
        self.loss_text: list[float] = []
        self.grad_max = -math.inf
        self.omega_sum: Optional[np.ndarray] = None
        self.omega_n = 0
        self.mem: Optional[float] = None

    def add(self, result: WindowResult) -> None:
        for micro, loss in zip(result.micros, result.losses):
            (self.loss_hybrid if micro.hybrid else self.loss_text).append(loss)
        self.micro += len(result.micros)
        self.hybrid += sum(1 for m in result.micros if m.hybrid)
        self.skipped += int(result.skipped)
        if not math.isnan(result.grad_norm_total):
            self.grad_max = max(self.grad_max, result.grad_norm_total)
        total = result.omega.sum(axis=0)
        self.omega_sum = total if self.omega_sum is None else self.omega_sum + total
        self.omega_n += result.omega.shape[0]
        if result.max_mem_gib is not None:
            self.mem = result.max_mem_gib if self.mem is None else max(self.mem, result.max_mem_gib)

    def line(self, *, step: int, total_steps: int, epoch: int, tau: int, p: float, result: WindowResult,
             skipped_total: int, now: float) -> str:
        def mean(xs: list[float]) -> str:
            return "-" if not xs else f"{sum(xs) / len(xs):.4f}"

        omega = "-" if self.omega_sum is None else ", ".join(f"{v / self.omega_n:.4f}" for v in self.omega_sum)
        lr_gate = result.lrs.get("gate")
        rate = self.micro / max(now - self.t0, 1e-9)
        return (f"{SUMMARY_PREFIX} {step}/{total_steps} epoch {epoch} tau {tau} p {p:.4f} "
                f"loss {mean(self.loss_hybrid + self.loss_text)} (hybrid {mean(self.loss_hybrid)} n={len(self.loss_hybrid)}, "
                f"text {mean(self.loss_text)} n={len(self.loss_text)}) lr lora {result.lrs['lora']:.6e} projector "
                f"{result.lrs['projector']:.6e} gate {'-' if lr_gate is None else f'{lr_gate:.6e}'} "
                f"scale {result.scale_after:g} skipped {self.skipped} (total {skipped_total}) grad_norm last "
                f"{result.grad_norm_total:.4g} max {self.grad_max:.4g} hybrid {self.hybrid}/{self.micro} omega [{omega}] "
                f"micro/s {rate:.2f} max_mem_gib {'-' if self.mem is None else f'{self.mem:.3f}'}")


# ----------------------------------------------------------------------------
# The train command (§14.6.4-§14.6.10)
# ----------------------------------------------------------------------------
class _Trainer:
    """State of one ``train`` process: prepared objects, step/tau counters, per-epoch results."""

    def __init__(self, cfg: "Config", run_dir: Path, *, repo_root: Path, llm_path: Optional[Path]) -> None:
        self.cfg, self.tc, self.run_dir, self.repo_root, self.llm_path = cfg, cfg.train, run_dir, repo_root, llm_path
        self.k = int(cfg.lora.num_experts)
        self.status = "failed"
        self.recorder: Any = None
        self.monitor: Any = None
        self.hard_gate: Any = None
        self.sc: Optional[StepCounts] = None
        self.step, self.tau_done = 0, 0
        self.val: dict[int, Any] = {}
        self.epoch_info: dict[int, dict] = {}
        self.test: dict[int, dict] = {}
        self.test_seconds = 0.0
        self.best: Optional[int] = None
        self.last: Optional[int] = None
        self.group_counts: dict[str, int] = {}
        self.counters = {"skipped_steps": 0, "nonfinite_loss": 0, "gate_calls": 0, "cluster_lookups": 0,
                         "train_s": 0.0}
        self.guard = SkipGuard(cfg.train.max_consecutive_skips)
        self.resumes = 0
        self.meta_train: dict[str, Any] = {}
        self.frozen_at_save: dict[str, float] = {}      # frozen_checksum recomputed at the end of the last epoch

    # ------------------------------------------------------------------ meta.json (M0 record functions)
    def _update_meta(self, fields: Mapping[str, object]) -> None:
        if (self.run_dir / "meta.json").is_file():
            run.update_meta(self.run_dir, fields)

    def _write_train_meta(self) -> None:
        self._update_meta({"train": _plain(self.meta_train)})

    def ckpt_dir(self, epoch: int) -> Path:
        return self.run_dir / CHECKPOINTS_DIR / f"epoch{epoch}"

    # ------------------------------------------------------------------ step 0: preparation (fixed order)
    def prepare(self, resume: bool) -> None:
        """§14.6.4 step 0.1-0.4: data, templates, tokenizer; cluster_hard artifacts -> mapping -> gate; model;
        live z check; optimizer and GradScaler."""
        from . import commands
        from .batching import load_tokenizer
        from .data import SPLITS, load_dataset
        from .prompts import load_templates

        cfg = self.cfg
        if (self.run_dir / "meta.json").is_file():
            meta = run.read_meta(self.run_dir)
            self.resumes = len(meta.get("resumes") or [])
            self.meta_train = dict(meta.get("train") or {})
        segments = self.meta_train.setdefault("segments", [])
        now = run.utc_now().isoformat()
        if resume:
            # A segment still "running" belongs to a process that ended without its finally block (SIGKILL, OOM
            # killer, SIGHUP; §4.8 #6). M0 run_command holds run.lock, so that process is gone. Its end time was never
            # measured and stays null [SPEC §0.11]; detected_utc is when this resume found it.
            for segment in segments:
                if segment.get("status") == "running":
                    segment.update({"status": "interrupted", "detected_utc": now})
        segments.append({"started_utc": now, "resume": bool(resume), "ended_utc": None, "status": "running"})
        self.meta_train.setdefault("epochs", {})
        self._write_train_meta()
        self.tokenizer_dir = commands.tokenizer_dir_for(cfg, self.llm_path)
        if self.tokenizer_dir is None:
            raise TrainingError("tokenizer folder not available (llama2_7b: LLM_PATH, tiny_random: ILORA_TEST_TOKENIZER)")
        self.data = load_dataset(cfg, self.repo_root)
        self.templates = load_templates(run.resolve_path(cfg.prompt.template_path, self.repo_root))
        self.bundle = load_tokenizer(self.tokenizer_dir, cfg.batching.tokenizer_sha256)
        self.sc = step_counts(len(self.data.train), cfg.train)
        artifacts = None
        if cfg.router.routing_mode == "cluster_hard":           # 19.6.7 order [SPEC §9, §11]; dynamic/none skip M15
            from .cluster import ClusterHardGate, load_cluster_artifacts, parse_expert_mapping

            expected_ids = [s.sample_id for split in SPLITS for s in self.data.samples(split)]
            state_path = run.resolve_path(cfg.sasrec.state_path, self.repo_root)
            artifacts = load_cluster_artifacts(run.resolve_path(cfg.cluster.artifact_dir, self.repo_root),
                                               expected_ids=expected_ids, sasrec_sha256=run.content_sha256(state_path),
                                               data_sha256=run.input_hashes(self.repo_root), ccfg=cfg.cluster)
            mapping = parse_expert_mapping(cfg.cluster.expert_mapping, n_clusters=artifacts.n_clusters,
                                           num_experts=self.k)
            self.hard_gate = ClusterHardGate(artifacts, mapping, self.k)
        from .model import build_model

        self.model = build_model(cfg, self.bundle, self.data.catalog, llm_path=self.llm_path, hard_gate=self.hard_gate)
        if artifacts is not None:
            from .cluster import verify_live_encoder

            live = verify_live_encoder(artifacts, self.model.sasrec, self.data.train, device=self.model.device,
                                       n=cfg.cluster.live_check_n, atol=cfg.cluster.live_check_atol)
            self._update_meta({"cluster_live_check": dict(live) if isinstance(live, Mapping) else {"result": live}})
        groups = self.model.param_groups()
        want = [g for g in GROUP_ORDER if g != "gate" or cfg.router.routing_mode == "dynamic"]
        if list(groups) != want:
            raise TrainingError(f"parameter groups {list(groups)} != {want} for routing_mode {cfg.router.routing_mode} "
                                "(gate only for dynamic, [SPEC §11])")
        self.optimizer = build_optimizer(groups, cfg.train, model=self.model)
        self.scaler = make_grad_scaler(cfg.train, self.model.device)
        self.group_counts = group_sizes(groups)
        self.provenance = current_provenance(cfg, self.model, repo_root=self.repo_root, llm_path=self.llm_path,
                                             tokenizer_dir=self.tokenizer_dir)
        if "gate" not in groups:
            logger.info("train.lr_scale_gate and train.weight_decay_gate are checked but not used (no gate group in "
                        "routing_mode %s)", cfg.router.routing_mode)
        if not self.scaler.is_enabled():
            logger.info("train.scaler_* keys are checked but not used (precision %s has no GradScaler)",
                        cfg.train.precision)
        sc = self.sc
        logger.info("training lengths: %d samples, %d micro-batches/epoch (tail %d), %d steps/epoch, T %d, warmup %d, "
                    "%d micro-batches in total; trainable parameters %s", sc.n_samples, sc.micro_per_epoch, sc.tail,
                    sc.steps_per_epoch, sc.total_steps, sc.warmup_steps, sc.total_micro, self.group_counts)
        self.meta_train.update({"step_counts": dataclasses.asdict(sc), "trainable_params": self.group_counts,
                                "parameter_report": self.model.parameter_report(),
                                "generated_frozen_sha256": tensor_digest(self.model.generated_frozen_state())})
        self._write_train_meta()

    # ------------------------------------------------------------------ step 0.5-0.6: records, start values
    def open_records(self, point: Optional[ResumePoint]) -> None:
        """M12 ``TrainRecorder`` (new) or ``TrainRecorder.resume`` at the loaded boundary (§14.6.10 step 3)."""
        from .records import TrainRecorder

        sc = self.sc
        common = {"num_experts": self.k, "total_micro_batches": sc.total_micro, "total_steps": sc.total_steps}
        if point is None:
            self.recorder = TrainRecorder(self.run_dir, self.cfg.records, **common)
        else:
            tau_done, global_step, epochs_complete = point.boundary(sc)
            self.recorder = TrainRecorder.resume(self.run_dir, self.cfg.records, **common, tau_done=tau_done,
                                                 global_step=global_step, epochs_complete=epochs_complete)
        self.interval = _Interval(self.recorder.elapsed())

    def attach_monitor(self) -> None:
        """``records.activation_max``: ``ActivationMaxMonitor(model, MoELoRALinear)`` attached (M12 16.6.3)."""
        if not self.cfg.records.activation_max:
            return
        from .moe_lora import MoELoRALinear
        from .records import ActivationMaxMonitor

        self.monitor = ActivationMaxMonitor(self.model, MoELoRALinear)
        self.monitor.attach()
        self.recorder.attach_monitor(self.monitor)

    def start_values(self) -> None:
        """Frozen-parameter checksum and SASRec content digest at the start (§14.6.4 step 0.6, M3 7.8 #13)."""
        from .sasrec import state_digest

        digest = state_digest(self.model.sasrec.state_dict())
        if digest != self.model.sasrec.state_digest:
            raise TrainingError(f"SASRec content digest {digest} != the loaded state digest {self.model.sasrec.state_digest}")
        self.digest0 = digest
        self.frozen0 = frozen_checksum(self.model)

    def counters_state(self) -> dict[str, object]:
        g = self.guard
        return {**self.counters, "skip_guard": {"consecutive": g.consecutive, "total": g.total,
                                                "run_start": g.run_start, "run_scale": g.run_scale}}

    def read_val_metrics(self, epoch: int) -> None:
        """Validation result of a COMPLETE epoch from its ``val_metrics.json`` (sha256 of ``COMPLETE``)."""
        from .evaluate import SplitMetrics

        ckpt = self.ckpt_dir(epoch)
        checkpoint_files(ckpt, need=(VAL_METRICS_FILE,))
        doc = _read_json(ckpt / VAL_METRICS_FILE)
        self.val[epoch] = SplitMetrics(**doc["metrics"])
        self.epoch_info[epoch] = doc

    def load(self, point: ResumePoint) -> dict[str, object]:
        """§14.6.10 step 2: strict ``load_checkpoint`` of ``point.load_epoch`` (RNG restored, no new seed)."""
        epoch = int(point.load_epoch)
        info = load_checkpoint(self.ckpt_dir(epoch), self.model, self.optimizer, self.scaler, self.cfg,
                               provenance=self.provenance)
        tau_done, global_step, _ = point.boundary(self.sc)
        if (info["epoch"], info["global_step"], info["tau_done"]) != (epoch, global_step, tau_done):
            raise TrainingError(f"checkpoint epoch{epoch} holds epoch {info['epoch']}, step {info['global_step']}, tau "
                                f"{info['tau_done']}; expected {epoch}, {global_step}, {tau_done}")
        self.step, self.tau_done = global_step, tau_done
        counters = dict(info["counters"])
        guard = dict(counters.pop("skip_guard"))
        self.counters.update(counters)
        self.guard.consecutive, self.guard.total = int(guard["consecutive"]), int(guard["total"])
        self.guard.run_start, self.guard.run_scale = guard["run_start"], guard["run_scale"]
        for done in point.complete:
            self.read_val_metrics(done)
        logger.info("resumed from %s (epoch %d, step %d, tau %d%s); moved aside: %s", self.ckpt_dir(epoch).name, epoch,
                    self.step, self.tau_done, ", validation pending" if point.state_saved_only else "",
                    list(point.moved) or "none")
        return info

    # ------------------------------------------------------------------ one epoch (§14.6.4 epoch loop)
    def run_epoch(self, epoch: int) -> dict[str, object]:
        """Plan, loader, train mode, the accumulation windows, the epoch-end checks; returns the epoch values."""
        from . import prompts
        from .batching import make_train_loader
        from .sasrec import state_digest

        cfg, sc = self.cfg, self.sc
        started = time.perf_counter()
        plan = prompts.build_batch_plan(self.data.train, epoch, cfg)
        check_plan(plan, epoch, sc, tau_done=self.tau_done)
        curriculum = cfg.prompt.train_curriculum
        expected = sum(curriculum_probability(curriculum, e.tau, sc.total_micro) for e in plan.entries)
        digest = plan.digest()
        logger.info("epoch %d plan %s: %d hybrid micro-batches of %d (expected sum p(tau) = %.1f, %s)", epoch, digest,
                    plan.hybrid_count(), len(plan), expected, curriculum)
        batches = make_train_loader(plan, self.data, cfg, epoch, bundle=self.bundle, templates=self.templates)
        enter_train_mode(self.model)
        if self.monitor is not None:
            self.monitor.resume()
        groups = self.model.param_groups()
        entries, pos = list(plan.entries), 0
        losses: list[float] = []
        n_hybrid = n_skipped = 0
        mem: list[Optional[float]] = [None, None]
        best_act: Optional[tuple[Tensor, Tensor]] = None
        for size in sc.window_sizes():
            window, pos = entries[pos:pos + size], pos + size
            result = train_window(self.model, self.optimizer, self.scaler, batches, window, self.data.train,
                                  step=self.step, sc=sc, tc=self.tc, monitor=self.monitor, clock=self.recorder.elapsed)
            rows, record = window_rows(result, epoch=epoch, curriculum=curriculum, total_micro=sc.total_micro,
                                       time_s=self.recorder.elapsed())
            for row in rows:
                self.recorder.log_micro_batch(row)
            self.recorder.log_optimizer_step(record.as_row())
            self.tau_done += len(window)
            losses.extend(result.losses)
            n_hybrid += record.n_hybrid
            n_skipped += int(result.skipped)
            self.counters["skipped_steps"] += int(result.skipped)
            self.counters["nonfinite_loss"] += record.nonfinite_loss
            self.counters["gate_calls"] += result.gate_calls
            self.counters["cluster_lookups"] += result.cluster_lookups
            for i, value in enumerate((result.max_mem_gib, result.max_mem_reserved_gib)):
                if value is not None:
                    mem[i] = value if mem[i] is None else max(mem[i], value)
            for micro in result.micros:
                if micro.act is not None:
                    value, index = micro.act
                    if best_act is None:
                        best_act = (value.clone(), index.clone())
                    else:                                       # on the device, NaN wins (M12 16.6.3)
                        better = (value > best_act[0]) | (torch.isnan(value) & ~torch.isnan(best_act[0]))
                        best_act = (torch.where(better, value, best_act[0]), torch.where(better, index, best_act[1]))
            self.guard.update(self.step, result.skipped, result.scale_before, result.scale_after,
                              grad_norm=result.grad_norm_total)
            check_parameters_finite(groups, f"step {self.step}")
            self.step += 1
            self.interval.add(result)
            if self.step % self.tc.log_every_steps == 0 or self.step == sc.steps_per_epoch * (epoch + 1):
                now = self.recorder.elapsed()
                tau = result.micros[-1].tau
                logger.info(self.interval.line(step=self.step, total_steps=sc.total_steps, epoch=epoch, tau=tau,
                                               p=curriculum_probability(curriculum, tau, sc.total_micro), result=result,
                                               skipped_total=self.counters["skipped_steps"], now=now))
                self.interval.reset(now)
        if next(batches, None) is not None:
            raise TrainingError(f"epoch {epoch}: the train loader yields more micro-batches than the plan")
        del batches
        if (self.step, self.tau_done) != (sc.steps_per_epoch * (epoch + 1), sc.micro_per_epoch * (epoch + 1)):
            raise TrainingError(f"epoch {epoch} ended at step {self.step}, tau {self.tau_done}; expected "
                                f"{sc.steps_per_epoch * (epoch + 1)}, {sc.micro_per_epoch * (epoch + 1)}")
        self.frozen_at_save = frozen_checksum(self.model)          # saved with the epoch's checkpoint
        if self.frozen_at_save != self.frozen0:
            raise TrainingError(f"epoch {epoch}: a frozen parameter changed (checksum differs from the start)")
        if state_digest(self.model.sasrec.state_dict()) != self.digest0:
            raise TrainingError(f"epoch {epoch}: the SASRec state digest changed (M3 7.8 #13)")
        train_s = time.perf_counter() - started
        self.counters["train_s"] += train_s
        act_abs, act_module = None, None
        if best_act is not None:
            act_abs = float(best_act[0].to(torch.float32).cpu())
            act_module = self.monitor.module_names[int(best_act[1].cpu())]
        return {"plan_digest": digest, "n_hybrid": n_hybrid, "train_loss_mean": _mean(losses),
                "n_skipped_steps": n_skipped, "act_max_abs": act_abs, "act_max_module": act_module,
                "max_mem_gib": mem[0], "max_mem_reserved_gib": mem[1], "train_s": train_s}

    # ------------------------------------------------------------------ epoch end (§14.6.4 step 7, §14.6.6-§14.6.7)
    def ckpt_meta(self, epoch: int, trainable_sha256: str, tags: Sequence[str]) -> dict[str, object]:
        """``ckpt_meta`` of M11 (metrics.json and NPZ meta), the keys of the eval command (15.6.11)."""
        return {"source_run": str(self.run_dir), "checkpoint_dir": str(self.ckpt_dir(epoch)), "epoch": int(epoch),
                "tags": list(tags), "trainable_sha256": trainable_sha256}

    def finish_epoch(self, epoch: int, stats: Mapping[str, object]) -> None:
        """``recorder.flush()`` -> checkpoint (``STATE_SAVED``) -> validation -> epoch row -> ``val_metrics.json``
        -> ``COMPLETE``."""
        from .sasrec import state_digest

        self.recorder.flush()
        started = time.perf_counter()
        prov = self.provenance
        state = {"epoch": epoch, "global_step": self.step, "tau_done": self.tau_done,
                 "config_hash": prov["config_hash"], "input_hashes": prov["input_hashes"], "cluster": prov["cluster"],
                 "sasrec_state_digest": state_digest(self.model.sasrec.state_dict()), "frozen_checksum": self.frozen0,
                 "frozen_checksum_at_save": self.frozen_at_save, "counters": self.counters_state(),
                 "epoch_stats": dict(stats),
                 "step_counts": dataclasses.asdict(self.sc)}
        files = save_checkpoint(self.ckpt_dir(epoch), self.model, self.optimizer, self.scaler, state)
        self.after_save(epoch, stats, files, time.perf_counter() - started)

    def validate(self, epoch: int, trainable_sha256: str) -> tuple[Any, float]:
        """M11 ``evaluate_split`` of the validation split with the monitor paused; a completed folder of an earlier
        process is read with ``load_eval_result`` instead (15.6.11)."""
        from .evaluate import eval_dir, evaluate_split, load_eval_result

        started = time.perf_counter()
        folder = eval_dir(self.run_dir, f"epoch{epoch}", "val")
        if (folder / "metrics.json").is_file():
            logger.info("validation of epoch %d is complete in %s: read, not generated again", epoch, folder)
            return load_eval_result(folder, num_experts=self.k), time.perf_counter() - started
        if self.monitor is not None:
            self.monitor.pause()
        try:
            result = evaluate_split(self.model, self.bundle, self.data, "val", self.cfg, self.run_dir,
                                    ckpt_tag=f"epoch{epoch}", ckpt_meta=self.ckpt_meta(epoch, trainable_sha256, []),
                                    templates=self.templates)
        finally:
            if self.monitor is not None:
                self.monitor.resume()
        return result, time.perf_counter() - started

    def after_save(self, epoch: int, stats: Mapping[str, object], files: Mapping[str, str], save_s: float) -> None:
        ckpt = self.ckpt_dir(epoch)
        result, val_s = self.validate(epoch, files[TRAINABLE_FILE])
        m = result.metrics
        self.recorder.log_epoch({
            "epoch": epoch, "last_step": self.step - 1, "tau_last": self.tau_done, "plan_digest": stats["plan_digest"],
            "n_hybrid": stats["n_hybrid"], "train_loss_mean": stats["train_loss_mean"],
            "n_skipped_steps": stats["n_skipped_steps"], "val_n_total": m.n_total, "val_n_valid": m.n_valid,
            "val_n_correct": m.n_correct, "val_valid_ratio": m.valid_ratio, "val_hr_at_1": m.hr_at_1,
            "val_combined_metric": m.combined_metric, "act_max_abs": stats["act_max_abs"],
            "act_max_module": stats["act_max_module"], "max_mem_gib": stats["max_mem_gib"],
            "max_mem_reserved_gib": stats["max_mem_reserved_gib"], "train_s": stats["train_s"], "val_s": val_s,
            "save_s": save_s, "time_s": self.recorder.elapsed()})
        records_file = result.out_dir / "records.jsonl"
        doc = {"epoch": epoch, "ckpt_tag": f"epoch{epoch}", "metrics": m,
               "eval_dir": result.out_dir.relative_to(self.run_dir).as_posix(),
               "records_sha256": run.content_sha256(records_file),
               "seconds": {"train": stats["train_s"], "save": save_s, "val": val_s},
               "max_mem_gib": stats["max_mem_gib"], "max_mem_reserved_gib": stats["max_mem_reserved_gib"]}
        _keep_stale_val_metrics(ckpt, run.utc_stamp())   # a resumed STATE_SAVED-only epoch: kept, never overwritten
        val_sha = write_marker(ckpt / VAL_METRICS_FILE, doc)
        write_marker(ckpt / COMPLETE_MARKER, {"epoch": epoch, "files": {**files, VAL_METRICS_FILE: val_sha},
                                              "utc": run.utc_now().isoformat()})
        self.val[epoch] = m
        self.epoch_info[epoch] = json.loads(json.dumps(_plain(doc)))
        logger.info("train epoch %d done: train %.1f s, validation %.1f s, save %.1f s, skipped steps %d, max_mem_gib "
                    "%s; val N %d valid %d correct %d ValidRatio %.4f HR@1 %.4f CombinedMetric %.4f", epoch,
                    stats["train_s"], val_s, save_s, stats["n_skipped_steps"], stats["max_mem_gib"], m.n_total,
                    m.n_valid, m.n_correct, m.valid_ratio, m.hr_at_1, m.combined_metric)
        self.meta_train["epochs"][str(epoch)] = {**doc["seconds"], "max_mem_gib": stats["max_mem_gib"],
                                                 "max_mem_reserved_gib": stats["max_mem_reserved_gib"],
                                                 "ended_utc": run.utc_now().isoformat()}
        self._write_train_meta()

    # ------------------------------------------------------------------ selection and test (§14.6.7, D11)
    def select_and_test(self) -> None:
        from .evaluate import eval_dir, evaluate_split, load_eval_result

        last_epoch = self.sc.epochs - 1
        if not (self.ckpt_dir(last_epoch) / COMPLETE_MARKER).is_file():
            raise TrainingError(f"test evaluation before epoch {last_epoch} is COMPLETE (D11: test never selects)")
        best, last = select_epochs(self.val)
        self.best, self.last = best, last
        doc = selection_document(self.val, best, last)
        path = self.run_dir / "eval" / SELECTION_FILE
        if path.is_file():                                     # a resumed run: the same rule gives the same file
            if _read_json(path) != json.loads(json.dumps(_plain(doc))):
                raise TrainingError(f"{path} differs from the selection recomputed from the validation results")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_marker(path, doc)
        logger.info("selection: best_val epoch %d (n_correct %d), last epoch %d; test order %s", best,
                    self.val[best].n_correct, last, list(epochs_to_test(best, last)))
        for epoch in epochs_to_test(best, last):
            tags = [name for name, chosen in (("best_val", best), ("last", last)) if chosen == epoch]
            folder = eval_dir(self.run_dir, f"epoch{epoch}", "test")
            if (folder / "metrics.json").is_file():
                logger.info("test of epoch %d is complete in %s: read, not generated again", epoch, folder)
                result = load_eval_result(folder, num_experts=self.k)
            else:
                sha = load_model_state(self.ckpt_dir(epoch), self.model)
                if self.monitor is not None:
                    self.monitor.pause()
                result = evaluate_split(self.model, self.bundle, self.data, "test", self.cfg, self.run_dir,
                                        ckpt_tag=f"epoch{epoch}", ckpt_meta=self.ckpt_meta(epoch, sha, tags),
                                        templates=self.templates)
            self.test_seconds += float(result.seconds)            # the evaluation time, also for a folder read back
            self.test[epoch] = {"tags": tags, "metrics": result.metrics, "aux_protocols": dict(result.aux),
                                "eval_dir": result.out_dir.relative_to(self.run_dir).as_posix()}

    # ------------------------------------------------------------------ summary and the whole run
    def summary(self) -> TrainSummary:
        seconds = {name: float(sum(info["seconds"][key] for info in self.epoch_info.values()))
                   for name, key in (("train", "train"), ("validation", "val"), ("save", "save"))}
        seconds["test"] = float(self.test_seconds)
        seconds["total"] = float(sum(seconds.values()))
        return TrainSummary(
            status=self.status, routing_mode=self.cfg.router.routing_mode, num_experts=self.k, counts=self.sc,
            epochs_complete=len(self.val), best_val_epoch=self.best, last_epoch=self.last,
            val=dict(sorted(self.val.items())), test=dict(sorted(self.test.items())),
            trainable_params=dict(self.group_counts), seconds=seconds,
            skipped_steps=int(self.counters["skipped_steps"]), nonfinite_loss=int(self.counters["nonfinite_loss"]),
            gate_calls=int(self.counters["gate_calls"]), cluster_lookups=int(self.counters["cluster_lookups"]),
            epoch_max_mem_gib={e: info.get("max_mem_gib") for e, info in sorted(self.epoch_info.items())},
            unused_experts=tuple(int(k) for k in getattr(self.hard_gate, "unused_experts", ())),
            resumes=self.resumes)

    def run(self, resume: bool) -> TrainSummary:
        """§14.6.4 step 0 (fixed order), the epochs, selection and test inside try/finally (step 0.7)."""
        try:
            self.prepare(resume)
            point = find_resume_point(self.run_dir, self.sc.epochs) if resume else None
            self.open_records(point)
        except BaseException:
            if self.meta_train.get("segments"):
                self.meta_train["segments"][-1].update({"ended_utc": run.utc_now().isoformat(), "status": "failed"})
                self._write_train_meta()
            raise
        summary: Optional[TrainSummary] = None
        try:
            self.attach_monitor()
            self.start_values()
            start = 0
            if point is None or point.load_epoch is None:
                seed = seeding.seed_torch_global(self.cfg.run.seed, "torch_global")      # §14.6.5, after build_model
                logger.info("global torch RNG seeded with derive_seed(%d, 'torch_global') = %d", self.cfg.run.seed, seed)
            else:
                info = self.load(point)
                start = len(point.complete)
                if point.state_saved_only:                     # §14.6.10 step 1: finish from the validation
                    saved = _read_json(self.ckpt_dir(start) / STATE_SAVED_MARKER)
                    self.after_save(start, info["epoch_stats"], info["files"], float(saved.get("seconds", 0.0)))
                    start += 1
            for epoch in range(start, self.sc.epochs):
                stats = self.run_epoch(epoch)
                self.finish_epoch(epoch, stats)
            self.select_and_test()
            self.status = "completed"
        finally:
            if self.monitor is not None:
                self.monitor.detach()
            summary = self.summary()
            segment = self.meta_train["segments"][-1]
            segment.update({"ended_utc": run.utc_now().isoformat(), "status": self.status})
            self.meta_train["summary"] = {"best_val_epoch": self.best, "last_epoch": self.last,
                                          "epochs_complete": len(self.val), "seconds": summary.seconds}
            try:
                self._write_train_meta()
            finally:
                self.recorder.close(self.status, summary)
        logger.info("training completed: best_val epoch %s, last epoch %s, %d skipped steps, %s", self.best, self.last,
                    summary.skipped_steps, {k: round(v, 1) for k, v in summary.seconds.items()})
        return summary


def train(cfg: "Config", run_dir: Path, *, repo_root: Path, llm_path: Optional[Path],
          resume: bool = False) -> TrainSummary:
    """The ``train`` command (§14.6.1) inside the run folder of M0 ``run_command``; ``resume`` continues a failed or
    unfinished run at its last epoch boundary (§14.6.10). ``llm_path`` only for ``model.backbone = llama2_7b``."""
    trainer = _Trainer(cfg, Path(run_dir), repo_root=Path(repo_root),
                       llm_path=Path(llm_path) if llm_path is not None else None)
    return trainer.run(bool(resume))


__all__ = [
    "TrainingError", "GROUP_ORDER", "CHECKPOINTS_DIR", "TRAINABLE_FILE", "TRAIN_STATE_FILE", "STATE_SAVED_MARKER",
    "VAL_METRICS_FILE", "COMPLETE_MARKER", "SELECTION_FILE", "TRAIN_STATE_FORMAT", "SUMMARY_PREFIX", "SELECTION_RULE",
    "STATE_KEYS", "PROVENANCE_KEYS", "StepCounts", "StepRecord", "TrainSummary", "MicroResult", "WindowResult",
    "ResumePoint", "group_hparams", "step_counts", "lr_at", "group_lrs", "curriculum_probability", "group_sizes",
    "build_optimizer", "set_lrs", "SkipGuard", "enter_train_mode", "check_plan", "check_batch", "write_marker",
    "trainable_names", "frozen_checksum", "tensor_digest", "current_provenance", "save_checkpoint", "checkpoint_files",
    "read_trainable_state", "load_model_state", "load_checkpoint", "select_epochs", "epochs_to_test",
    "selection_document", "accumulate_window", "gradient_norms", "check_grad_presence", "optimizer_step",
    "peak_memory", "check_parameters_finite", "train_window", "window_rows", "find_resume_point", "segment_problems",
    "train",
]
