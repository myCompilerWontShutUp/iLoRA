"""M12 기록 (DESIGN.md §16; 16.2 출력, 16.6.1 공개 API, 16.6.2-16.6.5, 16.6.8-16.6.10).

Everything M12 writes goes under one run folder, at a few fixed moments and never once per
training step [SPEC §4]:

- training records (``TrainRecorder``, §16.6.2): one row per micro-batch, per optimizer step and
  per epoch, kept in memory and appended to ``records/train/*.csv`` only every
  ``records.flush_every_steps`` steps, after each epoch row, when M10 calls ``flush()`` right before
  a checkpoint and in ``close``; ``close`` also writes ``activation_module_max.csv`` and
  ``run_summary.json``. ``TrainRecorder.resume`` reopens the files of an interrupted run;
- the activation probe (``ActivationMaxMonitor``, §16.6.3): read-only forward hooks on every M8
  ``MoELoRALinear`` that keep the largest |output| per module on the device, with no
  synchronisation (fp16 overflow diagnosis, D9);
- evaluation records (``write_eval``, §16.6.4): ``records.jsonl`` and ``metrics.json`` of one
  (checkpoint, split), checked before anything is written; ``read_eval_records`` reads them back;
- the gate and representation export (``run_export``, command ``export``, §16.6.5): z and omega of
  every train, validation and test sample computed with SASRec and the M7 ``Router`` of a finished
  checkpoint (no LLM, CPU, fp32), saved as ``RepresentationTable`` NPZ files and
  ``gates_{validation,test}.csv``, cross-checked against the omega used in generation
  (``crosscheck_gates``) and analysed with M12 ``analysis.run_gate_analysis``.

M12 draws no random numbers (§16.6.8). ``EvalRecord`` is the M11 type of ``evaluate.py``
(§15.6.1); ``write_eval`` and ``crosscheck_gates`` read its fields by name and only
``read_eval_records`` imports the class, inside the function, because ``evaluate.py`` imports this
module inside its own functions (no import cycle either way). Errors are ``RecordsError``, a
``ValueError`` (§16.6.2 asks for ``ValueError`` on bad rows), always with the cause in the message.
"""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import logging
import math
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from . import run
from .candidates import FLAG_NAMES
from .data import HISTORY_LEN, PAD_ID, SPLITS, Sample, pad_histories
from .router import ROW_SUM_ATOL
from .sasrec import strict_fp32

if TYPE_CHECKING:   # type names only (§16.6.1): M3 encoder, M11 record, M0 config sections
    from .evaluate import EvalRecord
    from .sasrec import SASRecEncoder
    from .sections import Config, RecordsConfig

logger = logging.getLogger("ilora.records")


class RecordsError(ValueError):
    """A record, table, file or export step breaks the M12 rules (§16.6.2, §16.6.4, §16.6.5, §16.6.8)."""


class GateCrosscheckError(RecordsError):
    """The export omega differs from the generation omega by more than ``records.gate_crosscheck_atol`` (§16.6.5
    step 7). ``result`` is the measured ``crosscheck_gates`` entry with ``passed`` false: max |d omega| and its
    worst sample, the argmax mismatches and max |d z|, which ``run_export`` writes to ``crosscheck.json`` before
    it stops. A |d z| near the M3 CUDA z limit 1e-4 tells a z rounding alarm from a real gate mismatch (§16.7)."""

    def __init__(self, message: str, result: Mapping[str, object]) -> None:
        super().__init__(message)
        self.result = dict(result)


# ----------------------------------------------------------------------------
# Training record columns (§16.6.2). The column definitions are the 3.1 StepRecord contract.
# ----------------------------------------------------------------------------
TRAIN_DIR: Final = ("records", "train")                     # §16.2 (training run folder)
MICRO_FILE: Final = "micro_batches.csv"
STEP_FILE: Final = "optimizer_steps.csv"
EPOCH_FILE: Final = "epochs.csv"
MODULE_MAX_FILE: Final = "activation_module_max.csv"
COLUMNS_FILE: Final = "columns.json"
SUMMARY_FILE: Final = "run_summary.json"

MICRO_FIELDS: Final = ("tau", "epoch", "step", "hybrid", "curriculum_p", "batch_size", "seq_len",
                       "n_label_tokens", "loss", "loss_finite", "act_max_abs", "act_max_module", "time_s")
_STEP_HEAD: Final = ("step", "epoch", "tau_first", "tau_last", "n_micro", "n_hybrid", "loss_mean",
                     "loss_hybrid_mean", "loss_text_mean", "nonfinite_loss", "lr_lora", "lr_projector", "lr_gate",
                     "grad_norm_lora", "grad_norm_projector", "grad_norm_gate", "grad_norm_total",
                     "projector_has_grad", "scale_before", "scale_after", "skipped")
_STEP_TAIL: Final = ("omega_entropy_mean", "gate_calls", "cluster_lookups", "max_mem_gib", "max_mem_reserved_gib",
                     "time_s")
EPOCH_FIELDS: Final = ("epoch", "last_step", "tau_last", "plan_digest", "n_hybrid", "train_loss_mean",
                       "n_skipped_steps", "val_n_total", "val_n_valid", "val_n_correct", "val_valid_ratio",
                       "val_hr_at_1", "val_combined_metric", "act_max_abs", "act_max_module", "max_mem_gib",
                       "max_mem_reserved_gib", "train_s", "val_s", "save_s", "time_s")
MODULE_MAX_FIELDS: Final = ("module_index", "module", "max_abs", "tau")
# Columns that accept a 0-dim device tensor (ActivationMaxMonitor.end_micro_batch output): the value is moved
# to the CPU once per flush (§16.6.3). act_max_module then is the int64 module index of the monitor.
_TENSOR_COLUMNS: Final = ("act_max_abs", "act_max_module")


@dataclass(frozen=True)
class Column:
    """One column of the training CSVs: ``kind`` is int, float, bool or str (``columns.json``)."""
    name: str
    kind: str
    nullable: bool
    unit: str
    description: str


def _col(name: str, kind: str, unit: str, description: str, nullable: bool = False) -> Column:
    return Column(name, kind, nullable, unit, description)


_MEM = "GiB = 2^30 byte. torch.cuda.max_memory_allocated() (reserved: max_memory_reserved()) since the previous step"
# [결정] §16.6.2: definitions are fixed here, M10 computes the values (§14.6.8).
_COLUMNS: dict[str, Column] = {c.name: c for c in (
    _col("tau", "int", "-", "micro-batch number over the whole training, from 1 (D3)"),
    _col("epoch", "int", "-", "epoch, from 0"),
    _col("step", "int", "-", "optimizer step (accumulation window), from 0; skipped steps count (§14.6.2)"),
    _col("hybrid", "bool", "0/1", "BatchPlan draw of this micro-batch: 1 hybrid prompt, 0 text-only (D3)"),
    _col("curriculum_p", "float", "-", "hybrid probability of this tau: linear tau/T, all_hybrid 1, reverse 1 - tau/T"),
    _col("batch_size", "int", "samples", "B of the micro-batch"),
    _col("seq_len", "int", "tokens", "L, padded length of the micro-batch"),
    _col("n_label_tokens", "int", "tokens", "answer tokens with a label (TrainForward.n_label_tokens)"),
    _col("loss", "float", "nat", "mean NLL of the answer tokens before the accumulation division "
                                 "(TrainForward.loss); nan/inf are written as such"),
    _col("loss_finite", "bool", "0/1", "1 when loss is finite"),
    _col("act_max_abs", "float", "-", "largest |output| over all MoE-LoRA layers in this micro-batch (module output "
                                      "dtype, written as float32); empty when the monitor is off", True),
    _col("act_max_module", "str", "-", "module name of act_max_abs (first index on ties); empty when the monitor "
                                       "is off", True),
    _col("time_s", "float", "s", "time.perf_counter() seconds since the recorder was created (a resumed recorder "
                                 "continues from the last kept row)"),
    _col("tau_first", "int", "-", "first tau of the window"),
    _col("tau_last", "int", "-", "last tau of the window (epoch rows: of the epoch)"),
    _col("n_micro", "int", "micro-batches", "micro-batches in the window (16, the epoch end 9, D24)"),
    _col("n_hybrid", "int", "micro-batches", "hybrid micro-batches in the window (epoch rows: in the epoch)"),
    _col("loss_mean", "float", "nat", "mean of the n_micro micro-batch losses"),
    _col("loss_hybrid_mean", "float", "nat", "mean loss of the hybrid micro-batches; empty without one", True),
    _col("loss_text_mean", "float", "nat", "mean loss of the text-only micro-batches; empty without one", True),
    _col("nonfinite_loss", "int", "micro-batches", "micro-batches with a non-finite loss"),
    _col("lr_lora", "float", "-", "lr of the lora group in this step (§14.6.2)"),
    _col("lr_projector", "float", "-", "lr of the projector group in this step"),
    _col("lr_gate", "float", "-", "lr of the gate group; empty without a gate group (cluster_hard, K = 1)", True),
    _col("grad_norm_lora", "float", "-", "fp32 L2 norm of the lora gradients after scaler.unscale_, before step"),
    _col("grad_norm_projector", "float", "-", "same for the projector; empty when it got no gradient "
                                              "(text-only window)", True),
    _col("grad_norm_gate", "float", "-", "same for the gate; empty without gate gradient (cluster_hard, K = 1)", True),
    _col("grad_norm_total", "float", "-", "L2 norm over every gradient of the step"),
    _col("projector_has_grad", "bool", "0/1", "1 when the projector got a gradient in this step"),
    _col("scale_before", "float", "-", "GradScaler.get_scale() before scaler.step (1.0 when the scaler is off)"),
    _col("scale_after", "float", "-", "GradScaler.get_scale() after scaler.update()"),
    _col("skipped", "bool", "0/1", "scale_after < scale_before: GradScaler skipped the step (inf/NaN gradients)"),
    _col("omega_entropy_mean", "float", "-", "mean normalized entropy H(omega)/ln K over the window samples "
                                             "(0 log 0 = 0); empty for K = 1", True),
    _col("gate_calls", "int", "calls", "increase of Router.gate_calls inside this training window (no evaluation)"),
    _col("cluster_lookups", "int", "samples", "increase of ClusterHardGate.lookups inside this training window; "
                                              "0 outside cluster_hard"),
    _col("max_mem_gib", "float", "GiB", _MEM + "; empty on CPU", True),
    _col("max_mem_reserved_gib", "float", "GiB", _MEM + "; empty on CPU", True),
    _col("last_step", "int", "-", "last optimizer step of the epoch"),
    _col("plan_digest", "str", "-", "BatchPlan.digest() of the epoch (M4 §8.6.6)"),
    _col("train_loss_mean", "float", "nat", "mean micro-batch loss of the epoch"),
    _col("n_skipped_steps", "int", "steps", "skipped steps in the epoch"),
    _col("val_n_total", "int", "samples", "validation N (D17)"),
    _col("val_n_valid", "int", "samples", "valid answers"),
    _col("val_n_correct", "int", "samples", "correct answers"),
    _col("val_valid_ratio", "float", "-", "ValidRatio = n_valid / n_total (D17)"),
    _col("val_hr_at_1", "float", "-", "HR@1 = n_correct / n_valid (0 when n_valid = 0, D17)"),
    _col("val_combined_metric", "float", "-", "CombinedMetric = n_correct / n_total (D17)"),
    _col("train_s", "float", "s", "training time of the epoch"),
    _col("val_s", "float", "s", "validation time of the epoch"),
    _col("save_s", "float", "s", "checkpoint save time of the epoch"),
)}
_EPOCH_ACT = {"act_max_abs": _col("act_max_abs", "float", "-", "largest act_max_abs of the epoch", True),
              "act_max_module": _col("act_max_module", "str", "-", "its module", True)}
_EPOCH_MEM = {"max_mem_gib": _col("max_mem_gib", "float", "GiB", "largest step value of the epoch; empty on CPU", True),
              "max_mem_reserved_gib": _col("max_mem_reserved_gib", "float", "GiB",
                                           "largest step value of the epoch; empty on CPU", True)}
_MODULE_MAX_COLUMNS = (
    _col("module_index", "int", "-", "index in ActivationMaxMonitor.module_names (named_modules order)"),
    _col("module", "str", "-", "module name"),
    _col("max_abs", "float", "-", "largest |output| seen while hooked (module output dtype, as float32); "
                                  "-inf when the module never ran", True),
    _col("tau", "int", "-", "micro-batch of that value (first one); empty when never observed", True),
)


def step_fields(num_experts: int) -> tuple[str, ...]:
    """``optimizer_steps.csv`` columns: ``omega_mean_0..omega_mean_{K-1}`` grow with K (§16.6.2)."""
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 1:
        raise RecordsError(f"num_experts must be an int >= 1, got {num_experts!r}")
    return (*_STEP_HEAD, *(f"omega_mean_{k}" for k in range(num_experts)), *_STEP_TAIL)


def _columns_for(kind: str, num_experts: int) -> tuple[Column, ...]:
    """Column definitions of ``micro`` / ``step`` / ``epoch`` rows in file order."""
    if kind == "micro":
        return tuple(_COLUMNS[name] for name in MICRO_FIELDS)
    if kind == "step":
        return tuple(_COLUMNS[name] if not name.startswith("omega_mean_") else
                     _col(name, "float", "-", f"mean omega_{name.rsplit('_', 1)[1]} over the window samples")
                     for name in step_fields(num_experts))
    if kind == "epoch":
        return tuple(_EPOCH_ACT.get(name) or _EPOCH_MEM.get(name) or _COLUMNS[name] for name in EPOCH_FIELDS)
    raise RecordsError(f"unknown record kind {kind!r}")


def columns_document(num_experts: int) -> dict:
    """Content of ``columns.json``: per file the columns with type, nullability, unit and meaning (§16.6.2)."""
    def describe(cols: Iterable[Column]) -> list[dict]:
        return [dataclasses.asdict(c) for c in cols]

    return {
        "format": {"encoding": "UTF-8", "newline": "\\n", "bool": "0/1", "null": "empty cell",
                   "float": "repr(float(x)) (round-trip exact)", "str": "csv quoting rules"},
        "num_experts": num_experts,
        MICRO_FILE: describe(_columns_for("micro", num_experts)),
        STEP_FILE: describe(_columns_for("step", num_experts)),
        EPOCH_FILE: describe(_columns_for("epoch", num_experts)),
        MODULE_MAX_FILE: describe(_MODULE_MAX_COLUMNS),
    }

# ----------------------------------------------------------------------------
# Value checks, CSV text and file helpers (§16.6.2 표기)
# ----------------------------------------------------------------------------
def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _check_value(col: Column, value: object, where: str) -> None:
    """Type check of one cell at log time (ValueError, §16.6.2: 누락·추가·타입 불일치)."""
    if value is None:
        if not col.nullable:
            raise RecordsError(f"{where}: column {col.name} must not be None")
        return
    if col.name in _TENSOR_COLUMNS and isinstance(value, Tensor):
        if value.dim() != 0:
            raise RecordsError(f"{where}: column {col.name} takes a 0-dim tensor, got shape {list(value.shape)}")
        if col.kind == "float" and not value.is_floating_point():
            raise RecordsError(f"{where}: column {col.name} tensor must be floating, got {value.dtype}")
        if col.kind == "str" and (value.is_floating_point() or value.dtype == torch.bool or value.is_complex()):
            raise RecordsError(f"{where}: column {col.name} tensor must hold an integer module index, got {value.dtype}")
        return
    ok = {"int": _is_int(value), "float": _is_number(value), "bool": isinstance(value, (bool, np.bool_)),
          "str": isinstance(value, str) and value != ""}[col.kind]
    if not ok:
        raise RecordsError(f"{where}: column {col.name} expects {col.kind}{' or None' if col.nullable else ''}, "
                           f"got {type(value).__name__} {value!r}")


def _check_row(kind: str, columns: Sequence[Column], row: Mapping[str, object], where: str) -> dict:
    if not isinstance(row, Mapping):
        raise RecordsError(f"{where}: a {kind} row must be a mapping, got {type(row).__name__}")
    names = [c.name for c in columns]
    missing = [n for n in names if n not in row]
    extra = sorted(str(k) for k in row if k not in set(names))
    if missing or extra:
        raise RecordsError(f"{where}: {kind} row keys differ from the columns: missing {missing}, extra {extra}")
    for col in columns:
        _check_value(col, row[col.name], where)
    return {n: row[n] for n in names}


def _text(col: Column, value: object) -> str:
    """CSV text of one resolved value: bool 0/1, None empty, float repr (round trip exact)."""
    if value is None:
        return ""
    if col.kind == "bool":
        return "1" if value else "0"
    if col.kind == "int":
        return str(int(value))
    if col.kind == "float":
        return repr(float(value))
    return str(value)


def _parse(col: Column, text: str, where: str) -> object:
    if text == "":
        if not col.nullable:
            raise RecordsError(f"{where}: column {col.name} is empty but not nullable")
        return None
    try:
        if col.kind == "int":
            return int(text)
        if col.kind == "float":
            return float(text)
        if col.kind == "bool":
            if text not in ("0", "1"):
                raise ValueError(text)
            return text == "1"
    except ValueError as exc:
        raise RecordsError(f"{where}: column {col.name} cannot parse {text!r} as {col.kind}") from exc
    return text


def _sha256(path: Path) -> str:
    return run.content_sha256(Path(path))


def _fsync_write(path: Path, data: bytes, *, append: bool) -> None:
    with open(path, "ab" if append else "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _csv_bytes(rows: Iterable[Sequence[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _append_rows(path: Path, rows: Sequence[Sequence[str]]) -> None:
    """The only write of training rows: one open-append-fsync per file and flush (§16.6.2 buffer)."""
    _fsync_write(path, _csv_bytes(rows), append=True)


def _replace_file(path: Path, data: bytes) -> None:
    """Write ``data`` to a temporary name and ``os.replace`` it (no half-written file at ``path``)."""
    tmp = Path(str(path) + ".tmp")
    _fsync_write(tmp, data, append=False)
    os.replace(tmp, path)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[list[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise RecordsError(f"{path}: empty file (no header)") from None
        return header, [list(r) for r in reader]


def read_rows(path: Path, fields: Sequence[str]) -> list[dict[str, object]]:
    """Read a training CSV back with the column types (header must equal ``fields``; empty cells are None).

    ``fields`` is ``MICRO_FIELDS``, ``step_fields(K)`` or ``EPOCH_FIELDS``; floats come back bit for bit.
    """
    fields = tuple(fields)
    kind = {MICRO_FIELDS: "micro", EPOCH_FIELDS: "epoch"}.get(fields)
    if kind is None:
        n_omega = sum(1 for name in fields if name.startswith("omega_mean_"))
        if fields != step_fields(max(n_omega, 1)):
            raise RecordsError(f"unknown column set {list(fields)[:5]}...")
        kind, columns = "step", _columns_for("step", n_omega)
    else:
        columns = _columns_for(kind, 1)
    header, raw = _read_csv(Path(path))
    if header != fields:
        raise RecordsError(f"{path}: header {list(header)} != columns {list(fields)}")
    out = []
    for i, values in enumerate(raw, start=2):
        if len(values) != len(columns):
            raise RecordsError(f"{path}:{i}: {len(values)} cells, expected {len(columns)}")
        out.append({c.name: _parse(c, v, f"{path}:{i}") for c, v in zip(columns, values)})
    return out


def _jsonable(obj: object) -> object:
    """Plain JSON values (dataclasses, mappings, sequences, numpy values, tensors, paths; other objects such as
    ``torch.device`` or ``torch.dtype`` become their ``str``, like M0 ``write_json_atomic``)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Tensor):
        return _jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj)]
    return str(obj)


def _write_json(path: Path, obj: object) -> None:
    """JSON with fixed key order, UTF-8, LF, temporary name + ``os.replace``; token-like strings refused (M0)."""
    data = _jsonable(obj)
    hits = run.find_secrets(data)
    if hits:
        raise run.SecretLeakError(f"refusing to write token-like values at {hits} to {path}")
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    _replace_file(Path(path), text.encode("utf-8"))


def _incomplete_name(path: Path, stamp: str) -> Path:
    """``<stem>.incomplete-<UTC><suffix>`` next to ``path``; a counter keeps names unique (§16.6.2 resume)."""
    base = path.with_name(f"{path.stem}.incomplete-{stamp}{path.suffix}")
    candidate, n = base, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.incomplete-{stamp}-{n}{path.suffix}")
        n += 1
    return candidate

# ----------------------------------------------------------------------------
# Activation maximum probe (§16.6.3) [결정]
# ----------------------------------------------------------------------------
class ActivationMaxMonitor:
    """Largest |output| of every MoE-LoRA layer per training micro-batch, kept on the device (§16.6.3).

    Targets are all instances of ``wrapper_type`` (M8 ``MoELoRALinear``; 224 in paper_spec, D7) in
    ``model.named_modules()`` order. A forward hook reads the layer output (base path + LoRA sum) only:
    ``mn, mx = torch.aminmax(out.detach())``, ``torch.maximum(-mn, mx)`` is folded into a float32 [M]
    device buffer with ``torch.maximum`` (NaN and Inf propagate and are recorded). The hook returns
    nothing and draws no random numbers, so loss, gradients and RNG state are the same with and without it
    (§16.8 #3). ``end_micro_batch(tau)`` returns the buffer maximum and its module index as 0-dim device
    tensors, updates the per-module maxima and their ``tau`` on the device and resets the buffer; nothing
    calls ``.item()``, so there is no synchronisation (values reach the CPU when ``TrainRecorder`` flushes).
    Training micro-batches only: M10 calls ``pause()`` around validation and generation (§14.6.7).
    """

    module_names: tuple[str, ...]

    def __init__(self, model: nn.Module, wrapper_type: type[nn.Module]) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not (isinstance(wrapper_type, type) and issubclass(wrapper_type, nn.Module)):
            raise TypeError(f"wrapper_type must be an nn.Module subclass, got {wrapper_type!r}")
        self.wrapper_type = wrapper_type
        self._targets = [(name, module) for name, module in model.named_modules() if isinstance(module, wrapper_type)]
        self.module_names = tuple(name for name, _ in self._targets)
        self._handles: list[Any] = []
        self._paused = False
        self._buffer: Optional[Tensor] = None      # float32 [M]: max |output| of the current micro-batch
        self._best: Optional[Tensor] = None        # float32 [M]: max |output| per module while hooked
        self._best_tau: Optional[Tensor] = None    # int64 [M]: tau of _best (-1 = never observed)

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    @staticmethod
    def _output_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
        base = getattr(module, "base", None)
        weight = getattr(base, "weight", None) if base is not None else None
        if not isinstance(weight, Tensor):
            weight = next(module.parameters(), None)
        if weight is None:
            raise RecordsError(f"{type(module).__name__} has no parameter to tell its device and dtype")
        return weight.device, weight.dtype

    def attach(self) -> None:
        """Register the hooks. No target, several devices or an ``aminmax`` the device cannot run -> error."""
        if self._handles:
            raise RecordsError("ActivationMaxMonitor is already attached")
        if not self._targets:
            raise RecordsError(f"no {self.wrapper_type.__name__} module in the model: the activation probe needs the "
                               "MoE-LoRA wrappers and is not switched off silently (16.6.3)")
        places = {self._output_dtype(module) for _, module in self._targets}
        devices = {device for device, _ in places}
        if len(devices) != 1:
            raise RecordsError(f"MoE-LoRA layers are on several devices: {sorted(str(d) for d in devices)}")
        device = devices.pop()
        for dtype in sorted({dtype for _, dtype in places}, key=str):
            try:                                     # torch 2.0 CPU has no Half aminmax (16.6.3)
                torch.aminmax(torch.zeros(2, dtype=dtype, device=device))
            except RuntimeError as exc:
                raise RecordsError(f"torch.aminmax is not available for {dtype} on {device}: {exc}") from exc
        m = len(self._targets)
        self._buffer = torch.full((m,), -math.inf, dtype=torch.float32, device=device)
        self._best = torch.full((m,), -math.inf, dtype=torch.float32, device=device)
        self._best_tau = torch.full((m,), -1, dtype=torch.int64, device=device)
        self._paused = False
        for index, (_, module) in enumerate(self._targets):
            self._handles.append(module.register_forward_hook(self._hook(index)))

    def _hook(self, index: int) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if self._paused:
                return None
            if not isinstance(output, Tensor):
                raise RecordsError(f"{self.module_names[index]}: the probe expects a tensor output, "
                                   f"got {type(output).__name__}")
            mn, mx = torch.aminmax(output.detach())
            current = torch.maximum(-mn, mx).to(torch.float32)
            self._buffer[index] = torch.maximum(self._buffer[index], current)
            return None

        return hook

    def pause(self) -> None:
        """Evaluation and generation: hooks return at once (§16.6.3)."""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def detach(self) -> None:
        """Remove every hook (idempotent; M10 calls it in ``finally``). The maxima stay readable."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def end_micro_batch(self, tau: int) -> tuple[Tensor, Tensor]:
        """``(max_abs [], module_idx [])`` on the device for this micro-batch; no synchronisation."""
        if self._buffer is None or self._best is None or self._best_tau is None:
            raise RecordsError("ActivationMaxMonitor.end_micro_batch before attach()")
        if isinstance(tau, bool) or not isinstance(tau, int) or tau < 1:
            raise RecordsError(f"tau must be an int >= 1, got {tau!r}")
        value, index = torch.max(self._buffer, dim=0)        # first index of the maximum; NaN wins
        better = (self._buffer > self._best) | (torch.isnan(self._buffer) & ~torch.isnan(self._best))
        self._best = torch.where(better, self._buffer, self._best)
        self._best_tau = torch.where(better, tau, self._best_tau)
        self._buffer.fill_(-math.inf)
        return value, index

    def module_maxima(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-module maximum (float32 [M]) and its tau (int64 [M], -1 = never) on the CPU (one sync)."""
        if self._best is None or self._best_tau is None:
            raise RecordsError("ActivationMaxMonitor.module_maxima before attach()")
        return self._best.detach().cpu().numpy().copy(), self._best_tau.detach().cpu().numpy().copy()

# ----------------------------------------------------------------------------
# Training records (§16.6.2) [결정]
# ----------------------------------------------------------------------------
_KINDS: Final = ("micro", "step", "epoch")
_FILES: Final = {"micro": MICRO_FILE, "step": STEP_FILE, "epoch": EPOCH_FILE}
# Row number column, first number and the boundary rule of resume (§16.6.2 resume, M10 §14.6.10).
_NUMBER_KEY: Final = {"micro": ("tau", 1), "step": ("step", 0), "epoch": ("epoch", 0)}


def _positive(name: str, value: object, low: int = 1) -> int:
    if not _is_int(value) or int(value) < low:
        raise RecordsError(f"{name} must be an int >= {low}, got {value!r}")
    return int(value)


def _tensors_to_cpu(tensors: Sequence[Tensor], dtype: torch.dtype) -> list:
    """Values of 0-dim tensors as Python numbers, one ``.cpu()`` per device (the only sync of the probe)."""
    out: list = [None] * len(tensors)
    by_device: dict[torch.device, list[int]] = {}
    for i, t in enumerate(tensors):
        by_device.setdefault(t.device, []).append(i)
    for device, idx in by_device.items():
        values = torch.cat([tensors[i].detach().reshape(1).to(dtype) for i in idx]).cpu().tolist()
        for i, v in zip(idx, values):
            out[i] = v
    return out


class TrainRecorder:
    """Training rows of M10 buffered in memory and appended to ``records/train/`` at fixed moments (§16.6.2).

    Files: ``micro_batches.csv`` (``MICRO_FIELDS``), ``optimizer_steps.csv`` (``step_fields(K)``),
    ``epochs.csv`` (``EPOCH_FIELDS``) with their header written once when created, ``columns.json``
    (types, units, meanings), and in ``close`` ``activation_module_max.csv`` and ``run_summary.json``.
    A row must have exactly the columns with the right types (else ``ValueError``); numbers must continue
    without gaps (tau from 1, step and epoch from 0) and a step row must close the micro-batches logged
    since the previous one. Writes happen only when ``step % flush_every_steps == 0`` (step from 0),
    after ``log_epoch``, when M10 calls ``flush()`` before a checkpoint, and in ``close``: one
    append + ``flush()`` + ``os.fsync()`` per file with pending rows (``flush_count`` counts them).
    ``act_max_abs``/``act_max_module`` may be the 0-dim device tensors of
    ``ActivationMaxMonitor.end_micro_batch``; the index becomes the module name when the rows are written,
    which needs ``attach_monitor(monitor)``. ``elapsed()`` gives the ``time_s`` value.
    """

    def __init__(self, run_dir: Path, cfg: "RecordsConfig", *, num_experts: int,
                 total_micro_batches: int, total_steps: int) -> None:
        self._setup(run_dir, cfg, num_experts, total_micro_batches, total_steps)
        names = [self._paths[k] for k in _KINDS] + [self._dir / COLUMNS_FILE]
        existing = [p.name for p in names if p.exists()]
        if existing:
            raise RecordsError(f"training records already exist in {self._dir} ({existing}); a resumed run uses "
                               "TrainRecorder.resume (16.6.2)")
        self._dir.mkdir(parents=True, exist_ok=True)
        _write_json(self._dir / COLUMNS_FILE, columns_document(self.num_experts))
        for kind in _KINDS:
            _replace_file(self._paths[kind], _csv_bytes([[c.name for c in self._columns[kind]]]))

    def _setup(self, run_dir: Path, cfg: "RecordsConfig", num_experts: int, total_micro_batches: int,
               total_steps: int) -> None:
        self.num_experts = _positive("num_experts", num_experts)
        self.total_micro_batches = _positive("total_micro_batches", total_micro_batches)
        self.total_steps = _positive("total_steps", total_steps)
        if self.total_steps > self.total_micro_batches:
            raise RecordsError(f"total_steps {total_steps} > total_micro_batches {total_micro_batches}")
        self.flush_every_steps = _positive("records.flush_every_steps", getattr(cfg, "flush_every_steps", None))
        self.activation_max = bool(getattr(cfg, "activation_max", False))
        self.run_dir = Path(run_dir)
        self._dir = self.run_dir.joinpath(*TRAIN_DIR)
        self._columns = {kind: _columns_for(kind, self.num_experts) for kind in _KINDS}
        self._paths = {kind: self._dir / _FILES[kind] for kind in _KINDS}
        self._pending: dict[str, list[dict]] = {kind: [] for kind in _KINDS}
        self._next_tau, self._next_step, self._next_epoch, self._window_end = 1, 0, 0, 0
        self._t0 = time.perf_counter()
        self._time_offset = 0.0
        self._monitor: Optional[ActivationMaxMonitor] = None
        self._closed = False
        self.flush_count = 0
        self._counts = {kind: 0 for kind in _KINDS}
        self._skipped_steps: list[int] = []
        self._gate_calls = 0
        self._cluster_lookups = 0
        self._nonfinite = 0
        self._act_best: Optional[tuple[float, Optional[str], int]] = None
        self._module_first_tau = 1

    # ------------------------------------------------------------------ resume (§16.6.2, M10 §14.6.10)
    @classmethod
    def resume(cls, run_dir: Path, cfg: "RecordsConfig", *, num_experts: int, total_micro_batches: int,
               total_steps: int, tau_done: int, global_step: int, epochs_complete: int) -> "TrainRecorder":
        """Reopen the records of an interrupted run at the loaded checkpoint boundary.

        The three headers must equal the current columns. Rows outside the boundary (micro tau > ``tau_done``,
        step >= ``global_step``, epoch >= ``epochs_complete``) move to ``<name>.incomplete-<UTC>.csv`` and the
        files are rewritten up to the boundary; the rows inside must be exactly tau 1..``tau_done``, steps
        0..``global_step`` - 1 and epochs 0..``epochs_complete`` - 1 (else ``ValueError``). A previous
        ``run_summary.json`` and ``activation_module_max.csv`` move aside the same way (nothing is
        overwritten). Logging continues at tau ``tau_done`` + 1, step ``global_step``, epoch ``epochs_complete``.
        """
        self = cls.__new__(cls)
        self._setup(run_dir, cfg, num_experts, total_micro_batches, total_steps)
        limits = {"micro": _positive("tau_done", tau_done, 0), "step": _positive("global_step", global_step, 0),
                  "epoch": _positive("epochs_complete", epochs_complete, 0)}
        if limits["micro"] > self.total_micro_batches or limits["step"] > self.total_steps:
            raise RecordsError(f"resume boundary tau_done {tau_done}, global_step {global_step} is beyond the totals "
                               f"{self.total_micro_batches}, {self.total_steps}")
        columns_path = self._dir / COLUMNS_FILE
        if columns_path.exists():
            if json.loads(columns_path.read_text(encoding="utf-8")) != _jsonable(columns_document(self.num_experts)):
                raise RecordsError(f"{columns_path} differs from the current column definitions")
        plans: dict[str, tuple[tuple[str, ...], list[list[str]], list[list[str]]]] = {}
        for kind in _KINDS:                                  # pass 1: read and check everything
            path, cols = self._paths[kind], self._columns[kind]
            header = tuple(c.name for c in cols)
            key, first = _NUMBER_KEY[kind]
            limit = limits[kind]
            if not path.exists():
                if limit:
                    raise RecordsError(f"{path} is missing but {limit} rows inside the boundary are expected")
                plans[kind] = (header, [], [])
                continue
            found, raw = _read_csv(path)
            if found != header:
                raise RecordsError(f"{path}: header {list(found)} differs from the current columns {list(header)}")
            k = header.index(key)
            keep, out = [], []
            for i, values in enumerate(raw, start=2):
                if len(values) != len(header):
                    raise RecordsError(f"{path}:{i}: {len(values)} cells, expected {len(header)}")
                number = _parse(cols[k], values[k], f"{path}:{i}")
                inside = number <= limit if kind == "micro" else number < limit
                (keep if inside else out).append(values)
            numbers = [int(v[k]) for v in keep]
            if numbers != list(range(first, first + limit)):
                raise RecordsError(f"{path}: rows inside the boundary ({key} {'<=' if kind == 'micro' else '<'} "
                                   f"{limit}) are {len(numbers)} ({numbers[:3]}...{numbers[-3:]}), expected {limit} "
                                   f"rows numbered from {first} without gaps")
            plans[kind] = (header, keep, out)
        typed = {kind: [{c.name: _parse(c, v, str(self._paths[kind])) for c, v in zip(self._columns[kind], values)}
                        for values in plans[kind][1]] for kind in _KINDS}
        if (limits["micro"] > 0) != (limits["step"] > 0) or limits["epoch"] > limits["step"]:
            raise RecordsError(f"inconsistent resume boundary: tau_done {tau_done}, global_step {global_step}, "
                               f"epochs_complete {epochs_complete}")
        if typed["step"] and typed["step"][-1]["tau_last"] != limits["micro"]:
            raise RecordsError(f"the last kept step row ends at tau {typed['step'][-1]['tau_last']}, not at tau_done "
                               f"{limits['micro']}")
        stamp = run.utc_stamp()
        self._dir.mkdir(parents=True, exist_ok=True)
        if not columns_path.exists():
            _write_json(columns_path, columns_document(self.num_experts))
        for kind in _KINDS:                                  # pass 2: move rows aside, rewrite to the boundary
            header, keep, out = plans[kind]
            if out:
                _replace_file(_incomplete_name(self._paths[kind], stamp), _csv_bytes([header, *out]))
            _replace_file(self._paths[kind], _csv_bytes([header, *keep]))
        for name in (MODULE_MAX_FILE, SUMMARY_FILE):
            old = self._dir / name
            if old.exists():
                os.replace(old, _incomplete_name(old, stamp))
        for kind in _KINDS:
            self._aggregate(kind, typed[kind])
        self._next_tau, self._next_step, self._next_epoch = limits["micro"] + 1, limits["step"], limits["epoch"]
        self._window_end = limits["micro"]
        times = [row["time_s"] for kind in _KINDS for row in typed[kind]]
        self._time_offset = max(times, default=0.0)
        self._module_first_tau = limits["micro"] + 1
        logger.info("training records resumed at tau %d, step %d, epoch %d (%s)", limits["micro"], limits["step"],
                    limits["epoch"], {k: len(plans[k][2]) for k in _KINDS})
        return self

    # ------------------------------------------------------------------ logging
    def attach_monitor(self, monitor: ActivationMaxMonitor) -> None:
        """Use ``monitor.module_names`` for tensor module indices and its maxima in ``close`` (§16.6.3)."""
        if not isinstance(monitor, ActivationMaxMonitor):
            raise TypeError(f"monitor must be an ActivationMaxMonitor, got {type(monitor).__name__}")
        self._monitor = monitor

    def elapsed(self) -> float:
        """``time_s``: ``time.perf_counter()`` seconds since the recorder was created (resume adds the last value)."""
        return self._time_offset + (time.perf_counter() - self._t0)

    def _check_open(self) -> None:
        if self._closed:
            raise RecordsError("TrainRecorder is closed")

    def _row(self, kind: str, row: Mapping[str, object]) -> dict:
        self._check_open()
        number_key = _NUMBER_KEY[kind][0]
        label = row.get(number_key) if isinstance(row, Mapping) else None
        values = _check_row(kind, self._columns[kind], row, f"{_FILES[kind]} row {number_key} {label!r}")
        if isinstance(values.get("act_max_module"), Tensor) and self._monitor is None:
            raise RecordsError("act_max_module is a module index tensor: call attach_monitor(monitor) first")
        return values

    def log_micro_batch(self, row: Mapping[str, object]) -> None:
        """One micro-batch row (``MICRO_FIELDS``); ``act_max_*`` may be 0-dim device tensors (no sync)."""
        values = self._row("micro", row)
        tau, epoch, step = int(values["tau"]), int(values["epoch"]), int(values["step"])
        if tau != self._next_tau or tau > self.total_micro_batches:
            raise RecordsError(f"micro row: tau must be {self._next_tau} (<= {self.total_micro_batches}), got {tau}")
        if epoch != self._next_epoch or step != self._next_step:
            raise RecordsError(f"micro row tau {tau}: (epoch, step) must be ({self._next_epoch}, {self._next_step}), "
                               f"got ({epoch}, {step})")
        self._pending["micro"].append(values)
        self._next_tau += 1

    def log_optimizer_step(self, row: Mapping[str, object]) -> None:
        """One ``StepRecord`` row (``step_fields(K)``) after its micro-batch rows; flushes when step % N == 0."""
        values = self._row("step", row)
        step, epoch = int(values["step"]), int(values["epoch"])
        first, last, n = int(values["tau_first"]), int(values["tau_last"]), int(values["n_micro"])
        if step != self._next_step or step >= self.total_steps or epoch != self._next_epoch:
            raise RecordsError(f"step row: (step, epoch) must be ({self._next_step}, {self._next_epoch}) with step < "
                               f"{self.total_steps}, got ({step}, {epoch})")
        if first != self._window_end + 1 or last != self._next_tau - 1 or n != last - first + 1 or n < 1:
            raise RecordsError(f"step row {step}: window tau {first}..{last} with n_micro {n} does not close the logged "
                               f"micro-batches {self._window_end + 1}..{self._next_tau - 1}")
        if not 0 <= int(values["n_hybrid"]) <= n:
            raise RecordsError(f"step row {step}: n_hybrid {values['n_hybrid']} outside 0..{n}")
        self._pending["step"].append(values)
        self._next_step += 1
        self._window_end = last
        if step % self.flush_every_steps == 0:
            self._write_pending()

    def log_epoch(self, row: Mapping[str, object]) -> None:
        """One epoch row (``EPOCH_FIELDS``) after its last step; always flushes."""
        values = self._row("epoch", row)
        epoch = int(values["epoch"])
        if epoch != self._next_epoch:
            raise RecordsError(f"epoch row: epoch must be {self._next_epoch}, got {epoch}")
        if int(values["last_step"]) != self._next_step - 1 or int(values["tau_last"]) != self._window_end \
                or self._window_end != self._next_tau - 1:
            raise RecordsError(f"epoch row {epoch}: last_step {values['last_step']} / tau_last {values['tau_last']} must "
                               f"be the last logged step {self._next_step - 1} / tau {self._next_tau - 1}, and every "
                               "micro-batch must belong to a step")
        self._pending["epoch"].append(values)
        self._next_epoch += 1
        self._write_pending()

    def flush(self) -> None:
        """Write pending rows now (M10 calls it right before saving a checkpoint, §14.6.4)."""
        self._check_open()
        self._write_pending()

    # ------------------------------------------------------------------ writing
    def _resolve_tensors(self) -> None:
        rows = [row for kind in ("micro", "epoch") for row in self._pending[kind]]
        values = [(row, "act_max_abs") for row in rows if isinstance(row["act_max_abs"], Tensor)]
        if values:
            numbers = _tensors_to_cpu([row[name] for row, name in values], torch.float32)
            for (row, name), number in zip(values, numbers):
                row[name] = float(number)
        indices = [(row, "act_max_module") for row in rows if isinstance(row["act_max_module"], Tensor)]
        if indices:
            names = self._monitor.module_names if self._monitor is not None else ()
            numbers = _tensors_to_cpu([row[name] for row, name in indices], torch.int64)
            for (row, name), number in zip(indices, numbers):
                if not 0 <= int(number) < len(names):
                    raise RecordsError(f"module index {number} outside the {len(names)} monitored modules")
                row[name] = names[int(number)]

    def _write_pending(self) -> None:
        kinds = [kind for kind in _KINDS if self._pending[kind]]
        if not kinds:
            return
        self._resolve_tensors()
        for kind in kinds:
            rows = self._pending[kind]
            cols = self._columns[kind]
            _append_rows(self._paths[kind], [[_text(c, row[c.name]) for c in cols] for row in rows])
            self._pending[kind] = []
            self._aggregate(kind, rows)
        self.flush_count += 1

    def _aggregate(self, kind: str, rows: Sequence[Mapping[str, object]]) -> None:
        """Totals for ``run_summary.json`` (also rebuilt from the kept rows on resume)."""
        self._counts[kind] += len(rows)
        for row in rows:
            if kind == "micro":
                self._nonfinite += 0 if row["loss_finite"] else 1
                value = row["act_max_abs"]
                if value is not None:
                    value = float(value)
                    best = self._act_best
                    if best is None or (not math.isnan(best[0]) and (math.isnan(value) or value > best[0])):
                        module = row["act_max_module"]
                        self._act_best = (value, None if module is None else str(module), int(row["tau"]))
            elif kind == "step":
                if row["skipped"]:
                    self._skipped_steps.append(int(row["step"]))
                self._gate_calls += int(row["gate_calls"])
                self._cluster_lookups += int(row["cluster_lookups"])

    def _routing_mode(self, summary: Mapping[str, object]) -> Optional[str]:
        mode = summary.get("routing_mode")
        if mode is None and (self.run_dir / "config.yaml").is_file():
            import yaml

            config = yaml.safe_load((self.run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
            mode = (config.get("router") or {}).get("routing_mode")
        return mode

    def close(self, status: Literal["completed", "failed"], summary: Mapping[str, object]) -> None:
        """Write the rest (M10 ``finally``): pending rows, ``activation_module_max.csv``, ``run_summary.json``.

        ``summary`` holds the M10 ``TrainSummary`` fields so far (a mapping or the dataclass); it is kept whole
        under ``train_summary``. Write errors propagate (chained to the exception being handled, if any).
        """
        if self._closed:
            raise RecordsError("TrainRecorder is already closed")
        if status not in ("completed", "failed"):
            raise RecordsError(f"status must be completed or failed, got {status!r}")
        self._closed = True
        self._write_pending()
        plain = _jsonable(summary) if summary is not None else {}
        if not isinstance(plain, dict):
            raise RecordsError(f"summary must be a mapping or a dataclass, got {type(summary).__name__}")
        names: tuple[str, ...] = ()
        rows: list[list[str]] = []
        if self._monitor is not None and self._monitor._best is not None:
            names = self._monitor.module_names
            maxima, taus = self._monitor.module_maxima()
            rows = [[str(i), name, _text(_MODULE_MAX_COLUMNS[2], float(v)), "" if t < 0 else str(int(t))]
                    for i, (name, v, t) in enumerate(zip(names, maxima.tolist(), taus.tolist()))]
        _replace_file(self._dir / MODULE_MAX_FILE, _csv_bytes([MODULE_MAX_FIELDS, *rows]))
        best = self._act_best
        report = {
            "status": status,
            "routing_mode": self._routing_mode(plain),
            "num_experts": self.num_experts,
            "micro_batches": self._counts["micro"],
            "optimizer_steps": self._counts["step"],
            "epochs": self._counts["epoch"],
            "expected": {"micro_batches": self.total_micro_batches, "optimizer_steps": self.total_steps},
            "skipped_steps": list(self._skipped_steps),
            "n_skipped_steps": len(self._skipped_steps),
            "nonfinite_loss_micro_batches": self._nonfinite,
            "gate_calls": self._gate_calls,
            "cluster_lookups": self._cluster_lookups,
            "activation_max": {"enabled": self.activation_max, "monitor": self._monitor is not None,
                               "hook_modules": len(names),
                               "max_abs": None if best is None else best[0],
                               "module": None if best is None else best[1],
                               "tau": None if best is None else best[2],
                               "module_maxima_from_tau": self._module_first_tau},
            "flushes": self.flush_count,
            "time_s": self.elapsed(),
            "closed_utc": run.utc_now().isoformat(),
            "train_summary": plain,
        }
        report["files"] = {name: _sha256(self._dir / name)
                           for name in (MICRO_FILE, STEP_FILE, EPOCH_FILE, MODULE_MAX_FILE, COLUMNS_FILE)
                           if (self._dir / name).exists()}
        _write_json(self._dir / SUMMARY_FILE, report)
        logger.info("training records closed (%s): %d micro-batches, %d steps, %d epochs, %d skipped steps", status,
                    self._counts["micro"], self._counts["step"], self._counts["epoch"], len(self._skipped_steps))

# ----------------------------------------------------------------------------
# Representation tables (§16.2 NPZ 형식, §16.6.5 steps 5-6, §16.6.8)
# ----------------------------------------------------------------------------
# File names use the SPEC spelling (validation), the split values inside the files the contract value (val)
# [결정] §16.2. M15 writes its cluster/ tables with the same names.
SPEC_SPLIT_NAMES: Final = {"train": "train", "val": "validation", "test": "test"}
NPZ_KEYS: Final = ("sample_id", "split", "row", "user_id", "z", "omega", "argmax_expert", "meta_json")
_NPZ_GATE_KEYS: Final = ("omega", "argmax_expert")
_ZIP_DATE: Final = (1980, 1, 1, 0, 0, 0)          # fixed member time: the same table gives the same bytes
CSV_FLOAT_FORMAT: Final = "%.9g"                 # float32 round trip exact; '%.8g' missed 3,524 of 300,000 (§16.7)


@dataclass(frozen=True, eq=False)
class RepresentationTable:
    """z (and omega) of the samples of one split in M1 row order (§16.6.1; NPZ format §16.2).

    ``rows`` and ``user_ids`` int64 [N], ``z`` float32 [N, d]; ``omega`` float32 [N, K] and ``argmax`` int64 [N]
    (first index of the row maximum), or both None when only z was computed (M15). ``sample_ids[i]`` is
    ``f"{split}:{rows[i]}"`` (3.1) with rows strictly ascending; ``meta`` is JSON data. ``eq=False``: arrays
    have no single truth value, so compare the fields with numpy.
    """
    split: str
    sample_ids: tuple[str, ...]
    rows: np.ndarray
    user_ids: np.ndarray
    z: np.ndarray                    # float32 [N, d]
    omega: Optional[np.ndarray]      # float32 [N, K]
    argmax: Optional[np.ndarray]     # int64 [N]
    meta: Mapping[str, object]

    def __post_init__(self) -> None:
        where = f"RepresentationTable(split={self.split!r})"
        if self.split not in SPLITS:
            raise RecordsError(f"{where}: split must be one of {SPLITS}")
        ids = self.sample_ids
        if not isinstance(ids, tuple) or not ids or not all(isinstance(s, str) for s in ids):
            raise RecordsError(f"{where}: sample_ids must be a non-empty tuple of str")
        n = len(ids)

        def check(name: str, value: object, dtype: type, ndim: int) -> np.ndarray:
            if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != ndim or value.shape[0] != n:
                got = (f"{value.dtype} {list(value.shape)}" if isinstance(value, np.ndarray) else type(value).__name__)
                raise RecordsError(f"{where}: {name} must be a {np.dtype(dtype).name} array of {ndim} dims with "
                                   f"{n} rows, got {got}")
            return value

        rows = check("rows", self.rows, np.int64, 1)
        check("user_ids", self.user_ids, np.int64, 1)
        z = check("z", self.z, np.float32, 2)
        if z.shape[1] < 1:
            raise RecordsError(f"{where}: z needs d >= 1 columns, got {list(z.shape)}")
        for i, (sid, row) in enumerate(zip(ids, rows.tolist())):
            if sid != f"{self.split}:{row}":
                raise RecordsError(f"{where}: sample_ids[{i}] = {sid!r} is not '{self.split}:{row}' (3.1 sample_id)")
        if n > 1 and not bool(np.all(np.diff(rows) > 0)):
            raise RecordsError(f"{where}: rows must be strictly ascending (M1 row order, 16.2)")
        if (self.omega is None) != (self.argmax is None):
            raise RecordsError(f"{where}: omega and argmax are both given or both None")
        if self.omega is not None:
            omega = check("omega", self.omega, np.float32, 2)
            argmax = check("argmax", self.argmax, np.int64, 1)
            if omega.shape[1] < 1:
                raise RecordsError(f"{where}: omega needs K >= 1 columns, got {list(omega.shape)}")
            if not np.array_equal(argmax, np.argmax(omega, axis=1).astype(np.int64)):
                raise RecordsError(f"{where}: argmax differs from the first index of the omega row maximum")
        if not isinstance(self.meta, Mapping):
            raise RecordsError(f"{where}: meta must be a mapping, got {type(self.meta).__name__}")
        object.__setattr__(self, "meta", dict(self.meta))

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def dim(self) -> int:
        """d, read from the z shape (64 is never written in code, [SPEC §4])."""
        return int(self.z.shape[1])

    @property
    def num_experts(self) -> Optional[int]:
        return None if self.omega is None else int(self.omega.shape[1])


def _bad_rows_message(values: Tensor, bad: Tensor, sample_ids: Optional[Sequence[str]]) -> str:
    rows = bad.nonzero().flatten()[:5].tolist()
    shown = [(sample_ids[r] if sample_ids is not None and r < len(sample_ids) else f"row {r}", values[r].tolist())
             for r in rows]
    return f"{int(bad.sum())} rows, first {shown}"


def validate_z(z: Tensor, *, batch: int, dim: int, sample_ids: Optional[Sequence[str]] = None) -> None:
    """z contract (3.1, §16.6.8): float32 tensor of shape exactly [batch, dim] (never squeezed), finite."""
    if not isinstance(z, Tensor):
        raise RecordsError(f"z must be a torch.Tensor, got {type(z).__name__}")
    if z.dtype != torch.float32:
        raise RecordsError(f"z must be float32 (3.1 z), got {z.dtype}; it is not converted silently")
    if tuple(z.shape) != (batch, dim):
        raise RecordsError(f"z must have shape [{batch}, {dim}], got {list(z.shape)} (no squeeze, [SPEC §4])")
    finite = torch.isfinite(z.detach()).all(dim=1)
    if not bool(finite.all()):
        raise RecordsError("z has NaN/Inf: " + _bad_rows_message(z.detach(), ~finite, sample_ids))


def validate_gate(omega: Tensor, *, batch: int, num_experts: int, rowsum_atol: float,
                  sample_ids: Optional[Sequence[str]] = None) -> None:
    """omega contract (3.1, §16.6.8): float32 [batch, K], no NaN/Inf, 0 <= omega <= 1 + atol, |row sum - 1| <= atol.

    The first five offending rows (sample_id when ``sample_ids`` is given, else the row index) and their values
    are in the message. fp32 softmax over K = 4 deviates at most 2.0e-7 from a row sum of 1 (16.6.8).
    """
    if not isinstance(omega, Tensor):
        raise RecordsError(f"omega must be a torch.Tensor, got {type(omega).__name__}")
    if not _is_number(rowsum_atol) or not math.isfinite(rowsum_atol) or rowsum_atol <= 0:
        raise RecordsError(f"rowsum_atol must be a finite number > 0, got {rowsum_atol!r}")
    if omega.dtype != torch.float32:
        raise RecordsError(f"omega must be float32 (3.1 omega), got {omega.dtype}; it is not converted silently")
    if tuple(omega.shape) != (batch, num_experts):
        raise RecordsError(f"omega must have shape [{batch}, {num_experts}] = [B, K], got {list(omega.shape)} "
                           "(no squeeze, [SPEC §4])")
    values = omega.detach()
    ok = (torch.isfinite(values).all(dim=1) & (values >= 0).all(dim=1) & (values <= 1 + rowsum_atol).all(dim=1)
          & ((values.double().sum(dim=1) - 1.0).abs() <= rowsum_atol))
    if not bool(ok.all()):
        raise RecordsError(f"omega breaks the gate contract (finite, 0 <= omega <= 1 + {rowsum_atol}, |row sum - 1| <= "
                           f"{rowsum_atol}): " + _bad_rows_message(values, ~ok, sample_ids))

def compute_representations(samples: Sequence[Sample], encoder: "SASRecEncoder",
                            gate_fn: Optional[Callable[[Tensor, list[str]], Tensor]], *, num_experts: int,
                            batch_size: int, num_threads: int, rowsum_atol: float) -> RepresentationTable:
    """z (and omega) of one split without the LLM (§16.6.5 step 5; M15 calls it with ``gate_fn=None``).

    ``samples`` are M1 samples of one split in row order. On the CPU with ``num_threads`` torch threads, no
    gradient and autocast off (D12, D23): consecutive batches of ``batch_size`` (the last one shorter), M1
    ``pad_histories`` (``hist_ids [B, 10]``, ``len_seq [B]``), ``z = encoder(hist_ids, len_seq)``, ``validate_z``,
    ``omega = gate_fn(z, sample_ids)`` (the M7 ``Router`` of all three modes), ``validate_gate``, then the
    ``.detach().float().cpu()`` copies are joined and ``argmax`` computed. d is ``encoder.hidden_size``.
    """
    samples = list(samples)
    if not samples or not all(isinstance(s, Sample) for s in samples):
        raise RecordsError("compute_representations needs a non-empty sequence of M1 Sample objects")
    split = samples[0].split
    if any(s.split != split for s in samples):
        raise RecordsError(f"compute_representations takes one split, got {sorted({s.split for s in samples})}")
    rows = [s.row for s in samples]
    if any(b <= a for a, b in zip(rows, rows[1:])):
        raise RecordsError(f"{split} samples must be in M1 row order (strictly ascending rows)")
    k = _positive("num_experts", num_experts)
    size = _positive("batch_size", batch_size)
    threads = _positive("num_threads", num_threads)
    params = list(encoder.parameters())
    if not params or any(p.device.type != "cpu" for p in params):
        raise RecordsError("compute_representations runs on the CPU (16.6.5 step 3); move the encoder to cpu")
    if encoder.training:
        raise RecordsError("the SASRec encoder must be in eval mode (D12)")
    d = int(encoder.hidden_size)
    previous = torch.get_num_threads()
    z_parts: list[Tensor] = []
    omega_parts: list[Tensor] = []
    seen: list[str] = []
    try:
        torch.set_num_threads(threads)
        with torch.no_grad(), strict_fp32(torch.device("cpu")):
            for start in range(0, len(samples), size):
                part = samples[start:start + size]
                ids = [s.sample_id for s in part]
                hist_ids, len_seq = pad_histories([s.history for s in part], PAD_ID, HISTORY_LEN)
                z = encoder(hist_ids, len_seq)
                validate_z(z, batch=len(part), dim=d, sample_ids=ids)
                z_parts.append(z.detach().float().cpu())
                if gate_fn is not None:
                    omega = gate_fn(z, ids)
                    validate_gate(omega, batch=len(part), num_experts=k, rowsum_atol=rowsum_atol, sample_ids=ids)
                    omega_parts.append(omega.detach().float().cpu())
                seen.extend(ids)
    finally:
        torch.set_num_threads(previous)
    expected = [s.sample_id for s in samples]
    if seen != expected:
        raise RecordsError(f"{split}: computed sample order differs from the M1 order")
    z_all = np.ascontiguousarray(torch.cat(z_parts).numpy(), dtype=np.float32)
    omega_all = argmax = None
    if gate_fn is not None:
        omega_all = np.ascontiguousarray(torch.cat(omega_parts).numpy(), dtype=np.float32)
        argmax = np.argmax(omega_all, axis=1).astype(np.int64)
    meta = {"source": "compute_representations", "split": split, "n": len(samples), "d": d,
            "num_experts": k if gate_fn is not None else None, "has_omega": gate_fn is not None,
            "batch_size": size, "num_threads": threads, "device": "cpu",
            "sasrec_state_digest": getattr(encoder, "state_digest", None),
            "torch_version": torch.__version__, "utc": run.utc_now().isoformat()}
    return RepresentationTable(split=split, sample_ids=tuple(expected), rows=np.array(rows, dtype=np.int64),
                               user_ids=np.array([s.user_id for s in samples], dtype=np.int64), z=z_all,
                               omega=omega_all, argmax=argmax, meta=meta)


def save_npz(path: Path, table: RepresentationTable) -> str:
    """Write ``table`` in the NPZ format of §16.2 and return the file sha256.

    Members ``sample_id``/``split`` ``<U`` [N], ``row``/``user_id`` int64 [N], ``z`` float32 [N, d],
    ``omega`` float32 [N, K] and ``argmax_expert`` int64 [N] (only with omega), ``meta_json`` 0-dim string.
    Readable with ``np.load(allow_pickle=False)``; zip members carry a fixed time, so the same table gives
    the same bytes. Temporary name + ``os.replace``; an existing file is never overwritten.
    """
    path = Path(path)
    if not isinstance(table, RepresentationTable):
        raise RecordsError(f"save_npz takes a RepresentationTable, got {type(table).__name__}")
    if path.exists():
        raise RecordsError(f"{path} exists; NPZ files are never overwritten")
    meta = _jsonable(table.meta)
    hits = run.find_secrets(meta)
    if hits:
        raise run.SecretLeakError(f"refusing to write token-like values at {hits} to {path}")
    n = len(table)
    members: list[tuple[str, np.ndarray]] = [
        ("sample_id", np.array(table.sample_ids, dtype=np.str_)),
        ("split", np.array([table.split] * n, dtype=np.str_)),
        ("row", np.ascontiguousarray(table.rows, dtype=np.int64)),
        ("user_id", np.ascontiguousarray(table.user_ids, dtype=np.int64)),
        ("z", np.ascontiguousarray(table.z, dtype=np.float32)),
    ]
    if table.omega is not None:
        members += [("omega", np.ascontiguousarray(table.omega, dtype=np.float32)),
                    ("argmax_expert", np.ascontiguousarray(table.argmax, dtype=np.int64))]
    members.append(("meta_json", np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True, allow_nan=False),
                                          dtype=np.str_)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, array in members:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3                       # same bytes on Windows and Linux
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, array, allow_pickle=False)
    os.replace(tmp, path)
    return _sha256(path)


def load_npz(path: Path) -> RepresentationTable:
    """Read an NPZ of §16.2 with ``allow_pickle=False``; members, dtypes and shapes are checked strictly."""
    path = Path(path)
    if not path.is_file():
        raise RecordsError(f"{path} not found")
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        required = set(NPZ_KEYS) - set(_NPZ_GATE_KEYS)
        if keys - set(NPZ_KEYS) or required - keys or len(keys & set(_NPZ_GATE_KEYS)) == 1:
            raise RecordsError(f"{path}: members {sorted(keys)} do not match the NPZ format {list(NPZ_KEYS)} "
                               "(omega and argmax_expert together or not at all)")
        data = {key: npz[key] for key in keys}
    for key in ("sample_id", "split"):
        if data[key].dtype.kind != "U" or data[key].ndim != 1:
            raise RecordsError(f"{path}: {key} must be a 1-dim unicode array, got {data[key].dtype} {data[key].shape}")
    meta_text = data["meta_json"]
    if meta_text.dtype.kind != "U" or meta_text.ndim != 0:
        raise RecordsError(f"{path}: meta_json must be a 0-dim unicode array")
    splits = set(data["split"].tolist())
    if len(splits) != 1:
        raise RecordsError(f"{path}: split values {sorted(splits)} (one split per file)")
    try:
        return RepresentationTable(split=splits.pop(), sample_ids=tuple(data["sample_id"].tolist()),
                                   rows=data["row"], user_ids=data["user_id"], z=data["z"],
                                   omega=data.get("omega"), argmax=data.get("argmax_expert"),
                                   meta=json.loads(str(meta_text[()])))
    except RecordsError as exc:
        raise RecordsError(f"{path}: {exc}") from exc


def write_gate_csv(path: Path, table: RepresentationTable) -> str:
    """``gates_{validation,test}.csv`` [SPEC §4]: ``sample_id, user_id, split, gate_0..gate_{K-1}, argmax_expert,
    sasrec_0..sasrec_{d-1}``; floats ``'%.9g'`` (float32 round trip exact). Returns the sha256."""
    path = Path(path)
    if table.omega is None or table.argmax is None:
        raise RecordsError(f"{path}: the gate CSV needs omega (the table has z only)")
    if path.exists():
        raise RecordsError(f"{path} exists; gate CSV files are never overwritten")
    k, d = int(table.omega.shape[1]), table.dim
    header = ["sample_id", "user_id", "split", *(f"gate_{i}" for i in range(k)), "argmax_expert",
              *(f"sasrec_{j}" for j in range(d))]
    lines: list[list[str]] = [header]
    for i, sid in enumerate(table.sample_ids):
        lines.append([sid, str(int(table.user_ids[i])), table.split,
                      *(CSV_FLOAT_FORMAT % float(v) for v in table.omega[i].tolist()), str(int(table.argmax[i])),
                      *(CSV_FLOAT_FORMAT % float(v) for v in table.z[i].tolist())])
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_file(path, _csv_bytes(lines))
    return _sha256(path)


def read_gate_csv(path: Path) -> RepresentationTable:
    """Read ``write_gate_csv`` output back (rows from ``sample_id``, float32 values bit for bit, ``meta`` empty)."""
    path = Path(path)
    header, raw = _read_csv(path)
    k = sum(1 for name in header if name.startswith("gate_"))
    d = sum(1 for name in header if name.startswith("sasrec_"))
    expected = ("sample_id", "user_id", "split", *(f"gate_{i}" for i in range(k)), "argmax_expert",
                *(f"sasrec_{j}" for j in range(d)))
    if k < 1 or d < 1 or header != expected:
        raise RecordsError(f"{path}: header {list(header)[:8]}... is not the gate CSV layout [SPEC §4]")
    if not raw:
        raise RecordsError(f"{path}: no rows")
    splits = {r[2] if len(r) > 2 else "" for r in raw}
    if len(splits) != 1 or any(len(r) != len(header) for r in raw):
        raise RecordsError(f"{path}: one split per file and {len(header)} cells per row")
    try:
        omega = np.array([[float(x) for x in r[3:3 + k]] for r in raw], dtype=np.float64).astype(np.float32)
        z = np.array([[float(x) for x in r[4 + k:]] for r in raw], dtype=np.float64).astype(np.float32)
        argmax = np.array([int(r[3 + k]) for r in raw], dtype=np.int64)
        rows = np.array([int(r[0].rsplit(":", 1)[1]) for r in raw], dtype=np.int64)
        users = np.array([int(r[1]) for r in raw], dtype=np.int64)
    except (ValueError, IndexError) as exc:
        raise RecordsError(f"{path}: cannot parse the gate CSV ({exc})") from exc
    try:
        return RepresentationTable(split=splits.pop(), sample_ids=tuple(r[0] for r in raw), rows=rows, user_ids=users,
                                   z=np.ascontiguousarray(z), omega=np.ascontiguousarray(omega), argmax=argmax, meta={})
    except RecordsError as exc:
        raise RecordsError(f"{path}: {exc}") from exc

# ----------------------------------------------------------------------------
# Evaluation records (§16.6.4; files of M11 §15.6.8, §15.6.9)
# ----------------------------------------------------------------------------
EVAL_RECORD_FIELDS: Final = ("sample_id", "split", "user_id", "template_id", "hybrid", "cand_ids", "cand_titles",
                             "target_id", "target_title", "raw_text", "first_line", "matched", "valid", "correct",
                             "gates", "flags", "gen_token_ids")         # 3.1 EvalRecord, M11 §15.6.1 order
RECORDS_FILE: Final = "records.jsonl"
METRICS_FILE: Final = "metrics.json"
REPRESENTATIONS_FILE: Final = "representations.npz"
EVAL_SPLITS: Final = ("val", "test")
METRICS_CORE: Final = ("split", "epoch", "n_total", "n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")
METRICS_BLOCKS: Final = ("ckpt_tag", "aux_protocols", "supplementary")   # M11 additions (§15.6.9)
SPLIT_METRICS_FIELDS: Final = ("split", "n_total", "n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")
# Auxiliary drop_last protocols of the test split: [코드] U main.py:82 (4), U train_movielens.sh:5 (8),
# U data/data_interface.py:158 (drop_last) = M11 AUX_DROP_LAST_BATCH_SIZES (§15.6.6).
AUX_BATCH_SIZES: Final = (4, 8)
METRIC_ATOL: Final = 1e-12                       # §16.6.4 절대 오차
_FIELD_KIND: Final = {"sample_id": "str", "split": "str", "user_id": "int", "template_id": "int", "hybrid": "bool",
                      "cand_ids": "ints", "cand_titles": "strs", "target_id": "int", "target_title": "str",
                      "raw_text": "str", "first_line": "str", "matched": "ints", "valid": "bool", "correct": "bool",
                      "gen_token_ids": "ints"}


def _plain_field(name: str, value: object, where: str) -> object:
    kind = _FIELD_KIND[name]
    if kind in ("ints", "strs"):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise RecordsError(f"{where}: {name} must be a sequence, got {type(value).__name__}")
        check = _is_int if kind == "ints" else (lambda x: isinstance(x, str))
        if not all(check(x) for x in value):
            raise RecordsError(f"{where}: {name} must hold {'int' if kind == 'ints' else 'str'} values")
        return [int(x) if kind == "ints" else x for x in value]
    ok = {"str": isinstance(value, str), "int": _is_int(value), "bool": isinstance(value, (bool, np.bool_))}[kind]
    if not ok:
        raise RecordsError(f"{where}: {name} must be {kind}, got {type(value).__name__} {value!r}")
    return int(value) if kind == "int" else bool(value) if kind == "bool" else value


def _record_pairs(record: object, num_experts: int, where: str) -> list[tuple[str, object]]:
    """``records.jsonl`` key/value pairs of one ``EvalRecord`` in field order, ``gates`` -> ``gate_0..gate_{K-1}``."""
    if not dataclasses.is_dataclass(record) or isinstance(record, type):
        raise RecordsError(f"{where}: expected an EvalRecord dataclass, got {type(record).__name__}")
    names = tuple(f.name for f in dataclasses.fields(record))
    if names != EVAL_RECORD_FIELDS:
        raise RecordsError(f"{where}: EvalRecord fields {names} differ from {EVAL_RECORD_FIELDS} (3.1, 15.6.1)")
    pairs: list[tuple[str, object]] = []
    for name in EVAL_RECORD_FIELDS:
        value = getattr(record, name)
        if name == "gates":
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != num_experts:
                raise RecordsError(f"{where}: gates must be {num_experts} numbers (K), got {value!r}")
            if not all(_is_number(g) and math.isfinite(float(g)) for g in value):
                raise RecordsError(f"{where}: gates must be finite numbers, got {list(value)}")
            pairs += [(f"gate_{k}", float(g)) for k, g in enumerate(value)]
        elif name == "flags":
            if not isinstance(value, Mapping) or tuple(value) != FLAG_NAMES or \
                    not all(isinstance(v, (bool, np.bool_)) for v in value.values()):
                raise RecordsError(f"{where}: flags must map FLAG_NAMES {FLAG_NAMES} (in order) to bool, got {value!r}")
            pairs.append((name, {k: bool(v) for k, v in value.items()}))
        else:
            pairs.append((name, _plain_field(name, value, where)))
    return pairs


def _check_ratios(block: Mapping[str, object], n: int, v: int, c: int, where: str) -> None:
    """D17: valid_ratio = v/n, hr_at_1 = c/v (0 when v = 0, U model/model_interface.py:487-491), combined = c/n."""
    expected = {"valid_ratio": v / n, "hr_at_1": c / v if v else 0.0, "combined_metric": c / n}
    for key, value in expected.items():
        got = block.get(key)
        if not _is_number(got) or not abs(float(got) - value) <= METRIC_ATOL:
            raise RecordsError(f"{where}: {key} = {got!r}, expected {value!r} from n={n}, valid={v}, correct={c}")


def _check_counts(block: Mapping[str, object], records: Sequence[Mapping[str, object]], where: str) -> None:
    n = len(records)
    v = sum(1 for r in records if r["valid"])
    c = sum(1 for r in records if r["correct"])
    for key, value in (("n_total", n), ("n_valid", v), ("n_correct", c)):
        got = block.get(key)
        if not _is_int(got) or int(got) != value:
            raise RecordsError(f"{where}: {key} = {got!r}, but the records count {value}")
    _check_ratios(block, n, v, c, where)


def _check_metrics(metrics: Mapping[str, object], plain: Sequence[Mapping[str, object]]) -> dict:
    """Checks of §16.6.4 on ``metrics.json``; returns it in the fixed key order (core, M11 blocks, meta)."""
    if not isinstance(metrics, Mapping):
        raise RecordsError(f"metrics must be a mapping, got {type(metrics).__name__}")
    metrics = _jsonable(metrics)            # SplitMetrics dataclasses inside become their field mappings
    missing = [k for k in (*METRICS_CORE, "meta") if k not in metrics]
    unknown = sorted(str(k) for k in metrics if k not in (*METRICS_CORE, *METRICS_BLOCKS, "meta"))
    if missing or unknown:
        raise RecordsError(f"metrics.json keys: missing {missing}, unknown {unknown} (16.6.4, 15.6.9)")
    split = metrics["split"]
    if split not in EVAL_SPLITS:
        raise RecordsError(f"metrics split must be one of {EVAL_SPLITS}, got {split!r}")
    epoch = metrics["epoch"]
    if epoch is not None and (not _is_int(epoch) or int(epoch) < 0):
        raise RecordsError(f"metrics epoch must be an int >= 0 or null (init), got {epoch!r}")
    if "ckpt_tag" in metrics:
        tag = metrics["ckpt_tag"]
        if not isinstance(tag, str) or not (tag == "init" or (tag.startswith("epoch") and tag[5:].isdigit()
                                                              and str(int(tag[5:])) == tag[5:])):
            raise RecordsError(f"ckpt_tag must be epoch<e> or init, got {tag!r}")
        if epoch != (None if tag == "init" else int(tag[5:])):
            raise RecordsError(f"epoch {epoch!r} does not match ckpt_tag {tag!r} (init -> null)")
    _check_counts(metrics, plain, "metrics.json")
    if not isinstance(metrics["meta"], Mapping):
        raise RecordsError("metrics meta must be a mapping")
    if "supplementary" in metrics and not isinstance(metrics["supplementary"], Mapping):
        raise RecordsError("metrics supplementary must be a mapping")
    if "aux_protocols" in metrics:
        aux = metrics["aux_protocols"]
        names = tuple(f"drop_last_bs{b}" for b in AUX_BATCH_SIZES)
        if not isinstance(aux, Mapping) or (split == "val" and len(aux)) or \
                (split == "test" and sorted(aux) != sorted(names)):
            raise RecordsError(f"aux_protocols must be {{}} for val and have {list(names)} for test, got {aux!r}")
        for b, name in zip(AUX_BATCH_SIZES, names):
            if split != "test":
                break
            block, n_b = aux[name], (len(plain) // b) * b
            if not isinstance(block, Mapping) or tuple(sorted(block)) != tuple(sorted(SPLIT_METRICS_FIELDS)):
                raise RecordsError(f"aux_protocols.{name} must have the SplitMetrics fields {SPLIT_METRICS_FIELDS}")
            if block["split"] != split or n_b < 1:
                raise RecordsError(f"aux_protocols.{name}: split {block['split']!r}, {n_b} records (needs >= {b})")
            _check_counts(block, plain[:n_b], f"aux_protocols.{name} (first {n_b} records)")
    ordered = {k: metrics[k] for k in METRICS_CORE}
    ordered.update({k: metrics[k] for k in METRICS_BLOCKS if k in metrics})
    ordered["meta"] = metrics["meta"]
    return ordered

def write_eval(out_dir: Path, records: Sequence["EvalRecord"], metrics: Mapping[str, object], *,
               expected_ids: Sequence[str], num_experts: int, rowsum_atol: float = ROW_SUM_ATOL) -> None:
    """``records.jsonl`` then ``metrics.json`` of one (checkpoint, split) evaluation (§16.6.4).

    Nothing is written unless every check passes: sample ids equal ``expected_ids`` (same order, no
    duplicates, full split: val 94 / test 95, smoke its size), K gate values per record, finite, in
    [0, 1 + atol] with |row sum - 1| <= atol (``rowsum_atol``, default the M7 value 1e-5 =
    ``records.gate_rowsum_atol``), ``correct => valid``, ``valid == (len(matched) == 1)``, matched among the
    candidates, ``metrics`` counts equal the records and the three ratios equal the D17 formulas within 1e-12,
    each ``aux_protocols`` block recounted from the first floor(N/b)*b records (test; val has ``{}``).
    ``epoch: null`` (``init``) and the M11 blocks pass. Lines are ``json.dumps(obj, ensure_ascii=False,
    separators=(",", ":"))`` + LF in field order, so the same records give the same sha256. Each file goes
    to a temporary name first; ``metrics.json`` last marks completion (§15.6.8).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise RecordsError(f"{out_dir} is not a folder (M11 makes eval/<ckpt_tag>/<split>/ first)")
    for name in (RECORDS_FILE, METRICS_FILE):
        if (out_dir / name).exists():
            raise RecordsError(f"{out_dir / name} exists; evaluation records are never overwritten (15.6.8)")
    k = _positive("num_experts", num_experts)
    if not _is_number(rowsum_atol) or not 0 < float(rowsum_atol) < 1:
        raise RecordsError(f"rowsum_atol must be in (0, 1), got {rowsum_atol!r}")
    records = list(records)
    expected = list(expected_ids)
    if not records:
        raise RecordsError("write_eval needs at least one record")
    plain = [dict(_record_pairs(rec, k, f"record {i}")) for i, rec in enumerate(records)]
    ids = [p["sample_id"] for p in plain]
    duplicates = sorted({s for s in ids if ids.count(s) > 1}) if len(set(ids)) != len(ids) else []
    if duplicates:
        raise RecordsError(f"duplicate sample ids in the records: {duplicates[:5]}")
    if len(set(expected)) != len(expected):
        raise RecordsError("expected_ids has duplicates")
    missing, extra = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
    if missing or extra:
        raise RecordsError(f"records cover {len(ids)} samples, expected {len(expected)}: missing {missing[:5]}, "
                           f"extra {extra[:5]} (D17: the whole split, no drop_last)")
    if ids != expected:
        raise RecordsError("records are not in the order of expected_ids (M1 row order)")
    split = metrics.get("split") if isinstance(metrics, Mapping) else None
    for p in plain:
        sid = p["sample_id"]
        gates = [p[f"gate_{j}"] for j in range(k)]
        if p["split"] != split:
            raise RecordsError(f"{sid}: record split {p['split']!r} != metrics split {split!r}")
        if min(gates) < 0 or max(gates) > 1 + rowsum_atol or abs(math.fsum(gates) - 1.0) > rowsum_atol:
            raise RecordsError(f"{sid}: gates {gates} break 0 <= g <= 1 + {rowsum_atol}, |sum - 1| <= {rowsum_atol}")
        if p["correct"] and not p["valid"]:
            raise RecordsError(f"{sid}: correct but not valid")
        if p["valid"] != (len(p["matched"]) == 1):
            raise RecordsError(f"{sid}: valid {p['valid']} but {len(p['matched'])} matched candidates (15.6.5)")
        if len(p["cand_titles"]) != len(p["cand_ids"]) or not set(p["matched"]) <= set(p["cand_ids"]):
            raise RecordsError(f"{sid}: matched {p['matched']} / candidate titles do not fit the candidates")
    ordered = _check_metrics(metrics, plain)
    data = "".join(json.dumps(p, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for p in plain)
    _replace_file(out_dir / RECORDS_FILE, data.encode("utf-8"))
    _write_json(out_dir / METRICS_FILE, ordered)


def read_eval_records(path: Path) -> list["EvalRecord"]:
    """``EvalRecord`` list of a ``records.jsonl`` (field order and types checked; K from the ``gate_*`` keys)."""
    from .evaluate import EvalRecord   # M11 type (§15.6.1); evaluate.py imports records inside its functions

    path = Path(path)
    if not path.is_file():
        raise RecordsError(f"{path} not found")
    text = path.read_bytes().decode("utf-8")
    if not text.endswith("\n"):
        raise RecordsError(f"{path}: the last line has no LF (incomplete file)")
    out: list = []
    num_experts: Optional[int] = None
    for i, line in enumerate(text[:-1].split("\n"), start=1):
        where = f"{path}:{i}"
        try:
            pairs = json.loads(line, object_pairs_hook=list)
        except json.JSONDecodeError as exc:
            raise RecordsError(f"{where}: not a JSON object ({exc})") from exc
        if not isinstance(pairs, list):
            raise RecordsError(f"{where}: not a JSON object")
        keys = [key for key, _ in pairs]
        k = sum(1 for key in keys if key.startswith("gate_"))
        layout = [*EVAL_RECORD_FIELDS[:14], *(f"gate_{j}" for j in range(k)), "flags", "gen_token_ids"]
        if keys != layout or k < 1 or (num_experts is not None and k != num_experts):
            raise RecordsError(f"{where}: keys {keys} are not the EvalRecord layout (K = {num_experts or k})")
        num_experts = k
        values = dict(pairs)
        flags = values["flags"]
        if not isinstance(flags, list):
            raise RecordsError(f"{where}: flags must be an object")
        fields = {name: values[name] for name in EVAL_RECORD_FIELDS if name not in ("gates", "flags")}
        for name in ("cand_ids", "cand_titles", "matched", "gen_token_ids"):
            if isinstance(fields[name], list):
                fields[name] = tuple(fields[name])
        record = EvalRecord(**fields, gates=tuple(values[f"gate_{j}"] for j in range(k)), flags=dict(flags))
        _record_pairs(record, k, where)          # the same type rules as write_eval
        out.append(record)
    return out


def crosscheck_gates(table: RepresentationTable, eval_table: RepresentationTable,
                     records: Sequence["EvalRecord"], *, atol: float) -> dict:
    """Export omega against the omega used in generation (§16.6.5 step 7). Raises ``RecordsError`` on a failure.

    ``eval_table`` is M11's ``representations.npz`` of the same checkpoint and split, ``records`` its
    ``records.jsonl``. Checked: the three sample sets are equal (matched by sample_id, not position), every
    ``gate_*`` of the records equals the evaluation NPZ omega bit for bit, and max |export omega - eval omega|
    <= ``atol`` (``records.gate_crosscheck_atol``). argmax disagreements and max |delta z| (GPU evaluation z
    against CPU export z, M11 15.6.14 제안 7) are recorded without a judgement. A tolerance failure is a
    ``GateCrosscheckError`` whose message and ``result`` (``passed`` false) give max |d omega| with the worst
    sample, the argmax mismatch count and max |d z|, so a z rounding alarm can be told from a real gate mismatch.
    """
    where = f"crosscheck of {table.split}"
    if table.omega is None or eval_table.omega is None or table.argmax is None or eval_table.argmax is None:
        raise RecordsError(f"{where}: both tables need omega")
    if table.split != eval_table.split:
        raise RecordsError(f"{where}: evaluation table split {eval_table.split!r}")
    k = int(table.omega.shape[1])
    if int(eval_table.omega.shape[1]) != k or eval_table.dim != table.dim:
        raise RecordsError(f"{where}: K or d differ ({k}, {table.dim}) vs ({eval_table.omega.shape[1]}, {eval_table.dim})")
    records = list(records)
    rec_ids = [r.sample_id for r in records]
    exported, evaluated = set(table.sample_ids), set(eval_table.sample_ids)
    if len(set(rec_ids)) != len(rec_ids) or exported != evaluated or set(rec_ids) != evaluated:
        raise RecordsError(f"{where}: sample sets differ: export-only {sorted(exported - evaluated)[:5]}, eval-only "
                           f"{sorted(evaluated - exported)[:5]}, records missing {sorted(evaluated - set(rec_ids))[:5]}, "
                           f"records extra {sorted(set(rec_ids) - evaluated)[:5]}, {len(rec_ids)} records")
    if any(r.split != table.split for r in records):
        raise RecordsError(f"{where}: a record has another split")
    index = {sid: i for i, sid in enumerate(eval_table.sample_ids)}
    gates = np.array([[float(g) for g in r.gates] for r in records], dtype=np.float64)
    if gates.shape != (len(records), k):
        raise RecordsError(f"{where}: records have gate shape {gates.shape}, expected ({len(records)}, {k})")
    gates32 = gates.astype(np.float32)
    npz_rows = eval_table.omega[[index[s] for s in rec_ids]]
    differ = np.nonzero((gates32.astype(np.float64) != gates).any(axis=1)
                        | (gates32.view(np.uint32) != npz_rows.view(np.uint32)).any(axis=1))[0]
    if differ.size:
        first = int(differ[0])
        raise RecordsError(f"{where}: records.jsonl gates differ from the evaluation NPZ omega in {differ.size} samples, "
                           f"first {rec_ids[first]}: {gates[first].tolist()} vs {npz_rows[first].tolist()}")
    order = [index[s] for s in table.sample_ids]
    delta = np.abs(table.omega.astype(np.float64) - eval_table.omega[order].astype(np.float64))
    worst = int(np.argmax(delta.max(axis=1)))
    max_omega = float(delta.max())
    mismatch = np.nonzero(table.argmax != eval_table.argmax[order])[0]
    max_z = float(np.abs(table.z.astype(np.float64) - eval_table.z[order].astype(np.float64)).max())
    result = {"split": table.split, "n": len(table), "num_experts": k, "atol": atol,
              "max_abs_diff_omega": max_omega, "max_abs_diff_omega_sample_id": table.sample_ids[worst],
              "argmax_mismatches": int(mismatch.size),
              "argmax_mismatch_sample_ids": [table.sample_ids[i] for i in mismatch[:20].tolist()],
              "max_abs_diff_z": max_z, "records_equal_eval_npz_bitwise": True, "passed": bool(max_omega <= atol)}
    if not result["passed"]:
        raise GateCrosscheckError(
            f"{where}: max |export omega - eval omega| = {max_omega!r} > {atol} (worst sample "
            f"{table.sample_ids[worst]}); max |export z - eval z| = {max_z!r}, argmax mismatches {int(mismatch.size)} "
            f"of {len(table)}. |d z| near the M3 CUDA z limit 1e-4 means the evaluation z differed (rounding, TF32); "
            f"a far smaller |d z| means a different gate (DESIGN 16.7)", result)
    return result

# ----------------------------------------------------------------------------
# Export command (§16.6.5) [결정]
# ----------------------------------------------------------------------------
EXPORT_SAME_SECTIONS: Final = ("data", "sasrec", "router", "lora", "cluster")    # §16.6.5 step 1
EXPORT_DIR: Final = ("records", "export")                                     # records/export/epoch{e}/
ANALYSIS_DIR: Final = ("records", "analysis")                                 # records/analysis/epoch{e}/{split}/
CROSSCHECK_FILE: Final = "crosscheck.json"
MANIFEST_FILE: Final = "manifest.json"
_SELECTION_KEYS: Final = {"best_val": "best_val_epoch", "last": "last_epoch"}   # M10 eval/selection.json (§14.6.7)


def config_differences(source: "Config", current: "Config",
                       sections: Sequence[str] = EXPORT_SAME_SECTIONS) -> list[str]:
    """``section.key: a != b`` for every key of ``sections`` whose value differs between two configs."""
    from .config import config_to_dict

    a, b = config_to_dict(source), config_to_dict(current)
    out = []
    for section in sections:
        for key in sorted(set(a[section]) | set(b[section])):
            if a[section].get(key) != b[section].get(key):
                out.append(f"{section}.{key}: {a[section].get(key)!r} != {b[section].get(key)!r}")
    return out


def export_epochs(source_run: Path, checkpoints: Sequence[str]) -> dict[int, tuple[str, ...]]:
    """epoch -> export tags from the source run's ``eval/selection.json`` (D11), ascending, each epoch once."""
    path = Path(source_run) / "eval" / "selection.json"
    if not path.is_file():
        raise RecordsError(f"{path} not found: export needs a training run whose selection is done (M10 14.6.7)")
    selection = json.loads(path.read_text(encoding="utf-8"))
    tags: dict[int, list[str]] = {}
    for tag in checkpoints:
        if tag not in _SELECTION_KEYS:
            raise RecordsError(f"records.export_checkpoints has {tag!r}; allowed {sorted(_SELECTION_KEYS)}")
        epoch = selection.get(_SELECTION_KEYS[tag])
        if not _is_int(epoch) or int(epoch) < 0:
            raise RecordsError(f"{path}: {_SELECTION_KEYS[tag]} = {epoch!r} is not an epoch number")
        tags.setdefault(int(epoch), []).append(tag)
    return {epoch: tuple(tags[epoch]) for epoch in sorted(tags)}


def export_router(cfg: "Config", state: Mapping[str, Tensor], *, z_dim: int, hard_gate: Any = None) -> nn.Module:
    """The M7 ``Router`` of the export (§16.6.5 step 4), CPU fp32, eval.

    Only the ``omega_source.`` keys of the checkpoint (M10 ``read_trainable_state``) are used, loaded with
    ``strict=True``: ``gate.proj.weight`` for dynamic, none for cluster_hard (``hard_gate`` = M15
    ``ClusterHardGate``) and none (omega = 1, K = 1).
    """
    from .model import OMEGA_SOURCE_PREFIX    # checkpoint key prefix of the gate (M9 §13.6.8)
    from .router import Router

    router = Router(cfg.router, num_experts=cfg.lora.num_experts, z_dim=z_dim, main_seed=cfg.run.seed,
                    hard_gate=hard_gate)
    gate_state = {key[len(OMEGA_SOURCE_PREFIX):]: value for key, value in state.items()
                  if key.startswith(OMEGA_SOURCE_PREFIX)}
    try:
        router.load_state_dict(gate_state, strict=True)
    except RuntimeError as exc:
        raise RecordsError(f"checkpoint keys {sorted(gate_state)} do not fit the {cfg.router.routing_mode} Router: "
                           f"{exc}") from exc
    return router.eval()


def _cluster_hard_gate(cfg: "Config", data: Any, encoder: "SASRecEncoder", repo_root: Path) -> tuple[Any, dict]:
    """cluster_hard: M15 artifacts -> mapping -> ``ClusterHardGate`` -> live z check (§16.6.5 step 4, 19.6.7)."""
    from .cluster import ClusterHardGate, load_cluster_artifacts, parse_expert_mapping, verify_live_encoder

    expected_ids = [s.sample_id for split in SPLITS for s in data.samples(split)]
    state_sha256 = run.content_sha256(run.resolve_path(cfg.sasrec.state_path, repo_root))
    artifacts = load_cluster_artifacts(run.resolve_path(cfg.cluster.artifact_dir, repo_root), expected_ids=expected_ids,
                                       sasrec_sha256=state_sha256, data_sha256=run.input_hashes(repo_root),
                                       ccfg=cfg.cluster)
    mapping = parse_expert_mapping(cfg.cluster.expert_mapping, n_clusters=artifacts.n_clusters,
                                   num_experts=cfg.lora.num_experts)
    gate = ClusterHardGate(artifacts, mapping, cfg.lora.num_experts)
    live = verify_live_encoder(artifacts, encoder, data.train, device=torch.device("cpu"),
                               n=cfg.cluster.live_check_n, atol=cfg.cluster.live_check_atol)
    return gate, dict(live) if isinstance(live, Mapping) else {"result": live}


def run_export(cfg: "Config", run_dir: Path, *, repo_root: Path) -> None:
    """``main.py export --set records.source_run=<training run>`` (§16.6.5), inside the M0 export run folder.

    1. the source ``config.yaml`` must equal this config in ``data``, ``sasrec``, ``router``, ``lora`` and
       ``cluster`` (the differing keys are listed); 2. ``records.export_checkpoints`` -> epochs through the
       source ``eval/selection.json``; 3. M1 samples and the frozen CPU SASRec; 4. per epoch the M7 ``Router``
       from M10 ``read_trainable_state`` (cluster_hard: M15 gate first); 5.-6. ``compute_representations`` per
       split, ``representations_{train,validation,test}.npz`` and ``gates_{validation,test}.csv`` in
       ``records/export/epoch{e}/``; 7. ``crosscheck.json`` against the source ``eval/epoch{e}/{val,test}/``
       with ``passed`` per split, written before a failed cross-check stops the export (``RecordsError``);
       8. for K >= 2 ``analysis.run_gate_analysis`` per ``records.analysis_splits`` into
       ``records/analysis/epoch{e}/{split}/``, then ``manifest.json`` (sha256 of every file, checkpoint,
       SASRec weights, source run).
    """
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    rc = cfg.records
    if rc.source_run is None:
        raise RecordsError("export needs --set records.source_run=<training run folder> (16.6.9)")
    source = run.resolve_path(rc.source_run, repo_root)
    if not (source / "config.yaml").is_file():
        raise RecordsError(f"{source}: no config.yaml, not a training run folder")
    from .config import load_config

    source_cfg, _ = load_config(source / "config.yaml")
    differences = config_differences(source_cfg, cfg)
    if differences:
        raise RecordsError("the export config differs from the training run in: " + "; ".join(differences))
    epochs = export_epochs(source, rc.export_checkpoints)
    from .data import load_dataset
    from .sasrec import SASRecEncoder

    data = load_dataset(cfg, repo_root)
    encoder = SASRecEncoder.from_config(cfg, data.catalog, HISTORY_LEN)       # CPU, eval, frozen (D12, D23)
    hard_gate, live = None, None
    if cfg.router.routing_mode == "cluster_hard":
        hard_gate, live = _cluster_hard_gate(cfg, data, encoder, repo_root)
    from .train import read_trainable_state       # M10 §14.6.6: the only checkpoint reader (markers, sha256)

    state_path = run.resolve_path(cfg.sasrec.state_path, repo_root)
    sasrec_info = {"state_path": str(state_path), "state_sha256": run.content_sha256(state_path),
                   "state_digest": encoder.state_digest}
    logger.info("export of %s: epochs %s, routing_mode %s, K %d", source, dict(epochs), cfg.router.routing_mode,
                cfg.lora.num_experts)
    for epoch, tags in epochs.items():
        ckpt_dir = source / "checkpoints" / f"epoch{epoch}"
        router = export_router(cfg, read_trainable_state(ckpt_dir), z_dim=int(encoder.hidden_size),
                               hard_gate=hard_gate)
        trainable = ckpt_dir / "trainable.pt"
        checkpoint = {"dir": str(ckpt_dir), "epoch": epoch, "tags": list(tags),
                      "trainable_sha256": run.content_sha256(trainable) if trainable.is_file() else None}
        _export_epoch(cfg, run_dir, source, epoch, data=data, encoder=encoder, router=router, checkpoint=checkpoint,
                      sasrec_info=sasrec_info, live=live)


def _export_epoch(cfg: "Config", run_dir: Path, source: Path, epoch: int, *, data: Any, encoder: "SASRecEncoder",
                  router: nn.Module, checkpoint: Mapping[str, object], sasrec_info: Mapping[str, object],
                  live: Optional[Mapping[str, object]]) -> None:
    """Steps 5-8 of §16.6.5 for one checkpoint epoch."""
    rc, k = cfg.records, cfg.lora.num_experts
    out = run_dir.joinpath(*EXPORT_DIR, f"epoch{epoch}")
    if out.exists():
        raise RecordsError(f"{out} exists; export folders are never overwritten")
    out.mkdir(parents=True)
    tables: dict[str, RepresentationTable] = {}
    files: dict[str, str] = {}
    for split in SPLITS:
        table = compute_representations(data.samples(split), encoder, router, num_experts=k,
                                         batch_size=rc.export_batch_size, num_threads=rc.export_num_threads,
                                         rowsum_atol=rc.gate_rowsum_atol)
        table = dataclasses.replace(table, meta={**table.meta, "source": "export", "routing_mode": cfg.router.routing_mode,
                                                 "checkpoint": dict(checkpoint), "sasrec": dict(sasrec_info),
                                                 "source_run": str(source)})
        name = SPEC_SPLIT_NAMES[split]
        files[f"representations_{name}.npz"] = save_npz(out / f"representations_{name}.npz", table)
        if split != "train":                       # train is NPZ only (66,126 x 72 as CSV) [SPEC §4]
            files[f"gates_{name}.csv"] = write_gate_csv(out / f"gates_{name}.csv", table)
        tables[split] = table
    cross: dict[str, dict] = {}
    failures: list[str] = []
    for split in EVAL_SPLITS:
        eval_dir = source / "eval" / f"epoch{epoch}" / split
        if not (eval_dir / METRICS_FILE).is_file():
            raise RecordsError(f"{eval_dir}: no completed evaluation (metrics.json) to cross-check (16.6.5 step 7)")
        try:
            result = crosscheck_gates(tables[split], load_npz(eval_dir / REPRESENTATIONS_FILE),
                                      read_eval_records(eval_dir / RECORDS_FILE), atol=rc.gate_crosscheck_atol)
        except GateCrosscheckError as exc:     # measured: kept in crosscheck.json, raised after both splits
            result = exc.result
            failures.append(str(exc))
        cross[SPEC_SPLIT_NAMES[split]] = {**result, "eval_dir": str(eval_dir),
                                          "eval_npz_sha256": _sha256(eval_dir / REPRESENTATIONS_FILE),
                                          "records_sha256": _sha256(eval_dir / RECORDS_FILE)}
        logger.info("export epoch%d %s: max |d omega| %.3g (passed %s), argmax mismatches %d, max |d z| %.3g", epoch,
                    split, result["max_abs_diff_omega"], result["passed"], result["argmax_mismatches"],
                    result["max_abs_diff_z"])
    _write_json(out / CROSSCHECK_FILE, {"epoch": epoch, "atol": rc.gate_crosscheck_atol, "splits": cross})
    files[CROSSCHECK_FILE] = _sha256(out / CROSSCHECK_FILE)
    if failures:                               # 16.6.5 step 7: crosscheck.json first, then the failure
        raise RecordsError(f"export epoch{epoch}: gate cross-check failed, {out / CROSSCHECK_FILE} written with "
                           f"passed false: " + "; ".join(failures))
    analysis_files: dict[str, str] = {}
    if k >= 2:
        from .analysis import run_gate_analysis

        for split in rc.analysis_splits:
            target = run_dir.joinpath(*ANALYSIS_DIR, f"epoch{epoch}", split)
            run_gate_analysis(tables[split], target, ckpt_tag=f"epoch{epoch}", formats=rc.figure_formats)
            analysis_files.update({p.relative_to(run_dir).as_posix(): _sha256(p)
                                   for p in sorted(target.rglob("*")) if p.is_file()})
    _write_json(out / MANIFEST_FILE, {
        "source_run": str(source), "epoch": epoch, "routing_mode": cfg.router.routing_mode, "num_experts": k,
        "z_dim": tables["train"].dim, "rows": {SPEC_SPLIT_NAMES[s]: len(tables[s]) for s in SPLITS},
        "checkpoint": dict(checkpoint), "sasrec": dict(sasrec_info), "cluster_live_check": live,
        "export": {"batch_size": rc.export_batch_size, "num_threads": rc.export_num_threads,
                   "torch_version": torch.__version__},
        "files": files, "analysis": list(rc.analysis_splits) if k >= 2 else "not run: gate analysis requires K >= 2",
        "analysis_files": analysis_files, "created_utc": run.utc_now().isoformat()})


__all__ = [
    "RecordsError", "GateCrosscheckError", "Column", "TRAIN_DIR", "MICRO_FILE", "STEP_FILE", "EPOCH_FILE",
    "MODULE_MAX_FILE", "COLUMNS_FILE",
    "SUMMARY_FILE", "MICRO_FIELDS", "EPOCH_FIELDS", "MODULE_MAX_FIELDS", "step_fields", "columns_document",
    "read_rows", "ActivationMaxMonitor", "TrainRecorder", "SPEC_SPLIT_NAMES", "NPZ_KEYS", "CSV_FLOAT_FORMAT",
    "RepresentationTable", "validate_z", "validate_gate", "compute_representations", "save_npz", "load_npz",
    "write_gate_csv", "read_gate_csv", "EVAL_RECORD_FIELDS", "RECORDS_FILE", "METRICS_FILE", "REPRESENTATIONS_FILE",
    "EVAL_SPLITS", "METRICS_CORE", "METRICS_BLOCKS", "SPLIT_METRICS_FIELDS", "AUX_BATCH_SIZES", "METRIC_ATOL",
    "write_eval", "read_eval_records", "crosscheck_gates", "EXPORT_SAME_SECTIONS", "EXPORT_DIR", "ANALYSIS_DIR",
    "CROSSCHECK_FILE", "MANIFEST_FILE", "config_differences", "export_epochs", "export_router", "run_export",
]
