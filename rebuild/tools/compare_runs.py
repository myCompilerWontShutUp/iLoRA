"""M14 실행 비교 도구 (DESIGN.md §18.6.8; 결과 표 §18.6.7, SPEC §20 색인 §18.6.9, 한계 §18.6.10, D11, D17).

    python -I rebuild/tools/compare_runs.py --run E1=<run> --run E2=<run> --run E3=<eval run> --run E4=<run> \
        [--export E2=<export run> ...] [--cluster <cluster run>] [--root runs/rebuild]

Makes ``<root>/compare_<UTC>/`` with M0 ``create_run_dir("compare")`` and keeps its ``meta.json`` status with M0
``start_meta``/``tracked_run`` (RUNBOOK R14 ``done_ok``). Tables are built only from the files the commands write:

- training run (``main.py train``): ``config.yaml``, ``meta.json``, ``eval/selection.json`` (M10 §14.6.7),
  ``eval/epoch{e}/test/metrics.json`` and ``records.jsonl`` (M11/M12 ``write_eval``), ``records/train/run_summary.json``
  (M12 ``TrainRecorder.close``, ``train_summary.trainable_params`` and ``train_summary.seconds.total``);
- eval run (``main.py eval``, E3): ``eval/summary.json`` (M11 §15.6.11), its own ``eval/epoch{e}/test/`` folders and
  the ``eval/selection.json`` of the training run it evaluated;
- export run (``--export``): ``records/export/epoch{e}/`` and ``records/analysis/epoch{e}/test/`` (M12 §16.6.5);
- cluster run (``--cluster``): ``cluster/`` (M15 ``cluster_summary.json``, ``manifest.json``, figures).

Outputs: ``comparison.csv`` (one row per experiment x checkpoint best_val/last x protocol full/drop_last_bs4/
drop_last_bs8, columns of §18.6.8), ``comparison.md`` (tables A-F, the D17 definitions, table D = the only place with
paper numbers, the limitations of §18.6.10 and, when a run is not float16 / fp16_amp, a precision note, D9),
``spec20_index.json`` (SPEC §20 names -> run files and sha256) and
``summary.md`` (SPEC §19 structure). A missing file or key is ``NOT_RUN`` with its reason, never an estimate or 0;
a run whose ``meta.json`` status is not ``completed`` gives ``NOT_RUN`` in every cell [SPEC §0.11, §19]. Metric
numbers are copied from ``metrics.json`` unchanged after checking them against their own counts within 1e-12; a file
that contradicts itself stops the comparison (``CompareError``, ``meta.json`` failed).

ElapsedTime (§18.6.7) of a training run is its compute time ``train_summary.seconds.total``: the training,
validation and saving time of its completed epochs (kept in each checkpoint's ``val_metrics.json``, so they carry
across ``--resume``) plus its test time. The idle time between a failure and its resume is not counted (nor the work
lost in an interrupted epoch). An evaluation run (E3, never resumed) keeps ``meta.json`` ``elapsed_s``. The wall
clock ``meta.json`` ``elapsed_s`` (first start to last end) is a separate column (``wall_clock_s``) with the number
of resumes; segment durations are not summed (a killed segment has no end time).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

REBUILD_DIR = Path(__file__).resolve().parents[1]
if str(REBUILD_DIR) not in sys.path:          # python -I rebuild/tools/compare_runs.py from any folder
    sys.path.insert(0, str(REBUILD_DIR))

import yaml  # noqa: E402

from ilora import run  # noqa: E402  (needs the sys.path entry above)

PathLike = Union[str, Path]
logger = logging.getLogger("ilora.compare")

NOT_RUN = "NOT_RUN"                                   # [SPEC §0.11, §19]
PROFILE = "compare"                                   # runs/rebuild/compare_<UTC>/ (RUNBOOK R14 ``latest compare``)
CHECKPOINTS = ("best_val", "last")                    # D11 (a) primary, (b)
SELECTION_KEYS = {"best_val": "best_val_epoch", "last": "last_epoch"}    # M10 eval/selection.json (§14.6.7)
PROTOCOLS = ("full", "drop_last_bs4", "drop_last_bs8")                  # D17 main + auxiliary (M11 §15.6.6)
AUX_PROTOCOLS = PROTOCOLS[1:]
CSV_COLUMNS = ("method", "run", "routing_mode", "num_moe", "lora_r", "seed", "checkpoint", "epoch", "protocol",
               "n_total", "n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric", "trainable_params",
               "elapsed_s", "wall_clock_s", "resumes", "source_path", "source_sha256")     # §18.6.8
ELAPSED_KEYS = ("train_summary", "seconds", "total")   # §18.6.7 ElapsedTime of a training run (§14.6.7)
IDENTITY_COLUMNS = ("method", "run", "checkpoint", "protocol")              # never NOT_RUN: they name the row
METRIC_FIELDS = ("n_total", "n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")
METRIC_ATOL = 1e-12                                   # §18.6.8 = M12 METRIC_ATOL (§16.6.4)
EXPERIMENTS = {"E1": "LLaRA baseline (E1)", "E2": "iLoRA Dynamic (E2)", "E3": "iLoRA text-only eval (E3)",
               "E4": "Cluster Hard (E4)"}           # §18.6.7 table A rows, always listed (NOT_RUN when not given)
TABLE_C_EXCLUDED = ("E3",)                            # §18.6.7 table C: E1, E2, E4 (and other training runs)
PAIRS = (("E2", "E1"), ("E4", "E2"), ("E3", "E2"))   # §18.6.7 table E
OUTPUT_FILES = ("comparison.csv", "comparison.md", "spec20_index.json", "summary.md")
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
BASELINE_FIGURES = ("gate_heatmap", "expert_mean_utilization", "argmax_expert_distribution", "entropy_distribution")
CLUSTER_FIGURES = ("sasrec_pca_clusters", "cluster_dynamic_gate_heatmap")
SPEC20_FIGURE_FORMAT = "png"                          # SPEC §20 names .png files

# §18.6.7 table D. The paper values appear only in comparison.md table D, never in a measured column; "ours" names
# the same-protocol values of our runs that sit next to them (E1 full protocol, E2 drop_last_bs4).
PAPER_REFERENCE = (
    {"source": "[LLaRA Table 2 p.8]", "row": "LLaRA (SASRec), MovieLens", "valid_ratio": "0.9684",
     "hit_ratio_at_1": "0.4421", "counts": (95, 92, 42), "denominator": "correct / total",
     "ours": ("E1", "full", ("valid_ratio", "combined_metric"))},
    {"source": "[iLoRA paper p.7 Table 1]", "row": "iLoRA, MovieLens", "valid_ratio": "0.9891",
     "hit_ratio_at_1": "0.5275", "counts": (92, 91, 48), "denominator": "correct / valid",
     "ours": ("E2", "drop_last_bs4", ("valid_ratio", "hr_at_1"))},
    {"source": "[iLoRA paper p.7 Table 1]", "row": "LLaRA, MovieLens", "valid_ratio": "0.9684",
     "hit_ratio_at_1": "0.4421", "counts": None, "denominator": None, "ours": None},
)
PAPER_IMPROVEMENT = "19.32%"                          # iLoRA Table 1 Imp.% = 0.5275 / 0.4421 - 1 (table D note only)
# §18.6.7 table A: calculated TrainableParams (M9 13.8 #13, M10 14.6.3) of Llama-2-7B, projector 17,047,552 + LoRA
# 19,988,480 + gate 256 (dynamic only); a run record that differs is reported with the difference.
EXPECTED_TRAINABLE = {"llama2_7b": {"dynamic": 37_036_288, "none": 37_036_032, "cluster_hard": 37_036_032}}
# paper_spec (model.torch_dtype, train.precision): fp16 base and fp16 autocast [논문 App E p.17] D9. Other values
# (variant_bf16, CPU smoke fp32) are written as a limitation (precision_note).
MAIN_PRECISION = ("float16", "fp16_amp")

# §18.6.10 limitations (written with every comparison; item 2 depends on whether a total_rank run was given).
LIMITATIONS = (
    "Active capacity (E4): r = 8 and K = 4 give rank r* = 2 per expert [paper p.5 eq. 9]. A one-hot gate uses one "
    "rank-2 expert per layer while the soft gate of E2 mixes four (rank 8 in total), so the E4 - E2 difference cannot "
    "be split into routing and active capacity [SPEC §13]. r and K are not changed in the main runs; per-expert full "
    "rank, top-k and matched active rank are follow-up candidates without a profile (NOT_RUN).",
    "D8 scaling: the main setting is alpha / r* = 16 (expert_rank). With a one-hot gate the active expert's update is "
    "16 B_k A_k; with a uniform soft gate (0.25 each) one expert contributes 4 B_k A_k, so a single active expert "
    "weighs four times as much. With a uniform gate E2's total update equals standard LoRA (alpha / r = 4) with the "
    "same A, B; the total_rank reading gives a quarter of that. Conclusions of E1, E2 and E4 depend on D8.",
    "One seed, small N: D2, seed 1234 only. One test sample is 1/95, about 0.0105; the spread of training re-runs "
    "(D16) is not measured (NOT_RUN).",
    "Title judgement defect (D20): with the fixed candidates, 1 validation and 2 test samples cannot be judged correct "
    "even with the right title, so CombinedMetric is at most 93/94 and 93/95 (M2 6.6.6). Flag-free metrics are "
    "supplementary only.",
    "Protocols of the paper table: the iLoRA row uses N = 92 and correct / valid, the LLaRA row N = 95 and correct / "
    "total (D17); the LLaRA row was copied from the LLaRA paper, not re-run by iLoRA [paper p.6 §4].",
    "E1 is not a run of the LLaRA code: it is K = 1 of this re-implementation (fp16 base load D9, no Lightning, no "
    "max_steps truncation D24, no save filter or CNN router D26), so its difference to LLaRA Table 2 is not explained "
    "by K alone.",
    "Clusters (E4): KMeans on samples (sequences, D25); cluster numbers carry no meaning [SPEC §9]. The user unit is a "
    "variant.",
    "E3 changes only the evaluation prompt. The model was trained with the curriculum (D3) on text-only prompts too, "
    "so E3 is not the result of a model trained text-only.",
)


class CompareError(RuntimeError):
    """A run file contradicts itself (metrics vs counts, summary vs files); nothing is compared from it."""


@dataclass(frozen=True)
class NotRun:
    """A value that no file provides: written as ``NOT_RUN``; ``reason`` goes to comparison.md and summary.md."""
    reason: str

    def __str__(self) -> str:
        return NOT_RUN


def missing(value: object) -> bool:
    return isinstance(value, NotRun)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def read_json(path: Path) -> dict:
    """A JSON object written by the commands (atomic writes, so a broken file is a ``CompareError``)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompareError(f"{path}: cannot be read as JSON ({type(exc).__name__}: {exc})") from exc
    if not isinstance(data, dict):
        raise CompareError(f"{path}: expected a JSON object")
    return data


def read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CompareError(f"{path}: cannot be read as YAML ({type(exc).__name__}: {exc})") from exc
    if not isinstance(data, dict):
        raise CompareError(f"{path}: expected a YAML mapping")
    return data


def dig(mapping: Any, keys: Sequence[str], where: str) -> Any:
    """``mapping[k1][k2]...`` or ``NotRun`` naming the first missing key (a null value is missing too); a
    ``"NOT_RUN: ..."`` string on the way (M15 dynamic items without an export) keeps its own reason."""
    value = mapping
    for i, key in enumerate(keys):
        if missing(value):
            return value
        if isinstance(value, str) and value.startswith(NOT_RUN):
            return NotRun(f"{where}: {value}")
        if not isinstance(value, Mapping) or key not in value or value[key] is None:
            return NotRun(f"{where} has no {'.'.join(keys[:i + 1])}")
        value = value[key]
    if isinstance(value, str) and value.startswith(NOT_RUN):
        return NotRun(f"{where} {'.'.join(keys)}: {value}")
    return value


def check_counts(block: Any, where: str) -> None:
    """D17 within 1e-12: valid_ratio = v / n, hr_at_1 = c / v (0 when v = 0), combined_metric = c / n (§18.6.8)."""
    if not isinstance(block, Mapping):
        raise CompareError(f"{where}: expected a mapping of metrics")
    n, v, c = (block.get(key) for key in ("n_total", "n_valid", "n_correct"))
    if not all(_is_int(x) for x in (n, v, c)) or not 0 <= c <= v <= n or n < 1:
        raise CompareError(f"{where}: counts n_total={n!r}, n_valid={v!r}, n_correct={c!r} are not 0 <= c <= v <= n")
    expected = {"valid_ratio": v / n, "hr_at_1": c / v if v else 0.0, "combined_metric": c / n}
    for key, value in expected.items():
        got = block.get(key)
        if not _is_number(got) or abs(float(got) - value) > METRIC_ATOL:
            raise CompareError(f"{where}: {key} = {got!r} differs from the recomputed {value!r} "
                               f"(n={n}, valid={v}, correct={c}) by more than {METRIC_ATOL}")


def cell(value: Any) -> str:
    """CSV/Markdown text of a value: ``NOT_RUN``, ints as written, floats with ``repr`` (exact round trip)."""
    if value is None or missing(value):
        return NOT_RUN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)

# ----------------------------------------------------------------------------
# Reading run folders
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Metrics:
    """One test ``metrics.json`` (M12 ``write_eval``), checked against its own counts."""
    path: Path
    sha256: str
    epoch: int
    data: Mapping[str, Any]

    def block(self, protocol: str) -> Any:
        """The ``full`` metrics (top level) or an ``aux_protocols`` block, or ``NotRun``."""
        if protocol == "full":
            return self.data
        aux = self.data.get("aux_protocols")
        if not isinstance(aux, Mapping) or protocol not in aux:
            return NotRun(f"{self.path} has no aux_protocols.{protocol}")
        return aux[protocol]


@dataclass
class RunView:
    """What the comparison reads from one run folder; ``problem`` set -> every cell of the run is ``NOT_RUN``."""
    name: str
    path: Optional[Path]
    role: str                                         # run (training or eval), export, cluster
    problem: Optional[str] = None
    meta: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    kind: Optional[str] = None                        # meta.json command
    source_run: Optional[Path] = None                 # eval/export run: the training run it read
    epochs: dict = field(default_factory=dict)        # best_val/last -> int | NotRun
    metrics: dict = field(default_factory=dict)       # best_val/last -> Metrics | NotRun
    records: dict = field(default_factory=dict)       # best_val/last -> records.jsonl Path | NotRun
    run_summary: Any = None                           # records/train/run_summary.json (dict) | NotRun
    run_summary_path: Optional[Path] = None

    @property
    def usable(self) -> bool:
        return self.problem is None

    def unusable(self) -> NotRun:
        return NotRun(f"{self.name}: {self.problem}")

    def config_value(self, section: str, key: str) -> Any:
        if not self.usable:
            return self.unusable()
        return dig(self.config, (section, key), f"{self.path / 'config.yaml'}")

    def elapsed(self) -> Any:
        """ElapsedTime (§18.6.7): a training run's ``run_summary.json`` ``train_summary.seconds.total`` (compute time
        of the completed epochs and the test, carried across resumes by ``val_metrics.json``, §14.6.7); any other
        run (E3 eval, export, cluster) its wall clock."""
        if not self.usable:
            return self.unusable()
        if self.kind == "train":
            return dig(self.run_summary, ELAPSED_KEYS, str(self.run_summary_path))
        return self.wall_clock()

    def wall_clock(self) -> Any:
        """``meta.json`` ``elapsed_s``: first start to last end, the idle time before each ``--resume`` included."""
        if not self.usable:
            return self.unusable()
        return dig(self.meta, ("elapsed_s",), f"{self.path / 'meta.json'}")

    def resumes(self) -> Any:
        """Number of ``meta.json`` ``resumes`` entries (M0 ``train --resume``; 0 for a run never resumed)."""
        if not self.usable:
            return self.unusable()
        return len(self.meta.get("resumes") or [])

    def trainable_params(self) -> Any:
        if not self.usable:
            return self.unusable()
        return dig(self.run_summary, ("train_summary", "trainable_params", "total"), str(self.run_summary_path))


ROLE_COMMANDS = {"run": ("train", "eval"), "export": ("export",), "cluster": ("cluster",)}


def open_run(name: str, path: Optional[Path], role: str) -> RunView:
    """Folder, ``meta.json`` status ``completed`` and the command of the role (M0 ``meta.json``), ``config.yaml``."""
    view = RunView(name=name, path=path, role=role)
    flag = "--cluster" if role == "cluster" else f"--{role} {name}=<folder>"
    if path is None:
        view.problem = f"not given ({flag})"
        return view
    if not path.is_dir():
        view.problem = f"run folder {path} not found"
        return view
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        view.problem = f"{meta_path} not found"
        return view
    view.meta = read_json(meta_path)
    status = view.meta.get("status")
    if status != "completed":
        error = view.meta.get("error")
        view.problem = f"meta.json status is {status!r}, not 'completed'" + (f" (error: {error})" if error else "")
        return view
    view.kind = view.meta.get("command")
    if view.kind not in ROLE_COMMANDS[role]:
        view.problem = f"meta.json command is {view.kind!r}, expected {' or '.join(ROLE_COMMANDS[role])}"
        return view
    config_path = path / "config.yaml"
    if not config_path.is_file():
        view.problem = f"{config_path} not found"
        return view
    view.config = read_yaml(config_path)
    return view


def read_metrics(path: Path, epoch: int) -> Any:
    """``Metrics`` of ``eval/epoch{e}/test/metrics.json`` or ``NotRun`` when the file is missing; a file of another
    split or epoch, or whose ratios differ from its counts, is a ``CompareError``."""
    if not path.is_file():
        return NotRun(f"{path} not found")
    data = read_json(path)
    if data.get("split") != "test" or data.get("epoch") != epoch:
        raise CompareError(f"{path}: split {data.get('split')!r} epoch {data.get('epoch')!r}, expected test epoch {epoch}")
    if "ckpt_tag" in data and data["ckpt_tag"] != f"epoch{epoch}":
        raise CompareError(f"{path}: ckpt_tag {data['ckpt_tag']!r} != epoch{epoch}")
    check_counts(data, str(path))
    aux = data.get("aux_protocols")
    if isinstance(aux, Mapping):
        for protocol in AUX_PROTOCOLS:
            if protocol in aux:
                check_counts(aux[protocol], f"{path} aux_protocols.{protocol}")
    return Metrics(path=path, sha256=run.content_sha256(path), epoch=epoch, data=data)


def selection_epochs(selection_path: Path) -> dict[str, Any]:
    """best_val/last -> epoch of a training run's ``eval/selection.json`` (D11), ``NotRun`` when absent."""
    if not selection_path.is_file():
        return {ckpt: NotRun(f"{selection_path} not found") for ckpt in CHECKPOINTS}
    selection = read_json(selection_path)
    out: dict[str, Any] = {}
    for ckpt in CHECKPOINTS:
        epoch = selection.get(SELECTION_KEYS[ckpt])
        if epoch is None:
            out[ckpt] = NotRun(f"{selection_path} has no {SELECTION_KEYS[ckpt]}")
        elif not _is_int(epoch) or epoch < 0:
            raise CompareError(f"{selection_path}: {SELECTION_KEYS[ckpt]} = {epoch!r} is not an epoch number")
        else:
            out[ckpt] = epoch
    return out


def load_run(name: str, path: Optional[Path]) -> RunView:
    """A ``--run`` folder: a training run, or an eval run (E3) of a training run's checkpoints."""
    view = open_run(name, path, "run")
    if not view.usable:
        return view
    assert path is not None
    summary_doc: Optional[dict] = None
    if view.kind == "train":
        train_root = path
    else:
        summary_path = path / "eval" / "summary.json"
        if not summary_path.is_file():
            view.problem = f"{summary_path} not found"
            return view
        summary_doc = read_json(summary_path)
        source = summary_doc.get("source_run")
        if not isinstance(source, str) or not source:
            raise CompareError(f"{summary_path}: no source_run")
        view.source_run = train_root = run.resolve_path(source)
    view.epochs = selection_epochs(train_root / "eval" / "selection.json")
    for ckpt in CHECKPOINTS:
        epoch = view.epochs[ckpt]
        if missing(epoch):
            view.metrics[ckpt] = view.records[ckpt] = epoch
            continue
        folder = path / "eval" / f"epoch{epoch}" / "test"
        metrics = read_metrics(folder / "metrics.json", epoch)
        if summary_doc is not None and not missing(metrics):
            item = (summary_doc.get("results") or {}).get(f"epoch{epoch}/test")
            if not isinstance(item, Mapping):
                metrics = NotRun(f"{path / 'eval' / 'summary.json'} has no result epoch{epoch}/test")
            elif (item.get("files") or {}).get("metrics.json") != metrics.sha256:
                raise CompareError(f"{metrics.path}: sha256 differs from the one in {path / 'eval' / 'summary.json'}")
        view.metrics[ckpt] = metrics
        records = folder / "records.jsonl"
        view.records[ckpt] = records if records.is_file() else NotRun(f"{records} not found")
    view.run_summary_path = train_root / "records" / "train" / "run_summary.json"
    view.run_summary = (read_json(view.run_summary_path) if view.run_summary_path.is_file()
                        else NotRun(f"{view.run_summary_path} not found"))
    return view


def load_aux(name: str, path: Optional[Path], role: str) -> RunView:
    """An export (``--export``) or cluster (``--cluster``) run: status, command, its source training run."""
    view = open_run(name, path, role)
    if view.usable and role == "export":
        source = view.meta.get("source_run")
        view.source_run = run.resolve_path(source) if isinstance(source, str) and source else None
    return view


def export_epoch(export: RunView, training: Optional[RunView]) -> Any:
    """Epoch a (best_val) of an export: from the training run given with ``--run``, else from the export's source."""
    if not export.usable:
        return export.unusable()
    if export.source_run is None:
        return NotRun(f"{export.path / 'meta.json'} has no source_run")
    if training is not None and training.usable and training.kind == "train":
        if export.source_run.resolve() != training.path.resolve():
            return NotRun(f"export {export.name} was made from {export.source_run}, not from the {training.name} run "
                          f"{training.path}")
        return training.epochs["best_val"]
    return selection_epochs(export.source_run / "eval" / "selection.json")["best_val"]


def read_records(path: Path) -> list:
    """M12 ``read_eval_records`` (field order and types checked); a malformed file is a ``CompareError``."""
    from ilora.records import read_eval_records

    try:
        return read_eval_records(path)
    except ValueError as exc:                  # RecordsError is a ValueError, and so is a JSON decoding error
        raise CompareError(f"{path}: {exc}") from exc

# ----------------------------------------------------------------------------
# Table values
# ----------------------------------------------------------------------------
def method_label(name: str) -> str:
    return EXPERIMENTS.get(name, name)


def comparison_row(view: RunView, ckpt: str, protocol: str) -> dict[str, Any]:
    """One ``comparison.csv`` row (§18.6.8): config values, the epoch of ``eval/selection.json``, the metrics of
    ``eval/epoch{e}/test/metrics.json`` (full or ``aux_protocols``), TrainableParams, ElapsedTime, the wall clock
    with the number of resumes and the source."""
    row: dict[str, Any] = {"method": method_label(view.name), "checkpoint": ckpt, "protocol": protocol,
                           "run": str(view.path) if view.path is not None else view.unusable()}
    if not view.usable:
        reason = view.unusable()
        return {col: row.get(col, reason) for col in CSV_COLUMNS}
    row.update(routing_mode=view.config_value("router", "routing_mode"),
               num_moe=view.config_value("lora", "num_experts"), lora_r=view.config_value("lora", "r"),
               seed=view.config_value("run", "seed"), epoch=view.epochs[ckpt])
    metrics = view.metrics[ckpt]
    block = metrics if missing(metrics) else metrics.block(protocol)
    for name in METRIC_FIELDS:
        row[name] = block if missing(block) else dig(block, (name,), f"{metrics.path} {protocol}")
    row["trainable_params"] = view.trainable_params()
    row["elapsed_s"] = view.elapsed()
    row["wall_clock_s"] = view.wall_clock()
    row["resumes"] = view.resumes()
    row["source_path"] = metrics if missing(metrics) else str(metrics.path)
    row["source_sha256"] = metrics if missing(metrics) else metrics.sha256
    return {col: row[col] for col in CSV_COLUMNS}


def pair_comparison(a: RunView, b: RunView) -> dict[str, Any]:
    """Table E for one pair (test, best_val, full protocol, §18.6.7): the same samples, candidates, templates and
    targets in both ``records.jsonl``; correct counts and their difference, samples only one side got right
    (discordant pairs) and CombinedMetric A / B - 1. Numbers only, no test or judgement [결정]."""
    out: dict[str, Any] = {"pair": f"{a.name} vs {b.name}", "a": a.name, "b": b.name}
    for view in (a, b):
        if not view.usable:
            out["problem"] = view.unusable().reason
            return out
        for value in (view.metrics["best_val"], view.records["best_val"]):
            if missing(value):
                out["problem"] = f"{view.name}: {value.reason}"
                return out
    ma, mb = a.metrics["best_val"], b.metrics["best_val"]
    ra, rb = read_records(a.records["best_val"]), read_records(b.records["best_val"])
    for view, metrics, recs in ((a, ma, ra), (b, mb, rb)):
        correct = sum(1 for r in recs if r.correct)
        if len(recs) != metrics.data["n_total"] or correct != metrics.data["n_correct"]:
            raise CompareError(f"{view.records['best_val']}: {len(recs)} records, {correct} correct; {metrics.path} has "
                               f"n_total {metrics.data['n_total']}, n_correct {metrics.data['n_correct']}")
    if [r.sample_id for r in ra] != [r.sample_id for r in rb]:
        out["problem"] = "the two test record files hold different samples or another order"
        return out
    differ = [x.sample_id for x, y in zip(ra, rb)
              if (tuple(x.cand_ids), x.template_id, x.target_id) != (tuple(y.cand_ids), y.template_id, y.target_id)]
    if differ:
        out["problem"] = f"candidates, template or target differ for {len(differ)} samples (first {differ[:3]})"
        return out
    only_a = sum(1 for x, y in zip(ra, rb) if x.correct and not y.correct)
    only_b = sum(1 for x, y in zip(ra, rb) if y.correct and not x.correct)
    ca, cb = ma.data["n_correct"], mb.data["n_correct"]
    combined_a, combined_b = ma.data["combined_metric"], mb.data["combined_metric"]
    out.update(n=len(ra), epoch_a=ma.epoch, epoch_b=mb.epoch, correct_a=ca, correct_b=cb, difference=ca - cb,
               only_a=only_a, only_b=only_b, discordant=only_a + only_b,
               combined_ratio_minus_1=(combined_a / combined_b - 1) if combined_b else
               NotRun(f"CombinedMetric of {b.name} is 0, the ratio is undefined"),
               sources={view.name: (str(view.records["best_val"]), run.content_sha256(view.records["best_val"]))
                        for view in (a, b)})
    return out


def gate_summary_view(export: RunView, epoch: Any) -> dict[str, Any]:
    """Test-split ``gate_summary.json`` of an export (M12 §16.6.6) at epoch a: mean utilization, argmax
    distribution, normalized entropy and JS statistics (§18.6.7 table F)."""
    if not export.usable:
        return {"problem": export.unusable().reason}
    if missing(epoch):
        return {"problem": epoch.reason}
    path = export.path / "records" / "analysis" / f"epoch{epoch}" / "test" / "gate_summary.json"
    if not path.is_file():
        return {"problem": f"{path} not found"}
    doc = read_json(path)
    where = str(path)
    out: dict[str, Any] = {"path": str(path), "sha256": run.content_sha256(path), "epoch": epoch,
                           "n": dig(doc, ("n",), where), "num_experts": dig(doc, ("num_experts",), where)}
    for key in ("mean_utilization", "argmax_count", "argmax_ratio", "expert_std"):
        out[key] = dig(doc, ("metrics", key, "per_expert"), where)
    for key in ("normalized_entropy", "js_to_mean_bits", "top1_minus_top2"):
        out[key] = dig(doc, ("metrics", key), where)
    text = path.parent / "gate_interpretation.txt"
    out["interpretation"] = str(text) if text.is_file() else NotRun(f"{text} not found")
    return out


def eval_gate_view(view: RunView) -> dict[str, Any]:
    """Evaluation gates of the test split at best_val (``records.jsonl`` ``gate_0..K-1``): rows that are exactly
    one-hot and the argmax (first maximum) count per expert (§18.6.7 table F, E4 one-hot check)."""
    if not view.usable:
        return {"problem": view.unusable().reason}
    path = view.records["best_val"]
    if missing(path):
        return {"problem": path.reason}
    recs = read_records(path)
    k = len(recs[0].gates) if recs else 0
    counts = [0] * k
    one_hot = 0
    for r in recs:
        gates = [float(g) for g in r.gates]
        counts[gates.index(max(gates))] += 1
        one_hot += int(sorted(gates) == [0.0] * (k - 1) + [1.0])
    return {"path": str(path), "sha256": run.content_sha256(path), "epoch": view.epochs["best_val"], "n": len(recs),
            "num_experts": k, "one_hot_rows": one_hot, "argmax_count": counts,
            "gate_calls": dig(view.run_summary, ("gate_calls",), str(view.run_summary_path)),
            "cluster_lookups": dig(view.run_summary, ("cluster_lookups",), str(view.run_summary_path))}


def cluster_view(cluster: RunView) -> dict[str, Any]:
    """``cluster/cluster_summary.json`` and ``manifest.json`` of the cluster run (M15 §19.6.5-§19.6.6)."""
    if not cluster.usable:
        return {"problem": cluster.unusable().reason}
    folder = cluster.path / "cluster"
    path = folder / "cluster_summary.json"
    if not path.is_file():
        return {"problem": f"{path} not found"}
    doc = read_json(path)
    where = str(path)
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else NotRun(f"{manifest_path} not found")
    return {"path": where, "sha256": run.content_sha256(path), "unit": dig(doc, ("unit",), where),
            "n_clusters": dig(doc, ("n_clusters",), where), "fit_split": dig(doc, ("fit", "split"), where),
            "fit_rows": dig(doc, ("fit", "rows"), where),
            "sizes": {split: dig(doc, (f"{split}_cluster_size",), where) for split in ("train", "validation", "test")},
            "silhouette_score_train": dig(doc, ("silhouette_score_train",), where),
            "nmi": {split: dig(doc, ("nmi_cluster_vs_dynamic_argmax", split), where)
                    for split in ("train", "validation", "test")},
            "reload_check": dig(doc, ("reload_check", "passed"), where),
            "z_max_abs_diff": dig(doc, ("dynamic_export", "z_max_abs_diff"), where),
            "kmeans_file": dig(manifest, ("kmeans_file",), str(manifest_path))}


def index_entry(value: Any) -> dict[str, Any]:
    if missing(value):
        return {"path": NOT_RUN, "sha256": NOT_RUN, "reason": value.reason}
    path = Path(value)
    if not path.is_file():
        return {"path": NOT_RUN, "sha256": NOT_RUN, "reason": f"{path} not found"}
    return {"path": str(path), "sha256": run.content_sha256(path)}


def spec20_index(runs: Mapping[str, RunView], exports: Mapping[str, RunView], cluster: RunView,
                 out_dir: Path) -> dict[str, dict[str, Any]]:
    """SPEC §20 file -> run file and sha256 (§18.6.9; a = E2 best_val epoch, a4 = E4's), ``NOT_RUN`` when absent."""
    e2x = exports.get("E2") or open_run("E2", None, "export")
    a = export_epoch(e2x, runs["E2"])

    def metrics_of(view: RunView) -> Any:
        if not view.usable:
            return view.unusable()
        metrics = view.metrics["best_val"]
        return metrics if missing(metrics) else metrics.path

    def in_export(*parts: str) -> Any:
        if not e2x.usable:
            return e2x.unusable()
        if missing(a):
            return a
        return e2x.path.joinpath("records", parts[0], f"epoch{a}", *parts[1:])

    def in_cluster(*parts: Any) -> Any:
        if not cluster.usable:
            return cluster.unusable()
        if any(missing(p) for p in parts):
            return next(p for p in parts if missing(p))
        return cluster.path.joinpath("cluster", *parts)

    clusters = cluster_view(cluster)
    kmeans = NotRun(clusters["problem"]) if "problem" in clusters else clusters["kmeans_file"]
    figure = f".{SPEC20_FIGURE_FORMAT}"
    entries: dict[str, Any] = {
        "results/baseline/metrics.json": metrics_of(runs["E2"]),
        "results/baseline/gates_test.csv": in_export("export", "gates_test.csv"),
        "results/baseline/gate_summary.json": in_export("analysis", "test", "gate_summary.json")}
    for name in BASELINE_FIGURES:
        entries[f"results/baseline/figures/{name}{figure}"] = in_export("analysis", "test", "figures", name + figure)
    entries["artifacts/kmeans_k4_seed1234.joblib"] = in_cluster(kmeans)       # M15 manifest kmeans_file
    entries["results/clustering/cluster_summary.json"] = in_cluster("cluster_summary.json")
    for name in CLUSTER_FIGURES:
        entries[f"results/clustering/figures/{name}{figure}"] = in_cluster("figures", name + figure)
    entries["results/cluster_hard/metrics.json"] = metrics_of(runs["E4"])
    entries["results/final_comparison.csv"] = out_dir / "comparison.csv"
    entries["results/final_comparison.md"] = out_dir / "comparison.md"
    entries["results/today_summary.md"] = out_dir / "summary.md"
    return {spec: index_entry(value) for spec, value in entries.items()}

# ----------------------------------------------------------------------------
# Report: every value read once
# ----------------------------------------------------------------------------
@dataclass
class Report:
    runs: dict[str, RunView]                      # E1-E4 first (always), then the other --run names in order
    exports: dict[str, RunView]
    cluster: RunView
    table: dict[tuple[str, str, str], dict]       # (name, checkpoint, protocol) -> comparison.csv row
    pairs: list[dict[str, Any]]
    gates_e2: dict[str, Any]
    gates_e4: dict[str, Any]
    clusters: dict[str, Any]
    created_utc: str

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self.table.values())


def build_report(runs: Mapping[str, Optional[Path]], exports: Mapping[str, Optional[Path]],
                 cluster: Optional[Path]) -> Report:
    names = list(EXPERIMENTS) + [name for name in runs if name not in EXPERIMENTS]
    views = {name: load_run(name, runs.get(name)) for name in names}
    export_views = {name: load_aux(name, path, "export") for name, path in exports.items()}
    cluster_v = load_aux("cluster", cluster, "cluster")
    table = {(name, ckpt, protocol): comparison_row(views[name], ckpt, protocol)
             for name in names for ckpt in CHECKPOINTS for protocol in PROTOCOLS}
    e2x = export_views.get("E2") or open_run("E2", None, "export")
    return Report(runs=views, exports=export_views, cluster=cluster_v, table=table,
                  pairs=[pair_comparison(views[a], views[b]) for a, b in PAIRS],
                  gates_e2=gate_summary_view(e2x, export_epoch(e2x, views["E2"])), gates_e4=eval_gate_view(views["E4"]),
                  clusters=cluster_view(cluster_v), created_utc=run.utc_now().isoformat())


def collect_reasons(report: Report) -> list[str]:
    """Every distinct NOT_RUN reason of the report, in reading order."""
    seen: dict[str, None] = {}

    def walk(value: Any) -> None:
        if missing(value):
            seen.setdefault(value.reason, None)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if key == "problem":
                    seen.setdefault(str(item), None)
                else:
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for view in (*report.runs.values(), *report.exports.values(), report.cluster):
        if not view.usable:
            seen.setdefault(view.unusable().reason, None)
    for item in (report.rows, report.pairs, report.gates_e2, report.gates_e4, report.clusters):
        walk(item)
    return list(seen)


# ----------------------------------------------------------------------------
# comparison.md
# ----------------------------------------------------------------------------
RATIO_COUNTS = {"valid_ratio": ("n_valid", "n_total"), "hr_at_1": ("n_correct", "n_valid"),
                "combined_metric": ("n_correct", "n_total")}


def md_cell(value: Any) -> str:
    text = value if isinstance(value, str) else cell(value)
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(md_cell(value) for value in row) + " |" for row in rows]


def ratio_text(row: Optional[Mapping[str, Any]], key: str) -> str:
    """``value (k/n)`` with the counts of the D17 definition, or ``NOT_RUN``."""
    value = None if row is None else row.get(key)
    if value is None or missing(value):
        return NOT_RUN
    num, den = RATIO_COUNTS[key]
    return f"{cell(value)} ({cell(row[num])}/{cell(row[den])})"


def seconds_text(value: Any) -> str:
    return NOT_RUN if value is None or missing(value) else f"{cell(value)} s"


def wall_clock_text(seconds: Any, resumes: Any) -> str:
    """The wall clock (``meta.json`` ``elapsed_s``) with the number of resumes (§18.6.7), or ``NOT_RUN``."""
    if seconds is None or missing(seconds):
        return NOT_RUN
    return f"{cell(seconds)} s (resumes {cell(resumes)})"


def table_ab(report: Report, ckpt: str) -> list[str]:
    header = ("Method", "Routing", "ValidRatio", "HR@1", "CombinedMetric", "Seed", "Epoch", "num_moe", "lora_r",
              "TrainableParams", "ElapsedTime", "WallClock")
    rows = []
    for name in report.runs:
        row = report.table[(name, ckpt, "full")]
        rows.append((row["method"], row["routing_mode"], ratio_text(row, "valid_ratio"), ratio_text(row, "hr_at_1"),
                     ratio_text(row, "combined_metric"), row["seed"], row["epoch"], row["num_moe"], row["lora_r"],
                     row["trainable_params"], seconds_text(row["elapsed_s"]),
                     wall_clock_text(row["wall_clock_s"], row["resumes"])))
    return md_table(header, rows)


def trainable_check(report: Report) -> list[str]:
    """TrainableParams of each run against the calculation of §18.6.7 (M9 13.8 #13, M10 14.6.3), with the difference."""
    lines = []
    for name, view in report.runs.items():
        actual = view.trainable_params()
        backbone, mode = view.config_value("model", "backbone"), view.config_value("router", "routing_mode")
        expected = None if missing(backbone) or missing(mode) else EXPECTED_TRAINABLE.get(backbone, {}).get(mode)
        if missing(actual):
            text = NOT_RUN
        elif expected is None:
            text = f"{actual} (no calculated value for backbone {cell(backbone)})"
        else:
            text = f"{actual} vs calculated {expected}: " + ("equal" if actual == expected else
                                                             f"differs by {actual - expected:+d}")
        lines.append(f"- {method_label(name)}: {text}")
    return lines


def table_c(report: Report) -> list[str]:
    header = ("Method", "Checkpoint", "Protocol", "N", "ValidRatio", "HR@1", "CombinedMetric")
    rows = []
    for name in report.runs:
        if name in TABLE_C_EXCLUDED:
            continue
        for ckpt in CHECKPOINTS:
            for protocol in AUX_PROTOCOLS:
                row = report.table[(name, ckpt, protocol)]
                rows.append((row["method"], ckpt, protocol, row["n_total"], ratio_text(row, "valid_ratio"),
                             ratio_text(row, "hr_at_1"), ratio_text(row, "combined_metric")))
    return md_table(header, rows)


def counterpart_text(report: Report, ref: Mapping[str, Any]) -> str:
    """Our same-protocol values next to a paper row (table D) and the differences in samples."""
    name, protocol, keys = ref["ours"]
    row = report.table.get((name, "best_val", protocol))
    parts = [f"{name} best_val {protocol}: " + ", ".join(f"{key} {ratio_text(row, key)}" for key in keys)]
    n_ref, v_ref, c_ref = ref["counts"]
    counts = [row.get(key) if row else None for key in ("n_total", "n_valid", "n_correct")]
    if any(value is None or missing(value) for value in counts):
        parts.append("sample differences NOT_RUN")
    elif counts[0] != n_ref:
        parts.append(f"sample differences NOT_RUN (our N {counts[0]} != {n_ref})")
    else:
        parts.append(f"valid {counts[1] - v_ref:+d}, correct {counts[2] - c_ref:+d} samples against the back-solution")
    return "; ".join(parts)


def table_d(report: Report) -> list[str]:
    header = ("Source", "Row", "ValidRatio", "HitRatio@1", "Integer back-solution (N, valid, correct)",
              "HitRatio@1 denominator", "Our counterpart (from the run files)")
    rows = []
    for ref in PAPER_REFERENCE:
        counts = ref["counts"]
        rows.append((ref["source"], ref["row"], ref["valid_ratio"], ref["hit_ratio_at_1"],
                     f"({counts[0]}, {counts[1]}, {counts[2]})" if counts else
                     "the LLaRA Table 2 values, copied (not re-run) by the iLoRA paper", ref["denominator"] or "-",
                     counterpart_text(report, ref) if ref["ours"] else "-"))
    llara, ilora = PAPER_REFERENCE[0], PAPER_REFERENCE[1]
    return ["Paper numbers appear only in this table. Each paper row is set against our values of the same "
            "protocol: E1 full protocol (CombinedMetric = correct / total) against LLaRA, E2 drop_last_bs4 (HR@1 = "
            "correct / valid) against iLoRA.", "", *md_table(header, rows), "",
            "- Back-solution: the only integer triple with N in 80-95 that rounds to both four-digit values (D17). "
            "The iLoRA row has N = 92 (batch 4 with drop_last), the LLaRA row N = 95.",
            f"- The paper's Imp.% {PAPER_IMPROVEMENT} = {ilora['hit_ratio_at_1']} / {llara['hit_ratio_at_1']} - 1 "
            "divides two numbers with different denominators and N; our runs are compared only within one protocol.",
            "- No reproduction threshold is set: one seed (D2), one test sample = 1/95, GPU non-determinism of "
            "training (D16) and the different run conditions of the reference values (limitations below). "
            "Differences are given in samples; Wilson intervals stay in the supplementary block of each metrics.json."]

def table_e(report: Report) -> list[str]:
    header = ("Pair (A vs B)", "Epochs (A, B)", "N", "Correct A", "Correct B", "A - B", "Only A correct",
              "Only B correct", "Discordant", "CombinedMetric A / B - 1")
    rows, sources = [], []
    for p in report.pairs:
        if "problem" in p:
            rows.append((p["pair"], *([NOT_RUN] * (len(header) - 1))))
            continue
        rows.append((p["pair"], f"{p['epoch_a']}, {p['epoch_b']}", p["n"], p["correct_a"], p["correct_b"],
                     f"{p['difference']:+d}", p["only_a"], p["only_b"], p["discordant"], p["combined_ratio_minus_1"]))
        sources += [f"- {p['pair']}: {name} `{path}` (sha256 {sha})" for name, (path, sha) in p["sources"].items()]
    return ["Test split, best_val checkpoints, full protocol: the same samples, candidates, templates and targets "
            "(D10, D15) are checked in both records.jsonl files. Counts only, no test or judgement.", "",
            *md_table(header, rows), "", *(sources or ["- no pair could be compared"])]


def _list_text(value: Any) -> str:
    if missing(value):
        return NOT_RUN
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cell(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return ", ".join(f"{key} {cell(item) if not isinstance(item, (list, tuple, Mapping)) else _list_text(item)}"
                         for key, item in value.items())
    return cell(value)


def table_f(report: Report) -> list[str]:
    g2, g4, cl = report.gates_e2, report.gates_e4, report.clusters
    out = ["### E2 dynamic gate (test split, export at the best_val epoch)", ""]
    if "problem" in g2:
        out.append(f"NOT_RUN: {g2['problem']}")
    else:
        out += md_table(("Statistic", "Value"), [
            ("source", f"`{g2['path']}` (sha256 {g2['sha256']})"), ("epoch", g2["epoch"]), ("n", g2["n"]),
            ("num_experts", g2["num_experts"]), ("mean utilization per expert", _list_text(g2["mean_utilization"])),
            ("argmax count per expert", _list_text(g2["argmax_count"])),
            ("argmax ratio per expert", _list_text(g2["argmax_ratio"])),
            ("expert std per expert", _list_text(g2["expert_std"])),
            ("normalized entropy", _list_text(g2["normalized_entropy"])),
            ("JS divergence to the mean gate (bits)", _list_text(g2["js_to_mean_bits"])),
            ("top-1 minus top-2", _list_text(g2["top1_minus_top2"]))])
    out += ["", "### E4 evaluation gates (test split, best_val)", ""]
    if "problem" in g4:
        out.append(f"NOT_RUN: {g4['problem']}")
    else:
        out += md_table(("Statistic", "Value"), [
            ("source", f"`{g4['path']}` (sha256 {g4['sha256']})"), ("epoch", g4["epoch"]), ("n", g4["n"]),
            ("one-hot rows", f"{g4['one_hot_rows']}/{g4['n']}"), ("argmax count per expert",
                                                                 _list_text(g4["argmax_count"])),
            ("training gate_calls (run_summary.json)", g4["gate_calls"]),
            ("training cluster_lookups (run_summary.json)", g4["cluster_lookups"])])
    out += ["", "### SASRec clusters", ""]
    if "problem" in cl:
        out.append(f"NOT_RUN: {cl['problem']}")
    else:
        out += md_table(("Statistic", "Value"), [
            ("source", f"`{cl['path']}` (sha256 {cl['sha256']})"), ("unit", cl["unit"]),
            ("n_clusters", cl["n_clusters"]), ("fit split, rows", f"{cell(cl['fit_split'])}, {cell(cl['fit_rows'])}"),
            *((f"{split} cluster size", _list_text(size)) for split, size in cl["sizes"].items()),
            ("silhouette_score_train", cl["silhouette_score_train"]),
            *((f"NMI cluster vs dynamic argmax ({split})", value) for split, value in cl["nmi"].items()),
            ("reload check passed", cl["reload_check"]),
            ("max abs z difference to the dynamic export", _list_text(cl["z_max_abs_diff"]))])
    return out


def runs_table(report: Report) -> list[str]:
    rows = []
    for role, views in (("run", report.runs.values()), ("export", report.exports.values()),
                        ("cluster", (report.cluster,))):
        for view in views:
            rows.append((view.name, role, f"`{view.path}`" if view.path is not None else NOT_RUN,
                         view.meta.get("status", NOT_RUN), view.meta.get("command", NOT_RUN),
                         "-" if view.usable else f"NOT_RUN: {view.problem}"))
    return md_table(("Name", "Role", "Folder", "Status", "Command", "Usable"), rows)


def precision_note(report: Report) -> Optional[str]:
    """A limitation when a compared run is not in the main precision ``MAIN_PRECISION`` (D9): the ``variant_bf16``
    fallback is a recorded deviation (§18.6.4), and runs of different precisions in one comparison mix precision
    into the paired differences (RUNBOOK bf16 block: E1, E3 and E4 follow E2). Values come from each run's
    ``config.yaml``; None when every usable run has the main precision."""
    found: dict[str, tuple[Any, Any]] = {}
    for name, view in report.runs.items():
        if not view.usable:
            continue
        dtype, precision = view.config_value("model", "torch_dtype"), view.config_value("train", "precision")
        if not missing(dtype) and not missing(precision):
            found[name] = (dtype, precision)
    kinds = set(found.values())
    if not kinds or kinds == {MAIN_PRECISION}:
        return None
    if len(kinds) > 1:
        listed = ", ".join(f"{name} {dtype} / {precision}" for name, (dtype, precision) in found.items())
        return (f"Precision (D9): the compared runs differ in model.torch_dtype / train.precision ({listed}), so the "
                "paired differences between them also differ in precision; the bf16 fallback runs E1, E3 and E4 "
                "with the precision of E2 (RUNBOOK).")
    ((dtype, precision),) = kinds
    return (f"Precision (D9): every compared run ({', '.join(found)}) uses {dtype} / {precision}, not the main "
            "setting float16 / fp16_amp: a recorded deviation (variant_bf16, DESIGN 18.6.4) or a CPU smoke run.")


def limitations(report: Report) -> list[str]:
    total_rank = [name for name, view in report.runs.items()
                  if view.usable and view.config_value("lora", "scaling_rule") == "total_rank"]
    note = (f" A total_rank run is included: {', '.join(total_rank)}." if total_rank else
            " No variant_total_rank_scaling result is included (NOT_RUN), so these comparisons rest on D8 alone.")
    items = list(LIMITATIONS)
    items[1] += note
    precision = precision_note(report)
    if precision is not None:
        items.append(precision)
    return [f"{i}. {text}" for i, text in enumerate(items, 1)]


def comparison_md(report: Report) -> str:
    lines = ["# Run comparison", "",
             f"Created {report.created_utc} by `rebuild/tools/compare_runs.py` (DESIGN 18.6.7-18.6.8). Every "
             "measured value is copied unchanged from the run file named in comparison.csv (source_path, "
             "source_sha256) after a check against its counts within 1e-12. NOT_RUN marks a value that no file "
             "provides; the reasons are listed at the end. Paper numbers appear only in table D.", "",
             "## Runs", "", *runs_table(report), "",
             "## Definitions (D17)", "",
             "- ValidRatio = n_valid / n_total; a response is valid when exactly one candidate title occurs in its "
             "first line.",
             "- HR@1 = n_correct / n_valid (released code `hr`, 0 when n_valid = 0).",
             "- CombinedMetric = n_correct / n_total = HR@1 x ValidRatio (released code `metric`).",
             "- Protocol full: every test sample (N = 95 on the full data). drop_last_bs4 / drop_last_bs8: the first "
             "floor(N/4)*4 / floor(N/8)*8 samples in row order (92 / 88 for N = 95), the sample sets of the released "
             "test loader with batch 4 and 8 and drop_last.",
             "- best_val: the epoch with the most correct validation answers, the later one on ties, fixed before any "
             "test result (D11 (a), the main result). last: the final epoch (D11 (b)).",
             "- Values are written as `value (numerator/denominator)`. ElapsedTime of a training run is "
             "records/train/run_summary.json train_summary.seconds.total: the training, validation and saving time "
             "of its completed epochs plus its test time, carried across --resume, so the idle time between a failure "
             "and its resume is not counted; for E3 (an evaluation run) it is meta.json elapsed_s. WallClock is "
             "meta.json elapsed_s (first start to last end, idle time included) with the number of resumes. "
             "TrainableParams of E3 come from the training run it evaluated.", "",
             "## Table A. Main results (test, best_val epoch, full protocol)", "", *table_ab(report, "best_val"), "",
             "TrainableParams against the calculation (M9 13.8 #13, M10 14.6.3):", "", *trainable_check(report), "",
             "## Table B. Last epoch (test, full protocol)", "", *table_ab(report, "last"), "",
             "## Table C. Auxiliary protocols (test)", "", *table_c(report), "",
             "## Table D. Paper reference values (not our results)", "", *table_d(report), "",
             "## Table E. Paired comparison of our runs", "", *table_e(report), "",
             "## Table F. Gate and cluster summary", "", *table_f(report), "",
             "## NOT_RUN reasons", ""]
    reasons = collect_reasons(report)
    lines += [f"- {reason}" for reason in reasons] or ["- none"]
    lines += ["", "## Limitations (DESIGN 18.6.10)", "", *limitations(report), ""]
    return "\n".join(lines)

# ----------------------------------------------------------------------------
# summary.md (SPEC §19 structure, file values only)
# ----------------------------------------------------------------------------
def _setup_rows(report: Report) -> list[tuple]:
    rows = []
    for role, views in (("run", report.runs.values()), ("export", report.exports.values()),
                        ("cluster", (report.cluster,))):
        for view in views:
            git = dig(view.meta, ("code", "git"), "meta.json") if view.usable else view.unusable()
            commit = git if missing(git) else f"{git.get('commit')} (dirty {git.get('dirty')})"
            rows.append((view.name, role, view.meta.get("status", NOT_RUN), view.meta.get("profile", NOT_RUN),
                         view.config_value("router", "routing_mode"), view.config_value("lora", "num_experts"),
                         view.config_value("lora", "r"), view.config_value("run", "seed"),
                         view.config_value("train", "epochs"), view.config_value("data", "subset"), commit,
                         seconds_text(view.elapsed()), seconds_text(view.wall_clock()), view.resumes(),
                         f"`{view.path}`" if view.path is not None else NOT_RUN))
    return rows


def _metric_lines(report: Report, name: str) -> list[str]:
    view = report.runs[name]
    if not view.usable:
        return [f"- NOT_RUN: {view.problem}"]
    lines = [f"- Run `{view.path}`; epochs best_val {cell(view.epochs['best_val'])}, last {cell(view.epochs['last'])} "
             "(eval/selection.json of the training run)."]
    for ckpt in CHECKPOINTS:
        for protocol in PROTOCOLS:
            row = report.table[(name, ckpt, protocol)]
            lines.append(f"- Test, {ckpt}, {protocol}: ValidRatio {ratio_text(row, 'valid_ratio')}, HR@1 "
                         f"{ratio_text(row, 'hr_at_1')}, CombinedMetric {ratio_text(row, 'combined_metric')}")
    row = report.table[(name, "best_val", "full")]
    lines.append(f"- TrainableParams {cell(row['trainable_params'])}; ElapsedTime {seconds_text(row['elapsed_s'])}; "
                 f"wall clock {wall_clock_text(row['wall_clock_s'], row['resumes'])}; "
                 f"source `{cell(row['source_path'])}` (sha256 {cell(row['source_sha256'])}).")
    return lines


def summary_md(report: Report) -> str:
    g2, g4, cl = report.gates_e2, report.gates_e4, report.clusters
    lines = ["# Experiment summary", "",
             f"Generated {report.created_utc} by `rebuild/tools/compare_runs.py` from the run files only (SPEC §19). "
             "NOT_RUN marks a value that no file provides. No paper number is copied here; the paper reference "
             "values are only in comparison.md table D.", "",
             "## 1. Experiment setup", "",
             *md_table(("Name", "Role", "Status", "Profile", "Routing", "num_moe", "lora_r", "Seed", "Epochs",
                        "Data subset", "git commit", "ElapsedTime", "Wall clock", "Resumes", "Folder"),
                       _setup_rows(report)),
             "", "ElapsedTime of a training run is its compute time (run_summary.json train_summary.seconds.total: "
             "completed epochs and test, idle time before a resume not counted); the other runs and the wall clock use "
             "meta.json elapsed_s.", "",
             "## 2. Baseline reproduction (iLoRA Dynamic, E2)", "", *_metric_lines(report, "E2"),
             "- Paper reference: comparison.md table D (same-protocol values only, differences in samples).", "",
             "## 3. Gate diversity (dynamic routing, test split)", ""]
    if "problem" in g2:
        lines.append(f"- NOT_RUN: {g2['problem']}")
    else:
        lines += [f"- Source `{g2['path']}` (sha256 {g2['sha256']}), export epoch {g2['epoch']}, n {cell(g2['n'])}, "
                  f"K {cell(g2['num_experts'])}.",
                  f"- Mean utilization per expert {_list_text(g2['mean_utilization'])}; argmax ratio "
                  f"{_list_text(g2['argmax_ratio'])} (counts {_list_text(g2['argmax_count'])}).",
                  f"- Normalized entropy: {_list_text(g2['normalized_entropy'])}.",
                  f"- JS divergence to the mean gate (bits): {_list_text(g2['js_to_mean_bits'])}.",
                  f"- Top-1 minus top-2: {_list_text(g2['top1_minus_top2'])}.",
                  f"- Interpretation template: `{cell(g2['interpretation'])}`."]
    lines += ["", "## 4. SASRec clustering", ""]
    if "problem" in cl:
        lines.append(f"- NOT_RUN: {cl['problem']}")
    else:
        lines += [f"- Source `{cl['path']}` (sha256 {cl['sha256']}); unit {cell(cl['unit'])}, n_clusters "
                  f"{cell(cl['n_clusters'])}, fit on {cell(cl['fit_split'])} ({cell(cl['fit_rows'])} rows).",
                  "- Cluster sizes: " + "; ".join(f"{split} {_list_text(size)}" for split, size in cl["sizes"].items())
                  + ".", f"- silhouette_score_train {cell(cl['silhouette_score_train'])}.",
                  "- NMI cluster vs dynamic argmax: " + "; ".join(f"{split} {cell(value)}"
                                                                 for split, value in cl["nmi"].items()) + ".",
                  f"- Reload check passed: {cell(cl['reload_check'])}; max abs z difference to the dynamic export: "
                  f"{_list_text(cl['z_max_abs_diff'])}."]
    lines += ["", "## 5. Cluster-hard routing (E4)", "", *_metric_lines(report, "E4")]
    if "problem" in g4:
        lines.append(f"- Evaluation gates: NOT_RUN: {g4['problem']}")
    else:
        lines.append(f"- Evaluation gates (test, best_val): one-hot rows {g4['one_hot_rows']}/{g4['n']}, argmax count "
                     f"per expert {_list_text(g4['argmax_count'])}; training gate_calls {cell(g4['gate_calls'])}, "
                     f"cluster_lookups {cell(g4['cluster_lookups'])} (run_summary.json).")
    lines += ["", "## 6. Performance comparison", "", "Test, best_val epoch, full protocol (comparison.md table A):", ""]
    for name in report.runs:
        row = report.table[(name, "best_val", "full")]
        lines.append(f"- {row['method']}: CombinedMetric {ratio_text(row, 'combined_metric')}, ValidRatio "
                     f"{ratio_text(row, 'valid_ratio')}, HR@1 {ratio_text(row, 'hr_at_1')}")
    lines += ["", "Paired comparison (comparison.md table E):", ""]
    for p in report.pairs:
        if "problem" in p:
            lines.append(f"- {p['pair']}: NOT_RUN: {p['problem']}")
        else:
            lines.append(f"- {p['pair']}: correct {p['correct_a']} vs {p['correct_b']} of {p['n']} ({p['difference']:+d}), "
                         f"only A {p['only_a']}, only B {p['only_b']}, CombinedMetric A / B - 1 "
                         f"{cell(p['combined_ratio_minus_1'])}")
    lines += ["", "Full tables: comparison.md, comparison.csv; SPEC §20 file index: spec20_index.json.", "",
              "## 7. Limitations", "", *limitations(report), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Output files, the compare run folder and the command line
# ----------------------------------------------------------------------------
def csv_text(report: Report) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in report.rows:
        writer.writerow([cell(row[col]) for col in CSV_COLUMNS])
    return buffer.getvalue()


def write_text(path: Path, text: str) -> str:
    """UTF-8 with LF, written once under a temporary name; token-like strings refused (M0). Returns the sha256."""
    if path.exists():
        raise CompareError(f"{path} exists; comparison files are never overwritten")
    hits = run.find_secrets([text])
    if hits:
        raise run.SecretLeakError(f"refusing to write token-like values to {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)
    return run.content_sha256(path)


def write_outputs(report: Report, out_dir: Path) -> dict[str, str]:
    """comparison.csv, comparison.md, summary.md, then spec20_index.json (which lists the other three)."""
    files = {"comparison.csv": write_text(out_dir / "comparison.csv", csv_text(report)),
             "comparison.md": write_text(out_dir / "comparison.md", comparison_md(report)),
             "summary.md": write_text(out_dir / "summary.md", summary_md(report))}
    e2 = report.runs["E2"]
    e2x = report.exports.get("E2") or open_run("E2", None, "export")
    epochs = {"E2_best_val": export_epoch(e2x, e2) if not e2.usable else e2.epochs["best_val"],
              "E4_best_val": report.runs["E4"].epochs.get("best_val", report.runs["E4"].unusable())}
    index = {"created_utc": report.created_utc,
             "rule": "SPEC §20 names -> files of the runs (DESIGN 18.6.9); the repository results/ folder is not "
                     "written; NOT_RUN when the file does not exist",
             "epochs": {key: cell(value) if missing(value) else value for key, value in epochs.items()},
             "files": spec20_index(report.runs, report.exports, report.cluster, out_dir)}
    run.write_json_atomic(out_dir / "spec20_index.json", index)
    files["spec20_index.json"] = run.content_sha256(out_dir / "spec20_index.json")
    return files


@contextlib.contextmanager
def _compare_logging(run_dir: Path) -> Iterator[logging.Logger]:
    """Console + ``run.log`` of the compare folder; the logger's previous state comes back afterwards."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")]
    saved = (logger.level, logger.propagate)
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(saved[0])
        logger.propagate = saved[1]


def _resolve(value: Optional[PathLike], repo_root: Path) -> Optional[Path]:
    """Run folder value -> path (M0 rule: absolute or relative to the repository root); empty -> not given."""
    if value is None or not str(value).strip():
        return None
    return run.resolve_path(value, repo_root)


def run_compare(runs: Mapping[str, Optional[PathLike]], exports: Optional[Mapping[str, Optional[PathLike]]] = None,
                cluster: Optional[PathLike] = None, *, root: PathLike = run.RUNS_ROOT,
                repo_root: PathLike = run.REPO_ROOT, argv: Optional[Sequence[str]] = None) -> Path:
    """Write ``<root>/compare_<UTC>/`` (``meta.json``, ``run.log`` and the four outputs) and return it (RUNBOOK R14).

    ``runs``/``exports`` map an experiment name (E1-E4 or a variant name) to a run folder; an empty or missing
    folder makes that experiment ``NOT_RUN``. A ``CompareError`` leaves ``meta.json`` with status ``failed``.
    """
    repo_root = Path(repo_root)
    for name in [*runs, *(exports or {})]:
        if not NAME_RE.fullmatch(str(name)):
            raise CompareError(f"run name {name!r} must match {NAME_RE.pattern}")
    run_paths = {name: _resolve(path, repo_root) for name, path in runs.items()}
    export_paths = {name: _resolve(path, repo_root) for name, path in (exports or {}).items()}
    cluster_path = _resolve(cluster, repo_root)
    inputs = {"runs": {k: str(v) if v else None for k, v in run_paths.items()},
              "exports": {k: str(v) if v else None for k, v in export_paths.items()},
              "cluster": str(cluster_path) if cluster_path else None}
    run_dir = run.create_run_dir(PROFILE, root=Path(root))
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
    run.start_meta(run_dir, profile=PROFILE, argv=list(sys.argv if argv is None else argv), config_hash=digest,
                   overrides={}, seed=None, extra={"command": PROFILE, "compare_inputs": inputs})
    with _compare_logging(run_dir) as log:
        log.info("run_dir=%s", run_dir)
        with run.tracked_run(run_dir):
            try:
                report = build_report(run_paths, export_paths, cluster_path)
                files = write_outputs(report, run_dir)
                reasons = collect_reasons(report)
                run.update_meta(run_dir, {"compare": {"files": files, "not_run_reasons": reasons}})
            except BaseException as exc:  # noqa: BLE001 - logged, then tracked_run marks failed
                log.error("compare failed: %s: %s", type(exc).__name__, exc)
                raise
        for reason in reasons:
            log.info("NOT_RUN: %s", reason)
        log.info("compare completed: %s", ", ".join(files))
    return run_dir


def _named(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(f"expected NAME=RUN_DIR (NAME like E2 or V_released_lr), got {text!r}")
    return name, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_runs.py",
        description="Comparison tables of rebuild runs (rebuild/DESIGN.md 18.6.7-18.6.9): comparison.csv, "
                    "comparison.md, spec20_index.json and summary.md in a new runs/rebuild/compare_<UTC>/ folder.")
    parser.add_argument("--run", dest="runs", action="append", type=_named, default=[], metavar="NAME=RUN_DIR",
                        help="training run (E1, E2, E4, variants) or eval run (E3) of an experiment; repeatable")
    parser.add_argument("--export", dest="exports", action="append", type=_named, default=[], metavar="NAME=RUN_DIR",
                        help="export run of an experiment (E2: gate summary and SPEC 20 files); repeatable")
    parser.add_argument("--cluster", metavar="RUN_DIR", help="cluster run (its cluster/ folder)")
    parser.add_argument("--root", metavar="DIR", help="folder that receives compare_<UTC>/ (default runs/rebuild)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.runs:
        parser.error("at least one --run NAME=RUN_DIR is required")
    for flag, pairs in (("--run", args.runs), ("--export", args.exports)):
        names = [name for name, _ in pairs]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            parser.error(f"{flag} names given more than once: {twice}")
    full_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    root = run.resolve_path(args.root) if args.root else run.RUNS_ROOT
    try:
        out = run_compare(dict(args.runs), dict(args.exports), args.cluster, root=root, argv=full_argv)
    except CompareError as exc:
        print(f"compare error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
