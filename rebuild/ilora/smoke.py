"""M13 smoke 계획과 검사 (DESIGN.md §17.6.6-§17.6.8, §17.6.12 제안 2; §3.1 시험 기반; D14, D16, D22).

``main.py smoke --config <profile>`` runs one of three plans, chosen from the profile values only
(``smoke_plan``, §17.6.6 계획 선택):

======== ====================================== ==========================================================
plan     profile                                steps (sub-run folders ``steps/<id>_<name>_<UTC>/``)
======== ====================================== ==========================================================
cpu      smoke_cpu (tiny random LLaMA, CPU)     S0 check-env, S1 train (E2), S2 eval (reload), S3 export,
                                                S4 cluster, S5 train (E4), S6 eval (E3), S7 train (E1),
                                                S8 report (§17.6.6)
gpu100   smoke_gpu_100 (A100, VESSL_REQUIRED)   G0-G5 as S0-S5, G6 train (E1), G7 report (§17.6.7)
overfit  overfit_gpu (A100, VESSL_REQUIRED)     O0 check-env, O1 train, O2 overfit generation check,
                                                O3 report (§17.6.8)
======== ====================================== ==========================================================

Every command step goes through M0 ``commands.run_command`` with the preparation of ``main.py``: the config is
``load_config(profile, smoke --set + step --set)`` and the run folder is made under ``<smoke run>/steps``
(§17.6.12 연결, 제안 3). After a step, the checks of its row of the 17.6.6 table (17.6.7 and 17.6.8 for the
GPU plans) are evaluated from the files the step wrote and kept as ``{id, step, description, passed,
measured, expected}``; ``passed`` None marks a value that is only recorded (TBD, no judgement, [SPEC §0.11]).
Expected counts come from the M1, M4 and M10 functions on the step's config (``load_dataset``,
``build_batch_plan``, ``step_counts``), which give the computed values of 17.6.6-17.6.8 for the profiles.
``smoke_report.json`` (the report step) holds the plan, every step with its folder, status and seconds, the
checks and ``passed``; ``main.py smoke`` exits non-zero when ``passed`` is false (M0 ``run_command``).
Accuracy is never judged: the CPU model is random (§17.6.6), the overfit plan reports only the number of
matching first lines (§17.6.8).
"""
from __future__ import annotations

import dataclasses
import functools
import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Mapping, Optional, Sequence, Union

from . import run
from .config import load_config

if TYPE_CHECKING:   # type names only: M5 tokenizer, M1 data, M4 templates, M0 config
    from .batching import TokenizerBundle
    from .data import LoadedData
    from .prompts import TemplateSet
    from .sections import Config

logger = logging.getLogger("ilora.smoke")

PathLike = Union[str, os.PathLike]
Plan = Literal["cpu", "gpu100", "overfit"]


class SmokeError(ValueError):
    """The profile cannot be smoked (``data.subset`` is not ``smoke``) or the smoke folder is not usable (§17.6.6)."""


# ----------------------------------------------------------------------------
# Constants (§17.2, §17.6.6-§17.6.8)
# ----------------------------------------------------------------------------
PLANS: Final = ("cpu", "gpu100", "overfit")                         # §17.6.6 계획 선택
PLAN_PREFIX: Final = {"cpu": "S", "gpu100": "G", "overfit": "O"}   # step ids S0-S8, G0-G7, O0-O3
REPORT_FILE: Final = "smoke_report.json"                            # §17.2 출력, S8
REPORT_FORMAT: Final = "ilora-rebuild-smoke-report-v1"              # smoke_report.json "format" [결정]
STEPS_DIR: Final = "steps"                                          # §17.6.6: steps/S<k>_<name>_<UTC>/
OVERFIT_LOSS_RATIO: Final = 0.1          # §17.6.8 통과 기준 2 [결정]: last-epoch mean loss <= first x 0.1
CHECKPOINT_FILES: Final = ("trainable.pt", "train_state.pt", "STATE_SAVED", "val_metrics.json", "COMPLETE")  # §14.6.6
PROJECTOR_KEYS: Final = 4                # M6 §10.6.1: mlp_proj.0 and mlp_proj.2, weight and bias each
SCALER_KEYS: Final = ("_growth_tracker", "backoff_factor", "growth_factor", "growth_interval", "scale")  # §14.6.6
ANALYSIS_FIGURES: Final = ("gate_heatmap", "expert_mean_utilization", "argmax_expert_distribution",
                           "entropy_distribution")                  # M12 §16.6.7 (every analysed split)
TEN_SEQUENCES_FIGURE: Final = "gate_ten_sequences"                  # M12 §16.6.7 (test only)
EXPORT_SPLIT_NAMES: Final = ("train", "validation", "test")         # M12 representations_{name}.npz [SPEC §4]


# ----------------------------------------------------------------------------
# Report values (§17.6.6 S8, §17.6.12 제안 2)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SmokeStep:
    """One step of a smoke run: its sub-run folder (None for in-process steps), ``status`` completed / failed /
    not_run (a step it needs did not complete), seconds and the error text of a failure."""
    id: str
    name: str
    command: str
    sets: tuple[str, ...]
    run_dir: Optional[str]
    status: str
    seconds: Optional[float]
    error: Optional[str]


@dataclass(frozen=True)
class SmokeCheck:
    """One check of the 17.6.6-17.6.8 tables: ``passed`` True/False, or None for a value recorded without a
    judgement (TBD); ``measured`` and ``expected`` are JSON values."""
    id: str
    step: str
    description: str
    passed: Optional[bool]
    measured: Any
    expected: Any = None


@dataclass(frozen=True)
class SmokeReport:
    """``SmokeReport(plan, run_dir, steps, checks, passed)`` (§17.6.12 제안 2), also ``smoke_report.json``."""
    plan: str
    run_dir: Path
    steps: tuple[SmokeStep, ...]
    checks: tuple[SmokeCheck, ...]
    passed: bool

    def step(self, step_id: str) -> SmokeStep:
        """The step ``step_id`` (``"S1"``); ``KeyError`` when the plan has no such step."""
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"smoke report of plan {self.plan} has no step {step_id!r}")

    def step_dir(self, step_id: str) -> Path:
        """Run folder of a command step (``S1`` -> ``steps/S1_train_<UTC>``); ``SmokeError`` when it has none."""
        step = self.step(step_id)
        if step.run_dir is None:
            raise SmokeError(f"step {step_id} ({step.status}) has no run folder")
        return Path(step.run_dir)

    def check(self, check_id: str) -> SmokeCheck:
        """The check ``check_id`` (``"S1.record_rows"``); ``KeyError`` when it was not evaluated."""
        for check in self.checks:
            if check.id == check_id:
                return check
        raise KeyError(f"smoke report has no check {check_id!r}")

    def failed_checks(self) -> list[str]:
        return [c.id for c in self.checks if c.passed is False]


# ----------------------------------------------------------------------------
# Plans and steps (§17.6.6 단계 표, §17.6.7 G0-G7, §17.6.8 O0-O3)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StepSpec:
    """One row of a plan: ``command`` is an M0 command, ``generation`` (O2, in process) or ``report``; ``kind``
    selects the checks; ``needs`` are the step indices that must have completed; ``sets`` are the step's
    ``--set`` texts, whose fields ``{train}`` (E2 training run), ``{export_epoch}`` (``<export>/records/export/
    epoch<a>``, a = best_val_epoch) and ``{cluster_dir}`` (``<cluster run>/cluster``) come from earlier steps."""
    index: int
    name: str
    command: str
    kind: str
    needs: tuple[int, ...] = ()
    sets: tuple[str, ...] = ()


_E4_SETS: Final = ("router.routing_mode=cluster_hard", "cluster.artifact_dir={cluster_dir}", "train.epochs=1")
_E1_SETS: Final = ("lora.num_experts=1", "router.routing_mode=none", "train.epochs=1")
_COMMON_STEPS: Final = (
    StepSpec(0, "check_env", "check-env", "check_env"),
    StepSpec(1, "train", "train", "train_e2", (0,)),
    StepSpec(2, "eval", "eval", "eval_reload", (1,), ("eval.source_run={train}",)),
    StepSpec(3, "export", "export", "export", (1,), ("records.source_run={train}",)),
    StepSpec(4, "cluster", "cluster", "cluster", (3,), ("cluster.dynamic_export_dir={export_epoch}",)),
    StepSpec(5, "train_e4", "train", "train_e4", (4,), _E4_SETS),
)
PLAN_STEPS: Final = {
    "cpu": _COMMON_STEPS + (
        StepSpec(6, "eval_e3", "eval", "eval_e3", (1,), ("eval.source_run={train}", "prompt.eval_prompt=text_only")),
        StepSpec(7, "train_e1", "train", "train_e1", (0,), _E1_SETS),
        StepSpec(8, "report", "report", "report")),
    "gpu100": _COMMON_STEPS + (                          # §17.6.7: no E3 step, G6 = the E1 path
        StepSpec(6, "train_e1", "train", "train_e1", (0,), _E1_SETS),
        StepSpec(7, "report", "report", "report")),
    "overfit": (
        StepSpec(0, "check_env", "check-env", "check_env"),
        StepSpec(1, "train", "train", "train_overfit", (0,)),
        StepSpec(2, "generation", "generation", "generation", (1,)),
        StepSpec(3, "report", "report", "report")),
}

_TRAIN_CHECKS: Final = ("status", "checkpoint_files", "trainable_keys", "optimizer_groups", "adam_state",
                        "scaler_rng", "frozen_sasrec", "record_rows", "window_sizes", "hybrid", "plan_digest",
                        "gate_calls", "cluster_lookups", "projector_grad", "consecutive_skips", "eval_folders")
KIND_CHECKS: Final = {
    "check_env": ("status", "checks", "environment", "input_hashes", "sasrec_reference", "tokenizer"),
    "train_e2": _TRAIN_CHECKS,
    "eval_reload": ("status", "reproduction", "records_sha256", "checkpoints"),
    "export": ("status", "export_epochs", "crosscheck", "npz_rows", "analysis", "manifest"),
    "cluster": ("status", "fit_train_only", "no_empty_cluster", "joblib_reload", "z_vs_export", "dynamic_items"),
    "train_e4": _TRAIN_CHECKS + ("live_check", "one_hot_gates"),
    "eval_e3": ("status", "text_only", "metrics_files", "not_comparable"),
    "train_e1": _TRAIN_CHECKS + ("gate_one",),
    "train_overfit": _TRAIN_CHECKS + ("all_hybrid", "loss_ratio"),
    "generation": ("checkpoint", "first_line_match"),
    "report": (),
}


def smoke_plan(cfg: "Config") -> Plan:
    """§17.6.6 계획 선택 from profile values only: ``overfit`` when ``data.train_rows`` is not empty, ``cpu`` for
    ``model.backbone == tiny_random``, ``gpu100`` otherwise. A profile whose ``data.subset`` is not ``smoke`` is a
    ``SmokeError``: the smoke never runs on the full data."""
    if cfg.data.subset != "smoke":
        raise SmokeError(f"smoke needs data.subset = smoke (smoke_cpu, smoke_gpu_100, overfit_gpu), got "
                         f"{cfg.data.subset!r}: the smoke never runs on the full data (17.6.6)")
    if cfg.data.train_rows:
        return "overfit"
    if cfg.model.backbone == "tiny_random":
        return "cpu"
    return "gpu100"


def required_checks(plan: str, cfg: "Config") -> tuple[str, ...]:
    """Check ids every complete report of ``plan`` holds (S1 adds ``batch_invariance`` when
    ``eval.batch_invariance_check`` is on, G0 adds ``gpu`` for a CUDA profile); ``SmokeError`` for an unknown plan."""
    if plan not in PLAN_STEPS:
        raise SmokeError(f"unknown smoke plan {plan!r}, expected one of {PLANS}")
    prefix, out = PLAN_PREFIX[plan], []
    for spec in PLAN_STEPS[plan]:
        names = list(KIND_CHECKS[spec.kind])
        if spec.kind == "check_env" and cfg.model.device.startswith("cuda"):
            names.append("gpu")
        if spec.kind == "train_e2" and cfg.eval.batch_invariance_check:
            names.append("batch_invariance")
        out.extend(f"{prefix}{spec.index}.{name}" for name in names)
    return tuple(out)


# ----------------------------------------------------------------------------
# O2: overfit generation check (§17.6.8)
# ----------------------------------------------------------------------------
def overfit_generation_check(model: Any, bundle: "TokenizerBundle", templates: "TemplateSet", data: "LoadedData",
                             cfg: "Config") -> dict:
    """Does the generation path reproduce what the overfit training fitted? (§17.6.8 O2, 통과 기준 3)

    ``model`` already holds the last checkpoint: ``run_smoke`` builds a new model with M9 ``build_model`` and
    loads ``checkpoints/epoch{E-1}`` with M10 ``load_model_state`` (the 17.6.12 제안 2 signature has no
    checkpoint argument). Every train sample (the 16 ``data.train_rows`` windows of ``overfit_gpu``) gets the keys
    of the last training epoch E-1 = ``train.epochs - 1``: candidates ``candidates_for(s, main_seed, epoch=E-1)``
    (M2) and template ``choose_template(cfg.prompt, seed, n, s, E-1)`` (M4); M5 ``EvalBatches`` takes only
    evaluation keys, so the public functions are called directly: ``build_prompt`` (hybrid, D4) ->
    ``encode_eval`` -> ``check_token_counts`` -> ``collate_eval`` (left padding, D22) in groups of
    ``eval.batch_size`` in row order -> ``model.generate`` inside ``autocast_for(train.precision)`` and
    ``torch.random.fork_rng`` with the model in eval mode (restored afterwards) -> M11 ``check_output``,
    ``trim_generated``, ``decode_generated``, ``first_line``. A sample matches when ``match_norm(first_line) ==
    match_norm(target title)``. Returns the match count and the per-sample first lines; no accuracy numbers.
    """
    import torch

    from .batching import check_token_counts, collate_eval, encode_eval
    from .candidates import candidates_for, match_norm
    from .evaluate import check_output, decode_generated, first_line, trim_generated
    from .precision import autocast_for
    from .prompts import build_prompt, check_catalog_titles, choose_template

    samples = tuple(data.train)
    if not samples:
        raise SmokeError("overfit generation check: the data has no train samples")
    epoch = int(cfg.train.epochs) - 1                       # the keys of the last training epoch (§17.6.8)
    catalog, ids = data.catalog, bundle.ids
    check_catalog_titles(catalog)                           # M4 §8.6.2; the M5 datasets do this for their own path
    n_templates = len(templates.templates)
    prepared = []
    for s in samples:
        cands = candidates_for(s, main_seed=cfg.run.seed, epoch=epoch, cfg=cfg.candidates)
        template_id = choose_template(cfg.prompt, cfg.run.seed, n_templates, s, epoch)
        encoded = encode_eval(bundle, build_prompt(s, cands, catalog, templates, template_id, True))
        check_token_counts(bundle, encoded, hybrid=True, len_seq=len(s.history), prompt_len=len(encoded) - 1,
                           n_bos=1, max_seq_len=cfg.batching.max_seq_len, where=f"overfit check {s.sample_id}")
        prepared.append((s, cands, template_id, encoded))
    dev = model.device
    batch_size = int(cfg.eval.batch_size)
    forked = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    results: list[dict] = []
    n_batches, was_training, started = 0, model.training, time.perf_counter()
    model.eval()                                            # LoRA dropout off, SASRec eval (D12)
    try:
        with torch.random.fork_rng(devices=forked):
            for start in range(0, len(prepared), batch_size):
                group = prepared[start:start + batch_size]
                batch = collate_eval([g[0] for g in group], [g[1] for g in group], [g[2] for g in group],
                                     [g[3] for g in group], catalog, bundle, hybrid=True)
                with autocast_for(cfg.train.precision, dev):                  # D9, M0 precision tools
                    out = model.generate(batch)
                n_batches += 1
                check_output(out, batch_size=len(group), num_experts=cfg.lora.num_experts,
                             z_dim=int(model.sasrec.hidden_size), max_new_tokens=cfg.model.gen_max_new_tokens,
                             eos_id=ids.eos, pad_id=ids.pad, rowsum_atol=cfg.records.gate_rowsum_atol,
                             sample_ids=batch.sample_ids)
                rows = [trim_generated(row, eos_id=ids.eos) for row in out.new_token_ids.tolist()]
                texts = decode_generated(bundle, rows)
                for r, (s, _, template_id, _) in enumerate(group):
                    line = first_line(texts[r])
                    target_title = batch.target_titles[r]
                    results.append({"sample_id": s.sample_id, "user_id": s.user_id, "target_id": s.target,
                                    "template_id": template_id, "target_title": target_title, "first_line": line,
                                    "match": match_norm(line) == match_norm(target_title),
                                    "n_new_tokens": len(rows[r]), "eos": ids.eos in rows[r]})
    finally:
        model.train(was_training)
    n_match = sum(1 for r in results if r["match"])
    return {"epoch": epoch, "n": len(results), "n_match": n_match, "all_match": n_match == len(results),
            "batch_size": batch_size, "n_batches": n_batches, "hybrid": True, "samples": results,
            "seconds": round(time.perf_counter() - started, 3)}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _plain(obj: object) -> object:
    """JSON values for the report: mappings, sequences, numbers (numpy included), paths, None."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj) if isinstance(obj, (set, frozenset)) else obj
        return [_plain(v) for v in items]
    item = getattr(obj, "item", None)                       # numpy and 0-dim torch scalars
    if callable(item):
        return _plain(item())
    return str(obj)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_value(path: Path) -> str:
    """A path as a ``--set`` value: a JSON string of the POSIX form (valid YAML, quotes and backslashes safe)."""
    return json.dumps(Path(path).as_posix())


def _torch_load(path: Path) -> Any:
    """``torch.load(weights_only=True)`` on CPU without torch 2.0.0's TypedStorage warning (M10 files)."""
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
        return torch.load(path, map_location="cpu", weights_only=True)


def _release_memory() -> None:
    """Drop the previous step's model before the next one is built (one A100 holds one LLaMA run at a time)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:   # pragma: no cover - torch is a requirement
        pass


def _longest_run(flags: Sequence[bool]) -> int:
    longest = streak = 0
    for flag in flags:
        streak = streak + 1 if flag else 0
        longest = max(longest, streak)
    return longest


# ----------------------------------------------------------------------------
# Reading one step's run folder
# ----------------------------------------------------------------------------
class _RunFiles:
    """Cached readers of one run folder (M10-M12 formats). A failed read is not cached, so it surfaces as a failed
    check in every check that needs the value."""

    def __init__(self, folder: Path, cfg: "Config") -> None:
        self.folder = Path(folder)
        self.cfg = cfg
        self._states: dict[int, dict] = {}

    @functools.cached_property
    def meta(self) -> dict:
        return run.read_meta(self.folder)

    def _rows(self, name: str, fields: Sequence[str]) -> list[dict]:
        from .records import read_rows

        return read_rows(self.folder / "records" / "train" / name, fields)

    @functools.cached_property
    def micro(self) -> list[dict]:
        from .records import MICRO_FIELDS

        return self._rows("micro_batches.csv", MICRO_FIELDS)

    @functools.cached_property
    def steps(self) -> list[dict]:
        from .records import step_fields

        return self._rows("optimizer_steps.csv", step_fields(self.cfg.lora.num_experts))

    @functools.cached_property
    def epochs(self) -> list[dict]:
        from .records import EPOCH_FIELDS

        return self._rows("epochs.csv", EPOCH_FIELDS)

    @functools.cached_property
    def run_summary(self) -> dict:
        return _read_json(self.folder / "records" / "train" / "run_summary.json")

    @functools.cached_property
    def selection(self) -> dict:
        return _read_json(self.folder / "eval" / "selection.json")

    @functools.cached_property
    def tested(self) -> tuple[int, ...]:
        """Epochs with a test evaluation: best_val and last ascending, once when equal (M10 §14.6.7)."""
        from .train import epochs_to_test

        return epochs_to_test(int(self.selection["best_val_epoch"]), int(self.selection["last_epoch"]))

    def ckpt(self, epoch: int) -> Path:
        return self.folder / "checkpoints" / f"epoch{epoch}"

    def train_state(self, epoch: int) -> dict:
        """The parts of ``train_state.pt`` the checks read, from one load per epoch (M10 §14.6.6). The Adam moments
        are not kept: a Llama-2 epoch holds about 283 MiB of them and the overfit plan has 25 epochs."""
        if epoch not in self._states:
            import torch

            state = _torch_load(self.ckpt(epoch) / "train_state.pt")
            opt = state["optimizer"]
            have = set(opt["state"])
            self._states[epoch] = {
                "groups": [pg.get("name") for pg in opt["param_groups"]],
                "group_param_names": list(state["group_param_names"]),
                "params": {pg.get("name"): len(pg["params"]) for pg in opt["param_groups"]},
                "with_state": {pg.get("name"): sum(1 for i in pg["params"] if i in have) for pg in opt["param_groups"]},
                "scaler_keys": sorted(state["scaler"]),
                "rng_cpu": isinstance(state["rng"].get("cpu"), torch.Tensor),
                "rng_cuda": state["rng"].get("cuda") is not None,
                "sasrec_state_digest": state["sasrec_state_digest"],
                "frozen_checksum": dict(state["frozen_checksum"]),
                "frozen_checksum_at_save": dict(state["frozen_checksum_at_save"]),
            }
        return self._states[epoch]

    def trainable(self, epoch: int) -> dict:
        from .train import read_trainable_state      # M10 §14.6.6: markers and sha256 checked

        return read_trainable_state(self.ckpt(epoch))

    def eval_folders(self) -> list[Path]:
        """Completed evaluation folders ``eval/{ckpt_tag}/{split}/`` (``metrics.json`` is written last, §15.6.8)."""
        return [p.parent for p in sorted(self.folder.glob("eval/*/*/metrics.json"))]

    @staticmethod
    def metrics(folder: Path) -> dict:
        return _read_json(Path(folder) / "metrics.json")

    @staticmethod
    def records(folder: Path) -> list:
        from .records import read_eval_records

        return read_eval_records(Path(folder) / "records.jsonl")


# ----------------------------------------------------------------------------
# The smoke run
# ----------------------------------------------------------------------------
class _SmokeRun:
    """Runs the steps of one plan in order, evaluates their checks and writes ``smoke_report.json``."""

    def __init__(self, plan: str, profile_path: Path, overrides: Sequence[str], run_dir: Path, *, repo_root: Path,
                 llm_path: Optional[Path], tokenizer_dir: Optional[Path], base_cfg: "Config") -> None:
        self.plan, self.prefix = plan, PLAN_PREFIX[plan]
        self.profile_path, self.overrides = Path(profile_path), list(overrides)
        self.run_dir, self.steps_root, self.repo_root = Path(run_dir), Path(run_dir) / STEPS_DIR, Path(repo_root)
        self.llm_path, self.tokenizer_dir = llm_path, tokenizer_dir
        self.base_cfg = base_cfg
        self.steps: list[SmokeStep] = []
        self.checks: list[SmokeCheck] = []
        self.status: dict[int, str] = {}
        self.folders: dict[str, Path] = {}       # step kind -> run folder of a completed step
        self.fields: dict[str, str] = {}         # {train}, {export_epoch}, {cluster_dir} of the step --set texts

    def sid(self, index: int) -> str:
        return f"{self.prefix}{index}"

    @functools.cached_property
    def data(self) -> "LoadedData":
        """M1 samples of the smoke config (every step shares the ``data`` section)."""
        from .data import load_dataset

        return load_dataset(self.base_cfg, self.repo_root)

    # -- checks ---------------------------------------------------------------
    def _add(self, check: SmokeCheck) -> None:
        self.checks.append(check)
        state = {True: "passed", False: "FAILED", None: "recorded"}[check.passed]
        logger.log(logging.WARNING if check.passed is False else logging.INFO, "smoke check %s: %s%s", check.id,
                   state, f" (measured {check.measured!r}, expected {check.expected!r})" if check.passed is False else "")

    def check(self, sid: str, name: str, description: str, fn: Callable[[], tuple]) -> None:
        """``fn() -> (measured, expected[, passed])``; without ``passed`` the two JSON values must be equal. An
        exception inside ``fn`` is a failed check carrying the error text."""
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - every failure is reported as a check, never swallowed
            self._add(SmokeCheck(f"{sid}.{name}", sid, description, False, f"error: {type(exc).__name__}: {exc}"))
            return
        measured, expected = _plain(out[0]), _plain(out[1])
        passed = bool(out[2]) if len(out) > 2 else measured == expected
        self._add(SmokeCheck(f"{sid}.{name}", sid, description, passed, measured, expected))

    def record(self, sid: str, name: str, description: str, fn: Callable[[], object]) -> None:
        """A value recorded without a judgement (TBD, [SPEC §0.11])."""
        try:
            measured = _plain(fn())
        except Exception as exc:  # noqa: BLE001 - a missing value is recorded as such
            measured = f"error: {type(exc).__name__}: {exc}"
        self._add(SmokeCheck(f"{sid}.{name}", sid, description, None, measured))

    # -- steps ----------------------------------------------------------------
    def _add_step(self, spec: StepSpec, sets: Sequence[str], folder: Optional[Path], status: str,
                  seconds: Optional[float], error: Optional[str]) -> None:
        sid = self.sid(spec.index)
        self.status[spec.index] = status
        self.steps.append(SmokeStep(sid, spec.name, spec.command, tuple(sets), None if folder is None else str(folder),
                                    status, seconds, error))
        logger.log(logging.INFO if status == "completed" else logging.WARNING, "smoke step %s %s (%s): %s%s%s", sid,
                   spec.name, spec.command, status, f" in {seconds:.1f} s" if seconds is not None else "",
                   f": {error}" if error else "")

    def _find_folder(self, sid: str, name: str) -> Optional[Path]:
        found = sorted(p for p in self.steps_root.glob(f"{sid}_{name}_*") if p.is_dir())
        return found[-1] if found else None

    def run_step(self, spec: StepSpec) -> None:
        sid = self.sid(spec.index)
        blocked = [self.sid(i) for i in spec.needs if self.status.get(i) != "completed"]
        if blocked:
            self._add_step(spec, spec.sets, None, "not_run", None, f"needs {blocked}, which did not complete")
            return
        try:
            sets = tuple(text.format(**self.fields) for text in spec.sets)
        except KeyError as exc:
            self._add_step(spec, spec.sets, None, "failed", None, f"no value for {exc} from the earlier steps")
            return
        logger.info("smoke step %s %s: %s %s", sid, spec.name, spec.command, " ".join(sets))
        cfg, folder, result, error = None, None, None, None
        started = time.perf_counter()
        try:
            if spec.kind == "generation":
                cfg = self.base_cfg
                result = self._generation(cfg, self.folders["train_overfit"])
            else:
                from .commands import run_command

                texts = [*self.overrides, *sets]
                cfg, _ = load_config(self.profile_path, texts)
                folder = run_command(spec.command, cfg, texts, profile=f"{sid}_{spec.name}", root=self.steps_root,
                                     repo_root=self.repo_root, llm_path=self.llm_path, tokenizer_dir=self.tokenizer_dir,
                                     profile_path=self.profile_path)
        except Exception as exc:  # noqa: BLE001 - the step fails, the smoke goes on with the independent steps
            error = f"{type(exc).__name__}: {exc}"
            folder = self._find_folder(sid, spec.name) if spec.kind != "generation" else None
        finally:
            _release_memory()
        seconds = round(time.perf_counter() - started, 3)
        self._add_step(spec, sets, folder, "failed" if error else "completed", seconds, error)
        if error:
            return
        if folder is not None:
            self.folders[spec.kind] = folder
        try:
            self._update_fields(spec, folder)
        except Exception as exc:  # noqa: BLE001 - later steps then fail with the missing field
            self._add(SmokeCheck(f"{sid}.fields", sid, "values for the next steps' --set", False,
                                 f"error: {type(exc).__name__}: {exc}"))
        try:
            _CHECKERS[spec.kind](self, sid, cfg, folder, result)
        except Exception as exc:  # noqa: BLE001 - a broken checker is a failed check, not a crash
            self._add(SmokeCheck(f"{sid}.checks", sid, "the checks of this step ran to the end", False,
                                 f"error: {type(exc).__name__}: {exc}"))

    def _update_fields(self, spec: StepSpec, folder: Optional[Path]) -> None:
        if spec.kind == "train_e2":
            self.fields["train"] = _path_value(folder)
        elif spec.kind == "export":                 # <S3>/records/export/epoch<a>, a = best_val_epoch (R11)
            best = int(_read_json(self.folders["train_e2"] / "eval" / "selection.json")["best_val_epoch"])
            self.fields["export_epoch"] = _path_value(Path(folder) / "records" / "export" / f"epoch{best}")
        elif spec.kind == "cluster":
            self.fields["cluster_dir"] = _path_value(Path(folder) / "cluster")

    def _generation(self, cfg: "Config", train_folder: Path) -> dict:
        """O2: a new model with the last checkpoint of O1 (M10 ``load_model_state``), then the generation check."""
        from .batching import load_tokenizer
        from .commands import llm_path_for, tokenizer_dir_for
        from .model import build_model
        from .prompts import load_templates
        from .train import load_model_state

        last = int(_read_json(train_folder / "eval" / "selection.json")["last_epoch"])
        llm = llm_path_for(cfg, self.llm_path)
        tokenizer = tokenizer_dir_for(cfg, llm, self.tokenizer_dir)
        if tokenizer is None:
            raise SmokeError("tokenizer folder not available (llama2_7b: LLM_PATH, tiny_random: ILORA_TEST_TOKENIZER)")
        bundle = load_tokenizer(Path(tokenizer), cfg.batching.tokenizer_sha256)
        templates = load_templates(run.resolve_path(cfg.prompt.template_path, self.repo_root))
        model = build_model(cfg, bundle, self.data.catalog, llm_path=llm)
        ckpt = train_folder / "checkpoints" / f"epoch{last}"
        sha = load_model_state(ckpt, model)
        result = overfit_generation_check(model, bundle, templates, self.data, cfg)
        return {**result, "checkpoint": {"dir": str(ckpt), "epoch": last, "trainable_sha256": sha}}

    # -- report ---------------------------------------------------------------
    def finish(self, spec: StepSpec, *, started_utc: str, started: float) -> SmokeReport:
        sid = self.sid(spec.index)
        present = {c.id for c in self.checks}
        missing = [cid for cid in required_checks(self.plan, self.base_cfg) if cid not in present]
        if missing:
            self._add(SmokeCheck(f"{sid}.required_checks", sid, "every check of the plan was evaluated", False,
                                 {"missing": missing}, {"missing": []}))
        passed = (all(s.status == "completed" for s in self.steps) and not missing
                  and all(c.passed is not False for c in self.checks))
        self.steps.append(SmokeStep(sid, spec.name, spec.command, (), str(self.run_dir), "completed", None, None))
        report = SmokeReport(self.plan, self.run_dir, tuple(self.steps), tuple(self.checks), passed)
        seconds = round(time.perf_counter() - started, 3)
        document = {
            "format": REPORT_FORMAT, "plan": self.plan, "passed": passed, "run_dir": str(self.run_dir),
            "profile_path": str(self.profile_path), "overrides": list(self.overrides),
            "started_utc": started_utc, "ended_utc": run.utc_now().isoformat(), "seconds": seconds,
            "counts": {"steps": len(report.steps), "steps_completed": sum(s.status == "completed" for s in report.steps),
                       "checks": len(report.checks), "passed": sum(c.passed is True for c in report.checks),
                       "failed": sum(c.passed is False for c in report.checks),
                       "recorded": sum(c.passed is None for c in report.checks)},
            "failed_steps": [s.id for s in report.steps if s.status != "completed"],
            "failed_checks": report.failed_checks(),
            "steps": [_plain(s) for s in report.steps],
            "checks": [_plain(c) for c in report.checks],
        }
        run.write_json_atomic(self.run_dir / REPORT_FILE, document)
        if (self.run_dir / "meta.json").is_file():          # main.py smoke: run_command made the folder
            run.update_meta(self.run_dir, {"smoke": {"plan": self.plan, "passed": passed, "report": REPORT_FILE,
                                                     "failed_steps": document["failed_steps"],
                                                     "failed_checks": document["failed_checks"]}})
        logger.info("smoke %s %s: %d checks, %d failed, %d recorded; report %s", self.plan,
                    "passed" if passed else "FAILED", len(report.checks), document["counts"]["failed"],
                    document["counts"]["recorded"], self.run_dir / REPORT_FILE)
        return report


# ----------------------------------------------------------------------------
# Checks: check-env (S0, G0, O0)
# ----------------------------------------------------------------------------
def _env_report_part(folder: Path, name: str) -> dict:
    doc = _read_json(folder / "check_env.json")
    return next(c for c in doc["checks"] if c["check"] == name)["result"]


def _check_env_step(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """§17.6.6 S0 / §17.6.7 G0: M0 environment check (GPU only for a CUDA profile), input hashes,
    ``candidates.self_check()``, SASRec ``verify_against_reference`` (CPU difference 0, CUDA <= 1e-4), tokenizer."""
    files = _RunFiles(folder, cfg)
    sr.check(sid, "status", "check-env run completed (meta.json status)",
             lambda: (files.meta.get("status"), "completed"))

    def checks() -> tuple:
        doc = _read_json(folder / "check_env.json")
        return ({"passed": doc.get("passed"), **{c["check"]: c["passed"] for c in doc["checks"]}},
                {"passed": True, "data": True, "sasrec": True, "tokenizer": True})

    sr.check(sid, "checks", "check_env.json: data load, SASRec and tokenizer checks passed (4.6.5)", checks)
    sr.check(sid, "environment", "M0 environment check (versions, transformers files, inputs, disk; GPU for cuda) and "
                                 "candidates.self_check() passed",
             lambda: ({key: (files.meta.get("preparation") or {}).get(key)
                       for key in ("environment_check", "candidates_self_check")},
                      {"environment_check": "passed", "candidates_self_check": "passed"}))
    sr.check(sid, "input_hashes", "data, prompt and SASRec source sha256 equal the expected values (4.6.5)",
             lambda: (files.meta.get("inputs"), {rel: sha for rel, (sha, _) in run.EXPECTED_INPUTS.items()}))

    def sasrec() -> tuple:
        from .sasrec import CUDA_REFERENCE_ATOL

        result = _env_report_part(folder, "sasrec")
        ref = result["reference"]
        measured = {"device": result["device"], "max_abs_diff": ref["max_abs_diff"], "bitwise": ref.get("bitwise"),
                    "state_digest": result["state_digest"]}
        same_digest = result["state_digest"] == cfg.sasrec.state_digest
        if str(result["device"]).startswith("cuda"):
            return (measured, {"max_abs_diff_at_most": CUDA_REFERENCE_ATOL, "state_digest": cfg.sasrec.state_digest},
                    same_digest and ref["max_abs_diff"] <= CUDA_REFERENCE_ATOL)
        return (measured, {"device": "cpu", "max_abs_diff": 0.0, "bitwise": True, "state_digest": cfg.sasrec.state_digest},
                same_digest and ref["max_abs_diff"] == 0.0 and ref.get("bitwise") is True)

    sr.check(sid, "sasrec_reference", "SASRec against the reference outputs: CPU bit for bit, CUDA <= 1e-4; digest = "
                                      "sasrec.state_digest (7.8 #5, #12)", sasrec)

    def tokenizer() -> tuple:
        from .batching import VOCAB_SIZE

        result = _env_report_part(folder, "tokenizer")
        return ({"model_sha256": result["model_sha256"], "vocab_size": result["vocab_size"]},
                {"model_sha256": cfg.batching.tokenizer_sha256, "vocab_size": VOCAB_SIZE})

    sr.check(sid, "tokenizer", "tokenizer.model sha256 = batching.tokenizer_sha256, vocabulary 32,005 (M5)", tokenizer)
    if cfg.model.device.startswith("cuda"):
        def gpu() -> tuple:
            env = files.meta.get("environment") or {}
            memory = [g.get("memory_gib") for g in env.get("gpus") or []]
            return ({"gpu_count": env.get("gpu_count"), "memory_gib": memory},
                    {"gpu_count": 1, "memory_gib_at_least": run.MIN_GPU_MEMORY_GIB},
                    env.get("gpu_count") == 1 and len(memory) == 1 and memory[0] >= run.MIN_GPU_MEMORY_GIB)

        sr.check(sid, "gpu", "exactly one visible GPU with at least 75 GiB ([SPEC §0.12], 17.6.7 G0)", gpu)


# ----------------------------------------------------------------------------
# Checks: training runs (S1, S5, S7; G1, G5, G6; O1)
# ----------------------------------------------------------------------------
def _expected_key_counts(cfg: "Config") -> dict:
    """``trainable.pt`` keys (§14.6.6): LoRA A and B of every wrapped module, projector 4, gate 1 (dynamic only),
    generated_frozen 3. Tiny model 28 + 4 + 1 + 3 = 36 (35 without gate), Llama-2 448 + 4 + 1 + 3 = 456 (455)."""
    from .model import GENERATED_FROZEN_KEYS, LLAMA2_7B_SHAPE, TINY_LLAMA_SHAPE

    shape = TINY_LLAMA_SHAPE if cfg.model.backbone == "tiny_random" else LLAMA2_7B_SHAPE
    counts = {"lora": 2 * shape["num_hidden_layers"] * len(cfg.lora.target_modules), "projector": PROJECTOR_KEYS,
              "gate": 1 if cfg.router.routing_mode == "dynamic" else 0,
              "generated_frozen": len(GENERATED_FROZEN_KEYS), "other": 0}
    counts["total"] = sum(counts.values())
    return counts


def _key_counts(state: Mapping[str, Any]) -> dict:
    from .model import GENERATED_FROZEN_PREFIX, OMEGA_SOURCE_PREFIX

    counts = {"lora": 0, "projector": 0, "gate": 0, "generated_frozen": 0, "other": 0}
    for key in state:
        if key.startswith("llm.") and (".lora_A." in key or ".lora_B." in key):
            counts["lora"] += 1
        elif key.startswith("projector."):
            counts["projector"] += 1
        elif key.startswith(OMEGA_SOURCE_PREFIX):
            counts["gate"] += 1
        elif key.startswith(GENERATED_FROZEN_PREFIX):
            counts["generated_frozen"] += 1
        else:
            counts["other"] += 1
    counts["total"] = len(state)
    return counts


def frozen_sasrec_values(files: Any, cfg: "Config", names: Sequence[str]) -> tuple[dict, dict]:
    """(measured, expected) of the ``frozen_sasrec`` check (§17.6.7 #2, M10 14.6.6, M3 7.8 #13): in every epoch's
    ``train_state.pt`` the frozen-parameter checksum recomputed at the epoch end (``frozen_checksum_at_save``) equals
    the start value (``frozen_checksum`` of epoch 0) and the SASRec digest recomputed at save time equals the
    profile's ``sasrec.state_digest``. ``files.train_state(e)`` is ``_RunFiles.train_state``."""
    start = files.train_state(0)["frozen_checksum"]
    measured = {name: {"sasrec_state_digest": files.train_state(e)["sasrec_state_digest"],
                       "frozen_checksum_at_save_equals_start": files.train_state(e)["frozen_checksum_at_save"] == start}
                for e, name in enumerate(names)}
    expected = {name: {"sasrec_state_digest": cfg.sasrec.state_digest, "frozen_checksum_at_save_equals_start": True}
                for name in names}
    return measured, expected


def _training_checks(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path) -> _RunFiles:
    """The checks every training step shares (§17.6.6 S1, §17.6.7 #1, #2, #4, §17.6.8 #1)."""
    from .prompts import build_batch_plan
    from .train import GROUP_ORDER, step_counts

    files, data = _RunFiles(folder, cfg), sr.data
    epochs, mode = int(cfg.train.epochs), cfg.router.routing_mode
    names = [f"epoch{e}" for e in range(epochs)]
    sc = functools.lru_cache(maxsize=None)(lambda: step_counts(len(data.train), cfg.train))
    plans = functools.lru_cache(maxsize=None)(lambda: [build_batch_plan(data.train, e, cfg) for e in range(epochs)])
    groups = [g for g in GROUP_ORDER if g != "gate" or mode == "dynamic"]

    sr.check(sid, "status", "training run completed (meta.json status)", lambda: (files.meta.get("status"), "completed"))
    sr.check(sid, "checkpoint_files", "checkpoints/epoch{e}: trainable.pt, train_state.pt, STATE_SAVED, val_metrics.json, "
                                      "COMPLETE for every epoch (14.6.6)",
             lambda: ({p.name: sorted(q.name for q in p.iterdir()) for p in sorted((folder / "checkpoints").iterdir())},
                      {name: sorted(CHECKPOINT_FILES) for name in names}))
    sr.check(sid, "trainable_keys", "trainable.pt keys per epoch: LoRA A/B, projector in every epoch, gate only for "
                                    "dynamic, generated_frozen 3 (36 / 35 tiny, 456 / 455 Llama-2)",
             lambda: ({name: _key_counts(files.trainable(e)) for e, name in enumerate(names)},
                      {name: _expected_key_counts(cfg) for name in names}))
    sr.check(sid, "optimizer_groups", "optimizer groups projector, gate (dynamic only), lora; no gate group for "
                                      "cluster_hard and K = 1 ([SPEC §11], 14.6.3)",
             lambda: ({name: {"optimizer": files.train_state(e)["groups"],
                              "group_param_names": files.train_state(e)["group_param_names"]}
                       for e, name in enumerate(names)},
                      {name: {"optimizer": groups, "group_param_names": groups} for name in names}))

    def adam() -> tuple:
        measured, expected = {}, {}
        for e, name in enumerate(names):
            state = files.train_state(e)
            done = [r for r in files.steps if r["epoch"] <= e and not r["skipped"]]
            got = {"projector": any(r["projector_has_grad"] for r in done), "gate": bool(done), "lora": bool(done)}
            measured[name] = state["with_state"]
            expected[name] = {group: n if got[group] else 0 for group, n in state["params"].items()}
        return measured, expected

    sr.check(sid, "adam_state", "Adam state per group = parameters that got a gradient in a non-skipped step up to that "
                                "epoch (17.6.7 #2, dynamic Llama-2 453)", adam)

    def scaler_rng() -> tuple:
        measured = {name: {key: files.train_state(e)[key] for key in ("scaler_keys", "rng_cpu", "rng_cuda")}
                    for e, name in enumerate(names)}
        want = {"scaler_keys": list(SCALER_KEYS) if cfg.train.precision == "fp16_amp" else [], "rng_cpu": True,
                "rng_cuda": cfg.model.device.startswith("cuda")}
        return measured, {name: want for name in names}

    sr.check(sid, "scaler_rng", "train_state.pt holds the GradScaler state (fp16_amp) and the CPU and CUDA random "
                                "states (17.6.7 #2)", scaler_rng)
    sr.check(sid, "frozen_sasrec", "frozen-parameter checksum and SASRec digest unchanged at every epoch end (14.6.6, "
                                   "M3 7.8 #13)", lambda: frozen_sasrec_values(files, cfg, names))
    sr.check(sid, "record_rows", "micro_batches.csv / optimizer_steps.csv / epochs.csv rows = total micro-batches, "
                                 "steps T, epochs; tau and step without gaps (M12 16.6.2)",
             lambda: ({"micro": len(files.micro), "step": len(files.steps), "epoch": len(files.epochs),
                       "tau_consecutive": [r["tau"] for r in files.micro] == list(range(1, len(files.micro) + 1)),
                       "step_consecutive": [r["step"] for r in files.steps] == list(range(len(files.steps)))},
                      {"micro": sc().total_micro, "step": sc().total_steps, "epoch": epochs, "tau_consecutive": True,
                       "step_consecutive": True}))
    sr.check(sid, "window_sizes", "n_micro of every step: grad_accum, the epoch-end window keeps the rest (D24)",
             lambda: ({name: [r["n_micro"] for r in files.steps if r["epoch"] == e] for e, name in enumerate(names)},
                      {name: list(sc().window_sizes()) for name in names}))
    sr.check(sid, "hybrid", "hybrid micro-batches per epoch and in the step rows = the BatchPlan draws (D3)",
             lambda: ({"epochs": [r["n_hybrid"] for r in files.epochs],
                       "steps": sum(r["n_hybrid"] for r in files.steps),
                       "micro_equal_plan": [[r["hybrid"] for r in files.micro if r["epoch"] == e]
                                            == [x.hybrid for x in plans()[e].entries] for e in range(epochs)]},
                      {"epochs": [p.hybrid_count() for p in plans()], "steps": sum(p.hybrid_count() for p in plans()),
                       "micro_equal_plan": [True] * epochs}))
    sr.check(sid, "plan_digest", "epochs.csv plan_digest = BatchPlan.digest() recomputed (M4 8.6.6)",
             lambda: ([r["plan_digest"] for r in files.epochs], [p.digest() for p in plans()]))
    calls = lambda: sc().total_micro if mode == "dynamic" else 0                            # noqa: E731
    sr.check(sid, "gate_calls", "gate calls of the training windows = micro-batches (dynamic), 0 otherwise; evaluation "
                                "calls excluded (M7 11.6.7)",
             lambda: ({"steps": sum(r["gate_calls"] for r in files.steps), "run_summary": files.run_summary["gate_calls"]},
                      {"steps": calls(), "run_summary": calls()}))
    lookups = lambda: sc().total_micro * sc().micro_batch if mode == "cluster_hard" else 0  # noqa: E731
    sr.check(sid, "cluster_lookups", "cluster lookups of the training windows = micro-batches x B (cluster_hard), 0 "
                                     "otherwise (M15 19.6.7)",
             lambda: ({"steps": sum(r["cluster_lookups"] for r in files.steps),
                       "run_summary": files.run_summary["cluster_lookups"]},
                      {"steps": lookups(), "run_summary": lookups()}))
    sr.check(sid, "projector_grad", "projector_has_grad = the window has a hybrid micro-batch (14.6.4 5.4)",
             lambda: ([r["step"] for r in files.steps if r["projector_has_grad"] != (r["n_hybrid"] > 0)], []))

    def skips() -> tuple:
        flags = [bool(r["skipped"]) for r in files.steps]
        longest = _longest_run(flags)
        return ({"skipped_steps": [r["step"] for r in files.steps if r["skipped"]], "longest_consecutive": longest},
                {"longest_consecutive_below": cfg.train.max_consecutive_skips},
                longest < cfg.train.max_consecutive_skips)

    sr.check(sid, "consecutive_skips", "no train.max_consecutive_skips (16) consecutive skipped steps (14.6.8)", skips)

    def eval_folders() -> tuple:
        found = {f"{p.parent.name}/{p.name}": files.metrics(p)["n_total"] for p in files.eval_folders()}
        expected = {f"epoch{e}/val": len(data.val) for e in range(epochs)}
        expected.update({f"epoch{b}/test": len(data.test) for b in files.tested})
        return found, expected

    sr.check(sid, "eval_folders", "validation of every epoch and test of best_val and last, each with its whole split "
                                  "(D11, D17)", eval_folders)
    sr.record(sid, "measurements", "recorded without a judgement (17.6.7 #6, TBD): times, step and evaluation peak "
                                   "memory, skips, non-finite losses",
              lambda: {"train_seconds": files.run_summary["train_summary"]["seconds"],
                       "step_max_mem_gib": max((r["max_mem_gib"] for r in files.steps if r["max_mem_gib"] is not None),
                                               default=None),
                       "step_max_mem_reserved_gib": max((r["max_mem_reserved_gib"] for r in files.steps
                                                         if r["max_mem_reserved_gib"] is not None), default=None),
                       "eval": {f"{p.parent.name}/{p.name}": {key: files.metrics(p)["meta"].get(key) for key in
                                                              ("seconds", "max_memory_allocated_gib")}
                                for p in files.eval_folders()},
                       "skipped_steps": sum(bool(r["skipped"]) for r in files.steps),
                       "nonfinite_loss": sum(r["nonfinite_loss"] for r in files.steps)})
    return files


def _batch_invariance(sr: _SmokeRun, sid: str, cfg: "Config", files: _RunFiles, *, judged: bool) -> None:
    """``batch_invariance.json`` of every evaluation (15.6.10): equal tokens for batch 8 and batch 1. Judged for the
    CPU E2 step (S1: 10 / 10, D22); recorded for the GPU (M11 #20, fp16 agreement TBD) and the other steps."""
    if not cfg.eval.batch_invariance_check:
        return

    def tokens() -> tuple:
        measured = {}
        for p in files.eval_folders():
            doc = _read_json(p / "batch_invariance.json")
            measured[f"{p.parent.name}/{p.name}"] = {"tokens_equal": doc["tokens_equal"], "n": doc["n"],
                                                     "ratios": doc["ratios"]}
        expected = {key: {"tokens_equal": value["n"], "n": value["n"], "ratios": value["ratios"]}
                    for key, value in measured.items()}
        return measured, expected, bool(measured) and all(v["tokens_equal"] == v["n"] for v in measured.values())

    description = "batch_invariance.json of every evaluation: generated tokens of batch 8 = batch 1 (D22)"
    if judged:
        sr.check(sid, "batch_invariance", description, tokens)
    else:
        sr.record(sid, "batch_invariance", description + " (recorded, TBD)", lambda: tokens()[0])


def _train_e2(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S1 / G1 (E2 path, dynamic): the shared training checks and the batch invariance of 17.6.6 S1."""
    files = _training_checks(sr, sid, cfg, folder)
    _batch_invariance(sr, sid, cfg, files, judged=sr.plan == "cpu")


def _train_e4(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S5 / G5 (E4 path, cluster_hard, 1 epoch): no gate group, 0 gate calls, lookups = micro-batches x B, the live z
    check passed, evaluation omega = one-hot of mapping[cluster of the assignment CSV], 35 / 455 keys (17.6.6 S5)."""
    import re

    files = _training_checks(sr, sid, cfg, folder)
    _batch_invariance(sr, sid, cfg, files, judged=False)

    def live() -> tuple:
        block = files.meta.get("cluster_live_check") or {}
        match = re.search(r"cluster live check (passed|FAILED): n=(\d+) max_abs_diff=(\S+) atol=(\S+) device=(\S+)",
                          (folder / "run.log").read_text(encoding="utf-8"))
        measured = {"passed": block.get("passed"), "n": block.get("n"), "max_abs_diff": block.get("max_abs_diff"),
                    "device": block.get("device"), "run_log": match.group(0) if match else None}
        ok = (block.get("passed") is True and block.get("n") == cfg.cluster.live_check_n
              and block.get("max_abs_diff") is not None and block["max_abs_diff"] <= cfg.cluster.live_check_atol
              and match is not None and match.group(1) == "passed")
        return measured, {"passed": True, "n": cfg.cluster.live_check_n,
                          "max_abs_diff_at_most": cfg.cluster.live_check_atol}, ok

    sr.check(sid, "live_check", "M15 verify_live_encoder passed (meta.json cluster_live_check and run.log, 19.6.7)", live)

    def one_hot() -> tuple:
        from .cluster import assignment_file, parse_expert_mapping, read_assignments
        from .data import SPLITS

        artifact = run.resolve_path(cfg.cluster.artifact_dir, sr.repo_root)
        assignment: dict[str, int] = {}
        for split in SPLITS:
            ids, _, _, clusters = read_assignments(artifact / assignment_file(split), split=split,
                                                   n_clusters=cfg.cluster.n_clusters)
            assignment.update(zip(ids, clusters.tolist()))
        k = cfg.lora.num_experts
        mapping = parse_expert_mapping(cfg.cluster.expert_mapping, n_clusters=cfg.cluster.n_clusters, num_experts=k)
        wrong, checked, calls = [], 0, {}
        for p in files.eval_folders():
            calls[f"{p.parent.name}/{p.name}"] = files.metrics(p)["meta"]["gate_calls"]
            for rec in files.records(p):
                expert = mapping[assignment[rec.sample_id]]
                checked += 1
                if [float(g) for g in rec.gates] != [1.0 if j == expert else 0.0 for j in range(k)]:
                    wrong.append(rec.sample_id)
        return ({"records": checked, "mismatches": wrong[:20], "eval_gate_calls": calls},
                {"mismatches": [], "eval_gate_calls": {key: 0 for key in calls}},
                checked > 0 and not wrong and all(v == 0 for v in calls.values()))

    sr.check(sid, "one_hot_gates", "every evaluation gate = one-hot of mapping[cluster in the assignment CSV], no gate "
                                   "call in evaluation (M7 #6, M15 #13)", one_hot)


def _train_e1(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S7 / G6 (E1 path, K = 1, routing none, 1 epoch): groups {projector, lora}, evaluation gate_0 = 1.0, step rows
    omega_mean_0 = 1.0 with an empty entropy, 35 / 455 keys (17.6.6 S7)."""
    files = _training_checks(sr, sid, cfg, folder)
    _batch_invariance(sr, sid, cfg, files, judged=False)

    def gate_one() -> tuple:
        not_one = [rec.sample_id for p in files.eval_folders() for rec in files.records(p)
                   if [float(g) for g in rec.gates] != [1.0]]
        n_eval = sum(len(files.records(p)) for p in files.eval_folders())
        return ({"eval_records": n_eval, "eval_gates_not_one": not_one[:20],
                 "step_omega_mean_0_not_one": [r["step"] for r in files.steps if r["omega_mean_0"] != 1.0],
                 "step_entropy_not_empty": [r["step"] for r in files.steps if r["omega_entropy_mean"] is not None]},
                {"eval_gates_not_one": [], "step_omega_mean_0_not_one": [], "step_entropy_not_empty": []},
                n_eval > 0 and not not_one and all(r["omega_mean_0"] == 1.0 and r["omega_entropy_mean"] is None
                                                   for r in files.steps))

    sr.check(sid, "gate_one", "K = 1: every evaluation gate_0 = 1.0 and every step omega_mean_0 = 1.0 (M7 #9)", gate_one)


def _train_overfit(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """O1 (§17.6.8 #1-#2): the shared training checks (25 epochs COMPLETE, no 16 consecutive skips), every micro-batch
    hybrid (all_hybrid) and last-epoch mean micro-batch loss <= first-epoch mean x 0.1 (the ratio is recorded)."""
    files = _training_checks(sr, sid, cfg, folder)
    sr.check(sid, "all_hybrid", "every micro-batch is hybrid (prompt.train_curriculum all_hybrid)",
             lambda: ({"micro": len(files.micro), "hybrid": sum(bool(r["hybrid"]) for r in files.micro)},
                      {"micro": len(files.micro), "hybrid": len(files.micro)}, bool(files.micro)
                      and all(r["hybrid"] for r in files.micro)))

    def ratio() -> tuple:
        last = int(cfg.train.epochs) - 1
        first = [r["loss"] for r in files.micro if r["epoch"] == 0]
        final = [r["loss"] for r in files.micro if r["epoch"] == last]
        m0, m1 = sum(first) / len(first), sum(final) / len(final)
        value = m1 / m0 if m0 else math.inf
        return ({"first_epoch_mean": m0, "last_epoch_mean": m1, "ratio": value,
                 "nonfinite": sum(not r["loss_finite"] for r in files.micro)},
                {"ratio_at_most": OVERFIT_LOSS_RATIO}, math.isfinite(value) and value <= OVERFIT_LOSS_RATIO)

    sr.check(sid, "loss_ratio", "last-epoch mean micro-batch loss <= first-epoch mean x 0.1 (17.6.8 #2; value TBD)",
             ratio)


# ----------------------------------------------------------------------------
# Checks: evaluation runs (S2, G2, S6)
# ----------------------------------------------------------------------------
def _eval_reload(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S2 / G2: ``main.py eval`` of the E2 checkpoints; every (best_val, last) x (val, test) ``records.jsonl`` has the
    sha256 of the training-time record (checkpoint round trip, 15.6.10)."""
    files, source = _RunFiles(folder, cfg), _RunFiles(sr.folders["train_e2"], cfg)
    summary = lambda: _read_json(folder / "eval" / "summary.json")                         # noqa: E731
    keys = lambda: [f"epoch{e}/{s}" for e in source.tested for s in cfg.eval.splits]       # noqa: E731
    sr.check(sid, "status", "eval run completed (meta.json status)", lambda: (files.meta.get("status"), "completed"))
    sr.check(sid, "reproduction", "summary.json reproduction of every (checkpoint, split) is identical",
             lambda: ({key: item["reproduction"]["status"] for key, item in summary()["results"].items()},
                      {key: "identical" for key in keys()}))
    sr.check(sid, "records_sha256", "records.jsonl sha256 = the training-time record of the same checkpoint and split",
             lambda: ({key: run.content_sha256(folder / "eval" / key / "records.jsonl")
                       == run.content_sha256(source.folder / "eval" / key / "records.jsonl") for key in keys()},
                      {key: True for key in keys()}))
    sr.check(sid, "checkpoints", "checkpoints loaded through M10 load_model_state: trainable sha256 = the file",
             lambda: ({item["ckpt_tag"]: item["trainable_sha256"] for item in summary()["checkpoints"]},
                      {f"epoch{e}": run.content_sha256(source.ckpt(e) / "trainable.pt") for e in source.tested}))


def _eval_e3(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S6 (E3 path): text-only prompts (every record hybrid false), metrics files present, reproduction marked
    ``not_comparable`` with differing key ``prompt.eval_prompt`` (17.6.12 불일치 3, 15.6.10)."""
    files, source = _RunFiles(folder, cfg), _RunFiles(sr.folders["train_e2"], cfg)
    summary = lambda: _read_json(folder / "eval" / "summary.json")                         # noqa: E731
    keys = lambda: [f"epoch{e}/{s}" for e in source.tested for s in cfg.eval.splits]       # noqa: E731
    sr.check(sid, "status", "eval run completed (meta.json status)", lambda: (files.meta.get("status"), "completed"))
    sr.check(sid, "text_only", "every evaluation record has hybrid false (prompt.eval_prompt text_only, D4)",
             lambda: ({f"{p.parent.name}/{p.name}": sum(bool(r.hybrid) for r in files.records(p))
                       for p in files.eval_folders()}, {key: 0 for key in keys()}))
    sr.check(sid, "metrics_files", "records.jsonl, metrics.json and representations.npz of every (checkpoint, split)",
             lambda: ({key: sorted(name for name in ("metrics.json", "records.jsonl", "representations.npz")
                                   if (folder / "eval" / key / name).is_file()) for key in summary()["results"]},
                      {key: ["metrics.json", "records.jsonl", "representations.npz"] for key in keys()}))
    sr.check(sid, "not_comparable", "summary.json marks the reproduction not_comparable because prompt.eval_prompt "
                                    "differs from the training run",
             lambda: ({key: [item["reproduction"]["status"], item["reproduction"].get("differing_keys")]
                       for key, item in summary()["results"].items()},
                      {key: ["not_comparable", ["prompt.eval_prompt"]] for key in keys()}))


# ----------------------------------------------------------------------------
# Checks: export (S3, G3) and cluster (S4, G4)
# ----------------------------------------------------------------------------
def _export_step(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S3 / G3 (§16.6.5): crosscheck.json passed (|d omega| <= 1e-5: the stored gate is the generation gate), NPZ rows
    of the three splits, the analysis of every ``records.analysis_splits`` with its figures, manifest sha256."""
    from .analysis import INTERPRETATION_TXT, SUMMARY_CSV, SUMMARY_JSON, TEN_SEQUENCES_CSV
    from .records import export_epochs, load_npz

    files, data, k = _RunFiles(folder, cfg), sr.data, cfg.lora.num_experts
    epochs = lambda: sorted(export_epochs(sr.folders["train_e2"], cfg.records.export_checkpoints))   # noqa: E731
    out = lambda e: folder / "records" / "export" / f"epoch{e}"                                      # noqa: E731
    sr.check(sid, "status", "export run completed (meta.json status)", lambda: (files.meta.get("status"), "completed"))
    sr.check(sid, "export_epochs", "records/export/epoch{e} for the best_val and last epochs of the training run",
             lambda: (sorted(p.name for p in (folder / "records" / "export").iterdir()), [f"epoch{e}" for e in epochs()]))

    def crosscheck() -> tuple:
        atol = cfg.records.gate_crosscheck_atol
        measured = {f"epoch{e}/{name}": {key: entry.get(key) for key in ("passed", "max_abs_diff_omega",
                                                                         "argmax_mismatches", "max_abs_diff_z")}
                    for e in epochs() for name, entry in _read_json(out(e) / "crosscheck.json")["splits"].items()}
        want = [f"epoch{e}/{name}" for e in epochs() for name in ("validation", "test")]
        ok = sorted(measured) == sorted(want) and all(v["passed"] is True and v["max_abs_diff_omega"] <= atol
                                                      for v in measured.values())
        return measured, {key: {"passed": True, "max_abs_diff_omega_at_most": atol} for key in want}, ok

    sr.check(sid, "crosscheck", "export omega = generation omega within records.gate_crosscheck_atol for validation and "
                                "test; |d z| recorded (16.6.5 step 7)", crosscheck)
    sizes = lambda: {"train": len(data.train), "validation": len(data.val), "test": len(data.test)}  # noqa: E731
    sr.check(sid, "npz_rows", "representations_{train,validation,test}.npz rows and K gate columns (16.2)",
             lambda: ({f"epoch{e}": {name: [len(t), int(t.omega.shape[1])] for name in EXPORT_SPLIT_NAMES
                                     for t in [load_npz(out(e) / f"representations_{name}.npz")]} for e in epochs()},
                      {f"epoch{e}": {name: [n, k] for name, n in sizes().items()} for e in epochs()}))

    def analysis() -> tuple:
        missing: dict[str, list[str]] = {}
        for e in epochs():
            base = folder / "records" / "analysis" / f"epoch{e}"
            if k < 2:                                   # M12 16.6.6: no gate analysis for K = 1
                missing[f"epoch{e}"] = ["(analysis folder present for K = 1)"] if base.exists() else []
                continue
            for split in cfg.records.analysis_splits:
                need = [SUMMARY_JSON, SUMMARY_CSV, INTERPRETATION_TXT]
                names = ANALYSIS_FIGURES + ((TEN_SEQUENCES_FIGURE,) if split == "test" else ())
                need += [TEN_SEQUENCES_CSV] if split == "test" else []
                need += [f"figures/{name}.{fmt}" for name in names for fmt in cfg.records.figure_formats]
                missing[f"epoch{e}/{split}"] = [n for n in need if not (base / split / n).is_file()]
        return missing, {key: [] for key in missing}, bool(missing) and not any(missing.values())

    sr.check(sid, "analysis", "gate_summary.json/.csv, gate_interpretation.txt and figures of every analysed split "
                              "(16.6.6, 16.6.7)", analysis)

    def manifest() -> tuple:
        wrong = {}
        for e in epochs():
            doc = _read_json(out(e) / "manifest.json")
            bad = [n for n, sha in doc["files"].items() if run.content_sha256(out(e) / n) != sha]
            bad += [n for n, sha in doc["analysis_files"].items() if run.content_sha256(folder / n) != sha]
            wrong[f"epoch{e}"] = bad
        return wrong, {key: [] for key in wrong}, bool(wrong) and not any(wrong.values())

    sr.check(sid, "manifest", "manifest.json sha256 of every export and analysis file", manifest)


def _cluster_step(sr: _SmokeRun, sid: str, cfg: "Config", folder: Path, _result: Any) -> None:
    """S4 / G4 (§19.6.2-§19.6.6): fit on the train rows only, no empty cluster, the reloaded joblib model predicts
    the CSV clusters, z equal to the E2 export z within 1e-5, the dynamic comparison items and figures."""
    from . import cluster as cl
    from .data import SPLITS
    from .records import SPEC_SPLIT_NAMES, load_npz

    files, data, out = _RunFiles(folder, cfg), sr.data, Path(folder) / cl.CLUSTER_DIR
    summary = lambda: _read_json(out / cl.SUMMARY_FILE)                                    # noqa: E731
    sr.check(sid, "status", "cluster run completed (meta.json status)", lambda: (files.meta.get("status"), "completed"))

    def fit() -> tuple:
        doc = _read_json(out / cl.MANIFEST_FILE)
        train_ids = [s.sample_id for s in data.train]
        if cfg.cluster.unit == "sequence":
            fit_ids = train_ids
        else:                                       # unit user (D25 variant): one window per train user
            fit_ids = [sid_ for sid_, _ in cl.read_user_assignments(out / cl.USER_ASSIGNMENTS_FILE,
                                                                       n_clusters=cfg.cluster.n_clusters).values()]
        measured = {key: doc.get(key) for key in ("fit_split", "fit_rows", "fit_ids_sha256", "rows")}
        measured["fit_ids_train_only"] = all(s.startswith("train:") and s in set(train_ids) for s in fit_ids)
        return measured, {"fit_split": "train", "fit_rows": len(fit_ids), "fit_ids_sha256": cl.ids_sha256(fit_ids),
                          "rows": {"train": len(data.train), "validation": len(data.val), "test": len(data.test)},
                          "fit_ids_train_only": True}

    sr.check(sid, "fit_train_only", "KMeans fitted on the train rows only (manifest fit_split, rows, id sha256; "
                                    "[SPEC §0.8])", fit)
    sr.check(sid, "no_empty_cluster", "every cluster has train samples; sizes add up to the train rows",
             lambda: (summary()["train_cluster_size"], {"clusters": cfg.cluster.n_clusters, "sum": len(data.train)},
                      len(summary()["train_cluster_size"]) == cfg.cluster.n_clusters
                      and min(summary()["train_cluster_size"]) > 0
                      and sum(summary()["train_cluster_size"]) == len(data.train)))

    def reload() -> tuple:
        km = cl.load_kmeans(out / cl.kmeans_file_name(cfg.cluster.n_clusters, cfg.cluster.random_state))
        mismatches = {}
        for split in SPLITS:
            table = load_npz(out / cl.representation_file(split))
            ids, _, _, clusters = cl.read_assignments(out / cl.assignment_file(split), split=split,
                                                      n_clusters=cfg.cluster.n_clusters)
            same_ids = tuple(ids) == tuple(table.sample_ids)
            mismatches[SPEC_SPLIT_NAMES[split]] = int((cl.predict(km, table.z) != clusters).sum()) if same_ids else -1
        return ({"reload_check": summary()["reload_check"].get("passed"), "mismatches": mismatches},
                {"reload_check": True, "mismatches": {name: 0 for name in mismatches}})

    sr.check(sid, "joblib_reload", "the joblib model read back predicts the assignment CSV of every split (19.6.5)",
             reload)

    def z_export() -> tuple:
        atol = cfg.cluster.z_crosscheck_atol
        export = run.resolve_path(cfg.cluster.dynamic_export_dir, sr.repo_root)
        again = {}
        for split in SPLITS:
            ours, theirs = load_npz(out / cl.representation_file(split)), load_npz(export / cl.representation_file(split))
            index = {s: i for i, s in enumerate(theirs.sample_ids)}
            diffs = [float(abs(theirs.z[index[s]].astype("float64") - ours.z[i].astype("float64")).max())
                     for i, s in enumerate(ours.sample_ids)]
            again[SPEC_SPLIT_NAMES[split]] = max(diffs) if len(index) == len(ours.sample_ids) else math.inf
        stored = summary()["dynamic_export"]["z_max_abs_diff"]
        return ({"summary": stored, "recomputed": again}, {"at_most": atol},
                sorted(stored) == sorted(again) and all(v <= atol for v in [*stored.values(), *again.values()]))

    sr.check(sid, "z_vs_export", "cluster z = E2 export z within cluster.z_crosscheck_atol for the three splits "
                                 "([SPEC §10], 19.6.6)", z_export)

    def dynamic() -> tuple:
        doc = summary()
        measured = {item: sorted(doc[item]) if isinstance(doc[item], Mapping) else doc[item] for item in cl.DYNAMIC_ITEMS}
        figures = [f"{name}.{fmt}" for name in (cl.PCA_FIGURE, cl.HEATMAP_FIGURE) for fmt in cfg.records.figure_formats]
        measured["missing_figures"] = [f for f in figures if not (out / cl.FIGURES_DIR / f).is_file()]
        expected = {item: sorted(EXPORT_SPLIT_NAMES) for item in cl.DYNAMIC_ITEMS}
        expected["missing_figures"] = []
        return measured, expected

    sr.check(sid, "dynamic_items", "dynamic comparison items for train/validation/test and both cluster figures (19.6.6)",
             dynamic)
    sr.record(sid, "measurements", "recorded without a judgement (TBD): silhouette, cluster sizes, NMI, timings",
              lambda: {key: summary().get(key) for key in ("silhouette_score_train", "train_cluster_size",
                                                           "nmi_cluster_vs_dynamic_argmax", "timings_s")})


# ----------------------------------------------------------------------------
# Checks: O2 (§17.6.8 #3)
# ----------------------------------------------------------------------------
def _generation_step(sr: _SmokeRun, sid: str, cfg: "Config", _folder: Optional[Path], result: Mapping) -> None:
    """O2: the last checkpoint (epoch E-1) was loaded, and 16 / 16 first lines equal the target titles."""
    sr.check(sid, "checkpoint", "the new model holds the last checkpoint, loaded with M10 load_model_state",
             lambda: ({"epoch": result["checkpoint"]["epoch"], "trainable_sha256": result["checkpoint"]["trainable_sha256"]},
                      {"epoch": int(cfg.train.epochs) - 1,
                       "trainable_sha256": run.content_sha256(Path(result["checkpoint"]["dir"]) / "trainable.pt")}))

    def match() -> tuple:
        n = len(sr.data.train)
        mismatched = [{key: s[key] for key in ("sample_id", "target_title", "first_line")}
                      for s in result["samples"] if not s["match"]]
        return ({"n": result["n"], "n_match": result["n_match"], "mismatched": mismatched}, {"n": n, "n_match": n},
                result["n"] == n and result["n_match"] == n)

    sr.check(sid, "first_line_match", "every overfit sample: match_norm(first line) == match_norm(target title) "
                                      "(17.6.8 #3)", match)
    sr.record(sid, "generation", "overfit generation settings, time and the first line of every sample (recorded)",
              lambda: {**{key: result[key] for key in ("epoch", "batch_size", "n_batches", "hybrid", "seconds")},
                       "samples": [{key: s[key] for key in ("sample_id", "target_title", "first_line", "match",
                                                            "n_new_tokens", "eos")} for s in result["samples"]]})


_CHECKERS: Final[dict[str, Callable[..., None]]] = {
    "check_env": _check_env_step, "train_e2": _train_e2, "eval_reload": _eval_reload, "export": _export_step,
    "cluster": _cluster_step, "train_e4": _train_e4, "eval_e3": _eval_e3, "train_e1": _train_e1,
    "train_overfit": _train_overfit, "generation": _generation_step,
}


# ----------------------------------------------------------------------------
# Entry (§17.6.12 제안 2)
# ----------------------------------------------------------------------------
def run_smoke(profile_path: PathLike, overrides: Sequence[str], run_dir: PathLike, *, repo_root: PathLike,
              llm_path: Optional[PathLike], tokenizer_dir: Optional[PathLike]) -> SmokeReport:
    """``main.py smoke`` (§17.6.6-§17.6.8): run the plan of the profile in ``run_dir`` and write ``smoke_report.json``.

    ``profile_path`` and ``overrides`` (the ``--set`` texts) are re-loaded for every step together with the
    step's own ``--set``; ``run_dir`` is the smoke run folder (M0 ``run_command`` or a test's tmp folder), the steps
    go to ``run_dir/steps/``. ``llm_path`` and ``tokenizer_dir`` are passed to every step. A failed step is recorded
    and the steps that need it are not run; the report is always written. Returns the ``SmokeReport``; M0
    ``run_command`` turns ``passed`` False into a failed run and a non-zero exit (17.6.12 제안 2).
    """
    profile_path, run_dir, repo_root = Path(profile_path), Path(run_dir), Path(repo_root)
    base_cfg, _ = load_config(profile_path, list(overrides))
    plan = smoke_plan(base_cfg)
    if not run_dir.is_dir():
        raise SmokeError(f"smoke run folder {run_dir} does not exist (M0 create_run_dir makes it)")
    if (run_dir / STEPS_DIR).exists() or (run_dir / REPORT_FILE).exists():
        raise SmokeError(f"{run_dir} already holds a smoke run; run folders are never reused (4.6.3)")
    (run_dir / STEPS_DIR).mkdir()
    started, started_utc = time.perf_counter(), run.utc_now().isoformat()
    smoke = _SmokeRun(plan, profile_path, overrides, run_dir, repo_root=repo_root,
                      llm_path=None if llm_path is None else Path(llm_path),
                      tokenizer_dir=None if tokenizer_dir is None else Path(tokenizer_dir), base_cfg=base_cfg)
    specs = PLAN_STEPS[plan]
    logger.info("smoke plan %s (%s): steps %s", plan, profile_path.name,
                ", ".join(f"{smoke.sid(s.index)} {s.name}" for s in specs))
    for spec in specs[:-1]:
        smoke.run_step(spec)
    return smoke.finish(specs[-1], started_utc=started_utc, started=started)


__all__ = [
    "SmokeError", "PLANS", "PLAN_PREFIX", "REPORT_FILE", "REPORT_FORMAT", "STEPS_DIR", "OVERFIT_LOSS_RATIO",
    "CHECKPOINT_FILES", "SmokeStep", "SmokeCheck", "SmokeReport", "StepSpec", "PLAN_STEPS", "KIND_CHECKS",
    "smoke_plan", "required_checks", "overfit_generation_check", "frozen_sasrec_values", "run_smoke",
]
