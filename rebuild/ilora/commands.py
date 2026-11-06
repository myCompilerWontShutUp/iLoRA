"""M0 명령 분기 (DESIGN.md §3.1 명령 분기, §4.6.1, §4.6.3, §4.6.5, §14.6.10 제안 4, §17.6.12 제안 3).

``run_command(command, cfg, overrides, *, profile, root)`` does the preparation
that every command shares and then calls the command function:

1. run folder ``<root>/<profile>_<UTC>/`` (M0 ``create_run_dir``, never reused),
   ``config.yaml`` and ``meta.json`` (status ``running``). ``train --resume``
   instead reopens a ``failed`` or unfinished (``running``) run folder and
   appends to ``meta.json`` ``resumes``. The process holds the folder's
   ``run.lock`` (M0 ``run.hold_run_lock``) until the command returns; a resume
   of a folder whose process is still alive is refused before anything is
   read or written.
2. logging to the console and ``run.log``; the first line carries ``run_dir=<path>``.
3. inside ``run.tracked_run`` (``meta.json`` becomes ``completed``/``failed``):
   disk space, the environment check (§4.6.5), ``candidates.self_check()``
   (M2 §6.6.5), then the command function:

   ========== ==========================================================
   train      M10 ``train.train(cfg, run_dir, *, repo_root, llm_path, resume)``
   eval       M11 ``evaluate.run_eval_command(cfg, run_dir, *, repo_root, llm_path)``
   export     M12 ``records.run_export(cfg, run_dir, *, repo_root)``
   cluster    M15 ``cluster.run_cluster(cfg, run_dir, *, repo_root)``
   smoke      M13 ``smoke.run_smoke(profile_path, overrides, run_dir, *, repo_root, llm_path, tokenizer_dir)``
   check-env  ``check_env`` below
   ========== ==========================================================

Module functions are imported only when their command runs, so this module
imports before those modules exist. ``main.py`` only parses arguments; the
package never imports ``main.py`` (§17.6.3 ``test_no_repo_imports``).
"""
from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

from . import run
from .config import ConfigError, config_hash, dump_config, load_config, parse_override

PathLike = Union[str, "os.PathLike[str]"]

COMMANDS = ("train", "eval", "export", "cluster", "smoke", "check-env")   # §4.6.1
# command -> (module of the ilora package, function name)
COMMAND_FUNCTIONS: dict[str, tuple[str, str]] = {
    "train": ("train", "train"),                      # M10 §14.6.1
    "eval": ("evaluate", "run_eval_command"),         # M11 §15.6.1
    "export": ("records", "run_export"),              # M12 §16.6.1
    "cluster": ("cluster", "run_cluster"),            # M15 §19.6.1
    "smoke": ("smoke", "run_smoke"),                  # M13 §17.6.12 제안 2
}
RESUMABLE_STATUSES = ("failed", "running")            # §3.1: failed or unfinished runs only
# Path-valued keys (§18.6.2, §18.6.12 제안 3), all read through run.resolve_path.
PATH_KEYS = ("sasrec.source_path", "sasrec.state_path", "sasrec.reference_path", "prompt.template_path",
             "eval.source_run", "records.source_run", "cluster.artifact_dir", "cluster.dynamic_export_dir")
# Disk check of §4.6.5 ("디스크 여유", no value in the design) [결정]: 1 GiB for any run plus the
# checkpoint size of a LLaMA training run, about 424 MiB per epoch (§14.6.6, §18.6.11).
MIN_FREE_DISK_GIB = 1.0
CHECKPOINT_GIB_PER_EPOCH = 424 / 1024

logger = logging.getLogger("ilora.commands")


class CommandError(RuntimeError):
    """A command cannot start or did not pass (resume of a completed run, failed smoke, ...)."""


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"{__package__}.{name}")


# ----------------------------------------------------------------------------
# Paths derived from the config and the environment
# ----------------------------------------------------------------------------
def llm_path_for(cfg: Any, llm_path: Optional[PathLike] = None) -> Optional[Path]:
    """LLaMA folder: only ``model.backbone == llama2_7b`` uses one (§13.6.1); default ``LLM_PATH`` (§4.2)."""
    if cfg.model.backbone != "llama2_7b":
        return None
    value = llm_path if llm_path is not None else os.environ.get("LLM_PATH")
    return Path(value) if value else None


def tokenizer_dir_for(cfg: Any, llm_path: Optional[PathLike],
                      tokenizer_dir: Optional[PathLike] = None) -> Optional[Path]:
    """Tokenizer folder (§9.2, §14.6.4): the LLaMA folder for llama2_7b, ``ILORA_TEST_TOKENIZER`` for tiny_random."""
    if tokenizer_dir is not None:
        return Path(tokenizer_dir)
    if cfg.model.backbone == "llama2_7b":
        return Path(llm_path) if llm_path is not None else None
    value = os.environ.get("ILORA_TEST_TOKENIZER")
    return Path(value) if value else None


def resolved_paths(cfg: Any, repo_root: Path = run.REPO_ROOT) -> dict[str, Optional[str]]:
    """Every path-valued key resolved with ``run.resolve_path`` (null stays None)."""
    out: dict[str, Optional[str]] = {}
    for key in PATH_KEYS:
        section, name = key.split(".")
        value = getattr(getattr(cfg, section), name)
        out[key] = None if value is None else str(run.resolve_path(value, repo_root))
    return out


def _file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": run.content_sha256(path) if path.is_file() else None}


def asset_record(cfg: Any, *, repo_root: Path, llm_path: Optional[Path],
                 tokenizer_dir: Optional[Path]) -> dict:
    """Input hashes outside ``run.EXPECTED_INPUTS`` for meta.json (§4.6.3).

    Converted SASRec state and reference (§7.6.1), ``tokenizer.model``, LLaMA
    ``config.json`` with its ``torch_dtype`` and the cluster manifest of a
    cluster_hard run. A missing file is recorded with ``sha256`` None.
    """
    out: dict[str, Any] = {
        "sasrec_state": _file_record(run.resolve_path(cfg.sasrec.state_path, repo_root)),
        "sasrec_reference": _file_record(run.resolve_path(cfg.sasrec.reference_path, repo_root)),
        "tokenizer_model": _file_record(Path(tokenizer_dir) / "tokenizer.model") if tokenizer_dir else None,
        "llama_config": None,
        "cluster_manifest": None,
    }
    if llm_path is not None:
        record = _file_record(Path(llm_path) / "config.json")
        dtype = None
        if record["sha256"] is not None:
            try:
                dtype = json.loads((Path(llm_path) / "config.json").read_text(encoding="utf-8")).get("torch_dtype")
            except (OSError, ValueError, AttributeError):
                dtype = None
        record["torch_dtype"] = dtype
        out["llama_config"] = record
    if cfg.cluster.artifact_dir is not None:
        out["cluster_manifest"] = _file_record(run.resolve_path(cfg.cluster.artifact_dir, repo_root) / "manifest.json")
    return out


# ----------------------------------------------------------------------------
# Shared preparation checks
# ----------------------------------------------------------------------------
def command_requirements(command: str, cfg: Any) -> None:
    """Keys that one command needs (checked before a run folder is made)."""
    if command == "eval" and cfg.eval.source_run is None:
        raise ConfigError("eval needs --set eval.source_run=<training run folder> (DESIGN §15.6.12)")
    if command == "export" and cfg.records.source_run is None:
        raise ConfigError("export needs --set records.source_run=<training run folder> (DESIGN §16.6.9)")


def required_free_gib(command: str, cfg: Any) -> float:
    need = MIN_FREE_DISK_GIB
    if command in ("train", "smoke") and cfg.model.backbone == "llama2_7b":
        need += cfg.train.epochs * CHECKPOINT_GIB_PER_EPOCH
    return need


def disk_check(run_dir: Path, command: str, cfg: Any) -> dict:
    """Run folder free space (§4.6.5 "실행 폴더 쓰기 가능, 디스크 여유")."""
    free = shutil.disk_usage(run_dir).free / 2**30
    need = required_free_gib(command, cfg)
    if free < need:
        raise run.EnvironmentCheckError(f"free disk at {run_dir} is {free:.2f} GiB < {need:.2f} GiB needed")
    return {"free_gib": round(free, 2), "required_gib": round(need, 2)}


def environment_check(cfg: Any, *, repo_root: Path, llm_path: Optional[Path]) -> None:
    """M0 environment check of §4.6.5 for this profile (GPU only for cuda:0, LLaMA only for llama2_7b)."""
    run.check_environment(require_gpu=cfg.model.device.startswith("cuda"), llm_path=llm_path,
                          require_llm=cfg.model.backbone == "llama2_7b", check_inputs=True, repo_root=repo_root)


def self_check() -> None:
    """M2 golden comparison (§6.6.5): numpy Generator output equals the design golden."""
    _module("candidates").self_check()


# ----------------------------------------------------------------------------
# check-env
# ----------------------------------------------------------------------------
def check_sasrec(cfg: Any, catalog: Any, *, repo_root: Path, device: Union[str, Any]) -> dict:
    """SASRec part of check-env (§7.6.8, §7.8 #11-#12, §17.6.6 S0, §17.6.7 G0).

    1. original pickle sha256 = ``sasrec.source_sha256``
    2. ``SASRecEncoder.from_config`` (M3): state digest = ``sasrec.state_digest``,
       catalog ``pad_id``/``num_items`` and history length agree with the state
    3. ``verify_against_reference(encoder, reference_path, device)`` (M3): CPU
       difference 0, CUDA within 1e-4 (§7.8 #5, #12). The function raises on a
       failure; a returned mapping with ``passed`` False is also a failure.
    Raises on the first failure; returns the measured values.
    """
    import torch

    source = run.resolve_path(cfg.sasrec.source_path, repo_root)
    if not source.is_file():
        raise run.EnvironmentCheckError(f"SASRec source {source} not found")
    got = run.content_sha256(source)
    if got != cfg.sasrec.source_sha256:
        raise run.EnvironmentCheckError(f"SASRec source sha256 {got} != sasrec.source_sha256 {cfg.sasrec.source_sha256}")
    sasrec = _module("sasrec")
    history_len = _module("data").HISTORY_LEN
    dev = torch.device(device)
    encoder = sasrec.SASRecEncoder.from_config(cfg, catalog, history_len).to(dev)
    reference = sasrec.verify_against_reference(encoder, run.resolve_path(cfg.sasrec.reference_path, repo_root), dev)
    if isinstance(reference, Mapping) and reference.get("passed") is False:
        raise run.EnvironmentCheckError(f"SASRec reference comparison failed on {dev}: {dict(reference)}")
    return {"source_sha256": got, "state_digest": encoder.state_digest, "device": str(dev),
            "reference": dict(reference) if isinstance(reference, Mapping) else reference}


def check_tokenizer(cfg: Any, tokenizer_dir: Optional[Path]) -> dict:
    """Tokenizer part of check-env: M5 ``load_tokenizer`` (sha256, versions, special ids, fingerprints)."""
    if tokenizer_dir is None:
        raise run.EnvironmentCheckError("tokenizer folder not available: llama2_7b uses LLM_PATH, "
                                        "tiny_random uses ILORA_TEST_TOKENIZER")
    bundle = _module("batching").load_tokenizer(Path(tokenizer_dir), cfg.batching.tokenizer_sha256)
    return {"tokenizer_dir": str(tokenizer_dir), "model_sha256": bundle.model_sha256,
            "vocab_size": bundle.ids.vocab_size}


def check_env(cfg: Any, run_dir: Path, *, repo_root: Path, llm_path: Optional[Path],
              tokenizer_dir: Optional[Path]) -> dict:
    """The ``check-env`` command (§4.6.5, §18.6.6 R4, §17.6.6 S0, §17.6.7 G0).

    ``run_command`` has already run the shared checks: versions and pins,
    transformers integrity, GPU, LLaMA folder (``config.json`` torch_dtype),
    input hashes, disk and ``candidates.self_check()``. This adds the checks
    that need module code: data load (M1), SASRec (M3, ``check_sasrec``) and the
    tokenizer (M5, ``check_tokenizer``). Every check runs; the report is written
    to ``check_env.json`` and any failure raises ``EnvironmentCheckError``.
    """
    checks: list[dict] = []
    loaded: dict[str, Any] = {}

    def attempt(name: str, fn: Callable[[], Any]) -> None:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - collected and raised together below
            checks.append({"check": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        checks.append({"check": name, "passed": True, "result": result})

    def load_data() -> dict:
        data = _module("data").load_dataset(cfg, repo_root)
        loaded["data"] = data
        return {"subset": data.subset, "train": len(data.train), "val": len(data.val), "test": len(data.test)}

    attempt("data", load_data)
    if "data" in loaded:
        attempt("sasrec", lambda: check_sasrec(cfg, loaded["data"].catalog, repo_root=repo_root,
                                               device=cfg.model.device))
    else:
        checks.append({"check": "sasrec", "passed": False, "error": "not run: data did not load"})
    attempt("tokenizer", lambda: check_tokenizer(cfg, tokenizer_dir))
    passed = all(c["passed"] for c in checks)
    report = {"passed": passed, "device": cfg.model.device, "checks": checks}
    run.write_json_atomic(Path(run_dir) / "check_env.json", report)
    for c in checks:
        logger.info("check-env %s: %s", c["check"], "passed" if c["passed"] else f"FAILED {c['error']}")
    if not passed:
        raise run.EnvironmentCheckError("check-env failed:\n  - " + "\n  - ".join(
            f"{c['check']}: {c['error']}" for c in checks if not c["passed"]))
    return report


# ----------------------------------------------------------------------------
# Run folder, resume, logging
# ----------------------------------------------------------------------------
def load_run_config(run_dir: PathLike) -> tuple[Any, str]:
    """Config and profile name of an existing run folder (``train --resume``)."""
    run_dir = Path(run_dir)
    cfg, _ = load_config(run_dir / "config.yaml")
    return cfg, run.read_meta(run_dir)["profile"]


def _complete_epochs(run_dir: Path) -> list[int]:
    done = []
    for path in (run_dir / "checkpoints").glob("epoch*"):
        match = re.fullmatch(r"epoch(0|[1-9][0-9]*)", path.name)
        if match and (path / "COMPLETE").is_file():
            done.append(int(match.group(1)))
    return sorted(done)


def _reopen_for_resume(run_dir: Path, cfg: Any, argv: Sequence[str]) -> Path:
    """Reopen a failed or unfinished train run (§3.1, §14.6.10 제안 4); nothing is overwritten. The caller holds the
    folder's ``run.lock``, so a ``running`` status here belongs to a process that has ended (killed, §4.8 #6)."""
    if not (run_dir / "meta.json").is_file() or not (run_dir / "config.yaml").is_file():
        raise CommandError(f"cannot resume {run_dir}: meta.json or config.yaml not found")
    meta = run.read_meta(run_dir)
    status = meta.get("status")
    if status not in RESUMABLE_STATUSES:
        raise CommandError(f"cannot resume {run_dir}: status is {status!r}; only failed or unfinished "
                           f"(running) runs are reopened")
    if meta.get("command", "train") != "train":
        raise CommandError(f"cannot resume {run_dir}: it is a {meta.get('command')!r} run, not train")
    saved, _ = load_config(run_dir / "config.yaml")
    if not (config_hash(cfg) == config_hash(saved) == meta.get("config_hash")):
        raise CommandError(f"cannot resume {run_dir}: the config differs from its config.yaml/meta.json")
    complete = _complete_epochs(run_dir)
    resumes = list(meta.get("resumes") or [])
    resumes.append({"utc": run.utc_now().isoformat(), "argv": run.scrub_argv(argv),
                    "previous_status": status, "previous_ended_utc": meta.get("ended_utc"),
                    "previous_error": meta.get("error"), "complete_epochs": complete,
                    "start_epoch": (complete[-1] + 1) if complete else 0})
    run.update_meta(run_dir, {"status": "running", "ended_utc": None, "elapsed_s": None, "error": None,
                              "resumes": resumes})
    return run_dir


@contextlib.contextmanager
def _run_logging(run_dir: Path) -> Iterator[logging.Logger]:
    """Console + ``run.log`` for one run; the previous handlers come back afterwards (nested smoke steps)."""
    root = logging.getLogger("ilora")
    saved = (list(root.handlers), root.level, root.propagate)
    run.setup_logging(run_dir)
    try:
        yield logger
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        handlers, level, propagate = saved
        for handler in handlers:   # a closed FileHandler (mode "a") reopens on the next record
            root.addHandler(handler)
        root.setLevel(level)
        root.propagate = propagate


def _dispatch(command: str, cfg: Any, run_dir: Path, *, repo_root: Path, llm_path: Optional[Path],
              tokenizer_dir: Optional[Path], overrides: Sequence[str], profile_path: Optional[Path],
              resume: bool) -> Any:
    if command == "check-env":
        return check_env(cfg, run_dir, repo_root=repo_root, llm_path=llm_path, tokenizer_dir=tokenizer_dir)
    module_name, function_name = COMMAND_FUNCTIONS[command]
    function = getattr(_module(module_name), function_name)
    if command == "train":
        return function(cfg, run_dir, repo_root=repo_root, llm_path=llm_path, resume=resume)
    if command == "eval":
        return function(cfg, run_dir, repo_root=repo_root, llm_path=llm_path)
    if command in ("export", "cluster"):
        return function(cfg, run_dir, repo_root=repo_root)
    report = function(Path(profile_path), list(overrides), run_dir, repo_root=repo_root, llm_path=llm_path,
                      tokenizer_dir=tokenizer_dir)
    if not getattr(report, "passed", False):
        raise CommandError(f"smoke checks failed (see {run_dir / 'smoke_report.json'})")
    return report


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------
def run_command(command: str, cfg: Any, overrides: Sequence[str] = (), *, profile: str,
                root: PathLike = run.RUNS_ROOT, repo_root: PathLike = run.REPO_ROOT,
                argv: Optional[Sequence[str]] = None, llm_path: Optional[PathLike] = None,
                tokenizer_dir: Optional[PathLike] = None, profile_path: Optional[PathLike] = None,
                resume: Optional[PathLike] = None, stamp: Optional[str] = None) -> Path:
    """Run one command with the shared preparation and return its run folder (§3.1 명령 분기).

    ``cfg`` is the loaded ``Config`` and ``overrides`` the ``--set`` texts it was
    loaded with (recorded in meta.json; ``smoke`` re-loads the profile with them).
    ``profile`` names the run folder ``<root>/<profile>_<UTC>/``; ``root`` is
    ``runs/rebuild`` or, for smoke steps, ``<smoke run>/steps`` (§17.6.12).
    ``llm_path`` defaults to ``LLM_PATH`` for llama2_7b; ``tokenizer_dir`` to
    the LLaMA folder (llama2_7b) or ``ILORA_TEST_TOKENIZER`` (tiny_random).
    ``profile_path`` is required for ``smoke``. ``resume`` (train only) reopens
    that run folder instead of making a new one; when another live process
    holds the folder's ``run.lock`` (a ``running`` training that a shell
    disconnect did not stop) it is a ``CommandError`` and nothing in the folder
    changes. Exceptions propagate after ``meta.json`` is marked ``failed``.
    """
    if command not in COMMANDS:
        raise CommandError(f"unknown command {command!r}, expected one of {COMMANDS}")
    if resume is not None and command != "train":
        raise CommandError("--resume is only for train")
    if command == "smoke" and profile_path is None:
        raise CommandError("smoke needs profile_path (the profile file it re-loads for each step)")
    command_requirements(command, cfg)
    repo_root = Path(repo_root)
    overrides = list(overrides)
    applied = dict(parse_override(text) for text in overrides)
    llm = llm_path_for(cfg, llm_path)
    tok_dir = tokenizer_dir_for(cfg, llm, tokenizer_dir)
    argv = list(sys.argv if argv is None else argv)

    with contextlib.ExitStack() as held:
        if resume is None:
            run_dir = run.create_run_dir(profile, stamp, Path(root))
            lock = held.enter_context(run.hold_run_lock(run_dir, command))    # a new folder: no other holder
            dump_config(cfg, run_dir / "config.yaml")
            paths = resolved_paths(cfg, repo_root)
            extra: dict[str, Any] = {
                "command": command,
                "profile_path": str(profile_path) if profile_path is not None else None,
                "inputs": run.input_hashes(repo_root),
                "assets": asset_record(cfg, repo_root=repo_root, llm_path=llm, tokenizer_dir=tok_dir),
                "paths": paths,
                "llm_path": str(llm) if llm is not None else None,
                "tokenizer_dir": str(tok_dir) if tok_dir is not None else None,
                "resumes": [],
            }
            if command == "eval":
                extra["source_run"] = paths["eval.source_run"]
            elif command == "export":
                extra["source_run"] = paths["records.source_run"]
            run.start_meta(run_dir, profile=profile, argv=argv, config_hash=config_hash(cfg), overrides=applied,
                           seed=cfg.run.seed, extra=extra)
        else:
            run_dir = Path(resume)
            if not (run_dir / "meta.json").is_file() or not (run_dir / "config.yaml").is_file():
                raise CommandError(f"cannot resume {run_dir}: meta.json or config.yaml not found")
            try:                               # before meta.json is read or written (§14.6.10 제안 4)
                lock = held.enter_context(run.hold_run_lock(run_dir, "train --resume"))
            except run.RunLockError as exc:
                raise CommandError(f"cannot resume {run_dir}: {exc}. Its process is still running (a shell "
                                   "disconnect does not stop a run inside tmux): reattach with tmux attach -t "
                                   "rebuild instead of resuming (RUNBOOK R7)") from exc
            run_dir = _reopen_for_resume(run_dir, cfg, argv)

        with _run_logging(run_dir) as log:
            log.info("run_dir=%s", run_dir)
            log.info("command=%s profile=%s resume=%s", command, profile, resume is not None)
            if lock.held:
                log.info("%s held by this process (pid %d) until the command returns", lock.path.name, os.getpid())
            else:
                log.warning("%s could not be locked (%s): this file system gives no guard against a second "
                            "process on the folder", lock.path.name, lock.problem)
            with run.tracked_run(run_dir):
                try:
                    disk = disk_check(run_dir, command, cfg)
                    environment_check(cfg, repo_root=repo_root, llm_path=llm)
                    log.info("environment check passed")
                    self_check()
                    log.info("candidates.self_check passed")
                    run.update_meta(run_dir, {"preparation": {"disk": disk, "environment_check": "passed",
                                                              "candidates_self_check": "passed"}})
                    _dispatch(command, cfg, run_dir, repo_root=repo_root, llm_path=llm, tokenizer_dir=tok_dir,
                              overrides=overrides, profile_path=Path(profile_path) if profile_path else None,
                              resume=resume is not None)
                except BaseException as exc:  # noqa: BLE001 - logged, then tracked_run marks failed
                    log.error("%s failed: %s: %s", command, type(exc).__name__, exc)
                    raise
            log.info("%s completed", command)
    return run_dir


__all__ = [
    "COMMANDS", "COMMAND_FUNCTIONS", "PATH_KEYS", "CommandError", "run_command", "check_env", "check_sasrec",
    "check_tokenizer", "environment_check", "self_check", "llm_path_for", "tokenizer_dir_for",
    "resolved_paths", "asset_record", "command_requirements", "disk_check", "load_run_config",
]
