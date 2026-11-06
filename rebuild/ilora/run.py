"""M0 실행 폴더, 출처 기록, 환경 점검, 경로·해시, 로깅 (DESIGN.md §4.6.3-§4.6.6, §3.1 경로와 해시)."""
from __future__ import annotations

import base64
import contextlib
import csv
import dataclasses
import datetime as _dt
import errno
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
RUNS_ROOT = REPO_ROOT / "runs" / "rebuild"

REQUIRED_TORCH = "2.0.0"
REQUIRED_TRANSFORMERS = "4.28.0"
MIN_GPU_MEMORY_GIB = 75.0

# Library pins checked before every run (DESIGN.md §4.6.5): numpy and pandas
# (unpickling the released .df files and numpy Generator output, M1·M2),
# sentencepiece (slow tokenizer splitting, M5), scikit-learn (KMeans, M15).
# Keys are import names; values are the required ``__version__``.
REQUIRED_LIBRARIES: dict[str, str] = {
    "numpy": "1.24.2",
    "pandas": "1.5.3",
    "sentencepiece": "0.1.97",
    "sklearn": "1.2.2",
}

# Distribution versions recorded in meta.json (DESIGN.md §4.6.3), import name -> distribution.
_META_DISTRIBUTIONS: dict[str, str] = {
    "torch": "torch", "transformers": "transformers", "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece", "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "yaml": "PyYAML", "matplotlib": "matplotlib", "scipy": "scipy", "joblib": "joblib",
    "threadpoolctl": "threadpoolctl", "safetensors": "safetensors", "pytest": "pytest",
}

# Expected content hashes of the inputs (DESIGN.md §4.6.5). Text files are
# hashed after CRLF -> LF normalization so that a Windows checkout with
# core.autocrlf=true and a Linux checkout give the same value (= git blob).
# train_data.df is the iLoRA copy (blob 73b47263, has an extra 'cluster'
# column that is never read).
EXPECTED_INPUTS: dict[str, tuple[str, bool]] = {
    "data/ref/movielens/train_data.df": ("bac4c88b11e6f76d914c2b22aadbb97315a78432bb79f424f7015775053b9546", False),
    "data/ref/movielens/Val_data.df": ("9a950900d0055649e2983ef403d189d5fdc5bd4f84d229a377806d371b32e460", False),
    "data/ref/movielens/Test_data.df": ("d38308a0078d4518dc9eeb619c8d1afb53d830712cc7d36ac51cc47eaafc97d6", False),
    "data/ref/movielens/u.item": ("553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701", True),
    "prompt/movie.txt": ("957fc1f27bb930ca5acbe1b545186ecc04fd93ea2c5307644c706ad44395ecd3", True),
    "rec_model/movielens.pt": ("6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9", False),
}

# Strings that must never reach meta.json or logs.
_SECRET_VALUE_RE = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})"
)
_SECRET_NAME_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)


class EnvironmentCheckError(RuntimeError):
    """The environment does not match what the profile requires."""


class SecretLeakError(RuntimeError):
    """A token-like string was about to be written."""


# ----------------------------------------------------------------------------
# Hashing and provenance
# ----------------------------------------------------------------------------
def content_sha256(path: Path, text: bool = False) -> str:
    data = Path(path).read_bytes()
    if text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def resolve_path(value: Union[str, "os.PathLike[str]"], repo_root: Path = REPO_ROOT) -> Path:
    """Interpret a path-valued config key (DESIGN.md §3.1 경로와 해시, §18.6.12 제안 3).

    The value is an absolute path or a path relative to the repository root
    (never to the working directory): ``sasrec.source_path``/``state_path``/
    ``reference_path``, ``prompt.template_path``, ``eval.source_run``,
    ``records.source_run``, ``cluster.artifact_dir``, ``cluster.dynamic_export_dir``.
    """
    if value is None:
        raise ValueError("path value is None (the key is null)")
    text = os.fspath(value)
    if not str(text).strip():
        raise ValueError("path value is empty")
    path = Path(text)
    return path if path.is_absolute() else Path(repo_root) / path


def git_info(repo_root: Path = REPO_ROOT) -> dict:
    def _git(*args: str) -> str:
        out = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        porcelain = _git("status", "--porcelain")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "branch": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"commit": commit, "branch": branch, "dirty": bool(porcelain)}


def rebuild_file_hashes(rebuild_dir: Path = REBUILD_DIR) -> dict[str, str]:
    out = {}
    for path in sorted(rebuild_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt"}:
            continue
        out[path.relative_to(rebuild_dir).as_posix()] = content_sha256(path, text=True)
    return out


def input_hashes(repo_root: Path = REPO_ROOT) -> dict[str, Optional[str]]:
    out = {}
    for rel, (_, is_text) in EXPECTED_INPUTS.items():
        path = repo_root / rel
        out[rel] = content_sha256(path, is_text) if path.exists() else None
    return out


def _dist_version(dist: str) -> Optional[str]:
    # Installed distribution metadata: does not import the package (no import side effects).
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info() -> dict:
    """Environment block of meta.json (DESIGN.md §4.6.3)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {name: _dist_version(dist) for name, dist in _META_DISTRIBUTIONS.items()},
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()     # CPU results are bitwise only at equal counts (15.6.10)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": props.name, "memory_gib": round(props.total_memory / 2**30, 2)})
        info["gpus"] = gpus
        info["gpu_count"] = len(gpus)
    except ImportError:
        info["cuda_available"] = None
    return info


# ----------------------------------------------------------------------------
# Secrets
# ----------------------------------------------------------------------------
def find_secrets(obj: Any, path: str = "$") -> list[str]:
    """Return JSON paths whose string value looks like a credential."""
    hits = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            hits += find_secrets(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits += find_secrets(v, f"{path}[{i}]")
    elif isinstance(obj, str) and _SECRET_VALUE_RE.search(obj):
        hits.append(path)
    return hits


def scrub_argv(argv: Sequence[str]) -> list[str]:
    """Mask values of arguments whose name looks secret (e.g. --hf_token x)."""
    out = []
    mask_next = False
    for arg in argv:
        if mask_next:
            out.append("***")
            mask_next = False
            continue
        if arg.startswith("-") and _SECRET_NAME_RE.search(arg):
            if "=" in arg:
                out.append(arg.split("=", 1)[0] + "=***")
            else:
                out.append(arg)
                mask_next = True
            continue
        out.append("***" if _SECRET_VALUE_RE.search(arg) else arg)
    return out


# ----------------------------------------------------------------------------
# Run folder and meta.json
# ----------------------------------------------------------------------------
def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def utc_stamp(now: Optional[_dt.datetime] = None) -> str:
    return (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(profile: str, stamp: Optional[str] = None, root: Path = RUNS_ROOT) -> Path:
    """Create ``<root>/<profile>_<UTC>/`` with its subfolders; never reuse one."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile):
        raise ValueError(f"profile name must be [A-Za-z0-9_.-]+: {profile!r}")
    path = Path(root) / f"{profile}_{stamp or utc_stamp()}"
    if path.exists():
        raise FileExistsError(f"run folder already exists, refusing to overwrite: {path}")
    path.mkdir(parents=True)
    for sub in ("checkpoints", "eval", "records"):
        (path / sub).mkdir()
    return path


def write_json_atomic(path: Path, obj: Any) -> None:
    hits = find_secrets(obj)
    if hits:
        raise SecretLeakError(f"refusing to write token-like values at {hits} to {path}")
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)


def start_meta(run_dir: Path, *, profile: str, argv: Sequence[str], config_hash: str,
               overrides: Mapping[str, Any], seed: int, extra: Optional[Mapping[str, Any]] = None) -> dict:
    meta = {
        "status": "running",
        "profile": profile,
        "argv": scrub_argv(argv),
        "config_hash": config_hash,
        "overrides": dict(overrides),
        "seed": seed,
        "started_utc": utc_now().isoformat(),
        "ended_utc": None,
        "elapsed_s": None,
        "error": None,
        "code": {"git": git_info(), "rebuild_files": rebuild_file_hashes()},
        "environment": environment_info(),
        "inputs": input_hashes(),
    }
    if extra:
        meta.update(extra)
    write_json_atomic(run_dir / "meta.json", meta)
    return meta


def finish_meta(run_dir: Path, status: str, error: Optional[str] = None,
                extra: Optional[Mapping[str, Any]] = None) -> dict:
    if status not in {"completed", "failed"}:
        raise ValueError(status)
    path = run_dir / "meta.json"
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ended = utc_now()
    started = _dt.datetime.fromisoformat(meta["started_utc"])
    meta.update({"status": status, "ended_utc": ended.isoformat(),
                 "elapsed_s": round((ended - started).total_seconds(), 3), "error": error})
    if extra:
        meta.update(extra)
    write_json_atomic(path, meta)
    return meta


def read_meta(run_dir: Path) -> dict:
    with open(Path(run_dir) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def update_meta(run_dir: Path, fields: Mapping[str, Any]) -> dict:
    """Merge top-level ``fields`` into ``meta.json`` (atomic; token-like values refused)."""
    meta = read_meta(run_dir)
    meta.update(fields)
    write_json_atomic(Path(run_dir) / "meta.json", meta)
    return meta


@contextlib.contextmanager
def tracked_run(run_dir: Path) -> Iterator[None]:
    """Mark meta.json completed/failed when the block ends (also on Ctrl-C)."""
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - record and re-raise everything
        finish_meta(run_dir, "failed", error="".join(traceback.format_exception_only(type(exc), exc)).strip())
        raise
    finish_meta(run_dir, "completed")


# ----------------------------------------------------------------------------
# Run folder lock: one live process per run folder
# ----------------------------------------------------------------------------
# A killed process (SIGKILL, OOM killer, SIGHUP) leaves meta.json "running" (§4.8 #6), and train --resume reopens such
# folders (§14.6.10 제안 4). Whether the process behind "running" is still alive is decided by an exclusive OS lock
# that the kernel releases when the holder ends in any way, so no stale lock and no pid probing (pid reuse) exist.
RUN_LOCK_FILE = "run.lock"
# msvcrt.locking locks a byte range; the locked byte lies far beyond the holder lines, which stay readable.
_WINDOWS_LOCK_OFFSET = 2**30
_LOCK_HELD_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK,
                               getattr(errno, "EDEADLOCK", errno.EDEADLK)})


class RunLockError(RuntimeError):
    """Another live process holds the run folder (its ``run.lock``)."""


@dataclasses.dataclass(frozen=True)
class RunLock:
    """``run.lock`` of a run folder: ``held`` False only on a file system without locks (``problem`` says why)."""
    path: Path
    held: bool
    problem: Optional[str] = None


def _os_lock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _last_lock_line(path: Path) -> str:
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


@contextlib.contextmanager
def hold_run_lock(run_dir: Path, label: str = "") -> Iterator[RunLock]:
    """Hold an exclusive, non-blocking OS lock on ``<run_dir>/run.lock`` for the block.

    Linux ``fcntl.flock(LOCK_EX | LOCK_NB)``, Windows ``msvcrt.locking(LK_NBLCK)``. The lock belongs to the open
    file, so it ends with the holding process however that ends (exception, SIGKILL, OOM killer, SIGHUP); a held
    lock therefore means a live process. ``RunLockError`` names the holder line when another process holds it. The
    file is opened for appending and never truncated or deleted: each holder appends one line (UTC, host, pid,
    ``label``) for people reading the folder. A file system that refuses locks gives ``held`` False with the reason
    (the run is then not guarded, as before this lock existed).
    """
    path = Path(run_dir) / RUN_LOCK_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o644)
    held, problem = False, None
    try:
        try:
            _os_lock(fd)
            held = True
        except OSError as exc:
            if exc.errno in _LOCK_HELD_ERRNOS:
                holder = _last_lock_line(path) or "no holder line"
                raise RunLockError(f"{run_dir} is in use by a live process ({RUN_LOCK_FILE}: {holder})") from exc
            problem = f"{type(exc).__name__}: {exc}"
        line = f"{utc_now().isoformat()} host={socket.gethostname()} pid={os.getpid()}"
        line += (f" {label}" if label else "") + ("" if held else f" (not locked: {problem})")
        os.write(fd, (line + "\n").encode("utf-8"))
        yield RunLock(path, held, problem)
    finally:
        if held:
            with contextlib.suppress(OSError):        # closing the file releases the lock anyway
                _os_unlock(fd)
        os.close(fd)


# ----------------------------------------------------------------------------
# Environment checks
# ----------------------------------------------------------------------------
def library_version_problems() -> list[str]:
    """Fail-fast pins of REQUIRED_LIBRARIES (DESIGN.md §4.6.5), read from the imported modules."""
    problems = []
    for name, required in REQUIRED_LIBRARIES.items():
        try:
            module = importlib.import_module(name)
        except ImportError:
            problems.append(f"{name} is not installed (required {required})")
            continue
        got = getattr(module, "__version__", None)
        if got != required:
            problems.append(f"{name} {got} != {required}")
    return problems


def _b64_sha256(data: bytes) -> str:
    # Wheel RECORD hash format: urlsafe base64 of the sha256 digest without '=' padding (PEP 376/427).
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def record_problems(site_dir: Path, record_text: str, package_dir: Path, package: str,
                    limit: int = 20) -> list[str]:
    """Compare every installed ``<package>/**/*.py`` with the sha256 in its wheel RECORD.

    A file that differs (for example the fork's ``debug/`` copies over
    ``generation/utils.py`` and ``models/llama/modeling_llama.py``,
    ``F setup_vessl.sh:87-106``), a ``.py`` file that the RECORD does not list
    and a listed file that is missing are all problems (D13, §17.6.12 제안 4).
    """
    expected: dict[str, str] = {}
    for row in csv.reader(io.StringIO(record_text)):
        if not row:
            continue
        rel = row[0].replace("\\", "/")
        if rel.startswith(package + "/") and rel.endswith(".py"):
            expected[rel] = row[1] if len(row) > 1 else ""
    site_dir = Path(site_dir)
    if (site_dir / package).resolve() != Path(package_dir).resolve():
        return [f"imported {package} at {package_dir} is not the installation its wheel RECORD describes "
                f"({site_dir / package})"]
    installed = {p.relative_to(site_dir).as_posix() for p in (site_dir / package).rglob("*.py")}
    problems = [f"{package} file {rel} is not listed in the wheel RECORD" for rel in sorted(installed - expected.keys())]
    problems += [f"{package} file {rel} from the wheel RECORD is missing" for rel in sorted(expected.keys() - installed)]
    for rel in sorted(installed & expected.keys()):
        algo, _, value = expected[rel].partition("=")
        if algo != "sha256" or _b64_sha256((site_dir / rel).read_bytes()) != value:
            problems.append(f"{package} file {rel} differs from the wheel RECORD sha256 (modified install, D13)")
    if len(problems) > limit:
        problems = problems[:limit] + [f"... and {len(problems) - limit} more {package} install problems"]
    return problems


def _transformers_install() -> tuple[Path, str, Path]:
    """(site folder, RECORD text, imported package folder) of the transformers in use."""
    import transformers

    dist = importlib.metadata.distribution("transformers")
    record = dist.read_text("RECORD")
    if record is None:
        raise FileNotFoundError("the transformers dist-info has no RECORD file")
    return Path(dist.locate_file("")), record, Path(transformers.__file__).resolve().parent


def transformers_integrity_problems() -> list[str]:
    """Every installed transformers ``.py`` equals its wheel RECORD sha256 (D13, §4.6.5)."""
    try:
        site_dir, record, package_dir = _transformers_install()
    except Exception as exc:  # noqa: BLE001 - reported as a check failure, never ignored
        return [f"cannot verify the transformers install against its wheel RECORD: {type(exc).__name__}: {exc}"]
    return record_problems(site_dir, record, package_dir, "transformers")


def check_environment(*, require_gpu: bool, llm_path: Optional[Path], check_inputs: bool = True,
                      repo_root: Path = REPO_ROOT, require_llm: bool = False) -> None:
    """Raise EnvironmentCheckError listing every problem found (DESIGN.md §4.6.5).

    Always: torch 2.0.0, transformers 4.28.0 (D13), REQUIRED_LIBRARIES, transformers
    install integrity. ``require_gpu``: CUDA available, exactly one visible GPU
    (SPEC §0.12), GPU 0 memory >= 75 GiB. ``llm_path``: LLaMA folder (config.json
    ``torch_dtype`` float16, shards, tokenizer.model); ``require_llm`` makes a
    missing ``llm_path`` (LLM_PATH not set) a problem. ``check_inputs``: input hashes.
    """
    problems: list[str] = []
    try:
        import torch
    except ImportError:
        raise EnvironmentCheckError("torch is not installed") from None
    try:
        import transformers
    except ImportError:
        raise EnvironmentCheckError("transformers is not installed") from None

    if torch.__version__.split("+")[0] != REQUIRED_TORCH:
        problems.append(f"torch {torch.__version__} != {REQUIRED_TORCH}")
    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        problems.append(f"transformers {transformers.__version__} != {REQUIRED_TRANSFORMERS}")
    problems += library_version_problems()
    problems += transformers_integrity_problems()

    if require_gpu:
        if not torch.cuda.is_available():
            problems.append("CUDA is not available")
        else:
            count = torch.cuda.device_count()
            if count != 1:
                problems.append(f"exactly one visible GPU is required (SPEC §0.12), found {count}")
            if count >= 1:
                mem = torch.cuda.get_device_properties(0).total_memory / 2**30
                if mem < MIN_GPU_MEMORY_GIB:
                    problems.append(f"GPU 0 has {mem:.1f} GiB < {MIN_GPU_MEMORY_GIB} GiB")

    if llm_path is not None:
        problems += check_llama_dir(Path(llm_path))
    elif require_llm:
        problems.append("LLM_PATH is not set: the profile needs the LLaMA folder (model.backbone llama2_7b)")

    if check_inputs:
        for rel, (expected, is_text) in EXPECTED_INPUTS.items():
            path = repo_root / rel
            if not path.exists():
                problems.append(f"missing input {rel}")
                continue
            got = content_sha256(path, is_text)
            if got != expected:
                problems.append(f"hash mismatch {rel}: {got} != {expected}")

    if problems:
        raise EnvironmentCheckError("environment check failed:\n  - " + "\n  - ".join(problems))


def check_llama_dir(path: Path) -> list[str]:
    problems = []
    if not path.is_dir():
        return [f"LLaMA path is not a directory: {path}"]
    cfg = path / "config.json"
    if not cfg.exists():
        return [f"missing {cfg}"]
    with open(cfg, "r", encoding="utf-8") as f:
        conf = json.load(f)
    if conf.get("torch_dtype") != "float16":
        problems.append(f"config.json torch_dtype is {conf.get('torch_dtype')!r}, expected 'float16' (D9)")
    shards = list(path.glob("pytorch_model*.bin")) + list(path.glob("model*.safetensors"))
    if not shards:
        problems.append(f"no weight shards in {path}")
    if not (path / "tokenizer.model").exists():
        problems.append(f"missing tokenizer.model in {path}")
    return problems


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def setup_logging(run_dir: Optional[Path], level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("ilora")
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if run_dir is not None:
        fh = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
