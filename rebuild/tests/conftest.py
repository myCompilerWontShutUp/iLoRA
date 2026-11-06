"""Shared pytest fixtures for the rebuild (DESIGN.md §17.6.2, §17.6.3, §17.8 #2 and #4).

Tests run on CPU. Optional local assets come from environment variables so that
nothing outside the repository is hard-coded:

- ``ILORA_TEST_TOKENIZER``: folder with the LLaMA SentencePiece test tokenizer
  (hf-internal-testing/llama-tokenizer, ``tokenizer.model`` sha256 9e556afd...d347).
- ``ILORA_SASREC_STATE``: converted SASRec state file of
  ``rebuild/tools/convert_sasrec.py``; default location
  ``runs/rebuild/assets/sasrec/sasrec_movielens_state.pt`` (§17.6.12 제안 6).

Fixtures import module code inside their bodies, so collection works before
the modules exist. A missing asset skips with one of the documented reasons
``SKIP_*`` below (§17.8 #2); the VESSL fixtures and reasons are in
``tests/vessl/conftest.py``. pytest 9 ignores marks on fixtures, so tests that
use ``converted_sasrec`` or ``cpu_smoke_run`` carry ``@pytest.mark.slow``
themselves (§17.6.3). Synthetic inputs come from ``seeding.np_rng(0, "test",
<name>)`` and ``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
if str(REBUILD_DIR) not in sys.path:
    sys.path.insert(0, str(REBUILD_DIR))

CONFIG_DIR = REBUILD_DIR / "configs"
RUNS_REBUILD = REPO_ROOT / "runs" / "rebuild"
DEFAULT_SASREC_DIR = RUNS_REBUILD / "assets" / "sasrec"      # §7.6.1 tool output, §17.6.2
SASREC_STATE_NAME = "sasrec_movielens_state.pt"
SASREC_REFERENCE_NAME = "sasrec_movielens_reference.pt"

# Documented skip reasons (§17.6.2, §17.8 #2). Tests skip only through these constants
# (and the VESSL ones in tests/vessl/conftest.py).
SKIP_NO_TOKENIZER = "test tokenizer not available (set ILORA_TEST_TOKENIZER to a folder with tokenizer.model)"
SKIP_NO_SASREC = ("converted SASRec weights not available (set ILORA_SASREC_STATE or run "
                  "rebuild/tools/convert_sasrec.py --out runs/rebuild/assets/sasrec)")
SKIP_NO_GIT_HISTORY = "full git history not available (git missing or shallow clone)"
DOCUMENTED_SKIP_REASONS = (SKIP_NO_TOKENIZER, SKIP_NO_SASREC, SKIP_NO_GIT_HISTORY)


# ----------------------------------------------------------------------------
# Helpers shared by fixtures
# ----------------------------------------------------------------------------
def sasrec_state_file() -> Path:
    """``ILORA_SASREC_STATE`` or the default converted-state location."""
    value = os.environ.get("ILORA_SASREC_STATE")
    return Path(value) if value else DEFAULT_SASREC_DIR / SASREC_STATE_NAME


def sasrec_overrides() -> list[str]:
    """``--set`` texts pointing a profile at ``ILORA_SASREC_STATE`` (none for the default location,
    which is already the profile value ``runs/rebuild/assets/sasrec/...``)."""
    value = os.environ.get("ILORA_SASREC_STATE")
    if not value:
        return []
    state = Path(value)
    return [f"sasrec.state_path={json.dumps(state.as_posix())}",
            f"sasrec.reference_path={json.dumps((state.parent / SASREC_REFERENCE_NAME).as_posix())}"]


def load_profile(name: str, *sets: str) -> Any:
    """``load_config(rebuild/configs/{name}.yaml, sets)`` (M0) with the SASRec paths of ``sasrec_overrides``."""
    from ilora.config import load_config

    cfg, _ = load_config(CONFIG_DIR / f"{name}.yaml", [*sasrec_overrides(), *sets])
    return cfg


def _git_history_available() -> bool:
    if shutil.which("git") is None:
        return False
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip() == "false"


def _repo_state() -> tuple[Optional[str], tuple[str, ...]]:
    """(``git status --porcelain``, listing of runs/rebuild/ without ``assets/`` and ``vessl_tests_*``)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=300)
        status: Optional[str] = out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        status = None
    listing = []
    if RUNS_REBUILD.is_dir():
        for path in sorted(RUNS_REBUILD.rglob("*")):
            rel = path.relative_to(RUNS_REBUILD)
            if rel.parts[0] == "assets" or rel.parts[0].startswith("vessl_tests_"):
                continue
            listing.append(rel.as_posix())
    return status, tuple(listing)


@pytest.fixture(scope="session", autouse=True)
def repo_guard() -> Any:
    """Tests write only to tmp_path (§17.1, §17.8 #4): checked at session teardown."""
    before = _repo_state()
    yield
    after = _repo_state()
    problems = []
    if before[0] != after[0]:
        problems.append(f"git status --porcelain changed\n--- before\n{before[0]}--- after\n{after[0]}")
    if before[1] != after[1]:
        added = sorted(set(after[1]) - set(before[1]))
        removed = sorted(set(before[1]) - set(after[1]))
        problems.append(f"runs/rebuild/ changed (assets/ and vessl_tests_* excluded): added {added[:20]}, "
                        f"removed {removed[:20]}")
    if problems:
        pytest.fail("the test session changed the repository:\n" + "\n".join(problems), pytrace=False)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def rebuild_dir() -> Path:
    return REBUILD_DIR


@pytest.fixture(scope="session")
def tokenizer_dir() -> Path:
    path = os.environ.get("ILORA_TEST_TOKENIZER")
    if not path or not (Path(path) / "tokenizer.model").exists():
        pytest.skip(SKIP_NO_TOKENIZER)
    return Path(path)


@pytest.fixture(scope="session")
def sasrec_state_path() -> Path:
    """``ILORA_SASREC_STATE`` or ``runs/rebuild/assets/sasrec/sasrec_movielens_state.pt``."""
    path = sasrec_state_file()
    if not path.is_file():
        pytest.skip(SKIP_NO_SASREC)
    return path


@pytest.fixture(scope="session")
def sasrec_reference_path(sasrec_state_path: Path) -> Path:
    """``sasrec_movielens_reference.pt`` next to the state file."""
    path = sasrec_state_path.parent / SASREC_REFERENCE_NAME
    if not path.is_file():
        pytest.skip(SKIP_NO_SASREC)
    return path


# ----------------------------------------------------------------------------
# Profiles and module objects (imports inside the fixtures)
# ----------------------------------------------------------------------------
@pytest.fixture
def profile_cfg() -> Callable[..., Any]:
    """Factory ``profile_cfg(name, *sets) -> Config``: ``load_config(rebuild/configs/{name}.yaml, sets)``."""
    return load_profile


@pytest.fixture(scope="session")
def paper_cfg() -> Any:
    return load_profile("paper_spec")


@pytest.fixture(scope="session")
def cpu_cfg() -> Any:
    return load_profile("smoke_cpu")


@pytest.fixture(scope="session")
def full_data(paper_cfg: Any) -> Any:
    """M1 ``load_dataset(paper_cfg, repo_root)`` on the released files."""
    from ilora.data import load_dataset

    return load_dataset(paper_cfg, REPO_ROOT)


@pytest.fixture(scope="session")
def smoke_data(full_data: Any) -> Any:
    """Data of ``smoke_cpu`` (§17.6.6): ``select_smoke(full_data, 1234, (2, 10, 10), train_rows=())``."""
    from ilora.data import select_smoke

    return select_smoke(full_data, 1234, (2, 10, 10), train_rows=())


@pytest.fixture(scope="session")
def bundle(tokenizer_dir: Path, paper_cfg: Any) -> Any:
    """M5 ``load_tokenizer(tokenizer_dir, paper_cfg.batching.tokenizer_sha256)``."""
    from ilora.batching import load_tokenizer

    return load_tokenizer(tokenizer_dir, paper_cfg.batching.tokenizer_sha256)


@pytest.fixture(scope="session")
def templates() -> Any:
    """M4 ``load_templates(repo_root / "prompt/movie.txt")``."""
    from ilora.prompts import load_templates

    return load_templates(REPO_ROOT / "prompt" / "movie.txt")


@pytest.fixture
def tiny_model(bundle: Any, smoke_data: Any, sasrec_state_path: Path) -> Callable[..., Any]:
    """Factory ``tiny_model(*sets, hard_gate=None)``: M9 ``build_model(smoke_cpu + sets, bundle,
    smoke_data.catalog, llm_path=None, hard_gate=...)`` (hidden 64, 2 layers, CPU fp32, §13.6.9)."""
    from ilora.model import build_model

    def make(*sets: str, hard_gate: Any = None) -> Any:
        cfg = load_profile("smoke_cpu", *sets)
        return build_model(cfg, bundle, smoke_data.catalog, llm_path=None, hard_gate=hard_gate)

    return make


class FakeHardGate:
    """Test object for the M7 ``HardGate`` protocol (§11.6.1): dict lookup -> one-hot ``[B, K]`` float32
    on ``device``, ``lookups`` counts looked-up samples, an unknown id raises ``KeyError``.

    ``fault`` makes it return a wrong output: ``"dtype"`` (float64), ``"shape"`` (``[B, K + 1]``),
    ``"batch_axis"`` (``[B, 1, K]`` like the released router, ``U model/router/nlpr.py:80``),
    ``"two_hot"`` (second 1 in row 0, row sum 2), ``"half"`` (row 0 = 0.5 and 0.5: row sum 1 but not one-hot),
    ``"nan"`` (NaN in row 0), ``"negative"`` (row 0 = -1 and 2), ``"rowsum"`` (row 0 halved),
    ``"zeros"`` (row 0 all 0), ``"device"`` (a device other than the requested one: ``meta`` when CPU is
    requested, CPU otherwise, so the fault is visible in CPU tests).
    """

    FAULTS = (None, "dtype", "shape", "batch_axis", "two_hot", "half", "nan", "negative", "rowsum", "zeros",
              "device")

    def __init__(self, table: Mapping[str, int], num_experts: int, *, fault: Optional[str] = None) -> None:
        if fault not in self.FAULTS:
            raise ValueError(f"unknown fault {fault!r}, expected one of {self.FAULTS}")
        self.table = dict(table)
        self.num_experts = num_experts
        self.fault = fault
        self.lookups = 0
        self.calls = 0

    def __call__(self, sample_ids: Sequence[str], device: Any) -> Any:
        import torch

        idx = torch.tensor([self.table[s] for s in sample_ids], dtype=torch.int64)   # KeyError for unknown ids
        self.lookups += len(sample_ids)
        self.calls += 1
        k = self.num_experts + 1 if self.fault == "shape" else self.num_experts
        omega = torch.zeros(len(sample_ids), k, dtype=torch.float32).scatter_(1, idx[:, None], 1.0)
        if self.fault == "two_hot":
            omega[0, (int(idx[0]) + 1) % k] = 1.0
        elif self.fault == "nan":
            omega[0, 0] = float("nan")
        elif self.fault == "negative":
            omega[0] = 0.0
            omega[0, 0] = -1.0
            omega[0, k - 1] = 2.0
        elif self.fault == "rowsum":
            omega[0] = omega[0] * 0.5
        elif self.fault == "zeros":
            omega[0] = 0.0
        elif self.fault == "dtype":
            omega = omega.double()
        elif self.fault == "half":
            omega[0] = 0.0
            omega[0, 0] = 0.5
            omega[0, k - 1] = 0.5
        elif self.fault == "batch_axis":
            omega = omega.unsqueeze(1)
        target = torch.device(device)
        if self.fault == "device":
            target = torch.device("meta" if target.type == "cpu" else "cpu")
        return omega.to(target)


@pytest.fixture
def fake_hard_gate() -> Callable[..., FakeHardGate]:
    """Factory ``fake_hard_gate(table, K, *, fault=None) -> FakeHardGate``."""
    def make(table: Mapping[str, int], K: int, *, fault: Optional[str] = None) -> FakeHardGate:
        return FakeHardGate(table, K, fault=fault)

    return make


@pytest.fixture(scope="session")
def smoke_cluster_dir(sasrec_state_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``cluster/`` made by M15 ``run_cluster(cpu_cfg, <tmp run folder>, repo_root=repo_root)`` (smoke train 156)."""
    from ilora import run
    from ilora.cluster import run_cluster
    from ilora.config import dump_config

    cfg = load_profile("smoke_cpu")
    run_dir = run.create_run_dir("smoke_cluster", root=tmp_path_factory.mktemp("smoke_cluster"))
    dump_config(cfg, run_dir / "config.yaml")
    run_cluster(cfg, run_dir, repo_root=REPO_ROOT)
    return run_dir / "cluster"


@pytest.fixture(scope="session")
def git_objects_dir() -> Path:
    """Absolute object folder of the repository (``git rev-parse --git-path objects``); M3 #1 shares it with
    temporary repositories through ``objects/info/alternates`` (read-only). Skips without git or full history."""
    if not _git_history_available():
        pytest.skip(SKIP_NO_GIT_HISTORY)
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-path", "objects"],
                         capture_output=True, text=True, timeout=60, check=True)
    return Path(out.stdout.strip())


@pytest.fixture(scope="session")
def converted_sasrec(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Output folder of ``tools/convert_sasrec.py --repo <repo_root> --out <tmp>`` (§7.6.1). Tests mark slow."""
    if not _git_history_available():
        pytest.skip(SKIP_NO_GIT_HISTORY)
    out_dir = tmp_path_factory.mktemp("converted_sasrec") / "sasrec"
    proc = subprocess.run([sys.executable, "-I", str(REBUILD_DIR / "tools" / "convert_sasrec.py"),
                           "--repo", str(REPO_ROOT), "--out", str(out_dir)],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        pytest.fail(f"convert_sasrec.py exited with {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}",
                    pytrace=False)
    return out_dir


@pytest.fixture(scope="session")
def cpu_smoke_run(tokenizer_dir: Path, sasrec_state_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """M13 ``smoke.run_smoke`` of ``smoke_cpu`` into a tmp run folder -> ``SmokeReport`` (§17.6.6). Tests mark slow."""
    from ilora import run
    from ilora.smoke import run_smoke

    run_dir = run.create_run_dir("smoke_cpu", root=tmp_path_factory.mktemp("cpu_smoke"))
    return run_smoke(CONFIG_DIR / "smoke_cpu.yaml", sasrec_overrides(), run_dir, repo_root=REPO_ROOT,
                     llm_path=None, tokenizer_dir=tokenizer_dir)


@pytest.fixture(scope="session")
def llama7b_skeleton() -> Any:
    """transformers 4.28.0 ``LlamaForCausalLM(LlamaConfig())`` (hidden 4096, 11008, 32 layers, 32 heads,
    vocab 32000) on the meta device: shapes only, no weights."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.device("meta"):
        return LlamaForCausalLM(LlamaConfig())


@pytest.fixture
def torch_threads1() -> Any:
    """``torch.set_num_threads(1)`` for bitwise comparisons, restored afterwards."""
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
