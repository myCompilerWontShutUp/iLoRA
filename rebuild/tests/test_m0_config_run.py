"""M0 설정과 실행 관리 (DESIGN.md §4.8 #1-#6, §17.6.9 M0 표, §3.1 경로·precision 도구·명령 분기)."""
import json
import os
import subprocess
import sys
import textwrap
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from ilora import commands, run
from ilora.config import ConfigError, build_config, config_hash, config_to_dict, dump_config, item, load_config

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"


@dataclass(frozen=True)
class _Train:
    lr: float = item("[논문 App E p.17]", "max lr")
    k: int = item("[논문 p.9 §4.3.2]", "experts")
    flag: bool = item("[결정]", "flag")
    mode: str = item("[결정]", "mode", choices=("a", "b"))
    mods: tuple = item("[코드]", "modules")


@dataclass(frozen=True)
class _Root:
    train: _Train


GOOD = {"train": {"lr": "1e-4", "k": 4, "flag": False, "mode": "a", "mods": ["q", "v"]}}


def _copy(d):
    return json.loads(json.dumps(d))


def _profile(name, *sets):
    cfg, _ = load_config(CONFIG_DIR / f"{name}.yaml", list(sets))
    return cfg


# ----------------------------------------------------------------------------
# M0-1 config round trip and rejection (synthetic schema; the real schema is in test_m0_sections.py)
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M0-1")
def test_valid_config_round_trip(tmp_path):
    cfg = build_config(_copy(GOOD), _Root)
    assert cfg.train.lr == pytest.approx(1e-4)
    path = tmp_path / "c.yaml"
    dump_config(cfg, path)
    again, _ = load_config(path, schema=_Root)
    assert config_to_dict(again) == config_to_dict(cfg)
    assert config_hash(again) == config_hash(cfg)


@pytest.mark.criterion("M0-1")
@pytest.mark.parametrize("key,value", [("k", 8.0), ("k", True), ("flag", "False"), ("mode", "c"), ("lr", "abc")])
def test_wrong_types_and_choices_are_rejected(key, value):
    d = _copy(GOOD)
    d["train"][key] = value
    with pytest.raises(ConfigError):
        build_config(d, _Root)


@pytest.mark.criterion("M0-1")
def test_unknown_and_missing_keys_are_rejected():
    d = _copy(GOOD)
    d["train"]["typo"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        build_config(d, _Root)
    d = _copy(GOOD)
    del d["train"]["lr"]
    with pytest.raises(ConfigError, match="missing"):
        build_config(d, _Root)
    with pytest.raises(ConfigError, match="unknown section"):
        build_config({**_copy(GOOD), "extra": {}}, _Root)


@pytest.mark.criterion("M0-1")
def test_extends_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  lr: 1.0e-4\n  k: 4\n  flag: false\n  mode: a\n  mods: [q, v]\n",
                                        encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain:\n  k: 1\n", encoding="utf-8")
    cfg, applied = load_config(tmp_path / "child.yaml", ["train.mode=b"], schema=_Root)
    assert cfg.train.k == 1 and cfg.train.mode == "b"
    assert applied == {"train.mode": "b"}
    with pytest.raises(ConfigError):
        load_config(tmp_path / "child.yaml", ["train.nope=1"], schema=_Root)


@pytest.mark.criterion("M0-2")
def test_item_rejects_missing_label():
    with pytest.raises(ValueError):
        item("no label", "doc")


@pytest.mark.criterion("M0-1")
def test_resolve_path(tmp_path):
    """Path-valued keys: absolute, or relative to the repository root (§3.1, §18.6.12 제안 3)."""
    assert run.resolve_path("rec_model/movielens.pt") == run.REPO_ROOT / "rec_model" / "movielens.pt"
    assert run.resolve_path("prompt/movie.txt", repo_root=tmp_path) == tmp_path / "prompt" / "movie.txt"
    absolute = (tmp_path / "x" / "cluster").resolve()
    assert run.resolve_path(str(absolute), repo_root=Path("elsewhere")) == absolute
    assert run.resolve_path(absolute) == absolute
    for bad in (None, "", "   "):
        with pytest.raises(ValueError):
            run.resolve_path(bad)
    cfg = _profile("paper_spec")
    paths = commands.resolved_paths(cfg, tmp_path)
    assert paths["sasrec.state_path"] == str(tmp_path / "runs/rebuild/assets/sasrec/sasrec_movielens_state.pt")
    assert paths["eval.source_run"] is None and set(paths) == set(commands.PATH_KEYS)


# ----------------------------------------------------------------------------
# M0-3 run folder and meta.json
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M0-3")
def test_run_dir_is_never_reused(tmp_path):
    path = run.create_run_dir("smoke_cpu", "20261106T000000Z", root=tmp_path)
    assert (path / "checkpoints").is_dir() and (path / "eval").is_dir() and (path / "records").is_dir()
    with pytest.raises(FileExistsError):
        run.create_run_dir("smoke_cpu", "20261106T000000Z", root=tmp_path)


@pytest.mark.criterion("M0-3", "M0-6")
def test_meta_status_and_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 34)
    rd = run.create_run_dir("smoke_cpu", "20261106T000001Z", root=tmp_path)
    run.start_meta(rd, profile="smoke_cpu", argv=["main.py", "--hf_token", "hf_" + "y" * 34],
                   config_hash="abc", overrides={}, seed=1234)
    text = (rd / "meta.json").read_text(encoding="utf-8")
    assert "hf_" + "x" * 34 not in text and "hf_" + "y" * 34 not in text
    meta = json.loads(text)
    assert meta["status"] == "running"
    for key in ("code", "environment", "inputs", "argv", "config_hash", "seed", "started_utc"):
        assert key in meta
    with pytest.raises(RuntimeError):
        with run.tracked_run(rd):
            raise RuntimeError("boom")
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and "boom" in meta["error"]


@pytest.mark.criterion("M0-3")
def test_secret_values_are_blocked(tmp_path):
    with pytest.raises(run.SecretLeakError):
        run.write_json_atomic(tmp_path / "x.json", {"note": "ghp_" + "a" * 36})


@pytest.mark.criterion("M0-3")
def test_meta_has_no_environment_dump(tmp_path, monkeypatch):
    """A planted environment variable value appears nowhere in meta.json (§4.6.3 기록 금지)."""
    planted = "planted-value-" + "z" * 24
    monkeypatch.setenv("ILORA_DUMMY_ENV", planted)
    rd = run.create_run_dir("smoke_cpu", "20261106T000002Z", root=tmp_path)
    run.start_meta(rd, profile="smoke_cpu", argv=["main.py", "check-env"], config_hash="abc", overrides={},
                   seed=1234)
    text = (rd / "meta.json").read_text(encoding="utf-8")
    assert planted not in text and "ILORA_DUMMY_ENV" not in text
    meta = json.loads(text)
    for key in ("status", "profile", "argv", "config_hash", "overrides", "seed", "started_utc", "ended_utc",
                "elapsed_s", "error", "code", "environment", "inputs"):
        assert key in meta, key
    assert {"commit", "dirty"} <= set(meta["code"]["git"]) and meta["code"]["rebuild_files"]
    env = meta["environment"]
    for key in ("python", "hostname", "packages", "cuda", "cudnn", "gpus", "gpu_count", "torch_num_threads"):
        assert key in env, key
    import torch

    assert env["torch_num_threads"] == torch.get_num_threads() >= 1          # 15.6.10: CPU bitwise needs equal counts
    for name in ("torch", "transformers", "tokenizers", "sentencepiece", "numpy", "pandas", "sklearn", "yaml",
                 "scipy", "joblib", "threadpoolctl"):
        assert name in env["packages"], name
    assert env["packages"]["numpy"] == "1.24.2" and env["packages"]["sklearn"] == "1.2.2"
    assert set(meta["inputs"]) == set(run.EXPECTED_INPUTS)


# ----------------------------------------------------------------------------
# M0-5 environment checks
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M0-5")
def test_input_hashes_match_expected():
    hashes = run.input_hashes()
    for rel, (expected, _) in run.EXPECTED_INPUTS.items():
        assert hashes[rel] == expected, rel


@pytest.mark.criterion("M0-5")
def test_environment_check_reports_problems(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.31.0")
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}), encoding="utf-8")
    with pytest.raises(run.EnvironmentCheckError) as err:
        run.check_environment(require_gpu=False, llm_path=llama)
    msg = str(err.value)
    assert "transformers 4.31.0" in msg and "torch_dtype" in msg and "no weight shards" in msg


@pytest.mark.criterion("M0-5")
def test_environment_check_detects_hash_mismatch(tmp_path, monkeypatch):
    fake = {"data/x.df": ("0" * 64, False)}
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.df").write_bytes(b"abc")
    monkeypatch.setattr(run, "EXPECTED_INPUTS", fake)
    with pytest.raises(run.EnvironmentCheckError, match="hash mismatch"):
        run.check_environment(require_gpu=False, llm_path=None, repo_root=tmp_path)


@pytest.mark.criterion("M0-5")
def test_text_hash_ignores_crlf(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"x\r\ny\r\n")
    b.write_bytes(b"x\ny\n")
    assert run.content_sha256(a, text=True) == run.content_sha256(b, text=True)


class _GpuProps:
    def __init__(self, gib: float) -> None:
        self.name = "NVIDIA A100-SXM4-80GB"
        self.total_memory = int(gib * 2**30)


def _fake_gpus(monkeypatch, count: int, gib: float = 80.0) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _GpuProps(gib))


@pytest.mark.criterion("M0-5")
@pytest.mark.parametrize("count", [0, 2])
def test_environment_check_gpu_count(count, monkeypatch):
    _fake_gpus(monkeypatch, count)
    with pytest.raises(run.EnvironmentCheckError, match=f"exactly one visible GPU is required .* found {count}"):
        run.check_environment(require_gpu=True, llm_path=None, check_inputs=False)


@pytest.mark.criterion("M0-5")
def test_environment_check_gpu_memory_and_cuda(monkeypatch):
    import torch

    _fake_gpus(monkeypatch, 1, gib=80.0)
    run.check_environment(require_gpu=True, llm_path=None, check_inputs=False)      # one A100 80GB passes
    _fake_gpus(monkeypatch, 1, gib=40.0)
    with pytest.raises(run.EnvironmentCheckError, match="GPU 0 has 40.0 GiB"):
        run.check_environment(require_gpu=True, llm_path=None, check_inputs=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(run.EnvironmentCheckError, match="CUDA is not available"):
        run.check_environment(require_gpu=True, llm_path=None, check_inputs=False)


@pytest.mark.criterion("M0-5")
def test_environment_check_torch_version(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.1.0+cu118")
    with pytest.raises(run.EnvironmentCheckError, match=r"torch 2\.1\.0\+cu118 != 2\.0\.0"):
        run.check_environment(require_gpu=False, llm_path=None, check_inputs=False)


@pytest.mark.criterion("M0-5")
def test_environment_check_missing_llama_path(tmp_path, monkeypatch):
    with pytest.raises(run.EnvironmentCheckError, match="LLaMA path is not a directory"):
        run.check_environment(require_gpu=False, llm_path=tmp_path / "no_llama", check_inputs=False)
    with pytest.raises(run.EnvironmentCheckError, match="LLM_PATH is not set"):
        run.check_environment(require_gpu=False, llm_path=None, require_llm=True, check_inputs=False)
    monkeypatch.delenv("LLM_PATH", raising=False)
    cfg = _profile("paper_spec")
    assert commands.llm_path_for(cfg) is None
    _fake_gpus(monkeypatch, 1)
    with pytest.raises(run.EnvironmentCheckError, match="LLM_PATH is not set"):
        commands.environment_check(cfg, repo_root=run.REPO_ROOT, llm_path=None)
    monkeypatch.setenv("LLM_PATH", str(tmp_path))
    assert commands.llm_path_for(cfg) == tmp_path
    assert commands.llm_path_for(_profile("smoke_cpu")) is None       # tiny_random never uses LLaMA


@pytest.mark.criterion("M0-5")
@pytest.mark.parametrize("module_name", ["numpy", "pandas", "sentencepiece", "sklearn"])
def test_environment_check_library_pins(module_name, monkeypatch):
    import importlib

    # Import everything the check imports before faking a version (pandas and scikit-learn
    # read numpy.__version__ at import time).
    for name in (*run.REQUIRED_LIBRARIES, "transformers"):
        importlib.import_module(name)
    module = importlib.import_module(module_name)
    required = run.REQUIRED_LIBRARIES[module_name]
    assert module.__version__ == required                       # the local test environment is pinned
    monkeypatch.setattr(module, "__version__", "0.0.1")
    with pytest.raises(run.EnvironmentCheckError, match=f"{module_name} 0.0.1 != {required}"):
        run.check_environment(require_gpu=False, llm_path=None, check_inputs=False)


def _fake_site(root: Path) -> tuple[Path, str]:
    """A wheel-like install of a package called transformers with a matching RECORD."""
    files = {"transformers/__init__.py": b"__version__ = '4.28.0'\n",
             "transformers/generation/utils.py": b"def greedy():\n    return 1\n",
             "transformers/models/llama/modeling_llama.py": b"class LlamaModel:\n    pass\n"}
    lines = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        lines.append(f"{rel},sha256={run._b64_sha256(data)},{len(data)}")
    lines += ["transformers-4.28.0.dist-info/RECORD,,", "transformers/__pycache__/__init__.cpython-310.pyc,,"]
    return root / "transformers", "\n".join(lines) + "\n"


@pytest.mark.criterion("M0-5")
def test_environment_check_patched_transformers(tmp_path, monkeypatch):
    """A transformers file that differs from its wheel RECORD (fork debug/ copies) fails the check (D13)."""
    assert run.transformers_integrity_problems() == []            # the installed 4.28.0 is stock
    site = tmp_path / "site"
    package, record = _fake_site(site)
    assert run.record_problems(site, record, package, "transformers") == []
    monkeypatch.setattr(run, "_transformers_install", lambda: (site, record, package))
    run.check_environment(require_gpu=False, llm_path=None, check_inputs=False)
    (package / "generation" / "utils.py").write_bytes(b"def greedy():\n    return 2  # patched\n")
    (package / "extra.py").write_bytes(b"x = 1\n")
    (package / "models" / "llama" / "modeling_llama.py").unlink()
    with pytest.raises(run.EnvironmentCheckError) as err:
        run.check_environment(require_gpu=False, llm_path=None, check_inputs=False)
    msg = str(err.value)
    assert "transformers file transformers/generation/utils.py differs from the wheel RECORD" in msg
    assert "transformers/extra.py is not listed in the wheel RECORD" in msg
    assert "transformers/models/llama/modeling_llama.py from the wheel RECORD is missing" in msg
    monkeypatch.setattr(run, "_transformers_install", lambda: (site, record, tmp_path / "elsewhere"))
    with pytest.raises(run.EnvironmentCheckError, match="is not the installation its wheel RECORD describes"):
        run.check_environment(require_gpu=False, llm_path=None, check_inputs=False)


# ----------------------------------------------------------------------------
# M0-6 abnormal endings
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M0-6")
def test_keyboard_interrupt_marks_failed(tmp_path):
    rd = run.create_run_dir("smoke_cpu", "20261106T000004Z", root=tmp_path)
    run.start_meta(rd, profile="smoke_cpu", argv=["main.py"], config_hash="abc", overrides={}, seed=1234)
    with pytest.raises(KeyboardInterrupt):
        with run.tracked_run(rd):
            raise KeyboardInterrupt()
    meta = run.read_meta(rd)
    assert meta["status"] == "failed" and "KeyboardInterrupt" in meta["error"]
    assert meta["ended_utc"] is not None and meta["elapsed_s"] >= 0


@pytest.mark.criterion("M0-6")
def test_killed_process_stays_running(tmp_path):
    """A process killed inside tracked_run leaves status running and ended_utc null (distinguishable)."""
    code = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(REBUILD_DIR)!r})
        from ilora import run
        rd = run.create_run_dir("killed", "20261106T000005Z", root=Path({str(tmp_path)!r}))
        run.start_meta(rd, profile="killed", argv=["child"], config_hash="abc", overrides={{}}, seed=1234)
        with run.tracked_run(rd):
            (rd / "ready").write_text("1")
            time.sleep(600)
    """)
    proc = subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    rd = tmp_path / "killed_20261106T000005Z"
    try:
        deadline = time.monotonic() + 180
        while not (rd / "ready").exists():
            assert proc.poll() is None, proc.stderr.read().decode(errors="replace")
            assert time.monotonic() < deadline, "child did not start"
            time.sleep(0.1)
    finally:
        proc.kill()
        proc.wait(timeout=60)
        proc.stderr.close()
    meta = run.read_meta(rd)
    assert meta["status"] == "running" and meta["ended_utc"] is None and meta["error"] is None


@pytest.mark.criterion("M0-6")
def test_run_lock_has_one_holder(tmp_path):
    """``run.lock`` (M0 ``hold_run_lock``): while one holder has it a second one is refused with the holder line, it
    is free again after the block, and the file only grows (one line per holder, never truncated)."""
    rd = run.create_run_dir("smoke_cpu", "20261106T000010Z", root=tmp_path)
    with run.hold_run_lock(rd, "first") as lock:
        assert lock.held and lock.problem is None and lock.path == rd / run.RUN_LOCK_FILE
        with pytest.raises(run.RunLockError, match=r"in use by a live process \(run\.lock: .* pid=\d+ first\)"):
            with run.hold_run_lock(rd, "second"):
                pass
    with run.hold_run_lock(rd, "third") as lock:
        assert lock.held
    lines = (rd / run.RUN_LOCK_FILE).read_text(encoding="utf-8").splitlines()
    assert [line.split()[-1] for line in lines] == ["first", "third"]
    assert all(f"pid={os.getpid()}" in line for line in lines)


def _wait_for_lock_release(run_dir: Path, timeout: float = 60.0) -> None:
    """The OS releases the lock of a killed process when its handles close (on Windows not always at once)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with run.hold_run_lock(run_dir, "test probe"):
                return
        except run.RunLockError:
            assert time.monotonic() < deadline, "run.lock still held after the process was killed"
            time.sleep(0.1)


@pytest.mark.criterion("M0-6")
def test_resume_refuses_a_live_run(tmp_path, stand_ins):
    """A train run whose process is alive (status ``running``, ``run.lock`` held, as after a shell disconnect inside
    tmux) is not reopened: ``run_command(resume=...)`` raises CommandError and meta.json, config.yaml and run.log stay
    byte for byte. After the process is killed (status stays ``running``, §4.8 #6) the same call reopens it with
    ``previous_status`` running (§14.6.10 제안 4)."""
    calls, _ = stand_ins
    code = textwrap.dedent(f"""
        import sys, time, types
        from pathlib import Path
        sys.path.insert(0, {str(REBUILD_DIR)!r})

        def train(cfg, run_dir, *, repo_root, llm_path, resume=False):
            (Path(run_dir) / "ready").write_text("1")
            time.sleep(600)

        for name, attrs in (("train", {{"train": train}}), ("candidates", {{"self_check": lambda: None}})):
            module = types.ModuleType("ilora." + name)
            module.__dict__.update(attrs)
            sys.modules["ilora." + name] = module
        from ilora import commands
        from ilora.config import load_config

        cfg, _ = load_config(Path({str(CONFIG_DIR / "smoke_cpu.yaml")!r}))
        commands.run_command("train", cfg, [], profile="smoke_cpu", root=Path({str(tmp_path)!r}), argv=["child"],
                             stamp="20261106T000011Z")
    """)
    proc = subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    rd = tmp_path / "smoke_cpu_20261106T000011Z"
    cfg = _profile("smoke_cpu")
    try:
        deadline = time.monotonic() + 180
        while not (rd / "ready").exists():
            assert proc.poll() is None, proc.stderr.read().decode(errors="replace")
            assert time.monotonic() < deadline, "child did not start"
            time.sleep(0.1)
        before = {name: (rd / name).read_bytes() for name in ("meta.json", "config.yaml", "run.log")}
        assert json.loads(before["meta.json"])["status"] == "running"
        with pytest.raises(commands.CommandError, match="in use by a live process.*tmux attach -t rebuild"):
            commands.run_command("train", cfg, [], profile="smoke_cpu", root=tmp_path, resume=rd, argv=["resume"])
        assert {name: (rd / name).read_bytes() for name in before} == before
        assert not [c for c in calls if c[0] == "train"]
    finally:
        proc.kill()
        proc.wait(timeout=60)
        proc.stderr.close()
    _wait_for_lock_release(rd)
    assert commands.run_command("train", cfg, [], profile="smoke_cpu", root=tmp_path, resume=rd,
                                argv=["resume"]) == rd
    meta = run.read_meta(rd)
    assert meta["status"] == "completed"
    (resume,) = meta["resumes"]
    assert resume["previous_status"] == "running" and resume["previous_ended_utc"] is None
    assert ("train", rd, run.REPO_ROOT, None, True) in calls


# ----------------------------------------------------------------------------
# Precision tools (§3.1 precision 도구, §14.6.10 제안 5, D9)
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M10-15")
def test_autocast_for_modes(monkeypatch):
    import torch

    from ilora.precision import autocast_for

    with autocast_for("fp32", "cpu"):
        assert not torch.is_autocast_cpu_enabled() and not torch.is_autocast_enabled()
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32
    with autocast_for("bf16_amp", torch.device("cpu")):
        assert torch.is_autocast_cpu_enabled()
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.bfloat16
    assert not torch.is_autocast_cpu_enabled()
    with pytest.raises(ValueError, match="CUDA"):
        autocast_for("fp16_amp", "cpu")
    with pytest.raises(ValueError, match="unknown precision"):
        autocast_for("fp8", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="is_available"):
        autocast_for("fp16_amp", "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    region = autocast_for("fp16_amp", "cuda:0")
    assert region.device == "cuda" and region.fast_dtype == torch.float16
    with pytest.raises(ValueError, match="cuda and cpu"):
        autocast_for("bf16_amp", "meta")


@pytest.mark.criterion("M10-15")
def test_make_grad_scaler(monkeypatch):
    import torch

    from ilora.precision import make_grad_scaler

    cpu = _profile("smoke_cpu").train
    scaler = make_grad_scaler(cpu, "cpu")
    assert not scaler.is_enabled() and scaler.get_scale() == 1.0
    paper = _profile("paper_spec").train
    with pytest.raises(ValueError, match="CUDA"):
        make_grad_scaler(paper, "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    scaler = make_grad_scaler(paper, torch.device("cuda:0"))
    assert scaler.is_enabled()
    assert scaler.state_dict() == {"scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                                   "growth_interval": 2000, "_growth_tracker": 0}
    bf16 = _profile("variant_bf16").train
    assert not make_grad_scaler(bf16, "cuda:0").is_enabled()


# ----------------------------------------------------------------------------
# Command dispatch (§3.1 명령 분기) with stand-in modules for the not-yet-built ones
# ----------------------------------------------------------------------------
class _Calls(list):
    pass


@pytest.fixture
def stand_ins(monkeypatch):
    """Stand-in ``ilora.candidates``/``train``/``data``/``sasrec``/``batching``/``smoke`` modules."""
    calls = _Calls()
    state = {"train_error": None, "tokenizer_error": None, "smoke_passed": True, "verify": {"passed": True}}

    def module(name, **attrs):
        mod = types.ModuleType(f"ilora.{name}")
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, f"ilora.{name}", mod)
        return mod

    def self_check():
        calls.append(("self_check",))

    def train(cfg, run_dir, *, repo_root, llm_path, resume=False):
        calls.append(("train", run_dir, repo_root, llm_path, resume))
        if state["train_error"] is not None:
            raise state["train_error"]

    def load_dataset(cfg, repo_root):
        calls.append(("load_dataset",))
        catalog = types.SimpleNamespace(pad_id=1682, num_items=1682)
        return types.SimpleNamespace(subset=cfg.data.subset, train=(1, 2, 3), val=(1,), test=(1,), catalog=catalog)

    class Encoder:
        state_digest = "d" * 64

        @classmethod
        def from_config(cls, cfg, catalog, history_len):
            calls.append(("from_config", catalog.pad_id, history_len))
            return cls()

        def to(self, device):
            return self

    def verify_against_reference(encoder, reference_path, device):
        calls.append(("verify", str(device)))
        return dict(state["verify"])

    def load_tokenizer(tokenizer_dir, expected_sha256):
        calls.append(("load_tokenizer", Path(tokenizer_dir)))
        if state["tokenizer_error"] is not None:
            raise state["tokenizer_error"]
        return types.SimpleNamespace(model_sha256=expected_sha256, ids=types.SimpleNamespace(vocab_size=32005))

    def run_smoke(profile_path, overrides, run_dir, *, repo_root, llm_path, tokenizer_dir):
        calls.append(("run_smoke", Path(profile_path), list(overrides)))
        return types.SimpleNamespace(passed=state["smoke_passed"])

    module("candidates", self_check=self_check)
    module("train", train=train)
    module("data", load_dataset=load_dataset, HISTORY_LEN=10)
    module("sasrec", SASRecEncoder=Encoder, verify_against_reference=verify_against_reference)
    module("batching", load_tokenizer=load_tokenizer)
    module("smoke", run_smoke=run_smoke)
    return calls, state


@pytest.mark.criterion("M0-3")
def test_run_command_prepares_run_folder(tmp_path, stand_ins):
    calls, _ = stand_ins
    cfg = _profile("smoke_cpu", "train.epochs=1")
    rd = commands.run_command("train", cfg, ["train.epochs=1"], profile="smoke_cpu", root=tmp_path,
                              argv=["main.py", "train"], profile_path=CONFIG_DIR / "smoke_cpu.yaml",
                              stamp="20261106T000006Z")
    assert rd == tmp_path / "smoke_cpu_20261106T000006Z"
    again, _ = load_config(rd / "config.yaml")
    assert again == cfg
    meta = run.read_meta(rd)
    assert meta["status"] == "completed" and meta["error"] is None and meta["elapsed_s"] >= 0
    assert meta["command"] == "train" and meta["profile"] == "smoke_cpu"
    assert meta["overrides"] == {"train.epochs": 1} and meta["seed"] == 1234
    assert meta["config_hash"] == config_hash(cfg) and meta["resumes"] == []
    assert set(meta["inputs"]) == set(run.EXPECTED_INPUTS)
    assert set(meta["assets"]) == {"sasrec_state", "sasrec_reference", "tokenizer_model", "llama_config",
                                   "cluster_manifest"}
    assert meta["preparation"]["environment_check"] == "passed"
    assert meta["paths"]["prompt.template_path"] == str(run.REPO_ROOT / "prompt" / "movie.txt")
    first_line = (rd / "run.log").read_text(encoding="utf-8").splitlines()[0]
    assert f"run_dir={rd}" in first_line
    assert ("self_check",) in calls
    assert ("train", rd, run.REPO_ROOT, None, False) in calls
    assert calls.index(("self_check",)) < calls.index(("train", rd, run.REPO_ROOT, None, False))


@pytest.mark.criterion("M0-6")
def test_run_command_failure_marks_failed(tmp_path, stand_ins):
    _, state = stand_ins
    state["train_error"] = RuntimeError("boom in train")
    cfg = _profile("smoke_cpu")
    with pytest.raises(RuntimeError, match="boom in train"):
        commands.run_command("train", cfg, [], profile="smoke_cpu", root=tmp_path, argv=["t"])
    (rd,) = list(tmp_path.iterdir())
    meta = run.read_meta(rd)
    assert meta["status"] == "failed" and "boom in train" in meta["error"]
    assert "train failed" in (rd / "run.log").read_text(encoding="utf-8")


@pytest.mark.criterion("M0-5", "M0-6")
def test_run_command_environment_failure_is_recorded(tmp_path, stand_ins, monkeypatch):
    calls, _ = stand_ins
    monkeypatch.setattr(run, "EXPECTED_INPUTS", {"prompt/movie.txt": ("0" * 64, True)})
    with pytest.raises(run.EnvironmentCheckError, match="hash mismatch prompt/movie.txt"):
        commands.run_command("train", _profile("smoke_cpu"), [], profile="smoke_cpu", root=tmp_path, argv=["t"])
    (rd,) = list(tmp_path.iterdir())
    assert run.read_meta(rd)["status"] == "failed"
    assert not [c for c in calls if c[0] in ("train", "self_check")]


@pytest.mark.criterion("M0-6")
def test_resume_reopens_failed_runs_only(tmp_path, stand_ins):
    calls, state = stand_ins
    cfg = _profile("smoke_cpu")
    state["train_error"] = RuntimeError("stopped")
    with pytest.raises(RuntimeError):
        commands.run_command("train", cfg, [], profile="smoke_cpu", root=tmp_path, argv=["t"],
                             stamp="20261106T000007Z")
    rd = tmp_path / "smoke_cpu_20261106T000007Z"
    (rd / "checkpoints" / "epoch0").mkdir()
    (rd / "checkpoints" / "epoch0" / "COMPLETE").write_text("{}", encoding="utf-8")
    state["train_error"] = None
    with pytest.raises(commands.CommandError, match="config differs"):
        commands.run_command("train", _profile("smoke_cpu", "train.epochs=3"), [], profile="smoke_cpu",
                             root=tmp_path, resume=rd)
    loaded, profile = commands.load_run_config(rd)
    assert profile == "smoke_cpu" and loaded == cfg
    again = commands.run_command("train", loaded, [], profile=profile, root=tmp_path, resume=rd,
                                 argv=["main.py", "train", "--resume", str(rd)])
    assert again == rd and sorted(p.name for p in tmp_path.iterdir()) == [rd.name]
    meta = run.read_meta(rd)
    assert meta["status"] == "completed"
    (resume,) = meta["resumes"]
    assert resume["previous_status"] == "failed" and "stopped" in resume["previous_error"]
    assert resume["complete_epochs"] == [0] and resume["start_epoch"] == 1
    assert ("train", rd, run.REPO_ROOT, None, True) in calls
    with pytest.raises(commands.CommandError, match="only failed or unfinished"):
        commands.run_command("train", loaded, [], profile=profile, root=tmp_path, resume=rd)
    with pytest.raises(commands.CommandError, match="only for train"):
        commands.run_command("eval", loaded, [], profile=profile, root=tmp_path, resume=rd)


@pytest.mark.criterion("M0-1")
def test_command_needs_its_keys_before_a_run_folder(tmp_path):
    cfg = _profile("paper_spec")
    with pytest.raises(ConfigError, match="eval.source_run"):
        commands.run_command("eval", cfg, [], profile="paper_spec", root=tmp_path)
    with pytest.raises(ConfigError, match="records.source_run"):
        commands.run_command("export", cfg, [], profile="paper_spec", root=tmp_path)
    with pytest.raises(commands.CommandError, match="unknown command"):
        commands.run_command("fit", cfg, [], profile="paper_spec", root=tmp_path)
    with pytest.raises(commands.CommandError, match="profile_path"):
        commands.run_command("smoke", cfg, [], profile="paper_spec", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.criterion("M0-5")
def test_check_env_command(tmp_path, stand_ins):
    """check-env runs the shared checks, then data, SASRec and tokenizer checks; any failure fails the run."""
    calls, state = stand_ins
    cfg = _profile("smoke_cpu")
    rd = commands.run_command("check-env", cfg, [], profile="smoke_cpu", root=tmp_path / "ok", argv=["t"],
                              tokenizer_dir=tmp_path)
    report = json.loads((rd / "check_env.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert [c["check"] for c in report["checks"]] == ["data", "sasrec", "tokenizer"]
    sasrec = report["checks"][1]["result"]
    assert sasrec["source_sha256"] == cfg.sasrec.source_sha256 and sasrec["state_digest"] == "d" * 64
    assert ("from_config", 1682, 10) in calls and ("verify", "cpu") in calls
    assert ("load_tokenizer", tmp_path) in calls
    assert run.read_meta(rd)["status"] == "completed"

    state["tokenizer_error"] = ValueError("tokenizer.model sha256 mismatch")
    state["verify"] = {"passed": False, "max_abs_diff": 0.5}
    with pytest.raises(run.EnvironmentCheckError) as err:
        commands.run_command("check-env", cfg, [], profile="smoke_cpu", root=tmp_path / "bad", argv=["t"],
                             tokenizer_dir=tmp_path)
    assert "tokenizer.model sha256 mismatch" in str(err.value) and "SASRec reference" in str(err.value)
    (rd,) = list((tmp_path / "bad").iterdir())
    report = json.loads((rd / "check_env.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert [c["passed"] for c in report["checks"]] == [True, False, False]
    assert run.read_meta(rd)["status"] == "failed"


@pytest.mark.criterion("M0-5")
def test_check_sasrec_source_hash(tmp_path, stand_ins):
    cfg = _profile("smoke_cpu", f"sasrec.source_path={json.dumps((tmp_path / 'other.pt').as_posix())}")
    (tmp_path / "other.pt").write_bytes(b"not the released pickle")
    catalog = types.SimpleNamespace(pad_id=1682)
    with pytest.raises(run.EnvironmentCheckError, match="SASRec source sha256"):
        commands.check_sasrec(cfg, catalog, repo_root=run.REPO_ROOT, device="cpu")


@pytest.mark.criterion("M0-6")
def test_smoke_command_fails_when_checks_fail(tmp_path, stand_ins):
    calls, state = stand_ins
    cfg = _profile("smoke_cpu")
    profile_path = CONFIG_DIR / "smoke_cpu.yaml"
    rd = commands.run_command("smoke", cfg, [], profile="smoke_cpu", root=tmp_path / "ok", argv=["t"],
                              profile_path=profile_path)
    assert run.read_meta(rd)["status"] == "completed"
    assert ("run_smoke", profile_path, []) in calls
    state["smoke_passed"] = False
    with pytest.raises(commands.CommandError, match="smoke checks failed"):
        commands.run_command("smoke", cfg, ["train.epochs=1"], profile="smoke_cpu", root=tmp_path / "bad",
                             argv=["t"], profile_path=profile_path)
    (rd,) = list((tmp_path / "bad").iterdir())
    assert run.read_meta(rd)["status"] == "failed"


@pytest.mark.criterion("M0-3")
def test_nested_runs_restore_logging(tmp_path, stand_ins):
    """A run started inside another (smoke steps) logs to its own run.log; the outer handlers come back."""
    import logging

    outer = run.create_run_dir("outer", "20261106T000008Z", root=tmp_path)
    run.setup_logging(outer)
    try:
        commands.run_command("train", _profile("smoke_cpu"), [], profile="S1_train", root=outer / "steps",
                             argv=["t"])
        logging.getLogger("ilora.test").info("after the step")
    finally:
        for handler in list(logging.getLogger("ilora").handlers):
            logging.getLogger("ilora").removeHandler(handler)
            handler.close()
    (step,) = list((outer / "steps").iterdir())
    assert "run_dir=" in (step / "run.log").read_text(encoding="utf-8").splitlines()[0]
    assert "after the step" not in (step / "run.log").read_text(encoding="utf-8")
    assert "after the step" in (outer / "run.log").read_text(encoding="utf-8")
