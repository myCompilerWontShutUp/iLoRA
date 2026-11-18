"""Steam ladder, stages S0 and S1 (DESIGN.md §24, D45-D47; rebuild/STEAM_LADDER.md).

S0: ``tools/sasrec_baseline.py``, the frozen SASRec alone on the fixed evaluation candidates, scored by the
released output layer ``s_fc``. S1: ``main.py zeroshot``, the untrained base LLM with the text-only prompt, through
``commands.run_command`` on the tiny random LLaMA. Also the stage profiles of ``configs/steam_ladder/`` and the bash
blocks of STEAM_LADDER.md. No accuracy is judged here; the counts below are computed values of this repository's
files (seed 1234).
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import pickle
import re
import shlex
import sys
from pathlib import Path

import pytest
import torch

from conftest import SKIP_NO_SASREC, sasrec_overrides
from ilora import commands, run
from ilora.config import ConfigError, load_config
from ilora.datasets import spec_of

pytestmark = [pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated"),
              pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")]          # torch 2.0.0 torch.load

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
CONFIG_DIR = REBUILD_DIR / "configs"
LADDER_DIR = CONFIG_DIR / "steam_ladder"
LADDER_DOC = REBUILD_DIR / "STEAM_LADDER.md"
TOOL_PATH = REBUILD_DIR / "tools" / "sasrec_baseline.py"

# S0 (§24.3): correct / n per split, tools/sasrec_baseline.py with seed 1234 (computed 2026-11-16).
BASELINE = {
    "movielens": (CONFIG_DIR / "paper_spec.yaml", {"val": (41, 94), "test": (32, 95)}),
    "steam": (CONFIG_DIR / "datasets" / "steam_paper_spec.yaml", {"val": (485, 1184), "test": (498, 1186)}),
}
# S1 (§24.4) on the CPU: the Steam smoke subset with the settings of steam_s1_zero_shot.yaml.
ZERO_SETS = ("lora.num_experts=1", "router.routing_mode=none", "prompt.eval_prompt=text_only")
STAGE_PROFILES = {                       # stem -> (lora.num_experts, router.routing_mode, prompt.eval_prompt)
    "steam_s1_zero_shot": (1, "none", "text_only"),
    "steam_s3_llara": (1, "none", "hybrid"),
    "steam_s4_two_experts": (2, "dynamic", "hybrid"),
    "steam_s5_ilora": (4, "dynamic", "hybrid"),
}


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    module = load_script(TOOL_PATH, "sasrec_baseline_under_test")
    yield module
    sys.modules.pop(module.__name__, None)


def baseline_cfg(dataset: str):
    """The dataset's paper profile, or a documented skip when its converted SASRec files are not there."""
    path, _ = BASELINE[dataset]
    cfg, _ = load_config(path, sasrec_overrides() if dataset == "movielens" else [])
    if not run.resolve_path(cfg.sasrec.state_path, REPO_ROOT).is_file():
        pytest.skip(SKIP_NO_SASREC)
    return cfg


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------- S0
@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(BASELINE))
def test_sasrec_baseline_counts(tool, dataset):
    """§24.3: hits of the frozen SASRec on the fixed candidates; the drop_last view is the same count on the first
    whole batches; every hit or miss is decided by a margin far above float32 rounding of z."""
    cfg = baseline_cfg(dataset)
    result = tool.sasrec_baseline(cfg, REPO_ROOT)
    assert result["dataset"] == dataset == spec_of(cfg).name and result["seed"] == 1234
    assert result["encoder_state_digest"] == cfg.sasrec.state_digest
    assert result["source_pickle"]["sha256"] == cfg.sasrec.source_sha256
    assert result["chance_level"] == 1 / 20
    for split, (correct, n) in BASELINE[dataset][1].items():
        r = result["splits"][split]
        assert (r["correct"], r["n"]) == (correct, n), split
        assert r["hr_at_1"] == correct / n
        low, high = r["wilson95"]
        assert low < r["hr_at_1"] < high
        d = r["drop_last"]
        assert d["batch_size"] == cfg.eval.batch_size and d["n"] == n - n % cfg.eval.batch_size
        assert 0 <= correct - d["correct"] <= n - d["n"]
        assert 1.0 <= r["mean_rank_of_target"] <= 20.0
        assert r["min_margin_target_vs_best_other"] > 1e-5


def test_output_layer_is_pinned_and_read_without_code(tool, tmp_path):
    """§24.3: the pickle must be the one the profile pins, and the unpickler admits only the known globals."""
    cfg, _ = load_config(BASELINE["movielens"][0])
    source = run.resolve_path(cfg.sasrec.source_path, REPO_ROOT)
    with pytest.raises(tool.BaselineError, match="sha256"):
        tool.load_output_layer(source, "0" * 64, 1682)
    with pytest.raises(tool.BaselineError, match="not found"):
        tool.load_output_layer(tmp_path / "missing.pt", cfg.sasrec.source_sha256, 1682)
    with pytest.raises(tool.BaselineError, match="do not fit"):
        tool.load_output_layer(source, cfg.sasrec.source_sha256, 1683)
    weight, bias, sha256 = tool.load_output_layer(source, cfg.sasrec.source_sha256, 1682)
    assert weight.shape == (1682, 64) and bias.shape == (1682,) and sha256 == cfg.sasrec.source_sha256
    other = tmp_path / "other.pt"
    torch.save({"counter": collections.Counter("abc")}, other)
    with pytest.raises(pickle.UnpicklingError, match="allow-list"):
        tool.load_output_layer(other, hashlib.sha256(other.read_bytes()).hexdigest(), 1682)


@pytest.mark.slow
def test_baseline_run_folder(tool, tmp_path, sasrec_state_path):
    """§24.3: one run folder ``<dataset>_sasrec_baseline_<UTC>`` with meta.json and baseline.json; the printed lines
    carry the counts of the file."""
    out = tool.run_baseline(BASELINE["movielens"][0], sasrec_overrides(), root=tmp_path, repo_root=REPO_ROOT,
                            argv=["sasrec_baseline.py", "--config", "paper_spec.yaml"])
    assert out.parent == tmp_path and re.fullmatch(r"movielens_sasrec_baseline_\d{8}T\d{6}Z", out.name)
    meta = read_json(out / "meta.json")
    assert (meta["status"], meta["command"], meta["seed"]) == ("completed", "sasrec-baseline", 1234)
    assert set(meta["inputs"]) == set(run.EXPECTED_INPUTS)
    result = read_json(out / tool.RESULT_FILE)
    assert {s: (r["correct"], r["n"]) for s, r in result["splits"].items()} == BASELINE["movielens"][1]
    lines = tool.format_lines(result)
    assert len(lines) == 3 and "val: correct 41/94" in lines[1] and "test: correct 32/95" in lines[2]


# ---------------------------------------------------------------------------------------------- S1
def steam_smoke_cfg(*sets: str):
    cfg, _ = load_config(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml", list(sets))
    if not run.resolve_path(cfg.sasrec.state_path, REPO_ROOT).is_file():
        pytest.skip(SKIP_NO_SASREC)
    return cfg


def test_zero_shot_requirements():
    """§24.4, D46: zeroshot needs one expert without a gate and the text-only prompt."""
    cfg, _ = load_config(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml")
    problems = commands.zero_shot_problems(cfg)
    assert len(problems) == 2 and "routing_mode" in problems[0] and "eval_prompt" in problems[1]
    hybrid, _ = load_config(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml", list(ZERO_SETS[:2]))
    assert len(commands.zero_shot_problems(hybrid)) == 1
    ok, _ = load_config(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml", list(ZERO_SETS))
    assert commands.zero_shot_problems(ok) == []
    assert "zeroshot" in commands.COMMANDS and commands.COMMAND_FUNCTIONS["zeroshot"] == ("evaluate",
                                                                                           "run_zero_shot_command")


def test_zeroshot_is_refused_before_a_run_folder(tmp_path):
    """A config that is not the base model stops with ConfigError and leaves nothing behind."""
    cfg, _ = load_config(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml")
    with pytest.raises(ConfigError, match="zeroshot"):
        commands.run_command("zeroshot", cfg, [], profile="zero", root=tmp_path, argv=["test_ladder"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_zeroshot_command_on_the_tiny_model(tmp_path, tokenizer_dir):
    """§24.4: the command evaluates validation and test of the seed-built model under ``eval/init/``: text-only
    records, the constant gate of one expert, no checkpoint named anywhere, and a summary of both splits."""
    cfg = steam_smoke_cfg(*ZERO_SETS)
    out = commands.run_command("zeroshot", cfg, list(ZERO_SETS), profile="steam_s1_zero_shot", root=tmp_path,
                               argv=["test_ladder", "zeroshot"], tokenizer_dir=tokenizer_dir)
    meta = read_json(out / "meta.json")
    assert (meta["status"], meta["command"]) == ("completed", "zeroshot") and "source_run" not in meta
    summary = read_json(out / "eval" / "summary.json")
    assert (summary["stage"], summary["dataset"], summary["routing_mode"], summary["eval_prompt"]) == \
        ("zero_shot", "steam", "none", "text_only")
    assert summary["lora_b_all_zero"] is True and summary["lora_wrapped_layers"] == 14     # tiny LLaMA: 2 x 7
    assert list(summary["results"]) == ["init/val", "init/test"]
    assert sorted(p.name for p in (out / "eval").iterdir()) == ["init", "summary.json"]
    assert not any((out / "checkpoints").iterdir())
    for split in ("val", "test"):
        folder = out / "eval" / "init" / split
        metrics = read_json(folder / "metrics.json")
        assert (metrics["ckpt_tag"], metrics["epoch"], metrics["n_total"]) == ("init", None, 10)
        assert metrics["meta"]["eval_prompt"] == "text_only" and metrics["meta"]["dataset"] == "steam"
        assert metrics["meta"]["ckpt_meta"] == {"source_run": None, "checkpoint_dir": None, "epoch": None,
                                                "tags": ["init"], "trainable_sha256": None}
        records = [json.loads(line) for line in (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(records) == 10 and {r["hybrid"] for r in records} == {False}
        assert {r["gate_0"] for r in records} == {1.0} and all("gate_1" not in r for r in records)
        result = summary["results"][f"init/{split}"]
        assert result["metrics"]["n_total"] == 10 and result["files"]["records.jsonl"] == run.content_sha256(
            folder / "records.jsonl")


@pytest.mark.slow
def test_zeroshot_refuses_a_trained_lora(tmp_path, tokenizer_dir, monkeypatch):
    """§24.4 step 3: a single non-zero ``lora_B`` value means the model is not the base model; nothing is evaluated."""
    from ilora import model as model_module
    from ilora.evaluate import EvalError
    from ilora.moe_lora import MoELoRALinear

    cfg = steam_smoke_cfg(*ZERO_SETS)
    real = model_module.build_model

    def trained(*args, **kwargs):
        built = real(*args, **kwargs)
        layer = next(m for m in built.modules() if isinstance(m, MoELoRALinear))
        with torch.no_grad():
            layer.lora_B.weight[0, 0] = 1e-3
        return built

    monkeypatch.setattr(model_module, "build_model", trained)
    with pytest.raises(EvalError, match="lora_B"):
        commands.run_command("zeroshot", cfg, list(ZERO_SETS), profile="steam_s1_zero_shot", root=tmp_path,
                             argv=["test_ladder", "zeroshot"], tokenizer_dir=tokenizer_dir)
    (out,) = list(tmp_path.iterdir())
    assert read_json(out / "meta.json")["status"] == "failed" and not any((out / "eval").iterdir())


# ---------------------------------------------------------------------------------------------- profiles, document
def test_stage_profiles():
    """§24.5: the stage profiles are Steam profiles with the stage's expert count, routing and evaluation prompt;
    every stem is unique among all profile folders; only S1 is a zero-shot profile."""
    assert sorted(p.stem for p in LADDER_DIR.iterdir()) == sorted(STAGE_PROFILES)
    others = [p.stem for p in (*CONFIG_DIR.glob("*.yaml"), *(CONFIG_DIR / "datasets").glob("*.yaml"))]
    assert not set(STAGE_PROFILES) & set(others)
    base, _ = load_config(CONFIG_DIR / "datasets" / "steam_paper_spec.yaml")
    for stem, (experts, routing, eval_prompt) in STAGE_PROFILES.items():
        text = (LADDER_DIR / f"{stem}.yaml").read_text(encoding="utf-8")
        assert text.isascii() and text.startswith(f"# {stem}.yaml"), stem
        cfg, _ = load_config(LADDER_DIR / f"{stem}.yaml")
        assert spec_of(cfg).name == "steam", stem
        assert (cfg.lora.num_experts, cfg.router.routing_mode, cfg.prompt.eval_prompt) == (experts, routing,
                                                                                           eval_prompt), stem
        for section in ("train", "sasrec", "candidates", "eval", "data"):
            assert getattr(cfg, section) == getattr(base, section), (stem, section)
        assert (commands.zero_shot_problems(cfg) == []) == (stem == "steam_s1_zero_shot"), stem


def ladder_commands() -> list[list[str]]:
    blocks = re.findall(r"```bash\n(.*?)```", LADDER_DOC.read_text(encoding="utf-8"), flags=re.S)
    lines = [line for block in blocks for line in block.splitlines() if line.strip() and not line.startswith("#")]
    return [shlex.split(line) for line in lines]


def test_ladder_document_commands():
    """rebuild/STEAM_LADDER.md: every bash line is ``python -I`` on main.py or the S0 tool, parses with the real
    parser, names an existing profile that loads, and uses the command of its stage."""
    main = load_script(REBUILD_DIR / "main.py", "rebuild_main_for_ladder")
    baseline = load_script(TOOL_PATH, "sasrec_baseline_for_ladder")
    try:
        seen = []
        for argv in ladder_commands():
            assert argv[:2] == ["python", "-I"], argv
            script, rest = argv[2], argv[3:]
            if script == "rebuild/main.py":
                ns = main.build_parser().parse_args(rest)
                assert ns.command in ("zeroshot", "train") and ns.config.parent == Path("rebuild/configs/steam_ladder")
                cfg, _ = load_config(REPO_ROOT / ns.config, ns.sets)
                assert (ns.command == "zeroshot") == (commands.zero_shot_problems(cfg) == []), argv
                commands.command_requirements(ns.command, cfg)
                seen.append(ns.config.stem)
            else:
                assert script == "rebuild/tools/sasrec_baseline.py", argv
                ns = baseline.build_parser().parse_args(rest)
                cfg, _ = load_config(REPO_ROOT / ns.config, ns.sets)
                assert spec_of(cfg).name == "steam"
                seen.append("s0")
        assert seen == ["s0", *sorted(STAGE_PROFILES)]            # one command per stage, in stage order
    finally:
        for name in ("rebuild_main_for_ladder", "sasrec_baseline_for_ladder"):
            sys.modules.pop(name, None)


def test_main_accepts_zeroshot():
    main = load_script(REBUILD_DIR / "main.py", "rebuild_main_for_zeroshot")
    try:
        ns = main.build_parser().parse_args(["zeroshot", "--config", "x.yaml"])
        assert ns.command == "zeroshot"
        assert main.main(["zeroshot", "--config", str(CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml")]) == 2
    finally:
        sys.modules.pop("rebuild_main_for_zeroshot", None)
