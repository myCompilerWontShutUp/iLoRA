"""M14 비교 도구 (DESIGN.md §18.6.7-§18.6.9, §18.8 #7): ``rebuild/tools/compare_runs.py``.

The tool is loaded from its file (like ``test_m3_convert.py``). The fast tests build run folders with the writers of
the commands themselves: M0 ``create_run_dir``/``start_meta``/``finish_meta``/``dump_config``, M11
``compute_metrics``/``aux_protocol_metrics`` + M12 ``write_eval`` (``records.jsonl``, ``metrics.json``), M10
``selection_document``/``write_marker`` (``eval/selection.json``), M12 ``TrainRecorder.close``
(``run_summary.json``) and M12 ``gate_summary``/``write_gate_summary``; the eval run's ``summary.json`` (M11 15.6.11)
and the M15 ``cluster/`` files follow their documented keys. ``test_compare_from_cpu_smoke`` (slow) reads the real
steps S1 (E2), S3 (E2 export), S4 (cluster), S5 (E4), S6 (E3) and S7 (E1) of M13 ``cpu_smoke_run``.
Synthetic inputs come from ``seeding.np_rng(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pytest

from ilora import run, seeding
from ilora.config import config_hash, dump_config, load_config

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
TOOL_PATH = REBUILD_DIR / "tools" / "compare_runs.py"
PAPER_NUMBERS = ("0.9684", "0.4421", "0.9891", "0.5275")     # §18.6.7 table D, nowhere else (§18.8 #7)
N_TEST = 12                                                  # synthetic test split: drop_last 12 (bs4) and 8 (bs8)
METRIC_FIELDS = ("n_total", "n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")
INT_FIELDS = ("n_total", "n_valid", "n_correct")
VALUE_COLUMNS = ("routing_mode", "num_moe", "lora_r", "seed", "epoch", *METRIC_FIELDS, "trainable_params", "elapsed_s",
                 "wall_clock_s", "resumes", "source_path", "source_sha256")
ARTIFACT = "runs/rebuild/cluster_run_for_tests/cluster"      # any path: the synthetic E4 run never reads it
TRAINABLE = {"dynamic": {"projector": 8320, "gate": 256, "lora": 17408, "total": 25984},   # tiny model (M10 notes)
             "other": {"projector": 8320, "lora": 17408, "total": 25728}}
# M10 TrainSummary.seconds of a synthetic training run (§14.6.7): training, validation, save and test time
TRAIN_SECONDS = {"train": 10.5, "validation": 2.25, "save": 0.5, "test": 1.25, "total": 14.5}


@pytest.fixture(scope="module")
def tool() -> Any:
    """``rebuild/tools/compare_runs.py`` imported from its file (registered so its dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location("compare_runs_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


# ----------------------------------------------------------------------------
# Synthetic run folders in the formats the commands write
# ----------------------------------------------------------------------------
def _profile(*sets: str) -> Any:
    cfg, _ = load_config(CONFIG_DIR / "smoke_cpu.yaml", list(sets))
    return cfg


def _start(root: Path, name: str, cfg: Any, command: str, **extra: Any) -> Path:
    run_dir = run.create_run_dir(name, root=root)
    dump_config(cfg, run_dir / "config.yaml")
    run.start_meta(run_dir, profile=name, argv=["synthetic"], config_hash=config_hash(cfg), overrides={},
                   seed=cfg.run.seed, extra={"command": command, **extra})
    return run_dir


def _finish(run_dir: Path, status: str = "completed") -> Path:
    run.finish_meta(run_dir, status, error=None if status == "completed" else "RuntimeError: synthetic failure")
    return run_dir


def make_records(correct: Iterable[int], valid: Iterable[int], *, num_experts: int, one_hot: bool = False,
                 hybrid: bool = True) -> list:
    """``N_TEST`` M11 ``EvalRecord`` of samples ``test:0..``: positions in ``correct`` answer the target, the other
    positions in ``valid`` name one wrong candidate, the rest name none. Candidates, templates and targets depend on
    the position only, so two runs share them (D10, D15)."""
    from ilora.candidates import FLAG_NAMES
    from ilora.evaluate import EvalRecord

    correct, valid = set(correct), set(valid) | set(correct)
    rng = seeding.np_rng(0, "test", "m14_compare_gates")
    out = []
    for i in range(N_TEST):
        cands = tuple(range(20 * i, 20 * i + 20))
        titles = tuple(f"Movie {c}" for c in cands)
        if one_hot or num_experts == 1:
            gates = tuple(1.0 if k == i % num_experts else 0.0 for k in range(num_experts))
        else:
            w = rng.random(num_experts) + 0.05
            gates = tuple(float(x) for x in w / w.sum())
        line = titles[0] if i in correct else titles[1] if i in valid else "no title"
        matched = (cands[0],) if i in correct else (cands[1],) if i in valid else ()
        out.append(EvalRecord(sample_id=f"test:{i}", split="test", user_id=848 + i, template_id=i % 3, hybrid=hybrid,
                              cand_ids=cands, cand_titles=titles, target_id=cands[0], target_title=titles[0],
                              raw_text=line + "\n", first_line=line, matched=matched, valid=i in valid,
                              correct=i in correct, gates=gates, flags={name: False for name in FLAG_NAMES},
                              gen_token_ids=(29871, 2)))
    return out


def write_test_eval(run_dir: Path, epoch: int, records: Sequence, num_experts: int) -> Path:
    """``eval/epoch{e}/test/`` with M12 ``write_eval`` (M11 metrics and auxiliary protocols)."""
    from ilora.evaluate import aux_protocol_metrics, compute_metrics
    from ilora.records import write_eval

    out = run_dir / "eval" / f"epoch{epoch}" / "test"
    out.mkdir(parents=True)
    metrics = {**dataclasses.asdict(compute_metrics(records, "test")), "epoch": epoch, "ckpt_tag": f"epoch{epoch}",
               "aux_protocols": aux_protocol_metrics(records, "test"), "meta": {"source": "synthetic test run"}}
    write_eval(out, records, metrics, expected_ids=[r.sample_id for r in records], num_experts=num_experts)
    return out


def make_train_run(root: Path, name: str, *, sets: Sequence[str] = (), best: int, last: int,
                   correct: Mapping[int, Iterable[int]], valid: Mapping[int, Iterable[int]], one_hot: bool = False,
                   status: str = "completed") -> Path:
    """A training run folder: ``config.yaml``, ``meta.json`` (command train), ``eval/selection.json`` (M10),
    ``eval/epoch{e}/test/`` of best_val and last, ``records/train/run_summary.json`` (M12 recorder)."""
    from ilora.evaluate import SplitMetrics
    from ilora.records import TrainRecorder
    from ilora.train import selection_document, write_marker

    cfg = _profile(*sets)
    k = cfg.lora.num_experts
    run_dir = _start(root, name, cfg, "train")
    for epoch in sorted({best, last}):
        write_test_eval(run_dir, epoch, make_records(correct[epoch], valid[epoch], num_experts=k, one_hot=one_hot), k)
    val = {e: SplitMetrics(split="val", n_total=10, n_valid=8, n_correct=5 if e == best else 1, valid_ratio=0.8,
                           hr_at_1=(5 if e == best else 1) / 8, combined_metric=(5 if e == best else 1) / 10)
           for e in range(last + 1)}
    write_marker(run_dir / "eval" / "selection.json", selection_document(val, best, last))
    mode = cfg.router.routing_mode
    recorder = TrainRecorder(run_dir, cfg.records, num_experts=k, total_micro_batches=1, total_steps=1)
    recorder.close("completed", {"routing_mode": mode,
                                 "trainable_params": TRAINABLE["dynamic" if mode == "dynamic" else "other"],
                                 "seconds": dict(TRAIN_SECONDS)})
    return _finish(run_dir, status)


def make_eval_run(root: Path, name: str, source: Path, *, correct: Mapping[int, Iterable[int]],
                  valid: Mapping[int, Iterable[int]]) -> Path:
    """An E3 eval run of ``source`` (text_only): its own ``eval/epoch{e}/test/`` and ``eval/summary.json`` with the
    keys of M11 ``run_eval_command`` (15.6.11)."""
    from ilora.records import SPLIT_METRICS_FIELDS

    cfg = _profile("prompt.eval_prompt=text_only", f"eval.source_run={json.dumps(source.as_posix())}")
    run_dir = _start(root, name, cfg, "eval", source_run=str(source))
    selection = json.loads((source / "eval" / "selection.json").read_text(encoding="utf-8"))
    best, last = selection["best_val_epoch"], selection["last_epoch"]
    results, listed = {}, []
    for epoch in sorted({best, last}):
        tags = [tag for tag, chosen in (("best_val", best), ("last", last)) if chosen == epoch]
        out = write_test_eval(run_dir, epoch, make_records(correct[epoch], valid[epoch], num_experts=4, hybrid=False), 4)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        listed.append({"ckpt_tag": f"epoch{epoch}", "source_run": str(source), "epoch": epoch, "tags": tags})
        results[f"epoch{epoch}/test"] = {
            "ckpt_tag": f"epoch{epoch}", "split": "test", "tags": tags,
            "metrics": {key: metrics[key] for key in SPLIT_METRICS_FIELDS}, "aux_protocols": metrics["aux_protocols"],
            "files": {f: run.content_sha256(out / f) for f in ("records.jsonl", "metrics.json")},
            "reproduction": {"status": "not_comparable", "differing_keys": ["prompt.eval_prompt"]}}
    run.write_json_atomic(run_dir / "eval" / "summary.json", {
        "source_run": str(source), "routing_mode": "dynamic", "splits": ["test"],
        "allowed_differences": ["prompt.eval_prompt: 'hybrid' != 'text_only'"],
        "reproduction_keys": ["eval.batch_size", "prompt.eval_prompt", "data.subset", "data.smoke_users"],
        "checkpoints": listed, "cluster_live_check": None, "results": results, "created_utc": run.utc_now().isoformat()})
    return _finish(run_dir)

def make_export_run(root: Path, source: Path, epoch: int) -> Path:
    """An E2 export run: ``records/export/epoch{e}/gates_test.csv`` and the test gate analysis written by M12
    ``gate_summary``/``write_gate_summary``/``write_interpretation`` with the four SPEC §20 figure files."""
    from ilora.analysis import gate_summary, write_gate_summary, write_interpretation

    cfg = _profile(f"records.source_run={json.dumps(source.as_posix())}")
    run_dir = _start(root, "export", cfg, "export", source_run=str(source))
    out = run_dir / "records" / "export" / f"epoch{epoch}"
    out.mkdir(parents=True)
    (out / "gates_test.csv").write_bytes(b"sample_id,split,row,gate_0,gate_1,gate_2,gate_3\n")
    omega = seeding.np_rng(0, "test", "m14_export_gates").dirichlet(np.ones(4), size=N_TEST).astype(np.float32)
    folder = run_dir / "records" / "analysis" / f"epoch{epoch}" / "test"
    summary = {"split": "test", "checkpoint": f"epoch{epoch}", "routing_mode": "dynamic", "source": {}}
    summary.update(gate_summary(omega, user_ids=None))
    write_gate_summary(folder, summary)
    write_interpretation(folder, summary)
    for name in ("gate_heatmap", "expert_mean_utilization", "argmax_expert_distribution", "entropy_distribution"):
        (folder / "figures").mkdir(exist_ok=True)
        (folder / "figures" / f"{name}.png").write_bytes(f"figure {name}".encode("ascii"))
    return _finish(run_dir)


def make_cluster_run(root: Path) -> Path:
    """A cluster run: ``cluster/`` with ``cluster_summary.json`` and ``manifest.json`` in the M15 keys (19.6.5-19.6.6),
    the joblib file and the two figures."""
    cfg = _profile()
    run_dir = _start(root, "cluster", cfg, "cluster")
    folder = run_dir / "cluster"
    (folder / "figures").mkdir(parents=True)
    (folder / "kmeans_k4_seed1234.joblib").write_bytes(b"synthetic kmeans")
    for name in ("sasrec_pca_clusters", "cluster_dynamic_gate_heatmap"):
        (folder / "figures" / f"{name}.png").write_bytes(f"figure {name}".encode("ascii"))
    sizes = seeding.np_rng(0, "test", "m14_cluster_sizes").multinomial(156, [0.25] * 4).tolist()
    run.write_json_atomic(folder / "cluster_summary.json", {
        "unit": "sequence", "n_clusters": 4, "fit": {"split": "train", "rows": 156},
        "train_cluster_size": sizes, "validation_cluster_size": [3, 3, 2, 2], "test_cluster_size": [3, 3, 3, 3],
        "silhouette_score_train": 0.125, "nmi_cluster_vs_dynamic_argmax": {"train": 0.5, "validation": 0.25,
                                                                           "test": 0.375},
        "dynamic_export": {"routing_mode": "dynamic", "z_max_abs_diff": {"train": 0.0, "validation": 0.0, "test": 0.0}},
        "reload_check": {"passed": True}})
    run.write_json_atomic(folder / "manifest.json", {"format": "synthetic", "kmeans_file": "kmeans_k4_seed1234.joblib"})
    return _finish(run_dir)


@pytest.fixture
def scenario(tmp_path: Path) -> dict[str, Path]:
    """E2 (dynamic, best 1, last 2), E1 (K = 1), E3 (text_only eval of E2), E4 (one-hot gates, best 0, last 1), the
    E2 export at epoch 1 and a cluster run. Correct positions (best_val): E2 {0-4}, E1 {0, 1, 5}, E4 {2, 3, 4, 9},
    E3 {0, 1}."""
    root = tmp_path / "runs"
    e2 = make_train_run(root, "e2", best=1, last=2, correct={1: range(5), 2: range(3)}, valid={1: range(9), 2: range(7)})
    e1 = make_train_run(root, "e1", sets=("lora.num_experts=1", "router.routing_mode=none"), best=0, last=0,
                        correct={0: (0, 1, 5)}, valid={0: (0, 1, 2, 5, 6)})
    e3 = make_eval_run(root, "e3", e2, correct={1: (0, 1), 2: (0,)}, valid={1: (0, 1, 2), 2: (0, 1)})
    e4 = make_train_run(root, "e4", sets=("router.routing_mode=cluster_hard", f"cluster.artifact_dir={ARTIFACT}"),
                        best=0, last=1, correct={0: (2, 3, 4, 9), 1: (1,)}, valid={0: (2, 3, 4, 9, 10), 1: (1, 2)},
                        one_hot=True)
    return {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E2X": make_export_run(root, e2, 1), "CL": make_cluster_run(root),
            "out": tmp_path / "compare"}


def run_tool(tool: Any, scenario: Mapping[str, Path], **override: Optional[Path]) -> Path:
    runs = {name: scenario[name] for name in ("E1", "E2", "E3", "E4")}
    runs.update({k: v for k, v in override.items() if k not in ("E2X", "CL")})
    return tool.run_compare(runs, {"E2": override.get("E2X", scenario["E2X"])}, override.get("CL", scenario["CL"]),
                            root=scenario["out"], argv=["compare_runs.py", "test"])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_rows_against_files(rows: Sequence[Mapping[str, str]], runs: Mapping[str, Path]) -> int:
    """Every measured cell equals the metrics.json it names, the epoch equals eval/selection.json, the sha256 the file;
    config and TrainableParams equal config.yaml and run_summary.json; ElapsedTime equals run_summary.json
    train_summary.seconds.total of a training run and meta.json elapsed_s of the E3 eval run (§18.6.7); the wall clock
    and resumes equal meta.json elapsed_s and its resumes. Returns the rows checked."""
    import yaml

    checked = 0
    labels = {"LLaRA baseline (E1)": "E1", "iLoRA Dynamic (E2)": "E2", "iLoRA text-only eval (E3)": "E3",
              "Cluster Hard (E4)": "E4"}
    for row in rows:
        name = labels[row["method"]]
        run_dir = runs[name]
        assert Path(row["run"]) == run_dir
        source = run_dir
        if name == "E3":
            source = Path(json.loads((run_dir / "eval" / "summary.json").read_text(encoding="utf-8"))["source_run"])
        selection = json.loads((source / "eval" / "selection.json").read_text(encoding="utf-8"))
        epoch = selection["best_val_epoch" if row["checkpoint"] == "best_val" else "last_epoch"]
        path = run_dir / "eval" / f"epoch{epoch}" / "test" / "metrics.json"
        assert Path(row["source_path"]) == path and row["source_sha256"] == run.content_sha256(path)
        assert int(row["epoch"]) == epoch
        metrics = json.loads(path.read_text(encoding="utf-8"))
        block = metrics if row["protocol"] == "full" else metrics["aux_protocols"][row["protocol"]]
        for key in METRIC_FIELDS:
            value = int(row[key]) if key in INT_FIELDS else float(row[key])
            assert value == block[key] and type(value) is type(block[key]), (row["method"], row["protocol"], key)
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        assert (row["routing_mode"], int(row["num_moe"]), int(row["lora_r"]), int(row["seed"])) == (
            config["router"]["routing_mode"], config["lora"]["num_experts"], config["lora"]["r"], config["run"]["seed"])
        summary = json.loads((source / "records" / "train" / "run_summary.json").read_text(encoding="utf-8"))
        assert int(row["trainable_params"]) == summary["train_summary"]["trainable_params"]["total"]
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        elapsed = meta["elapsed_s"] if name == "E3" else summary["train_summary"]["seconds"]["total"]
        assert float(row["elapsed_s"]) == elapsed, (row["method"], row["elapsed_s"], elapsed)
        assert float(row["wall_clock_s"]) == meta["elapsed_s"] and int(row["resumes"]) == len(meta.get("resumes") or [])
        checked += 1
    return checked


def check_paper_numbers(out: Path) -> None:
    """The four paper values appear in comparison.md table D only, never in the other outputs (§18.8 #7)."""
    for name in ("comparison.csv", "spec20_index.json", "summary.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert not [n for n in PAPER_NUMBERS if n in text], name
    text = (out / "comparison.md").read_text(encoding="utf-8")
    start = text.index("## Table D.")
    end = text.index("\n## ", start + 1)
    assert all(n in text[start:end] for n in PAPER_NUMBERS)
    outside = text[:start] + text[end:]
    assert not [n for n in PAPER_NUMBERS if n in outside]


def check_spec20(out: Path, expect_missing: Sequence[str] = ()) -> dict:
    """``spec20_index.json``: every listed file exists with the recorded sha256; ``expect_missing`` entries are
    NOT_RUN with a reason."""
    index = json.loads((out / "spec20_index.json").read_text(encoding="utf-8"))["files"]
    for spec, entry in index.items():
        if spec in expect_missing:
            assert entry["path"] == "NOT_RUN" and entry["sha256"] == "NOT_RUN" and entry["reason"], spec
        else:
            assert entry["sha256"] == run.content_sha256(Path(entry["path"])), spec
    return index

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M14-7")
def test_compare_values_and_sources(tool, scenario):
    """All four experiments: every comparison.csv cell equals the metrics.json it names (exactly), table E counts the
    samples of both records.jsonl, E4's evaluation gates are one-hot, the SPEC §20 index carries correct sha256 and the
    paper values stay in table D."""
    out = run_tool(tool, scenario)
    assert out.parent == scenario["out"] and out.name.startswith("compare_")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["command"] == "compare" and meta["profile"] == "compare"
    assert sorted(meta["compare"]["files"]) == sorted(tool.OUTPUT_FILES)
    for name, sha in meta["compare"]["files"].items():
        assert run.content_sha256(out / name) == sha, name
    rows = read_csv_rows(out / "comparison.csv")
    assert tuple(rows[0]) == tool.CSV_COLUMNS
    assert [(r["method"].split("(")[-1][:2], r["checkpoint"], r["protocol"]) for r in rows] == [
        (name, ckpt, protocol) for name in ("E1", "E2", "E3", "E4") for ckpt in ("best_val", "last")
        for protocol in ("full", "drop_last_bs4", "drop_last_bs8")]
    assert check_rows_against_files(rows, scenario) == 24
    full = {r["method"]: r for r in rows if r["checkpoint"] == "best_val" and r["protocol"] == "full"}
    assert full["iLoRA Dynamic (E2)"]["n_correct"] == "5" and full["Cluster Hard (E4)"]["epoch"] == "0"
    assert {r["n_total"] for r in rows if r["protocol"] == "drop_last_bs8"} == {"8"}
    report = tool.build_report({k: scenario[k] for k in ("E1", "E2", "E3", "E4")}, {"E2": scenario["E2X"]},
                               scenario["CL"])
    pairs = {p["pair"]: p for p in report.pairs}
    expected = {"E2 vs E1": (5, 3, 3, 1), "E4 vs E2": (4, 5, 1, 2), "E3 vs E2": (2, 5, 0, 3)}
    for pair, (ca, cb, only_a, only_b) in expected.items():
        p = pairs[pair]
        assert "problem" not in p, p
        assert (p["n"], p["correct_a"], p["correct_b"], p["only_a"], p["only_b"]) == (N_TEST, ca, cb, only_a, only_b)
        assert p["difference"] == ca - cb and p["discordant"] == only_a + only_b
        assert p["combined_ratio_minus_1"] == (ca / N_TEST) / (cb / N_TEST) - 1
    assert report.gates_e4["one_hot_rows"] == N_TEST and report.gates_e4["argmax_count"] == [3, 3, 3, 3]
    assert report.gates_e2["epoch"] == 1 and report.gates_e2["n"] == N_TEST
    assert report.clusters["fit_rows"] == 156 and report.clusters["reload_check"] is True
    md = (out / "comparison.md").read_text(encoding="utf-8")
    assert "| E2 vs E1 | 1, 0 | 12 | 5 | 3 | +2 | 3 | 1 | 4 | 0.6666666666666667 |" in md
    assert f"| iLoRA Dynamic (E2) | dynamic | 0.75 (9/12) | {5 / 9!r} (5/9) | {5 / 12!r} (5/12) | 1234 | 1 |" in md
    assert "No variant_total_rank_scaling result is included" in md
    assert "- iLoRA Dynamic (E2): 25984 (no calculated value for backbone tiny_random)" in md
    check_paper_numbers(out)
    index = check_spec20(out)
    assert Path(index["results/baseline/metrics.json"]["path"]) == scenario["E2"] / "eval/epoch1/test/metrics.json"
    assert Path(index["results/cluster_hard/metrics.json"]["path"]) == scenario["E4"] / "eval/epoch0/test/metrics.json"
    assert Path(index["results/baseline/gates_test.csv"]["path"]) == scenario["E2X"] / "records/export/epoch1/gates_test.csv"
    assert Path(index["artifacts/kmeans_k4_seed1234.joblib"]["path"]) == scenario["CL"] / "cluster/kmeans_k4_seed1234.joblib"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    for heading in ("## 1. Experiment setup", "## 2. Baseline reproduction", "## 3. Gate diversity",
                    "## 4. SASRec clustering", "## 5. Cluster-hard routing", "## 6. Performance comparison",
                    "## 7. Limitations"):
        assert heading in summary
    assert "one-hot rows 12/12" in summary


@pytest.mark.criterion("M14-7")
def test_missing_and_failed_runs_are_not_run(tool, scenario, tmp_path):
    """A run folder that does not exist (E1), a run whose meta.json status is failed (E4), an experiment not given
    (E3), a missing key (aux_protocols of E2's last epoch) and a missing cluster run give NOT_RUN cells with reasons,
    never estimates; the usable cells are unchanged."""
    failed = make_train_run(tmp_path / "more", "e4_failed", sets=("router.routing_mode=cluster_hard",
                                                                  f"cluster.artifact_dir={ARTIFACT}"),
                            best=0, last=0, correct={0: (1,)}, valid={0: (1,)}, one_hot=True, status="failed")
    last = scenario["E2"] / "eval" / "epoch2" / "test" / "metrics.json"
    doc = json.loads(last.read_text(encoding="utf-8"))
    del doc["aux_protocols"]
    last.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    out = tool.run_compare({"E1": tmp_path / "no_such_run", "E2": scenario["E2"], "E4": failed},
                           {"E2": scenario["E2X"]}, None, root=scenario["out"], argv=["compare_runs.py"])
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["status"] == "completed"
    rows = read_csv_rows(out / "comparison.csv")
    by = {}
    for row in rows:
        by.setdefault(row["method"], []).append(row)
    for method in ("LLaRA baseline (E1)", "iLoRA text-only eval (E3)", "Cluster Hard (E4)"):
        assert len(by[method]) == 6
        assert all(row[col] == "NOT_RUN" for row in by[method] for col in VALUE_COLUMNS), method
    assert by["iLoRA text-only eval (E3)"][0]["run"] == "NOT_RUN"
    e2 = {(r["checkpoint"], r["protocol"]): r for r in by["iLoRA Dynamic (E2)"]}
    assert check_rows_against_files([e2[("best_val", p)] for p in ("full", "drop_last_bs4", "drop_last_bs8")]
                                    + [e2[("last", "full")]], scenario) == 4
    for protocol in ("drop_last_bs4", "drop_last_bs8"):
        row = e2[("last", protocol)]
        assert all(row[key] == "NOT_RUN" for key in METRIC_FIELDS) and row["source_path"] == str(last)
    md = (out / "comparison.md").read_text(encoding="utf-8")
    reasons = md[md.index("## NOT_RUN reasons"):md.index("## Limitations")]
    assert f"run folder {tmp_path / 'no_such_run'} not found" in reasons
    assert "status is 'failed'" in reasons and "not given (--run E3=<folder>)" in reasons
    assert "has no aux_protocols.drop_last_bs4" in reasons and "not given (--cluster)" in reasons
    assert "| E2 vs E1 | NOT_RUN |" in md and "| E3 vs E2 | NOT_RUN |" in md
    check_paper_numbers(out)
    check_spec20(out, expect_missing=("artifacts/kmeans_k4_seed1234.joblib", "results/clustering/cluster_summary.json",
                                      "results/clustering/figures/sasrec_pca_clusters.png",
                                      "results/clustering/figures/cluster_dynamic_gate_heatmap.png",
                                      "results/cluster_hard/metrics.json"))
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## 4. SASRec clustering\n\n- NOT_RUN: cluster: not given (--cluster)" in summary


@pytest.mark.criterion("M14-7")
def test_limitation_e1_names_only_max_steps_for_d24(tool):
    """18.6.10 #6 after the D24 decision of 2026-11-06: the tail window follows Lightning (every micro-batch / 16), so
    the E1 limitation names only the missing max_steps truncation for D24, not a partial-accumulation difference."""
    (e1,) = [text for text in tool.LIMITATIONS if text.startswith("E1 is not a run of the LLaRA code")]
    assert "no max_steps truncation D24" in e1 and "partial accumulation" not in e1


@pytest.mark.criterion("M14-7")
def test_trainable_params_against_calculation(tool, tmp_path):
    """Table A note (§18.6.7): a llama2_7b run's TrainableParams from run_summary.json are set against the
    calculated 37,036,288 (dynamic) / 37,036,032 (K = 1) and the difference is written, not replaced."""
    llama = ("model.backbone=llama2_7b", "model.torch_dtype=float16", "model.device=cuda:0", "train.precision=fp16_amp")
    e2 = make_train_run(tmp_path / "runs", "e2", sets=llama, best=0, last=0, correct={0: (0,)}, valid={0: (0, 1)})
    e1 = make_train_run(tmp_path / "runs", "e1", sets=(*llama, "lora.num_experts=1", "router.routing_mode=none"),
                        best=0, last=0, correct={0: (1,)}, valid={0: (1,)})
    out = tool.run_compare({"E1": e1, "E2": e2}, {}, None, root=tmp_path / "compare", argv=["compare_runs.py"])
    md = (out / "comparison.md").read_text(encoding="utf-8")
    assert "- iLoRA Dynamic (E2): 25984 vs calculated 37036288: differs by -37010304" in md
    assert "- LLaRA baseline (E1): 25728 vs calculated 37036032: differs by -37010304" in md
    rows = read_csv_rows(out / "comparison.csv")
    assert {row["trainable_params"] for row in rows if row["method"] == "iLoRA Dynamic (E2)"} == {"25984"}


@pytest.mark.criterion("M14-7")
def test_elapsed_time_is_training_compute_time(tool, tmp_path):
    """§18.6.7: a training run resumed after an overnight stop (meta.json ``resumes``, ``elapsed_s`` from the first
    start to the last end) shows its compute time run_summary.json ``train_summary.seconds.total`` as ElapsedTime in
    comparison.csv, table A and summary.md, and meta.json ``elapsed_s`` only as the separate wall clock with its
    number of resumes; the E3 eval run keeps meta.json ``elapsed_s`` as ElapsedTime."""
    import datetime as dt

    root = tmp_path / "runs"
    e2 = make_train_run(root, "e2", best=0, last=0, correct={0: (0, 1)}, valid={0: (0, 1, 2)})
    started = dt.datetime.fromisoformat(run.read_meta(e2)["started_utc"]) - dt.timedelta(hours=25)
    run.update_meta(e2, {"started_utc": started.isoformat(), "resumes": [     # the M0 resume entry (commands.py)
        {"utc": run.utc_now().isoformat(), "argv": ["main.py", "train", "--resume", str(e2)],
         "previous_status": "running", "previous_ended_utc": None, "previous_error": None, "complete_epochs": [0],
         "start_epoch": 1}]})
    run.finish_meta(e2, "completed")           # elapsed_s: first start to last end, the 25 idle hours included
    wall = run.read_meta(e2)["elapsed_s"]
    total = TRAIN_SECONDS["total"]
    assert wall > 25 * 3600 > total
    e3 = make_eval_run(root, "e3", e2, correct={0: (0,)}, valid={0: (0, 1)})
    e3_wall = run.read_meta(e3)["elapsed_s"]
    out = tool.run_compare({"E2": e2, "E3": e3}, {}, None, root=tmp_path / "compare", argv=["compare_runs.py"])
    rows = read_csv_rows(out / "comparison.csv")
    times = {}
    for row in rows:
        times.setdefault(row["method"], set()).add((row["elapsed_s"], row["wall_clock_s"], row["resumes"]))
    assert times["iLoRA Dynamic (E2)"] == {(repr(total), repr(wall), "1")}
    assert times["iLoRA text-only eval (E3)"] == {(repr(e3_wall), repr(e3_wall), "0")}
    assert check_rows_against_files([r for r in rows if r["method"] in ("iLoRA Dynamic (E2)",
                                                                         "iLoRA text-only eval (E3)")],
                                    {"E2": e2, "E3": e3}) == 12
    md = (out / "comparison.md").read_text(encoding="utf-8")
    table_a = md[md.index("## Table A."):md.index("## Table B.")]
    assert "| TrainableParams | ElapsedTime | WallClock |" in table_a
    assert f"| 25984 | {total!r} s | {wall!r} s (resumes 1) |" in table_a
    assert f"| {e3_wall!r} s | {e3_wall!r} s (resumes 0) |" in table_a
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert f"ElapsedTime {total!r} s; wall clock {wall!r} s (resumes 1)" in summary


@pytest.mark.criterion("M14-7")
def test_precision_of_the_runs_is_stated(tool, tmp_path):
    """D9, §18.6.4: runs in the main precision float16 / fp16_amp add no note; a bf16 E2 (the variant_bf16 fallback)
    next to an fp16 E1 is named as a precision difference in the limitations of comparison.md and summary.md, and
    runs that are all bf16 are named as the recorded deviation. The values come from each run's config.yaml."""
    llama = ("model.backbone=llama2_7b", "model.device=cuda:0")
    fp16 = (*llama, "model.torch_dtype=float16", "train.precision=fp16_amp")
    bf16 = (*llama, "model.torch_dtype=bfloat16", "train.precision=bf16_amp")
    k1 = ("lora.num_experts=1", "router.routing_mode=none")
    root = tmp_path / "runs"

    def compare(case: str, e1: Sequence[str], e2: Sequence[str]) -> tuple[str, str]:
        runs = {"E1": make_train_run(root, f"e1_{case}", sets=(*e1, *k1), best=0, last=0, correct={0: (1,)},
                                     valid={0: (1,)}),
                "E2": make_train_run(root, f"e2_{case}", sets=e2, best=0, last=0, correct={0: (0,)},
                                     valid={0: (0, 1)})}
        out = tool.run_compare(runs, {}, None, root=tmp_path / f"compare_{case}", argv=["compare_runs.py"])
        md = (out / "comparison.md").read_text(encoding="utf-8")
        return md[md.index("## Limitations"):], (out / "summary.md").read_text(encoding="utf-8")

    limits, summary = compare("fp16", fp16, fp16)
    assert "Precision (D9)" not in limits and "Precision (D9)" not in summary
    limits, summary = compare("mixed", fp16, bf16)
    assert "9. Precision (D9): the compared runs differ in model.torch_dtype / train.precision (E1 float16 / " \
           "fp16_amp, E2 bfloat16 / bf16_amp)" in limits
    assert "Precision (D9): the compared runs differ" in summary
    limits, _ = compare("bf16", bf16, bf16)
    assert "9. Precision (D9): every compared run (E1, E2) uses bfloat16 / bf16_amp, not the main setting" in limits


@pytest.mark.criterion("M14-7")
def test_export_of_another_run_is_not_used(tool, scenario):
    """An export whose meta.json source_run is not the E2 run given with --run is not read for E2: the gate summary
    and the SPEC §20 export files are NOT_RUN with that reason (no values of one run under another's name)."""
    out = tool.run_compare({"E2": scenario["E1"]}, {"E2": scenario["E2X"]}, scenario["CL"], root=scenario["out"],
                           argv=["compare_runs.py"])
    md = (out / "comparison.md").read_text(encoding="utf-8")
    assert f"export E2 was made from {scenario['E2']}, not from the E2 run {scenario['E1']}" in md
    index = check_spec20(out, expect_missing=("results/baseline/gates_test.csv", "results/baseline/gate_summary.json",
                                              *(f"results/baseline/figures/{name}.png" for name in (
                                                  "gate_heatmap", "expert_mean_utilization",
                                                  "argmax_expert_distribution", "entropy_distribution")),
                                              "results/cluster_hard/metrics.json"))
    assert "not from the E2 run" in index["results/baseline/gate_summary.json"]["reason"]


@pytest.mark.criterion("M14-7")
@pytest.mark.parametrize("case", ["ratio", "count", "summary_sha"])
def test_inconsistent_files_stop_the_comparison(tool, scenario, case):
    """A metrics.json whose ratio differs from its counts by more than 1e-12, a records.jsonl that does not match its
    metrics.json, or an eval summary.json whose metrics.json sha256 differs is a CompareError; meta.json is failed."""
    if case == "ratio":
        path = scenario["E2"] / "eval" / "epoch1" / "test" / "metrics.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["hr_at_1"] += 1e-9
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    elif case == "count":
        path = scenario["E1"] / "eval" / "epoch0" / "test" / "records.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-1]), encoding="utf-8")
    else:
        path = scenario["E3"] / "eval" / "epoch1" / "test" / "metrics.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(tool.CompareError):
        run_tool(tool, scenario)
    (out,) = scenario["out"].glob("compare_*")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and "CompareError" in meta["error"]


@pytest.mark.criterion("M14-7")
def test_command_line(tool, scenario):
    """``python -I rebuild/tools/compare_runs.py --run ... --root ...`` exits 0 and prints the new folder; malformed
    or repeated names and a missing --run exit 2 before any folder is made."""
    def call(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", str(TOOL_PATH), *args], capture_output=True, text=True,
                              timeout=600, cwd=str(scenario["out"].parent))

    for bad in (("--run", "E2"), ("--run", f"E2={scenario['E2']}", "--run", f"E2={scenario['E1']}"),
                ("--cluster", str(scenario["CL"])), ("--run", f"2x={scenario['E2']}")):
        proc = call(*bad, "--root", str(scenario["out"]))
        assert proc.returncode == 2, (bad, proc.stderr)
    assert not scenario["out"].exists()
    proc = call("--run", f"E2={scenario['E2']}", "--run", f"E1={scenario['E1']}", "--export", f"E2={scenario['E2X']}",
                "--cluster", str(scenario["CL"]), "--root", str(scenario["out"]))
    assert proc.returncode == 0, proc.stderr
    (out,) = scenario["out"].glob("compare_*")
    assert proc.stdout.strip().splitlines()[-1] == str(out)
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert "run_dir=" in (out / "run.log").read_text(encoding="utf-8").splitlines()[0]
    check_spec20(out, expect_missing=("results/cluster_hard/metrics.json",))


@pytest.mark.slow
@pytest.mark.criterion("M14-7")
@pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated")   # matplotlib 3.7.1 figures
def test_compare_from_cpu_smoke(tool, cpu_smoke_run, tmp_path):
    """M13 ``cpu_smoke_run``: S1 (E2), S5 (E4), S6 (E3), S7 (E1), the S3 export and the S4 cluster run. Every value
    equals its metrics.json, a missing run gives NOT_RUN cells, the paper values stay in table D and the SPEC §20
    index carries the right sha256 (§18.8 #7)."""
    def step(k: int) -> Path:
        found = Path(cpu_smoke_run.step_dir(f"S{k}"))          # steps/S<k>_<name>_<UTC>/ (M13 SmokeReport)
        assert json.loads((found / "meta.json").read_text(encoding="utf-8"))["status"] == "completed", found
        return found

    runs = {"E1": step(7), "E2": step(1), "E3": step(6), "E4": step(5)}
    out = tool.run_compare(runs, {"E2": step(3)}, step(4), root=tmp_path / "compare", argv=["compare_runs.py"])
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["status"] == "completed"
    rows = read_csv_rows(out / "comparison.csv")
    assert check_rows_against_files(rows, runs) == 24
    check_paper_numbers(out)
    check_spec20(out)
    missing_out = tool.run_compare({**runs, "E1": tmp_path / "no_such_run"}, {"E2": step(3)}, step(4),
                                   root=tmp_path / "compare_missing", argv=["compare_runs.py"])
    rows = read_csv_rows(missing_out / "comparison.csv")
    e1 = [row for row in rows if row["method"] == "LLaRA baseline (E1)"]
    assert len(e1) == 6 and all(row[col] == "NOT_RUN" for row in e1 for col in VALUE_COLUMNS)
    assert check_rows_against_files([row for row in rows if row["method"] != "LLaRA baseline (E1)"], runs) == 18
    check_paper_numbers(missing_out)
