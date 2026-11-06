"""실험 실행 폴더 검사 (DESIGN.md §17.6.1, §17.6.9, §17.8 #9, §18.6.6). VESSL_REQUIRED: skipped locally.

After each RUNBOOK step, ``ILORA_RUN_<NAME>`` points at the finished run folder and
the step's tests are selected with ``-k <prefix>`` (``smoke_gpu``, ``overfit``,
``e2_run``, ``e2_reeval``, ``e2_export``, ``cluster``, ``e4``, ``e1``; §17.6.1).
The whole module carries ``criterion("M13-9")`` (§17.8 #9). Values the design does
not judge (skip lists, activation maxima) are recorded in ``measurements`` and
stay TBD; only the counts fixed by the design are asserted.

File formats read here: ``records/train/*.csv`` columns of M12 §16.6.2,
``eval/{ckpt_tag}/{split}/records.jsonl`` (``EvalRecord`` with ``gate_0..K-1``,
§3.1) and ``metrics.json`` (``meta.n_batches``, ``meta.gate_calls`` = increment of
the evaluation, §15.6.9), the cluster run's ``cluster/`` (``{train,validation,test}_assignments.csv``,
``manifest.json``, ``cluster_summary.json``, ``figures/``, §19.2, §19.6.5-§19.6.6), the E4 run's
``meta.json`` ``cluster_live_check`` and ``run.log`` (M15 ``verify_live_encoder``, §19.6.7), the export run's ``records/export/epoch{e}/`` (NPZ, gate CSV, ``crosscheck.json``, ``manifest.json``)
and ``records/analysis/epoch{e}/{split}/`` (M12 §16.2, §16.6.5), the training run's ``checkpoints/epoch{e}/``
markers and ``val_metrics.json`` and ``meta.json`` ``train`` block (M10 §14.6.6, §14.6.8) and each run's ``config.yaml``.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.vessl, pytest.mark.criterion("M13-9")]

# paper_spec counts (§14.6.2, D3, D24): 8,265 micro-batches per epoch = 16 x 516 + 9, 517 steps per epoch, 5 epochs
EPOCHS, STEPS_PER_EPOCH, MICRO_PER_EPOCH = 5, 517, 8_265
TOTAL_STEPS, TOTAL_MICRO = EPOCHS * STEPS_PER_EPOCH, EPOCHS * MICRO_PER_EPOCH        # 2,585 and 41,325
EVAL_BATCHES = {"val": 12, "test": 12}          # eval.batch_size 8 over 94 and 95 samples (§15.6.2)
EVAL_ROWS = {"val": 94, "test": 95}
ASSIGNMENT_FILES = ("train", "validation", "test")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def _eval_folders(run_dir: Path) -> list[Path]:
    """Completed evaluation folders ``eval/{ckpt_tag}/{split}/`` (``metrics.json`` marks completion, §15.6.8)."""
    return [p.parent for p in sorted(run_dir.glob("eval/*/*/metrics.json"))]


def _eval_records(folder: Path) -> list[dict]:
    lines = (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _gates(record: dict, num_experts: int) -> list[float]:
    return [float(record[f"gate_{k}"]) for k in range(num_experts)]


@pytest.mark.criterion("M6-9", "M12-15")
def test_e2_run_records(run_dir_env, measurements):
    """E2 training records: 41,325 micro rows, 2,585 step rows (epoch-end n_micro 9), 5 epoch rows, gate calls."""
    run_dir = run_dir_env("E2")
    train = run_dir / "records" / "train"
    micro = _csv_rows(train / "micro_batches.csv")
    steps = sorted(_csv_rows(train / "optimizer_steps.csv"), key=lambda row: int(row["step"]))
    epochs = _csv_rows(train / "epochs.csv")
    assert len(micro) == TOTAL_MICRO
    assert len(steps) == TOTAL_STEPS
    assert len(epochs) == EPOCHS
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in steps:
        by_epoch[int(row["epoch"])].append(row)
    assert sorted(by_epoch) == list(range(EPOCHS))
    for epoch, rows in by_epoch.items():
        assert [int(row["n_micro"]) for row in rows] == [16] * (STEPS_PER_EPOCH - 1) + [9], epoch   # D24
    assert sum(int(row["gate_calls"]) for row in steps) == TOTAL_MICRO
    # GradScaler skips are written per step (M6-9, §10.6.5): 0/1 in every row, the list is recorded
    assert all(row["skipped"] in ("0", "1") for row in steps)
    skipped = [int(row["step"]) for row in steps if row["skipped"] == "1"]
    summary = json.loads((train / "run_summary.json").read_text(encoding="utf-8"))   # M12 close (§16.6.2)
    assert summary["status"] == "completed" and summary["skipped_steps"] == skipped
    assert (summary["micro_batches"], summary["optimizer_steps"], summary["epochs"]) == (TOTAL_MICRO, TOTAL_STEPS, EPOCHS)
    assert summary["gate_calls"] == TOTAL_MICRO
    measurements["e2_run_records"] = {
        "skipped_steps": skipped,
        "n_skipped_steps": len(skipped),
        "epoch_act_max_abs": [row["act_max_abs"] for row in epochs],
        "epoch_act_max_module": [row["act_max_module"] for row in epochs],
        "run_activation_max": summary["activation_max"],
    }


@pytest.mark.criterion("M3-13")
def test_e2_run_sasrec_digest(run_dir_env, measurements):
    """E2: ``sasrec_state_digest`` of every ``epoch{e}/train_state.pt`` equals the start value (§7.8 #13).

    M10 recomputes ``state_digest(model.sasrec.state_dict())`` at each epoch end and compares it with the
    value at the start (§14.6.4); the start value is the digest the encoder was loaded and checked
    against, ``sasrec.state_digest`` of the run's ``config.yaml`` (M3 ``load_sasrec_state``).
    """
    import torch

    run_dir = run_dir_env("E2")
    start = _config(run_dir)["sasrec"]["state_digest"]
    digests = {}
    for epoch in range(EPOCHS):
        state = torch.load(run_dir / "checkpoints" / f"epoch{epoch}" / "train_state.pt", map_location="cpu",
                           weights_only=True)
        digests[epoch] = state["sasrec_state_digest"]
    measurements["e2_run_sasrec_digest"] = {"start": start, "epochs": digests}
    assert digests == {epoch: start for epoch in range(EPOCHS)}


@pytest.mark.criterion("M7-14")
def test_e2_run_gate_counts(run_dir_env, measurements):
    """One gate call per training micro-batch and per evaluation batch; recorded omega rows are probability vectors."""
    run_dir = run_dir_env("E2")
    config = _config(run_dir)
    assert config["router"]["routing_mode"] == "dynamic"
    num_experts = int(config["lora"]["num_experts"])
    steps = _csv_rows(run_dir / "records" / "train" / "optimizer_steps.csv")
    assert sum(int(row["gate_calls"]) for row in steps) == TOTAL_MICRO
    folders = _eval_folders(run_dir)
    assert folders, "no completed evaluation folder"
    counted = {}
    for folder in folders:
        split = folder.name
        meta = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["meta"]
        assert meta["n_batches"] == EVAL_BATCHES[split], folder
        assert meta["gate_calls"] == meta["n_batches"], folder
        records = _eval_records(folder)
        assert len(records) == EVAL_ROWS[split], folder
        for record in records:
            gates = _gates(record, num_experts)
            assert all(math.isfinite(g) for g in gates), record["sample_id"]
            assert abs(sum(gates) - 1.0) <= 1e-5, record["sample_id"]
        counted[f"{folder.parent.name}/{split}"] = len(records)
    measurements["e2_run_gate_counts"] = {"eval_folders": counted}


@pytest.mark.criterion("M7-15", "M15-13")
def test_e4_run_no_gate_calls_and_one_hot(run_dir_env, measurements):
    """E4 (cluster_hard): no gate call in training, 330,600 lookups, evaluation omega = one-hot of the mapping."""
    from ilora.cluster import parse_expert_mapping
    from ilora.run import resolve_path

    e4 = run_dir_env("E4")
    cluster_run = run_dir_env("CLUSTER")
    config = _config(e4)
    assert config["router"]["routing_mode"] == "cluster_hard"
    assert resolve_path(config["cluster"]["artifact_dir"]).resolve() == (cluster_run / "cluster").resolve()
    num_experts = int(config["lora"]["num_experts"])
    steps = _csv_rows(e4 / "records" / "train" / "optimizer_steps.csv")
    assert sum(int(row["gate_calls"]) for row in steps) == 0
    assert sum(int(row["cluster_lookups"]) for row in steps) == TOTAL_MICRO * 8      # 5 x 8,265 x 8 = 330,600
    mapping = parse_expert_mapping(config["cluster"]["expert_mapping"], n_clusters=int(config["cluster"]["n_clusters"]),
                                   num_experts=num_experts)
    assignment: dict[str, int] = {}
    for name in ASSIGNMENT_FILES:
        for row in _csv_rows(cluster_run / "cluster" / f"{name}_assignments.csv"):
            assignment[row["sample_id"]] = int(row["cluster_id"])
    folders = _eval_folders(e4)
    assert folders, "no completed evaluation folder"
    checked = 0
    for folder in folders:
        meta = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["meta"]
        assert meta["gate_calls"] == 0, folder
        for record in _eval_records(folder):
            expert = mapping[assignment[record["sample_id"]]]
            assert _gates(record, num_experts) == [1.0 if k == expert else 0.0 for k in range(num_experts)], \
                record["sample_id"]
            checked += 1
    measurements["e4_run_one_hot"] = {"records_checked": checked, "eval_folders": len(folders)}


@pytest.mark.criterion("M15-12")
def test_e4_run_live_check(run_dir_env, measurements):
    """E4 start: M15 ``verify_live_encoder`` on the A100 passed, max |dz| <= ``cluster.live_check_atol`` (1e-4) over
    ``cluster.live_check_n`` train samples (§19.8 #12). Read from ``meta.json`` ``cluster_live_check`` (the dict M10
    records) and from the ``cluster live check`` line the function writes to ``run.log``; max |dz| is recorded (TBD)."""
    import re

    e4 = run_dir_env("E4")
    config = _config(e4)
    assert config["router"]["routing_mode"] == "cluster_hard"
    atol, n = float(config["cluster"]["live_check_atol"]), int(config["cluster"]["live_check_n"])
    assert atol <= 1e-4
    found = {}
    block = json.loads((e4 / "meta.json").read_text(encoding="utf-8")).get("cluster_live_check")
    if block is not None:
        assert block["passed"] is True and block["n"] == n and block["max_abs_diff"] <= atol, block
        assert str(block["device"]).startswith("cuda"), block
        found["meta_max_abs_diff"] = block["max_abs_diff"]
    pattern = re.compile(r"cluster live check (passed|FAILED): n=(\d+) max_abs_diff=(\S+) atol=(\S+) device=(\S+)")
    lines = list(pattern.finditer((e4 / "run.log").read_text(encoding="utf-8")))
    for match in lines:
        assert match.group(1) == "passed" and int(match.group(2)) == n, match.group(0)
        assert float(match.group(3)) <= atol and match.group(5).startswith("cuda"), match.group(0)
    if lines:
        found["log_max_abs_diff"] = [float(match.group(3)) for match in lines]
    assert found, "no verify_live_encoder result in meta.json (cluster_live_check) or run.log"
    measurements["e4_run_live_check"] = {"n": n, "atol": atol, **found}


@pytest.mark.criterion("M15-14")
def test_cluster_dynamic_comparison(run_dir_env, measurements):
    """Cluster run with the E2 export (RUNBOOK R11): every artifact matches its manifest sha256, fit split train with
    66,126 rows, no empty train cluster, z within ``cluster.z_crosscheck_atol`` of the dynamic export, the dynamic
    comparison items for the three splits and ``cluster_dynamic_gate_heatmap`` present. Values are recorded (TBD)."""
    from ilora.run import content_sha256

    cluster_run = run_dir_env("CLUSTER")
    config = _config(cluster_run)
    folder = cluster_run / "cluster"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["files"].items():
        assert content_sha256(folder / name) == sha, name
    assert manifest["fit_split"] == "train" and manifest["fit_rows"] == 66_126 and manifest["data"]["subset"] == "full"
    assert manifest["rows"] == {"train": 66_126, "validation": EVAL_ROWS["val"], "test": EVAL_ROWS["test"]}
    kmeans = f"kmeans_k{config['cluster']['n_clusters']}_seed{config['cluster']['random_state']}.joblib"
    assert {kmeans, "centroids.npz", "cluster_summary.json", *(f"{n}_assignments.csv" for n in ASSIGNMENT_FILES)} <= \
        set(manifest["files"])
    summary = json.loads((folder / "cluster_summary.json").read_text(encoding="utf-8"))
    assert min(summary["train_cluster_size"]) > 0 and sum(summary["train_cluster_size"]) == 66_126
    assert config["cluster"]["dynamic_export_dir"] is not None
    dynamic = summary["dynamic_export"]
    assert isinstance(dynamic, dict) and dynamic["routing_mode"] == "dynamic"
    atol = float(config["cluster"]["z_crosscheck_atol"])
    assert set(dynamic["z_max_abs_diff"]) == set(ASSIGNMENT_FILES)
    assert all(value <= atol for value in dynamic["z_max_abs_diff"].values()), dynamic["z_max_abs_diff"]
    for item in ("cluster_dynamic_gate_mean", "cluster_dynamic_gate_std", "contingency_cluster_vs_dynamic_argmax",
                 "cluster_dynamic_argmax_distribution", "nmi_cluster_vs_dynamic_argmax"):
        assert isinstance(summary[item], dict) and set(summary[item]) == set(ASSIGNMENT_FILES), item
    for fmt in config["records"]["figure_formats"]:
        for figure in ("sasrec_pca_clusters", "cluster_dynamic_gate_heatmap"):
            assert (folder / "figures" / f"{figure}.{fmt}").is_file(), (figure, fmt)
    measurements["cluster_dynamic_comparison"] = {
        "dynamic_export": {key: dynamic.get(key) for key in ("path", "epoch", "z_max_abs_diff")},
        "nmi_cluster_vs_dynamic_argmax": summary["nmi_cluster_vs_dynamic_argmax"],
        "train_cluster_size": summary["train_cluster_size"], "silhouette_score_train": summary["silhouette_score_train"],
    }


@pytest.mark.criterion("M12-16")
def test_e2_export(run_dir_env, measurements):
    """E2 export (``main.py export``, §16.6.5): best_val and last epochs of the training run, validation and test
    cross-check passed with max |d omega| <= 1e-5, NPZ/CSV/manifest present with their sha256, the analysis of every
    ``records.analysis_splits`` with its figures. max |d omega|, max |d z| and argmax mismatches are recorded (TBD)."""
    import numpy as np

    from ilora.run import content_sha256

    export = run_dir_env("E2_EXPORT")
    config = _config(export)
    num_experts = int(config["lora"]["num_experts"])
    source = Path(json.loads((export / "meta.json").read_text(encoding="utf-8"))["source_run"])
    selection = json.loads((source / "eval" / "selection.json").read_text(encoding="utf-8"))
    epochs = sorted({int(selection["best_val_epoch"]), int(selection["last_epoch"])})
    assert sorted(p.name for p in (export / "records" / "export").iterdir()) == [f"epoch{e}" for e in epochs]
    sizes = {"train": 66_126, "validation": EVAL_ROWS["val"], "test": EVAL_ROWS["test"]}
    figures = ("gate_heatmap", "expert_mean_utilization", "argmax_expert_distribution", "entropy_distribution")
    recorded = {}
    for epoch in epochs:
        out = export / "records" / "export" / f"epoch{epoch}"
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        for name, sha in manifest["files"].items():
            assert content_sha256(out / name) == sha, name
        for name, n in sizes.items():
            with np.load(out / f"representations_{name}.npz", allow_pickle=False) as npz:
                assert npz["sample_id"].shape == (n,) and npz["omega"].shape == (n, num_experts), name
        for name in ("validation", "test"):
            rows = _csv_rows(out / f"gates_{name}.csv")
            assert len(rows) == sizes[name] and list(rows[0])[3:3 + num_experts] == \
                [f"gate_{k}" for k in range(num_experts)]
        cross = json.loads((out / "crosscheck.json").read_text(encoding="utf-8"))["splits"]
        for name in ("validation", "test"):
            assert cross[name]["passed"] and cross[name]["max_abs_diff_omega"] <= 1e-5, (epoch, name)
        for split in config["records"]["analysis_splits"]:
            folder = export / "records" / "analysis" / f"epoch{epoch}" / split
            for name in ("gate_summary.json", "gate_summary.csv", "gate_interpretation.txt"):
                assert (folder / name).is_file(), folder / name
            for name in figures + (("gate_ten_sequences",) if split == "test" else ()):
                for fmt in config["records"]["figure_formats"]:
                    assert (folder / "figures" / f"{name}.{fmt}").is_file(), (folder, name, fmt)
        recorded[f"epoch{epoch}"] = {name: {key: cross[name][key] for key in
                                            ("max_abs_diff_omega", "max_abs_diff_z", "argmax_mismatches")}
                                     for name in ("validation", "test")}
    measurements["e2_export"] = recorded


# ----------------------------------------------------------------------------
# M11 evaluation folders of the E2 run and its re-evaluations (§15.8 #18-#20, #22)
# ----------------------------------------------------------------------------
def _metrics(folder: Path) -> dict:
    return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))


@pytest.mark.criterion("M11-18")
def test_e2_run_eval_folders(run_dir_env, measurements):
    """E2: completed ``eval/epoch{0..4}/val/`` (94) and ``eval/epoch{e}/test/`` (95) for best_val and last of
    ``eval/selection.json``, with the three files; seconds and max GPU memory per split are recorded (TBD), and so
    are the metrics (TBD, no judgement)."""
    run_dir = run_dir_env("E2")
    selection = json.loads((run_dir / "eval" / "selection.json").read_text(encoding="utf-8"))
    tests = sorted({int(selection["best_val_epoch"]), int(selection["last_epoch"])})
    found = {f"{folder.parent.name}/{folder.name}" for folder in _eval_folders(run_dir)}
    assert found == {f"epoch{e}/val" for e in range(EPOCHS)} | {f"epoch{e}/test" for e in tests}
    recorded = {}
    for folder in _eval_folders(run_dir):
        split = folder.name
        metrics = _metrics(folder)
        assert {"records.jsonl", "metrics.json", "representations.npz"} <= {p.name for p in folder.iterdir()}
        assert metrics["n_total"] == EVAL_ROWS[split] and len(_eval_records(folder)) == EVAL_ROWS[split], folder
        meta = metrics["meta"]
        assert math.isfinite(meta["seconds"]) and meta["seconds"] > 0, folder
        assert meta["max_memory_allocated_gib"] is not None and math.isfinite(meta["max_memory_allocated_gib"])
        recorded[f"{folder.parent.name}/{split}"] = {
            "seconds": meta["seconds"], "max_memory_allocated_gib": meta["max_memory_allocated_gib"],
            **{key: metrics[key] for key in ("n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")}}
    measurements["e2_run_eval_folders"] = recorded


@pytest.mark.criterion("M10-17", "M11-19")
def test_e2_reeval_reproduces(run_dir_env, measurements):
    """RUNBOOK R8: ``main.py eval`` of the E2 checkpoints in new processes (E2_REEVAL, and E2_REEVAL_BI with
    ``eval.batch_invariance_check=true``): every ``records.jsonl`` has the sha256 of the training-time record and
    ``summary.json`` reports ``identical`` (D16). A difference prints the samples and the first differing step and
    fails; results are not final until its cause is known (§15.8 #19)."""
    from ilora.evaluate import compare_records
    from ilora.records import read_eval_records
    from ilora.run import content_sha256

    e2 = run_dir_env("E2")
    recorded, failures = {}, []
    for name in ("E2_REEVAL", "E2_REEVAL_BI"):
        reeval = run_dir_env(name)
        summary = json.loads((reeval / "eval" / "summary.json").read_text(encoding="utf-8"))
        assert Path(summary["source_run"]).resolve() == e2.resolve()
        assert summary["results"], name
        for key, item in summary["results"].items():
            original = e2 / "eval" / item["ckpt_tag"] / item["split"] / "records.jsonl"
            new = reeval / "eval" / item["ckpt_tag"] / item["split"] / "records.jsonl"
            same = content_sha256(original) == content_sha256(new)
            recorded[f"{name}/{key}"] = {"identical": same, "reproduction": item["reproduction"]["status"]}
            if not same or item["reproduction"]["status"] != "identical":
                diff = compare_records(read_eval_records(original), read_eval_records(new))
                print(f"{name} {key}: records differ; first differences "
                      f"{[(d['sample_id'], d['first_diff_step']) for d in diff['differences'][:20]]}, "
                      f"tokens equal {diff['tokens_equal']}/{diff['n']}, max |d omega| {diff['max_abs_diff_omega']}")
                failures.append(f"{name}/{key}")
    measurements["e2_reeval_reproduces"] = recorded
    assert not failures, f"re-evaluation differs from the training-time records: {failures}"


@pytest.mark.criterion("M11-20")
def test_e2_reeval_batch_invariance(run_dir_env, measurements):
    """E2_REEVAL_BI: every evaluated (checkpoint, split) has ``batch_invariance.json`` (batch 8 vs batch 1); the
    token, first-line and judgement agreement and max |d omega|, |d z| are recorded without a requirement (TBD)."""
    reeval = run_dir_env("E2_REEVAL_BI")
    folders = _eval_folders(reeval)
    assert folders, "no completed evaluation folder"
    recorded = {}
    for folder in folders:
        report = json.loads((folder / "batch_invariance.json").read_text(encoding="utf-8"))
        assert report["batch_sizes"] == [_metrics(folder)["meta"]["eval_batch_size"], 1]
        assert report["n"] == EVAL_ROWS[folder.name], folder
        recorded[f"{folder.parent.name}/{folder.name}"] = {
            key: report[key] for key in ("tokens_equal", "first_line_equal", "judgement_equal", "gates_equal",
                                         "ratios", "max_abs_diff_omega", "max_abs_diff_z")}
        recorded[f"{folder.parent.name}/{folder.name}"]["n_differences"] = len(report["differences"])
    measurements["e2_reeval_batch_invariance"] = recorded


@pytest.mark.criterion("M11-22")
def test_e2_run_generation_diagnostics(run_dir_env, measurements):
    """E2 evaluation folders: ``supplementary.generation`` holds the six diagnostic counts of §15.6.7 (integers in
    [0, N], valid = N - n_no_match - n_multi_match); the values are recorded (TBD)."""
    run_dir = run_dir_env("E2")
    names = ("n_no_eos", "n_empty_first_line", "n_blank_leading_segment", "n_cut_at_max_tokens", "n_no_match",
             "n_multi_match")
    recorded = {}
    for folder in _eval_folders(run_dir):
        metrics = _metrics(folder)
        generation = metrics["supplementary"]["generation"]
        n = metrics["n_total"]
        assert set(generation) == set(names), folder
        assert all(isinstance(generation[k], int) and 0 <= generation[k] <= n for k in names), generation
        assert metrics["n_valid"] == n - generation["n_no_match"] - generation["n_multi_match"], folder
        recorded[f"{folder.parent.name}/{folder.name}"] = generation
    measurements["e2_run_generation_diagnostics"] = recorded


@pytest.mark.criterion("M10-15")
def test_e2_run_complete(run_dir_env, measurements):
    """E2 on the A100 in fp16 (§14.8 #15): the first step starts at scale 65,536, every step row records ``skipped``
    and no ``train.max_consecutive_skips`` (16) consecutive skips occur, 2,585 steps and 41,325 micro-batches, the five
    epochs have ``COMPLETE`` with a validation of 94 (``val_metrics.json``, M10 14.6.6), and ``meta.json`` holds the
    start and end times and the per-epoch maximum memory (M10 ``train`` block). The ``train.segments`` follow M10
    ``segment_problems``: a process killed before closing its segment (SIGKILL, OOM killer) leaves it ``interrupted``
    with ``ended_utc`` null after the RUNBOOK R7 resume. With the bf16 fallback (``variant_bf16``, RUNBOOK, D9
    deviation) there is no GradScaler: every step has scale 1.0 and none is skipped. Skips, times and memory are
    recorded (TBD), not judged."""
    from ilora.train import segment_problems

    run_dir = run_dir_env("E2")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["started_utc"] and meta["ended_utc"]
    train = run_dir / "records" / "train"
    steps = sorted(_csv_rows(train / "optimizer_steps.csv"), key=lambda row: int(row["step"]))
    assert len(steps) == TOTAL_STEPS and len(_csv_rows(train / "micro_batches.csv")) == TOTAL_MICRO
    precision = _config(run_dir)["train"]["precision"]
    if precision == "fp16_amp":
        assert float(steps[0]["scale_before"]) == 65536.0
    else:
        assert all(float(row["scale_before"]) == float(row["scale_after"]) == 1.0 for row in steps), precision
        assert all(row["skipped"] == "0" for row in steps), precision
    assert all(row["skipped"] in ("0", "1") for row in steps)
    longest = streak = 0
    for row in steps:
        streak = streak + 1 if row["skipped"] == "1" else 0
        longest = max(longest, streak)
    assert longest < int(_config(run_dir)["train"]["max_consecutive_skips"])
    for epoch in range(EPOCHS):
        ckpt = run_dir / "checkpoints" / f"epoch{epoch}"
        assert (ckpt / "STATE_SAVED").is_file() and (ckpt / "COMPLETE").is_file(), ckpt
        val = json.loads((ckpt / "val_metrics.json").read_text(encoding="utf-8"))
        assert val["metrics"]["n_total"] == EVAL_ROWS["val"], ckpt
    block = meta["train"]
    assert segment_problems(block["segments"], meta.get("resumes") or []) == []
    assert sorted(block["epochs"], key=int) == [str(e) for e in range(EPOCHS)]
    for info in block["epochs"].values():
        assert info["max_mem_gib"] is not None and math.isfinite(info["max_mem_gib"])
    epochs = _csv_rows(train / "epochs.csv")
    assert len(epochs) == EPOCHS and all(row["max_mem_gib"] != "" for row in epochs)
    measurements["e2_run_complete"] = {
        "precision": precision, "torch_dtype": _config(run_dir)["model"]["torch_dtype"],
        "skipped_steps": [int(row["step"]) for row in steps if row["skipped"] == "1"],
        "longest_consecutive_skips": longest,
        "first_scale_before": float(steps[0]["scale_before"]), "last_scale_after": float(steps[-1]["scale_after"]),
        "started_utc": meta["started_utc"], "ended_utc": meta["ended_utc"], "elapsed_s": meta["elapsed_s"],
        "segments": block["segments"], "epochs": block["epochs"],
        "epoch_max_mem_gib": [row["max_mem_gib"] for row in epochs],
        "epoch_max_mem_reserved_gib": [row["max_mem_reserved_gib"] for row in epochs]}


# ----------------------------------------------------------------------------
# Smoke runs of RUNBOOK R5 and R6 (M13 §17.6.7, §17.6.8, §17.8 #8)
# ----------------------------------------------------------------------------
def _smoke_checks(run_dir: Path, plan: str) -> tuple[dict, dict[str, dict]]:
    """``smoke_report.json`` of a ``main.py smoke`` run: the plan of its config.yaml, passed, every step completed and
    every check of the plan present (M13 ``smoke.required_checks``) with none failed. Returns (report, id -> check)."""
    from ilora import smoke
    from ilora.config import load_config

    report = json.loads((run_dir / smoke.REPORT_FILE).read_text(encoding="utf-8"))
    cfg, _ = load_config(run_dir / "config.yaml")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["smoke"]["passed"] is True
    assert report["plan"] == plan == smoke.smoke_plan(cfg) and report["passed"] is True
    assert all(step["status"] == "completed" for step in report["steps"]), report["failed_steps"]
    checks = {c["id"]: c for c in report["checks"]}
    missing = sorted(set(smoke.required_checks(plan, cfg)) - set(checks))
    assert not missing, missing
    assert report["failed_checks"] == [] and all(c["passed"] is not False for c in checks.values())
    return report, checks


@pytest.mark.criterion("M13-8")
def test_smoke_gpu_report(run_dir_env, measurements):
    """R5 (§17.6.7): smoke_gpu_100 passed with every check of G0-G6, and the pass criteria hold: 456 trainable.pt keys
    (455 for E4 and E1), Adam state 453 (dynamic), records 1,806 / 114 / 2 with epoch-end windows of 7, gate calls
    1,806, hybrid 218 + 670 = 888, re-evaluation identical, export |d omega| <= 1e-5, E4 one-hot with 0 gate calls and
    7,224 lookups. Memory, times, skips and the batch-invariance agreement are recorded (TBD)."""
    report, checks = _smoke_checks(run_dir_env("SMOKE_GPU"), "gpu100")
    m = {cid: check["measured"] for cid, check in checks.items()}
    for epoch in ("epoch0", "epoch1"):
        assert m["G1.trainable_keys"][epoch]["total"] == 456 and m["G1.trainable_keys"][epoch]["lora"] == 448
        assert sum(m["G1.adam_state"][epoch].values()) == 453
    assert m["G5.trainable_keys"]["epoch0"]["total"] == m["G6.trainable_keys"]["epoch0"]["total"] == 455
    assert m["G1.record_rows"] == {"micro": 1_806, "step": 114, "epoch": 2, "tau_consecutive": True,
                                   "step_consecutive": True}
    assert [sizes[-1] for sizes in m["G1.window_sizes"].values()] == [7, 7]
    assert m["G1.gate_calls"]["steps"] == 1_806 and m["G1.hybrid"]["epochs"] == [218, 670]
    assert m["G1.hybrid"]["steps"] == 888
    assert set(m["G2.reproduction"].values()) == {"identical"}
    assert all(entry["max_abs_diff_omega"] <= 1e-5 for entry in m["G3.crosscheck"].values())
    assert m["G5.gate_calls"]["steps"] == 0 and m["G5.cluster_lookups"]["steps"] == 7_224
    assert m["G5.one_hot_gates"]["mismatches"] == []
    assert m["G0.gpu"]["gpu_count"] == 1
    measurements["smoke_gpu_report"] = {
        "steps": {step["id"]: {"status": step["status"], "seconds": step["seconds"]} for step in report["steps"]},
        "recorded": {cid: check["measured"] for cid, check in checks.items() if check["passed"] is None},
        "consecutive_skips": {cid: m[cid] for cid in m if cid.endswith(".consecutive_skips")},
        "sasrec_reference": m["G0.sasrec_reference"], "crosscheck": m["G3.crosscheck"]}


@pytest.mark.criterion("M13-8")
def test_overfit_report(run_dir_env, measurements):
    """R6 (§17.6.8): overfit_gpu passed: 25 epochs COMPLETE, 100 micro-batches all hybrid, no 16 consecutive skips,
    last-epoch mean loss <= first-epoch mean x 0.1 and 16 / 16 first lines equal to the target titles after loading
    epoch 24 into a new model. The loss ratio and the match count are recorded (TBD)."""
    report, checks = _smoke_checks(run_dir_env("OVERFIT"), "overfit")
    m = {cid: check["measured"] for cid, check in checks.items()}
    assert set(m["O1.checkpoint_files"]) == {f"epoch{e}" for e in range(25)}
    assert m["O1.record_rows"] == {"micro": 100, "step": 100, "epoch": 25, "tau_consecutive": True,
                                   "step_consecutive": True}
    assert m["O1.all_hybrid"] == {"micro": 100, "hybrid": 100}
    assert math.isfinite(m["O1.loss_ratio"]["ratio"]) and m["O1.loss_ratio"]["ratio"] <= 0.1
    assert m["O2.checkpoint"]["epoch"] == 24
    assert m["O2.first_line_match"]["n"] == m["O2.first_line_match"]["n_match"] == 16
    measurements["overfit_report"] = {
        "loss_ratio": m["O1.loss_ratio"], "first_line_match": {k: m["O2.first_line_match"][k] for k in ("n", "n_match")},
        "consecutive_skips": m["O1.consecutive_skips"], "seconds": report["seconds"],
        "steps": {step["id"]: step["seconds"] for step in report["steps"]}}


# ----------------------------------------------------------------------------
# E1 (RUNBOOK R13, M14 §18.8 #8) and the E4 export (R12)
# ----------------------------------------------------------------------------
# §14.6.3 groups without gate (K = 1, cluster_hard): projector 17,047,552 + LoRA 19,988,480 = 37,036,032 (M9 13.8 #13)
E1_TRAINABLE_PARAMS = {"projector": 17_047_552, "lora": 19_988_480, "total": 37_036_032}
@pytest.mark.criterion("M14-8")
def test_e1_run_outputs(run_dir_env, measurements):
    """R13: E1 (llara_baseline, K = 1, routing none): optimizer groups {projector, lora} in every epoch, every
    evaluation gate_0 = 1.0, as many records as E2 (41,325 / 2,585 / 5) with no gate call, trainable parameters
    37,036,032 (§14.6.3), and the export (E1_EXPORT) done without gate analysis (M12-11). Metrics are recorded (TBD)."""
    import torch

    e1, export = run_dir_env("E1"), run_dir_env("E1_EXPORT")
    config = _config(e1)
    assert int(config["lora"]["num_experts"]) == 1 and config["router"]["routing_mode"] == "none"
    assert json.loads((e1 / "meta.json").read_text(encoding="utf-8"))["status"] == "completed"
    for epoch in range(EPOCHS):
        state = torch.load(e1 / "checkpoints" / f"epoch{epoch}" / "train_state.pt", map_location="cpu",
                           weights_only=True)
        assert [pg["name"] for pg in state["optimizer"]["param_groups"]] == ["projector", "lora"], epoch
    train = e1 / "records" / "train"
    steps = _csv_rows(train / "optimizer_steps.csv")
    assert (len(_csv_rows(train / "micro_batches.csv")), len(steps), len(_csv_rows(train / "epochs.csv"))) == \
        (TOTAL_MICRO, TOTAL_STEPS, EPOCHS)
    assert sum(int(row["gate_calls"]) for row in steps) == 0
    assert all(row["omega_mean_0"] == "1.0" and row["omega_entropy_mean"] == "" for row in steps)
    folders = _eval_folders(e1)
    assert folders, "no completed evaluation folder"
    for folder in folders:
        for record in _eval_records(folder):
            assert _gates(record, 1) == [1.0] and "gate_1" not in record, record["sample_id"]
    summary = json.loads((train / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["train_summary"]["trainable_params"] == E1_TRAINABLE_PARAMS
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and Path(meta["source_run"]).resolve() == e1.resolve()
    selection = json.loads((e1 / "eval" / "selection.json").read_text(encoding="utf-8"))
    epochs = sorted({int(selection["best_val_epoch"]), int(selection["last_epoch"])})
    for epoch in epochs:
        out = export / "records" / "export" / f"epoch{epoch}"
        cross = json.loads((out / "crosscheck.json").read_text(encoding="utf-8"))["splits"]
        assert all(cross[name]["passed"] for name in ("validation", "test")), epoch
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["num_experts"] == 1 and manifest["analysis"] == "not run: gate analysis requires K >= 2"
    assert not (export / "records" / "analysis").exists()
    measurements["e1_run_outputs"] = {
        "trainable_params": summary["train_summary"]["trainable_params"],
        "metrics": {f"{f.parent.name}/{f.name}": {k: _metrics(f)[k] for k in ("n_valid", "n_correct", "valid_ratio",
                                                                               "hr_at_1", "combined_metric")}
                    for f in folders}}


def test_e4_export(run_dir_env, measurements):
    """R12 (``outputs e4`` after the E4 export): the cluster_hard export of best_val and last passes the cross-check,
    its omega is the one-hot of mapping[cluster] for every split (the M15 gate, no KMeans), the live z check of the
    export passed, and the gate analysis of K = 4 exists. max |d z| is recorded (TBD)."""
    import numpy as np

    from ilora.cluster import parse_expert_mapping

    e4, export, cluster_run = run_dir_env("E4"), run_dir_env("E4_EXPORT"), run_dir_env("CLUSTER")
    config = _config(export)
    assert config["router"]["routing_mode"] == "cluster_hard"
    num_experts, n_clusters = int(config["lora"]["num_experts"]), int(config["cluster"]["n_clusters"])
    mapping = parse_expert_mapping(config["cluster"]["expert_mapping"], n_clusters=n_clusters, num_experts=num_experts)
    assignment = {row["sample_id"]: int(row["cluster_id"]) for name in ASSIGNMENT_FILES
                  for row in _csv_rows(cluster_run / "cluster" / f"{name}_assignments.csv")}
    meta = json.loads((export / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and Path(meta["source_run"]).resolve() == e4.resolve()
    selection = json.loads((e4 / "eval" / "selection.json").read_text(encoding="utf-8"))
    recorded = {}
    for epoch in sorted({int(selection["best_val_epoch"]), int(selection["last_epoch"])}):
        out = export / "records" / "export" / f"epoch{epoch}"
        cross = json.loads((out / "crosscheck.json").read_text(encoding="utf-8"))["splits"]
        assert all(cross[name]["passed"] and cross[name]["max_abs_diff_omega"] <= 1e-5 for name in ("validation", "test"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["routing_mode"] == "cluster_hard" and manifest["cluster_live_check"]["passed"] is True
        for name in ASSIGNMENT_FILES:
            with np.load(out / f"representations_{name}.npz", allow_pickle=False) as npz:
                expected = np.array([mapping[assignment[str(s)]] for s in npz["sample_id"]], dtype=np.int64)
                omega = npz["omega"]
                assert np.array_equal(npz["argmax_expert"], expected), (epoch, name)
                assert bool(((omega == 0) | (omega == 1)).all()) and bool((omega.sum(axis=1) == 1).all())
        for split in config["records"]["analysis_splits"]:
            assert (export / "records" / "analysis" / f"epoch{epoch}" / split / "gate_summary.json").is_file()
        recorded[f"epoch{epoch}"] = {name: cross[name]["max_abs_diff_z"] for name in ("validation", "test")}
    measurements["e4_export"] = recorded
