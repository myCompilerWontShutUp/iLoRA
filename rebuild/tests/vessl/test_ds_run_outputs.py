"""실행 폴더 검사: Steam, LastFM (DESIGN.md §23.6, §23.8 #16-#20). VESSL_REQUIRED: skipped locally.

After a step of ``rebuild/RUNBOOK_datasets.md``, ``ILORA_RUN_<DATASET>_<NAME>`` (``rec``) points at the finished run
folder and the step's tests are selected with ``dsoutputs <dataset>_<step>`` (pytest ``-k``, RUNBOOK_datasets DS0): the
test ids carry ``<dataset>_<step>`` because ``-k`` selects by id. The checks read the same files as
``tests/vessl/test_run_outputs.py`` (§17.6.1; the file formats are listed in that module's docstring) with the counts of
the dataset (§23.2, §23.6), computed with M10 ``step_counts`` from the run's own ``config.yaml``. Test modules cannot
import each other (test_m13_coverage ``test_no_repo_imports``), so its small readers are repeated here. §23.8: no
``criterion`` marker in this module. Values the design does not judge (skip lists, times, memory, agreement rates) are
recorded in ``measurements`` and stay TBD; only the counts fixed by the design are asserted.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

pytestmark = [pytest.mark.vessl]

DATASETS = ("lastfm", "steam")
# Step (the part after the dataset name in ``dsoutputs <dataset>_<step>``, RUNBOOK_datasets DS0) -> the
# ILORA_RUN_<DATASET>_<NAME> folders its tests read (tests/test_ds_runbook.py parses this literal, keep it a literal).
STEP_RUNS = {"smoke_gpu": ("SMOKE_GPU",), "overfit": ("OVERFIT",), "e2_run": ("E2",),
             "e2_reeval": ("E2", "E2_REEVAL", "E2_REEVAL_BI"), "e1": ("E1",), "e2_export": ("E2_EXPORT",),
             "cluster": ("CLUSTER",), "e4": ("E4", "E4_EXPORT", "CLUSTER")}
# §23.2: train samples, validation and test N, the E2 lengths of step_counts (micro-batches per epoch, tail, steps per
# epoch, T, micro-batches in total), the converted-state digest, the candidate hashes (§6.6.5 form, seed 1234);
# §23.6: train samples of <d>_smoke_gpu_100 and T of its 2-epoch run.
FACTS = {
    "lastfm": {"train": 47_872, "val": 122, "test": 122, "micro_per_epoch": 5_984, "tail": 0, "steps_per_epoch": 374,
               "total_steps": 1_870, "total_micro": 29_920,
               "state_digest": "6b0d161351901a272fe07ef4c2bf3ccc99e06a487bcaee1c9b88392f59435d85",
               "candidates": {"val": "3caad6a1fb669dc2e82b234afa474ddd2ed880002fb20b64e70b94fa298d1294",
                              "test": "0eb6f0af94ea28d6526b8a34f63ff88d94ce8cc5a243a07eb409dac57bc7c2d1"},
               "smoke_train": 3_626, "smoke_steps": 58},
    "steam": {"train": 151_048, "val": 1_184, "test": 1_186, "micro_per_epoch": 37_762, "tail": 2,
              "steps_per_epoch": 1_181, "total_steps": 5_905, "total_micro": 188_810,
              "state_digest": "d5d13fc0bc0f3c8ebc2f64e4247e9da3b486e6984c931bb8454ebbb17c8b3932",
              "candidates": {"val": "ba0df0feb1ce97dfac44a5786bf1f0b2674cb2501fcb54f302419c468a4120bb",
                             "test": "3dd8399ec36b739bfb25102e32a20569ad5bfdd1eddb87a00ca3c6612985813f"},
              "smoke_train": 1_385, "smoke_steps": 22},
}
EPOCHS = 5
ASSIGNMENT_FILES = ("train", "validation", "test")
# §14.6.3 groups without gate, the same model on every dataset (§23.5 M9): projector + LoRA (M9 13.8 #13)
E1_TRAINABLE_PARAMS = {"projector": 17_047_552, "lora": 19_988_480, "total": 37_036_032}


def step(name: str) -> Any:
    """Parametrize a test over the datasets with the ids ``<dataset>_<name>``: the selector of
    ``dsoutputs <dataset>_<name>`` (the ids are what ``-k`` matches, RUNBOOK_datasets DS0)."""
    return pytest.mark.parametrize("dataset", DATASETS, ids=[f"{d}_{name}" for d in DATASETS])


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def _run_dir(run_dir_env: Callable[[str], Path], dataset: str, name: str) -> Path:
    """``ILORA_RUN_<DATASET>_<NAME>`` (the ``rec`` names of RUNBOOK_datasets; a missing folder skips, SKIP_NO_RUN)."""
    return run_dir_env(f"{dataset.upper()}_{name}")


def _step_counts(run_dir: Path, n_samples: int) -> Any:
    """M10 ``step_counts`` of ``train.*`` of the run's ``config.yaml`` for ``n_samples`` train samples."""
    from ilora.config import load_config
    from ilora.train import step_counts

    return step_counts(n_samples, load_config(run_dir / "config.yaml")[0].train)


def _eval_folders(run_dir: Path) -> list[Path]:
    """Completed evaluation folders ``eval/{ckpt_tag}/{split}/`` (``metrics.json`` marks completion, §15.6.8)."""
    return [p.parent for p in sorted(run_dir.glob("eval/*/*/metrics.json"))]


def _eval_records(folder: Path) -> list[dict]:
    lines = (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _metrics(folder: Path) -> dict:
    return _read_json(folder / "metrics.json")


def _gates(record: dict, num_experts: int) -> list[float]:
    return [float(record[f"gate_{k}"]) for k in range(num_experts)]


def _split_n(dataset: str, split: str) -> int:
    """Evaluation N of §23.2 for ``val`` or ``test`` (no drop_last, D41)."""
    return FACTS[dataset][split]


def _tested_epochs(run_dir: Path) -> list[int]:
    selection = _read_json(run_dir / "eval" / "selection.json")
    return sorted({int(selection["best_val_epoch"]), int(selection["last_epoch"])})


# ----------------------------------------------------------------------------
# DS4, DS5: the smoke runs (§23.6, §17.6.7, §17.6.8)
# ----------------------------------------------------------------------------
def _smoke_checks(run_dir: Path, plan: str) -> tuple[dict, dict[str, dict]]:
    """``smoke_report.json`` of a ``main.py smoke`` run: the plan of its config.yaml, passed, every step completed and
    every check of the plan present (M13 ``smoke.required_checks``) with none failed. Returns (report, id -> check)."""
    from ilora import smoke
    from ilora.config import load_config

    report = _read_json(run_dir / smoke.REPORT_FILE)
    cfg, _ = load_config(run_dir / "config.yaml")
    meta = _read_json(run_dir / "meta.json")
    assert meta["status"] == "completed" and meta["smoke"]["passed"] is True
    assert report["plan"] == plan == smoke.smoke_plan(cfg) and report["passed"] is True
    assert all(item["status"] == "completed" for item in report["steps"]), report["failed_steps"]
    checks = {c["id"]: c for c in report["checks"]}
    missing = sorted(set(smoke.required_checks(plan, cfg)) - set(checks))
    assert not missing, missing
    assert report["failed_checks"] == [] and all(c["passed"] is not False for c in checks.values())
    return report, checks


@step("smoke_gpu")
def test_smoke_gpu_report(dataset, run_dir_env, measurements):
    """DS4 (§17.6.7): <d>_smoke_gpu_100 passed with every check of G0-G6, and the sizes hold: train samples and T of
    §23.6, records and window sizes of ``step_counts`` for the 2-epoch run, 456 / 455 trainable.pt keys, Adam state
    453, re-evaluation identical, export |d omega| <= 1e-5, E4 path one-hot with 0 gate calls and lookups = one epoch of
    micro-batches x micro-batch size. Memory, times and skips are recorded (TBD)."""
    run_dir = _run_dir(run_dir_env, dataset, "SMOKE_GPU")
    report, checks = _smoke_checks(run_dir, "gpu100")
    m = {cid: check["measured"] for cid, check in checks.items()}
    facts = FACTS[dataset]
    sc = _step_counts(run_dir, facts["smoke_train"])
    assert (sc.epochs, sc.total_steps) == (2, facts["smoke_steps"])
    for epoch in ("epoch0", "epoch1"):
        assert m["G1.trainable_keys"][epoch]["total"] == 456 and m["G1.trainable_keys"][epoch]["lora"] == 448
        assert sum(m["G1.adam_state"][epoch].values()) == 453
    assert m["G5.trainable_keys"]["epoch0"]["total"] == m["G6.trainable_keys"]["epoch0"]["total"] == 455
    assert m["G1.record_rows"] == {"micro": sc.total_micro, "step": sc.total_steps, "epoch": 2, "tau_consecutive": True,
                                   "step_consecutive": True}
    assert m["G1.window_sizes"] == {name: list(sc.window_sizes()) for name in ("epoch0", "epoch1")}
    assert m["G1.gate_calls"]["steps"] == sc.total_micro and m["G1.hybrid"]["steps"] == sum(m["G1.hybrid"]["epochs"])
    assert set(m["G2.reproduction"].values()) == {"identical"}
    assert all(entry["max_abs_diff_omega"] <= 1e-5 for entry in m["G3.crosscheck"].values())
    assert m["G5.gate_calls"]["steps"] == 0
    assert m["G5.cluster_lookups"]["steps"] == sc.micro_per_epoch * sc.micro_batch
    assert m["G5.one_hot_gates"]["mismatches"] == []
    assert m["G0.gpu"]["gpu_count"] == 1
    measurements[f"{dataset}_smoke_gpu_report"] = {
        "steps": {item["id"]: {"status": item["status"], "seconds": item["seconds"]} for item in report["steps"]},
        "recorded": {cid: check["measured"] for cid, check in checks.items() if check["passed"] is None},
        "sasrec_reference": m["G0.sasrec_reference"], "crosscheck": m["G3.crosscheck"]}


@step("overfit")
def test_overfit_report(dataset, run_dir_env, measurements):
    """DS5 (§17.6.8, the same rule as RUNBOOK R6): <d>_overfit_gpu passed: 25 epochs COMPLETE, 100 micro-batches all
    hybrid, last-epoch mean loss <= first-epoch mean x 0.1 and 16 / 16 first lines equal to the target titles after
    loading epoch 24 into a new model. The loss ratio and the match count are recorded (TBD)."""
    report, checks = _smoke_checks(_run_dir(run_dir_env, dataset, "OVERFIT"), "overfit")
    m = {cid: check["measured"] for cid, check in checks.items()}
    assert set(m["O1.checkpoint_files"]) == {f"epoch{e}" for e in range(25)}
    assert m["O1.record_rows"] == {"micro": 100, "step": 100, "epoch": 25, "tau_consecutive": True,
                                   "step_consecutive": True}
    assert m["O1.all_hybrid"] == {"micro": 100, "hybrid": 100}
    assert math.isfinite(m["O1.loss_ratio"]["ratio"]) and m["O1.loss_ratio"]["ratio"] <= 0.1
    assert m["O2.checkpoint"]["epoch"] == 24
    assert m["O2.first_line_match"]["n"] == m["O2.first_line_match"]["n_match"] == 16
    measurements[f"{dataset}_overfit_report"] = {
        "loss_ratio": m["O1.loss_ratio"],
        "first_line_match": {k: m["O2.first_line_match"][k] for k in ("n", "n_match")},
        "consecutive_skips": m["O1.consecutive_skips"], "seconds": report["seconds"],
        "steps": {item["id"]: item["seconds"] for item in report["steps"]}}


# ----------------------------------------------------------------------------
# DS6: the E2 run (§23.2 lengths and N, §23.8)
# ----------------------------------------------------------------------------
@step("e2_run")
def test_e2_records_equal_step_counts(dataset, run_dir_env, measurements):
    """DS6: the §23.2 lengths equal M10 ``step_counts`` of the run's config.yaml, and the training records have exactly
    those counts: micro_batches.csv, optimizer_steps.csv (window sizes per epoch, D24), epochs.csv, gate calls, run
    summary; every step records ``skipped``. Skip lists and activation maxima are recorded (TBD)."""
    run_dir = _run_dir(run_dir_env, dataset, "E2")
    facts = FACTS[dataset]
    sc = _step_counts(run_dir, facts["train"])
    assert (sc.micro_per_epoch, sc.tail, sc.steps_per_epoch, sc.total_steps, sc.total_micro, sc.epochs) == \
        (facts["micro_per_epoch"], facts["tail"], facts["steps_per_epoch"], facts["total_steps"], facts["total_micro"],
         EPOCHS)
    train = run_dir / "records" / "train"
    micro = _csv_rows(train / "micro_batches.csv")
    steps = sorted(_csv_rows(train / "optimizer_steps.csv"), key=lambda row: int(row["step"]))
    epochs = _csv_rows(train / "epochs.csv")
    assert (len(micro), len(steps), len(epochs)) == (sc.total_micro, sc.total_steps, EPOCHS)
    by_epoch: dict[int, list[int]] = defaultdict(list)
    for row in steps:
        by_epoch[int(row["epoch"])].append(int(row["n_micro"]))
    assert sorted(by_epoch) == list(range(EPOCHS))
    assert all(sizes == list(sc.window_sizes()) for sizes in by_epoch.values()), {e: s[-1] for e, s in by_epoch.items()}
    assert sum(int(row["gate_calls"]) for row in steps) == sc.total_micro
    assert all(row["skipped"] in ("0", "1") for row in steps)
    skipped = [int(row["step"]) for row in steps if row["skipped"] == "1"]
    summary = _read_json(train / "run_summary.json")
    assert summary["status"] == "completed" and summary["skipped_steps"] == skipped
    assert (summary["micro_batches"], summary["optimizer_steps"], summary["epochs"]) == (sc.total_micro, sc.total_steps,
                                                                                          EPOCHS)
    assert summary["gate_calls"] == sc.total_micro
    measurements[f"{dataset}_e2_records"] = {
        "skipped_steps": skipped, "n_skipped_steps": len(skipped),
        "epoch_act_max_abs": [row["act_max_abs"] for row in epochs],
        "epoch_act_max_module": [row["act_max_module"] for row in epochs],
        "run_activation_max": summary["activation_max"]}


@step("e2_run")
def test_e2_epochs_complete(dataset, run_dir_env, measurements):
    """DS6: status completed, every epoch has ``COMPLETE`` and ``STATE_SAVED`` and a validation of the whole split
    (§23.2 N), the first step starts at scale 65,536 in fp16 (with the bf16 alternative of RUNBOOK_datasets there is no
    GradScaler: scale 1.0, no skip), no ``train.max_consecutive_skips`` consecutive skips, the SASRec digest of every
    ``train_state.pt`` equals the §23.2 value (frozen encoder, §7.8 #13). Times, memory, skips are recorded (TBD)."""
    import torch

    from ilora.train import segment_problems

    run_dir = _run_dir(run_dir_env, dataset, "E2")
    facts = FACTS[dataset]
    meta = _read_json(run_dir / "meta.json")
    assert meta["status"] == "completed" and meta["command"] == "train" and meta["started_utc"] and meta["ended_utc"]
    config = _config(run_dir)
    assert config["sasrec"]["state_digest"] == facts["state_digest"]
    train = run_dir / "records" / "train"
    steps = sorted(_csv_rows(train / "optimizer_steps.csv"), key=lambda row: int(row["step"]))
    precision = config["train"]["precision"]
    if precision == "fp16_amp":
        assert float(steps[0]["scale_before"]) == 65536.0
    else:
        assert all(float(row["scale_before"]) == float(row["scale_after"]) == 1.0 for row in steps), precision
        assert all(row["skipped"] == "0" for row in steps), precision
    longest = streak = 0
    for row in steps:
        streak = streak + 1 if row["skipped"] == "1" else 0
        longest = max(longest, streak)
    assert longest < int(config["train"]["max_consecutive_skips"])
    digests = {}
    for epoch in range(EPOCHS):
        ckpt = run_dir / "checkpoints" / f"epoch{epoch}"
        assert (ckpt / "STATE_SAVED").is_file() and (ckpt / "COMPLETE").is_file(), ckpt
        assert _read_json(ckpt / "val_metrics.json")["metrics"]["n_total"] == facts["val"], ckpt
        state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
        digests[epoch] = state["sasrec_state_digest"]
    assert digests == {epoch: facts["state_digest"] for epoch in range(EPOCHS)}
    block = meta["train"]
    assert segment_problems(block["segments"], meta.get("resumes") or []) == []
    assert sorted(block["epochs"], key=int) == [str(e) for e in range(EPOCHS)]
    epochs = _csv_rows(train / "epochs.csv")
    measurements[f"{dataset}_e2_complete"] = {
        "precision": precision, "torch_dtype": config["model"]["torch_dtype"],
        "skipped_steps": [int(row["step"]) for row in steps if row["skipped"] == "1"],
        "longest_consecutive_skips": longest,
        "started_utc": meta["started_utc"], "ended_utc": meta["ended_utc"], "elapsed_s": meta["elapsed_s"],
        "segments": block["segments"], "epoch_max_mem_gib": [row["max_mem_gib"] for row in epochs],
        "epoch_max_mem_reserved_gib": [row["max_mem_reserved_gib"] for row in epochs]}


@step("e2_run")
def test_e2_eval_folders(dataset, run_dir_env, measurements):
    """DS6 (§23.2, D17, D41): ``eval/selection.json`` exists; completed ``eval/epoch{0..4}/val/`` and
    ``eval/epoch{e}/test/`` for best_val and last; every folder holds the whole split (N of §23.2) and
    ``n_batches = ceil(N / eval.batch_size)``, the full-data candidates hash of §23.2, the dataset name and one gate
    call per batch; recorded gates are probability vectors. Seconds, memory and metrics are recorded (TBD, no
    judgement)."""
    run_dir = _run_dir(run_dir_env, dataset, "E2")
    facts = FACTS[dataset]
    config = _config(run_dir)
    assert config["router"]["routing_mode"] == "dynamic"
    num_experts, batch_size = int(config["lora"]["num_experts"]), int(config["eval"]["batch_size"])
    tests = _tested_epochs(run_dir)                                # eval/selection.json must exist
    folders = _eval_folders(run_dir)
    assert {f"{folder.parent.name}/{folder.name}" for folder in folders} == (
        {f"epoch{e}/val" for e in range(EPOCHS)} | {f"epoch{e}/test" for e in tests})
    recorded = {}
    for folder in folders:
        split = folder.name
        metrics, records = _metrics(folder), _eval_records(folder)
        meta = metrics["meta"]
        assert metrics["n_total"] == _split_n(dataset, split) and len(records) == _split_n(dataset, split), folder
        assert meta["n_batches"] == math.ceil(_split_n(dataset, split) / batch_size), folder
        assert meta["gate_calls"] == meta["n_batches"], folder
        assert (meta["dataset"], meta["data_subset"], meta["eval_batch_size"]) == (dataset, "full", batch_size), folder
        assert meta["candidates_sha256"] == facts["candidates"][split], folder
        for record in records:
            gates = _gates(record, num_experts)
            assert all(math.isfinite(g) for g in gates) and abs(sum(gates) - 1.0) <= 1e-5, record["sample_id"]
        assert {"records.jsonl", "metrics.json", "representations.npz"} <= {p.name for p in folder.iterdir()}
        recorded[f"{folder.parent.name}/{split}"] = {
            "seconds": meta["seconds"], "max_memory_allocated_gib": meta["max_memory_allocated_gib"],
            **{key: metrics[key] for key in ("n_valid", "n_correct", "valid_ratio", "hr_at_1", "combined_metric")}}
    measurements[f"{dataset}_e2_eval_folders"] = recorded


# ----------------------------------------------------------------------------
# DS7: the two re-evaluations of the E2 checkpoints (the form of RUNBOOK R8)
# ----------------------------------------------------------------------------
@step("e2_reeval")
def test_e2_reeval_reproduces(dataset, run_dir_env, measurements):
    """DS7 (§23.8 #19): ``main.py eval`` of the E2 checkpoints in new processes (E2_REEVAL, and E2_REEVAL_BI with
    ``eval.batch_invariance_check=true``): every ``records.jsonl`` has the sha256 of the training-time record and
    ``summary.json`` reports ``identical`` (D16). A difference prints the samples and the first differing step and
    fails; results are not final until its cause is known (RUNBOOK_datasets DS7)."""
    from ilora.evaluate import compare_records
    from ilora.records import read_eval_records
    from ilora.run import content_sha256

    e2 = _run_dir(run_dir_env, dataset, "E2")
    recorded, failures = {}, []
    for name in ("E2_REEVAL", "E2_REEVAL_BI"):
        reeval = _run_dir(run_dir_env, dataset, name)
        summary = _read_json(reeval / "eval" / "summary.json")
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
    measurements[f"{dataset}_e2_reeval_reproduces"] = recorded
    assert not failures, f"re-evaluation differs from the training-time records: {failures}"


@step("e2_reeval")
def test_e2_reeval_batch_invariance(dataset, run_dir_env, measurements):
    """DS7 second run: every evaluated (checkpoint, split) of E2_REEVAL_BI has ``batch_invariance.json`` (the evaluation
    batch size against batch 1, D32) for the whole split; the agreement counts and max |d omega|, |d z| are recorded
    without a requirement (TBD, §23.8 #20)."""
    reeval = _run_dir(run_dir_env, dataset, "E2_REEVAL_BI")
    folders = _eval_folders(reeval)
    assert folders, "no completed evaluation folder"
    recorded = {}
    for folder in folders:
        report = _read_json(folder / "batch_invariance.json")
        assert report["batch_sizes"] == [_metrics(folder)["meta"]["eval_batch_size"], 1]
        assert report["n"] == _split_n(dataset, folder.name), folder
        recorded[f"{folder.parent.name}/{folder.name}"] = {
            key: report[key] for key in ("tokens_equal", "first_line_equal", "judgement_equal", "gates_equal",
                                         "ratios", "max_abs_diff_omega", "max_abs_diff_z")}
        recorded[f"{folder.parent.name}/{folder.name}"]["n_differences"] = len(report["differences"])
    measurements[f"{dataset}_e2_reeval_batch_invariance"] = recorded


# ----------------------------------------------------------------------------
# DS8: the E1 run (K = 1, the same lengths and N as E2)
# ----------------------------------------------------------------------------
@step("e1")
def test_e1_outputs(dataset, run_dir_env, measurements):
    """DS8: E1 (<d>_llara_baseline, K = 1, routing none): epochs complete, optimizer groups {projector, lora} in every
    epoch, every evaluation gate_0 = 1.0 with no gate call, the records and N of the dataset (``step_counts``, §23.2)
    with no gate call in training, trainable parameters 37,036,032 (§14.6.3, dataset-independent, §23.5 M9). Metrics
    are recorded (TBD)."""
    import torch

    e1 = _run_dir(run_dir_env, dataset, "E1")
    facts = FACTS[dataset]
    config = _config(e1)
    assert int(config["lora"]["num_experts"]) == 1 and config["router"]["routing_mode"] == "none"
    assert _read_json(e1 / "meta.json")["status"] == "completed"
    sc = _step_counts(e1, facts["train"])
    for epoch in range(EPOCHS):
        ckpt = e1 / "checkpoints" / f"epoch{epoch}"
        assert (ckpt / "COMPLETE").is_file(), ckpt
        state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
        assert [pg["name"] for pg in state["optimizer"]["param_groups"]] == ["projector", "lora"], epoch
    train = e1 / "records" / "train"
    steps = _csv_rows(train / "optimizer_steps.csv")
    assert (len(_csv_rows(train / "micro_batches.csv")), len(steps), len(_csv_rows(train / "epochs.csv"))) == \
        (sc.total_micro, sc.total_steps, EPOCHS)
    assert sum(int(row["gate_calls"]) for row in steps) == 0
    assert all(row["omega_mean_0"] == "1.0" and row["omega_entropy_mean"] == "" for row in steps)
    folders = _eval_folders(e1)
    assert folders, "no completed evaluation folder"
    tests = _tested_epochs(e1)
    assert {f"{f.parent.name}/{f.name}" for f in folders} == ({f"epoch{e}/val" for e in range(EPOCHS)} |
                                                               {f"epoch{e}/test" for e in tests})
    for folder in folders:
        records = _eval_records(folder)
        assert len(records) == _split_n(dataset, folder.name), folder
        for record in records:
            assert _gates(record, 1) == [1.0] and "gate_1" not in record, record["sample_id"]
        assert _metrics(folder)["meta"]["dataset"] == dataset, folder
    summary = _read_json(train / "run_summary.json")
    assert summary["train_summary"]["trainable_params"] == E1_TRAINABLE_PARAMS
    measurements[f"{dataset}_e1_outputs"] = {
        "trainable_params": summary["train_summary"]["trainable_params"],
        "metrics": {f"{f.parent.name}/{f.name}": {k: _metrics(f)[k] for k in ("n_valid", "n_correct", "valid_ratio",
                                                                               "hr_at_1", "combined_metric")}
                    for f in folders}}


# ----------------------------------------------------------------------------
# DS11-DS13: the optional E2 export, cluster and E4 runs
# ----------------------------------------------------------------------------
@step("e2_export")
def test_e2_export(dataset, run_dir_env, measurements):
    """DS11 (R10): best_val and last epochs of the training run, validation and test cross-check passed with max
    |d omega| <= 1e-5, NPZ/CSV/manifest present with their sha256 and the row counts of the dataset, the analysis of
    every ``records.analysis_splits`` with its figures. max |d omega|, max |d z| and argmax mismatches are recorded."""
    import numpy as np

    from ilora.run import content_sha256

    export = _run_dir(run_dir_env, dataset, "E2_EXPORT")
    facts = FACTS[dataset]
    config = _config(export)
    num_experts = int(config["lora"]["num_experts"])
    source = Path(_read_json(export / "meta.json")["source_run"])
    epochs = _tested_epochs(source)
    assert sorted(p.name for p in (export / "records" / "export").iterdir()) == [f"epoch{e}" for e in epochs]
    sizes = {"train": facts["train"], "validation": facts["val"], "test": facts["test"]}
    figures = ("gate_heatmap", "expert_mean_utilization", "argmax_expert_distribution", "entropy_distribution")
    recorded = {}
    for epoch in epochs:
        out = export / "records" / "export" / f"epoch{epoch}"
        manifest = _read_json(out / "manifest.json")
        for name, sha in manifest["files"].items():
            assert content_sha256(out / name) == sha, name
        for name, n in sizes.items():
            with np.load(out / f"representations_{name}.npz", allow_pickle=False) as npz:
                assert npz["sample_id"].shape == (n,) and npz["omega"].shape == (n, num_experts), name
        for name in ("validation", "test"):
            rows = _csv_rows(out / f"gates_{name}.csv")
            assert len(rows) == sizes[name] and list(rows[0])[3:3 + num_experts] == \
                [f"gate_{k}" for k in range(num_experts)]
        cross = _read_json(out / "crosscheck.json")["splits"]
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
    measurements[f"{dataset}_e2_export"] = recorded


@step("cluster")
def test_cluster_outputs(dataset, run_dir_env, measurements):
    """DS12 (R11): every artifact matches its manifest sha256, fit split train with the dataset's train samples (§23.2),
    no empty train cluster, z within ``cluster.z_crosscheck_atol`` of the dynamic export, the dynamic comparison items
    for the three splits and ``cluster_dynamic_gate_heatmap`` present. Values are recorded (TBD)."""
    from ilora.run import content_sha256

    cluster_run = _run_dir(run_dir_env, dataset, "CLUSTER")
    facts = FACTS[dataset]
    config = _config(cluster_run)
    folder = cluster_run / "cluster"
    manifest = _read_json(folder / "manifest.json")
    for name, sha in manifest["files"].items():
        assert content_sha256(folder / name) == sha, name
    assert manifest["fit_split"] == "train" and manifest["fit_rows"] == facts["train"]
    assert manifest["data"]["subset"] == "full"
    assert manifest["rows"] == {"train": facts["train"], "validation": facts["val"], "test": facts["test"]}
    kmeans = f"kmeans_k{config['cluster']['n_clusters']}_seed{config['cluster']['random_state']}.joblib"
    assert {kmeans, "centroids.npz", "cluster_summary.json", *(f"{n}_assignments.csv" for n in ASSIGNMENT_FILES)} <= \
        set(manifest["files"])
    summary = _read_json(folder / "cluster_summary.json")
    assert min(summary["train_cluster_size"]) > 0 and sum(summary["train_cluster_size"]) == facts["train"]
    assert config["cluster"]["dynamic_export_dir"] is not None
    dynamic = summary["dynamic_export"]
    assert isinstance(dynamic, dict) and dynamic["routing_mode"] == "dynamic"
    assert set(dynamic["z_max_abs_diff"]) == set(ASSIGNMENT_FILES)
    atol = float(config["cluster"]["z_crosscheck_atol"])
    assert all(value <= atol for value in dynamic["z_max_abs_diff"].values()), dynamic["z_max_abs_diff"]
    for item in ("cluster_dynamic_gate_mean", "cluster_dynamic_gate_std", "contingency_cluster_vs_dynamic_argmax",
                 "cluster_dynamic_argmax_distribution", "nmi_cluster_vs_dynamic_argmax"):
        assert isinstance(summary[item], dict) and set(summary[item]) == set(ASSIGNMENT_FILES), item
    for fmt in config["records"]["figure_formats"]:
        for figure in ("sasrec_pca_clusters", "cluster_dynamic_gate_heatmap"):
            assert (folder / "figures" / f"{figure}.{fmt}").is_file(), (figure, fmt)
    measurements[f"{dataset}_cluster_outputs"] = {
        "z_max_abs_diff": dynamic.get("z_max_abs_diff"),
        "nmi_cluster_vs_dynamic_argmax": summary["nmi_cluster_vs_dynamic_argmax"],
        "train_cluster_size": summary["train_cluster_size"],
        "silhouette_score_train": summary["silhouette_score_train"]}


@step("e4")
def test_e4_run(dataset, run_dir_env, measurements):
    """DS13 (R12): <d>_cluster_hard: no gate call in training, cluster lookups = micro-batches x micro-batch size of
    ``step_counts`` (§23.2), the live z check on the GPU passed within ``cluster.live_check_atol``, every evaluation
    gate = one-hot of mapping[assignment]. Records of the dataset's lengths; max |dz| is recorded (TBD)."""
    from ilora.cluster import parse_expert_mapping
    from ilora.run import resolve_path

    e4 = _run_dir(run_dir_env, dataset, "E4")
    cluster_run = _run_dir(run_dir_env, dataset, "CLUSTER")
    facts = FACTS[dataset]
    config = _config(e4)
    assert config["router"]["routing_mode"] == "cluster_hard"
    assert resolve_path(config["cluster"]["artifact_dir"]).resolve() == (cluster_run / "cluster").resolve()
    num_experts = int(config["lora"]["num_experts"])
    sc = _step_counts(e4, facts["train"])
    train = e4 / "records" / "train"
    steps = _csv_rows(train / "optimizer_steps.csv")
    assert (len(_csv_rows(train / "micro_batches.csv")), len(steps)) == (sc.total_micro, sc.total_steps)
    assert sum(int(row["gate_calls"]) for row in steps) == 0
    assert sum(int(row["cluster_lookups"]) for row in steps) == sc.total_micro * sc.micro_batch
    block = _read_json(e4 / "meta.json").get("cluster_live_check")
    atol, n = float(config["cluster"]["live_check_atol"]), int(config["cluster"]["live_check_n"])
    assert block is not None and block["passed"] is True and block["n"] == n and block["max_abs_diff"] <= atol, block
    assert str(block["device"]).startswith("cuda"), block
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
        assert _metrics(folder)["meta"]["gate_calls"] == 0, folder
        for record in _eval_records(folder):
            expert = mapping[assignment[record["sample_id"]]]
            assert _gates(record, num_experts) == [1.0 if k == expert else 0.0 for k in range(num_experts)], \
                record["sample_id"]
            checked += 1
    measurements[f"{dataset}_e4_run"] = {"records_checked": checked, "eval_folders": len(folders),
                                          "live_check_max_abs_diff": block["max_abs_diff"]}


@step("e4")
def test_e4_export(dataset, run_dir_env, measurements):
    """DS13 export (``test_e4_export`` of test_run_outputs): best_val and last pass the cross-check, omega is the
    one-hot of mapping[cluster] for every split with the dataset's row counts, the live z check of the export passed,
    and the gate analysis of K = 4 exists. max |d z| is recorded (TBD)."""
    import numpy as np

    from ilora.cluster import parse_expert_mapping

    e4, export = _run_dir(run_dir_env, dataset, "E4"), _run_dir(run_dir_env, dataset, "E4_EXPORT")
    cluster_run = _run_dir(run_dir_env, dataset, "CLUSTER")
    facts = FACTS[dataset]
    config = _config(export)
    assert config["router"]["routing_mode"] == "cluster_hard"
    num_experts, n_clusters = int(config["lora"]["num_experts"]), int(config["cluster"]["n_clusters"])
    mapping = parse_expert_mapping(config["cluster"]["expert_mapping"], n_clusters=n_clusters, num_experts=num_experts)
    assignment = {row["sample_id"]: int(row["cluster_id"]) for name in ASSIGNMENT_FILES
                  for row in _csv_rows(cluster_run / "cluster" / f"{name}_assignments.csv")}
    meta = _read_json(export / "meta.json")
    assert meta["status"] == "completed" and Path(meta["source_run"]).resolve() == e4.resolve()
    sizes = {"train": facts["train"], "validation": facts["val"], "test": facts["test"]}
    recorded = {}
    for epoch in _tested_epochs(e4):
        out = export / "records" / "export" / f"epoch{epoch}"
        cross = _read_json(out / "crosscheck.json")["splits"]
        assert all(cross[name]["passed"] and cross[name]["max_abs_diff_omega"] <= 1e-5
                   for name in ("validation", "test"))
        manifest = _read_json(out / "manifest.json")
        assert manifest["routing_mode"] == "cluster_hard" and manifest["cluster_live_check"]["passed"] is True
        for name, n in sizes.items():
            with np.load(out / f"representations_{name}.npz", allow_pickle=False) as npz:
                expected = np.array([mapping[assignment[str(s)]] for s in npz["sample_id"]], dtype=np.int64)
                omega = npz["omega"]
                assert npz["sample_id"].shape == (n,) and np.array_equal(npz["argmax_expert"], expected), (epoch, name)
                assert bool(((omega == 0) | (omega == 1)).all()) and bool((omega.sum(axis=1) == 1).all())
        for split in config["records"]["analysis_splits"]:
            assert (export / "records" / "analysis" / f"epoch{epoch}" / split / "gate_summary.json").is_file()
        recorded[f"epoch{epoch}"] = {name: cross[name]["max_abs_diff_z"] for name in ("validation", "test")}
    measurements[f"{dataset}_e4_export"] = recorded
