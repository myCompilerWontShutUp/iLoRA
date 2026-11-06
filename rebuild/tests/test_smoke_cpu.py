"""M13 smoke 1단계: CPU 전체 경로 (DESIGN.md §17.6.6, §17.6.8, §17.6.12 제안 2, §17.8 #3, #6).

``cpu_smoke_run`` (tests/conftest.py) runs ``smoke.run_smoke`` of ``smoke_cpu`` into a tmp run folder: S0 check-env,
S1 train (E2), S2 eval (reload), S3 export, S4 cluster, S5 train (E4), S6 eval (E3), S7 train (E1), S8 report. The
tests below assert every check of the 17.6.6 table twice: through the report's checks and again from the files
the steps wrote, with the 17.6.6 computed values (seed 1234: 156 train samples, 39 micro-batches per epoch, windows
4 x 9 + 3, T = 20, micro-batch rows 78, hybrid 10 / 28, gate calls 78; the 1-epoch E4 and E1 paths 39 rows,
hybrid 23, lookups 156 = 39 x 4). The overfit plan (§17.6.8) runs here on the tiny CPU model only to check its
mechanics; its pass criteria (loss ratio, 16 / 16 first lines) belong to the GPU (tests/vessl/test_run_outputs.py).
Synthetic values come from ``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest

from ilora import seeding

pytestmark = [pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated"),
              pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")]          # torch 2.0.0 torch.load

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
K = 4
PLAN_OF = {"smoke_cpu": "cpu", "smoke_gpu_100": "gpu100", "overfit_gpu": "overfit"}     # 17.8 #6
NEEDS_ARTIFACT_DIR = ("cluster_hard", "variant_user_level_cluster")   # cluster.artifact_dir only on the command line
OVERFIT_ROWS = (71, 196, 377, 554, 797, 1001, 1129, 1234, 1595, 1651, 1905, 2070, 2311, 2693, 2765, 2874)  # 17.6.8
# 17.6.6 computed values (design/tmp/m1314/smoke_sizes.py, curriculum_counts.py)
TRAIN_SAMPLES, MICRO_PER_EPOCH, WINDOWS = 156, 39, [4] * 9 + [3]
S1_MICRO, S1_STEPS, S1_HYBRID = 78, 20, [10, 28]
ONE_EPOCH_MICRO, ONE_EPOCH_HYBRID = 39, 23
SMOKE_EVAL = 10                       # data.smoke_users [2, 10, 10]: validation and test users (D14)


def _profile_names() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _load(profile_cfg, name: str, *sets: str):
    extra = ('cluster.artifact_dir="runs/rebuild/any_cluster_run/cluster"',) if name in NEEDS_ARTIFACT_DIR else ()
    return profile_cfg(name, *extra, *sets)


# ----------------------------------------------------------------------------------------- plan selection
@pytest.mark.criterion("M13-6")
@pytest.mark.parametrize("name", _profile_names())
def test_smoke_plan_selection(name, profile_cfg):
    """17.8 #6: overfit_gpu -> overfit, smoke_cpu -> cpu, smoke_gpu_100 -> gpu100; every profile whose data.subset is
    full (paper_spec, E1, E4, the variants) -> SmokeError (no smoke on the full data)."""
    from ilora.smoke import SmokeError, smoke_plan

    cfg = _load(profile_cfg, name)
    if name in PLAN_OF:
        assert smoke_plan(cfg) == PLAN_OF[name]
    else:
        assert cfg.data.subset == "full"
        with pytest.raises(SmokeError, match="data.subset = smoke"):
            smoke_plan(cfg)


@pytest.mark.criterion("M13-6")
def test_smoke_plan_uses_profile_values_only(profile_cfg):
    """The plan follows data.train_rows first, then model.backbone (17.6.6 계획 선택): train_rows on smoke_cpu gives
    overfit, the tiny CPU backbone on smoke_gpu_100 gives cpu, paper_spec with data.subset=smoke gives gpu100."""
    from ilora.smoke import smoke_plan

    rows = f"data.train_rows={list(OVERFIT_ROWS)}"
    assert smoke_plan(profile_cfg("smoke_cpu", "data.smoke_users=[16,10,10]", rows)) == "overfit"
    tiny = ("model.backbone=tiny_random", "model.torch_dtype=float32", "model.device=cpu", "train.precision=fp32")
    assert smoke_plan(profile_cfg("smoke_gpu_100", *tiny)) == "cpu"
    assert smoke_plan(profile_cfg("paper_spec", "data.subset=smoke")) == "gpu100"
    assert smoke_plan(profile_cfg("overfit_gpu")) == "overfit"


@pytest.mark.criterion("M13-3", "M13-6")
def test_plan_steps_follow_the_design():
    """The step tables of 17.6.6 (S0-S8), 17.6.7 (G0-G7: no E3 step, G6 = E1 path) and 17.6.8 (O0-O3): commands,
    folder names and the step --set texts (fields filled from earlier steps)."""
    from ilora.smoke import PLAN_PREFIX, PLAN_STEPS, PLANS

    assert PLANS == ("cpu", "gpu100", "overfit") and set(PLAN_STEPS) == set(PLANS)
    assert PLAN_PREFIX == {"cpu": "S", "gpu100": "G", "overfit": "O"}
    table = {plan: [(s.index, s.name, s.command, s.needs, s.sets) for s in PLAN_STEPS[plan]] for plan in PLANS}
    e4 = ("router.routing_mode=cluster_hard", "cluster.artifact_dir={cluster_dir}", "train.epochs=1")
    e1 = ("lora.num_experts=1", "router.routing_mode=none", "train.epochs=1")
    common = [(0, "check_env", "check-env", (), ()), (1, "train", "train", (0,), ()),
              (2, "eval", "eval", (1,), ("eval.source_run={train}",)),
              (3, "export", "export", (1,), ("records.source_run={train}",)),
              (4, "cluster", "cluster", (3,), ("cluster.dynamic_export_dir={export_epoch}",)),
              (5, "train_e4", "train", (4,), e4)]
    assert table["cpu"] == common + [(6, "eval_e3", "eval", (1,), ("eval.source_run={train}",
                                                                     "prompt.eval_prompt=text_only")),
                                     (7, "train_e1", "train", (0,), e1), (8, "report", "report", (), ())]
    assert table["gpu100"] == common + [(6, "train_e1", "train", (0,), e1), (7, "report", "report", (), ())]
    assert table["overfit"] == [(0, "check_env", "check-env", (), ()), (1, "train", "train", (0,), ()),
                                (2, "generation", "generation", (1,), ()), (3, "report", "report", (), ())]


@pytest.mark.criterion("M13-3", "M13-6")
def test_required_checks(profile_cfg):
    """Every row of the 17.6.6-17.6.8 tables has its checks: S1 with batch invariance (eval.batch_invariance_check),
    G0 with the GPU check of a CUDA profile, O1 with all_hybrid and loss_ratio, O2 with the first-line match."""
    from ilora.smoke import SmokeError, required_checks

    cpu = required_checks("cpu", profile_cfg("smoke_cpu"))
    gpu = required_checks("gpu100", profile_cfg("smoke_gpu_100"))
    overfit = required_checks("overfit", profile_cfg("overfit_gpu"))
    assert len(set(cpu)) == len(cpu) and len(set(gpu)) == len(gpu) and len(set(overfit)) == len(overfit)
    assert {"S0.sasrec_reference", "S1.record_rows", "S1.plan_digest", "S1.batch_invariance", "S2.reproduction",
            "S3.crosscheck", "S4.joblib_reload", "S4.z_vs_export", "S5.live_check", "S5.one_hot_gates",
            "S6.not_comparable", "S7.gate_one"} <= set(cpu)
    assert not any(c.startswith(("S0.gpu", "S8.")) for c in cpu)
    assert {"G0.gpu", "G1.batch_invariance", "G2.records_sha256", "G5.cluster_lookups", "G6.gate_one"} <= set(gpu)
    assert not any(c.startswith(("G7.", "G6.not_comparable")) for c in gpu)
    assert {"O1.all_hybrid", "O1.loss_ratio", "O1.consecutive_skips", "O2.checkpoint", "O2.first_line_match"} <= \
        set(overfit)
    assert "O1.batch_invariance" not in overfit and "O0.gpu" in overfit
    assert "S1.batch_invariance" not in required_checks("cpu", profile_cfg("smoke_cpu",
                                                                            "eval.batch_invariance_check=false"))
    with pytest.raises(SmokeError):
        required_checks("full", profile_cfg("smoke_cpu"))


@pytest.mark.criterion("M13-3")
def test_frozen_sasrec_check_compares_saved_values(cpu_cfg):
    """The ``frozen_sasrec`` check (17.6.7 #2) compares each epoch's ``frozen_checksum_at_save`` (recomputed by M10
    at the epoch end) with the start value of epoch 0 and the saved SASRec digest with the profile digest, so a
    changed frozen parameter or SASRec state fails it (the stored start value alone is equal by construction)."""
    from ilora.smoke import frozen_sasrec_values

    start = {"llm.model.embed_tokens.weight": 1.5, "sasrec.item_emb.weight": -2.25}
    digest = cpu_cfg.sasrec.state_digest

    class Files:
        def __init__(self, *states: dict) -> None:
            self.states = states

        def train_state(self, epoch: int) -> dict:
            return self.states[epoch]

    def state(at_save: dict, sasrec: str = digest) -> dict:
        return {"frozen_checksum": dict(start), "frozen_checksum_at_save": dict(at_save), "sasrec_state_digest": sasrec}

    names = ["epoch0", "epoch1"]
    measured, expected = frozen_sasrec_values(Files(state(start), state(start)), cpu_cfg, names)
    assert measured == expected
    changed = {**start, "llm.model.embed_tokens.weight": 1.5 + 2.0 ** -20}
    measured, expected = frozen_sasrec_values(Files(state(start), state(changed)), cpu_cfg, names)
    assert measured["epoch1"]["frozen_checksum_at_save_equals_start"] is False and measured != expected
    measured, expected = frozen_sasrec_values(Files(state(start), state(start, sasrec="0" * 64)), cpu_cfg, names)
    assert measured["epoch1"]["sasrec_state_digest"] == "0" * 64 and measured != expected


# ----------------------------------------------------------------------------------------- helpers
def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def train_rows(run_dir: Path, num_experts: int) -> tuple[list, list, list]:
    """micro_batches.csv, optimizer_steps.csv and epochs.csv read back with the M12 column types."""
    from ilora import records

    folder = run_dir / "records" / "train"
    return (records.read_rows(folder / "micro_batches.csv", records.MICRO_FIELDS),
            records.read_rows(folder / "optimizer_steps.csv", records.step_fields(num_experts)),
            records.read_rows(folder / "epochs.csv", records.EPOCH_FIELDS))


def completed_evals(run_dir: Path) -> dict[str, Path]:
    """``"epoch{e}/{split}"`` -> folder of every completed evaluation (``metrics.json`` written last)."""
    return {f"{p.parent.parent.name}/{p.parent.name}": p.parent for p in sorted(run_dir.glob("eval/*/*/metrics.json"))}


def epochs_tested(run_dir: Path) -> tuple[int, ...]:
    from ilora.train import epochs_to_test

    selection = read_json(run_dir / "eval" / "selection.json")
    return epochs_to_test(selection["best_val_epoch"], selection["last_epoch"])


def key_counts(state: dict) -> dict:
    return {"lora": sum(".lora_A." in k or ".lora_B." in k for k in state),
            "projector": sum(k.startswith("projector.") for k in state),
            "gate": sum(k.startswith("omega_source.") for k in state),
            "generated_frozen": sum(k.startswith("generated_frozen.") for k in state), "total": len(state)}


def optimizer_groups(ckpt: Path) -> list[str]:
    import torch

    state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
    return [pg["name"] for pg in state["optimizer"]["param_groups"]]


def passed_step(report, sid: str, kind: str) -> Path:
    """The step completed, every check of its 17.6.6 row is in the report and passed; returns its run folder."""
    from ilora.smoke import KIND_CHECKS, STEPS_DIR

    step = report.step(sid)
    assert step.status == "completed" and step.error is None, (sid, step.error)
    for name in KIND_CHECKS[kind]:
        check = report.check(f"{sid}.{name}")
        assert check.passed is True, (check.id, check.measured, check.expected)
    run_dir = report.step_dir(sid)
    assert run_dir.parent == report.run_dir / STEPS_DIR
    assert re.fullmatch(rf"{sid}_{step.name}_\d{{8}}T\d{{6}}Z", run_dir.name), run_dir.name
    assert read_json(run_dir / "meta.json")["status"] == "completed"
    return run_dir


# ----------------------------------------------------------------------------------------- 17.8 #3
@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_cpu_smoke_passes(cpu_smoke_run, cpu_cfg):
    """17.8 #3: the CPU smoke passes (every step completed, every check of the plan present and passed, report
    written) and its records hold the 17.6.6 values: 156 samples, micro 78, step 20, hybrid 10 / 28, gate calls 78,
    cluster lookups 156."""
    from ilora import smoke

    report = cpu_smoke_run
    assert isinstance(report, smoke.SmokeReport) and report.plan == "cpu"
    assert report.passed is True, (report.failed_checks(), [(s.id, s.error) for s in report.steps if s.error])
    assert [s.id for s in report.steps] == [f"S{i}" for i in range(9)]
    assert all(s.status == "completed" for s in report.steps)
    ids = [c.id for c in report.checks]
    assert len(ids) == len(set(ids)) and set(smoke.required_checks("cpu", cpu_cfg)) <= set(ids)
    assert all(c.passed is not False for c in report.checks)
    doc = read_json(report.run_dir / smoke.REPORT_FILE)
    assert (doc["format"], doc["plan"], doc["passed"], doc["failed_checks"], doc["failed_steps"]) == \
        (smoke.REPORT_FORMAT, "cpu", True, [], [])
    measured = {c.id: c.measured for c in report.checks}
    assert measured["S1.record_rows"] == {"micro": S1_MICRO, "step": S1_STEPS, "epoch": 2, "tau_consecutive": True,
                                          "step_consecutive": True}
    assert measured["S1.hybrid"] == {"epochs": S1_HYBRID, "steps": sum(S1_HYBRID), "micro_equal_plan": [True, True]}
    assert measured["S1.gate_calls"] == {"steps": S1_MICRO, "run_summary": S1_MICRO}
    assert measured["S4.fit_train_only"]["fit_rows"] == TRAIN_SAMPLES
    assert measured["S5.cluster_lookups"] == {"steps": TRAIN_SAMPLES, "run_summary": TRAIN_SAMPLES}   # 39 x 4
    assert measured["S5.gate_calls"] == {"steps": 0, "run_summary": 0}


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s0_check_env(cpu_smoke_run):
    """S0: M0 environment check (no GPU on CPU), input hashes, candidates.self_check(), SASRec against the reference
    with CPU difference 0, tokenizer sha256."""
    from ilora import run

    run_dir = passed_step(cpu_smoke_run, "S0", "check_env")
    report = read_json(run_dir / "check_env.json")
    checks = {c["check"]: c for c in report["checks"]}
    assert report["passed"] is True and report["device"] == "cpu" and set(checks) == {"data", "sasrec", "tokenizer"}
    assert checks["data"]["result"] == {"subset": "smoke", "train": TRAIN_SAMPLES, "val": SMOKE_EVAL, "test": SMOKE_EVAL}
    reference = checks["sasrec"]["result"]["reference"]
    assert reference["max_abs_diff"] == 0.0 and reference["bitwise"] is True
    assert checks["tokenizer"]["result"]["vocab_size"] == 32_005
    meta = read_json(run_dir / "meta.json")
    assert meta["preparation"]["environment_check"] == "passed"
    assert meta["preparation"]["candidates_self_check"] == "passed"
    assert meta["inputs"] == {rel: sha for rel, (sha, _) in run.EXPECTED_INPUTS.items()}


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s1_train_e2(cpu_smoke_run, smoke_data):
    """S1: five files per epoch, 36 trainable.pt keys with the projector in both epochs, 78 / 20 / 2 record rows
    (epoch-end window 3), hybrid 10 + 28 = 38, gate calls 78, projector_has_grad = hybrid in the window, plan digests
    recomputed, validation twice and test of best and last with 10 samples, batch invariance 10 / 10 tokens."""
    from ilora.config import load_config
    from ilora.prompts import build_batch_plan
    from ilora.smoke import CHECKPOINT_FILES
    from ilora.train import read_trainable_state

    report = cpu_smoke_run
    run_dir = passed_step(report, "S1", "train_e2")
    assert report.check("S1.batch_invariance").passed is True
    cfg, _ = load_config(run_dir / "config.yaml")
    for epoch in (0, 1):
        ckpt = run_dir / "checkpoints" / f"epoch{epoch}"
        assert sorted(p.name for p in ckpt.iterdir()) == sorted(CHECKPOINT_FILES)
        assert key_counts(read_trainable_state(ckpt)) == {"lora": 28, "projector": 4, "gate": 1, "generated_frozen": 3,
                                                          "total": 36}
        assert optimizer_groups(ckpt) == ["projector", "gate", "lora"]
    micro, steps, epochs = train_rows(run_dir, K)
    assert (len(micro), len(steps), len(epochs)) == (S1_MICRO, S1_STEPS, 2)
    assert [sum(r["epoch"] == e for r in micro) for e in (0, 1)] == [MICRO_PER_EPOCH] * 2
    assert [r["n_micro"] for r in steps] == WINDOWS * 2
    assert [r["n_hybrid"] for r in epochs] == S1_HYBRID and sum(r["n_hybrid"] for r in steps) == sum(S1_HYBRID)
    assert sum(r["gate_calls"] for r in steps) == S1_MICRO and all(r["cluster_lookups"] == 0 for r in steps)
    assert all(r["projector_has_grad"] == (r["n_hybrid"] > 0) for r in steps)
    assert [r["plan_digest"] for r in epochs] == [build_batch_plan(smoke_data.train, e, cfg).digest() for e in (0, 1)]
    evals = completed_evals(run_dir)
    assert set(evals) == {"epoch0/val", "epoch1/val"} | {f"epoch{b}/test" for b in epochs_tested(run_dir)}
    for folder in evals.values():
        assert read_json(folder / "metrics.json")["n_total"] == SMOKE_EVAL
        invariance = read_json(folder / "batch_invariance.json")
        assert invariance["n"] == invariance["tokens_equal"] == SMOKE_EVAL


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s2_eval_reload(cpu_smoke_run):
    """S2: main.py eval of the S1 checkpoints; for (best, last) x (val, test) records.jsonl has the sha256 of the
    training-time record and summary.json says identical (checkpoint round trip)."""
    from ilora.run import content_sha256

    report = cpu_smoke_run
    run_dir, source = passed_step(report, "S2", "eval_reload"), report.step_dir("S1")
    summary = read_json(run_dir / "eval" / "summary.json")
    assert Path(summary["source_run"]).resolve() == source.resolve()
    keys = [f"epoch{e}/{split}" for e in epochs_tested(source) for split in ("val", "test")]
    assert list(summary["results"]) == keys
    for key in keys:
        assert summary["results"][key]["reproduction"]["status"] == "identical"
        assert content_sha256(run_dir / "eval" / key / "records.jsonl") == \
            content_sha256(source / "eval" / key / "records.jsonl")


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s3_export(cpu_smoke_run):
    """S3: crosscheck.json passed (|d omega| <= 1e-5), NPZ rows train 156, validation 10, test 10, the analysis
    folders of test, val and train with their figures."""
    from ilora import records

    report = cpu_smoke_run
    run_dir, source = passed_step(report, "S3", "export"), report.step_dir("S1")
    epochs = epochs_tested(source)
    assert sorted(p.name for p in (run_dir / "records" / "export").iterdir()) == [f"epoch{e}" for e in epochs]
    for epoch in epochs:
        out = run_dir / "records" / "export" / f"epoch{epoch}"
        cross = read_json(out / "crosscheck.json")["splits"]
        assert all(cross[name]["passed"] and cross[name]["max_abs_diff_omega"] <= 1e-5 for name in ("validation", "test"))
        rows = [len(records.load_npz(out / f"representations_{name}.npz")) for name in ("train", "validation", "test")]
        assert rows == [TRAIN_SAMPLES, SMOKE_EVAL, SMOKE_EVAL]
        for split in ("test", "val", "train"):
            folder = run_dir / "records" / "analysis" / f"epoch{epoch}" / split
            assert (folder / "gate_summary.json").is_file() and (folder / "gate_interpretation.txt").is_file()
            assert (folder / "figures" / "gate_heatmap.png").is_file()
            assert (folder / "figures" / "gate_ten_sequences.png").is_file() == (split == "test")


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s4_cluster(cpu_smoke_run, smoke_data):
    """S4: fit on the 156 train rows only, no empty cluster, the reloaded joblib model predicts the CSV clusters,
    z equal to the S3 export z within 1e-5, the dynamic comparison items and heatmap present."""
    from ilora import cluster, records

    report = cpu_smoke_run
    run_dir = passed_step(report, "S4", "cluster")
    out = run_dir / "cluster"
    manifest, summary = read_json(out / "manifest.json"), read_json(out / "cluster_summary.json")
    train_ids = [s.sample_id for s in smoke_data.train]
    assert (manifest["fit_split"], manifest["fit_rows"]) == ("train", TRAIN_SAMPLES)
    assert manifest["fit_ids_sha256"] == cluster.ids_sha256(train_ids)
    assert [row["sample_id"] for row in read_csv(out / "train_assignments.csv")] == train_ids
    assert min(summary["train_cluster_size"]) > 0 and sum(summary["train_cluster_size"]) == TRAIN_SAMPLES
    km = cluster.load_kmeans(out / "kmeans_k4_seed1234.joblib")
    for name in ("train", "validation", "test"):
        table = records.load_npz(out / f"representations_{name}.npz")
        rows = read_csv(out / f"{name}_assignments.csv")
        assert [r["sample_id"] for r in rows] == list(table.sample_ids)
        assert cluster.predict(km, table.z).tolist() == [int(r["cluster_id"]) for r in rows]
    best = read_json(report.step_dir("S1") / "eval" / "selection.json")["best_val_epoch"]
    export = report.step_dir("S3") / "records" / "export" / f"epoch{best}"
    assert Path(summary["dynamic_export"]["path"]).resolve() == export.resolve()
    assert all(value <= 1e-5 for value in summary["dynamic_export"]["z_max_abs_diff"].values())
    for item in cluster.DYNAMIC_ITEMS:
        assert set(summary[item]) == {"train", "validation", "test"}, item
    assert (out / "figures" / "cluster_dynamic_gate_heatmap.png").is_file()


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s5_train_e4(cpu_smoke_run):
    """S5: no gate group, gate calls 0, cluster lookups 156 (39 x 4), the live z check passed, evaluation gates are
    the one-hot of mapping[cluster of the S4 assignment CSV], 35 trainable.pt keys."""
    from ilora import cluster, run
    from ilora.config import load_config
    from ilora.train import read_trainable_state

    report = cpu_smoke_run
    run_dir = passed_step(report, "S5", "train_e4")
    cfg, _ = load_config(run_dir / "config.yaml")
    artifacts = report.step_dir("S4") / "cluster"
    assert cfg.router.routing_mode == "cluster_hard" and cfg.train.epochs == 1
    assert run.resolve_path(cfg.cluster.artifact_dir).resolve() == artifacts.resolve()
    ckpt = run_dir / "checkpoints" / "epoch0"
    assert optimizer_groups(ckpt) == ["projector", "lora"]
    assert key_counts(read_trainable_state(ckpt)) == {"lora": 28, "projector": 4, "gate": 0, "generated_frozen": 3,
                                                      "total": 35}
    micro, steps, epochs = train_rows(run_dir, K)
    assert (len(micro), len(steps), [r["n_hybrid"] for r in epochs]) == (ONE_EPOCH_MICRO, 10, [ONE_EPOCH_HYBRID])
    assert sum(r["gate_calls"] for r in steps) == 0 and sum(r["cluster_lookups"] for r in steps) == TRAIN_SAMPLES
    assert [r["cluster_lookups"] for r in steps] == [4 * r["n_micro"] for r in steps]
    live = read_json(run_dir / "meta.json")["cluster_live_check"]
    assert live["passed"] is True and live["n"] == 64 and live["max_abs_diff"] <= 1e-4
    assert "cluster live check passed: n=64" in (run_dir / "run.log").read_text(encoding="utf-8")
    mapping = cluster.parse_expert_mapping(cfg.cluster.expert_mapping, n_clusters=4, num_experts=K)
    assignment = {r["sample_id"]: int(r["cluster_id"]) for name in ("train", "validation", "test")
                  for r in read_csv(artifacts / f"{name}_assignments.csv")}
    evals = completed_evals(run_dir)
    assert evals
    for folder in evals.values():
        assert read_json(folder / "metrics.json")["meta"]["gate_calls"] == 0
        for line in read_jsonl(folder / "records.jsonl"):
            expert = mapping[assignment[line["sample_id"]]]
            assert [line[f"gate_{k}"] for k in range(K)] == [1.0 if k == expert else 0.0 for k in range(K)]


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s6_eval_e3(cpu_smoke_run):
    """S6: text-only evaluation of the S1 checkpoints: every record hybrid false, metrics files present, the
    reproduction marked not_comparable because prompt.eval_prompt differs (17.6.12 불일치 3)."""
    report = cpu_smoke_run
    run_dir, source = passed_step(report, "S6", "eval_e3"), report.step_dir("S1")
    summary = read_json(run_dir / "eval" / "summary.json")
    keys = [f"epoch{e}/{split}" for e in epochs_tested(source) for split in ("val", "test")]
    assert list(summary["results"]) == keys
    for key in keys:
        reproduction = summary["results"][key]["reproduction"]
        assert reproduction["status"] == "not_comparable" and reproduction["differing_keys"] == ["prompt.eval_prompt"]
        folder = run_dir / "eval" / key
        assert {"records.jsonl", "metrics.json", "representations.npz"} <= {p.name for p in folder.iterdir()}
        assert read_json(folder / "metrics.json")["meta"]["eval_prompt"] == "text_only"
        assert all(line["hybrid"] is False for line in read_jsonl(folder / "records.jsonl"))


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s7_train_e1(cpu_smoke_run):
    """S7: K = 1 (routing none): groups {projector, lora}, evaluation gate_0 = 1.0, step rows omega_mean_0 = 1.0,
    35 trainable.pt keys."""
    from ilora.train import read_trainable_state

    run_dir = passed_step(cpu_smoke_run, "S7", "train_e1")
    ckpt = run_dir / "checkpoints" / "epoch0"
    assert optimizer_groups(ckpt) == ["projector", "lora"]
    assert key_counts(read_trainable_state(ckpt))["total"] == 35
    micro, steps, _ = train_rows(run_dir, 1)
    assert len(micro) == ONE_EPOCH_MICRO and all(r["omega_mean_0"] == 1.0 for r in steps)
    assert all(r["omega_entropy_mean"] is None and r["gate_calls"] == 0 for r in steps)
    evals = completed_evals(run_dir)
    assert evals
    for folder in evals.values():
        for line in read_jsonl(folder / "records.jsonl"):
            assert line["gate_0"] == 1.0 and "gate_1" not in line


@pytest.mark.slow
@pytest.mark.criterion("M13-3")
def test_s8_report(cpu_smoke_run):
    """S8: smoke_report.json holds the plan, every step with its run folder, status and seconds, every check with
    id, description, passed and measured value, and passed; it agrees with the returned SmokeReport."""
    from ilora import smoke

    report = cpu_smoke_run
    doc = read_json(report.run_dir / smoke.REPORT_FILE)
    assert doc["plan"] == report.plan == "cpu" and doc["passed"] is report.passed is True
    assert [s["id"] for s in doc["steps"]] == [s.id for s in report.steps]
    for step in doc["steps"][:-1]:
        assert step["status"] == "completed" and step["seconds"] > 0
        assert Path(step["run_dir"]).parent == report.run_dir / smoke.STEPS_DIR
    assert doc["steps"][-1]["id"] == "S8" and Path(doc["steps"][-1]["run_dir"]) == report.run_dir
    assert sorted(p.name.split("_20")[0] for p in (report.run_dir / smoke.STEPS_DIR).iterdir()) == \
        ["S0_check_env", "S1_train", "S2_eval", "S3_export", "S4_cluster", "S5_train_e4", "S6_eval_e3", "S7_train_e1"]
    assert [c["id"] for c in doc["checks"]] == [c.id for c in report.checks]
    for check, item in zip(report.checks, doc["checks"]):
        assert set(item) == {"id", "step", "description", "passed", "measured", "expected"}
        assert item["id"].startswith(f"{item['step']}.") and item["description"] and item["passed"] is check.passed
    recorded = {c.id for c in report.checks if c.passed is None}
    assert recorded and all(c.split(".")[1] in ("measurements", "batch_invariance") for c in recorded)
    assert doc["counts"]["failed"] == 0 and doc["counts"]["checks"] == len(report.checks)


# ----------------------------------------------------------------------------------------- overfit plan (17.6.8)
OVERFIT_CPU_SETS = ("data.smoke_users=[16,10,10]", f"data.train_rows={list(OVERFIT_ROWS)}",
                    "prompt.train_curriculum=all_hybrid", "train.micro_batch_size=4", "train.grad_accum=1")


class EchoModel:
    """Stand-in for the M9 model in ``overfit_generation_check``: ``generate`` answers the target title of each row
    (a wrong title for the sample ids in ``wrong``) followed by a newline and EOS, pads with [PAD] like greedy
    generation, and keeps the batches it was given. omega is uniform, z comes from ``torch_seed_scope(0, "test",
    "overfit_z")`` (17.6.2)."""

    def __init__(self, bundle, *, wrong=()):
        import torch

        self.bundle, self.wrong, self.batches = bundle, set(wrong), []
        self.device = torch.device("cpu")
        self.training = True
        self.sasrec = type("Encoder", (), {"hidden_size": 64})()

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def generate(self, batch):
        import torch

        from ilora.batching import encode_text
        from ilora.model import GenerationOutput

        assert not self.training, "generate in eval mode"
        self.batches.append(batch)
        ids = self.bundle.ids
        rows = [encode_text(self.bundle, ("Wrong Title (1999)" if sid in self.wrong else title) + "\n") + [ids.eos]
                for sid, title in zip(batch.sample_ids, batch.target_titles)]
        width = max(len(r) for r in rows)
        tokens = torch.tensor([r + [ids.pad] * (width - len(r)) for r in rows], dtype=torch.int64)
        with seeding.torch_seed_scope(0, "test", "overfit_z"):
            z = torch.randn(len(rows), 64)
        return GenerationOutput(new_token_ids=tokens, omega=torch.full((len(rows), K), 0.25), z=z)


@pytest.mark.criterion("M13-8")
def test_overfit_generation_check_keys(profile_cfg, full_data, bundle, templates):
    """17.6.8 O2 on a stand-in model: every train sample gets the candidates and template of the last training epoch
    E-1 (M2 candidates_for, M4 choose_template with epoch E-1), the hybrid prompt [BOS] + p left-padded by
    collate_eval in groups of eval.batch_size in row order; a sample matches when match_norm(first line) =
    match_norm(target title): 16 / 16 for the echo, 13 / 16 with three wrong answers; the model mode is restored."""
    from ilora.batching import encode_eval
    from ilora.candidates import candidates_for
    from ilora.data import select_smoke
    from ilora.prompts import build_prompt, choose_template
    from ilora.smoke import overfit_generation_check

    cfg = profile_cfg("smoke_cpu", *OVERFIT_CPU_SETS, "train.epochs=25")
    data = select_smoke(full_data, cfg.run.seed, (16, 10, 10), train_rows=OVERFIT_ROWS)
    assert [s.row for s in data.train] == list(OVERFIT_ROWS)
    model = EchoModel(bundle)
    result = overfit_generation_check(model, bundle, templates, data, cfg)
    assert model.training is True                                       # restored
    assert (result["epoch"], result["n"], result["n_match"], result["all_match"]) == (24, 16, 16, True)
    assert [len(b.sample_ids) for b in model.batches] == [8, 8] and all(b.hybrid for b in model.batches)
    batches = {sid: (b, i) for b in model.batches for i, sid in enumerate(b.sample_ids)}
    assert list(batches) == [s.sample_id for s in data.train]
    changed = 0
    for s in data.train:
        batch, i = batches[s.sample_id]
        cands = candidates_for(s, main_seed=cfg.run.seed, epoch=24, cfg=cfg.candidates)
        template_id = choose_template(cfg.prompt, cfg.run.seed, len(templates.templates), s, 24)
        assert tuple(batch.cand_ids[i].tolist()) == cands and batch.template_ids[i] == template_id
        expected = encode_eval(bundle, build_prompt(s, cands, data.catalog, templates, template_id, True))
        mask = batch.attention_mask[i].tolist()
        n = sum(mask)
        assert mask == [0] * (len(mask) - n) + [1] * n                          # left padding (D22)
        assert batch.input_ids[i].tolist()[len(mask) - n:] == expected
        changed += cands != candidates_for(s, main_seed=cfg.run.seed, epoch=0, cfg=cfg.candidates)
    assert changed > 0                                                  # the epoch key matters
    wrong = [data.train[i].sample_id for i in (0, 7, 15)]
    result = overfit_generation_check(EchoModel(bundle, wrong=wrong), bundle, templates, data, cfg)
    assert result["n_match"] == 13 and not result["all_match"]
    assert [s["sample_id"] for s in result["samples"] if not s["match"]] == wrong


@pytest.mark.slow
@pytest.mark.criterion("M13-8")
def test_overfit_plan_runs_on_cpu(tokenizer_dir, sasrec_state_path, tmp_path):
    """The overfit plan (O0 check-env, O1 train, O2 generation check on a new model with the last checkpoint, O3
    report) through M0 run_command('smoke') on the tiny CPU model, 2 epochs: the steps complete and every check is
    evaluated; the random tiny model neither reaches the loss ratio nor the 16 first lines, so the report fails
    exactly there, the run is marked failed and the command raises (main.py exits non-zero)."""
    from ilora import commands, smoke
    from ilora.config import load_config

    sets = [*OVERFIT_CPU_SETS, "train.epochs=2", "eval.batch_invariance_check=false"]
    profile = CONFIG_DIR / "smoke_cpu.yaml"
    cfg, _ = load_config(profile, sets)
    assert smoke.smoke_plan(cfg) == "overfit"
    with pytest.raises(commands.CommandError, match="smoke checks failed"):
        commands.run_command("smoke", cfg, sets, profile="overfit_cpu", root=tmp_path, profile_path=profile,
                             tokenizer_dir=tokenizer_dir, argv=["test_overfit_plan_runs_on_cpu"])
    (run_dir,) = tmp_path.iterdir()
    meta, doc = read_json(run_dir / "meta.json"), read_json(run_dir / smoke.REPORT_FILE)
    assert meta["status"] == "failed" and meta["smoke"]["plan"] == "overfit" and meta["smoke"]["passed"] is False
    assert doc["plan"] == "overfit" and doc["passed"] is False
    assert [(s["id"], s["status"]) for s in doc["steps"]] == [("O0", "completed"), ("O1", "completed"),
                                                              ("O2", "completed"), ("O3", "completed")]
    checks = {c["id"]: c for c in doc["checks"]}
    assert set(smoke.required_checks("overfit", cfg)) <= set(checks)
    assert set(doc["failed_checks"]) <= {"O1.loss_ratio", "O2.first_line_match"}
    assert checks["O2.first_line_match"]["passed"] is False                 # a random model answers no title
    assert checks["O1.record_rows"]["measured"] == {"micro": 8, "step": 8, "epoch": 2, "tau_consecutive": True,
                                                    "step_consecutive": True}
    assert checks["O1.all_hybrid"]["measured"] == {"micro": 8, "hybrid": 8}
    assert math.isfinite(checks["O1.loss_ratio"]["measured"]["ratio"])
    assert checks["O2.checkpoint"]["measured"]["epoch"] == 1 and checks["O2.first_line_match"]["measured"]["n"] == 16
