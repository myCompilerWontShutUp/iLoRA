"""M13 CPU smoke of the other datasets (DESIGN.md §23.6 로컬 CPU smoke, §23.8 #12, §17.6.6).

``ds_smoke`` runs ``smoke.run_smoke`` of ``steam_smoke_cpu`` and ``lastfm_smoke_cpu`` into a tmp run folder, the call of
the ``cpu_smoke_run`` fixture of tests/conftest.py with the plan S0-S8 on the tiny random LLaMA (S0 check-env, S1 train
(E2), S2 eval, S3 export, S4 cluster, S5 train (E4), S6 eval (E3), S7 train (E1), S8 report); one run per dataset,
shared by the tests below. It needs the test tokenizer (``tokenizer_dir``, ``SKIP_NO_TOKENIZER``) and the converted
SASRec files of the dataset (``runs/rebuild/assets/sasrec_<d>/``, ``tools/convert_sasrec.py --dataset <d>``,
``SKIP_NO_SASREC``). The checks of the plan are those of the MovieLens smoke (tests/test_smoke_cpu.py), evaluated on
the dataset's own data; the numbers below are the computed values of §23.2 and §23.6. No accuracy is judged.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import SKIP_NO_SASREC
from ilora import run, smoke
from ilora.config import load_config
from ilora.data import load_dataset
from ilora.datasets import SPECS

pytestmark = [pytest.mark.slow,
              pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated"),
              pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")]          # torch 2.0.0 torch.load

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
PROFILE_DIR = REBUILD_DIR / "configs" / "datasets"

K = 4                                 # lora.num_experts of the E2 path (D1)
SMOKE_EVAL = 10                       # data.smoke_users [2, 10, 10]: validation and test users (D14, §23.6)
# §23.6 table (multi/critic/c7_steps.py, multi/critic/c03_smoke_math.py): train samples, micro-batches per epoch
# (size 4, drop_last: 80 // 4, 41 // 4), windows of train.grad_accum 4 (D24), steps of the 2-epoch run S1 and of the
# 1-epoch runs S5 and S7, cluster.live_check_n. state_digest: converted SASRec state, §23.2 hash table.
DESIGN = {
    "steam": SimpleNamespace(train=80, micro=20, windows=[4, 4, 4, 4, 4], steps=(10, 5), live_check_n=64,
                             state_digest="d5d13fc0bc0f3c8ebc2f64e4247e9da3b486e6984c931bb8454ebbb17c8b3932"),
    "lastfm": SimpleNamespace(train=41, micro=10, windows=[4, 4, 2], steps=(6, 3), live_check_n=32,
                              state_digest="6b0d161351901a272fe07ef4c2bf3ccc99e06a487bcaee1c9b88392f59435d85"),
}
COMMAND_STEPS = tuple(f"S{i}" for i in range(8))        # S8 is the report step, it writes in the smoke folder itself
EVAL_STEPS = ("S1", "S2", "S5", "S6", "S7")             # the steps that write eval/{ckpt}/{split}/metrics.json


@pytest.fixture(scope="module", params=list(DESIGN))
def ds_smoke(request, tokenizer_dir, tmp_path_factory):
    """``smoke.run_smoke`` of ``<name>_smoke_cpu`` into a tmp run folder (§23.6) -> name, config, report, data, design.
    Missing converted weights skip with the documented ``SKIP_NO_SASREC`` (§23.8); pytest 9 ignores marks on fixtures,
    so ``slow`` is in ``pytestmark`` above."""
    name = request.param
    profile = PROFILE_DIR / f"{name}_smoke_cpu.yaml"
    cfg, _ = load_config(profile)
    converted = (cfg.sasrec.state_path, cfg.sasrec.reference_path)
    if not all(run.resolve_path(path, REPO_ROOT).is_file() for path in converted):
        pytest.skip(SKIP_NO_SASREC)
    run_dir = run.create_run_dir(f"{name}_smoke_cpu", root=tmp_path_factory.mktemp(f"{name}_smoke"))
    report = smoke.run_smoke(profile, [], run_dir, repo_root=REPO_ROOT, llm_path=None, tokenizer_dir=tokenizer_dir)
    return SimpleNamespace(name=name, cfg=cfg, report=report, design=DESIGN[name], data=load_dataset(cfg, REPO_ROOT))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_folders(step_dir: Path) -> list[Path]:
    """Completed evaluation folders of a step (``metrics.json`` is written last, §15.6.8)."""
    return [p.parent for p in sorted(Path(step_dir).glob("eval/*/*/metrics.json"))]


def test_smoke_passes(ds_smoke):
    """§23.8 #12: the steps S0-S8 completed, every check of the plan is present and none failed, and the report file
    says the same."""
    report = ds_smoke.report
    assert isinstance(report, smoke.SmokeReport) and report.plan == "cpu"
    assert report.passed is True, (report.failed_checks(), [(s.id, s.error) for s in report.steps if s.error])
    assert [s.id for s in report.steps] == [f"S{i}" for i in range(9)]
    assert all(s.status == "completed" and s.error is None for s in report.steps), report.steps
    ids = [c.id for c in report.checks]
    assert len(ids) == len(set(ids)) and set(smoke.required_checks("cpu", ds_smoke.cfg)) <= set(ids)
    assert not report.failed_checks()
    doc = _read_json(report.run_dir / smoke.REPORT_FILE)
    assert (doc["format"], doc["plan"], doc["passed"], doc["failed_checks"], doc["failed_steps"]) == \
        (smoke.REPORT_FORMAT, "cpu", True, [], [])
    assert (doc["counts"]["checks"], doc["counts"]["failed"], doc["counts"]["steps_completed"]) == (len(ids), 0, 9)


def test_s0_loads_the_smoke_selection(ds_smoke):
    """S0 (§23.6): the smoke selection [2, 10, 10] of the dataset, 80 train samples for Steam and 41 for LastFM from two
    users, 10 validation and 10 test samples (D14), on CPU."""
    doc = _read_json(ds_smoke.report.step_dir("S0") / "check_env.json")
    checks = {c["check"]: c for c in doc["checks"]}
    assert doc["passed"] is True and doc["device"] == "cpu" and set(checks) == {"data", "sasrec", "tokenizer"}
    assert all(c["passed"] is True for c in checks.values())
    assert checks["data"]["result"] == {"subset": "smoke", "train": ds_smoke.design.train, "val": SMOKE_EVAL,
                                        "test": SMOKE_EVAL}
    data = ds_smoke.data
    assert (len(data.train), len(data.val), len(data.test)) == (ds_smoke.design.train, SMOKE_EVAL, SMOKE_EVAL)
    assert len({s.user_id for s in data.train}) == 2
    assert data.catalog.num_items == SPECS[ds_smoke.name].num_items


def test_inputs_are_the_table_of_the_dataset(ds_smoke):
    """§23.4, §23.5 M0, §23.8: ``meta.json`` ``inputs`` of S0 and of every other command step equal the hashes of
    ``run.expected_inputs(dataset)``: the data files, the template and the SASRec pickle of this dataset."""
    report, name = ds_smoke.report, ds_smoke.name
    expected = {rel: sha for rel, (sha, _) in run.expected_inputs(name).items()}
    spec = SPECS[name]
    assert sorted(expected) == sorted([*spec.data_files, spec.template_file, spec.sasrec_source])
    check = report.check("S0.input_hashes")
    assert check.passed is True and check.measured == expected
    for sid in COMMAND_STEPS:
        assert _read_json(report.step_dir(sid) / "meta.json")["inputs"] == expected, sid


def test_metrics_meta_names_the_dataset(ds_smoke):
    """§23.5 M11: ``metrics.json`` ``meta.dataset`` is the name of the dataset in every evaluation folder of every step
    that evaluates (S1, S2, S5, S6, S7)."""
    for sid in EVAL_STEPS:
        folders = _eval_folders(ds_smoke.report.step_dir(sid))
        assert folders, sid
        for folder in folders:
            assert _read_json(folder / "metrics.json")["meta"]["dataset"] == ds_smoke.name, (sid, folder.name)


def test_run_sizes_follow_the_design(ds_smoke):
    """§23.6: micro-batch and step rows of S1 (2 epochs) and S5, S7 (1 epoch), window sizes, gate calls, cluster lookups
    (micro-batches x 4), the live z check, NPZ rows of S3 and fit rows of S4. The report's checks compare these with
    ``step_counts`` and ``load_dataset`` at run time; here they equal the values of the design."""
    design, check = ds_smoke.design, ds_smoke.report.check
    (long_run, short_run), nb = design.steps, design.micro
    gapless = {"tau_consecutive": True, "step_consecutive": True}
    assert check("S1.record_rows").measured == {"micro": 2 * nb, "step": long_run, "epoch": 2, **gapless}
    assert check("S5.record_rows").measured == {"micro": nb, "step": short_run, "epoch": 1, **gapless}
    assert check("S7.record_rows").measured == {"micro": nb, "step": short_run, "epoch": 1, **gapless}
    assert check("S1.window_sizes").measured == {"epoch0": design.windows, "epoch1": design.windows}
    assert check("S5.window_sizes").measured == check("S7.window_sizes").measured == {"epoch0": design.windows}
    assert check("S1.gate_calls").measured == {"steps": 2 * nb, "run_summary": 2 * nb}
    assert check("S5.gate_calls").measured == {"steps": 0, "run_summary": 0}
    assert check("S5.cluster_lookups").measured == {"steps": nb * 4, "run_summary": nb * 4}
    live = check("S5.live_check").measured
    assert (live["passed"], live["n"], live["device"]) == (True, design.live_check_n, "cpu")
    rows = {"train": [design.train, K], "validation": [SMOKE_EVAL, K], "test": [SMOKE_EVAL, K]}
    exported = check("S3.npz_rows").measured
    assert exported and all(value == rows for value in exported.values())
    assert check("S4.fit_train_only").measured["fit_rows"] == design.train


def test_s0_compares_the_sasrec_state_of_the_dataset(ds_smoke):
    """§17.6.6 S0, §23.2: the dataset's own converted state, digest of the design, bit for bit on CPU against its
    reference outputs; the training steps keep that digest (frozen_sasrec, 7.8 #13)."""
    report = ds_smoke.report
    assert report.check("S0.sasrec_reference").measured == {"device": "cpu", "max_abs_diff": 0.0, "bitwise": True,
                                                            "state_digest": ds_smoke.design.state_digest}
    assert ds_smoke.cfg.sasrec.state_digest == ds_smoke.design.state_digest
    for sid in ("S1", "S5", "S7"):
        assert report.check(f"{sid}.frozen_sasrec").passed is True, sid


def test_records_use_the_catalog_of_the_dataset(ds_smoke):
    """D28, §23.5 M2, M4, M11: every evaluation record of S1 is a sample of this smoke selection, its 20 candidates are
    ids below ``num_items`` with the target once, and the titles come from the dataset's own title file."""
    data, spec = ds_smoke.data, SPECS[ds_smoke.name]
    titles = data.catalog.titles
    samples = {s.sample_id: s for split in ("val", "test") for s in data.samples(split)}
    folders = _eval_folders(ds_smoke.report.step_dir("S1"))
    assert folders
    for folder in folders:
        rows = _read_jsonl(folder / "records.jsonl")
        assert len(rows) == SMOKE_EVAL
        for row in rows:
            sample = samples[row["sample_id"]]
            assert row["target_id"] == sample.target and row["target_title"] == titles[sample.target]
            assert len(row["cand_ids"]) == 20 and row["cand_ids"].count(sample.target) == 1
            assert all(0 <= item < spec.num_items for item in row["cand_ids"])
            assert row["cand_titles"] == [titles[item] for item in row["cand_ids"]]


def test_crosscheck_covers_padded_windows(ds_smoke):
    """D28: the evaluation windows of the selection include histories shorter than 10 (seed 1234: Steam 4 validation and
    4 test windows, LastFM 7 and 5), so padded histories go through the generation path (M5, M9) and the export path
    (M12) whose gates S3 compares (|d omega| <= records.gate_crosscheck_atol), and ``Sample.num_items`` is the dataset's
    padding id (3581, 4606, §23.4). A wrong padding id is refused even earlier, by the SASRec input check (M3)."""
    for split in ("val", "test"):
        assert any(len(s.history) < 10 for s in ds_smoke.data.samples(split)), split
    assert all(s.num_items == SPECS[ds_smoke.name].pad_id for s in ds_smoke.data.val)
    assert ds_smoke.report.check("S3.crosscheck").passed is True
