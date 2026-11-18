"""M12 gate 분석·그림과 export (DESIGN.md §16.8 #8-#11; test names of §17.6.9).

Known answers of ``gate_summary`` (uniform, SPEC Case B, one-hot, a two-point mixture and eta^2 by hand),
the summary files and the judgement-free interpretation template, the figures (format, 300 dpi, no
annotations, normal fonts, at most 500 heatmap rows, the Fig. 3c counterpart = first ten test rows) and the
``export`` command end to end on the smoke data. Synthetic inputs come from ``seeding.np_rng(0, "test",
<name>)`` (§17.6.2).

``run_export`` reads checkpoints only through M10 ``train.read_trainable_state`` and evaluation records
through M11 ``EvalRecord`` (§14.6.6, §15.6.1). Neither module exists at this stage, so the export tests build
a synthetic source run (``config.yaml``, ``eval/selection.json``, ``checkpoints/epoch{e}/trainable.pt``,
``eval/epoch{e}/{val,test}/`` written with M12 ``save_npz``/``write_eval``) and register stand-ins for
``ilora.train`` (a reader of that ``trainable.pt``) and, while evaluate.py is missing, ``ilora.evaluate``
for the duration of a test. The export path with the real M10 reader is M10's to test (cpu_smoke_run S3).
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
import torch

from ilora import analysis, records, seeding
from ilora.analysis import AnalysisError
from ilora.candidates import FLAG_NAMES
from ilora.records import RepresentationTable

K = 4
# matplotlib 3.7.1 calls pyparsing names that newer pyparsing releases mark deprecated (library internals)
pytestmark = pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class StandInEvalRecord:
    """The 3.1 ``EvalRecord`` fields in the M11 §15.6.1 order (used only while ilora/evaluate.py is missing)."""
    sample_id: str
    split: str
    user_id: int
    template_id: int
    hybrid: bool
    cand_ids: tuple
    cand_titles: tuple
    target_id: int
    target_title: str
    raw_text: str
    first_line: str
    matched: tuple
    valid: bool
    correct: bool
    gates: tuple
    flags: Mapping
    gen_token_ids: tuple


@pytest.fixture
def eval_record_cls(monkeypatch):
    if importlib.util.find_spec("ilora.evaluate") is not None:
        from ilora.evaluate import EvalRecord

        return EvalRecord
    module = types.ModuleType("ilora.evaluate")
    module.EvalRecord = StandInEvalRecord
    monkeypatch.setitem(sys.modules, "ilora.evaluate", module)
    return StandInEvalRecord


def softmax_rows(name: str, n: int, k: int = K, scale: float = 1.5) -> np.ndarray:
    rng = seeding.np_rng(0, "test", name)
    logits = rng.normal(size=(n, k)) * scale
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def table_of(split: str, omega: np.ndarray, *, d: int = 8, users: np.ndarray = None, name: str = "table",
             meta=None) -> RepresentationTable:
    n = omega.shape[0]
    rows = np.arange(n, dtype=np.int64)
    z = seeding.np_rng(0, "test", f"{name}_z").normal(size=(n, d)).astype(np.float32)
    return RepresentationTable(split=split, sample_ids=tuple(f"{split}:{r}" for r in rows.tolist()), rows=rows,
                               user_ids=np.asarray(users if users is not None else rows // 7, dtype=np.int64), z=z,
                               omega=omega, argmax=np.argmax(omega, axis=1).astype(np.int64),
                               meta=meta or {"routing_mode": "dynamic", "source": "export"})


def js_bits_by_hand(p: list[float], q: list[float]) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda x, y: sum(a * math.log(a / b) for a, b in zip(x, y) if a > 0)      # noqa: E731
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


# ----------------------------------------------------------------------------------------------- #8
@pytest.mark.criterion("M12-8")
def test_gate_summary_known_answers():
    """Uniform, Case B, one-hot, a two-point mixture (JS by hand, 1e-12) and eta^2 by hand."""
    n = 12
    uniform = analysis.gate_summary(np.full((n, K), 0.25, dtype=np.float32), user_ids=None)["metrics"]
    for stat in analysis.STATISTICS[2:] + ("mean",):
        assert abs(uniform["normalized_entropy"][stat] - 1.0) <= 1e-12
        assert abs(uniform["js_to_mean_bits"][stat]) <= 1e-15 and uniform["top1_minus_top2"][stat] == 0.0
    assert uniform["expert_std"]["per_expert"] == [0.0] * K and uniform["argmax_tie_rows"]["value"] == n
    assert uniform["argmax_count"]["per_expert"] == [n, 0, 0, 0]          # first index on ties

    case_b = analysis.gate_summary(np.tile(np.float32([0.90, 0.03, 0.03, 0.04]), (n, 1)), user_ids=None)["metrics"]
    assert round(case_b["normalized_entropy"]["mean"], 6) == 0.313045
    assert case_b["js_to_mean_bits"]["max"] <= 1e-15 and case_b["expert_std"]["per_expert"] == [0.0] * K
    assert case_b["argmax_count"]["per_expert"] == [n, 0, 0, 0] and case_b["argmax_ratio"]["per_expert"][0] == 1.0
    assert abs(case_b["top1_minus_top2"]["mean"] - 0.86) <= 1e-7               # float32 inputs

    one_hot = np.eye(K, dtype=np.float32)[[0, 1, 2, 3, 3, 1]]
    h = analysis.normalized_entropy(one_hot)
    assert h.tolist() == [0.0] * 6 and not np.isnan(h).any()

    a, b = [0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]
    mixture = np.asarray([a, a, a, b], dtype=np.float64)
    mean = [(3 * x + y) / 4 for x, y in zip(a, b)]
    js_a, js_b = js_bits_by_hand(a, mean), js_bits_by_hand(b, mean)
    got = analysis.gate_summary(mixture, user_ids=None)["metrics"]["js_to_mean_bits"]
    assert abs(got["mean"] - (3 * js_a + js_b) / 4) <= 1e-12 and abs(got["min"] - js_a) <= 1e-12
    assert abs(got["max"] - js_b) <= 1e-12 and abs(got["median"] - js_a) <= 1e-12
    assert abs(analysis.js_divergence_bits(np.asarray([a]), np.asarray(b))[0] - js_bits_by_hand(a, b)) <= 1e-12

    w = [0.2, 0.4, 0.6, 0.6, 0.9]
    users = np.asarray([5, 5, 9, 9, 9])
    eta = analysis.gate_summary(np.asarray([[x, 1 - x] for x in w]), user_ids=users)
    m = sum(w) / 5
    between = 2 * ((0.2 + 0.4) / 2 - m) ** 2 + 3 * ((0.6 + 0.6 + 0.9) / 3 - m) ** 2
    total = sum((x - m) ** 2 for x in w)
    assert all(abs(v - between / total) <= 1e-12 for v in eta["metrics"]["user_eta_squared"]["per_expert"])
    assert eta["user_view"]["n_users"] == 2 and eta["user_view"]["windows_per_user"] == {"min": 2, "median": 2.5, "max": 3}
    constant = analysis.gate_summary(np.asarray([[x, 1 - x, 0.0] for x in w]), user_ids=users)["user_view"]
    assert constant["eta_squared"][2] is None and "variance" in constant["eta_squared_null_reason"][2]
    for bad in (np.ones((5, 1), dtype=np.float32), np.ones((5, 1, K), dtype=np.float32)):
        with pytest.raises(AnalysisError, match="K >= 2|shape"):
            analysis.gate_summary(bad, user_ids=None)

# ----------------------------------------------------------------------------------------------- #9
# Judgement sentences that must not appear in gate_interpretation.txt (SPEC §5: no TRUE/FALSE collapse verdict
# from a threshold). The reference descriptions may name "Case B" and "collapse" themselves (§17.6.10).
JUDGEMENT_PATTERNS = (
    r"\b(?:TRUE|FALSE|True|False|true|false)\b",
    r"(?i)collapse\w*\W{0,20}(?:yes|no|true|false|detected|occurred|confirmed)",
    r"(?i)\b(?:is|was|are|were)\s+(?:not\s+)?collapsed\b",
    r"[<>≤≥]\s*=?\s*[0-9]",
    r"(?i)threshold",
    r"임계",
    r"붕괴(?:되었|됐|됨|했|하였|이다)",
    r"(?:판정|결론|진단)\s*[:：=]",
)


def _csv_value(text: str) -> object:
    if text == "":
        return None
    return int(text) if re.fullmatch(r"-?[0-9]+", text) else float(text)


@pytest.mark.criterion("M12-9")
def test_summary_files_and_interpretation(tmp_path):
    """gate_summary.json and .csv hold the same numbers; the interpretation has the reference values and no verdict."""
    omega = softmax_rows("m12_summary", 300)
    users = np.repeat(np.arange(30), 10)
    summary = {"split": "train", "checkpoint": "epoch2", "routing_mode": "dynamic",
               **analysis.gate_summary(omega, user_ids=users)}
    analysis.write_gate_summary(tmp_path, summary)
    analysis.write_interpretation(tmp_path, summary)
    data = json.loads((tmp_path / "gate_summary.json").read_text(encoding="utf-8"))
    assert data["split"] == "train" and data["n"] == 300 and data["num_experts"] == K
    assert set(data["metrics"]) >= {"mean_utilization", "argmax_count", "argmax_ratio", "argmax_tie_rows",
                                    "normalized_entropy", "expert_std", "js_to_mean_bits", "top1_minus_top2",
                                    "user_eta_squared"}
    assert set(data["definitions"]) >= {"normalized_entropy", "js_to_mean_bits", "expert_std"}
    with open(tmp_path / "gate_summary.csv", "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(analysis.SUMMARY_CSV_FIELDS)
    from_csv = [(m, e, s, _csv_value(v)) for m, e, s, v in rows[1:]]
    from_json = analysis.summary_rows(data)
    assert len(from_csv) == len(from_json) > 50
    for got, want in zip(from_csv, from_json):
        assert got[:3] == want[:3] and (got[3] == want[3] or (got[3] is None and want[3] is None)), (got, want)
    assert ("mean_utilization", "3", "value", data["metrics"]["mean_utilization"]["per_expert"][3]) in from_csv
    assert ("normalized_entropy", "", "median", data["metrics"]["normalized_entropy"]["median"]) in from_csv

    text = (tmp_path / "gate_interpretation.txt").read_text(encoding="utf-8")
    for value in ("0.3130", "0.8600", "1.0000", "0.9599", "0.9185", "0.9937", "0.0929", "0.0199",
                  f"{data['metrics']['normalized_entropy']['mean']:.4f}", f"{data['metrics']['js_to_mean_bits']['mean']:.4f}"):
        assert value in text, value
    for heading in ("1. 수치", "2. 참고점", "3. 지표 읽는 법", "4. 판정하지 않음. 해석은 사람이 작성한다."):
        assert heading in text
    for pattern in JUDGEMENT_PATTERNS:
        assert not re.search(pattern, text), pattern
    refs = data["reference_points"]
    assert round(refs["case_b_one_expert"]["normalized_entropy"], 6) == 0.313045
    assert refs["case_a_uniform"]["js_to_mean_bits"] == 0.0 and refs["case_a_uniform"]["argmax_tie_row_ratio"] == 1.0
    fig = refs["paper_fig3c"]
    assert [round(fig["normalized_entropy"][s], 4) for s in ("mean", "min", "max")] == [0.9599, 0.9185, 0.9937]
    assert [round(fig["top1_minus_top2"][s], 4) for s in ("mean", "min", "max")] == [0.0929, 0.004, 0.2773]
    assert round(fig["js_to_mean_bits"]["mean"], 4) == 0.0199
    assert "note" in analysis.reference_points(3)["paper_fig3c"]          # Fig. 3c has K = 4
    with pytest.raises(AnalysisError, match="never overwritten"):
        analysis.write_gate_summary(tmp_path, summary)


# ----------------------------------------------------------------------------------------------- #10
@pytest.fixture
def saved_figures(monkeypatch):
    """Every Figure saved through matplotlib, with the path (the figure is inspected after drawing)."""
    from matplotlib.figure import Figure

    saved = []
    original = Figure.savefig

    def spy(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        saved.append((Path(fname), self))
        return result

    monkeypatch.setattr(Figure, "savefig", spy)
    return saved


def _check_figure_text(fig) -> None:
    from matplotlib.text import Text

    for ax in fig.axes:
        assert len(ax.texts) == 0, "annotation inside the axes"
    for text in fig.findobj(match=Text):
        assert text.get_fontstyle() == "normal" and text.get_fontweight() in ("normal", 400), text
        assert text.get_text().isascii(), text.get_text()
        assert text.get_fontname() == "DejaVu Sans"


def _png_dpi(path: Path) -> tuple[float, float]:
    from PIL import Image

    with Image.open(path) as image:
        return image.info["dpi"]


@pytest.mark.criterion("M12-10")
def test_figures(tmp_path, saved_figures):
    """Formats, 300 dpi PNG, no annotations, normal fonts, train heatmap <= 500 rows, Fig. 3c bars = test:0..test:9."""
    train = table_of("train", softmax_rows("m12_fig_train", 1203), users=np.arange(1203) // 9, name="fig_train")
    test = table_of("test", softmax_rows("m12_fig_test", 95), name="fig_test")
    analysis.run_gate_analysis(train, tmp_path / "train", ckpt_tag="epoch0", formats=("png", "pdf"))
    analysis.run_gate_analysis(test, tmp_path / "test", ckpt_tag="epoch0", formats=("png",))
    names = ("gate_heatmap", "expert_mean_utilization", "argmax_expert_distribution", "entropy_distribution")
    for name in names:
        for fmt in ("png", "pdf"):
            assert (tmp_path / "train" / "figures" / f"{name}.{fmt}").is_file()
        assert (tmp_path / "test" / "figures" / f"{name}.png").is_file()
        assert not (tmp_path / "test" / "figures" / f"{name}.pdf").exists()
    assert not (tmp_path / "train" / "figures" / "gate_ten_sequences.png").exists()      # test split only
    pngs = sorted((tmp_path).rglob("*.png"))
    assert len(pngs) == 9
    for path in pngs:
        assert [round(v) for v in _png_dpi(path)] == [300, 300], path
    by_name = {}
    for path, fig in saved_figures:
        _check_figure_text(fig)
        by_name[(path.parent.parent.name, path.stem)] = fig
    heatmap = by_name[("train", "gate_heatmap")]
    image = heatmap.axes[0].images[0]
    assert image.get_array().shape == (500, K) and image.get_clim() == (0.0, 1.0)
    assert heatmap.axes[0].get_yticks().size == 0
    assert [t.get_text() for t in heatmap.axes[0].get_xticklabels()] == [f"Expert {k}" for k in range(1, K + 1)]
    assert by_name[("test", "gate_heatmap")].axes[0].images[0].get_array().shape == (95, K)
    ten = by_name[("test", "gate_ten_sequences")].axes[0]
    assert [t.get_text() for t in ten.get_yticklabels()] == [f"test:{i}" for i in range(10)]
    assert len(ten.patches) == 10 * K and ten.get_xlim() == (0.0, 1.0)
    assert [t.get_text() for t in ten.get_legend().get_texts()] == [f"Expert {k}" for k in range(1, K + 1)]
    with open(tmp_path / "test" / "gate_ten_sequences.csv", "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert [r[0] for r in rows[1:]] == [f"test:{i}" for i in range(10)]
    values = np.asarray([[float(x) for x in r[2:]] for r in rows[1:]]).astype(np.float32)
    assert np.array_equal(values, test.omega[:10])
    mean_ax = by_name[("test", "expert_mean_utilization")].axes[0]
    assert [t.get_text() for t in mean_ax.get_legend().get_texts()] == ["1/K"]
    # M15 figures: PCA for pictures only, cluster scatter and the cluster x expert heatmap
    coords = analysis.pca_2d(seeding.np_rng(0, "test", "m12_pca").normal(size=(200, 8)).astype(np.float32))
    assert coords.shape == (200, 2)
    labels = np.arange(200) % 4
    analysis.plot_pca_clusters(tmp_path / "m15" / "sasrec_pca_clusters", coords, labels, n_clusters=4,
                               title="SASRec representations (PCA)")
    analysis.plot_cluster_gate_heatmap(tmp_path / "m15" / "cluster_dynamic_gate_heatmap",
                                       softmax_rows("m12_cluster_means", 4), title="Mean dynamic gate per cluster")
    pca = saved_figures[-2][1].axes[0]
    assert [t.get_text() for t in pca.get_legend().get_texts()] == [f"Cluster {c}" for c in range(4)]
    assert len(pca.collections) == 1                        # one scatter call in row order
    cluster = saved_figures[-1][1].axes[0]
    assert cluster.images[0].get_clim() == (0.0, 1.0)
    assert [t.get_text() for t in cluster.get_yticklabels()] == [f"Cluster {c}" for c in range(4)]
    for _, fig in saved_figures[-2:]:
        _check_figure_text(fig)
    with pytest.raises(AnalysisError, match="ASCII"):
        analysis.plot_entropy_hist(tmp_path / "x", np.zeros(3), title="엔트로피")

# ----------------------------------------------------------------------------------------------- #11, export
def _records_for(cls, table: RepresentationTable) -> list:
    rng = seeding.np_rng(0, "test", f"m12_export_records_{table.split}")
    out = []
    for i, sid in enumerate(table.sample_ids):
        cands = tuple(int(c) for c in rng.choice(1682, size=20, replace=False))
        target = cands[i % 20]
        valid, correct = i % 4 != 3, i % 4 == 0
        matched = (target,) if correct else (cands[(i + 1) % 20],) if valid else ()
        out.append(cls(sample_id=sid, split=table.split, user_id=int(table.user_ids[i]), template_id=i % 3,
                       hybrid=True, cand_ids=cands, cand_titles=tuple(f"Movie {c}" for c in cands), target_id=target,
                       target_title=f"Movie {target}", raw_text=f"Movie {target}\n", first_line=f"Movie {target}",
                       matched=matched, valid=valid, correct=correct,
                       gates=tuple(float(g) for g in table.omega[i].tolist()),
                       flags={flag: False for flag in FLAG_NAMES}, gen_token_ids=(29871, 2)))
    return out


def _metrics(split: str, recs: list, epoch: int) -> dict:
    def block(rs: list) -> dict:
        n, v, c = len(rs), sum(r.valid for r in rs), sum(r.correct for r in rs)
        return {"split": split, "n_total": n, "n_valid": v, "n_correct": c, "valid_ratio": v / n,
                "hr_at_1": c / v if v else 0.0, "combined_metric": c / n}

    return {**block(recs), "epoch": epoch, "ckpt_tag": f"epoch{epoch}",
            "aux_protocols": {} if split == "val" else {f"drop_last_bs{b}": block(recs[:(len(recs) // b) * b])
                                                         for b in records.AUX_BATCH_SIZES},
            "supplementary": {}, "meta": {"seed": 1234, "dataset": "movielens"}}


def make_source_run(root: Path, cfg, data, encoder, cls, monkeypatch, *, epochs: int, best: int, last: int,
                    swap_top2: tuple = None) -> Path:
    """A finished training run as export reads it (§16.6.5): config.yaml, eval/selection.json,
    checkpoints/epoch{e}/trainable.pt + COMPLETE and eval/epoch{e}/{val,test}/ of every epoch (validation) and of
    best/last (test), whose omega comes from the checkpoint gate and whose z differs from the CPU z by about 1e-6
    (as a GPU evaluation would). ``swap_top2 = (split, i)`` swaps the two largest omega entries of sample i in that
    split's evaluation files (NPZ and records.jsonl alike): a real gate mismatch with an unchanged z. ``ilora.train``
    becomes a stand-in reader of that trainable.pt."""
    from ilora.config import dump_config
    from ilora.router import Router

    source = root / "source_run"
    (source / "eval").mkdir(parents=True)
    dump_config(cfg, source / "config.yaml")
    (source / "eval" / "selection.json").write_text(json.dumps(
        {"rule": "largest validation n_correct, later epoch on ties", "val": {}, "best_val_epoch": best,
         "last_epoch": last, "test_dirs": {"best_val": f"eval/epoch{best}/test", "last": f"eval/epoch{last}/test"},
         "primary": "best_val"}), encoding="utf-8")
    for epoch in range(epochs):
        with seeding.torch_seed_scope(0, "test", f"m12_export_ckpt_{epoch}"):
            state = {"projector.mlp_proj.0.weight": torch.randn(3, 3), "generated_frozen.embed_rows": torch.randn(5, 2)}
            if cfg.router.routing_mode == "dynamic":
                state["omega_source.gate.proj.weight"] = torch.randn(cfg.lora.num_experts, encoder.hidden_size)
        ckpt = source / "checkpoints" / f"epoch{epoch}"
        ckpt.mkdir(parents=True)
        torch.save(state, ckpt / "trainable.pt")
        (ckpt / "COMPLETE").write_text("{}", encoding="utf-8")
        router = Router(cfg.router, num_experts=cfg.lora.num_experts, z_dim=encoder.hidden_size, main_seed=cfg.run.seed,
                        hard_gate=None)
        router.load_state_dict({k[len("omega_source."):]: v for k, v in state.items() if k.startswith("omega_source.")})
        for split in ("val", "test") if epoch in (best, last) else ("val",):
            table = records.compute_representations(data.samples(split), encoder, router,
                                                    num_experts=cfg.lora.num_experts, batch_size=8, num_threads=1,
                                                    rowsum_atol=1e-5)
            noise = seeding.np_rng(0, "test", f"m12_gpu_z_{epoch}_{split}").normal(size=table.z.shape) * 1e-6
            table = dataclasses.replace(table, z=(table.z + noise).astype(np.float32), meta={"source": "evaluate"})
            if swap_top2 is not None and swap_top2[0] == split:
                omega = table.omega.copy()
                row = omega[swap_top2[1]]
                top, second = np.argsort(row)[-1], np.argsort(row)[-2]
                row[[top, second]] = row[[second, top]]
                table = dataclasses.replace(table, omega=omega, argmax=np.argmax(omega, axis=1).astype(np.int64))
            out = source / "eval" / f"epoch{epoch}" / split
            out.mkdir(parents=True)
            records.save_npz(out / "representations.npz", table)
            recs = _records_for(cls, table)
            records.write_eval(out, recs, _metrics(split, recs, epoch), expected_ids=list(table.sample_ids),
                               num_experts=cfg.lora.num_experts)

    def read_trainable_state(ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        assert (ckpt_dir / "COMPLETE").is_file(), ckpt_dir
        return torch.load(ckpt_dir / "trainable.pt")

    module = types.ModuleType("ilora.train")
    module.read_trainable_state = read_trainable_state
    monkeypatch.setitem(sys.modules, "ilora.train", module)
    return source


@pytest.fixture
def export_setup(profile_cfg, smoke_data, sasrec_state_path, eval_record_cls, monkeypatch, tmp_path, torch_threads1):
    """Factory ``export_setup(*sets, epochs, best, last, swap_top2=None) -> (export cfg, source run, export run
    folder)``."""
    from ilora import run
    from ilora.data import HISTORY_LEN
    from ilora.sasrec import SASRecEncoder

    def make(*sets: str, epochs: int, best: int, last: int, swap_top2: tuple = None):
        cfg = profile_cfg("smoke_cpu", *sets)
        encoder = SASRecEncoder.from_config(cfg, smoke_data.catalog, HISTORY_LEN)
        source = make_source_run(tmp_path, cfg, smoke_data, encoder, eval_record_cls, monkeypatch, epochs=epochs,
                                 best=best, last=last, swap_top2=swap_top2)
        export_cfg = profile_cfg("smoke_cpu", *sets, f"records.source_run={json.dumps(source.as_posix())}")
        return export_cfg, source, run.create_run_dir("export", root=tmp_path / "runs")

    return make


@pytest.mark.criterion("M12-11")
def test_k1_export_and_analysis_error(export_setup, repo_root):
    """K = 1 (routing none): export succeeds with gate_0 = 1 and no analysis folder; the analysis is an explicit error."""
    cfg, source, run_dir = export_setup("lora.num_experts=1", "router.routing_mode=none", epochs=1, best=0, last=0)
    records.run_export(cfg, run_dir, repo_root=repo_root)
    out = run_dir / "records" / "export" / "epoch0"
    for split, n in (("train", 156), ("validation", 10), ("test", 10)):
        table = records.load_npz(out / f"representations_{split}.npz")
        assert len(table) == n and table.omega.shape == (n, 1) and bool((table.omega == 1.0).all())
        assert table.meta["routing_mode"] == "none" and table.meta["source"] == "export"
    with open(out / "gates_test.csv", "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][:5] == ["sample_id", "user_id", "split", "gate_0", "argmax_expert"]
    assert {r[3] for r in rows[1:]} == {"1"} and {r[4] for r in rows[1:]} == {"0"}
    cross = json.loads((out / "crosscheck.json").read_text(encoding="utf-8"))
    assert all(cross["splits"][s]["passed"] for s in ("validation", "test"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis"] == "not run: gate analysis requires K >= 2" and manifest["analysis_files"] == {}
    assert not (run_dir / "records" / "analysis").exists()
    table = records.load_npz(out / "representations_test.npz")
    with pytest.raises(AnalysisError, match="gate analysis requires K >= 2"):
        analysis.run_gate_analysis(table, run_dir / "records" / "analysis" / "epoch0" / "test", ckpt_tag="epoch0")
    with pytest.raises(AnalysisError, match="gate analysis requires K >= 2"):
        analysis.gate_summary(table.omega, user_ids=None)


@pytest.mark.criterion("M12-7")
def test_failed_crosscheck_is_written(export_setup, repo_root):
    """16.6.5 step 7: one test sample whose evaluation gate differs (its two largest omega entries swapped, z as
    before) stops the export with a RecordsError naming max |d omega| with its worst sample, max |d z| and the argmax
    mismatch count, after crosscheck.json was written with passed true for validation and passed false with those
    numbers for test; manifest.json and the analysis are not written."""
    from ilora import run

    cfg, source, run_dir = export_setup(epochs=1, best=0, last=0, swap_top2=("test", 2))
    with pytest.raises(records.RecordsError) as info:
        records.run_export(cfg, run_dir, repo_root=repo_root)
    message = str(info.value)
    out = run_dir / "records" / "export" / "epoch0"
    cross = json.loads((out / "crosscheck.json").read_text(encoding="utf-8"))
    assert cross["atol"] == 1e-5 and sorted(cross["splits"]) == ["test", "validation"]
    val, test = cross["splits"]["validation"], cross["splits"]["test"]
    assert val["passed"] is True and val["max_abs_diff_omega"] <= 1e-5 and val["argmax_mismatches"] == 0
    sample = records.load_npz(source / "eval" / "epoch0" / "test" / "representations.npz").sample_ids[2]
    assert test["passed"] is False and test["max_abs_diff_omega"] > 1e-5
    assert test["max_abs_diff_omega_sample_id"] == sample and test["argmax_mismatch_sample_ids"] == [sample]
    assert test["argmax_mismatches"] == 1 and test["n"] == 10 and 0.0 < test["max_abs_diff_z"] < 1e-4
    assert test["records_sha256"] == run.content_sha256(source / "eval" / "epoch0" / "test" / "records.jsonl")
    assert f"{out / 'crosscheck.json'} written with passed false" in message
    worst = f"max |export omega - eval omega| = {test['max_abs_diff_omega']!r} > 1e-05 (worst sample {sample})"
    assert worst in message
    assert f"max |export z - eval z| = {test['max_abs_diff_z']!r}, argmax mismatches 1 of 10" in message
    assert not (out / "manifest.json").exists() and not (run_dir / "records" / "analysis").exists()


@pytest.mark.slow
@pytest.mark.criterion("M12-7")
def test_export_end_to_end(export_setup, repo_root):
    """Dynamic export of best != last: NPZ/CSV per split, crosscheck.json (|d omega| 0, |d z| recorded), analysis of
    the three splits with figures, manifest sha256; a differing config section is refused."""
    from ilora import run

    cfg, source, run_dir = export_setup(epochs=3, best=1, last=2)
    records.run_export(cfg, run_dir, repo_root=repo_root)
    for epoch in (1, 2):
        out = run_dir / "records" / "export" / f"epoch{epoch}"
        tables = {s: records.load_npz(out / f"representations_{s}.npz") for s in ("train", "validation", "test")}
        assert [len(t) for t in tables.values()] == [156, 10, 10]
        assert tables["validation"].split == "val" and tables["train"].dim == 64 and tables["train"].num_experts == K
        for split, name in (("val", "validation"), ("test", "test")):
            evaluated = records.load_npz(source / "eval" / f"epoch{epoch}" / split / "representations.npz")
            # evaluation batches of 8 vs export batches of 1,024: z and omega agree to rounding (16.6.5 step 7)
            assert np.abs(tables[name].omega.astype(np.float64) - evaluated.omega).max() <= 1e-5
            csv_table = records.read_gate_csv(out / f"gates_{name}.csv")
            assert np.array_equal(csv_table.omega, tables[name].omega) and np.array_equal(csv_table.z, tables[name].z)
        assert not (out / "gates_train.csv").exists()
        cross = json.loads((out / "crosscheck.json").read_text(encoding="utf-8"))
        for name in ("validation", "test"):
            entry = cross["splits"][name]
            assert entry["passed"] and entry["max_abs_diff_omega"] <= 1e-5 and entry["argmax_mismatches"] == 0
            assert 0.0 < entry["max_abs_diff_z"] < 1e-4                  # recorded, not judged
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["checkpoint"]["tags"] == (["best_val"] if epoch == 1 else ["last"])
        assert manifest["checkpoint"]["trainable_sha256"] == run.content_sha256(
            source / "checkpoints" / f"epoch{epoch}" / "trainable.pt")
        for name, sha in manifest["files"].items():
            assert run.content_sha256(out / name) == sha, name
        for rel, sha in manifest["analysis_files"].items():
            assert run.content_sha256(run_dir / rel) == sha, rel
        for split in ("test", "val", "train"):
            folder = run_dir / "records" / "analysis" / f"epoch{epoch}" / split
            summary = json.loads((folder / "gate_summary.json").read_text(encoding="utf-8"))
            assert summary["checkpoint"] == f"epoch{epoch}" and summary["routing_mode"] == "dynamic"
            assert ("user_view" in summary) == (split == "train")
            assert (folder / "gate_interpretation.txt").is_file() and (folder / "figures" / "gate_heatmap.png").is_file()
            assert (folder / "figures" / "gate_ten_sequences.png").is_file() == (split == "test")
    assert not (run_dir / "records" / "export" / "epoch0").exists()
    with pytest.raises(records.RecordsError, match="never overwritten"):
        records.run_export(cfg, run_dir, repo_root=repo_root)
    # a different lora/router section than the training run is refused before anything is computed
    other = dataclasses.replace(cfg, lora=dataclasses.replace(cfg.lora, dropout=0.0))
    with pytest.raises(records.RecordsError, match="lora.dropout"):
        records.run_export(other, run_dir, repo_root=repo_root)
