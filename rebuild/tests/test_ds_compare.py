"""M14 비교 도구의 데이터셋 선택 (DESIGN.md §23.5 M14, §23.8, D41): ``rebuild/tools/compare_runs.py`` on Steam and LastFM.

The tool is loaded from its file. Synthetic run folders are written by the writers of the commands, as in
tests/test_m14_compare.py (test modules cannot import each other, test_m13_coverage ``test_no_repo_imports``, so its
small makers are repeated here), from the ``<d>_smoke_cpu`` profiles, whose ``prompt.template_path`` selects the
dataset (D27). Table D, its notes, the protocol line of the definitions and the limitation items with N or title
rules are compared as exact output text, MovieLens included (the comparison of the four whole output files with the
54e5ae0 tool is the measurement multi/rebuild_parametrize/compare_regress.py, §23.7 (f)); the paper numbers and integer
solutions are recomputed from the printed values. Synthetic inputs come from ``seeding.np_rng(0, "test", <name>)``
(§17.6.2).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pytest
import yaml

from ilora import datasets, run, seeding
from ilora.config import config_hash, dump_config, load_config

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
TOOL_PATH = REBUILD_DIR / "tools" / "compare_runs.py"
PROFILES = {"movielens": CONFIG_DIR / "smoke_cpu.yaml", "steam": CONFIG_DIR / "datasets" / "steam_smoke_cpu.yaml",
            "lastfm": CONFIG_DIR / "datasets" / "lastfm_smoke_cpu.yaml"}                    # §23.6
N_TEST = 12                                                  # synthetic test split: drop_last 12 (bs4) and 8 (bs8)
K1 = ("lora.num_experts=1", "router.routing_mode=none")      # E1 overrides of §18.6.4
TRAINABLE = {"dynamic": {"projector": 8320, "gate": 256, "lora": 17408, "total": 25984},   # tiny model (M10 notes)
             "other": {"projector": 8320, "lora": 17408, "total": 25728}}
TRAIN_SECONDS = {"train": 10.5, "validation": 2.25, "save": 0.5, "test": 1.25, "total": 14.5}   # §14.6.7
# E1 / E2 of the synthetic scenario: best_val E2 epoch 1 (5 correct of 9 valid), E1 epoch 0 (3 correct of 5 valid).
E1_CELL = f"valid_ratio {5 / 12!r} (5/12), combined_metric {3 / 12!r} (3/12)"
E2_BS4_CELL = f"valid_ratio {9 / 12!r} (9/12), hr_at_1 {5 / 9!r} (5/9)"
HEADER = ("| Source | Row | ValidRatio | HitRatio@1 | Integer back-solution (N, valid, correct) | HitRatio@1 "
          "denominator | Our counterpart (from the run files) |")
RULE = "|---|---|---|---|---|---|---|"
COPIED = "the LLaRA Table 2 values, copied (not re-run) by the iLoRA paper"
THRESHOLD = ("- No reproduction threshold is set: one seed (D2), one test sample = 1/{n}, GPU non-determinism of "
             "training (D16) and the different run conditions of the reference values (limitations below). Differences "
             "are given in samples; Wilson intervals stay in the supplementary block of each metrics.json.")
INTRO_LLARA_ONLY = ("Paper numbers appear only in this table. The LLaRA row is set against our values of the same "
                    "protocol: E1 full protocol (CombinedMetric = correct / total). The iLoRA row has no counterpart "
                    "because its protocol cannot be identified (D41); our own values are in tables A and C.")


def table_d_lines(dataset: str) -> list[str]:
    """The exact rendered table D (§23.5 M14, D41): rows, ``counts_text`` cells and notes (the E1/E2 cells of the
    scenario below)."""
    head = ["## Table D. Paper reference values (not our results)", ""]
    if dataset == "steam":
        return [*head, INTRO_LLARA_ONLY, "", HEADER, RULE,
                "| [LLaRA Table 2 p.8] | LLaRA (SASRec), Steam | 0.9975 | 0.4949 | (1186, 1183, 587) | correct / total "
                f"| E1 best_val full: {E1_CELL}; sample differences NOT_RUN (our N {N_TEST} != 1186) |",
                "| [iLoRA paper p.7 Table 1] | iLoRA, Steam | 0.9981 | 0.5264 | no integer solution at N = 1184, 1186, "
                "1192 or 1194 (D41) | - | - |",
                f"| [iLoRA paper p.7 Table 1] | LLaRA, Steam | 0.9975 | 0.4949 | {COPIED} | - | - |", "",
                "- Back-solution of the LLaRA row: at the N the released code can produce (1184, 1186, 1192 and 1194, "
                "D41) two integer triples round to both four-digit values, (1186, 1183, 587) and (1184, 1181, 586), "
                "both correct / total. N = 1186 is used: the released LLaRA test loader has no drop_last and "
                "ValidRatio is over all sequences [LLaRA §5.1.3 p.7]; N = 1184 is the row count after drop_last with "
                "batch 4 or 8.",
                "- The paper's Imp.% is printed as 6.36% and computes to 6.36% = 0.5264 / 0.4949 - 1. The N and "
                "denominator of the iLoRA row are not identified, so this ratio may mix two protocols; our runs are "
                "compared only within one protocol.",
                THRESHOLD.format(n=1186)]
    if dataset == "lastfm":
        return [*head, INTRO_LLARA_ONLY, "", HEADER, RULE,
                "| [LLaRA Table 2 p.8] | LLaRA (SASRec), LastFM | 1.0000 | 0.4508 | (122, 122, 55) | correct / total "
                f"| E1 best_val full: {E1_CELL}; sample differences NOT_RUN (our N {N_TEST} != 122) |",
                "| [iLoRA paper p.7 Table 1] | iLoRA, LastFM | 1.0000 | 0.5000 | (122, 122, 61) or (120, 120, 60): "
                "N not identifiable (D41) | - | - |",
                f"| [iLoRA paper p.7 Table 1] | LLaRA, LastFM | 1.0000 | 0.4508 | {COPIED} | - | - |", "",
                "- Back-solution of the LLaRA row: (122, 122, 55) is the only integer triple at the N of the released "
                "code (122 and the drop_last 120) that rounds to both four-digit values; valid = total, so "
                "correct / valid and correct / total are the same number (D41).",
                "- The paper's Imp.% is printed as 8.51% but computes to 10.91% = 0.5000 / 0.4508 - 1 (the other Imp.% "
                "cells of the table agree). The N of the iLoRA row is not identifiable, so this ratio may mix two "
                "protocols; our runs are compared only within one protocol.",
                THRESHOLD.format(n=122)]
    return [*head, "Paper numbers appear only in this table. Each paper row is set against our values of the same "
            "protocol: E1 full protocol (CombinedMetric = correct / total) against LLaRA, E2 drop_last_bs4 (HR@1 = "
            "correct / valid) against iLoRA.", "", HEADER, RULE,
            "| [LLaRA Table 2 p.8] | LLaRA (SASRec), MovieLens | 0.9684 | 0.4421 | (95, 92, 42) | correct / total | "
            f"E1 best_val full: {E1_CELL}; sample differences NOT_RUN (our N {N_TEST} != 95) |",
            "| [iLoRA paper p.7 Table 1] | iLoRA, MovieLens | 0.9891 | 0.5275 | (92, 91, 48) | correct / valid | "
            f"E2 best_val drop_last_bs4: {E2_BS4_CELL}; sample differences NOT_RUN (our N {N_TEST} != 92) |",
            f"| [iLoRA paper p.7 Table 1] | LLaRA, MovieLens | 0.9684 | 0.4421 | {COPIED} | - | - |", "",
            "- Back-solution: the only integer triple with N in 80-95 that rounds to both four-digit values (D17). "
            "The iLoRA row has N = 92 (batch 4 with drop_last), the LLaRA row N = 95.",
            "- The paper's Imp.% 19.32% = 0.5275 / 0.4421 - 1 divides two numbers with different denominators and N; "
            "our runs are compared only within one protocol.",
            THRESHOLD.format(n=95)]


# §23.2 (seed 1234, target_unscorable: multi/datasets/s5_flags.py), §6.6.6 form, D38: the limitation items of a dataset.
ITEMS = {
    "steam": ("3. One seed, N = 1186: D2, seed 1234 only. One test sample is 1/1186, about 0.0008; the spread of "
              "training re-runs (D16) is not measured (NOT_RUN).",
              "4. Title judgement defect (D38): with the fixed candidates, 13 validation and 5 test samples cannot be "
              "judged correct even with the right title, so CombinedMetric is at most 1171/1184 and 1181/1186 (M2 "
              "6.6.6, DESIGN 23.2). Flag-free metrics are supplementary only.",
              "5. Protocols of the paper table: the LLaRA row uses N = 1186 and correct / total (D41); the N and "
              "denominator of the iLoRA row are not identified (no integer solution at the N of the released code), "
              "so our runs have no same-protocol value for it; the LLaRA row was copied from the LLaRA paper, not "
              "re-run by iLoRA [paper p.6 §4]."),
    "lastfm": ("3. One seed, N = 122: D2, seed 1234 only. One test sample is 1/122, about 0.0082; the spread of "
               "training re-runs (D16) is not measured (NOT_RUN).",
               "4. Title judgement defect (D38): with the fixed candidates, 0 validation and 0 test samples are "
               "unscorable, so CombinedMetric can reach 122/122 on both splits (M2 6.6.6, DESIGN 23.2). Flag-free "
               "metrics are supplementary only.",
               "5. Protocols of the paper table: the LLaRA row uses N = 122 and correct / total (D41; valid = total, "
               "so correct / valid is the same number); the N of the iLoRA row is not identifiable (two integer "
               "solutions, table D), so our runs have no same-protocol value for it; the LLaRA row was copied from "
               "the LLaRA paper, not re-run by iLoRA [paper p.6 §4]."),
    # the MovieLens items of the 54e5ae0 tool, byte for byte (18.6.10 #3-#5)
    "movielens": ("3. One seed, small N: D2, seed 1234 only. One test sample is 1/95, about 0.0105; the spread of "
                  "training re-runs (D16) is not measured (NOT_RUN).",
                  "4. Title judgement defect (D20): with the fixed candidates, 1 validation and 2 test samples cannot "
                  "be judged correct even with the right title, so CombinedMetric is at most 93/94 and 93/95 (M2 "
                  "6.6.6). Flag-free metrics are supplementary only.",
                  "5. Protocols of the paper table: the iLoRA row uses N = 92 and correct / valid, the LLaRA row N = "
                  "95 and correct / total (D17); the LLaRA row was copied from the LLaRA paper, not re-run by iLoRA "
                  "[paper p.6 §4]."),
}
# The paper numbers and back-solutions of a dataset: only comparison.md table D may contain them (§18.6.8).
PAPER_TOKENS = {
    "steam": ("0.9975", "0.4949", "0.9981", "0.5264", "6.36%", "(1186, 1183, 587)", "(1184, 1181, 586)"),
    "lastfm": ("1.0000", "0.4508", "0.5000", "8.51%", "10.91%", "(122, 122, 55)", "(122, 122, 61)", "(120, 120, 60)"),
}


@pytest.fixture(scope="module")
def tool() -> Any:
    """``rebuild/tools/compare_runs.py`` imported from its file (registered so its dataclasses resolve)."""
    spec = importlib.util.spec_from_file_location("compare_runs_ds_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


# ----------------------------------------------------------------------------
# Synthetic run folders in the formats the commands write (tests/test_m14_compare.py makers)
# ----------------------------------------------------------------------------
def _profile(dataset: str, *sets: str) -> Any:
    cfg, _ = load_config(PROFILES[dataset], list(sets))
    return cfg


def _start(root: Path, name: str, cfg: Any, command: str, **extra: Any) -> Path:
    run_dir = run.create_run_dir(name, root=root)
    dump_config(cfg, run_dir / "config.yaml")
    run.start_meta(run_dir, profile=name, argv=["synthetic"], config_hash=config_hash(cfg), overrides={},
                   seed=cfg.run.seed, extra={"command": command, **extra})
    return run_dir


def _finish(run_dir: Path) -> Path:
    run.finish_meta(run_dir, "completed")
    return run_dir


def make_records(correct: Iterable[int], valid: Iterable[int], *, num_experts: int) -> list:
    """``N_TEST`` M11 ``EvalRecord`` of samples ``test:0..`` in the M12 form (test_m14_compare ``make_records``)."""
    from ilora.candidates import FLAG_NAMES
    from ilora.evaluate import EvalRecord

    correct, valid = set(correct), set(valid) | set(correct)
    rng = seeding.np_rng(0, "test", "ds_compare_gates")
    out = []
    for i in range(N_TEST):
        cands = tuple(range(20 * i, 20 * i + 20))
        titles = tuple(f"Title {c}" for c in cands)
        if num_experts == 1:
            gates = (1.0,)
        else:
            w = rng.random(num_experts) + 0.05
            gates = tuple(float(x) for x in w / w.sum())
        line = titles[0] if i in correct else titles[1] if i in valid else "no title"
        matched = (cands[0],) if i in correct else (cands[1],) if i in valid else ()
        out.append(EvalRecord(sample_id=f"test:{i}", split="test", user_id=900 + i, template_id=i % 3, hybrid=True,
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


def make_train_run(root: Path, name: str, dataset: str, *, sets: Sequence[str] = (), best: int, last: int,
                   correct: Mapping[int, Iterable[int]], valid: Mapping[int, Iterable[int]]) -> Path:
    """A training run folder of ``dataset``: ``config.yaml``, ``meta.json`` (command train), ``eval/selection.json``
    (M10), ``eval/epoch{e}/test/`` of best_val and last, ``records/train/run_summary.json`` (M12 recorder)."""
    from ilora.evaluate import SplitMetrics
    from ilora.records import TrainRecorder
    from ilora.train import selection_document, write_marker

    cfg = _profile(dataset, *sets)
    k = cfg.lora.num_experts
    run_dir = _start(root, name, cfg, "train")
    for epoch in sorted({best, last}):
        write_test_eval(run_dir, epoch, make_records(correct[epoch], valid[epoch], num_experts=k), k)
    val = {e: SplitMetrics(split="val", n_total=10, n_valid=8, n_correct=5 if e == best else 1, valid_ratio=0.8,
                           hr_at_1=(5 if e == best else 1) / 8, combined_metric=(5 if e == best else 1) / 10)
           for e in range(last + 1)}
    write_marker(run_dir / "eval" / "selection.json", selection_document(val, best, last))
    mode = cfg.router.routing_mode
    recorder = TrainRecorder(run_dir, cfg.records, num_experts=k, total_micro_batches=1, total_steps=1)
    recorder.close("completed", {"routing_mode": mode,
                                 "trainable_params": TRAINABLE["dynamic" if mode == "dynamic" else "other"],
                                 "seconds": dict(TRAIN_SECONDS)})
    return _finish(run_dir)


def make_aux_run(root: Path, name: str, dataset: str, command: str, **extra: Any) -> Path:
    """An export or cluster run the tool accepts (status completed, the command of its role, ``config.yaml``); its
    files are absent, so its tables are ``NOT_RUN`` and only its dataset matters here."""
    return _finish(_start(root, name, _profile(dataset), command, **extra))


def scenario_runs(dataset: str, root: Path) -> dict[str, Path]:
    """E2 (dynamic, best 1 of 3 epochs) and E1 (K = 1, epoch 0) of ``dataset``: E2's best_val test has 9 valid and 5
    correct of 12 (E2_BS4_CELL: the first 12 samples, floor(12/4)*4), E1's 5 valid and 3 correct of 12 (E1_CELL)."""
    e2 = make_train_run(root, "e2", dataset, best=1, last=2, correct={1: range(5), 2: range(3)},
                        valid={1: range(9), 2: range(7)})
    e1 = make_train_run(root, "e1", dataset, sets=K1, best=0, last=0, correct={0: (0, 1, 5)},
                        valid={0: (0, 1, 2, 5, 6)})
    return {"E1": e1, "E2": e2}


def read_outputs(out: Path) -> SimpleNamespace:
    """The four output files of a comparison and its ``meta.json``."""
    return SimpleNamespace(md=(out / "comparison.md").read_text(encoding="utf-8"),
                           summary=(out / "summary.md").read_text(encoding="utf-8"),
                           csv=(out / "comparison.csv").read_text(encoding="utf-8"),
                           index=(out / "spec20_index.json").read_text(encoding="utf-8"),
                           meta=json.loads((out / "meta.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module", params=("steam", "lastfm"))
def compared(request, tool, tmp_path_factory) -> SimpleNamespace:
    """A comparison of the synthetic E1 and E2 runs of the dataset (its profile selects it, D27): name, folder and
    outputs."""
    root = tmp_path_factory.mktemp(f"{request.param}_compare")
    runs = scenario_runs(request.param, root / "runs")
    out = tool.run_compare(runs, {}, None, root=root / "compare", argv=["compare_runs.py", "synthetic"])
    return SimpleNamespace(name=request.param, out=out, **vars(read_outputs(out)))


def table_d_of(md: str) -> list[str]:
    """The lines of the table D section of comparison.md (up to the next section, blank lines at its end dropped)."""
    return md[md.index("## Table D."):md.index("\n## Table E.")].rstrip("\n").split("\n")


def limitation_items(md: str) -> list[str]:
    """The numbered items of the Limitations section of comparison.md."""
    return [line for line in md[md.index("## Limitations"):].splitlines() if re.match(r"\d+\. ", line)]


def has_token(text: str, token: str) -> bool:
    """``token`` is in ``text`` as a number of its own (not inside a longer number, a timestamp or a float)."""
    return re.search(rf"(?<![\d.]){re.escape(token)}(?!\d)", text) is not None


def protocol_line(n: int, bs4: int, bs8: int) -> str:
    return (f"- Protocol full: every test sample (N = {n} on the full data). drop_last_bs4 / drop_last_bs8: the first "
            f"floor(N/4)*4 / floor(N/8)*8 samples in row order ({bs4} / {bs8} for N = {n}), the sample sets of the "
            "released test loader with batch 4 and 8 and drop_last.")


# §23.2 평가 N (val / test) and 보조 프로토콜 test: N, floor(N/4)*4, floor(N/8)*8 (D17, D41).
PROTOCOL_N = {"movielens": (95, 92, 88), "steam": (1186, 1184, 1184), "lastfm": (122, 120, 120)}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_table_d_of_the_dataset(compared):
    """D41, §23.5 M14, §23.8: table D of a Steam or LastFM comparison as rendered: the LLaRA row (LLaRA Table 2) with
    its integer back-solution and our E1 full-protocol counterpart, the iLoRA row with the ``counts_text`` wording
    instead of counts and no counterpart, the copied LLaRA row in the wording of MovieLens, and the notes (Steam: both
    integer solutions and why N = 1186; Imp.% printed and computed; the no-threshold note with the dataset's N)."""
    assert table_d_of(compared.md) == table_d_lines(compared.name)


def test_paper_numbers_only_in_table_d(compared):
    """§18.6.8, §18.8 #7: the paper metric values, Imp.% values and integer back-solutions of the dataset appear in
    comparison.md table D and nowhere else; the other three outputs hold none of them."""
    tokens = PAPER_TOKENS[compared.name]
    for name, text in (("comparison.csv", compared.csv), ("spec20_index.json", compared.index),
                       ("summary.md", compared.summary)):
        assert not [token for token in tokens if has_token(text, token)], name
    start = compared.md.index("## Table D.")
    end = compared.md.index("\n## ", start + 1)
    assert all(has_token(compared.md[start:end], token) for token in tokens)
    outside = compared.md[:start] + compared.md[end:]
    assert not [token for token in tokens if has_token(outside, token)]


def test_definitions_and_limitations_follow_the_dataset(compared, tool):
    """§23.5 M14, D38, D41, §23.2: the protocol line carries the dataset's N (full and drop_last), the limitation items
    #3-#5 its N, title rule and unscorable counts, in comparison.md and summary.md; the other items are the common
    ones of ``LIMITATIONS``. No MovieLens number is left in these texts."""
    n, bs4, bs8 = PROTOCOL_N[compared.name]
    assert n == datasets.SPECS[compared.name].expected_samples["test"]
    assert protocol_line(n, bs4, bs8) in compared.md
    items = limitation_items(compared.md)
    assert items[2:5] == list(ITEMS[compared.name])
    assert items[0] == f"1. {tool.LIMITATIONS[0]}" and items[1].startswith(f"2. {tool.LIMITATIONS[1]}")
    assert items[5:8] == [f"{i}. {tool.LIMITATIONS[i - 1]}" for i in (6, 7, 8)]
    for item in ITEMS[compared.name]:
        assert item in compared.summary
    for movielens in ("N = 95", "N = 92", "1/95", "93/94", "93/95", "(D20)", "small N"):
        assert movielens not in compared.md and movielens not in compared.summary


def test_meta_inputs_are_the_dataset_table(compared):
    """§23.5 M14: the ``meta.json`` ``inputs`` of the comparison are the input table of the dataset that was read
    (``run.update_meta`` after ``build_report``), not the MovieLens default of ``start_meta``."""
    inputs = compared.meta["inputs"]
    assert list(inputs) == list(run.expected_inputs(compared.name))
    assert inputs == run.input_hashes(run.REPO_ROOT, compared.name)
    assert not set(inputs) & set(run.expected_inputs("movielens"))
    assert compared.meta["status"] == "completed"


def test_movielens_texts_are_unchanged(tool, tmp_path):
    """§23.7 (f): the texts that moved into the per-dataset table render for a MovieLens comparison exactly as in
    the 54e5ae0 tool (table D with its notes, the protocol line, limitation items #3-#5), and ``meta.json`` keeps the
    MovieLens input table written by ``start_meta``."""
    runs = scenario_runs("movielens", tmp_path / "runs")
    outputs = read_outputs(tool.run_compare(runs, {}, None, root=tmp_path / "compare", argv=["compare_runs.py"]))
    assert table_d_of(outputs.md) == table_d_lines("movielens")
    assert protocol_line(*PROTOCOL_N["movielens"]) in outputs.md
    assert limitation_items(outputs.md)[2:5] == list(ITEMS["movielens"])
    assert list(outputs.meta["inputs"]) == list(run.expected_inputs("movielens"))
    assert outputs.meta["inputs"] == run.input_hashes(run.REPO_ROOT, "movielens")


def test_no_usable_run_keeps_the_movielens_text(tool, tmp_path):
    """A comparison with no usable run names no dataset; it keeps the MovieLens wording of the earlier tool and stays a
    completed ``NOT_RUN`` comparison (module docstring of the tool)."""
    outputs = read_outputs(tool.run_compare({"E2": tmp_path / "no_such_run"}, {}, None, root=tmp_path / "compare",
                                            argv=["compare_runs.py"]))
    assert outputs.meta["status"] == "completed"
    assert "LLaRA (SASRec), MovieLens" in outputs.md and "N = 95 on the full data" in outputs.md
    assert list(outputs.meta["inputs"]) == list(run.expected_inputs("movielens"))


@pytest.mark.parametrize("case", ["lastfm_run", "movielens_run", "lastfm_export", "lastfm_cluster"])
def test_mixed_datasets_are_a_compare_error(tool, tmp_path, case):
    """§23.1, §23.5 M14: every usable run of one comparison must name the same dataset: a training run, an export or a
    cluster run of another dataset stops the comparison with ``CompareError`` naming both datasets (``meta.json``
    failed, no output file)."""
    root = tmp_path / "runs"
    e2 = make_train_run(root, "e2", "steam", best=0, last=0, correct={0: (0,)}, valid={0: (0, 1)})
    runs: dict[str, Path] = {"E2": e2}
    exports: dict[str, Path] = {}
    cluster = None
    if case.endswith("_run"):
        runs["E1"] = make_train_run(root, "e1", case.split("_")[0], sets=K1, best=0, last=0, correct={0: (1,)},
                                    valid={0: (1,)})
    elif case == "lastfm_export":
        exports["E2"] = make_aux_run(root, "export", "lastfm", "export", source_run=str(e2))
    else:
        cluster = make_aux_run(root, "cluster", "lastfm", "cluster")
    with pytest.raises(tool.CompareError, match="one comparison holds one dataset") as info:
        tool.run_compare(runs, exports, cluster, root=tmp_path / "compare", argv=["compare_runs.py"])
    assert "steam" in str(info.value) and case.split("_")[0] in str(info.value)
    (out,) = (tmp_path / "compare").glob("compare_*")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and "CompareError" in meta["error"]
    assert not (out / "comparison.md").exists() and not (out / "comparison.csv").exists()


def test_command_line_refuses_mixed_datasets(tool, tmp_path, capsys):
    """§18.6.8 종료 코드: a comparison of a Steam and a LastFM run exits 1 with a compare error (no traceback)."""
    root = tmp_path / "runs"
    e2 = make_train_run(root, "e2", "steam", best=0, last=0, correct={0: (0,)}, valid={0: (0, 1)})
    e1 = make_train_run(root, "e1", "lastfm", sets=K1, best=0, last=0, correct={0: (1,)}, valid={0: (1,)})
    code = tool.main(["--run", f"E2={e2}", "--run", f"E1={e1}", "--root", str(tmp_path / "compare")])
    assert code == 1
    assert "compare error:" in capsys.readouterr().err


@pytest.mark.parametrize("case", ["unknown_path", "missing_key"])
def test_unreadable_dataset_is_a_compare_error(tool, tmp_path, case):
    """§23.4, D27: a usable run whose ``prompt.template_path`` names no dataset, or that lacks the key, stops the
    comparison with ``CompareError`` naming its ``config.yaml``."""
    e2 = make_train_run(tmp_path / "runs", "e2", "steam", best=0, last=0, correct={0: (0,)}, valid={0: (0, 1)})
    path = e2 / "config.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if case == "unknown_path":
        doc["prompt"]["template_path"] = "prompt/book.txt"
    else:
        del doc["prompt"]["template_path"]
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(tool.CompareError) as info:
        tool.run_compare({"E2": e2}, {}, None, root=tmp_path / "compare", argv=["compare_runs.py"])
    assert ("no dataset uses the template file" if case == "unknown_path" else "cannot be read") in str(info.value)
    assert str(path) in str(info.value)


@pytest.mark.parametrize("dataset, n", [("steam", 1186), ("lastfm", 122)])
def test_sample_differences_at_the_dataset_n(tool, dataset, n):
    """D41: a real E1 full-protocol run has the back-solved N (§23.2 평가 N), so the LLaRA row gives the differences in
    samples against the back-solution (the test runs above have N = 12, ``sample differences NOT_RUN``)."""
    ref = tool.PAPER_TEXTS[dataset].reference[0]
    assert ref["counts"][0] == n == datasets.SPECS[dataset].expected_samples["test"]
    v, c = n - 5, 50
    row = {"n_total": n, "n_valid": v, "n_correct": c, "valid_ratio": v / n, "hr_at_1": c / v, "combined_metric": c / n}
    report = SimpleNamespace(table={("E1", "best_val", "full"): row})
    _, v_ref, c_ref = ref["counts"]
    assert tool.counterpart_text(report, ref) == (
        f"E1 best_val full: valid_ratio {v / n!r} ({v}/{n}), combined_metric {c / n!r} ({c}/{n}); "
        f"valid {v - v_ref:+d}, correct {c - c_ref:+d} samples against the back-solution")


# ----------------------------------------------------------------------------
# The recorded numbers, recomputed from the printed paper values (back-solve of 18.6.7, D41)
# ----------------------------------------------------------------------------
HALF_STEP = Fraction(1, 20000)                                 # half of the printed 1e-4
RELEASED_N = {"steam": (1184, 1186, 1192, 1194), "lastfm": (120, 122)}      # D41, §23.2 (1192: older test file, D33)


def close(numerator: int, denominator: int, printed: str) -> bool:
    return denominator > 0 and abs(Fraction(numerator, denominator) - Fraction(printed)) <= HALF_STEP


def solutions(n: int, valid_ratio: str, hit_ratio: str) -> list[tuple[int, int, int, str]]:
    out = []
    for v in range(n + 1):
        if not close(v, n, valid_ratio):
            continue
        for c in range(v + 1):
            if close(c, v, hit_ratio):
                out.append((n, v, c, "correct / valid"))
            if close(c, n, hit_ratio):
                out.append((n, v, c, "correct / total"))
    return out


def test_integer_solutions_are_the_recorded_ones():
    """D41, §23.2: at the N the released code can produce the LLaRA Steam row has exactly the two solutions of the table
    D note (both correct / total) and the iLoRA Steam row none; the LastFM LLaRA row has (122, 122, 55) only and the
    iLoRA row (120, 120, 60) and (122, 122, 61) (valid = total, so the denominators coincide)."""
    steam_n, lastfm_n = RELEASED_N["steam"], RELEASED_N["lastfm"]
    assert sorted(s for n in steam_n for s in solutions(n, "0.9975", "0.4949")) == [
        (1184, 1181, 586, "correct / total"), (1186, 1183, 587, "correct / total")]
    assert not [s for n in steam_n for s in solutions(n, "0.9981", "0.5264")]
    assert sorted(s for n in lastfm_n for s in solutions(n, "1.0000", "0.4508")) == [
        (122, 122, 55, "correct / total"), (122, 122, 55, "correct / valid")]
    assert sorted(s for n in lastfm_n for s in solutions(n, "1.0000", "0.5000")) == [
        (120, 120, 60, "correct / total"), (120, 120, 60, "correct / valid"),
        (122, 122, 61, "correct / total"), (122, 122, 61, "correct / valid")]


def test_table_d_rows_agree_with_their_numbers(tool):
    """Every table D row with counts, MovieLens included, rounds to its printed values under its denominator (D17, D41),
    and the LLaRA row of each dataset is at the dataset's full test N (D41: "over all sequences")."""
    for name, paper in tool.PAPER_TEXTS.items():
        assert paper.reference[0]["counts"][0] == datasets.SPECS[name].expected_samples["test"], name
        for ref in paper.reference:
            if ref["counts"] is None:
                continue
            n, v, c = ref["counts"]
            den = n if ref["denominator"] == "correct / total" else v
            assert close(v, n, ref["valid_ratio"]) and close(c, den, ref["hit_ratio_at_1"]), (name, ref["row"])


def test_imp_note_values_are_computed():
    """[논문 p.7 Table 1] Imp.% = iLoRA / LLaRA - 1 in percent: MovieLens 19.32% and Steam 6.36% as printed; the
    LastFM cell is printed as 8.51% but the printed values give 10.91% (§23.2)."""
    def imp(ilora: str, llara: str) -> str:
        return f"{float((Fraction(ilora) / Fraction(llara) - 1) * 100):.2f}"

    assert imp("0.5275", "0.4421") == "19.32"
    assert imp("0.5264", "0.4949") == "6.36"
    assert imp("0.5000", "0.4508") == "10.91"
