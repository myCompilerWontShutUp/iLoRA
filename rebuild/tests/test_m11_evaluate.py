"""M11 생성과 평가 (DESIGN.md §15.8 #1-#17 and M7-12; test names of §17.6.9).

The judgement is compared with ``refs_upstream.first_line_and_hr1`` (the copy of
``U model/model_interface.py:89-90, :165, :468-492``) on hand cases and 10,000 seeded cases built from
catalog titles; decoding uses the test tokenizer (``bundle``); generation uses the CPU test model
(``tiny_model``: tiny random LLaMA, real SASRec, CPU fp32) on the smoke data or, for the slow tests, on the
whole validation and test splits (``data.subset=full``). #18-#22 are VESSL tests
(``tests/vessl/test_run_outputs.py``, ``tests/vessl/test_vessl_env.py``).

``main.py eval`` (#16) reads checkpoints only through M10 ``load_model_state`` (§14.6.6). train.py is not
built yet, so ``test_eval_command`` registers a stand-in ``ilora.train`` (a reader of the synthetic
``trainable.pt`` written here, with the generated_frozen.* bit comparison) and a recording stand-in
``ilora.cluster`` for the M15 preparation order, both through ``monkeypatch.setitem(sys.modules, ...)``.
Synthetic values come from ``seeding.np_rng(0, "test", <name>)`` and ``seeding.torch_seed_scope(0, "test",
<name>)`` (§17.6.2).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import random
import shutil
import statistics
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import refs_upstream
from ilora import evaluate, records, run, seeding
from ilora.batching import collate_eval, encode_eval, encode_text
from ilora.candidates import FLAG_NAMES, candidate_flags, candidates_for
from ilora.evaluate import EvalError, EvalRecord, SplitMetrics
from ilora.prompts import build_prompt, choose_template
from ilora.records import RecordsError
from ilora.sections import CandidatesConfig

K = 4                                   # lora.num_experts (paper_spec, smoke_cpu)
SEED = 1234                             # run.seed
SPLIT_SHA256 = {"val": "9a07320651cadd0d049efc8644cb9a6103a78bdc2b45f2bd793ae204a545f01a",     # M2 6.6.5
                "test": "1ffdb6499d3a316fb8cf94c322f09e5af509520f8cdcc6574634b99a7a30bef3"}
TEMPLATE_COUNTS = {"val": [30, 33, 31], "test": [22, 36, 37]}                                  # M4 8.6.3
TEST_FLAG_ROWS = (5, 11, 38, 39, 43)    # test rows with a candidate flag (M2 6.6.6, code titles)
SPECIAL_IDS = (0, 1, 2, 32000, 32001, 32002, 32003, 32004)   # unk, bos, eos, [PAD], [PH], [HistoryEmb], [CansEmb], [ItemEmb]
NO_BI = "eval.batch_invariance_check=false"
FULL = "data.subset=full"
CLUSTER_HARD = ("router.routing_mode=cluster_hard", "cluster.artifact_dir=unused_by_m11/cluster")
K1 = ("lora.num_experts=1", "router.routing_mode=none")
EVAL_FILES = {"records.jsonl", "metrics.json", "representations.npz"}
CC20 = CandidatesConfig(num_candidates=20)
EVALUATE_PY = Path(__file__).resolve().parents[1] / "ilora" / "evaluate.py"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def new_run_dir(root: Path, name: str = "m11") -> Path:
    return run.create_run_dir(name, root=root / "runs")


def rng_states():
    return torch.get_rng_state(), random.getstate(), np.random.get_state()


def assert_rng_states_equal(a, b) -> None:
    assert torch.equal(a[0], b[0]), "torch CPU generator changed"
    assert a[1] == b[1], "Python random changed"
    assert a[2][0] == b[2][0] and np.array_equal(a[2][1], b[2][1]) and a[2][2:] == b[2][2:], "numpy global changed"


def disturb_global_rng(name: str) -> None:
    """Move the three global generators away from their defaults (so a reset would be visible)."""
    torch.rand(int(seeding.np_rng(0, "test", name).integers(1, 9)))
    random.random()
    np.random.random(3)


def perturb_trainable(model, name: str, std: float = 0.02) -> None:
    """Seeded noise on every trainable parameter (LoRA B no longer 0, so omega changes the generation)."""
    with torch.no_grad(), seeding.torch_seed_scope(0, "test", name):
        for items in model.param_groups().values():
            for _, p in items:
                p.add_(torch.randn(p.shape, dtype=p.dtype) * std)


def eval_batch(cfg, data, bundle, templates, split: str = "val", n: int = 2, start: int = 0):
    """An ``EvalBatch`` of the keyed candidates/templates through the public M2/M4/M5 functions."""
    samples = list(data.samples(split)[start:start + n])
    cands, tids, encoded = [], [], []
    hybrid = cfg.prompt.eval_prompt == "hybrid"
    for s in samples:
        c = candidates_for(s, main_seed=cfg.run.seed, epoch=None, cfg=cfg.candidates)
        k = choose_template(cfg.prompt, cfg.run.seed, len(templates.templates), s, None)
        cands.append(c)
        tids.append(k)
        encoded.append(encode_eval(bundle, build_prompt(s, c, data.catalog, templates, k, hybrid)))
    return collate_eval(samples, cands, tids, encoded, data.catalog, bundle, hybrid=hybrid)


def synthetic_record(split: str, i: int, *, valid: bool, correct: bool, k: int = K, flagged: bool = False):
    """A minimal EvalRecord for metric tests (ids/titles consistent with the judgement flags given)."""
    ids = tuple(range(100, 120))
    titles = tuple(f"Movie {c}" for c in ids)
    matched = (ids[0] if correct else ids[1],) if valid else ()
    return EvalRecord(sample_id=f"{split}:{i}", split=split, user_id=754 + i, template_id=i % 3, hybrid=True,
                      cand_ids=ids, cand_titles=titles, target_id=ids[0], target_title=titles[0],
                      raw_text=titles[0] if correct else "x", first_line=titles[0] if correct else "x",
                      matched=matched, valid=valid, correct=correct, gates=tuple([1.0 / k] * k),
                      flags={name: (flagged and name == FLAG_NAMES[0]) for name in FLAG_NAMES}, gen_token_ids=(2,))


def records_with_counts(split: str, n: int, v: int, c: int) -> list:
    assert c <= v <= n
    return [synthetic_record(split, i, valid=i < v, correct=i < c) for i in range(n)]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line, object_pairs_hook=lambda pairs: dict(pairs)) for line in
            path.read_text(encoding="utf-8").splitlines()]


def first_keys(path: Path) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8").splitlines()[0]))


@contextlib.contextmanager
def capture_generate(model):
    """Collect every ``GenerationOutput`` of ``model.generate`` (instance attribute in front of the method)."""
    outputs = []
    original = model.generate

    def spy(batch):
        out = original(batch)
        outputs.append(out)
        return out

    model.generate = spy
    try:
        yield outputs
    finally:
        del model.generate


@contextlib.contextmanager
def capture_gate_record(model):
    """Collect the ``GateRecord`` that ``generate_records`` gets from the M7 recorder."""
    got = []
    router = model.omega_source
    original = router.stop_recording

    def spy(expected):
        rec = original(expected)
        got.append(rec)
        return rec

    router.stop_recording = spy
    try:
        yield got
    finally:
        del router.stop_recording


def bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# ----------------------------------------------------------------------------------------------- #1, #2
WORDS = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliett", "Kilo",
         "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango")
HAND_IDS = tuple(range(100, 120))
# case -> (raw generation, {candidate index: title override}, target index, expected (valid, correct))
HAND_CASES = {
    "exact": (" Charlie Story\nAlpha Story", {}, 2, (True, True)),
    "other_candidate": ("Delta Story", {}, 2, (True, False)),
    "empty_title_candidate": ("Charlie Story", {5: ""}, 2, (False, False)),
    "empty_title_alone": ("no title here", {5: ""}, 5, (True, True)),
    "same_normalized_titles": ("Charlie Story", {3: "  CHARLIE story "}, 2, (False, False)),
    "substring_title": ("Charlie Story", {7: "Story"}, 2, (False, False)),
    "case": ("CHARLIE STORY", {}, 2, (True, True)),
    "spaces_and_cr": ("  Charlie Story \r\nAlpha Story", {}, 2, (True, True)),
    "starts_with_newline": ("\nCharlie Story\nDelta Story\n", {}, 2, (True, True)),
    "blank_first_segment": (" \n \nDelta Story", {}, 2, (True, False)),
    "no_newline": ("I recommend Charlie Story to you", {}, 2, (True, True)),
    "two_titles_one_line": ("Charlie Story or Delta Story", {}, 2, (False, False)),
    "title_on_second_line": ("Something else\nCharlie Story", {}, 2, (False, False)),
    "empty_generation": ("", {}, 2, (False, False)),
    "target_trailing_space": ("Charlie Story", {2: "Charlie Story "}, 2, (True, True)),
}


def hand_titles(overrides: dict) -> list[str]:
    titles = [f"{w} Story" for w in WORDS]
    for index, title in overrides.items():
        titles[index] = title
    return titles


@pytest.mark.criterion("M11-1")
@pytest.mark.parametrize("case", list(HAND_CASES))
def test_judge_matches_upstream(case):
    """first_line + judge == the released rule copy (first line, valid, correct) on the hand cases of 15.8 #1."""
    raw, overrides, target, expected = HAND_CASES[case]
    titles = hand_titles(overrides)
    line = evaluate.first_line(raw)
    verdict = evaluate.judge(line, HAND_IDS, titles, titles[target])
    ref_line, ref_valid, ref_correct = refs_upstream.first_line_and_hr1(raw, titles, titles[target])
    assert (line, verdict.valid, verdict.correct) == (ref_line, ref_valid, ref_correct)
    assert (verdict.valid, verdict.correct) == expected
    assert verdict.valid == (len(verdict.matched) == 1)
    assert list(verdict.matched) == [c for c, t in zip(HAND_IDS, titles)
                                     if t.strip().lower().strip() in line.strip().lower().strip()]


def random_cases(catalog, n: int, name: str):
    """Seeded generations from catalog titles: a candidate or the target title, two lines, two titles in one line,
    a truncated title, any catalog title, title words; random case, spaces, ``\\r`` and leading newlines."""
    rng = seeding.np_rng(0, "test", name)
    titles = catalog.titles
    for _ in range(n):
        ids = tuple(int(x) for x in rng.choice(len(titles), size=20, replace=False))
        target = ids[int(rng.integers(20))]
        cand_titles = [titles[c] for c in ids]
        pick = lambda: cand_titles[int(rng.integers(20))]          # noqa: E731
        kind = int(rng.integers(8))
        if kind == 0:
            text = pick()
        elif kind == 1:
            text = titles[target]
        elif kind == 2:
            text = pick() + "\n" + pick()
        elif kind == 3:
            text = pick() + " " + pick()
        elif kind == 4:
            title = pick()
            text = title[:int(rng.integers(len(title) + 1))]
        elif kind == 5:
            text = titles[int(rng.integers(len(titles)))]
        elif kind == 6:
            words = " ".join(cand_titles).split()
            text = " ".join(words[int(i)] for i in rng.integers(0, max(len(words), 1), size=int(rng.integers(1, 4))))
        else:
            text = ""
        style = int(rng.integers(6))
        if style == 1:
            text = text.upper()
        elif style == 2:
            text = "  " + text.lower() + " \r\n" + pick()
        elif style == 3:
            text = "\n" + text + "\n"
        elif style == 4:
            text = " \n " + text
        yield ids, cand_titles, target, titles[target], text


@pytest.mark.criterion("M11-1")
def test_judge_random_10000(full_data):
    """10,000 seeded cases from catalog titles: first line, valid and correct equal the released rule copy."""
    counts = Counter()
    for ids, cand_titles, target, target_title, text in random_cases(full_data.catalog, 10_000, "m11_judge_random"):
        line = evaluate.first_line(text)
        verdict = evaluate.judge(line, ids, cand_titles, target_title)
        assert (line, verdict.valid, verdict.correct) == refs_upstream.first_line_and_hr1(text, cand_titles,
                                                                                         target_title), text
        counts[(verdict.valid, verdict.correct)] += 1
    # every outcome occurs (the cases exercise the rule, not only one branch)
    assert counts[(True, True)] > 500 and counts[(True, False)] > 500 and counts[(False, False)] > 500, counts


@pytest.mark.criterion("M11-2")
def test_correct_iff_target_id(full_data):
    """In every valid case the string judgement is correct <=> matched[0] == target id: the 10,000 seeded cases and
    each candidate title of every validation and test sample (keyed candidates, D20 duplicates and item 266)."""
    checked = 0
    for ids, cand_titles, target, target_title, text in random_cases(full_data.catalog, 10_000, "m11_judge_random"):
        verdict = evaluate.judge(evaluate.first_line(text), ids, cand_titles, target_title)
        if verdict.valid:
            assert verdict.correct == (verdict.matched[0] == target), text
            checked += 1
    titles = full_data.catalog.titles
    for s in (*full_data.val, *full_data.test):
        cands = candidates_for(s, main_seed=SEED, epoch=None, cfg=CC20)
        cand_titles = [titles[c] for c in cands]
        for line in cand_titles:
            verdict = evaluate.judge(line, cands, cand_titles, titles[s.target])
            if verdict.valid:
                assert verdict.correct == (verdict.matched[0] == s.target), (s.sample_id, line)
                checked += 1
    assert checked > 5_000


@pytest.mark.criterion("M11-2")
@pytest.mark.parametrize("case", ["19_candidates", "21_candidates", "duplicate_id", "no_target"])
def test_judge_rejects(case, full_data):
    """19 or 21 candidates, a repeated candidate id and a candidate set without the target -> EvalError."""
    titles = hand_titles({})
    if case == "19_candidates":
        call = lambda: evaluate.judge("Alpha Story", HAND_IDS[:19], titles[:19], titles[0])      # noqa: E731
    elif case == "21_candidates":
        call = lambda: evaluate.judge("Alpha Story", HAND_IDS + (999,), titles + ["Extra"], titles[0])   # noqa: E731
    elif case == "duplicate_id":
        call = lambda: evaluate.judge("Alpha Story", HAND_IDS[:19] + (HAND_IDS[0],), titles, titles[0])  # noqa: E731
    else:
        s = full_data.val[0]
        cands = candidates_for(s, main_seed=SEED, epoch=None, cfg=CC20)
        other = next(i for i in range(1682) if i not in cands and i not in s.history and i != s.target)
        broken = tuple(other if c == s.target else c for c in cands)
        catalog = full_data.catalog
        expected = evaluate._Expected(sample=s, cands=broken, template_id=0, flags={n: False for n in FLAG_NAMES})
        call = lambda: evaluate._build_record(                                                      # noqa: E731
            expected, catalog=catalog, cand_titles=[catalog.titles[c] for c in broken],
            target_title=catalog.titles[s.target], template_id=0, hybrid=True, raw_text="x", gates=[0.25] * 4,
            gen_token_ids=(2,))
    with pytest.raises(EvalError):
        call()


# ----------------------------------------------------------------------------------------------- #3, #4
DECODE_TEXTS = ("Star Wars\n", "\nStar Wars\nHeat\n", "Good Will Hunting\nThe Rock", " \n Toy Story\n",
                "Monty Python and the Holy Grail\n", "Á köldum klaka (Cold Fever)\n")


@pytest.mark.criterion("M11-3")
def test_decode_rules(bundle):
    """Decoding (15.6.4, design measurement D2-D4): the BOS placeholder and [PAD] after EOS change nothing, the 8
    special ids are dropped anywhere, and among the 32,000 pieces only id 13 decodes to a newline (25 to \\r)."""
    assert sorted(bundle.tokenizer.all_special_ids) == list(SPECIAL_IDS)
    for text in DECODE_TEXTS:
        a = encode_text(bundle, text)
        padded = a + [2] + [32000] * 5
        decoded = evaluate.decode_generated(bundle, [a, [1] + a, padded, list(evaluate.trim_generated(padded, eos_id=2))])
        assert decoded == [text] * 4, text
    a = encode_text(bundle, "Star Wars\n")
    mixed = [1, 32001] + a[:1] + [32002, 0, 32003] + a[1:] + [32000, 32004, 2]
    assert set(mixed) >= set(SPECIAL_IDS)
    assert evaluate.decode_generated(bundle, [mixed]) == ["Star Wars\n"]
    assert evaluate.decode_generated(bundle, [[i] for i in SPECIAL_IDS]) == [""] * len(SPECIAL_IDS)
    texts = evaluate.decode_generated(bundle, [[i] for i in range(32000)])
    assert [i for i, t in enumerate(texts) if "\n" in t] == [13]
    assert sum(1 for t in texts if "\r" in t) == 25


@pytest.mark.criterion("M11-4")
def test_newline_counterexample(bundle, full_data):
    """Stopping at the first newline would change the first line (15.6.4, D5): ``"\\nStar Wars\\nHeat\\n"`` gives
    ``"Star Wars"`` while the text cut after the first 13 gives ``""``, and the judgement differs."""
    for gen in ([29871, 13, 16213, 9149, 13, 3868, 271, 13], [29871, 13, 7828, 9149, 13]):
        text = evaluate.decode_generated(bundle, [gen])[0]
        assert evaluate.first_line(text) == "Star Wars"
        stopped = evaluate.decode_generated(bundle, [gen[:gen.index(13) + 1]])[0]
        assert evaluate.first_line(stopped) == ""
        assert evaluate.first_line(text) == refs_upstream.first_line_and_hr1(text, ["x"] * 20, "x")[0]
    assert encode_text(bundle, "\nStar Wars\nHeat\n") == [29871, 13, 16213, 9149, 13, 3868, 271, 13]
    titles = full_data.catalog.titles
    star, empty = titles.index("Star Wars"), 266
    assert titles[empty] == ""
    others = [i for i in range(1682) if titles[i] and i != star][:18]
    cands = (empty, star, *others)
    cand_titles = [titles[c] for c in cands]
    full = evaluate.judge("Star Wars", cands, cand_titles, titles[star])
    cut = evaluate.judge("", cands, cand_titles, titles[star])
    assert (full.matched, full.valid) == ((empty, star), False)        # the empty title matches every line
    assert (cut.matched, cut.valid, cut.correct) == ((empty,), True, False)


@pytest.mark.criterion("M11-4")
def test_gen_config_stops(tiny_model, cpu_cfg, smoke_data, bundle, templates, monkeypatch):
    """``model.gen_config`` ends generation at EOS 2 (one id) or the maximum length only: the stopping criteria of
    a real generate call are exactly one MaxLengthCriteria (T4.28 generation/utils.py:912-921); a list of EOS ids
    or a max_time is refused by the evaluation check."""
    from transformers.generation.stopping_criteria import MaxLengthCriteria

    model = tiny_model()
    gc = model.gen_config
    assert gc.eos_token_id == 2 and isinstance(gc.eos_token_id, int) and gc.max_time is None
    assert (gc.do_sample, gc.num_beams, gc.max_new_tokens, gc.pad_token_id) == (False, 1, 64, 32000)
    seen = []
    original = type(model.llm)._get_stopping_criteria

    def spy(self, generation_config, stopping_criteria):
        criteria = original(self, generation_config, stopping_criteria)
        seen.append([type(c) for c in criteria])
        return criteria

    monkeypatch.setattr(type(model.llm), "_get_stopping_criteria", spy)
    out = model.generate(eval_batch(cpu_cfg, smoke_data, bundle, templates, n=2))
    assert seen == [[MaxLengthCriteria]] and out.new_token_ids.shape[0] == 2
    evaluate._check_gen_config(model, cpu_cfg, 2, 32000)
    for change in ({"eos_token_id": [2, 13]}, {"max_time": 5.0}, {"num_beams": 2}, {"do_sample": True}):
        changed = model.gen_config
        for key, value in change.items():
            setattr(changed, key, value)
        with pytest.raises(EvalError):
            evaluate._check_gen_config(SimpleNamespace(gen_config=changed), cpu_cfg, 2, 32000)


# ----------------------------------------------------------------------------------------------- #5
@pytest.mark.criterion("M11-5")
def test_trim_generated():
    """Cut after the first EOS (EOS kept, [PAD] dropped); 64 tokens without EOS stay 64 tokens."""
    assert evaluate.trim_generated([711, 712, 2, 32000, 32000], eos_id=2) == (711, 712, 2)
    assert evaluate.trim_generated([2, 32000], eos_id=2) == (2,)
    assert evaluate.trim_generated(np.array([5, 2, 32000], dtype=np.int64), eos_id=2) == (5, 2)
    row = [int(x) for x in seeding.np_rng(0, "test", "m11_trim").integers(3, 32000, size=64)]
    assert evaluate.trim_generated(row, eos_id=2) == tuple(row) and len(row) == 64
    for bad in ([1.5, 2], [True, 2], "ab"):
        with pytest.raises(EvalError):
            evaluate.trim_generated(bad, eos_id=2)


def generation_output(name: str, **changes) -> SimpleNamespace:
    """A valid GenerationOutput of 2 rows (T = 4 < 64, every row has EOS) with optional replacements."""
    with seeding.torch_seed_scope(0, "test", name):
        omega = torch.softmax(torch.randn(2, K), dim=-1)
        z = torch.randn(2, 64)
    values = {"new_token_ids": torch.tensor([[5, 6, 2, 32000], [7, 8, 9, 2]], dtype=torch.int64),
              "omega": omega, "z": z}
    values.update(changes)
    return SimpleNamespace(**values)


CHECK_OUTPUT_CASES = {
    "token_after_eos": ({"new_token_ids": torch.tensor([[5, 2, 7, 32000], [7, 8, 9, 2]])}, EvalError),
    "t_0": ({"new_token_ids": torch.zeros(2, 0, dtype=torch.int64)}, EvalError),
    "t_65": ({"new_token_ids": torch.full((2, 65), 7, dtype=torch.int64)}, EvalError),
    "short_without_eos": ({"new_token_ids": torch.tensor([[5, 6, 2, 32000], [7, 8, 9, 10]])}, EvalError),
    "omega_batch_axis": ({"omega": torch.full((2, 1, K), 1.0 / K)}, RecordsError),
    "omega_nan": ({"omega": torch.tensor([[float("nan"), 0.5, 0.25, 0.25], [0.25] * 4])}, RecordsError),
    "omega_float16": ({"omega": torch.full((2, K), 1.0 / K, dtype=torch.float16)}, RecordsError),
    "z_dim": ({"z": torch.zeros(2, 63)}, RecordsError),
}


@pytest.mark.criterion("M11-5")
@pytest.mark.parametrize("case", list(CHECK_OUTPUT_CASES))
def test_check_output_rejects(case):
    """check_output refuses a non-[PAD] token after EOS, T = 0, T = 65, a row without EOS when T < 64, omega
    [B, 1, K] / NaN / float16 and a z of the wrong width (M12 checks raise RecordsError unchanged, 15.6.13)."""
    kwargs = dict(batch_size=2, num_experts=K, z_dim=64, max_new_tokens=64, eos_id=2, pad_id=32000,
                  rowsum_atol=1e-5, sample_ids=["val:1", "val:7"])
    evaluate.check_output(generation_output("m11_check_output"), **kwargs)          # the valid output passes
    changes, error = CHECK_OUTPUT_CASES[case]
    with pytest.raises(error):
        evaluate.check_output(generation_output("m11_check_output", **changes), **kwargs)


# ----------------------------------------------------------------------------------------------- #6
METRIC_CASES = ["95_90_40", "95_0_0", "n_0", "correct_not_valid", "other_split"]


@pytest.mark.criterion("M11-6")
@pytest.mark.parametrize("case", METRIC_CASES)
def test_compute_metrics(case):
    """(95, 90, 40) gives exactly 90/95, 40/90, 40/95; (95, 0, 0) gives HR@1 0.0; n = 0, correct without valid
    and a record of another split raise EvalError."""
    if case == "95_90_40":
        m = evaluate.compute_metrics(records_with_counts("test", 95, 90, 40), "test")
        assert m == SplitMetrics("test", 95, 90, 40, 90 / 95, 40 / 90, 40 / 95)
        assert (m.valid_ratio, m.hr_at_1, m.combined_metric) == (90 / 95, 40 / 90, 40 / 95)   # one division each
        return
    if case == "95_0_0":
        m = evaluate.compute_metrics(records_with_counts("val", 95, 0, 0), "val")
        assert (m.n_valid, m.n_correct, m.valid_ratio, m.hr_at_1, m.combined_metric) == (0, 0, 0.0, 0.0, 0.0)
        assert isinstance(m.hr_at_1, float)
        return
    if case == "n_0":
        recs = []
    elif case == "correct_not_valid":
        recs = records_with_counts("val", 10, 5, 2)
        recs[7] = dataclasses.replace(recs[7], correct=True)
    else:
        recs = records_with_counts("val", 10, 5, 2)
        recs[3] = dataclasses.replace(recs[3], split="test")
    with pytest.raises(EvalError):
        evaluate.compute_metrics(recs, "val")


# ----------------------------------------------------------------------------------------------- #7
@pytest.mark.criterion("M11-7")
def test_aux_protocols(full_data):
    """test 95 -> drop_last_bs4 = first 92, drop_last_bs8 = first 88; validation -> {}; smoke 10 -> 8 and 8. The
    flagged test rows 5, 11, 38, 39, 43 (M2 6.6.6) are all among the first 88."""
    assert records.AUX_BATCH_SIZES == evaluate.AUX_DROP_LAST_BATCH_SIZES == (4, 8)       # M12 recount = M11 blocks
    assert tuple(f.name for f in dataclasses.fields(EvalRecord)) == records.EVAL_RECORD_FIELDS     # 3.1, 15.6.1
    assert tuple(f.name for f in dataclasses.fields(SplitMetrics)) == records.SPLIT_METRICS_FIELDS
    assert evaluate.FULL_SPLIT_SIZE == {"val": 94, "test": 95} and evaluate.EVAL_SPLITS == records.EVAL_SPLITS
    recs = [synthetic_record("test", i, valid=i % 3 != 0, correct=i % 3 != 0 and i % 5 == 1) for i in range(95)]
    aux = evaluate.aux_protocol_metrics(recs, "test")
    assert list(aux) == ["drop_last_bs4", "drop_last_bs8"]
    assert aux["drop_last_bs4"] == evaluate.compute_metrics(recs[:92], "test") and aux["drop_last_bs4"].n_total == 92
    assert aux["drop_last_bs8"] == evaluate.compute_metrics(recs[:88], "test") and aux["drop_last_bs8"].n_total == 88
    assert evaluate.aux_protocol_metrics([synthetic_record("val", i, valid=True, correct=False) for i in range(94)],
                                         "val") == {}
    smoke = evaluate.aux_protocol_metrics(recs[:10], "test")
    assert [m.n_total for m in smoke.values()] == [8, 8]
    with pytest.raises(EvalError):
        evaluate.aux_protocol_metrics(recs[:5], "test")                 # fewer than 8 records: no drop_last_bs8
    flagged = []
    for position, s in enumerate(full_data.test):
        cands = candidates_for(s, main_seed=SEED, epoch=None, cfg=CC20)
        if any(candidate_flags(cands, s.target, full_data.catalog).values()):
            flagged.append((s.row, position))
    assert [row for row, _ in flagged] == list(TEST_FLAG_ROWS)
    assert all(position < 88 for _, position in flagged)


@pytest.mark.slow
@pytest.mark.criterion("M11-7")
def test_drop_last_bs8_prefix_bitwise(tiny_model, profile_cfg, full_data, bundle, templates):
    """With eval.batch_size 8 the first 88 test records of the whole split equal, bit for bit, an evaluation of the
    first 88 samples alone (same batch composition, so drop_last_bs8 needs no batch-independence assumption)."""
    cfg = profile_cfg("smoke_cpu", FULL, NO_BI)
    model = tiny_model()
    perturb_trainable(model, "m11_prefix")
    whole = evaluate.generate_records(model, bundle, full_data, "test", cfg, batch_size=8, templates=templates,
                                      record_gates=False)
    first88 = dataclasses.replace(full_data, test=full_data.test[:88])
    part = evaluate.generate_records(model, bundle, first88, "test", cfg, batch_size=8, templates=templates,
                                     record_gates=False)
    assert len(whole.records) == 95 and len(part.records) == 88 and part.n_batches == 11
    assert whole.records[:88] == part.records
    assert np.array_equal(bits(whole.omega[:88]), bits(part.omega)) and np.array_equal(bits(whole.z[:88]), bits(part.z))
    assert evaluate.aux_protocol_metrics(whole.records, "test")["drop_last_bs8"] == \
        evaluate.compute_metrics(part.records, "test")


# ----------------------------------------------------------------------------------------------- #8
@pytest.mark.criterion("M11-8")
def test_wilson():
    """Design reference values within 1e-6 (design/tmp/m11_wilson.py), n = 0 -> None, every interval inside [0, 1],
    WILSON_Z = NormalDist().inv_cdf(0.975)."""
    assert evaluate.WILSON_Z == statistics.NormalDist().inv_cdf(0.975) and repr(evaluate.WILSON_Z) == "1.9599639845400536"
    reference = {(42, 95): (0.346422, 0.542289), (48, 91): (0.425878, 0.626842), (0, 10): (0.0, 0.277533),
                 (10, 10): (0.722467, 1.0), (86, 95): (0.829670, 0.949356), (1, 1): (0.206549, 1.0)}
    for (k, n), (lo, hi) in reference.items():
        got = evaluate.wilson_interval(k, n)
        assert abs(got[0] - lo) <= 1e-6 and abs(got[1] - hi) <= 1e-6, (k, n, got)
    assert evaluate.wilson_interval(0, 0) is None
    for n in (1, 2, 10, 94, 95):
        for k in range(n + 1):
            lo, hi = evaluate.wilson_interval(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0
    for k, n in ((5, 4), (-1, 4), (1.0, 4), (True, 4)):
        with pytest.raises(EvalError):
            evaluate.wilson_interval(k, n)


# ----------------------------------------------------------------------------------------------- #9
def _swap_rows(batch, i: int, j: int):
    order = list(range(len(batch.sample_ids)))
    order[i], order[j] = order[j], order[i]
    tensors = {name: getattr(batch, name)[order] for name in ("input_ids", "attention_mask", "hist_ids", "len_seq",
                                                              "cand_ids", "target_ids")}
    lists = {name: [getattr(batch, name)[k] for k in order] for name in ("cand_titles", "target_titles",
                                                                         "sample_ids", "template_ids")}
    return dataclasses.replace(batch, **tensors, **lists)


def _mutate(batch, case: str):
    if case == "row_order":
        return _swap_rows(batch, 0, 1)
    if case == "template_id":
        return dataclasses.replace(batch, template_ids=[(batch.template_ids[0] + 1) % 3, *batch.template_ids[1:]])
    if case == "candidate_order":
        cand_ids = batch.cand_ids.clone()
        cand_ids[0] = cand_ids[0].flip(0)
        titles = [list(reversed(batch.cand_titles[0])), *batch.cand_titles[1:]]
        return dataclasses.replace(batch, cand_ids=cand_ids, cand_titles=titles)
    if case == "hybrid":
        return dataclasses.replace(batch, hybrid=not batch.hybrid)
    if case == "target_id":
        targets = batch.target_ids.clone()
        cands = batch.cand_ids[0].tolist()
        targets[0] = cands[0] if cands[0] != int(targets[0]) else cands[1]
        return dataclasses.replace(batch, target_ids=targets)
    titles = [list(t) for t in batch.cand_titles]
    titles[0][3] = titles[0][3] + " (Other Cut)"
    return dataclasses.replace(batch, cand_titles=titles)


@pytest.mark.criterion("M11-9")
@pytest.mark.parametrize("case", ["row_order", "template_id", "candidate_order", "hybrid", "target_id",
                                  "candidate_title"])
def test_batch_crosscheck_rejects(case, tiny_model, profile_cfg, smoke_data, bundle, templates, monkeypatch):
    """A mock loader whose second batch has two rows swapped, one template id changed, a candidate order reversed,
    the hybrid flag flipped, another target id or another candidate title -> EvalError before that batch is
    generated; the recorder is discarded and the model mode restored."""
    cfg = profile_cfg("smoke_cpu", "eval.batch_size=4", NO_BI)
    model = tiny_model()
    model.train()

    def mock_loader(dataset, num_workers, pin_memory):
        items = [dataset[i] for i in range(len(dataset))]
        items[1] = _mutate(items[1], case)
        return items

    monkeypatch.setattr(evaluate, "make_loader", mock_loader)
    with capture_generate(model) as outputs:
        with pytest.raises(EvalError, match="batch 1"):
            evaluate.generate_records(model, bundle, smoke_data, "val", cfg, batch_size=4, templates=templates,
                                      record_gates=True)
    assert len(outputs) == 1                                         # only batch 0 was generated
    assert model.training and model.sasrec_is_eval()
    model.omega_source.start_recording("val")                        # the recording was discarded
    model.omega_source.discard_recording()


# ----------------------------------------------------------------------------------------------- #10-#12
def check_eval_folder(out_dir: Path, result, gate_record, *, split: str, n: int, n_batches: int, data, k: int = K):
    """Files, rows, gate keys, flags, write_eval checks and the one collection point of an evaluation folder."""
    assert {p.name for p in out_dir.iterdir()} == EVAL_FILES
    lines = read_jsonl(out_dir / "records.jsonl")
    assert [r["sample_id"] for r in lines] == [s.sample_id for s in data.samples(split)] and len(lines) == n
    assert first_keys(out_dir / "records.jsonl") == [*records.EVAL_RECORD_FIELDS[:14],
                                                     *(f"gate_{j}" for j in range(k)), "flags", "gen_token_ids"]
    for line, s in zip(lines, data.samples(split)):
        assert line["flags"] == candidate_flags(tuple(line["cand_ids"]), s.target, data.catalog)
        assert list(line["flags"]) == list(FLAG_NAMES)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_total"] == n and metrics["meta"]["n_batches"] == n_batches
    gates = np.array([[line[f"gate_{j}"] for j in range(k)] for line in lines], dtype=np.float64)
    assert np.array_equal(gates.astype(np.float32).astype(np.float64), gates)        # float32 values, exact
    npz = records.load_npz(out_dir / "representations.npz")
    assert npz.meta["source"] == "evaluate" and npz.sample_ids == tuple(r["sample_id"] for r in lines)
    assert np.array_equal(bits(npz.omega), bits(gates.astype(np.float32)))
    assert np.array_equal(bits(npz.omega), bits(result.omega)) and np.array_equal(bits(npz.z), bits(result.z))
    if gate_record is not None:                                                     # GateRecord = records = NPZ
        assert gate_record.sample_ids == npz.sample_ids
        assert np.array_equal(bits(gate_record.omega.numpy()), bits(npz.omega))
        assert np.array_equal(bits(gate_record.z.numpy()), bits(npz.z))
    back = evaluate.load_eval_result(out_dir, num_experts=k)                         # write_eval output reads back
    assert back.metrics == result.metrics and back.records == result.records
    return metrics


@pytest.mark.slow
@pytest.mark.criterion("M11-10")
@pytest.mark.parametrize("split", ["val", "test"])
def test_full_split_eval_tiny(split, tiny_model, profile_cfg, full_data, bundle, templates, tmp_path):
    """Whole split (94 / 95) with the tiny model: three files, rows in row order, K gate keys, flags = M2
    candidate_flags, M12 write_eval checks, GateRecord = records = NPZ (bitwise), 12 gate calls,
    candidates_sha256 and template_counts of the design (M2 6.6.5, M4 8.6.3)."""
    cfg = profile_cfg("smoke_cpu", FULL, NO_BI)
    model = tiny_model()
    n = evaluate.FULL_SPLIT_SIZE[split]
    with capture_gate_record(model) as got:
        result = evaluate.evaluate_split(model, bundle, full_data, split, cfg, new_run_dir(tmp_path), ckpt_tag="epoch0",
                                         templates=templates)
    assert len(got) == 1 and got[0].gate_calls == 12
    metrics = check_eval_folder(result.out_dir, result, got[0], split=split, n=n, n_batches=12, data=full_data)
    meta = metrics["meta"]
    assert meta["gate_calls"] == 12 and meta["candidates_sha256"] == SPLIT_SHA256[split]
    assert meta["torch_num_threads"] == torch.get_num_threads()       # 15.6.10: CPU records bitwise at equal counts
    assert meta["template_counts"] == TEMPLATE_COUNTS[split]
    assert (metrics["split"], metrics["epoch"], metrics["ckpt_tag"]) == (split, 0, "epoch0")
    assert list(metrics["aux_protocols"]) == ([] if split == "val" else ["drop_last_bs4", "drop_last_bs8"])
    assert set(metrics["supplementary"]) == {"wilson95", "flags", "generation"}
    assert sum(metrics["supplementary"]["flags"]["counts"].values()) >= metrics["supplementary"]["flags"]["any"] > 0
    assert result.out_dir == evaluate.eval_dir(result.out_dir.parents[2], "epoch0", split)


@pytest.mark.slow
@pytest.mark.criterion("M11-11")
def test_eval_deterministic(tiny_model, profile_cfg, full_data, bundle, templates, tmp_path):
    """The validation evaluation of #10 twice, in new run folders, with 0 and 2 DataLoader workers: the
    records.jsonl and representations.npz sha256 are the same (1.3, D16; worker count irrelevant, M5)."""
    model = tiny_model()
    perturb_trainable(model, "m11_deterministic")
    hashes = []
    for workers, name in ((0, "a"), (2, "b")):
        cfg = profile_cfg("smoke_cpu", FULL, NO_BI, f"batching.num_workers={workers}")
        result = evaluate.evaluate_split(model, bundle, full_data, "val", cfg, new_run_dir(tmp_path / name),
                                         ckpt_tag="epoch3", templates=templates)
        hashes.append({f: run.content_sha256(result.out_dir / f) for f in ("records.jsonl", "representations.npz")})
    assert hashes[0] == hashes[1]


@pytest.mark.slow
@pytest.mark.criterion("M11-12")
def test_batch_invariance_tiny(tiny_model, profile_cfg, full_data, bundle, templates, tmp_path):
    """eval.batch_invariance_check = true: batch_invariance.json reports 94 of 94 equal token sequences (CPU fp32,
    design measurement 95/95 on a plain tiny LLaMA), equal first lines and judgements; main results unchanged."""
    cfg = profile_cfg("smoke_cpu", FULL)
    assert cfg.eval.batch_invariance_check is True and cfg.eval.batch_size == 8
    model = tiny_model()
    perturb_trainable(model, "m11_batch_invariance")
    result = evaluate.evaluate_split(model, bundle, full_data, "val", cfg, new_run_dir(tmp_path), ckpt_tag="epoch0",
                                     templates=templates)
    report = json.loads((result.out_dir / "batch_invariance.json").read_text(encoding="utf-8"))
    assert report["batch_sizes"] == [8, 1] and report["n"] == 94
    assert (report["tokens_equal"], report["first_line_equal"], report["judgement_equal"]) == (94, 94, 94)
    assert report["differences"] == [] and report["n_batches_batch1"] == 94 and report["gate_calls_batch1"] == 94
    assert report["metrics_batch1"] == dataclasses.asdict(result.metrics)
    for key in ("max_abs_diff_omega", "max_abs_diff_z"):       # recorded without a requirement (15.6.10)
        assert isinstance(report[key], float) and np.isfinite(report[key])
    assert 0 <= report["gates_equal"] <= 94 and report["records_equal"] <= report["tokens_equal"]
    metrics = json.loads((result.out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["meta"]["gate_calls"] == 12 and metrics["meta"]["n_batches"] == 12
    assert {p.name for p in result.out_dir.iterdir()} == EVAL_FILES | {"batch_invariance.json"}


@pytest.mark.criterion("M11-12")
def test_compare_records():
    """compare_records: equal records give no difference; a changed token at step 1 is listed with that step and
    both judgements; gate values differing in the last bit are counted (not listed); other samples -> EvalError."""
    a = [synthetic_record("val", i, valid=i % 2 == 0, correct=i % 4 == 0) for i in range(6)]
    a = [dataclasses.replace(r, gen_token_ids=(11, 12, 13, 2)) for r in a]
    same = evaluate.compare_records(a, list(a))
    assert (same["n"], same["records_equal"], same["tokens_equal"], same["differences"]) == (6, 6, 6, [])
    b = list(a)
    b[2] = dataclasses.replace(b[2], gen_token_ids=(11, 99, 13, 2), first_line="other", valid=False, correct=False,
                               matched=())
    gates = list(b[3].gates)
    gates[0] = float(np.nextafter(np.float32(gates[0]), np.float32(1.0)))
    b[3] = dataclasses.replace(b[3], gates=tuple(gates))
    diff = evaluate.compare_records(a, b)
    assert (diff["records_equal"], diff["tokens_equal"], diff["first_line_equal"], diff["judgement_equal"],
            diff["gates_equal"]) == (4, 5, 5, 5, 5)
    assert [d["sample_id"] for d in diff["differences"]] == ["val:2"]
    entry = diff["differences"][0]
    assert entry["first_diff_step"] == 1 and entry["judgement_a"] == [True, False] and entry["judgement_b"] == [False, False]
    assert 0 < diff["max_abs_diff_omega"] < 1e-7
    with pytest.raises(EvalError):
        evaluate.compare_records(a, a[::-1])


# ----------------------------------------------------------------------------------------------- #13, #14
@pytest.mark.criterion("M11-13")
def test_rng_and_mode_restored(tiny_model, profile_cfg, smoke_data, bundle, templates, tmp_path):
    """torch CPU, Python random and numpy global states are the same before and after evaluate_split (the
    DataLoader draw is undone by fork_rng; nothing else draws); a model in train mode comes back in train mode
    and generates in eval mode."""
    cfg = profile_cfg("smoke_cpu", NO_BI)
    model = tiny_model()
    model.train()
    disturb_global_rng("m11_rng")
    before = rng_states()
    modes = []
    original = model.generate
    model.generate = lambda batch: (modes.append(model.training), original(batch))[1]
    try:
        evaluate.evaluate_split(model, bundle, smoke_data, "val", cfg, new_run_dir(tmp_path), ckpt_tag="epoch0",
                                templates=templates)
    finally:
        del model.generate
    assert_rng_states_equal(before, rng_states())
    assert modes == [False, False] and model.training and model.sasrec_is_eval()
    model.eval()
    evaluate.evaluate_split(model, bundle, smoke_data, "test", cfg, new_run_dir(tmp_path, "m11b"), ckpt_tag="init",
                            templates=templates)
    assert not model.training
    assert_rng_states_equal(before, rng_states())


@pytest.mark.criterion("M11-14")
def test_exception_cleanup(tiny_model, profile_cfg, smoke_data, bundle, templates, tmp_path):
    """An exception at the third batch: recorder emptied (start_recording works again), mode and random state
    restored, no metrics.json. Evaluating again moves the old folder to val.incomplete-<UTC>; evaluating into the
    completed folder raises EvalError."""
    cfg = profile_cfg("smoke_cpu", "eval.batch_size=4", NO_BI)
    model = tiny_model()
    model.train()
    run_dir = new_run_dir(tmp_path)
    disturb_global_rng("m11_exception")
    before = rng_states()
    calls = []
    original = model.generate

    def failing(batch):
        calls.append(len(batch.sample_ids))
        if len(calls) == 3:
            raise RuntimeError("injected failure at batch 3")
        return original(batch)

    model.generate = failing
    try:
        with pytest.raises(RuntimeError, match="injected"):
            evaluate.evaluate_split(model, bundle, smoke_data, "val", cfg, run_dir, ckpt_tag="epoch1",
                                    templates=templates)
    finally:
        del model.generate
    assert calls == [4, 4, 2]
    out = evaluate.eval_dir(run_dir, "epoch1", "val")
    assert out.is_dir() and list(out.iterdir()) == []                       # nothing written
    assert model.training and model.sasrec_is_eval()
    assert_rng_states_equal(before, rng_states())
    model.omega_source.start_recording("val")                                # recorder was discarded
    model.omega_source.discard_recording()
    result = evaluate.evaluate_split(model, bundle, smoke_data, "val", cfg, run_dir, ckpt_tag="epoch1",
                                     templates=templates)
    kept = [p.name for p in out.parent.iterdir() if p.name.startswith("val.incomplete-")]
    assert len(kept) == 1 and (result.out_dir / "metrics.json").is_file()
    sha = run.content_sha256(result.out_dir / "records.jsonl")
    with pytest.raises(EvalError, match="never overwritten"):
        evaluate.evaluate_split(model, bundle, smoke_data, "val", cfg, run_dir, ckpt_tag="epoch1", templates=templates)
    assert run.content_sha256(result.out_dir / "records.jsonl") == sha
    assert sorted(p.name for p in out.parent.iterdir()) == sorted(["val", *kept])


# ----------------------------------------------------------------------------------------------- #15
@pytest.mark.criterion("M11-15")
@pytest.mark.parametrize("variant", ["text_only", "cluster_hard", "none"])
def test_eval_variants(variant, tiny_model, profile_cfg, smoke_data, bundle, templates, fake_hard_gate, tmp_path):
    """text_only (E3): records hybrid false, no projector forward; cluster_hard: one-hot omega of the mapping, no
    gate call, one lookup per sample; none (K = 1): omega = 1, key gate_0 only, no gate call."""
    table = {s.sample_id: s.row % K for split in ("train", "val", "test") for s in smoke_data.samples(split)}
    gate = None
    if variant == "text_only":
        sets, k = ("prompt.eval_prompt=text_only",), K
    elif variant == "cluster_hard":
        sets, k = CLUSTER_HARD, K
        gate = fake_hard_gate(table, K)
    else:
        sets, k = K1, 1
    cfg = profile_cfg("smoke_cpu", *sets, NO_BI)
    model = tiny_model(*sets, hard_gate=gate)
    projector_calls = []
    handle = model.projector.register_forward_hook(lambda module, args, output: projector_calls.append(1))
    try:
        result = evaluate.evaluate_split(model, bundle, smoke_data, "val", cfg, new_run_dir(tmp_path),
                                         ckpt_tag="epoch0", templates=templates)
    finally:
        handle.remove()
    lines = read_jsonl(result.out_dir / "records.jsonl")
    metrics = check_eval_folder(result.out_dir, result, None, split="val", n=10, n_batches=2, data=smoke_data, k=k)
    assert first_keys(result.out_dir / "records.jsonl")[14:14 + k] == [f"gate_{j}" for j in range(k)]
    assert result.metrics.n_total == 10 and metrics["meta"]["routing_mode"] == cfg.router.routing_mode
    if variant == "text_only":
        assert all(line["hybrid"] is False for line in lines) and projector_calls == []
        assert metrics["meta"]["gate_calls"] == 2 and metrics["meta"]["eval_prompt"] == "text_only"
        return
    assert projector_calls == [1, 1]                                    # hybrid: one projector call per batch
    assert metrics["meta"]["gate_calls"] == 0
    if variant == "cluster_hard":
        for line in lines:
            assert [line[f"gate_{j}"] for j in range(K)] == [1.0 if j == table[line["sample_id"]] else 0.0
                                                             for j in range(K)]
        assert gate.lookups == 10
    else:
        assert all(line["gate_0"] == 1.0 for line in lines) and result.omega.shape == (10, 1)


# ----------------------------------------------------------------------------------------------- #17
@pytest.mark.criterion("M11-17")
def test_load_eval_result(tiny_model, profile_cfg, smoke_data, bundle, templates, tmp_path):
    """load_eval_result gives back the EvalResult of evaluate_split (metrics recomputed = metrics.json, records,
    omega and z); a removed line, a changed ``valid``, an incomplete folder or another K fail."""
    cfg = profile_cfg("smoke_cpu", NO_BI)
    model = tiny_model()
    perturb_trainable(model, "m11_load")
    run_dir = new_run_dir(tmp_path)
    for split in ("val", "test"):
        result = evaluate.evaluate_split(model, bundle, smoke_data, split, cfg, run_dir, ckpt_tag="epoch2",
                                         templates=templates)
        back = evaluate.load_eval_result(result.out_dir, num_experts=K)
        assert (back.split, back.ckpt_tag, back.metrics, back.aux) == (split, "epoch2", result.metrics, result.aux)
        assert back.records == result.records and back.out_dir == result.out_dir
        assert np.array_equal(bits(back.omega), bits(result.omega)) and np.array_equal(bits(back.z), bits(result.z))
        assert back.seconds == json.loads((result.out_dir / "metrics.json").read_text(encoding="utf-8"))["meta"]["seconds"]
    source = evaluate.eval_dir(run_dir, "epoch2", "val")

    def tampered(name: str, edit) -> Path:
        target = tmp_path / name / "eval" / "epoch2" / "val"
        shutil.copytree(source, target)
        path = target / "records.jsonl"
        path.write_bytes(edit(path.read_bytes().decode("utf-8")).encode("utf-8"))
        return target

    removed = tampered("removed", lambda text: "".join(text.splitlines(keepends=True)[:-1]))
    flip = lambda text: text.replace('"valid":false', '"valid":true', 1) if '"valid":false' in text \
        else text.replace('"valid":true', '"valid":false', 1)                                   # noqa: E731
    flipped = tampered("flipped", flip)
    for folder in (removed, flipped):
        with pytest.raises(EvalError):
            evaluate.load_eval_result(folder, num_experts=K)
    incomplete = tmp_path / "incomplete" / "eval" / "epoch2" / "val"
    shutil.copytree(source, incomplete)
    (incomplete / "metrics.json").unlink()
    with pytest.raises(EvalError, match="not complete"):
        evaluate.load_eval_result(incomplete, num_experts=K)
    with pytest.raises(EvalError):
        evaluate.load_eval_result(source, num_experts=1)


# ----------------------------------------------------------------------------------------------- M7-12
@pytest.mark.criterion("M7-12")
def test_gate_record_single_collection_point(tiny_model, cpu_cfg, smoke_data, bundle, templates):
    """The GateRecord of an evaluation equals, bit for bit, the omega and z of every GenerationOutput, and the
    output of a Router rebuilt without the LLM from the checkpoint gate key and M3 z of the same batches."""
    from ilora.data import HISTORY_LEN, PAD_ID, pad_histories
    from ilora.sasrec import SASRecEncoder

    model = tiny_model()
    perturb_trainable(model, "m7_collection_point")
    with capture_generate(model) as outputs:
        gen = evaluate.generate_records(model, bundle, smoke_data, "val", cpu_cfg, batch_size=4, templates=templates,
                                        record_gates=True)
    rec = gen.gate_record
    assert rec is not None and rec.sample_ids == tuple(s.sample_id for s in smoke_data.val) and rec.gate_calls == 3
    assert torch.equal(rec.omega.view(torch.int32), torch.cat([o.omega for o in outputs]).view(torch.int32))
    assert torch.equal(rec.z.view(torch.int32), torch.cat([o.z for o in outputs]).view(torch.int32))
    assert np.array_equal(bits(gen.omega), bits(rec.omega.numpy()))
    assert np.array_equal(bits(np.array([r.gates for r in gen.records])), bits(rec.omega.numpy()))
    router = records.export_router(cpu_cfg, model.trainable_state(), z_dim=64)       # omega_source.* keys only
    encoder = SASRecEncoder.from_config(cpu_cfg, smoke_data.catalog, HISTORY_LEN)     # M3 on CPU, no LLM
    omegas, zs = [], []
    samples = smoke_data.val
    with torch.no_grad():
        for i in range(0, len(samples), 4):
            group = samples[i:i + 4]
            hist_ids, len_seq = pad_histories([s.history for s in group], PAD_ID, HISTORY_LEN)
            z = encoder(hist_ids, len_seq)
            zs.append(z)
            omegas.append(router(z, [s.sample_id for s in group]))
    assert torch.equal(torch.cat(zs).view(torch.int32), rec.z.view(torch.int32))
    assert torch.equal(torch.cat(omegas).view(torch.int32), rec.omega.view(torch.int32))


# ----------------------------------------------------------------------------------------------- #16
class TableGate:
    """``HardGate`` protocol (M7 11.6.1) for the cluster_hard runs: dict lookup -> one-hot, lookups counted."""

    def __init__(self, table, num_experts: int) -> None:
        self.table, self.num_experts, self.lookups = dict(table), num_experts, 0

    def __call__(self, sample_ids, device):
        idx = torch.tensor([self.table[s] for s in sample_ids], dtype=torch.int64)
        self.lookups += len(sample_ids)
        omega = torch.zeros(len(sample_ids), self.num_experts, dtype=torch.float32).scatter_(1, idx[:, None], 1.0)
        return omega.to(device)


class TrainStub:
    """Stand-in ``ilora.train`` with the M10 14.6.6 reading rules for the synthetic checkpoints: COMPLETE marker,
    generated_frozen.* compared bit for bit with the rebuilt model, the rest loaded with ``strict=True``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def read_trainable_state(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        if not (ckpt_dir / "COMPLETE").is_file():
            raise RuntimeError(f"{ckpt_dir}: no COMPLETE marker")
        self.calls.append(("read", ckpt_dir.name))
        return torch.load(ckpt_dir / "trainable.pt")

    def load_model_state(self, ckpt_dir, model):
        state = dict(self.read_trainable_state(ckpt_dir))
        for key, value in model.generated_frozen_state().items():
            saved = state.pop(key)
            if saved.dtype != value.dtype or saved.shape != value.shape or not torch.equal(
                    saved.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)):
                raise RuntimeError(f"{key} differs from the rebuilt model's generated_frozen rows (M10 14.6.6)")
        model.load_trainable_state(state, strict=True)
        self.calls.append(("load", Path(ckpt_dir).name))
        return run.content_sha256(Path(ckpt_dir) / "trainable.pt")

    def install(self, monkeypatch) -> None:
        module = types.ModuleType("ilora.train")
        module.read_trainable_state = self.read_trainable_state
        module.load_model_state = self.load_model_state
        monkeypatch.setitem(sys.modules, "ilora.train", module)


class ClusterStub:
    """Stand-in ``ilora.cluster`` recording the M15 call order and arguments (19.6.1 call shapes); with
    ``allowed=False`` any call fails (dynamic runs never call M15, [SPEC §22])."""

    def __init__(self, table, *, allowed: bool) -> None:
        self.table, self.allowed, self.calls = dict(table), allowed, []

    def _call(self, name: str, **details) -> None:
        if not self.allowed:
            raise AssertionError(f"M15 {name} called by a dynamic run [SPEC §22]")
        self.calls.append((name, details))

    def load_cluster_artifacts(self, artifact_dir, *, expected_ids, sasrec_sha256, data_sha256, ccfg):
        self._call("load_cluster_artifacts", artifact_dir=Path(artifact_dir), expected_ids=list(expected_ids),
                   sasrec_sha256=sasrec_sha256, data_sha256=dict(data_sha256), ccfg=ccfg)
        return SimpleNamespace(dir=Path(artifact_dir), n_clusters=K, assignments=dict(self.table))

    def parse_expert_mapping(self, text, *, n_clusters, num_experts):
        self._call("parse_expert_mapping", text=text, n_clusters=n_clusters, num_experts=num_experts)
        return tuple(range(n_clusters))

    def ClusterHardGate(self, artifacts, mapping, num_experts):        # noqa: N802 - M15 class name
        self._call("ClusterHardGate", mapping=tuple(mapping), num_experts=num_experts)
        return TableGate(artifacts.assignments, num_experts)

    def verify_live_encoder(self, artifacts, encoder, samples, *, device, n, atol):
        self._call("verify_live_encoder", device=str(device), n=n, atol=atol, n_samples=len(samples))
        return {"passed": True, "n": n, "max_abs_diff": 0.0}

    def install(self, monkeypatch) -> None:
        module = types.ModuleType("ilora.cluster")
        for name in ("load_cluster_artifacts", "parse_expert_mapping", "ClusterHardGate", "verify_live_encoder"):
            setattr(module, name, getattr(self, name))
        monkeypatch.setitem(sys.modules, "ilora.cluster", module)


def build_source_run(root: Path, name: str, cfg, *, data, bundle, templates, repo_root, tokenizer_dir,
                     hard_gate=None, epochs: int, best: int, last: int) -> Path:
    """A finished training run as ``main.py eval`` reads it (15.6.11): config.yaml, meta.json (inputs, assets),
    checkpoints/epoch{e}/{trainable.pt, COMPLETE} (trainable_state + generated_frozen_state, M9 13.6.8),
    eval/selection.json (M10 14.6.7 keys) and eval/epoch{e}/val, eval/epoch{best,last}/test from evaluate_split."""
    from ilora.commands import asset_record
    from ilora.config import config_hash, dump_config
    from ilora.model import build_model

    source = run.create_run_dir(name, root=root)
    dump_config(cfg, source / "config.yaml")
    run.start_meta(source, profile=name, argv=[name], config_hash=config_hash(cfg), overrides={}, seed=cfg.run.seed,
                   extra={"command": "train", "inputs": run.input_hashes(repo_root),
                          "assets": asset_record(cfg, repo_root=repo_root, llm_path=None, tokenizer_dir=tokenizer_dir)})
    model = build_model(cfg, bundle, data.catalog, llm_path=None, hard_gate=hard_gate)
    states = {}
    for epoch in range(epochs):
        perturb_trainable(model, f"{name}_{epoch}")
        ckpt = source / "checkpoints" / f"epoch{epoch}"
        ckpt.mkdir()
        torch.save({**model.trainable_state(), **model.generated_frozen_state()}, ckpt / "trainable.pt")
        (ckpt / "COMPLETE").write_text("{}", encoding="utf-8")
        states[epoch] = model.trainable_state()
        evaluate.evaluate_split(model, bundle, data, "val", cfg, source, ckpt_tag=f"epoch{epoch}", templates=templates)
    (source / "eval" / "selection.json").write_text(json.dumps(
        {"rule": "largest validation n_correct, later epoch on ties", "val": {}, "best_val_epoch": best,
         "last_epoch": last, "test_dirs": {"best_val": f"eval/epoch{best}/test", "last": f"eval/epoch{last}/test"},
         "primary": "best_val"}), encoding="utf-8")
    for epoch in sorted({best, last}):
        model.load_trainable_state(states[epoch])
        evaluate.evaluate_split(model, bundle, data, "test", cfg, source, ckpt_tag=f"epoch{epoch}", templates=templates)
    run.finish_meta(source, "completed")
    return source


@pytest.fixture(scope="module")
def eval_sources(cpu_cfg, smoke_data, bundle, templates, repo_root, tokenizer_dir, sasrec_state_path,
                 tmp_path_factory):
    """Two synthetic training runs of smoke_cpu (eval.batch_invariance_check false): dynamic (2 epochs, best 0,
    last 1) and cluster_hard (1 epoch, a sample -> cluster table, identity mapping)."""
    from ilora.sections import RouterConfig

    root = tmp_path_factory.mktemp("m11_sources")
    base = dataclasses.replace(cpu_cfg, eval=dataclasses.replace(cpu_cfg.eval, batch_invariance_check=False))
    common = dict(data=smoke_data, bundle=bundle, templates=templates, repo_root=repo_root, tokenizer_dir=tokenizer_dir)
    dynamic = build_source_run(root, "m11_source_dynamic", base, epochs=2, best=0, last=1, **common)
    table = {s.sample_id: (s.row * 7) % K for split in ("train", "val", "test") for s in smoke_data.samples(split)}
    artifact_dir = (root / "cluster_run" / "cluster").as_posix()
    hard = dataclasses.replace(base, router=RouterConfig(routing_mode="cluster_hard"),
                               cluster=dataclasses.replace(base.cluster, artifact_dir=artifact_dir))
    cluster = build_source_run(root, "m11_source_cluster", hard, hard_gate=TableGate(table, K), epochs=1, best=0,
                               last=0, **common)
    return SimpleNamespace(dynamic=dynamic, cluster=cluster, table=table, artifact_dir=artifact_dir)


EVAL_COMMAND_CASES = ["same_config", "allowed_differences", "batch_size", "init_and_epoch", "forbidden_lora",
                      "forbidden_router", "forbidden_seed", "input_hash", "test_with_epoch", "frozen_bit",
                      "cluster_hard"]


@pytest.mark.slow
@pytest.mark.criterion("M11-16")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
@pytest.mark.parametrize("case", EVAL_COMMAND_CASES)
def test_eval_command(case, eval_sources, profile_cfg, tokenizer_dir, repo_root, sasrec_state_path, tiny_model,
                      smoke_data, bundle, templates, monkeypatch, tmp_path):
    """``main.py eval`` through M0 ``run_command`` (15.6.11, 15.8 #16): allowed differences pass, lora/router/seed
    differences and changed input hashes fail; best_val/last follow eval/selection.json; test with epoch1 is a
    ConfigError; init evaluates the seed-built model; summary.json has identical / not_comparable (with the keys)
    / absent; checkpoints are read only through M10 load_model_state (one generated_frozen bit fails);
    cluster_hard follows the M15 order and gives one-hot omega, dynamic never calls M15."""
    import ilora.model as model_module
    from ilora import commands
    from ilora.config import ConfigError

    train = TrainStub()
    train.install(monkeypatch)
    cluster = ClusterStub(eval_sources.table, allowed=case == "cluster_hard")
    cluster.install(monkeypatch)
    original_build = model_module.build_model

    def build_spy(*args, **kwargs):
        cluster.calls.append(("build_model", {"hard_gate": kwargs.get("hard_gate")}))
        return original_build(*args, **kwargs)

    monkeypatch.setattr(model_module, "build_model", build_spy)
    source = eval_sources.cluster if case == "cluster_hard" else eval_sources.dynamic
    if case in ("input_hash", "frozen_bit"):
        source = Path(shutil.copytree(source, tmp_path / "source_copy"))
        if case == "input_hash":
            meta = run.read_meta(source)
            meta["inputs"]["prompt/movie.txt"] = "0" * 64
            run.write_json_atomic(source / "meta.json", meta)
        else:
            path = source / "checkpoints" / "epoch1" / "trainable.pt"
            state = torch.load(path)
            rows = state["generated_frozen.embed_rows"].clone()
            rows.view(torch.int32)[0, 0] ^= 1                              # one bit of one new token row
            state["generated_frozen.embed_rows"] = rows
            torch.save(state, path)
    base = [f"eval.source_run={json.dumps(source.as_posix())}", NO_BI]
    if case == "cluster_hard":
        base += ["router.routing_mode=cluster_hard", f"cluster.artifact_dir={json.dumps(eval_sources.artifact_dir)}"]

    def run_eval(*sets: str) -> Path:
        texts = [*base, *sets]
        return commands.run_command("eval", profile_cfg("smoke_cpu", *texts), texts, profile="m11_eval",
                                    root=tmp_path / "runs", argv=["test_eval_command", case], tokenizer_dir=tokenizer_dir)

    def only_run() -> Path:
        (folder,) = list((tmp_path / "runs").iterdir())
        return folder

    def summary_of(run_dir: Path) -> dict:
        return json.loads((run_dir / "eval" / "summary.json").read_text(encoding="utf-8"))

    loads = lambda: [name for kind, name in train.calls if kind == "load"]      # noqa: E731
    if case == "test_with_epoch":
        with pytest.raises(ConfigError):
            profile_cfg("smoke_cpu", *base, "eval.splits=[test]", "eval.checkpoints=[epoch1]")
        cfg = profile_cfg("smoke_cpu", *base)
        bad = dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, splits=("test",), checkpoints=("epoch1",)))
        with pytest.raises(ConfigError):
            evaluate.run_eval_command(bad, new_run_dir(tmp_path), repo_root=repo_root, llm_path=None)
        assert train.calls == [] and cluster.calls == []
        return
    if case in ("forbidden_lora", "forbidden_router", "forbidden_seed", "input_hash"):
        sets, keys = {"forbidden_lora": (("lora.dropout=0.2",), ["lora.dropout"]),
                      "forbidden_router": (("router.routing_mode=cluster_hard", "cluster.artifact_dir=x/cluster"),
                                           ["router.routing_mode", "cluster.artifact_dir"]),
                      "forbidden_seed": (("run.seed=1",), ["run.seed"]),
                      "input_hash": ((), ["prompt/movie.txt"])}[case]
        with pytest.raises(EvalError) as info:
            run_eval(*sets)
        assert all(key in str(info.value) for key in keys), str(info.value)
        assert run.read_meta(only_run())["status"] == "failed"
        assert train.calls == [] and cluster.calls == []                  # nothing built, loaded or M15-called
        return
    if case == "frozen_bit":
        with pytest.raises(RuntimeError, match="generated_frozen"):
            run_eval("eval.checkpoints=[last]", "eval.splits=[val]")
        folder = only_run()
        assert run.read_meta(folder)["status"] == "failed" and loads() == []
        assert not (folder / "eval" / "epoch1" / "val" / "metrics.json").exists()
        return
    if case == "same_config":
        run_dir = run_eval()
        summary = summary_of(run_dir)
        assert list(summary["results"]) == ["epoch0/val", "epoch0/test", "epoch1/val", "epoch1/test"]
        assert [item["reproduction"]["status"] for item in summary["results"].values()] == ["identical"] * 4
        assert [(c["ckpt_tag"], c["tags"]) for c in summary["checkpoints"]] == [("epoch0", ["best_val"]),
                                                                              ("epoch1", ["last"])]
        assert loads() == ["epoch0", "epoch1"] and [name for name, _ in cluster.calls] == ["build_model"]
        meta = run.read_meta(run_dir)
        assert meta["status"] == "completed" and meta["source_run"] == str(source)
        for key, item in summary["results"].items():
            for name, sha in item["files"].items():
                assert run.content_sha256(run_dir / "eval" / key / name) == sha
        assert "torch.load" not in EVALUATE_PY.read_text(encoding="utf-8")     # only M10 load_model_state reads
        return
    if case == "allowed_differences":
        run_dir = run_eval("prompt.eval_prompt=text_only", "batching.num_workers=1", "eval.checkpoints=[last]",
                           "eval.splits=[val]")
        summary = summary_of(run_dir)
        (item,) = summary["results"].values()
        assert (item["ckpt_tag"], item["split"], item["tags"]) == ("epoch1", "val", ["last"])
        assert item["reproduction"]["status"] == "not_comparable"
        assert item["reproduction"]["differing_keys"] == ["prompt.eval_prompt"]
        allowed = " ".join(summary["allowed_differences"])
        assert all(key in allowed for key in ("prompt.eval_prompt", "batching.num_workers", "eval.checkpoints",
                                              "eval.splits", "eval.source_run"))
        assert all(line["hybrid"] is False for line in read_jsonl(run_dir / "eval" / "epoch1" / "val" / "records.jsonl"))
        return
    if case == "batch_size":
        run_dir = run_eval("eval.batch_size=4", "eval.checkpoints=[best_val]", "eval.splits=[val]")
        (item,) = summary_of(run_dir)["results"].values()
        assert item["ckpt_tag"] == "epoch0" and item["reproduction"]["status"] == "not_comparable"
        assert item["reproduction"]["differing_keys"] == ["eval.batch_size"]
        metrics = json.loads((run_dir / "eval" / "epoch0" / "val" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["meta"]["n_batches"] == 3 and loads() == ["epoch0"]
        return
    if case == "init_and_epoch":
        run_dir = run_eval("eval.checkpoints=[epoch1,init]", "eval.splits=[val]")
        summary = summary_of(run_dir)
        assert list(summary["results"]) == ["init/val", "epoch1/val"] and loads() == ["epoch1"]
        assert summary["results"]["init/val"]["reproduction"]["status"] == "absent"
        assert summary["results"]["epoch1/val"]["reproduction"]["status"] == "identical"
        init = run_dir / "eval" / "init" / "val"
        metrics = json.loads((init / "metrics.json").read_text(encoding="utf-8"))
        assert (metrics["epoch"], metrics["ckpt_tag"], metrics["meta"]["ckpt_meta"]["checkpoint_dir"]) == (None, "init", None)
        fresh = evaluate.evaluate_split(tiny_model(), bundle, smoke_data, "val", profile_cfg("smoke_cpu", NO_BI),
                                        new_run_dir(tmp_path, "fresh"), ckpt_tag="init", templates=templates)
        assert run.content_sha256(init / "records.jsonl") == run.content_sha256(fresh.out_dir / "records.jsonl")
        return
    run_dir = run_eval()                                                           # cluster_hard
    assert [name for name, _ in cluster.calls] == ["load_cluster_artifacts", "parse_expert_mapping",
                                                   "ClusterHardGate", "build_model", "verify_live_encoder"]
    details = dict(cluster.calls)
    load = details["load_cluster_artifacts"]
    assert load["artifact_dir"] == Path(eval_sources.artifact_dir)
    assert load["expected_ids"] == [s.sample_id for split in ("train", "val", "test") for s in smoke_data.samples(split)]
    assert len(load["expected_ids"]) == 176 and load["data_sha256"] == run.input_hashes(repo_root)
    assert load["sasrec_sha256"] == run.content_sha256(sasrec_state_path)
    assert details["parse_expert_mapping"] == {"text": "0:0,1:1,2:2,3:3", "n_clusters": K, "num_experts": K}
    assert isinstance(details["build_model"]["hard_gate"], TableGate)
    assert (details["verify_live_encoder"]["n"], details["verify_live_encoder"]["device"]) == (64, "cpu")
    summary = summary_of(run_dir)
    assert summary["cluster_live_check"]["passed"] is True and loads() == ["epoch0"]
    assert [item["reproduction"]["status"] for item in summary["results"].values()] == ["identical", "identical"]
    for key in ("epoch0/val", "epoch0/test"):
        folder = run_dir / "eval" / key
        assert json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["meta"]["gate_calls"] == 0
        for line in read_jsonl(folder / "records.jsonl"):
            expert = eval_sources.table[line["sample_id"]]
            assert [line[f"gate_{j}"] for j in range(K)] == [1.0 if j == expert else 0.0 for j in range(K)]
