"""M2 후보 생성 시험 (DESIGN.md §6.8 #1-#8; §17.6.9 M2 표). #9 is ``vessl/test_vessl_env.py``;
the worker-count part of #2 is ``test_workers.py``.

Golden values are the §6.6.5-§6.6.6 numbers (seed 1234, numpy 1.24.2).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import pickle
from collections import Counter

import numpy as np
import pytest
from scipy.stats import chisquare

import refs_upstream
from ilora import candidates, data
from ilora.candidates import (FLAG_NAMES, NUM_CANDIDATES, CandidateError, candidate_flags, candidate_key,
                              candidates_for, check_candidates, match_norm, negative_pool)
from ilora.config import ConfigError
from ilora.data import NUM_ITEMS, PAD_ID, ItemCatalog, Sample
from ilora.sections import CandidatesConfig
from ilora.seeding import derive_seed

GOLDEN_VAL0 = (980, 156, 321, 1240, 892, 1299, 448, 1094, 1502, 728, 1598, 208, 1042, 1386, 902, 1282, 1029, 378,
               515, 554)
GOLDEN_TEST0 = (1625, 1239, 800, 944, 769, 316, 840, 381, 1450, 228, 1367, 1393, 634, 3, 1497, 365, 101, 832, 1160,
                638)
SPLIT_SHA256 = {"val": "9a07320651cadd0d049efc8644cb9a6103a78bdc2b45f2bd793ae204a545f01a",
                "test": "1ffdb6499d3a316fb8cf94c322f09e5af509520f8cdcc6574634b99a7a30bef3"}
CC20 = CandidatesConfig(num_candidates=20)


def _cands(sample: Sample, epoch=None, seed: int = 1234) -> tuple[int, ...]:
    return candidates_for(sample, main_seed=seed, epoch=epoch, cfg=CC20)


# ----------------------------------------------------------------------------
# M2-1 contract
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M2-1")
def test_candidate_contract(full_data, paper_cfg):
    assert paper_cfg.candidates.num_candidates == NUM_CANDIDATES == 20
    cases = [(s, None) for s in (*full_data.val, *full_data.test)]
    cases += [(s, e) for s in full_data.train[:2000] for e in (0, 1)]
    assert len(cases) == 189 + 4000
    for s, e in cases:
        c = candidates_for(s, main_seed=paper_cfg.run.seed, epoch=e, cfg=paper_cfg.candidates)
        assert isinstance(c, tuple) and len(c) == 20, s.sample_id
        assert all(type(x) is int and 0 <= x <= 1681 for x in c), s.sample_id
        assert len(set(c)) == 20 and c.count(s.target) == 1, s.sample_id
        assert not set(c) & set(s.history), s.sample_id
        check_candidates(c, s, num_items=NUM_ITEMS, num_candidates=20)
    pool = negative_pool(full_data.val[0].history, full_data.val[0].target, NUM_ITEMS)
    assert pool.dtype == np.int64 and len(pool) == 1682 - 10 - 1 and np.all(np.diff(pool) > 0)


CHECK_CASES = {
    "list": lambda c, s: list(c),
    "19_items": lambda c, s: c[:19] if s.target in c[:19] else c[1:],
    "repeat": lambda c, s: (c[0],) + c[:19] if c[0] != s.target else (c[1],) + c[1:],
    "no_target": lambda c, s: tuple(x if x != s.target else next(i for i in range(NUM_ITEMS)
                                                                 if i not in c and i not in s.history) for x in c),
    "history_item": lambda c, s: tuple(s.history[0] if x == next(y for y in c if y != s.target) else x for x in c),
    "bool": lambda c, s: (True,) + c[1:] if c[0] != s.target else (c[0], True) + c[2:],
    "id_1682": lambda c, s: (1682,) + c[1:] if c[0] != s.target else (c[0], 1682) + c[2:],
    "numpy_int": lambda c, s: tuple(np.int64(x) for x in c),
}


@pytest.mark.criterion("M2-1")
@pytest.mark.parametrize("case", list(CHECK_CASES))
def test_check_candidates_rejects(case, full_data):
    s = full_data.val[3]
    good = _cands(s)
    check_candidates(good, s, num_items=NUM_ITEMS, num_candidates=20)
    with pytest.raises(CandidateError, match=s.sample_id):
        check_candidates(CHECK_CASES[case](good, s), s, num_items=NUM_ITEMS, num_candidates=20)


# ----------------------------------------------------------------------------
# M2-2 determinism (worker counts: test_workers.py)
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M2-2")
def test_deterministic_and_epoch_dependent(full_data, paper_cfg):
    seed, cc = paper_cfg.run.seed, paper_cfg.candidates
    s3 = next(s for s in full_data.train if s.row == 3)
    assert candidate_key(s3, 0) == ("candidates", "train", 3, 0)
    assert candidate_key(s3, 1) == ("candidates", "train", 3, 1)
    assert candidate_key(full_data.val[0], None) == ("candidates", "val", 0)
    assert candidate_key(full_data.test[7], None) == ("candidates", "test", 7)
    e0 = candidates_for(s3, main_seed=seed, epoch=0, cfg=cc)
    assert candidates_for(s3, main_seed=seed, epoch=0, cfg=cc) == e0
    assert candidates_for(s3, main_seed=seed, epoch=1, cfg=cc) != e0
    assert candidates_for(s3, main_seed=seed + 1, epoch=0, cfg=cc) != e0
    # independent of call order and of other calls in between (no state, no cache)
    sample_list = list(full_data.val[:30]) + list(full_data.train[:30])
    epochs = [None] * 30 + [2] * 30
    forward = [candidates_for(s, main_seed=seed, epoch=e, cfg=cc) for s, e in zip(sample_list, epochs)]
    backward = [candidates_for(s, main_seed=seed, epoch=e, cfg=cc)
                for s, e in zip(reversed(sample_list), reversed(epochs))]
    assert forward == backward[::-1]
    # CandidateFn (§3.1, §6.6.9): picklable partial, epoch keyword-only
    fn = functools.partial(candidates_for, main_seed=seed, cfg=cc)
    again = pickle.loads(pickle.dumps(fn))
    assert again(s3, epoch=0) == fn(s3, epoch=0) == e0
    assert again(full_data.val[0], epoch=None) == fn(full_data.val[0], epoch=None)
    with pytest.raises(TypeError):
        fn(s3, 0)
    # the result is a function of (history, target, Generator state) only
    rng_a = np.random.Generator(np.random.PCG64(derive_seed(seed, "candidates", "train", 3, 0)))
    assert candidates.draw_candidates(s3.history, s3.target, rng_a, num_items=NUM_ITEMS,
                                      num_candidates=20) == e0


# ----------------------------------------------------------------------------
# M2-3 golden
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M2-3")
def test_golden_rows_and_split_sha256(full_data, paper_cfg):
    assert paper_cfg.run.seed == 1234
    assert derive_seed(1234, "candidates", "val", 0) == 4835709385070275973
    v0, t0 = full_data.val[0], full_data.test[0]
    assert (v0.row, v0.history, v0.target) == (0, (880, 269, 361, 1024, 881, 890, 1242, 1104, 989, 333), 321)
    assert (t0.row, t0.target) == (0, 3)
    assert _cands(v0) == GOLDEN_VAL0
    assert _cands(t0) == GOLDEN_TEST0
    for split, expected in SPLIT_SHA256.items():
        blob = json.dumps([[s.row, list(_cands(s))] for s in full_data.samples(split)], separators=(",", ":"))
        assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == expected, split


@pytest.mark.criterion("M2-3")
def test_self_check_passes(monkeypatch):
    assert np.__version__ == "1.24.2"
    assert candidates.self_check() is None
    real = candidates.draw_candidates
    # a Generator that orders the draw differently (as another numpy version may) is caught
    monkeypatch.setattr(candidates, "draw_candidates", lambda *a, **k: tuple(reversed(real(*a, **k))))
    with pytest.raises(CandidateError, match="numpy"):
        candidates.self_check()


# ----------------------------------------------------------------------------
# M2-4 smoke = full
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M2-4")
def test_smoke_candidates_equal_full(full_data, smoke_data, cpu_cfg, repo_root):
    full_by_id = {s.sample_id: s for split in data.SPLITS for s in full_data.samples(split)}
    # smoke data loaded on its own (smoke_cpu profile, new Sample objects) and the smoke_data fixture
    loaded = data.load_dataset(cpu_cfg, repo_root)
    assert loaded == smoke_data
    n = 0
    for split in ("val", "test"):
        for s in loaded.samples(split):
            f = full_by_id[s.sample_id]
            assert s == f
            assert _cands(s) == _cands(f), s.sample_id
            n += 1
    assert n == 20
    for s in loaded.train[:40]:
        assert _cands(s, epoch=0) == _cands(full_by_id[s.sample_id], epoch=0)


# ----------------------------------------------------------------------------
# M2-5 flags
# ----------------------------------------------------------------------------
def _hand_catalog(special: dict) -> ItemCatalog:
    """1,682 fixed-width filler titles (none is a substring of another) with some titles replaced."""
    titles = [f"filler title {i:04d}" for i in range(NUM_ITEMS)]
    for i, title in special.items():
        titles[i] = title
    return ItemCatalog(titles=tuple(titles), title_mode="code", num_items=NUM_ITEMS, pad_id=PAD_ID)


HAND_CANDS = tuple(range(20))      # target = 0, negatives = 1..19
CARMEN = "Carmen Miranda: Bananas Is My Business"
# flag -> (titles where it fires, flags fired, near-miss titles, flags fired). target_title_duplicated and
# target_unscorable cannot fire alone (§6.6.4 definitions): the same title is also a duplicate pair and an
# unscorable target; an unscorable negative is empty, equal or a substring of the target.
FLAG_CASES = {
    "empty_title_candidate": ({0: ""}, {"empty_title_candidate"},
                              {0: "-"}, set()),
    "duplicate_title_candidates": ({1: "Twin Peaks", 2: " TWIN PEAKS"}, {"duplicate_title_candidates"},
                                   {1: "Twin Peaks", 2: "Twin Peaks: Fire Walk"}, {"substring_title_candidates"}),
    "substring_title_candidates": ({1: "Heat", 2: "Heat Wave"}, {"substring_title_candidates"},
                                   {1: "Heat", 2: "Hate Wave"}, set()),
    "target_title_duplicated": ({0: "Sabrina", 1: "sabrina "},
                                {"duplicate_title_candidates", "target_title_duplicated", "target_unscorable"},
                                {0: "Sabrina", 1: "Sabrina 2"}, {"substring_title_candidates"}),
    "target_unscorable": ({0: CARMEN, 1: "Ran"}, {"substring_title_candidates", "target_unscorable"},
                          {0: "Ran", 1: CARMEN}, {"substring_title_candidates"}),
}


def _fired(special: dict) -> set:
    flags = candidate_flags(HAND_CANDS, 0, _hand_catalog(special))
    assert tuple(flags) == FLAG_NAMES and all(type(v) is bool for v in flags.values())
    return {name for name, on in flags.items() if on}


@pytest.mark.criterion("M2-5")
@pytest.mark.parametrize("flag", FLAG_NAMES)
def test_flags_single_trigger(flag):
    assert _fired({}) == set()
    trigger, fired, near_miss, near_fired = FLAG_CASES[flag]
    assert flag in fired and _fired(trigger) == fired
    assert flag not in near_fired and _fired(near_miss) == near_fired
    if flag == "empty_title_candidate":
        # an empty (or blank) negative is in every target title: the realistic item-266 case
        assert _fired({5: ""}) == {"empty_title_candidate", "target_unscorable"}
        assert _fired({5: "   "}) == {"empty_title_candidate", "target_unscorable"}
    assert match_norm("  Star Wars \r") == "star wars"
    with pytest.raises(CandidateError):
        candidate_flags(HAND_CANDS[:19], 0, _hand_catalog({}))
    with pytest.raises(CandidateError):
        candidate_flags(HAND_CANDS, 25, _hand_catalog({}))


def _flag_table(full_data, catalog) -> dict:
    out = {}
    for split in ("val", "test"):
        rows = {name: [] for name in FLAG_NAMES}
        any_rows = []
        for s in full_data.samples(split):
            flags = candidate_flags(_cands(s), s.target, catalog)
            for name, on in flags.items():
                if on:
                    rows[name].append(s.row)
            if any(flags.values()):
                any_rows.append(s.row)
        out[split] = (rows, any_rows)
    return out


@pytest.mark.criterion("M2-5")
def test_flag_counts_seed1234(full_data, repo_root):
    code = _flag_table(full_data, full_data.catalog)
    val_rows, val_any = code["val"]
    assert [len(val_rows[n]) for n in FLAG_NAMES] == [1, 0, 1, 0, 1] and len(val_any) == 2
    assert val_rows["empty_title_candidate"] == [66] and val_rows["substring_title_candidates"] == [36]
    assert val_rows["target_unscorable"] == [66]
    test_rows, test_any = code["test"]
    assert [len(test_rows[n]) for n in FLAG_NAMES] == [1, 0, 4, 0, 2] and len(test_any) == 5
    assert test_rows["empty_title_candidate"] == [5]
    assert test_rows["substring_title_candidates"] == [11, 38, 39, 43]
    assert test_rows["target_unscorable"] == [5, 11]
    # cases of §6.6.6
    titles = full_data.catalog.titles
    by_row = {(s.split, s.row): s for s in (*full_data.val, *full_data.test)}

    def pairs(split, row):
        norm = [match_norm(titles[c]) for c in _cands(by_row[(split, row)])]
        return {(a, b) for a in norm for b in norm if a != "" and a != b and a in b}

    assert 266 in _cands(by_row[("val", 66)]) and titles[by_row[("val", 66)].target] == \
        "Monty Python and the Holy Grail"
    assert 266 in _cands(by_row[("test", 5)]) and titles[by_row[("test", 5)].target] == "Dirty Dancing"
    row11 = by_row[("test", 11)]
    assert 655 in _cands(row11) and titles[655] == "M" and titles[row11.target] == "Mr. Holland's Opus"
    assert sum("m" in match_norm(titles[c]) for c in _cands(row11) if c != 655) == 5
    assert ("manhattan", "sudden manhattan") in pairs("val", 36)
    assert ("ran", "carmen miranda: bananas is my business") in pairs("test", 38)
    assert ("die hard", "die hard 2") in pairs("test", 39)
    assert ("glory", "paths of glory") in pairs("test", 43)
    # catalog-wide hazard behind the substring flag (§6.5): 79 code titles inside other titles, 774 pairs
    norm = [match_norm(t) for t in titles]
    inside = {i: sum(1 for j, b in enumerate(norm) if j != i and b != a and a in b)
              for i, a in enumerate(norm) if a != ""}
    inside = {i: n for i, n in inside.items() if n}
    assert (len(inside), sum(inside.values())) == (79, 774)
    assert (inside[655], inside[1286], inside[646]) == (541, 80, 25)
    # variant_fixed_titles reference values (§6.6.6)
    fixed = data.build_catalog(data.read_raw_titles(repo_root / data.DATA_DIR / data.ITEM_FILE), "fixed")
    fx = _flag_table(full_data, fixed)
    assert [len(fx["val"][0][n]) for n in FLAG_NAMES] == [0, 0, 1, 0, 0] and len(fx["val"][1]) == 1
    assert [len(fx["test"][0][n]) for n in FLAG_NAMES] == [0, 0, 4, 0, 1] and len(fx["test"][1]) == 4
    assert fx["test"][0]["target_unscorable"] == [11]


# ----------------------------------------------------------------------------
# M2-6 target_unscorable == the judgement of the exact target title
# ----------------------------------------------------------------------------
def _judges():
    """Judgement functions: the released rule copy (always) and M11 ``evaluate.judge`` once it exists."""
    out = [("refs_upstream", lambda first, ids, titles, target: refs_upstream.first_line_and_hr1(
        first, titles, target)[2])]
    if importlib.util.find_spec("ilora.evaluate") is not None:
        from ilora.evaluate import judge

        out.append(("evaluate.judge", lambda first, ids, titles, target: judge(first, ids, titles, target).correct))
    return out


@pytest.mark.criterion("M2-6")
def test_unscorable_matches_judge(full_data, repo_root):
    raw = data.read_raw_titles(repo_root / data.DATA_DIR / data.ITEM_FILE)
    checked = 0
    for catalog in (data.build_catalog(raw, "code"), data.build_catalog(raw, "fixed")):
        for s in (*full_data.val, *full_data.test):
            cands = _cands(s)
            titles = [catalog.titles[c] for c in cands]
            target_title = catalog.titles[s.target]
            unscorable = candidate_flags(cands, s.target, catalog)["target_unscorable"]
            for name, correct in _judges():
                assert unscorable == (not correct(target_title, cands, titles, target_title)), \
                    (catalog.title_mode, s.sample_id, name)
            checked += 1
    assert checked == 2 * 189


# ----------------------------------------------------------------------------
# M2-7 invalid calls
# ----------------------------------------------------------------------------
def _unchecked_sample(**fields) -> Sample:
    """A Sample that bypasses M1's checks, to reach M2's own double check."""
    sample = object.__new__(Sample)
    for name, value in fields.items():
        object.__setattr__(sample, name, value)
    return sample


INVALID_CASES = ["train_epoch_none", "val_epoch_0", "epoch_negative", "epoch_bool", "target_in_history",
                 "num_candidates_21"]


@pytest.mark.criterion("M2-7")
@pytest.mark.parametrize("case", INVALID_CASES)
def test_invalid_calls_rejected(case, profile_cfg):
    train = Sample("train", 3, 0, (1, 2, 3), 4)
    val = Sample("val", 0, 754, (1, 2, 3), 4)
    if case == "train_epoch_none":
        with pytest.raises(CandidateError, match="train sample needs epoch"):
            _cands(train, epoch=None)
    elif case == "val_epoch_0":
        with pytest.raises(CandidateError, match="evaluation sample takes epoch=None"):
            _cands(val, epoch=0)
    elif case == "epoch_negative":
        with pytest.raises(CandidateError, match="train sample needs epoch"):
            _cands(train, epoch=-1)
    elif case == "epoch_bool":
        with pytest.raises(CandidateError, match="train sample needs epoch"):
            _cands(train, epoch=True)
    elif case == "target_in_history":
        with pytest.raises(data.DataError):
            Sample("val", 0, 754, (1, 2, 3), 2)
        bad = _unchecked_sample(split="val", row=0, user_id=754, history=(1, 2, 3), target=2)
        with pytest.raises(CandidateError, match="in the history"):
            _cands(bad)
        with pytest.raises(CandidateError, match="in the history"):
            negative_pool((1, 2, 3), 2, NUM_ITEMS)
    elif case == "num_candidates_21":
        with pytest.raises(ConfigError, match="candidates.num_candidates"):
            profile_cfg("paper_spec", "candidates.num_candidates=21")
        with pytest.raises(CandidateError, match="num_candidates must be 20"):
            candidates_for(val, main_seed=1234, epoch=None, cfg=CandidatesConfig(num_candidates=21))
    else:
        raise AssertionError(case)


# ----------------------------------------------------------------------------
# M2-8 uniformity
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M2-8")
def test_uniformity_chi_square(full_data, paper_cfg, record_property):
    s = next(s for s in full_data.train if len(s.history) == 10)
    assert s.row == 10
    pool = negative_pool(s.history, s.target, NUM_ITEMS)
    assert len(pool) == 1671
    negatives, positions = Counter(), Counter()
    epochs = 10_000
    for epoch in range(epochs):
        c = candidates_for(s, main_seed=paper_cfg.run.seed, epoch=epoch, cfg=paper_cfg.candidates)
        positions[c.index(s.target)] += 1
        negatives.update(x for x in c if x != s.target)
    assert set(negatives) <= set(pool.tolist())
    observed = np.array([negatives[int(i)] for i in pool])
    assert observed.sum() == epochs * 19
    p_item = float(chisquare(observed, np.full(len(pool), epochs * 19 / len(pool))).pvalue)
    p_position = float(chisquare([positions[k] for k in range(20)]).pvalue)
    record_property("item_chi2_p", p_item)            # design-time value 0.380 (§6.8 #8)
    record_property("position_chi2_p", p_position)    # design-time value 0.398
    assert p_item > 0.001 and p_position > 0.001
