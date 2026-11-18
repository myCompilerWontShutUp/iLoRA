"""M2 후보 생성 시험: Steam, LastFM (DESIGN.md §23.2, §23.8 #7; D10, D21, D28, D34, D35, D38).

The serialized hash of §6.6.5 and the flag counts of §6.6.4 on the released validation and test files, seed 1234
(numpy 1.24.2, the condition of ``candidates.self_check``), are the §23.2 values. The data come from ``load_dataset`` of
``<name>_paper_spec``, which checks the sha256 of the four files first (§5.6.3 step 2). The released loaders are not
used: LastFM returns one negative and no target there (``U data/lastfm_data.py:43-46``, D34), so its candidates are
the §6.6.2 draw like the other two datasets. Tests that read the real frames carry ``slow`` (§23.8).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from ilora import data
from ilora.candidates import (FLAG_NAMES, NUM_CANDIDATES, candidate_flags, candidate_key, candidates_for,
                              check_candidates, draw_candidates, match_norm, negative_pool)
from ilora.config import load_config
from ilora.datasets import MOVIELENS, SPECS
from ilora.seeding import np_rng

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
PROFILE_DIR = REBUILD_DIR / "configs" / "datasets"
NAMES = ("steam", "lastfm")
EVAL_SPLITS = ("val", "test")
SEED = 1234                                       # run.seed of every profile (D2)
LAST_MOVIELENS_ID = MOVIELENS.num_items - 1       # 1681

# §23.2 후보 직렬화 hash, the §6.6.5 form (multi/verify/check_numbers.py).
SPLIT_SHA256 = {
    "steam": {"val": "ba0df0feb1ce97dfac44a5786bf1f0b2674cb2501fcb54f302419c468a4120bb",
              "test": "3dd8399ec36b739bfb25102e32a20569ad5bfdd1eddb87a00ca3c6612985813f"},
    "lastfm": {"val": "3caad6a1fb669dc2e82b234afa474ddd2ed880002fb20b64e70b94fa298d1294",
               "test": "0eb6f0af94ea28d6526b8a34f63ff88d94ce8cc5a243a07eb409dac57bc7c2d1"},
}
# Flag counts in FLAG_NAMES order. §23.2 제목 모호 표시 gives substring and unscorable (multi/datasets/s5_flags.py,
# multi/verify/check_numbers.py); the other three are 0 by the §23.2 제목 결함 기록 (no empty title, no exact duplicate,
# and the one match_norm pair of Steam never meets in these candidates).
FLAG_COUNTS = {
    "steam": {"val": [0, 0, 43, 0, 13], "test": [0, 0, 37, 0, 5]},
    "lastfm": {"val": [0, 0, 3, 0, 0], "test": [0, 0, 3, 0, 0]},
}
# §23.2 CombinedMetric 상한: N - target_unscorable, the §6.6.6 form (computed, not a result).
SCORABLE = {"steam": {"val": (1171, 1184), "test": (1181, 1186)},
            "lastfm": {"val": (122, 122), "test": (122, 122)}}
MIN_POOL = {"steam": 3570, "lastfm": 4595}        # §23.2 음성 풀 크기 (D21): N - 11, the pool of a full history
SUBSTRING_PAIRS = {"steam": (164, 2497), "lastfm": (261, 1167)}     # §23.2 부분 문자열 제목 쌍 (titles, pairs)


@lru_cache(maxsize=None)
def _released(name: str) -> SimpleNamespace:
    """``load_dataset`` of ``<name>_paper_spec`` on the released files (it checks the sha256 of the four files and all
    counts itself) and the candidates of every validation and test sample, computed once."""
    cfg, _ = load_config(PROFILE_DIR / f"{name}_paper_spec.yaml")
    full = data.load_dataset(cfg, REPO_ROOT)
    cands = {s.sample_id: candidates_for(s, main_seed=cfg.run.seed, epoch=None, cfg=cfg.candidates)
             for split in EVAL_SPLITS for s in full.samples(split)}
    return SimpleNamespace(name=name, spec=SPECS[name], cfg=cfg, full=full, cands=cands)


@pytest.fixture(scope="module", params=NAMES)
def released(request: pytest.FixtureRequest) -> SimpleNamespace:
    return _released(request.param)


# ----------------------------------------------------------------------------
# §23.2: the serialized hash and the flag counts of the fixed evaluation candidates (§23.8 #7)
# ----------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_serialized_hash_equals_design(released, split):
    """§23.2 후보 직렬화 hash, the §6.6.5 form: ``json.dumps([[row, list(cands)], ...], separators=(",", ":"))`` in row
    order, seed 1234 and ``candidates.num_candidates`` 20."""
    assert released.cfg.run.seed == SEED and released.cfg.candidates.num_candidates == NUM_CANDIDATES == 20
    samples = released.full.samples(split)
    assert len(samples) == released.spec.expected_samples[split]
    assert [s.row for s in samples] == sorted(s.row for s in samples)
    blob = json.dumps([[s.row, list(released.cands[s.sample_id])] for s in samples], separators=(",", ":"))
    assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == SPLIT_SHA256[released.name][split]


@pytest.mark.slow
@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_flag_counts_equal_design(released, split):
    """§23.2, §6.6.4: the five flags of every evaluation sample, and N - ``target_unscorable`` = the CombinedMetric
    bound of §23.2 (a computed bound, not a result)."""
    samples = released.full.samples(split)
    rows = {name: [] for name in FLAG_NAMES}
    for s in samples:
        flags = candidate_flags(released.cands[s.sample_id], s.target, released.full.catalog)
        assert tuple(flags) == FLAG_NAMES and all(type(v) is bool for v in flags.values()), s.sample_id
        for name in FLAG_NAMES:
            if flags[name]:
                rows[name].append(s.row)
    assert [len(rows[name]) for name in FLAG_NAMES] == FLAG_COUNTS[released.name][split]
    n = len(samples)
    assert (n - len(rows["target_unscorable"]), n) == SCORABLE[released.name][split]
    # no empty title and no duplicate here (§23.2), so an unscorable target always comes with a substring pair (§6.6.4)
    assert set(rows["target_unscorable"]) <= set(rows["substring_title_candidates"])


# ----------------------------------------------------------------------------
# D10, D21, D34: the candidate contract on the fixed evaluation candidates (§23.8 #7)
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_candidate_contract_on_every_evaluation_sample(released):
    """D10, D21, D34: 20 Python ints below ``num_items``, the target exactly once, no history id, on every validation
    and test sample; the negative pool is every item outside the window and the target."""
    spec = released.spec
    num_items = spec.num_items
    pool_sizes = []
    for split in EVAL_SPLITS:
        for s in released.full.samples(split):
            c = released.cands[s.sample_id]
            assert s.num_items == num_items, s.sample_id                # D28: read from the sample
            assert isinstance(c, tuple) and len(c) == 20, s.sample_id
            assert all(type(x) is int and 0 <= x < num_items for x in c), s.sample_id
            assert len(set(c)) == 20 and c.count(s.target) == 1, s.sample_id
            assert not set(c) & set(s.history), s.sample_id
            check_candidates(c, s, num_items=num_items, num_candidates=released.cfg.candidates.num_candidates)
            pool = negative_pool(s.history, s.target, num_items)
            assert len(pool) == num_items - len(s.history) - 1 and s.target not in pool, s.sample_id
            pool_sizes.append(len(pool))
    assert len(pool_sizes) == spec.expected_samples["val"] + spec.expected_samples["test"]
    assert min(pool_sizes) == MIN_POOL[released.name] == num_items - 11
    assert max(map(max, released.cands.values())) > LAST_MOVIELENS_ID          # ids above 1681 occur (D28)


@pytest.mark.slow
def test_train_sample_below_the_movielens_bound_draws_from_all_items(released):
    """D28: the id bound is carried by the sample, so a train row whose history and target are all below 1,682 still
    draws from all 3,581 or 4,606 items. Such rows exist in both datasets, and a pool of 1,682 would have accepted them
    silently (the failure §23.3 D28 names)."""
    spec, full, cc = released.spec, released.full, released.cfg.candidates
    low = [s for s in full.train if max((*s.history, s.target)) <= LAST_MOVIELENS_ID]
    assert low and {s.num_items for s in low} == {spec.num_items}
    for s in low:
        c = candidates_for(s, main_seed=SEED, epoch=0, cfg=cc)
        assert max(c) > LAST_MOVIELENS_ID and len(c) == 20, s.sample_id
    first = low[0]
    ours = candidates_for(first, main_seed=SEED, epoch=0, cfg=cc)
    rng = np_rng(SEED, *candidate_key(first, 0))
    assert ours == draw_candidates(first.history, first.target, rng, num_items=spec.num_items, num_candidates=20)
    rng = np_rng(SEED, *candidate_key(first, 0))
    narrow = draw_candidates(first.history, first.target, rng, num_items=MOVIELENS.num_items, num_candidates=20)
    assert max(narrow) <= LAST_MOVIELENS_ID and narrow != ours
    assert len(negative_pool(first.history, first.target, spec.num_items)) == spec.num_items - len(first.history) - 1


@pytest.mark.slow
def test_random_keys_do_not_name_the_dataset(released):
    """D35: the keys of §6.6.2 are the MovieLens keys, ``("candidates", split, row)`` and ``("candidates", "train",
    row, epoch)``; the draw is a function of (history, target, key). Training candidates are redrawn every epoch and
    evaluation candidates are fixed (D10)."""
    full, cc, num_items = released.full, released.cfg.candidates, released.spec.num_items
    val, test, train = full.val[0], full.test[0], full.train[0]
    assert candidate_key(val, None) == ("candidates", "val", val.row)
    assert candidate_key(test, None) == ("candidates", "test", test.row)
    assert candidate_key(train, 2) == ("candidates", "train", train.row, 2)
    for s, epoch in ((val, None), (test, None), (train, 0), (train, 2)):
        rng = np_rng(SEED, *candidate_key(s, epoch))
        assert candidates_for(s, main_seed=SEED, epoch=epoch, cfg=cc) == draw_candidates(
            s.history, s.target, rng, num_items=num_items, num_candidates=20), s.sample_id
    epoch0 = candidates_for(train, main_seed=SEED, epoch=0, cfg=cc)
    assert epoch0 != candidates_for(train, main_seed=SEED, epoch=1, cfg=cc)
    assert released.cands[val.sample_id] == candidates_for(val, main_seed=SEED, epoch=None, cfg=cc)


# ----------------------------------------------------------------------------
# D38: the recorded title defects are flagged, not repaired
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_recorded_title_defects_are_flagged_not_repaired(released):
    """§23.2 제목 결함 기록 (D38): no empty title and no exact duplicate, the catalog-wide substring hazard behind the
    flag (counted as ``test_m2_candidates.py::test_flag_counts_seed1234`` counts it), and the one ``match_norm``
    collision of Steam, which never meets in the seed 1234 evaluation candidates and is flagged by
    ``candidate_flags`` in a set that holds both titles."""
    catalog = released.full.catalog
    titles = catalog.titles
    assert "" not in titles and len(set(titles)) == len(titles) == released.spec.num_items
    norm = [match_norm(t) for t in titles]
    inside = {i: sum(1 for j, b in enumerate(norm) if j != i and b != a and a in b)
              for i, a in enumerate(norm) if a != ""}
    inside = {i: n for i, n in inside.items() if n}
    assert (len(inside), sum(inside.values())) == SUBSTRING_PAIRS[released.name]
    if released.name == "lastfm":
        assert len(set(norm)) == len(norm)                                # no match_norm collision in this file
        return
    pair = (1064, 2201)
    assert (titles[1064], titles[2201]) == ("FORTIFY", "Fortify") and norm[1064] == norm[2201] == "fortify"
    assert not any(set(pair) <= set(c) for c in released.cands.values())
    assert any(set(pair) & set(c) for c in released.cands.values())      # one of the two does occur
    negatives = tuple(i for i in range(released.spec.num_items) if i not in pair)[:18]
    cands = (*pair, *negatives)
    as_target = candidate_flags(cands, 1064, catalog)
    assert as_target["duplicate_title_candidates"] and as_target["target_title_duplicated"]
    assert as_target["target_unscorable"]                                # the other spelling equals the target title
    among_negatives = candidate_flags(cands, negatives[0], catalog)
    assert among_negatives["duplicate_title_candidates"] and not among_negatives["target_title_duplicated"]
