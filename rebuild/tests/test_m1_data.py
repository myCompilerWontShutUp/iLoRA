"""M1 데이터 시험 (DESIGN.md §5.8 #1-#11, #13, #14; §17.6.9 M1 표). #12 is ``vessl/test_vessl_env.py``.

The released files are read only after their sha256 matched ``run.EXPECTED_INPUTS``.
Synthetic frames come from ``seeding.np_rng(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import dataclasses
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import refs_upstream
from ilora import data, run, seeding
from ilora.config import ConfigError
from ilora.data import DataError, LoadedData, Sample

# Design values (§5.6.2-§5.6.9), written out here so that the module constants are checked too.
FINGERPRINTS = {
    "train": "85bafab832f1f78e5095b1fd880a418097054d3ea7dbb7d4fe9e1528f019aa99",
    "val": "de7ac66b6703dc6935e1339183c00e08af2ad40f6d0877e7556b483a9aa65523",
    "test": "01636285c16faf95cc7136bc937b9aa315c66de5e23fed5efec0e2b2e84dff09",
}
ROWS = {"train": 68388, "val": 94, "test": 95}
SAMPLES = {"train": 66126, "val": 94, "test": 95}
SMOKE_VAL_ROWS = [1, 7, 15, 20, 26, 44, 59, 61, 77, 90]
SMOKE_TEST_ROWS = [4, 9, 14, 20, 50, 66, 71, 72, 82, 92]
OVERFIT_ROWS = (71, 196, 377, 554, 797, 1001, 1129, 1234, 1595, 1651, 1905, 2070, 2311, 2693, 2765, 2874)
FIXED_DIFF_IDS = (176, 266, 435, 601, 965, 1004, 1018, 1127, 1200, 1226, 1411, 1634)


@pytest.fixture(scope="module")
def released(repo_root: Path) -> dict:
    """The three released frames (hash-checked before unpickling), projected, parsed, with pseudo users."""
    out = {}
    for split in data.SPLITS:
        rel = f"{data.DATA_DIR}/{data.SPLIT_FILES[split]}"
        expected, is_text = run.EXPECTED_INPUTS[rel]
        assert run.content_sha256(repo_root / rel, is_text) == expected, rel
        original = pd.read_pickle(repo_root / rel)
        frame = data.project_frame(original, split)
        rows = data.parse_rows(frame, split)
        out[split] = SimpleNamespace(original=original, frame=frame, rows=rows, users=data.assign_users(rows, split))
    return out


# ----------------------------------------------------------------------------
# M1-1 hash before unpickle
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-1")
def test_hash_checked_before_unpickle(repo_root, paper_cfg, tmp_path, monkeypatch):
    """A copy with one changed byte in any of the four files fails with DataError before pandas.read_pickle."""
    names = (*data.SPLIT_FILES.values(), data.ITEM_FILE)
    target_dir = tmp_path / data.DATA_DIR
    target_dir.mkdir(parents=True)
    for name in names:
        shutil.copyfile(repo_root / data.DATA_DIR / name, target_dir / name)
    calls = []
    real_read_pickle = pd.read_pickle

    def spy(*args, **kwargs):
        calls.append(args)
        return real_read_pickle(*args, **kwargs)

    monkeypatch.setattr(pd, "read_pickle", spy)
    # Val_data.df first (the case named in §17.6.9), then the other three files.
    for name in ("Val_data.df", "train_data.df", "Test_data.df", data.ITEM_FILE):
        path = target_dir / name
        original = path.read_bytes()
        changed = bytearray(original)
        changed[len(changed) // 2] ^= 0x01
        path.write_bytes(bytes(changed))
        rel = f"{data.DATA_DIR}/{name}"
        with pytest.raises(DataError, match=rf"{rel}: sha256 [0-9a-f]{{64}} != expected {run.EXPECTED_INPUTS[rel][0]}"):
            data.load_dataset(paper_cfg, tmp_path)
        assert calls == [], name
        path.write_bytes(original)
    (target_dir / "Test_data.df").unlink()
    with pytest.raises(DataError, match="Test_data.df: file not found"):
        data.load_dataset(paper_cfg, tmp_path)
    assert calls == []


# ----------------------------------------------------------------------------
# M1-2 counts
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-2")
def test_row_and_sample_counts(full_data, released):
    for split in data.SPLITS:
        assert len(released[split].rows) == ROWS[split] == data.EXPECTED_ROWS[split]
        assert len(full_data.samples(split)) == SAMPLES[split] == data.EXPECTED_SAMPLES[split]
    # (len_seq, k) of the unfiltered rows (§5.6.2)
    dist = {split: Counter((r.len_seq, len(r.history)) for r in released[split].rows) for split in data.SPLITS}
    assert dist["train"] == {(1, 0): 754, (1, 1): 754, (2, 2): 754, (3, 3): 754, (4, 4): 754, (5, 5): 754,
                             (6, 6): 753, (7, 7): 753, (8, 8): 752, (9, 9): 752, (10, 10): 60854}
    assert dist["val"] == {(9, 9): 2, (10, 10): 92}
    assert dist["test"] == {(10, 10): 95}
    # the filter removes 2,262 = 754 x 3 train rows and nothing else; kept rows have len(history) == len_seq
    for split in data.SPLITS:
        kept = [r for r in released[split].rows if r.len_seq >= data.MIN_LEN_SEQ]
        assert len(kept) == SAMPLES[split]
        assert all(len(r.history) == r.len_seq for r in kept)
        samples = full_data.samples(split)
        assert [s.row for s in samples] == [r.row for r in kept]
        assert [s.history for s in samples] == [r.history for r in kept]
        assert [s.target for s in samples] == [r.target for r in kept]
    lengths = {split: Counter(len(s.history) for s in full_data.samples(split)) for split in data.SPLITS}
    assert (min(lengths["train"]), max(lengths["train"])) == (3, 10)
    assert set(lengths["val"]) == {9, 10} and set(lengths["test"]) == {10}
    assert full_data.subset == "full" and full_data.smoke_users is None and full_data.title_mode == "code"
    assert full_data.catalog.num_items == data.NUM_ITEMS == 1682 and full_data.catalog.pad_id == data.PAD_ID == 1682


# ----------------------------------------------------------------------------
# M1-3 pseudo users
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-3")
def test_user_blocks(full_data, released):
    rows, users = released["train"].rows, released["train"].users
    starts = [i for i in range(len(users)) if i == 0 or users[i] != users[i - 1]]
    assert len(starts) == 754 and users[0] == 0 and users[-1] == 753 and list(users) == sorted(users)
    assert rows[starts[0]].row == 0 and rows[starts[1]].row == 72 and rows[starts[753]].row == 68368
    # block starts are exactly the empty-history rows
    assert [rows[i].row for i in starts] == [r.row for r in rows if not r.history]
    bounds = [*starts, len(rows)]
    sizes = [bounds[b + 1] - bounds[b] for b in range(754)]
    assert (min(sizes), max(sizes), sum(sizes)) == (6, 509, 68388)
    kept_by_user = defaultdict(list)
    for s in full_data.train:
        kept_by_user[s.user_id].append(s.row)
    assert sorted(kept_by_user) == list(range(754))                 # no empty block after the filter
    for b in range(754):
        block = rows[bounds[b]:bounds[b + 1]]
        assert [len(r.history) for r in block] == [min(j, 10) for j in range(len(block))], b
        assert [r.len_seq for r in block[:3]] == [1, 1, 2], b        # the filter drops exactly these 3 rows
        assert kept_by_user[b] == [r.row for r in block[3:]], b
    # the same rule on the filtered samples gives the same 754 blocks (§5.6.5)
    again, block = [], -1
    for i, s in enumerate(full_data.train):
        prev = full_data.train[i - 1] if i else None
        if prev is None or s.history != (prev.history + (prev.target,))[-10:]:
            block += 1
        again.append(block)
    assert again == [s.user_id for s in full_data.train]
    # validation and test: one row = one user
    assert [s.user_id for s in full_data.val] == list(range(754, 848)) == [754 + s.row for s in full_data.val]
    assert [s.user_id for s in full_data.test] == list(range(848, 943)) == [848 + s.row for s in full_data.test]
    assert data.USER_OFFSET == {"train": 0, "val": 754, "test": 848}
    # validation/test histories are all distinct and appear nowhere in train (§5.6.5)
    val_h, test_h = {s.history for s in full_data.val}, {s.history for s in full_data.test}
    train_h = {r.history for r in rows}
    assert len(val_h) == 94 and len(test_h) == 95 and not val_h & test_h
    assert not (val_h | test_h) & train_h


# ----------------------------------------------------------------------------
# M1-4 fingerprints and projection
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-4")
def test_fingerprints_and_projection(released):
    assert data.EXPECTED_FINGERPRINTS == FINGERPRINTS
    for split in data.SPLITS:
        original, frame = released[split].original, released[split].frame
        expected_cols = ["seq", "len_seq", "next", "cluster"] if split == "train" else ["seq", "len_seq", "next"]
        assert list(original.columns) == expected_cols
        assert list(frame.columns) == ["seq", "len_seq", "next"]
        assert data.frame_fingerprint(frame) == FINGERPRINTS[split]
    # the fingerprint ignores the cluster column: LLaRA layout (no cluster) and changed cluster values agree
    train = released["train"].original
    llara_layout = train.drop(columns=["cluster"])
    assert data.frame_fingerprint(data.project_frame(llara_layout, "train")) == FINGERPRINTS["train"]
    other_cluster = train.copy()
    other_cluster["cluster"] = np.int32(0)
    assert data.frame_fingerprint(data.project_frame(other_cluster, "train")) == FINGERPRINTS["train"]
    # no cluster information reaches Sample or LoadedData
    assert [f.name for f in dataclasses.fields(Sample)] == ["split", "row", "user_id", "history", "target"]
    assert [f.name for f in dataclasses.fields(LoadedData)] == ["catalog", "train", "val", "test", "subset",
                                                                "smoke_users"]
    assert [f.name for f in dataclasses.fields(data.RawRow)] == ["row", "len_seq", "history", "target"]


# ----------------------------------------------------------------------------
# M1-5, M1-6 titles
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-5")
def test_code_titles_reproduce_defects(full_data, repo_root):
    titles = full_data.catalog.titles
    assert full_data.catalog.title_mode == "code" and len(titles) == 1682
    assert titles[435] == "A American Werewolf in Londonn"
    assert titles[601] == "A American in Parisn"
    assert titles[965] == "A Affair to Remembern"
    assert titles[1226] == "A Awfully Big Adventuren"
    assert titles[176] == "The Good Bad and The Ugly"
    assert titles[1004] == "The Double vie de Véronique, La (Double Life of Veronique)"
    assert titles[1018] == "The Die xue shuang xiong (Killer)"
    assert titles[266] == ""
    assert titles[1411] == "Land Before Time III: The Time of the Great Giving (19"
    assert titles[1127] == "Heidi Fleiss: Hollywood Madam "
    assert titles[1200] == "Marlene Dietrich: Shadow and Light "
    assert titles[1634] == "Two Friends "
    assert titles[1632] == "Á köldum klaka (Cold Fever)"      # ISO-8859-1 decode kept as is
    assert sum(any(ord(ch) > 127 for ch in t) for t in titles) == 9
    for name, norm in (("exact", lambda t: t), ("match", lambda t: t.strip().lower().strip())):
        groups = defaultdict(list)
        for i, t in enumerate(titles):
            groups[norm(t)].append(i)
        dup = {t: ids for t, ids in groups.items() if len(ids) > 1}
        assert (len(dup), sum(len(ids) for ids in dup.values())) == (22, 45), name
    report = data.title_report(data.read_raw_titles(repo_root / data.DATA_DIR / data.ITEM_FILE))
    assert tuple(report) == data.TITLE_REPORT_CATEGORIES
    assert report["article_an"] == (435, 601, 965, 1226)
    assert report["article_double_the"] == (176,)
    assert report["article_in_parentheses"] == (1004, 1018)
    assert report["empty"] == (266,)
    assert report["truncated"] == (1411,)
    assert report["trailing_space"] == (1127, 1200, 1634)
    assert len(report["duplicate_title"]) == 45
    assert {titles[i] for i in report["duplicate_title"]} >= {"Cape Fear", "Sabrina", "That Darn Cat!"}
    # item 266 (empty title): 6 train targets, 76 history occurrences, absent from validation and test (§5.5)
    assert sum(s.target == 266 for s in full_data.train) == 6
    assert sum(s.history.count(266) for s in full_data.train) == 76
    assert not any(s.target == 266 or 266 in s.history for s in (*full_data.val, *full_data.test))


@pytest.mark.criterion("M1-6")
def test_fixed_titles_differ_in_12_ids(repo_root):
    raw = data.read_raw_titles(repo_root / data.DATA_DIR / data.ITEM_FILE)
    code = data.build_catalog(raw, "code")
    fixed = data.build_catalog(raw, "fixed")
    assert fixed.title_mode == "fixed" and code.title_mode == "code"
    assert tuple(i for i in range(1682) if code.titles[i] != fixed.titles[i]) == FIXED_DIFF_IDS
    assert fixed.titles[435] == "An American Werewolf in London"
    assert fixed.titles[176] == "The Good, The Bad and The Ugly"
    assert fixed.titles[1004] == "Double vie de Véronique, La (Double Life of Veronique, The)"
    assert fixed.titles[1411] == "Land Before Time III: The Time of the Great Giving"
    assert fixed.titles[266] == "unknown"
    assert fixed.titles[1127] == "Heidi Fleiss: Hollywood Madam"
    assert [i for i, t in enumerate(fixed.titles) if t.strip() == ""] == []
    assert [i for i, t in enumerate(fixed.titles) if i != 266 and "unknown" in t.lower()] == []
    groups = Counter(t.strip().lower().strip() for t in fixed.titles)
    dup = {t: n for t, n in groups.items() if n > 1}
    assert (len(dup), sum(dup.values())) == (22, 45)
    with pytest.raises(DataError):
        data.build_catalog(raw, "other")


# ----------------------------------------------------------------------------
# M1-7 CRLF
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-7")
def test_item_file_crlf_invariance(repo_root, tmp_path):
    lf = (repo_root / data.DATA_DIR / data.ITEM_FILE).read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert lf.count(b"\n") == 1682 and crlf.count(b"\r\n") == 1682
    lf_path, crlf_path = tmp_path / "lf" / data.ITEM_FILE, tmp_path / "crlf" / data.ITEM_FILE
    for path, content in ((lf_path, lf), (crlf_path, crlf)):
        path.parent.mkdir()
        path.write_bytes(content)
    expected = run.EXPECTED_INPUTS[f"{data.DATA_DIR}/{data.ITEM_FILE}"][0]
    assert run.content_sha256(lf_path, text=True) == run.content_sha256(crlf_path, text=True) == expected
    assert run.content_sha256(lf_path) != run.content_sha256(crlf_path)
    titles_lf, titles_crlf = data.read_raw_titles(lf_path), data.read_raw_titles(crlf_path)
    assert titles_lf == titles_crlf and len(titles_lf) == 1682
    for mode in data.TITLE_MODES:
        assert data.build_catalog(titles_lf, mode) == data.build_catalog(titles_crlf, mode)
    # only "\n" ends a line: an ISO-8859-1 \x85 (NEL) inside a title stays in the title (§5.6.3 step 3)
    nel_path = tmp_path / "nel" / data.ITEM_FILE
    nel_path.parent.mkdir()
    nel_path.write_bytes(lf.replace(b"1|Toy Story (1995)|", b"1|Toy\x85Story (1995)|", 1))
    titles_nel = data.read_raw_titles(nel_path)
    assert titles_nel[0] == "Toy\x85Story (1995)" and titles_nel[1:] == titles_lf[1:]
    # a missing last newline is fine, an extra empty line is not
    no_final = tmp_path / "no_final" / data.ITEM_FILE
    no_final.parent.mkdir()
    no_final.write_bytes(lf[:-1])
    assert data.read_raw_titles(no_final) == titles_lf
    extra = tmp_path / "extra" / data.ITEM_FILE
    extra.parent.mkdir()
    extra.write_bytes(lf + b"\n")
    with pytest.raises(DataError, match="expected 1682 lines"):
        data.read_raw_titles(extra)


# ----------------------------------------------------------------------------
# M1-8 fork numbering
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-8")
def test_row_maps_to_fork_sample_idx(full_data, released):
    train_idx = refs_upstream.fork_sample_idx(released["train"].original)
    assert len(train_idx) == 66126 and set(train_idx) == {s.row for s in full_data.train}
    assert all(s.row == train_idx[s.row] + 3 * (s.user_id + 1) for s in full_data.train)
    for split in ("val", "test"):
        idx = refs_upstream.fork_sample_idx(released[split].original)
        assert len(idx) == SAMPLES[split]
        assert all(s.row == idx[s.row] for s in full_data.samples(split))


# ----------------------------------------------------------------------------
# M1-9 smoke subset
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-9")
def test_smoke_selection_seed1234(full_data, smoke_data, cpu_cfg, repo_root):
    smoke = data.select_smoke(full_data, 1234, (80, 10, 10), train_rows=())
    users = sorted({s.user_id for s in smoke.train})
    assert len(users) == 80 and len(smoke.train) == 7230 and users[:5] == [43, 51, 60, 66, 69]
    assert [s.row for s in smoke.val] == SMOKE_VAL_ROWS
    assert [s.user_id for s in smoke.val] == [755, 761, 769, 774, 780, 798, 813, 815, 831, 844]
    assert [s.row for s in smoke.test] == SMOKE_TEST_ROWS
    assert [s.user_id for s in smoke.test] == [852, 857, 862, 868, 898, 914, 919, 920, 930, 940]
    assert smoke.subset == "smoke" and smoke.smoke_users == (80, 10, 10) and smoke.catalog is full_data.catalog
    # every window of the chosen users, in row order, rows and user ids not renumbered
    assert smoke.train == tuple(s for s in full_data.train if s.user_id in set(users))
    assert set(smoke.val) <= set(full_data.val) and set(smoke.test) <= set(full_data.test)
    assert data.select_smoke(full_data, 1234, (80, 10, 10), train_rows=()) == smoke      # twice, same
    # smoke_cpu sizes (2, 10, 10): users 117 and 129, 156 samples (§17.6.6)
    small = data.select_smoke(full_data, 1234, (2, 10, 10), train_rows=())
    assert sorted({s.user_id for s in small.train}) == [117, 129] and len(small.train) == 156
    assert [s.row for s in small.val] == SMOKE_VAL_ROWS and [s.row for s in small.test] == SMOKE_TEST_ROWS
    assert small == smoke_data
    # load_dataset with data.subset = smoke gives the same subset (§5.6.3 step 8)
    assert cpu_cfg.data.subset == "smoke" and cpu_cfg.data.smoke_users == (2, 10, 10)
    assert data.load_dataset(cpu_cfg, repo_root) == small
    assert full_data.subset == "full" and len(full_data.train) == 66126             # the full data is untouched


@pytest.mark.criterion("M1-9")
@pytest.mark.parametrize("sizes", [(0, 10, 10), (755, 10, 10), (80, 0, 10), (80, 95, 10), (80, 10, 96),
                                   (80, 10), (True, 10, 10)])
def test_smoke_size_range(full_data, sizes):
    with pytest.raises(DataError, match="smoke size"):
        data.select_smoke(full_data, 1234, sizes, train_rows=())


# ----------------------------------------------------------------------------
# M1-10 synthetic frames
# ----------------------------------------------------------------------------
def _seq(items, ratings) -> list:
    """Released layout: real entries are numpy.int64 pairs, padding is the Python pair (1682, 0)."""
    entries = [(np.int64(i), np.int64(r)) for i, r in zip(items, ratings)]
    return entries + [(data.PAD_ID, data.PAD_RATING)] * (data.HISTORY_LEN - len(entries))


def _block(items, ratings) -> list[dict]:
    """One user block: row p has history items[max(0, p - 10):p] and target items[p]."""
    rows = []
    for p in range(len(items)):
        start = max(0, p - data.HISTORY_LEN)
        rows.append({"seq": _seq(items[start:p], ratings[start:p]), "len_seq": max(1, p - start),
                     "next": (np.int64(items[p]), np.int64(ratings[p]))})
    return rows


def _user_sequences(name: str, n_users: int, length: int) -> list[tuple[list[int], list[int]]]:
    rng = seeding.np_rng(0, "test", name)
    out = []
    for _ in range(n_users):
        items = [int(x) for x in rng.choice(data.NUM_ITEMS, size=length, replace=False)]
        ratings = [int(x) for x in rng.integers(1, 6, size=length)]
        out.append((items, ratings))
    return out


def _frame(rows: list[dict], *, cluster: bool = False) -> pd.DataFrame:
    df = pd.DataFrame({"seq": [r["seq"] for r in rows], "len_seq": np.array([r["len_seq"] for r in rows], np.int64),
                       "next": [r["next"] for r in rows]})
    if cluster:
        df["cluster"] = np.zeros(len(df), dtype=np.int32)
    return df


def _pipeline(df: pd.DataFrame, split: str) -> tuple[Sample, ...]:
    frame = data.project_frame(df, split)
    rows = data.parse_rows(frame, split)
    return data.to_samples(rows, data.assign_users(rows, split), split)


def _train_rows(name: str, *, change=None) -> list[dict]:
    """Two users with 14 interactions each (windows 0..10, 10, 10, 10); ``change(user, items)`` edits items."""
    rows = []
    for u, (items, ratings) in enumerate(_user_sequences(name, 2, 14)):
        if change is not None:
            change(u, items)
        rows += _block(items, ratings)
    return rows


@pytest.mark.criterion("M1-10")
def test_synthetic_pipeline_accepts_valid_frames():
    """Control for M1-10: the unmodified synthetic frames load (both train layouts, val, test)."""
    for cluster in (False, True):
        samples = _pipeline(_frame(_train_rows("parse_rows"), cluster=cluster), "train")
        assert len(samples) == 2 * (14 - 3) and [s.user_id for s in samples] == [0] * 11 + [1] * 11
        assert [s.row for s in samples] == [*range(3, 14), *range(17, 28)]
    eval_rows = [_block(items, ratings)[-1] for items, ratings in _user_sequences("parse_rows_eval", 3, 11)]
    for split in ("val", "test"):
        samples = _pipeline(_frame(eval_rows), split)
        assert [s.user_id for s in samples] == [data.USER_OFFSET[split] + r for r in range(3)]


def _case_rows(case: str) -> tuple[pd.DataFrame, str]:
    rows = _train_rows("parse_rows")
    pad = (data.PAD_ID, data.PAD_RATING)
    if case == "left_padding":
        rows[3]["seq"] = [pad] + rows[3]["seq"][:9]
    elif case == "middle_padding":
        rows[5]["seq"] = rows[5]["seq"][:2] + [pad] + rows[5]["seq"][2:9]
    elif case == "len_seq_mismatch":
        rows[4]["len_seq"] = 5
    elif case == "target_1682":
        rows[6]["next"] = (np.int64(data.PAD_ID), np.int64(4))
    elif case == "item_out_of_range":
        rows[7]["seq"][0] = (np.int64(1683), np.int64(4))
    elif case == "rating_0":
        rows[8]["seq"][1] = (rows[8]["seq"][1][0], np.int64(0))
    elif case == "duplicate_in_history":
        # item 1 repeats item 0: rows 1-2 are filtered, row 3 (kept) holds the repeat; blocks stay continuous
        rows = _train_rows("parse_rows", change=lambda u, items: items.__setitem__(1, items[0]) if u == 0 else None)
    elif case == "target_in_history":
        rows = _train_rows("parse_rows", change=lambda u, items: items.__setitem__(5, items[2]) if u == 0 else None)
    elif case == "columns":
        eval_rows = [_block(items, ratings)[-1] for items, ratings in _user_sequences("parse_rows_eval", 3, 11)]
        return _frame(eval_rows, cluster=True), "val"            # cluster is allowed in train only
    elif case == "not_range_index":
        df = _frame(rows)
        df.index = pd.Index(np.arange(len(df), dtype=np.int64))  # same labels, not a RangeIndex
        return df, "train"
    elif case == "block_not_starting_empty":
        del rows[14]                                            # user 1 now starts with a 1-item history
    else:
        raise AssertionError(case)
    return _frame(rows), "train"


PARSE_CASES = {
    "left_padding": "padding", "middle_padding": "padding", "len_seq_mismatch": "len_seq",
    "target_1682": "next: item id 1682", "item_out_of_range": "item id 1683", "rating_0": "rating 0",
    "duplicate_in_history": "repeated item", "target_in_history": "is in the history",
    "columns": "columns", "not_range_index": "RangeIndex", "block_not_starting_empty": "empty history",
}


@pytest.mark.criterion("M1-10")
@pytest.mark.parametrize("case", list(PARSE_CASES))
def test_parse_rows_rejects(case):
    df, split = _case_rows(case)
    with pytest.raises(DataError, match=PARSE_CASES[case]):
        _pipeline(df, split)
    # extra train-layout checks that do not fit one case each
    if case == "columns":
        with pytest.raises(DataError, match="columns"):
            _pipeline(_frame(_train_rows("parse_rows")).assign(other=1), "train")
        with pytest.raises(DataError, match="columns"):
            _pipeline(_frame(_train_rows("parse_rows")).drop(columns=["next"]), "train")
    if case == "not_range_index":
        df = _frame(_train_rows("parse_rows"))
        df.index = pd.RangeIndex(1, len(df) + 1)
        with pytest.raises(DataError, match="RangeIndex"):
            _pipeline(df, "train")
        with pytest.raises(DataError, match="DataFrame"):
            data.project_frame([1, 2, 3], "train")


# ----------------------------------------------------------------------------
# M1-11 Sample checks
# ----------------------------------------------------------------------------
SAMPLE_CASES = {
    "history_len_2": dict(history=(1, 2)),
    "history_len_11": dict(history=tuple(range(11))),
    "bool_in_history": dict(history=(True, 2, 3)),
    "bool_target": dict(target=True),
    "history_id_1682": dict(history=(1682, 2, 3)),
    "target_id_negative": dict(target=-1),
    "numpy_id": dict(history=(np.int64(1), 2, 3)),
    "repeated_history": dict(history=(1, 2, 1)),
    "target_in_history": dict(target=2),
    "unknown_split": dict(split="valid"),
    "negative_row": dict(row=-1),
    "list_history": dict(history=[1, 2, 3]),
}


@pytest.mark.criterion("M1-11")
@pytest.mark.parametrize("case", list(SAMPLE_CASES))
def test_sample_rejects(case):
    good = dict(split="val", row=0, user_id=754, history=(1, 2, 3), target=4)
    Sample(**good)
    with pytest.raises(DataError):
        Sample(**{**good, **SAMPLE_CASES[case]})


@pytest.mark.criterion("M1-11")
def test_sample_id_format(full_data):
    assert Sample("train", 3, 0, (1, 2, 3), 4).sample_id == "train:3"
    assert Sample("val", 0, 754, tuple(range(10)), 11).sample_id == "val:0"
    assert full_data.val[0].sample_id == "val:0" and full_data.test[-1].sample_id == "test:94"
    assert full_data.train[0].sample_id == "train:3"
    sample = full_data.train[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.row = 4
    with pytest.raises(DataError, match="unknown split"):
        full_data.samples("validation")


# ----------------------------------------------------------------------------
# M1-13 train_rows
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-13")
def test_train_rows_subset(full_data, profile_cfg, repo_root, monkeypatch):
    cfg = profile_cfg("overfit_gpu")
    assert cfg.data.train_rows == OVERFIT_ROWS and cfg.data.smoke_users == (16, 10, 10)
    keys = []
    real_np_rng = data.np_rng

    def spy(main_seed, *k):
        keys.append(k)
        return real_np_rng(main_seed, *k)

    monkeypatch.setattr(data, "np_rng", spy)
    sub = data.select_smoke(full_data, cfg.run.seed, cfg.data.smoke_users, train_rows=cfg.data.train_rows)
    assert sorted(keys) == [("smoke_users", "test"), ("smoke_users", "val")]       # no ("smoke_users", "train")
    assert [s.row for s in sub.train] == list(OVERFIT_ROWS)
    assert [s.user_id for s in sub.train] == list(range(16))
    last_row = {}
    for s in full_data.train:
        last_row[s.user_id] = s.row
    assert [last_row[u] for u in range(16)] == list(OVERFIT_ROWS)                    # last window of users 0-15
    assert all(len(s.history) == 10 for s in sub.train)
    assert len({s.target for s in sub.train}) == 16 and 266 not in {s.target for s in sub.train}
    assert [s.row for s in sub.val] == SMOKE_VAL_ROWS and [s.row for s in sub.test] == SMOKE_TEST_ROWS
    assert sub.subset == "smoke" and sub.smoke_users == (16, 10, 10)
    monkeypatch.undo()
    assert data.load_dataset(cfg, repo_root) == sub


TRAIN_ROWS_CASES = ["filtered_row", "outside_train", "user_count", "duplicate", "descending", "too_many",
                    "subset_full"]


@pytest.mark.criterion("M1-13")
@pytest.mark.parametrize("case", TRAIN_ROWS_CASES)
def test_train_rows_rejects(case, full_data, profile_cfg):
    if case == "filtered_row":
        with pytest.raises(DataError, match="not train samples"):
            data.select_smoke(full_data, 1234, (1, 10, 10), train_rows=(0,))
    elif case == "outside_train":
        with pytest.raises(DataError, match="not train samples"):
            data.select_smoke(full_data, 1234, (1, 10, 10), train_rows=(68388,))
    elif case == "user_count":
        with pytest.raises(DataError, match="smoke_users"):
            data.select_smoke(full_data, 1234, (15, 10, 10), train_rows=OVERFIT_ROWS)
    else:
        sets = {
            "duplicate": "data.train_rows=[71, 71, 196]",
            "descending": "data.train_rows=[196, 71]",
            "too_many": f"data.train_rows={list(range(3, 68))}",
            "subset_full": "data.subset=full",
        }[case]
        with pytest.raises(ConfigError, match="train_rows"):
            profile_cfg("overfit_gpu", sets)


# ----------------------------------------------------------------------------
# M1-14 pad_histories
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M1-14")
def test_pad_histories():
    rng = seeding.np_rng(0, "test", "pad_histories")
    items = [int(x) for x in rng.choice(data.NUM_ITEMS, size=13, replace=False)]
    short, full = tuple(items[:3]), tuple(items[3:])
    hist_ids, len_seq = data.pad_histories([short, full], data.PAD_ID, data.HISTORY_LEN)
    assert hist_ids.dtype == torch.int64 and len_seq.dtype == torch.int64
    assert hist_ids.shape == (2, 10) and len_seq.shape == (2,)
    assert hist_ids.device.type == "cpu" and hist_ids.is_contiguous() and len_seq.is_contiguous()
    assert hist_ids.tolist() == [list(short) + [1682] * 7, list(full)]
    assert len_seq.tolist() == [3, 10]
    bad_cases = [
        [],                                   # N = 0
        [()],                                 # length 0
        [tuple(range(11))],                   # length 11
        [(1, 2, 1682)],                       # id = pad id
        [(1, 2, 5000)],                       # id above pad id
        [(1, -1, 3)],                         # negative id
        [(1, True, 3)],                       # bool id
        [(1.0, 2, 3)],                        # not an int
        [short, "abc"],                       # not a sequence of ids
    ]
    for case in bad_cases:
        with pytest.raises(DataError):
            data.pad_histories(case, data.PAD_ID, data.HISTORY_LEN)
