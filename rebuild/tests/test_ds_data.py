"""M1 데이터 시험: Steam, LastFM (DESIGN.md §23.2, §23.6, §23.8 #5, #6; D33, D38, D40, D42).

The released files are read through ``load_dataset`` (sha256 first, §5.6.3 step 2). The numbers of the design are
written out here so that ``datasets.py`` is checked against them too. Synthetic frames copy the layout of the two
datasets (§23.2: ``seq`` a list of 10 plain ints, ``next`` and ``len_seq`` int64 columns) and come from
``seeding.np_rng(0, "test", <name>)`` (§17.6.2). Tests that parse the full train frames carry ``slow`` (§23.8).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ilora import data, run, seeding
from ilora.candidates import match_norm
from ilora.config import load_config
from ilora.data import DataError, RawRow
from ilora.datasets import LASTFM, MOVIELENS, SPECS, STEAM

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
CONFIG_DIR = REBUILD_DIR / "configs"
PROFILE_DIR = CONFIG_DIR / "datasets"
NAMES = ("steam", "lastfm")

# §23.2 (multi/verify/check_numbers.py): counts, offsets, the columns of the files and the fingerprints.
DESIGN = {
    "steam": SimpleNamespace(
        columns=["seq", "next", "len_seq"], num_items=3581, blocks=9550, users=9542,
        offsets={"val": 9550, "test": 10744},
        rows={"train": 151056, "val": 1194, "test": 1194}, samples={"train": 151048, "val": 1184, "test": 1186},
        dropped={"train": 8, "val": 10, "test": 8}, sequences=11938,
        fingerprints={"train": "d1468b1f36548bdc9c71dca9c918b710695e5efc9052cbe973357d5a10266ee9",
                      "val": "bc63919adc858273ecde739a6eedab8e48bd4359fb204f5179bd55e389d42b75",
                      "test": "987376eb6cb7ac1bc6fdf04b7655bbc467c18277e60bcd305b3717cb89784705"}),
    "lastfm": SimpleNamespace(
        columns=["seq", "len_seq", "next"], num_items=4606, blocks=976, users=976,
        offsets={"val": 976, "test": 1098},
        rows={"train": 50800, "val": 122, "test": 122}, samples={"train": 47872, "val": 122, "test": 122},
        dropped={"train": 2928, "val": 0, "test": 0}, sequences=1220,
        fingerprints={"train": "890287c1d4f8fc1015d2e7ebe2167e8d098b0a1ff56721a3c73afaecf0cc2640",
                      "val": "9d3440714637ebbf2b19ca9a0f8037fe8a391aa587c75adc736d4d84769bf52c",
                      "test": "13d9b892f7cdf28fc25d61458f3fda873015305cc3486870da9ed24b95a4c354"}),
}
# §23.2, §23.6 (multi/verify/check_numbers2.py): train samples of the smoke selections of seed 1234.
SMOKE_TRAIN = {"steam": {(2, 10, 10): 80, (80, 10, 10): 1385}, "lastfm": {(2, 10, 10): 41, (80, 10, 10): 3626}}
# §23.6 data.train_rows of the overfit profiles; the users left out before the 16th and the length of their last window.
OVERFIT_ROWS = {"steam": (39, 94, 129, 133, 210, 230, 316, 329, 344, 601, 748, 749, 762, 795, 812, 939),
                "lastfm": (61, 94, 168, 181, 199, 433, 525, 546, 564, 591, 617, 628, 643, 657, 680, 732)}
OVERFIT_SKIPPED = {"steam": {1: 6}, "lastfm": {0: 7, 7: 7, 12: 3, 18: 6}}


def _config(stem: str):
    cfg, _ = load_config(PROFILE_DIR / f"{stem}.yaml")
    return cfg


@lru_cache(maxsize=None)
def _released(name: str) -> SimpleNamespace:
    """``load_dataset`` of ``<name>_paper_spec`` on the released files (it checks the sha256 of the four files and all
    counts itself), then the same frames projected and parsed before the filter."""
    spec, cfg = SPECS[name], _config(f"{name}_paper_spec")
    full = data.load_dataset(cfg, REPO_ROOT)
    columns, projected, rows, users = {}, {}, {}, {}
    for split in data.SPLITS:
        frame = pd.read_pickle(REPO_ROOT / spec.data_dir / data.SPLIT_FILES[split])
        columns[split] = list(frame.columns)
        projected[split] = data.project_frame(frame, split)
        rows[split] = data.parse_rows(projected[split], split, spec=spec)
        users[split] = data.assign_users(rows[split], split, spec=spec)
    return SimpleNamespace(name=name, spec=spec, cfg=cfg, full=full, columns=columns, projected=projected, rows=rows,
                           users=users)


@pytest.fixture(scope="module", params=NAMES)
def released(request: pytest.FixtureRequest) -> SimpleNamespace:
    return _released(request.param)


# ----------------------------------------------------------------------------
# Rows, samples, blocks, users, fingerprints (§23.2, §23.8 #5)
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_rows_samples_blocks_and_user_offsets(released):
    """§23.2, D40, D42: rows, samples, blocks, users with samples, the user id offsets, and ``num_items`` of the
    catalog and of every sample (D28)."""
    spec, full, design = released.spec, released.full, DESIGN[released.name]
    assert (full.subset, full.smoke_users, full.title_mode) == ("full", None, "code")
    assert spec.expected_rows == design.rows and spec.expected_samples == design.samples
    assert (spec.train_blocks, spec.train_users) == (design.blocks, design.users)
    assert (spec.num_items, spec.pad_id) == (design.num_items, design.num_items)
    assert spec.user_offset == {"train": 0, **design.offsets}
    catalog = full.catalog
    assert (catalog.num_items, catalog.pad_id, len(catalog.titles)) == (design.num_items, design.num_items,
                                                                       design.num_items)
    for split in data.SPLITS:
        rows, samples = released.rows[split], full.samples(split)
        assert len(rows) == design.rows[split] and len(samples) == design.samples[split]
        assert design.rows[split] - design.samples[split] == design.dropped[split]
        kept = [r for r in rows if r.len_seq >= data.MIN_LEN_SEQ]        # D42: the filter only removes rows
        assert [s.row for s in samples] == [r.row for r in kept]
        assert [s.history for s in samples] == [r.history for r in kept]
        assert [s.target for s in samples] == [r.target for r in kept]
        assert all(len(s.history) == r.len_seq for s, r in zip(samples, kept))
        assert {s.num_items for s in samples} == {design.num_items}
        assert (min(len(s.history) for s in samples), max(len(s.history) for s in samples)) == (3, 10)
    train_users = released.users["train"]
    assert sorted(set(train_users)) == list(range(design.blocks))
    assert len({s.user_id for s in full.train}) == design.users
    assert design.blocks + design.rows["val"] + design.rows["test"] == design.sequences     # [LLaRA Table 1 p.7]
    for split, offset in design.offsets.items():                          # one row = one user (D14, D40)
        assert released.users[split] == tuple(offset + r.row for r in released.rows[split])
        assert [s.user_id for s in full.samples(split)] == [offset + s.row for s in full.samples(split)]
    # validation and test user ids continue after the blocks (the offsets are block-based, not user-based, §23.2)
    assert min(s.user_id for s in full.val) >= design.blocks
    assert min(s.user_id for s in full.test) >= design.offsets["test"]


@pytest.mark.slow
def test_fingerprints_and_projection(released):
    """§23.2 hash table: the int-row fingerprints of ``(seq, len_seq, next)``, with the columns in the order of the file
    (Steam stores ``seq, next, len_seq``) and after the projection."""
    design = DESIGN[released.name]
    assert released.spec.fingerprints == design.fingerprints
    for split in data.SPLITS:
        assert released.columns[split] == design.columns
        frame = released.projected[split]
        assert list(frame.columns) == ["seq", "len_seq", "next"]
        assert data.frame_fingerprint(frame, spec=released.spec) == design.fingerprints[split]


@pytest.mark.slow
def test_block_shape_follows_the_first_window_rule(released):
    """D40, §23.2 블록 모양: Steam blocks start with a window of any length 1-10 (the chain rule gives ``min(h0 + i, 10)``)
    and all later windows have length 10; LastFM blocks have the MovieLens shape (0, 1, ..., 10, 10). The real train
    rows of Steam fail under the empty-first-window rule and those of LastFM pass under both."""
    rows, users, design = released.rows["train"], released.users["train"], DESIGN[released.name]
    blocks: dict[int, list[RawRow]] = defaultdict(list)
    for row, user in zip(rows, users):
        blocks[user].append(row)
    assert sorted(blocks) == list(range(design.blocks)) and sum(map(len, blocks.values())) == design.rows["train"]
    for block in blocks.values():
        h0 = len(block[0].history)
        assert [len(r.history) for r in block] == [min(h0 + i, 10) for i in range(len(block))]
    first_lengths = Counter(len(block[0].history) for block in blocks.values())
    if released.name == "steam":
        assert set(first_lengths) == set(range(1, 11)) and not any(not block[0].history for block in blocks.values())
        assert all(len(r.history) == 10 for block in blocks.values() for r in block[1:])
        for empty_rule in (MOVIELENS, LASTFM):
            with pytest.raises(DataError, match="does not start with an empty history"):
                data.assign_users(rows, "train", spec=empty_rule)
    else:
        assert first_lengths == {0: design.blocks}
        assert data.assign_users(rows, "train", spec=STEAM) == users         # an empty start also fits the Steam rule


@pytest.mark.slow
def test_load_dataset_catalog_is_the_id2name_catalog(released):
    """``load_dataset`` builds the catalog from the ``id2name.txt`` of the dataset with the ``code`` rule (D38)."""
    assert released.full.catalog == data.build_catalog(_raw_titles(released.name), "code", spec=released.spec)


# ----------------------------------------------------------------------------
# id2name titles (D38, §23.2, §23.8 #6)
# ----------------------------------------------------------------------------
def _raw_titles(name: str) -> tuple[str, ...]:
    spec = SPECS[name]
    return data.read_raw_titles(REPO_ROOT / spec.data_dir / spec.item_file, spec=spec)


def _norm_collisions(titles) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, title in enumerate(titles):
        groups[match_norm(title)].append(i)
    return {key: ids for key, ids in groups.items() if len(ids) > 1}


def test_steam_titles_keep_their_recorded_defects():
    """§23.2 제목 결함 기록 (D38, D39): the second ``::`` of id 354 cuts the title as the released loader does, one
    ``match_norm`` pair, brackets, entities and the control character stay, nothing is repaired; UTF-8 titles decode."""
    raw = _raw_titles("steam")
    titles = data.build_catalog(raw, "code", spec=STEAM).titles
    assert len(raw) == len(titles) == 3581
    assert raw[354] == "Depths of Fear " and titles[354] == "Depths of Fear"
    assert [i for i, (r, t) in enumerate(zip(raw, titles)) if r != t] == [354]      # strip() changes only that line
    assert _norm_collisions(titles) == {"fortify": [1064, 2201]}
    assert (titles[1064], titles[2201]) == ("FORTIFY", "Fortify")
    assert "" not in titles and len(set(titles)) == 3581                            # no empty title, no exact duplicate
    assert [i for i, t in enumerate(titles) if "[" in t or "]" in t] == [2224, 2592]
    assert titles[2224] == "[the Sequence]"
    assert titles[2592] == "Corrosion: Cold Winter Waiting [Enhanced Edition]"
    assert sum("&amp;" in t for t in titles) == 37
    assert titles[3073] == "Starlight Inception™\x99"
    assert titles[1530] == "ШП" and titles[0] == "Ironbound"              # Cyrillic ШП: decoded as UTF-8
    assert [titles[i] for i in (1653, 2978, 1801, 527, 687)] == ["b", "Z", "OR", "iO", "oO"]


def test_lastfm_titles_keep_their_recorded_defects():
    """§23.2 (D38, D39): LastFM ids equal line numbers, 1016 is ``[unknown]``, brackets stay, no ``match_norm``
    collision, and UTF-8 titles decode (``Café Del Mar``, the CJK artist 嵐)."""
    raw = _raw_titles("lastfm")
    titles = data.build_catalog(raw, "code", spec=LASTFM).titles
    assert len(raw) == len(titles) == 4606 and list(raw) == list(titles)            # no surrounding spaces in this file
    assert titles[1016] == "[unknown]" and titles[3429] == "Various Artists"
    assert [i for i, t in enumerate(titles) if "[" in t or "]" in t] == [678, 690, 1016, 2985, 3568]
    assert titles[678] == "[:SITD:]" and titles[690] == "Nurzery [Rhymes]"
    assert titles[2985] == "Dungeon Elite [**]" and titles[3568] == "Superchic[k]"
    assert _norm_collisions(titles) == {} and "" not in titles and len(set(titles)) == 4606
    assert titles[2] == "Caf\xe9 Del Mar" and titles[3383] == "嵐"
    assert [titles[i] for i in (256, 608, 803, 1528, 2139, 2564, 4347, 4384)] == ["U2", "Zu", "BT", "L7", "Ef", "jj",
                                                                                  "Lu", "3"]


@pytest.mark.parametrize("name", NAMES)
def test_id2name_crlf_lf_and_final_newline(name, tmp_path):
    """§23.8 #6: CRLF and LF copies give the same titles and the same text hash (§4.6.5); a missing last newline is fine
    and an extra empty line is not (the rules of ``u.item``, §5.6.3 step 3)."""
    spec = SPECS[name]
    rel = f"{spec.data_dir}/{spec.item_file}"
    lf = (REPO_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert lf.count(b"\n") == crlf.count(b"\r\n") == spec.num_items
    paths = {}
    for label, content in (("lf", lf), ("crlf", crlf), ("no_final", lf[:-1]), ("extra", lf + b"\n")):
        paths[label] = tmp_path / label / spec.item_file
        paths[label].parent.mkdir()
        paths[label].write_bytes(content)
    expected, is_text = run.expected_inputs(name)[rel]
    assert is_text and run.content_sha256(paths["lf"], True) == run.content_sha256(paths["crlf"], True) == expected
    assert run.content_sha256(paths["lf"]) != run.content_sha256(paths["crlf"])
    titles = data.read_raw_titles(paths["lf"], spec=spec)
    assert len(titles) == spec.num_items
    assert data.read_raw_titles(paths["crlf"], spec=spec) == titles
    assert data.read_raw_titles(paths["no_final"], spec=spec) == titles
    assert (data.build_catalog(titles, "code", spec=spec)
            == data.build_catalog(data.read_raw_titles(paths["crlf"], spec=spec), "code", spec=spec))
    with pytest.raises(DataError, match=f"expected {spec.num_items} lines, got {spec.num_items + 1}"):
        data.read_raw_titles(paths["extra"], spec=spec)


def _small_spec(**changes):
    """A three-item id2name spec for hand-written files: ``read_raw_titles`` reads only the format, the count and
    ``extra_separator_ids``."""
    return dataclasses.replace(LASTFM, num_items=3, **changes)


def _write(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "id2name.txt"
    path.write_bytes(content)
    return path


def test_id2name_splits_on_newlines_only_and_decodes_utf8(tmp_path):
    """Only ``"\\n"`` ends a line (§5.6.3): U+0085 and U+2028 inside a title stay. The bytes are decoded as UTF-8
    whatever the locale is (D38), and the raw field is returned unstripped."""
    path = _write(tmp_path, "0::A\u0085B\n1::C\u2028D\n2:: 嵐 Caf\xe9 \n".encode("utf-8"))
    assert data.read_raw_titles(path, spec=_small_spec()) == ("A\x85B", "C\u2028D", " 嵐 Caf\xe9 ")


# (file content, text of the DataError) (§23.8 #6, §5.6.10)
ID2NAME_BAD = {
    "no_separator": (b"0::A\n1 B\n2::C\n", "expected 'id::title'"),
    "second_separator": (b"0::A\n1::B :: Extra\n2::C\n", "second '::'"),
    "line_number": (b"0::A\n2::B\n1::C\n", "item id"),
    "id_not_a_number": (b"0::A\nx::B\n2::C\n", "item id"),
    "id_with_a_space": (b"0::A\n 1::B\n2::C\n", "item id"),
    "empty_file": (b"", "expected 3 lines, got 0"),
    "too_few_lines": (b"0::A\n1::B\n", "expected 3 lines, got 2"),
    "too_many_lines": (b"0::A\n1::B\n2::C\n3::D\n", "expected 3 lines, got 4"),
    "extra_empty_line": (b"0::A\n1::B\n2::C\n\n", "expected 3 lines, got 4"),
    "not_utf8": (b"0::A\n1::B\xff\n2::C\n", "not valid UTF-8"),
}


@pytest.mark.parametrize("case", list(ID2NAME_BAD))
def test_id2name_rejects(case, tmp_path):
    """§23.8 #6: wrong field counts, wrong or unordered ids, wrong line counts, an empty file and non-UTF-8 bytes."""
    content, message = ID2NAME_BAD[case]
    with pytest.raises(DataError, match=message):
        data.read_raw_titles(_write(tmp_path, content), spec=_small_spec())


def test_second_separator_only_for_registered_ids(tmp_path):
    """D38: only the ids of ``extra_separator_ids`` may hold a second ``::`` (the title stays the second field); on the
    real Steam file that is id 354 and a second ``::`` anywhere else is refused."""
    path = _write(tmp_path, b"0::A :: cut\n1::B\n2::C\n")
    with pytest.raises(DataError, match="second '::'"):
        data.read_raw_titles(path, spec=_small_spec())
    assert data.read_raw_titles(path, spec=_small_spec(extra_separator_ids=(0,))) == ("A ", "B", "C")
    lines = (REPO_ROOT / STEAM.data_dir / STEAM.item_file).read_bytes().replace(b"\r\n", b"\n").split(b"\n")
    assert lines[354] == b"354::Depths of Fear :: Knossos"
    lines[353] += b" :: added"
    copy = _write(tmp_path, b"\n".join(lines))
    with pytest.raises(DataError, match=r"line 354: 3 fields; a second '::' is allowed only for item ids \[354\]"):
        data.read_raw_titles(copy, spec=STEAM)


@pytest.mark.parametrize("spec", [STEAM, LASTFM], ids=NAMES)
def test_build_catalog_strips_and_refuses_fixed(spec):
    """D38: the ``code`` rule of these datasets is ``str.strip`` only (no ``[:-7]``, no article move) and ``fixed`` does
    not exist for them; a wrong number of raw titles is refused."""
    raw = ["  Portal, The (2007)\t", *[f"item {i}" for i in range(1, spec.num_items)]]
    catalog = data.build_catalog(raw, "code", spec=spec)
    assert catalog.titles[0] == "Portal, The (2007)" and catalog.title_mode == "code"
    assert catalog.num_items == catalog.pad_id == spec.num_items
    with pytest.raises(DataError, match="title_mode"):
        data.build_catalog(raw, "fixed", spec=spec)
    with pytest.raises(DataError, match=f"expected {spec.num_items} raw titles"):
        data.build_catalog(raw[:-1], "code", spec=spec)


# ----------------------------------------------------------------------------
# Int rows (§23.2 row layout, the checks of §5.6.10 for the layout without ratings)
# ----------------------------------------------------------------------------
def _int_row(history, target, spec, *, len_seq=None) -> dict:
    """One row of the Steam and LastFM layout: 10 plain ints with right padding, ``next`` an int."""
    seq = [*history, *[spec.pad_id] * (data.HISTORY_LEN - len(history))]
    return {"seq": seq, "len_seq": max(1, len(history)) if len_seq is None else len_seq, "next": target}


def _int_block(items, spec, *, first: int) -> list[dict]:
    """A train block: row ``p >= first`` holds the history ``items[max(0, p - 10):p]`` and the target ``items[p]``."""
    return [_int_row(items[max(0, p - data.HISTORY_LEN):p], items[p], spec) for p in range(first, len(items))]


def _items(spec, name: str, n: int) -> list[int]:
    return [int(x) for x in seeding.np_rng(0, "test", name).choice(spec.num_items, size=n, replace=False)]


def _int_frame(rows: list[dict]) -> pd.DataFrame:
    """The columns of the real frames: object ``seq`` holding lists, int64 ``len_seq`` and ``next`` when valid."""
    return pd.DataFrame({"seq": [r["seq"] for r in rows], "len_seq": [r["len_seq"] for r in rows],
                         "next": [r["next"] for r in rows]})


def _pipeline(df: pd.DataFrame, split: str, spec) -> tuple:
    frame = data.project_frame(df, split)
    rows = data.parse_rows(frame, split, spec=spec)
    return data.to_samples(rows, data.assign_users(rows, split, spec=spec), split, spec=spec)


def _train_rows(spec, name: str, *, change=None) -> list[dict]:
    """Two blocks in the shape of ``spec`` (D40): from the empty window when ``first_window_empty``, otherwise from a
    6-item and a 10-item first window. ``change(block, items)`` edits the items of a block before its windows exist."""
    first_a, first_b = (0, 0) if spec.first_window_empty else (6, 10)
    items_a, items_b = _items(spec, f"{name}/a", 14), _items(spec, f"{name}/b", 12)
    if change is not None:
        change(0, items_a)
        change(1, items_b)
    return _int_block(items_a, spec, first=first_a) + _int_block(items_b, spec, first=first_b)


@pytest.mark.parametrize("spec", [STEAM, LASTFM], ids=NAMES)
def test_int_pipeline_accepts_valid_frames(spec):
    """Control for the rejections below: the synthetic int frames load through the whole pipeline; item ids reach
    ``num_items - 1`` (D28), the padding id only pads, ``len_seq < 3`` rows are dropped (D42)."""
    rows = _train_rows(spec, "int_valid")
    top = spec.num_items - 1
    assert top > data.NUM_ITEMS
    rows[-1]["next"] = top
    samples = _pipeline(_int_frame(rows), "train", spec)
    kept = [r for r in rows if r["len_seq"] >= 3]
    assert len(samples) == len(kept) and samples[-1].target == top
    assert all(s.num_items == spec.num_items for s in samples)
    eval_items = [_items(spec, f"int_valid/eval{i}", 8) for i in range(3)]
    eval_rows = [_int_row(items[:-1], items[-1], spec) for items in eval_items]
    for split in ("val", "test"):
        result = _pipeline(_int_frame(eval_rows), split, spec)
        assert [s.user_id for s in result] == [spec.user_offset[split] + r for r in range(3)]
        assert [s.history for s in result] == [tuple(r["seq"][:7]) for r in eval_rows]


def _bad_int_rows(case: str, spec) -> list[dict]:
    if case == "duplicate_in_history":                    # item 1 repeats item 0 inside a window kept by the filter
        return _train_rows(spec, "int_bad", change=lambda b, items: items.__setitem__(1, items[0]) if b == 0 else None)
    if case == "target_in_history":
        return _train_rows(spec, "int_bad", change=lambda b, items: items.__setitem__(7, items[2]) if b == 0 else None)
    rows = _train_rows(spec, "int_bad")
    row = rows[5]                                         # at least 5 real items in every shape
    pad, seq = spec.pad_id, row["seq"]
    if case == "left_padding":
        row["seq"] = [pad] + seq[:9]
    elif case == "middle_padding":
        row["seq"] = seq[:2] + [pad] + seq[2:9]
    elif case == "len_seq_mismatch":
        row["len_seq"] = 2
    elif case == "item_above_padding_id":
        seq[0] = spec.num_items + 1
    elif case == "item_negative":
        seq[0] = -1
    elif case == "target_is_padding_id":
        row["next"] = pad
    elif case == "target_above_padding_id":
        row["next"] = pad + 1
    elif case == "float_item":
        seq[0] = 1.5
    elif case == "bool_item":
        seq[0] = True
    elif case == "str_item":
        seq[0] = "7"
    elif case == "pair_item":
        seq[0] = (7, 4)
    elif case == "float_next":
        row["next"] = 1.5
    elif case == "bool_next":
        row["next"] = True
    elif case == "pair_next":
        row["next"] = (7, 4)
    elif case == "float_len_seq":
        row["len_seq"] = 6.5
    elif case == "seq_too_short":
        row["seq"] = seq[:9]
    elif case == "seq_is_a_tuple":
        row["seq"] = tuple(seq)
    else:
        raise AssertionError(case)
    return rows


INT_CASES = {
    "left_padding": "padding", "middle_padding": "padding", "len_seq_mismatch": "len_seq",
    "item_above_padding_id": "item id", "item_negative": "item id -1", "target_is_padding_id": "next: item id",
    "target_above_padding_id": "next: item id", "float_item": "integer item id", "bool_item": "integer item id",
    "str_item": "integer item id", "pair_item": "integer item id", "float_next": "next: expected an integer",
    "bool_next": "next: expected an integer", "pair_next": "next: expected an integer",
    "float_len_seq": "len_seq must be an integer", "seq_too_short": "list of 10", "seq_is_a_tuple": "list of 10",
    "duplicate_in_history": "repeated item", "target_in_history": "is in the history",
}


@pytest.mark.parametrize("spec", [STEAM, LASTFM], ids=NAMES)
@pytest.mark.parametrize("case", list(INT_CASES))
def test_int_rows_rejected(case, spec):
    """§5.6.10 for the int layout: padding only on the right, ``len_seq == max(1, k)``, ids inside the catalog (the
    padding id is never a target), non-integers (float, bool, str, an (item, rating) pair), ``Sample`` rules."""
    with pytest.raises(DataError, match=INT_CASES[case]):
        _pipeline(_int_frame(_bad_int_rows(case, spec)), "train", spec)


def test_int_and_pair_layouts_do_not_mix():
    """§23.2: ``spec.has_ratings`` picks the layout. An int frame fails under the MovieLens layout, a pair frame under
    the int layout, in ``parse_rows`` and in ``frame_fingerprint``; int rows have no rating to check."""
    int_df = data.project_frame(_int_frame(_train_rows(STEAM, "layouts")), "train")
    with pytest.raises(DataError, match=r"expected an \(item, rating\) tuple"):
        data.parse_rows(int_df, "train", spec=MOVIELENS)
    with pytest.raises(DataError, match="fingerprint"):
        data.frame_fingerprint(int_df, spec=MOVIELENS)
    pair = (np.int64(3), np.int64(4))
    pair_df = pd.DataFrame({"seq": [[pair] * 3 + [(data.PAD_ID, data.PAD_RATING)] * 7], "len_seq": [3],
                            "next": [(np.int64(7), np.int64(5))]})
    with pytest.raises(DataError, match="integer item id"):
        data.parse_rows(pair_df, "train", spec=STEAM)
    with pytest.raises(DataError, match="fingerprint"):
        data.frame_fingerprint(pair_df, spec=STEAM)


def test_int_fingerprint_line_format():
    """§23.2: one ``json.dumps([p, [item x 10], len_seq, next], separators=(",", ":"))`` per row, joined by ``"\\n"``,
    sha256 of the UTF-8 text; the column order of the file does not matter."""
    rows = _train_rows(STEAM, "fingerprint")
    frame = data.project_frame(_int_frame(rows), "train")
    lines = [json.dumps([p, r["seq"], r["len_seq"], r["next"]], separators=(",", ":")) for p, r in enumerate(rows)]
    assert data.frame_fingerprint(frame, spec=STEAM) == hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    swapped = data.project_frame(_int_frame(rows)[["seq", "next", "len_seq"]], "train")
    assert data.frame_fingerprint(swapped, spec=STEAM) == data.frame_fingerprint(frame, spec=STEAM)


# ----------------------------------------------------------------------------
# Pseudo users (D14, D40)
# ----------------------------------------------------------------------------
def test_assign_users_steam_frame_needs_the_steam_rule():
    """D40: a train frame whose blocks start with a non-empty window is refused under the empty first-window rule
    (MovieLens, LastFM) and accepted under the Steam rule: window lengths ``min(h0 + i, 10)``, chain rule unchanged, a
    chain break starts a new block. A block may still start empty under the Steam rule."""
    df = data.project_frame(_int_frame(_train_rows(STEAM, "assign_users")), "train")
    rows = data.parse_rows(df, "train", spec=STEAM)
    assert [len(r.history) for r in rows] == [6, 7, 8, 9, 10, 10, 10, 10, 10, 10]
    assert data.assign_users(rows, "train", spec=STEAM) == (0,) * 8 + (1,) * 2
    for empty_rule in (MOVIELENS, LASTFM):
        with pytest.raises(DataError, match="does not start with an empty history"):
            data.assign_users(rows, "train", spec=empty_rule)
    empty_df = data.project_frame(_int_frame(_int_block(_items(STEAM, "assign_empty", 13), STEAM, first=0)), "train")
    empty_start = data.parse_rows(empty_df, "train", spec=STEAM)
    assert [len(r.history) for r in empty_start] == [min(i, 10) for i in range(13)]
    assert data.assign_users(empty_start, "train", spec=STEAM) == (0,) * 13


def test_assign_users_offsets_of_validation_and_test():
    """§23.2, D14: one row = one user, with the offsets of the spec (the blocks of train, then the validation rows)."""
    rows = [RawRow(row=r, len_seq=3, history=(1, 2, 3), target=4) for r in range(4)]
    for spec, (val, test) in ((MOVIELENS, (754, 848)), (STEAM, (9550, 10744)), (LASTFM, (976, 1098))):
        assert data.assign_users(rows, "val", spec=spec) == tuple(val + r for r in range(4))
        assert data.assign_users(rows, "test", spec=spec) == tuple(test + r for r in range(4))


# ----------------------------------------------------------------------------
# Smoke subsets and overfit rows (§23.2, §23.6, D14)
# ----------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("sizes", [(2, 10, 10), (80, 10, 10)], ids=["2_users", "80_users"])
def test_smoke_selection_counts(released, sizes):
    """§23.2, §23.6: train samples of the selections of seed 1234 (80 and 41; 1,385 and 3,626), every window of the
    picked users in row order, nothing renumbered; validation and test follow ``np_rng(seed, "smoke_users", split)``
    (§5.6.7); the same call gives the same subset."""
    spec, full = released.spec, released.full
    smoke = data.select_smoke(full, 1234, sizes, train_rows=(), spec=spec)
    users = {s.user_id for s in smoke.train}
    assert len(smoke.train) == SMOKE_TRAIN[released.name][sizes] and len(users) == sizes[0]
    assert smoke.train == tuple(s for s in full.train if s.user_id in users)
    assert (smoke.subset, smoke.smoke_users) == ("smoke", sizes) and smoke.catalog is full.catalog
    assert set(smoke.val) <= set(full.val) and set(smoke.test) <= set(full.test)
    for split, n in (("val", sizes[1]), ("test", sizes[2])):
        pick = seeding.np_rng(1234, "smoke_users", split).choice(len(full.samples(split)), size=n, replace=False)
        assert smoke.samples(split) == tuple(full.samples(split)[i] for i in sorted(int(x) for x in pick))
    assert data.select_smoke(full, 1234, sizes, train_rows=(), spec=spec) == smoke


@pytest.mark.slow
@pytest.mark.parametrize("profile,sizes", [("smoke_cpu", (2, 10, 10)), ("smoke_gpu_100", (80, 10, 10))],
                         ids=["smoke_cpu", "smoke_gpu_100"])
def test_smoke_profiles_through_load_dataset(released, profile, sizes):
    """§5.6.3 step 8: ``load_dataset`` of the §23.6 smoke profiles equals ``select_smoke`` on the full data."""
    cfg = _config(f"{released.name}_{profile}")
    assert (cfg.data.subset, cfg.data.smoke_users, cfg.data.train_rows, cfg.run.seed) == ("smoke", sizes, (), 1234)
    assert (data.load_dataset(cfg, REPO_ROOT)
            == data.select_smoke(released.full, 1234, sizes, train_rows=(), spec=released.spec))


@pytest.mark.slow
def test_overfit_rows_are_the_last_windows(released):
    """§23.6: ``data.train_rows`` are, in user order, the last windows of the first 16 users whose last window has 10
    items, with 16 distinct targets; the users left out are the ones the table lists. ``load_dataset`` of the overfit
    profile gives the same subset."""
    name, spec, full, rows = released.name, released.spec, released.full, OVERFIT_ROWS[released.name]
    cfg = _config(f"{name}_overfit_gpu")
    assert (cfg.data.subset, cfg.data.train_rows, cfg.data.smoke_users) == ("smoke", rows, (16, 10, 10))
    sub = data.select_smoke(full, cfg.run.seed, (16, 10, 10), train_rows=rows, spec=spec)
    assert [s.row for s in sub.train] == list(rows) and sub.smoke_users == (16, 10, 10)
    assert all(len(s.history) == 10 for s in sub.train)
    assert len({s.user_id for s in sub.train}) == 16 and len({s.target for s in sub.train}) == 16
    last = {}
    for sample in full.train:
        last[sample.user_id] = sample
    eligible = [u for u in sorted(last) if len(last[u].history) == 10]
    assert [last[u].row for u in eligible[:16]] == list(rows)
    skipped = {u: len(last[u].history) for u in sorted(last) if u < eligible[15] and u not in eligible}
    assert skipped == OVERFIT_SKIPPED[name]
    assert data.load_dataset(cfg, REPO_ROOT) == sub


@pytest.mark.slow
def test_smoke_size_limits_come_from_the_data(released):
    """§5.6.7, §23.2: the limits are the users with samples (Steam 9,542, not its 9,550 blocks) and the filtered sizes
    of validation and test; the upper limits themselves are allowed."""
    spec, full = released.spec, released.full
    n_val, n_test = len(full.val), len(full.test)
    assert (n_val, n_test) == (spec.expected_samples["val"], spec.expected_samples["test"])
    everything = data.select_smoke(full, 1234, (spec.train_users, n_val, n_test), train_rows=(), spec=spec)
    assert (everything.train, everything.val, everything.test) == (full.train, full.val, full.test)
    for sizes in [(0, 10, 10), (spec.train_users + 1, 10, 10), (80, 0, 10), (80, n_val + 1, 10), (80, 10, n_test + 1),
                  (80, 10), (True, 10, 10)]:
        with pytest.raises(DataError, match="smoke size"):
            data.select_smoke(full, 1234, sizes, train_rows=(), spec=spec)


@pytest.mark.slow
def test_load_dataset_range_checks_smoke_users_with_the_spec():
    """§5.6.8, §23.5: ``load_dataset`` checks ``data.smoke_users`` against the dataset's users, also for ``full``."""
    cfg = _config("lastfm_paper_spec")
    broken = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, smoke_users=(977, 10, 10)))
    with pytest.raises(DataError, match=r"smoke size for train must be in 1\.\.976, got 977"):
        data.load_dataset(broken, REPO_ROOT)


# ----------------------------------------------------------------------------
# File hashes and the checks of load_dataset (§5.6.3)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_changed_file_hash_is_refused_before_unpickling(name, tmp_path, monkeypatch):
    """§5.6.3 step 2, D33: one changed byte in any of the four files fails with ``DataError`` before ``pd.read_pickle``
    runs (the expected values of ``run.expected_inputs``); a missing file fails the same way."""
    spec, cfg = SPECS[name], _config(f"{name}_paper_spec")
    names = (*data.SPLIT_FILES.values(), spec.item_file)
    target_dir = tmp_path / spec.data_dir
    target_dir.mkdir(parents=True)
    for file_name in names:
        shutil.copyfile(REPO_ROOT / spec.data_dir / file_name, target_dir / file_name)
    calls = []
    monkeypatch.setattr(pd, "read_pickle", lambda *args, **kwargs: calls.append(args))
    for file_name in names:
        path = target_dir / file_name
        original = path.read_bytes()
        changed = bytearray(original)
        changed[len(changed) // 2] ^= 0x01
        path.write_bytes(bytes(changed))
        rel = f"{spec.data_dir}/{file_name}"
        expected = run.expected_inputs(name)[rel][0]
        with pytest.raises(DataError, match=rf"{rel}: sha256 [0-9a-f]{{64}} != expected {expected}"):
            data.load_dataset(cfg, tmp_path)
        assert calls == [], file_name
        path.write_bytes(original)
    (target_dir / spec.item_file).unlink()
    with pytest.raises(DataError, match=rf"{spec.item_file}: file not found"):
        data.load_dataset(cfg, tmp_path)
    assert calls == []


@pytest.mark.slow
@pytest.mark.parametrize("profile,calls", [("paper_spec", 1), ("datasets/steam_paper_spec", 0),
                                           ("datasets/lastfm_paper_spec", 0)], ids=["movielens", "steam", "lastfm"])
def test_title_report_only_for_the_uitem_format(profile, calls, monkeypatch):
    """§23.5: ``title_report`` describes the ``u.item`` normalization (§5.6.6); the id2name datasets skip it (D38)."""
    cfg, _ = load_config(CONFIG_DIR / f"{profile}.yaml")
    seen = []
    real = data.title_report
    monkeypatch.setattr(data, "title_report", lambda raw: (seen.append(len(raw)), real(raw))[1])
    assert data.load_dataset(cfg, REPO_ROOT).subset == "full"
    assert seen == [data.NUM_ITEMS] * calls


# (profile under configs/, changes to its spec, text of the DataError), §5.6.3 step 7, D40.
TRAIN_CHECK = [
    ("datasets/lastfm_paper_spec", {"train_blocks": 977},
     "976 blocks and 976 users with samples, expected 977 blocks and 976 users with samples$"),
    ("datasets/lastfm_paper_spec", {"train_users": 975},
     "976 blocks and 976 users with samples, expected 976 blocks and 975 users with samples$"),
    ("paper_spec", {"train_blocks": 755, "train_users": 755}, "754 blocks and 754 users with samples, expected 755$"),
]


@pytest.mark.slow
@pytest.mark.parametrize("profile,changes,message", TRAIN_CHECK, ids=["blocks", "users", "movielens_text"])
def test_train_check_compares_blocks_and_users(profile, changes, message, monkeypatch):
    """§23.5, D40: blocks are compared with ``train_blocks`` and users with samples with ``train_users`` (Steam 9,550
    against 9,542); with equal expected numbers the error keeps the MovieLens text of §5.6.3 step 7."""
    cfg, _ = load_config(CONFIG_DIR / f"{profile}.yaml")
    spec = dataclasses.replace(data.spec_of(cfg), **changes)
    monkeypatch.setattr(data, "spec_of", lambda _cfg: spec)
    with pytest.raises(DataError, match=message):
        data.load_dataset(cfg, REPO_ROOT)
