"""M1 데이터 (DESIGN.md §5, §3.1 ``ItemCatalog``·``Sample``·``LoadedData``·``pad_histories``).

Reads the four MovieLens100K files that LLaRA distributes and iLoRA uses
unchanged (``data/ref/movielens/``: ``train_data.df``, ``Val_data.df``,
``Test_data.df``, ``u.item``) and turns them into an ``ItemCatalog`` and
``Sample`` tuples. ``load_dataset(cfg, repo_root)`` is the only way the other
modules get samples (§5.6.12):

1. the sha256 of all four files is compared with ``run.EXPECTED_INPUTS``
   before anything is unpickled (pickle can run code, §5.6.3 step 2),
2. ``u.item`` titles are normalized as the released code does (``code``,
   paper_spec) or with the minimal fixes of ``variant_fixed_titles``
   (``fixed``), D20, §5.6.6,
3. each frame is projected to ``(seq, len_seq, next)`` (the ``cluster``
   column of the iLoRA copy is never read), its content fingerprint is
   checked, rows are parsed strictly, pseudo users are assigned (D14,
   §5.6.5), rows with ``len_seq < 3`` are dropped and the counts are checked,
4. ``data.subset == smoke`` selects the smoke subset (§5.6.7).

pandas objects never leave this module, no file is written, and torch is
used only by ``pad_histories`` (§5.6.10, §5.6.13). Every problem raises
``DataError`` at once; nothing is retried, repaired or switched to another
file (§5.6.10).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Optional, Sequence

import numpy as np
import pandas as pd

from . import run
from .seeding import np_rng

if TYPE_CHECKING:
    import torch

    from .sections import Config

logger = logging.getLogger("ilora.data")

# ----------------------------------------------------------------------------
# Constants (§5.6.1, §5.6.8: fixed rules, not config keys)
# ----------------------------------------------------------------------------
NUM_ITEMS: Final = 1682              # [LLaRA Table 1 p.7] 1,682 items, ids 0-1681
PAD_ID: Final = 1682                 # [코드] U data/movielens_data.py:18
PAD_RATING: Final = 0                # [코드] U data/movielens_data.py:19
HISTORY_LEN: Final = 10              # [LLaRA §5.1.1 p.7] "retain the last 10 interactions"
MIN_LEN_SEQ: Final = 3               # [코드] U data/movielens_data.py:79 (no paper evidence)
SPLITS: Final = ("train", "val", "test")
DATA_DIR: Final = "data/ref/movielens"   # repository-relative, [코드] U train_movielens.sh:8
SPLIT_FILES: Final = {"train": "train_data.df", "val": "Val_data.df", "test": "Test_data.df"}  # [코드] :52-57
ITEM_FILE: Final = "u.item"          # [코드] U data/movielens_data.py:70
USER_OFFSET: Final = {"train": 0, "val": 754, "test": 848}   # D14, §5.6.5

ITEM_ENCODING: Final = "ISO-8859-1"  # [코드] U data/movielens_data.py:71
ITEM_FIELDS: Final = 24              # fields per u.item line (computed, §5.2)
RATINGS: Final = (1, 2, 3, 4, 5)     # allowed rating values; the files hold only 3, 4, 5 (§5.6.2)
TITLE_MODES: Final = ("code", "fixed")   # D20
SUBSETS: Final = ("full", "smoke")       # D14
FRAME_COLUMNS: Final = ("seq", "len_seq", "next")
CLUSTER_COLUMN: Final = "cluster"    # iLoRA copy of train_data.df only; never read [SPEC §0.8, §7]

# Expected counts (§5.6.2, §5.6.3 step 7, §5.6.5).
EXPECTED_ROWS: Final = {"train": 68388, "val": 94, "test": 95}
EXPECTED_SAMPLES: Final = {"train": 66126, "val": 94, "test": 95}
TRAIN_USERS: Final = 754

# Content fingerprints of (seq, len_seq, next), §5.6.9 (design/tmp/m1_fingerprint.py). The train
# value is the same for the iLoRA copy (sha256 bac4c88b...) and the LLaRA copy (sha256 a73b119d...).
EXPECTED_FINGERPRINTS: Final = {
    "train": "85bafab832f1f78e5095b1fd880a418097054d3ea7dbb7d4fe9e1528f019aa99",
    "val": "de7ac66b6703dc6935e1339183c00e08af2ad40f6d0877e7556b483a9aa65523",
    "test": "01636285c16faf95cc7136bc937b9aa315c66de5e23fed5efec0e2b2e84dff09",
}

# title_report categories (§5.6.6), the defects of the code normalization listed in §5.5.
TITLE_REPORT_CATEGORIES: Final = ("article_an", "article_double_the", "article_in_parentheses", "empty",
                                  "truncated", "trailing_space", "duplicate_title")

# [코드] U data/movielens_data.py:62, order kept (", A" is tested before ", An", D20).
_CODE_ARTICLES: Final = (", The", ", A", ", An")
# variant_fixed_titles rules, §5.6.6 [결정]
_FIXED_YEAR_RE: Final = re.compile(r"(?P<body>.+) \((?P<year>\d{4})\)(?: \(V\))?")
_FIXED_ARTICLE_RE: Final = re.compile(r"(?P<main>.+?), (?P<art>The|A|An)(?P<alias> \(.+\))?")
_MOVIE_ID_RE: Final = re.compile(r"[0-9]+")


class DataError(RuntimeError):
    """Data input or content does not match the design (§5.6.10)."""


def _is_int(value: Any) -> bool:
    """Python or numpy integer, bool excluded (the released items are numpy.int64, padding is int)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_py_int(value: Any) -> bool:
    """Python int, bool excluded (the contract values of §3.1 are Python ints)."""
    return isinstance(value, int) and not isinstance(value, bool)


# ----------------------------------------------------------------------------
# Contract values (§3.1, §5.6.1)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ItemCatalog:
    """§3.1 ``ItemCatalog``: ``titles[i]`` is the normalized title of item id ``i`` (0-1681).

    ``title_mode`` is ``code`` (paper_spec) or ``fixed`` (``variant_fixed_titles``), D20;
    M11 and M12 record which titles were judged.
    """
    titles: tuple[str, ...]
    title_mode: str
    num_items: int                       # always 1682
    pad_id: int                          # always 1682

    def __post_init__(self) -> None:
        if not isinstance(self.titles, tuple):
            raise DataError(f"ItemCatalog.titles must be a tuple, got {type(self.titles).__name__}")
        if self.num_items != NUM_ITEMS or not _is_py_int(self.num_items):
            raise DataError(f"ItemCatalog.num_items must be {NUM_ITEMS}, got {self.num_items!r}")
        if self.pad_id != PAD_ID or not _is_py_int(self.pad_id):
            raise DataError(f"ItemCatalog.pad_id must be {PAD_ID}, got {self.pad_id!r}")
        if len(self.titles) != self.num_items:
            raise DataError(f"ItemCatalog needs {self.num_items} titles, got {len(self.titles)}")
        bad = [i for i, t in enumerate(self.titles) if not isinstance(t, str)]
        if bad:
            raise DataError(f"ItemCatalog titles must be str, item ids {bad[:10]} are not")
        if self.title_mode not in TITLE_MODES:
            raise DataError(f"ItemCatalog.title_mode must be one of {TITLE_MODES}, got {self.title_mode!r}")


@dataclass(frozen=True)
class Sample:
    """§3.1 ``Sample``: one (history, next item) window.

    ``row`` is the row number of the released ``.df`` file before the
    ``len_seq`` filter (§5.6.4), ``user_id`` the pseudo user (D14),
    ``history`` the unpadded window, oldest first, 3-10 items.
    """
    split: str
    row: int
    user_id: int
    history: tuple[int, ...]
    target: int

    def __post_init__(self) -> None:
        where = f"sample {self.split!r} row {self.row!r}"
        if self.split not in SPLITS:
            raise DataError(f"{where}: split must be one of {SPLITS}")
        if not _is_py_int(self.row) or self.row < 0:
            raise DataError(f"{where}: row must be an int >= 0")
        if not _is_py_int(self.user_id) or self.user_id < 0:
            raise DataError(f"{where}: user_id must be an int >= 0, got {self.user_id!r}")
        if not isinstance(self.history, tuple):
            raise DataError(f"{where}: history must be a tuple, got {type(self.history).__name__}")
        if not MIN_LEN_SEQ <= len(self.history) <= HISTORY_LEN:
            raise DataError(f"{where}: history length {len(self.history)} is outside "
                            f"[{MIN_LEN_SEQ}, {HISTORY_LEN}]")
        for name, value in [*((f"history[{j}]", x) for j, x in enumerate(self.history)), ("target", self.target)]:
            if not _is_py_int(value) or not 0 <= value < NUM_ITEMS:
                raise DataError(f"{where}: {name} = {value!r} is not an item id (int 0-{NUM_ITEMS - 1}, bool excluded)")
        if len(set(self.history)) != len(self.history):
            raise DataError(f"{where}: history has a repeated item: {self.history}")
        if self.target in self.history:
            raise DataError(f"{where}: target {self.target} is in the history {self.history}")

    @property
    def sample_id(self) -> str:
        return f"{self.split}:{self.row}"


@dataclass(frozen=True)
class RawRow:
    """One parsed row before the ``len_seq`` filter (§5.6.1). ``history`` may be empty."""
    row: int
    len_seq: int
    history: tuple[int, ...]
    target: int


@dataclass(frozen=True)
class LoadedData:
    """§3.1 ``LoadedData``: the catalog and the samples of each split in ``row`` order.

    ``subset`` is ``full`` or ``smoke``; ``smoke_users`` is the chosen
    (train, val, test) user counts for smoke and None for full (M11 §15.6.2).
    """
    catalog: ItemCatalog
    train: tuple[Sample, ...]
    val: tuple[Sample, ...]
    test: tuple[Sample, ...]
    subset: str
    smoke_users: Optional[tuple[int, int, int]]

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, ItemCatalog):
            raise DataError(f"LoadedData.catalog must be an ItemCatalog, got {type(self.catalog).__name__}")
        if self.subset not in SUBSETS:
            raise DataError(f"LoadedData.subset must be one of {SUBSETS}, got {self.subset!r}")
        if self.subset == "full" and self.smoke_users is not None:
            raise DataError(f"full data has smoke_users None, got {self.smoke_users!r}")
        if self.subset == "smoke" and not (isinstance(self.smoke_users, tuple) and len(self.smoke_users) == 3
                                           and all(_is_py_int(n) for n in self.smoke_users)):
            raise DataError(f"smoke data needs smoke_users = (train, val, test) ints, got {self.smoke_users!r}")
        for split in SPLITS:
            samples = getattr(self, split)
            if not isinstance(samples, tuple):
                raise DataError(f"LoadedData.{split} must be a tuple, got {type(samples).__name__}")
            previous = -1
            for s in samples:
                if not isinstance(s, Sample) or s.split != split:
                    raise DataError(f"LoadedData.{split} holds {s!r}, not a {split} Sample")
                if s.row <= previous:
                    raise DataError(f"LoadedData.{split} is not in strictly ascending row order "
                                    f"(row {s.row} after {previous})")
                previous = s.row

    def samples(self, split: str) -> tuple[Sample, ...]:
        if split not in SPLITS:
            raise DataError(f"unknown split {split!r}, expected one of {SPLITS}")
        return getattr(self, split)

    @property
    def title_mode(self) -> str:
        return self.catalog.title_mode


# ----------------------------------------------------------------------------
# u.item and titles (§5.6.3 step 3, §5.6.6)
# ----------------------------------------------------------------------------
def read_raw_titles(path: Path) -> tuple[str, ...]:
    """Raw ``u.item`` titles (field 1), index = item id = movie_id - 1.

    Text mode with universal newlines and ISO-8859-1 (``U data/movielens_data.py:71``),
    so a CRLF checkout (Windows, core.autocrlf) and an LF checkout parse the same
    (§4.6.5). The text is split on ``"\\n"`` only, not ``str.splitlines()``, which
    would also break at ISO-8859-1 ``\\x85`` and other separators (§5.6.3).
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"{path}: file not found")
    with open(path, "r", encoding=ITEM_ENCODING, newline=None) as f:
        text = f.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]                      # only the one empty element after the last newline
    if len(lines) != NUM_ITEMS:
        raise DataError(f"{path}: expected {NUM_ITEMS} lines, got {len(lines)}")
    titles = []
    for i, line in enumerate(lines):
        fields = line.split("|")                # [코드] U data/movielens_data.py:73
        if len(fields) != ITEM_FIELDS:
            raise DataError(f"{path} line {i + 1}: expected {ITEM_FIELDS} '|' fields, got {len(fields)}")
        if not _MOVIE_ID_RE.fullmatch(fields[0]) or int(fields[0]) != i + 1:
            raise DataError(f"{path} line {i + 1}: movie_id {fields[0]!r} != line number {i + 1}")
        titles.append(fields[1])                # item id = movie_id - 1, [코드] :74
    return tuple(titles)


def _get_mv_title(s: str) -> str:
    # [코드] U data/movielens_data.py:61-66, unchanged (defects kept, D20)
    for sub_s in _CODE_ARTICLES:
        if sub_s in s:
            return sub_s[2:] + " " + s.replace(sub_s, "")
    return s


def code_title(raw: str) -> str:
    """Released normalization (paper_spec, D20): ``get_mv_title(raw[:-7])``, ``U data/movielens_data.py:74``."""
    return _get_mv_title(raw[:-7])


def fixed_title(raw: str) -> str:
    """``variant_fixed_titles`` normalization (§5.6.6) [결정].

    1. ``strip()``; 2. drop a final `` (YYYY)`` with an optional `` (V)``
    (item 266 ``unknown`` has no year and stays); 3. move a final ``, The``/``, A``/``, An``
    (optionally followed by one parenthesized alias) to the front. Articles inside
    parentheses and foreign articles (``, La``) are not moved.
    """
    s = raw.strip()
    m = _FIXED_YEAR_RE.fullmatch(s)
    if m:
        s = m.group("body")
    m = _FIXED_ARTICLE_RE.fullmatch(s)
    if m:
        s = m.group("art") + " " + m.group("main") + (m.group("alias") or "")
    return s


def build_catalog(raw_titles: Sequence[str], title_mode: str) -> ItemCatalog:
    """``ItemCatalog`` from the raw ``u.item`` titles with ``code`` or ``fixed`` normalization (D20)."""
    if title_mode not in TITLE_MODES:
        raise DataError(f"title_mode must be one of {TITLE_MODES}, got {title_mode!r}")
    if len(raw_titles) != NUM_ITEMS:
        raise DataError(f"expected {NUM_ITEMS} raw titles, got {len(raw_titles)}")
    bad = [i for i, t in enumerate(raw_titles) if not isinstance(t, str)]
    if bad:
        raise DataError(f"raw titles must be str, item ids {bad[:10]} are not")
    normalize = code_title if title_mode == "code" else fixed_title
    return ItemCatalog(titles=tuple(normalize(t) for t in raw_titles), title_mode=title_mode,
                       num_items=NUM_ITEMS, pad_id=PAD_ID)


def _paren_depth(text: str, pos: int) -> int:
    head = text[:pos]
    return head.count("(") - head.count(")")


def title_report(raw_titles: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Item ids per defect of the ``code`` normalization (§5.5, §5.6.6), keys ``TITLE_REPORT_CATEGORIES``.

    - ``article_an``: ``", An"`` titles, corrupted because ``", A"`` is tested first (435, 601, 965, 1226)
    - ``article_double_the``: ``", The"`` more than once, ``replace`` removes all (176)
    - ``article_in_parentheses``: the moved article stands inside parentheses (1004, 1018)
    - ``empty``: empty title (266 ``unknown`` has no year, ``[:-7]`` leaves nothing)
    - ``truncated``: the 7 removed characters were not the year and part of the suffix stays (1411)
    - ``trailing_space``: the raw title ends with ``"(YYYY) "`` (1127, 1200, 1634)
    - ``duplicate_title``: ids whose normalized title another id also has (22 titles, 45 ids)
    """
    out: dict[str, list[int]] = {name: [] for name in TITLE_REPORT_CATEGORIES}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, raw in enumerate(raw_titles):
        cut = raw[:-7]
        title = _get_mv_title(cut)
        groups[title].append(i)
        article = next((a for a in _CODE_ARTICLES if a in cut), None)
        if article == ", A" and ", An" in cut:
            out["article_an"].append(i)
        if cut.count(", The") > 1:
            out["article_double_the"].append(i)
        if article is not None:
            positions = [m.start() for m in re.finditer(re.escape(article), cut)]
            if any(_paren_depth(cut, pos) > 0 for pos in positions):
                out["article_in_parentheses"].append(i)
        if title == "":
            out["empty"].append(i)
        if title != title.rstrip():
            out["trailing_space"].append(i)
        m = _FIXED_YEAR_RE.fullmatch(raw.strip())
        body = m.group("body") if m else raw.strip()
        if cut != "" and cut.rstrip() != body.rstrip():
            out["truncated"].append(i)
    out["duplicate_title"] = sorted(i for ids in groups.values() if len(ids) > 1 for i in ids)
    return {name: tuple(ids) for name, ids in out.items()}


# ----------------------------------------------------------------------------
# Released frames (§5.6.2-§5.6.5, §5.6.9)
# ----------------------------------------------------------------------------
def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise DataError(f"unknown split {split!r}, expected one of {SPLITS}")


def project_frame(obj: object, split: str) -> pd.DataFrame:
    """Check the unpickled object and keep only ``(seq, len_seq, next)`` (§5.6.3 step 4).

    It must be a ``DataFrame`` with ``RangeIndex(0, len, 1)``. Allowed columns:
    train ``{seq, len_seq, next, cluster}`` (iLoRA copy) or ``{seq, len_seq, next}``
    (LLaRA copy, for fingerprint comparisons and tests), validation and test
    ``{seq, len_seq, next}``. The ``cluster`` values are never read.
    """
    _check_split(split)
    if not isinstance(obj, pd.DataFrame):
        raise DataError(f"{split}: expected a pandas DataFrame, got {type(obj).__name__}")
    index = obj.index
    if not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1
            and index.stop == len(obj)):
        raise DataError(f"{split}: index must be RangeIndex(0, {len(obj)}, 1), got {index!r}")
    columns = list(obj.columns)
    allowed = [set(FRAME_COLUMNS)]
    if split == "train":
        allowed.insert(0, set(FRAME_COLUMNS) | {CLUSTER_COLUMN})
    if len(set(columns)) != len(columns) or set(columns) not in allowed:
        raise DataError(f"{split}: columns {columns} are not one of {[sorted(a) for a in allowed]}")
    return obj.loc[:, list(FRAME_COLUMNS)].copy()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content fingerprint of ``(seq, len_seq, next)`` (§5.6.9).

    One line per unfiltered row ``p``: ``json.dumps([p, [[item, rating] x 10], len_seq,
    [next_item, next_rating]], separators=(",", ":"))``; lines joined by ``"\\n"``,
    sha256 of the UTF-8 text. Other columns (``cluster``) are not included.
    """
    lines = []
    for p, (seq, len_seq, nxt) in enumerate(zip(df["seq"].tolist(), df["len_seq"].tolist(),
                                                 df["next"].tolist())):
        try:
            entry = [p, [[int(item), int(rating)] for item, rating in seq], int(len_seq),
                     [int(nxt[0]), int(nxt[1])]]
        except (TypeError, ValueError, IndexError) as exc:
            raise DataError(f"fingerprint: row {p} is not (seq, len_seq, next) of integers: {exc}") from exc
        lines.append(json.dumps(entry, separators=(",", ":")))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _pair(value: Any, where: str) -> tuple[int, int]:
    """An ``(item, rating)`` 2-tuple of integers (numpy.int64 in the files; bool rejected)."""
    if not isinstance(value, tuple) or len(value) != 2 or not all(_is_int(v) for v in value):
        raise DataError(f"{where}: expected an (item, rating) tuple of 2 integers, got {value!r}")
    return int(value[0]), int(value[1])


def _check_item_rating(item: int, rating: int, where: str) -> None:
    if not 0 <= item < NUM_ITEMS:
        raise DataError(f"{where}: item id {item} is outside 0-{NUM_ITEMS - 1}")
    if rating not in RATINGS:
        raise DataError(f"{where}: rating {rating} is not in 1-5")


def parse_rows(df: pd.DataFrame, split: str) -> tuple[RawRow, ...]:
    """Parse a projected frame row by row (§5.6.2, §5.6.10).

    ``seq`` is a list of 10 ``(item, rating)`` tuples; padding ``(1682, 0)`` only on
    the right (left or middle padding is an error, ``U data/movielens_data.py:80-88``
    removed it without checking positions); real items 0-1681 with rating 1-5;
    ``len_seq == max(1, k)`` with k real items; ``next`` an ``(item, rating)`` tuple with
    item 0-1681 (1682 is never a target). Ratings are dropped (``:95-101``).
    """
    _check_split(split)
    missing = [c for c in FRAME_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{split}: missing columns {missing}")
    rows = []
    for p, (seq, len_seq, nxt) in enumerate(zip(df["seq"].tolist(), df["len_seq"].tolist(),
                                                 df["next"].tolist())):
        where = f"{split} row {p}"
        if not isinstance(seq, list) or len(seq) != HISTORY_LEN:
            raise DataError(f"{where}: seq must be a list of {HISTORY_LEN} (item, rating) tuples, got {seq!r}")
        history: list[int] = []
        padded = False
        for j, entry in enumerate(seq):
            item, rating = _pair(entry, f"{where} seq[{j}]")
            if (item, rating) == (PAD_ID, PAD_RATING):
                padded = True
                continue
            if padded:
                raise DataError(f"{where}: seq[{j}] = {entry!r} follows padding; padding (1682, 0) must be on "
                                f"the right only (left or middle padding): {seq!r}")
            _check_item_rating(item, rating, f"{where} seq[{j}]")
            history.append(item)
        if not _is_int(len_seq):
            raise DataError(f"{where}: len_seq must be an integer, got {len_seq!r}")
        k = len(history)
        if int(len_seq) != max(1, k):
            raise DataError(f"{where}: len_seq {int(len_seq)} != max(1, k) = {max(1, k)} for k = {k} real items")
        target, rating = _pair(nxt, f"{where} next")
        _check_item_rating(target, rating, f"{where} next")
        rows.append(RawRow(row=p, len_seq=int(len_seq), history=tuple(history), target=target))
    return tuple(rows)


def assign_users(rows: Sequence[RawRow], split: str) -> tuple[int, ...]:
    """Pseudo user ids (D14, §5.6.5), one per row.

    train (rows before the filter, file order): row ``p`` starts a new block when
    ``p = 0`` or ``history[p] != (history[p-1] + (target[p-1],))[-10:]``; the user id
    is the block number. Every block must start with an empty history and its
    windows must have lengths 0, 1, 2, ..., 10, 10, .... Validation and test: one
    row = one user, ``754 + row`` and ``848 + row``.
    """
    _check_split(split)
    if split != "train":
        return tuple(USER_OFFSET[split] + r.row for r in rows)
    users: list[int] = []
    block = -1
    position = 0
    previous: Optional[RawRow] = None
    for r in rows:
        if previous is None or r.history != (previous.history + (previous.target,))[-HISTORY_LEN:]:
            block += 1
            position = 0
            if r.history:
                raise DataError(f"train row {r.row}: block {block} does not start with an empty history "
                                f"(history {r.history}, the previous row does not continue into it)")
        else:
            position += 1
        if len(r.history) != min(position, HISTORY_LEN):
            raise DataError(f"train row {r.row}: window {position} of block {block} has length "
                            f"{len(r.history)}, expected {min(position, HISTORY_LEN)}")
        users.append(USER_OFFSET["train"] + block)
        previous = r
    return tuple(users)


def to_samples(rows: Sequence[RawRow], users: Sequence[int], split: str) -> tuple[Sample, ...]:
    """``Sample`` tuple of the rows that pass ``len_seq >= MIN_LEN_SEQ``, in row order (§5.6.3 step 6).

    The filter (``U data/movielens_data.py:79``) only drops rows; ``row`` keeps the
    unfiltered number (§5.6.4). Kept rows have ``len(history) == len_seq``.
    """
    _check_split(split)
    if len(rows) != len(users):
        raise DataError(f"{split}: {len(rows)} rows but {len(users)} user ids")
    out = []
    previous = -1
    for r, user in zip(rows, users):
        if r.row <= previous:
            raise DataError(f"{split}: rows are not in strictly ascending order (row {r.row} after {previous})")
        previous = r.row
        if r.len_seq < MIN_LEN_SEQ:
            continue
        if len(r.history) != r.len_seq:
            raise DataError(f"{split} row {r.row}: history length {len(r.history)} != len_seq {r.len_seq}")
        out.append(Sample(split=split, row=r.row, user_id=user, history=r.history, target=r.target))
    return tuple(out)


# ----------------------------------------------------------------------------
# Loading (§5.6.3)
# ----------------------------------------------------------------------------
def _verified_paths(repo_root: Path) -> dict[str, Path]:
    """Steps 1-2: the four files exist and their sha256 equals ``run.EXPECTED_INPUTS``.

    All four are checked before any file is unpickled (pickle can execute code).
    ``u.item`` is hashed after CRLF -> LF normalization (the ``is_text`` flag of the table).
    """
    paths = {}
    for name in (*SPLIT_FILES.values(), ITEM_FILE):
        rel = f"{DATA_DIR}/{name}"
        path = Path(repo_root) / rel
        if not path.is_file():
            raise DataError(f"{rel}: file not found under {repo_root}")
        if rel not in run.EXPECTED_INPUTS:
            raise DataError(f"{rel}: no expected sha256 in run.EXPECTED_INPUTS")
        expected, is_text = run.EXPECTED_INPUTS[rel]
        got = run.content_sha256(path, is_text)
        if got != expected:
            raise DataError(f"{rel}: sha256 {got} != expected {expected} (file not read)")
        paths[name] = path
    return paths


def _check_smoke_sizes(sizes: Sequence[int], n_train_users: int, n_val: int, n_test: int) -> tuple[int, int, int]:
    """(train users, val users, test users) within 1..available (§5.6.7: 754, 94, 95 for the full data)."""
    if not isinstance(sizes, (tuple, list)) or len(sizes) != 3 or not all(_is_py_int(n) for n in sizes):
        raise DataError(f"smoke sizes must be 3 ints (train, val, test), got {sizes!r}")
    for name, n, high in zip(SPLITS, sizes, (n_train_users, n_val, n_test)):
        if not 1 <= n <= high:
            raise DataError(f"smoke size for {name} must be in 1..{high}, got {n}")
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


def load_dataset(cfg: "Config", repo_root: Path) -> LoadedData:
    """The only path to the samples (§5.6.3, §5.6.12). ``cfg.data`` and, for smoke, ``cfg.run.seed``."""
    repo_root = Path(repo_root)
    paths = _verified_paths(repo_root)                                       # steps 1-2
    raw_titles = read_raw_titles(paths[ITEM_FILE])                           # step 3
    catalog = build_catalog(raw_titles, cfg.data.title_mode)
    samples: dict[str, tuple[Sample, ...]] = {}
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        frame = project_frame(pd.read_pickle(paths[SPLIT_FILES[split]]), split)   # step 4
        fingerprint = frame_fingerprint(frame)                               # step 5
        if fingerprint != EXPECTED_FINGERPRINTS[split]:
            raise DataError(f"{split}: content fingerprint {fingerprint} != expected "
                            f"{EXPECTED_FINGERPRINTS[split]} (pandas/numpy unpickling differs?)")
        rows = parse_rows(frame, split)                                      # step 6
        users = assign_users(rows, split)
        kept = to_samples(rows, users, split)
        n_users = len({s.user_id for s in kept})
        if len(rows) != EXPECTED_ROWS[split] or len(kept) != EXPECTED_SAMPLES[split]:   # step 7
            raise DataError(f"{split}: {len(rows)} rows / {len(kept)} samples, expected "
                            f"{EXPECTED_ROWS[split]} / {EXPECTED_SAMPLES[split]}")
        if split == "train" and (len(set(users)) != TRAIN_USERS or n_users != TRAIN_USERS):
            raise DataError(f"train: {len(set(users))} blocks and {n_users} users with samples, "
                            f"expected {TRAIN_USERS}")
        samples[split] = kept
        summary[split] = {"rows": len(rows), "samples": len(kept), "users": n_users}
    data = LoadedData(catalog=catalog, train=samples["train"], val=samples["val"], test=samples["test"],
                      subset="full", smoke_users=None)
    # §5.6.8: smoke_users is range-checked also for full (every key is required by M0).
    sizes = _check_smoke_sizes(cfg.data.smoke_users, TRAIN_USERS, len(data.val), len(data.test))
    if cfg.data.subset == "smoke":                                           # step 8
        data = select_smoke(data, cfg.run.seed, sizes, train_rows=tuple(cfg.data.train_rows))
    elif cfg.data.subset != "full":
        raise DataError(f"data.subset must be one of {SUBSETS}, got {cfg.data.subset!r}")
    report = title_report(raw_titles)                                        # step 9
    logger.info("data: %s", ", ".join(f"{s} rows {v['rows']} -> samples {v['samples']} (users {v['users']})"
                                     for s, v in summary.items()))
    logger.info("data: title_mode=%s, code title defects %s", catalog.title_mode,
                {name: len(ids) for name, ids in report.items()})
    if data.subset == "smoke":
        logger.info("data: smoke subset %s (train_rows %s): train %d samples of %d users, val rows %s, "
                    "test rows %s", data.smoke_users, list(cfg.data.train_rows) or "none", len(data.train),
                    len({s.user_id for s in data.train}), [s.row for s in data.val], [s.row for s in data.test])
    return data


def select_smoke(data: LoadedData, main_seed: int, sizes: tuple[int, int, int], *,
                 train_rows: tuple[int, ...]) -> LoadedData:
    """Smoke subset of the full data (D14, §5.6.7). ``row`` and ``user_id`` are kept.

    train: ``users = sorted(train user ids)``, ``pick = np_rng(seed, "smoke_users",
    "train").choice(len(users), size=n_train, replace=False)``, then every window of
    the picked users in row order. With non-empty ``train_rows`` (``overfit_gpu``) the
    train samples are exactly those rows instead and the ``("smoke_users", "train")``
    key is not used; each row must be a train sample after the filter and the
    samples must have ``sizes[0]`` distinct users. val/test: ``pick = np_rng(seed,
    "smoke_users", split).choice(N, size=n, replace=False)`` positions, in row order.
    """
    if data.subset != "full":
        raise DataError(f"select_smoke selects from the full data, got subset {data.subset!r}")
    users = sorted({s.user_id for s in data.train})
    n_train, n_val, n_test = _check_smoke_sizes(sizes, len(users), len(data.val), len(data.test))
    rows = tuple(train_rows)
    if rows:
        if not all(_is_py_int(r) for r in rows) or list(rows) != sorted(set(rows)):
            raise DataError(f"train_rows must be strictly ascending ints, got {list(rows)}")
        by_row = {s.row: s for s in data.train}
        missing = [r for r in rows if r not in by_row]
        if missing:
            raise DataError(f"train_rows {missing} are not train samples: rows removed by the len_seq >= "
                            f"{MIN_LEN_SEQ} filter or outside train_data.df (0-{EXPECTED_ROWS['train'] - 1})")
        train = tuple(by_row[r] for r in rows)
        chosen = {s.user_id for s in train}
        if len(chosen) != n_train:
            raise DataError(f"train_rows give {len(chosen)} users {sorted(chosen)}, but smoke_users[0] = {n_train}")
    else:
        pick = np_rng(main_seed, "smoke_users", "train").choice(len(users), size=n_train, replace=False)
        chosen = {users[int(i)] for i in pick}
        train = tuple(s for s in data.train if s.user_id in chosen)
    val_pick = np_rng(main_seed, "smoke_users", "val").choice(len(data.val), size=n_val, replace=False)
    test_pick = np_rng(main_seed, "smoke_users", "test").choice(len(data.test), size=n_test, replace=False)
    val = tuple(data.val[i] for i in sorted(int(x) for x in val_pick))
    test = tuple(data.test[i] for i in sorted(int(x) for x in test_pick))
    return LoadedData(catalog=data.catalog, train=train, val=val, test=test, subset="smoke",
                      smoke_users=(n_train, n_val, n_test))


# ----------------------------------------------------------------------------
# Shared padding (§5.6.13)
# ----------------------------------------------------------------------------
def pad_histories(histories: Sequence[Sequence[int]], pad_id: int, length: int
                  ) -> tuple["torch.Tensor", "torch.Tensor"]:
    """``(hist_ids [N, length], len_seq [N])``, CPU contiguous int64, right padding with ``pad_id``.

    ``hist_ids[i] = list(h) + [pad_id] * (length - len(h))``, ``len_seq[i] = len(h)``.
    Callers pass ``PAD_ID`` (1682) and ``HISTORY_LEN`` (10): M5 collate, M3
    ``encode_all`` and M12 ``compute_representations`` share this one rule.
    """
    import torch

    if not _is_py_int(pad_id) or pad_id < 1:
        raise DataError(f"pad_id must be an int >= 1, got {pad_id!r}")
    if not _is_py_int(length) or length < 1:
        raise DataError(f"length must be an int >= 1, got {length!r}")
    rows = list(histories)
    if not rows:
        raise DataError("pad_histories needs at least one history (N >= 1)")
    padded: list[list[int]] = []
    lengths: list[int] = []
    for i, history in enumerate(rows):
        if isinstance(history, (str, bytes)) or not hasattr(history, "__len__"):
            raise DataError(f"history {i}: expected a sequence of item ids, got {type(history).__name__}")
        if not 1 <= len(history) <= length:
            raise DataError(f"history {i}: length {len(history)} is outside [1, {length}]")
        for j, item in enumerate(history):
            if not _is_int(item) or not 0 <= int(item) < pad_id:
                raise DataError(f"history {i} position {j}: {item!r} is not an item id "
                                f"(int 0-{pad_id - 1}, bool excluded)")
        padded.append([int(x) for x in history] + [pad_id] * (length - len(history)))
        lengths.append(len(history))
    hist_ids = torch.tensor(padded, dtype=torch.int64, device="cpu").contiguous()
    len_seq = torch.tensor(lengths, dtype=torch.int64, device="cpu").contiguous()
    return hist_ids, len_seq


__all__ = [
    "NUM_ITEMS", "PAD_ID", "PAD_RATING", "HISTORY_LEN", "MIN_LEN_SEQ", "SPLITS", "DATA_DIR", "SPLIT_FILES",
    "ITEM_FILE", "USER_OFFSET", "ITEM_ENCODING", "ITEM_FIELDS", "RATINGS", "TITLE_MODES", "SUBSETS",
    "FRAME_COLUMNS", "CLUSTER_COLUMN", "EXPECTED_ROWS", "EXPECTED_SAMPLES", "TRAIN_USERS",
    "EXPECTED_FINGERPRINTS", "TITLE_REPORT_CATEGORIES", "DataError", "ItemCatalog", "Sample", "RawRow",
    "LoadedData", "read_raw_titles", "code_title", "fixed_title", "build_catalog", "title_report",
    "project_frame", "frame_fingerprint", "parse_rows", "assign_users", "to_samples", "load_dataset",
    "select_smoke", "pad_histories",
]
