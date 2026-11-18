"""M0·M1 데이터셋 규격 (DESIGN.md §23.2, §23.3 D27·D28, §23.4).

``DatasetSpec`` holds every value that differs between datasets: folder and file names, the item
count (which is also the padding id and the SASRec ``item_num``), the row layout, the expected
counts, the frame fingerprints, the template and SASRec files and the title rules. One frozen
module constant per dataset carries them (``MOVIELENS``, ``STEAM``, ``LASTFM``); the constants of
``data.py`` (``NUM_ITEMS``, ``PAD_ID``, ...) stay as aliases of ``MOVIELENS`` (§23.4).

A run uses exactly one dataset. No config key selects it: the closed key ``prompt.template_path``
does, through ``for_template``/``spec_of`` (D27), so every existing ``config.yaml`` keeps meaning
MovieLens and the 87 keys stay. Standard library only and no other ilora module is imported, so
``sections.py`` and the torch-free parts of M0 can import this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

# Split files of every dataset folder: [코드] U data/movielens_data.py:52-57, U data/steam_data.py:57-62.
SPLIT_FILES: Final = {"train": "train_data.df", "val": "Val_data.df", "test": "Test_data.df"}


class DatasetError(ValueError):
    """A template path that belongs to no dataset (§23.4 ``for_template``)."""


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset (§23.4). Mapping-valued fields have the keys ``train``, ``val``, ``test``.

    The MovieLens constant carries exactly the values of ``data.py``; ``data.py``, ``cluster.py``,
    ``evaluate.py``, ``prompts.py`` and ``batching.py`` keep their module constants as they were.
    """
    name: str                                # "movielens"
    data_dir: str                            # repository-relative folder of the four data files
    item_file: str                           # titles file (§5.6.2)
    item_format: str                         # "uitem": ISO-8859-1, 24 '|' fields, id = movie_id - 1
    #                                          "id2name": UTF-8, "id::title", id = line number (D38)
    num_items: int                           # also the padding id and SASRec item_num (U main.py:142-147)
    has_ratings: bool                        # seq and next are (item, rating) pairs (§5.6.2)
    first_window_empty: bool                 # train blocks start with an empty history (D14, D40)
    extra_separator_ids: tuple[int, ...]     # ids whose id2name line may hold a second "::" (D38)
    expected_rows: Mapping[str, int]         # rows before the len_seq filter (§5.6.2)
    expected_samples: Mapping[str, int]      # samples with len_seq >= 3 (§5.6.3 step 7)
    train_blocks: int                        # train blocks before the filter (§5.6.5)
    train_users: int                         # train users with samples after the filter (§5.6.5)
    fingerprints: Mapping[str, str]          # content fingerprints of (seq, len_seq, next) (§5.6.9)
    template_file: str                       # prompt templates, the key of the M0 input table (§8.6.8, D37)
    sasrec_source: str                       # original SASRec pickle, the key of the M0 input table (D36)
    title_modes: tuple[str, ...]             # allowed data.title_mode values (D20, D38)
    title_forbidden: tuple[str, ...]         # strings check_catalog_titles refuses (§8.6.2, D39)

    @property
    def pad_id(self) -> int:
        """Padding item id, equal to the item count ([코드] U main.py:142-147)."""
        return self.num_items

    @property
    def data_files(self) -> tuple[str, ...]:
        """Repository-relative (train, val, test, item_file) paths, the data keys of the M0 input table."""
        return tuple(f"{self.data_dir}/{name}" for name in (*SPLIT_FILES.values(), self.item_file))

    @property
    def user_offset(self) -> dict[str, int]:
        """Pseudo user id of train block 0, validation row 0 and test row 0 (D14, §5.6.5)."""
        return {"train": 0, "val": self.train_blocks, "test": self.train_blocks + self.expected_rows["val"]}


MOVIELENS: Final = DatasetSpec(
    name="movielens",
    data_dir="data/ref/movielens",           # [코드] U train_movielens.sh:8
    item_file="u.item",                      # [코드] U data/movielens_data.py:70
    item_format="uitem",                     # [코드] U data/movielens_data.py:71-74 (§5.6.2)
    num_items=1682,                          # [LLaRA Table 1 p.7]; padding id [코드] U data/movielens_data.py:18
    has_ratings=True,
    first_window_empty=True,                 # D14: windows 0, 1, 2, ..., 10, 10, ... (§5.6.5)
    extra_separator_ids=(),                  # u.item has no "::" field (D38 is about id2name files)
    expected_rows={"train": 68388, "val": 94, "test": 95},          # §5.6.2
    expected_samples={"train": 66126, "val": 94, "test": 95},       # §5.6.3 step 7
    train_blocks=754,                        # D14, §5.6.5
    train_users=754,                         # D14, §5.6.5
    # §5.6.9 (design/tmp/m1_fingerprint.py); the train value is the same for the iLoRA copy
    # (sha256 bac4c88b...) and the LLaRA copy (sha256 a73b119d...).
    fingerprints={
        "train": "85bafab832f1f78e5095b1fd880a418097054d3ea7dbb7d4fe9e1528f019aa99",
        "val": "de7ac66b6703dc6935e1339183c00e08af2ad40f6d0877e7556b483a9aa65523",
        "test": "01636285c16faf95cc7136bc937b9aa315c66de5e23fed5efec0e2b2e84dff09",
    },
    template_file="prompt/movie.txt",        # [코드] U train_movielens.sh:10 (§8.6.8)
    sasrec_source="rec_model/movielens.pt",  # [코드] U train_movielens.sh:14, blob 41c057dc (§7.4)
    title_modes=("code", "fixed"),           # D20
    title_forbidden=("[", "]", "<s>", "</s>", "<unk>"),   # §8.6.2 check_catalog_titles
)

# D39: the slot and special token strings; real Steam and LastFM titles contain single brackets (§23.2).
_ID2NAME_TITLE_FORBIDDEN: Final = ("[HistoryHere]", "[CansHere]", "[HistoryEmb]", "[CansEmb]", "[PH]", "[ItemEmb]",
                                   "[PAD]", "<s>", "</s>", "<unk>")

STEAM: Final = DatasetSpec(
    name="steam",
    data_dir="data/ref/steam",               # [코드] U train_steam.sh:8
    item_file="id2name.txt",                 # [코드] U data/steam_data.py:70
    item_format="id2name",                   # D38, [코드] U data/steam_data.py:73-74
    num_items=3581,                          # [LLaRA Table 1 p.7]; padding id [코드] U main.py:142-147 (§23.2)
    has_ratings=False,                       # §23.2 row layout: seq and next are plain item ids
    first_window_empty=False,                # D40: first window length 1-10, then 10
    extra_separator_ids=(354,),              # D38: "Depths of Fear :: Knossos" is cut at the second "::"
    expected_rows={"train": 151056, "val": 1194, "test": 1194},     # §23.2
    expected_samples={"train": 151048, "val": 1184, "test": 1186},  # §23.2, D42
    train_blocks=9550,                       # D40, §23.2
    train_users=9542,                        # D40, §23.2
    # §23.2 hash table (multi/verify/check_numbers.py)
    fingerprints={
        "train": "d1468b1f36548bdc9c71dca9c918b710695e5efc9052cbe973357d5a10266ee9",
        "val": "bc63919adc858273ecde739a6eedab8e48bd4359fb204f5179bd55e389d42b75",
        "test": "987376eb6cb7ac1bc6fdf04b7655bbc467c18277e60bcd305b3717cb89784705",
    },
    template_file="prompt/game.txt",         # [코드] U train_steam.sh:10 (D37)
    sasrec_source="rec_model/steam.pt",      # [코드] U train_steam.sh:14, blob db784032 (D36)
    title_modes=("code",),                   # D38
    title_forbidden=_ID2NAME_TITLE_FORBIDDEN,
)

LASTFM: Final = DatasetSpec(
    name="lastfm",
    data_dir="data/ref/lastfm",              # [코드] U train_lastfm.sh:8
    item_file="id2name.txt",                 # [코드] U data/lastfm_data.py:62
    item_format="id2name",                   # D38, [코드] U data/lastfm_data.py:65-66
    num_items=4606,                          # [LLaRA Table 1 p.7]; padding id [코드] U main.py:142-147 (§23.2)
    has_ratings=False,                       # §23.2 row layout: seq and next are plain item ids
    first_window_empty=True,                 # D40: the same block shape as MovieLens (windows 0, 1, ..., 10, 10)
    extra_separator_ids=(),                  # D38: no title of the file holds a second "::"
    expected_rows={"train": 50800, "val": 122, "test": 122},        # §23.2
    expected_samples={"train": 47872, "val": 122, "test": 122},     # §23.2, D42
    train_blocks=976,                        # D40, §23.2
    train_users=976,                         # D40, §23.2
    # §23.2 hash table (multi/verify/check_numbers.py)
    fingerprints={
        "train": "890287c1d4f8fc1015d2e7ebe2167e8d098b0a1ff56721a3c73afaecf0cc2640",
        "val": "9d3440714637ebbf2b19ca9a0f8037fe8a391aa587c75adc736d4d84769bf52c",
        "test": "13d9b892f7cdf28fc25d61458f3fda873015305cc3486870da9ed24b95a4c354",
    },
    template_file="prompt/artist.txt",       # [코드] U train_lastfm.sh:10 (D37)
    sasrec_source="rec_model/lastfm.pt",     # [코드] U train_lastfm.sh:14, blob 4cad4b22 (D36)
    title_modes=("code",),                   # D38
    title_forbidden=_ID2NAME_TITLE_FORBIDDEN,
)

SPECS: Final[Mapping[str, DatasetSpec]] = {spec.name: spec for spec in (MOVIELENS, STEAM, LASTFM)}
_BY_TEMPLATE: Final[Mapping[str, DatasetSpec]] = {spec.template_file: spec for spec in SPECS.values()}


def for_template(path: str) -> DatasetSpec:
    """The dataset of a ``prompt.template_path`` value (D27, closed correspondence, §23.4)."""
    spec = _BY_TEMPLATE.get(path)
    if spec is None:
        raise DatasetError(f"no dataset uses the template file {path!r}; known template files: "
                           f"{sorted(_BY_TEMPLATE)} (DESIGN 23.4)")
    return spec


def spec_of(cfg: Any) -> DatasetSpec:
    """The dataset of a config: ``for_template(cfg.prompt.template_path)`` (D27, §23.4)."""
    return for_template(cfg.prompt.template_path)


__all__ = ["SPLIT_FILES", "DatasetError", "DatasetSpec", "MOVIELENS", "STEAM", "LASTFM", "SPECS", "for_template",
           "spec_of"]
