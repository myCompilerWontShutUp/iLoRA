"""M2 후보 생성 (DESIGN.md §6, §3.1 후보·``CandidateFn``·``candidate_flags``·``match_norm``·``self_check``).

Every sample gets 20 candidates: its target plus 19 negatives drawn uniformly
without replacement from the items that are neither in the visible history
window nor the target (D10, D21). The random source is one numpy Generator per
sample, keyed ``("candidates", split, row)`` for evaluation (fixed) and
``("candidates", "train", row, epoch)`` for training (redrawn every epoch),
§4.6.4. Nothing depends on DataLoader workers, call order, batch composition or
the smoke subset, and there is no state or cache.

The draw keeps the released order (``random.sample`` of the negatives, the
target appended, ``random.shuffle``, ``U data/movielens_data.py:46-47``) with
``Generator.choice`` and ``Generator.shuffle``. numpy fixes the BitGenerator
stream but not the algorithms of those methods across versions (NEP 19), so the
output is tied to numpy 1.24.2 and ``self_check`` compares it with the design
golden (§6.6.5).

``candidate_flags`` marks evaluation samples whose candidate titles make the
released judgement rule ambiguous (D20, §6.6.4). ``match_norm`` is the only
definition of the judgement normalization; M11 uses it too.

M5 passes ``functools.partial(candidates_for, main_seed=cfg.run.seed,
cfg=cfg.candidates)`` (``CandidateFn``, pickles for spawn workers) and calls it
as ``fn(sample, epoch=e)`` or ``fn(sample, epoch=None)``; ``epoch`` is
keyword-only, so ``fn(sample, 0)`` is a ``TypeError`` (§6.6.9).
"""
from __future__ import annotations

from typing import Final, Optional, Union

import numpy as np

from .data import NUM_ITEMS, ItemCatalog, Sample
from .sections import CandidatesConfig
from .seeding import derive_seed, np_rng

NUM_CANDIDATES: Final = 20          # §3.1, [논문 App B p.16] [LLaRA §5.1.3 p.7] [코드] U main.py:122 (D10)
FLAG_NAMES: Final = ("empty_title_candidate", "duplicate_title_candidates",
                     "substring_title_candidates", "target_title_duplicated",
                     "target_unscorable")
EVAL_SPLITS: Final = ("val", "test")

# Golden of §6.6.5 (design/tmp/m2_flags.py; seed 1234, numpy 1.24.2, M0 derive_seed).
_GOLDEN_SEED: Final = 1234
_GOLDEN_DERIVED_SEED: Final = 4835709385070275973          # derive_seed(1234, "candidates", "val", 0)
_GOLDEN_SAMPLE: Final = ("val", 0, 754, (880, 269, 361, 1024, 881, 890, 1242, 1104, 989, 333), 321)
_GOLDEN_CANDIDATES: Final = (980, 156, 321, 1240, 892, 1299, 448, 1094, 1502, 728,
                             1598, 208, 1042, 1386, 902, 1282, 1029, 378, 515, 554)

Key = tuple[Union[str, int], ...]


class CandidateError(ValueError):
    """Wrong call or a candidate tuple that breaks the §3.1 contract (§6.6.3, §6.6.8)."""


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def candidate_key(sample: Sample, epoch: Optional[int]) -> Key:
    """§6.6.2 step 1: ``("candidates", "train", row, epoch)`` or ``("candidates", split, row)``.

    A train sample needs ``epoch`` = int >= 0 (bool rejected, the same epoch number
    as the ``train_order`` and ``template`` keys); a val/test sample needs ``epoch=None``.
    """
    split = sample.split
    if split == "train":
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise CandidateError(f"{sample.sample_id}: a train sample needs epoch = int >= 0 (training key "
                                 f"('candidates', 'train', row, epoch)), got {epoch!r}")
        return ("candidates", "train", sample.row, epoch)
    if split in EVAL_SPLITS:
        if epoch is not None:
            raise CandidateError(f"{sample.sample_id}: an evaluation sample takes epoch=None (fixed key "
                                 f"('candidates', {split!r}, row)), got {epoch!r}")
        return ("candidates", split, sample.row)
    raise CandidateError(f"{sample.sample_id}: unknown split {split!r}")


def negative_pool(history: tuple[int, ...], target: int, num_items: int) -> np.ndarray:
    """Negative pool (D21): ids 0..num_items-1 without the history window and the target, ascending int64.

    ``U data/movielens_data.py:45`` builds the same set in the same order (u.item
    order) with a list scan; a boolean mask gives it directly (§6.5).
    """
    ids = (*history, target)
    bad = [x for x in ids if not _is_int(x) or not 0 <= int(x) < num_items]
    if bad:
        raise CandidateError(f"history/target ids {bad} are not item ids 0-{num_items - 1}")
    if target in history:
        raise CandidateError(f"target {target} is in the history {tuple(history)}")
    mask = np.ones(num_items, dtype=bool)
    mask[[int(x) for x in history]] = False
    mask[int(target)] = False
    return np.flatnonzero(mask).astype(np.int64)


def draw_candidates(history: tuple[int, ...], target: int, rng: np.random.Generator, *,
                    num_items: int, num_candidates: int) -> tuple[int, ...]:
    """§6.6.2 steps 3-7: pool, ``rng.choice`` (call 1), target appended, ``rng.shuffle`` (call 2).

    ``rng`` is used for these two calls only.
    """
    pool = negative_pool(history, target, num_items)
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) \
            or not 1 <= num_candidates <= len(pool) + 1:
        raise CandidateError(f"num_candidates must be an int in 1..{len(pool) + 1}, got {num_candidates!r}")
    neg = rng.choice(pool, size=num_candidates - 1, replace=False)      # [코드] random.sample, U :46
    cands = np.concatenate([neg, np.array([target], dtype=np.int64)])  # target last, U :46
    rng.shuffle(cands)                                                  # [코드] random.shuffle, U :47
    return tuple(int(x) for x in cands)


def check_candidates(cands: tuple[int, ...], sample: Sample, *, num_items: int,
                     num_candidates: int) -> None:
    """§6.6.3 contract: a tuple of ``num_candidates`` ints (bool excluded) in 0..num_items-1,
    no repeats, the target exactly once and no history item. Raises ``CandidateError``."""
    where = sample.sample_id
    if not isinstance(cands, tuple):
        raise CandidateError(f"{where}: candidates must be a tuple, got {type(cands).__name__}")
    if len(cands) != num_candidates:
        raise CandidateError(f"{where}: {len(cands)} candidates, expected {num_candidates}")
    bad = [c for c in cands if isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < num_items]
    if bad:
        raise CandidateError(f"{where}: candidates {bad} are not Python int item ids 0-{num_items - 1}")
    if len(set(cands)) != len(cands):
        raise CandidateError(f"{where}: repeated candidates in {cands}")
    if cands.count(sample.target) != 1:
        raise CandidateError(f"{where}: target {sample.target} appears {cands.count(sample.target)} times "
                             f"in {cands}")
    overlap = sorted(set(cands) & set(sample.history))
    if overlap:
        raise CandidateError(f"{where}: candidates {overlap} are in the history {sample.history}")


def candidates_for(sample: Sample, *, main_seed: int, epoch: Optional[int],
                   cfg: CandidatesConfig) -> tuple[int, ...]:
    """The 20 candidates of ``sample`` (D10, §6.6.2): ``epoch`` int for training, None for evaluation."""
    if isinstance(cfg.num_candidates, bool) or cfg.num_candidates != NUM_CANDIDATES:
        raise CandidateError(f"candidates.num_candidates must be {NUM_CANDIDATES} (§3.1 cand_ids [B, 20], "
                             f"template text '20 movie titles'), got {cfg.num_candidates!r}")
    key = candidate_key(sample, epoch)
    if sample.target in sample.history:          # checked before the pool (M1 Sample also rejects it)
        raise CandidateError(f"{sample.sample_id} (key {key}): target {sample.target} is in the history "
                             f"{sample.history}")
    rng = np_rng(main_seed, *key)                # one Generator per sample and purpose (§4.6.4)
    cands = draw_candidates(sample.history, sample.target, rng, num_items=NUM_ITEMS,
                            num_candidates=cfg.num_candidates)
    try:
        check_candidates(cands, sample, num_items=NUM_ITEMS, num_candidates=cfg.num_candidates)
    except CandidateError as exc:
        raise CandidateError(f"{exc} (key {key})") from exc
    return cands


def match_norm(title: str) -> str:
    """Judgement normalization ``strip().lower().strip()`` (``U model/model_interface.py:476-478``).

    The only definition: the flags below and the M11 judgement both use it (§6.6.4, §6.6.10).
    """
    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title).__name__}")
    return title.strip().lower().strip()


def candidate_flags(cands: tuple[int, ...], target: int, catalog: ItemCatalog) -> dict[str, bool]:
    """Title ambiguity flags of one evaluation candidate set (§6.6.4), keys ``FLAG_NAMES`` in order.

    With n(c) = ``match_norm(catalog.titles[c])`` and the 19 negatives = ``cands``
    without ``target``:

    - ``empty_title_candidate``: some n(c) == ""
    - ``duplicate_title_candidates``: n(c) == n(c') for two different candidates
    - ``substring_title_candidates``: n(c) != "", n(c) != n(c') and n(c) in n(c') for some pair
    - ``target_title_duplicated``: some negative has n(c) == n(target)
    - ``target_unscorable``: some negative has n(c) in n(target) (empty, equal or substring), so
      the released judgement never counts the exact target title as correct
    """
    cands = tuple(cands)
    titles = catalog.titles
    bad = [c for c in (*cands, target) if not _is_int(c) or not 0 <= int(c) < len(titles)]
    if bad:
        raise CandidateError(f"ids {bad} are not item ids 0-{len(titles) - 1}")
    if len(cands) != NUM_CANDIDATES or len(set(cands)) != len(cands):
        raise CandidateError(f"expected {NUM_CANDIDATES} distinct candidates, got {cands}")
    if cands.count(target) != 1:
        raise CandidateError(f"target {target} appears {cands.count(target)} times in {cands}")
    norm = [match_norm(titles[c]) for c in cands]
    norm_target = match_norm(titles[target])
    negatives = [match_norm(titles[c]) for c in cands if c != target]
    flags = {
        "empty_title_candidate": any(n == "" for n in norm),
        "duplicate_title_candidates": len(set(norm)) < len(norm),
        "substring_title_candidates": any(a != "" and a != b and a in b
                                          for i, a in enumerate(norm) for j, b in enumerate(norm) if i != j),
        "target_title_duplicated": any(n == norm_target for n in negatives),
        "target_unscorable": any(n in norm_target for n in negatives),
    }
    return {name: bool(flags[name]) for name in FLAG_NAMES}


def self_check() -> None:
    """Golden comparison (§6.6.5): validation row 0 at seed 1234 must give the design tuple.

    Needs no data file. The entry points call it once right after the environment
    check (M0 ``commands.run_command``).
    """
    seed = derive_seed(_GOLDEN_SEED, "candidates", "val", 0)
    if seed != _GOLDEN_DERIVED_SEED:
        raise CandidateError(f"derive_seed(1234, 'candidates', 'val', 0) = {seed} differs from the design golden "
                             f"{_GOLDEN_DERIVED_SEED}: check M0 seeding.derive_seed (§3.1)")
    split, row, user_id, history, target = _GOLDEN_SAMPLE
    sample = Sample(split=split, row=row, user_id=user_id, history=history, target=target)
    got = candidates_for(sample, main_seed=_GOLDEN_SEED, epoch=None,
                         cfg=CandidatesConfig(num_candidates=NUM_CANDIDATES))
    if got != _GOLDEN_CANDIDATES:
        raise CandidateError(f"numpy Generator 출력이 설계 golden과 다름: numpy 판 확인 (numpy {np.__version__}, "
                             f"required 1.24.2; validation row 0 gave {got}, expected {_GOLDEN_CANDIDATES})")


__all__ = [
    "NUM_CANDIDATES", "FLAG_NAMES", "EVAL_SPLITS", "CandidateError", "candidate_key", "negative_pool",
    "draw_candidates", "candidates_for", "check_candidates", "match_norm", "candidate_flags", "self_check",
]
