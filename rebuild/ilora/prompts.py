"""M4 prompt와 curriculum (DESIGN.md §8, §3.1 prompt·``BatchPlan``).

Templates come from ``prompt/movie.txt`` only (``TEMPLATE_FILE``, the key of
``run.EXPECTED_INPUTS``): ``load_templates`` compares the CRLF -> LF
normalized sha256 with the M0 expected value before reading, then
``parse_templates`` keeps the released reading (``splitlines()`` and
``strip()``, ``U data/data_interface.py:182-186``) and rejects anything but
exactly one ``[HistoryHere]`` and one ``[CansHere]`` per template (§8.6.2).

Every random choice is keyed by ``derive_seed`` coordinates, never by
DataLoader worker state (§4.6.4, D3, D15):

- template (D15): ``np_rng(seed, "template", "train", row, epoch)`` for training,
  ``np_rng(seed, "template", split, row)`` for evaluation, one ``integers(0, n)``;
- curriculum (D3): ``uniform(seed, "curriculum", tau)``, hybrid when ``u < tau / T``
  (``linear``), always (``all_hybrid``) or when ``u < 1 - tau / T`` (``reverse``);
- micro-batch order: ``torch.randperm(N, generator=torch_generator(seed, "train_order",
  epoch))`` cut into ``N // B`` micro-batches (drop_last).

``BatchPlan`` is built in the main process (``make_batch_plan``/``build_batch_plan``)
and DataLoader workers only tokenize what it lists (M5). Prompts are assembled in
one step from the five template parts, so text inside a title is never substituted
again (§8.5). Only strings, Python ints and Python floats are handled here; the
permutation is the only torch call (CPU generator, §8.6.7).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterator, NamedTuple, Optional, Sequence, Union

import torch

from . import run, seeding
from .candidates import CandidateError, check_candidates
from .data import HISTORY_LEN, MIN_LEN_SEQ, NUM_ITEMS, ItemCatalog, Sample
from .sections import PromptConfig

if TYPE_CHECKING:
    from .sections import Config


class PromptError(ValueError):
    """Template, prompt, curriculum or plan problem (§8.6.9). Raised at once, never repaired."""


# ----------------------------------------------------------------------------
# Constants (§8.6.1, §8.6.8: code constants, not config keys)
# ----------------------------------------------------------------------------
# [코드] U data/data_interface.py:43-63 (L data/data_interface.py:43-59, 코드 계보 일치)
HISTORY_SLOT, CANDS_SLOT = "[HistoryHere]", "[CansHere]"
HISTORY_TAG, CANS_TAG, PH_TAG = " [HistoryEmb]", " [CansEmb]", " [PH]"
ITEM_SEP = ", "
NUM_CANDIDATES: Final = 20                                   # §3.1 후보, D10
CURRICULA: Final = ("linear", "all_hybrid", "reverse")       # D3 and its variants
TEMPLATE_CHOICES: Final = ("random", "fixed")                # D15 and variant_fixed_template
EVAL_PROMPTS: Final = ("hybrid", "text_only")                # D4 and E3
TEMPLATE_FILE: Final = "prompt/movie.txt"                    # key of M0 run.EXPECTED_INPUTS (§8.6.8)

_BRACKET_RE: Final = re.compile(r"\[[^\[\]]*\]")             # §8.6.2 step 4
_SPECIAL_STRINGS: Final = ("<s>", "</s>", "<unk>")           # LLaMA special token strings
_TITLE_FORBIDDEN: Final = ("[", "]", *_SPECIAL_STRINGS)      # §8.6.2 check_catalog_titles


def _is_int(value: object) -> bool:
    """Python int, bool excluded."""
    return isinstance(value, int) and not isinstance(value, bool)


# ----------------------------------------------------------------------------
# Templates (§8.6.2)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TemplateSet:
    """The stripped template lines in file order and their five parts.

    ``parts[k] = (before, first slot, middle, second slot, after)``; the slot order
    is the order of the template (all three released templates have the history first).
    """
    templates: tuple[str, ...]
    parts: tuple[tuple[str, str, str, str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.templates, tuple) or not isinstance(self.parts, tuple):
            raise PromptError("TemplateSet.templates and .parts must be tuples")
        if not self.templates or len(self.templates) != len(self.parts):
            raise PromptError(f"TemplateSet needs >= 1 template and one parts tuple per template, got "
                              f"{len(self.templates)} templates and {len(self.parts)} parts")
        for k, (text, part) in enumerate(zip(self.templates, self.parts)):
            if not (isinstance(part, tuple) and len(part) == 5 and all(isinstance(x, str) for x in part)):
                raise PromptError(f"TemplateSet.parts[{k}] must be a tuple of 5 strings, got {part!r}")
            if {part[1], part[3]} != {HISTORY_SLOT, CANDS_SLOT} or "".join(part) != text:
                raise PromptError(f"TemplateSet.parts[{k}] {part!r} does not rebuild template {k} {text!r}")


def _split_template(text: str, where: str) -> tuple[str, str, str, str, str]:
    """§8.6.2 steps 4-5 for one stripped template line."""
    groups = _BRACKET_RE.findall(text)
    other = [g for g in groups if g not in (HISTORY_SLOT, CANDS_SLOT)]
    if other:   # [TargetHere] (answer leak, U data/data_interface.py:49-51), [PH], [HistoryEmb], ...
        raise PromptError(f"{where}: bracket group(s) {other} are not allowed, only {HISTORY_SLOT} and "
                          f"{CANDS_SLOT}: {text!r}")
    counts = Counter(groups)
    for slot in (HISTORY_SLOT, CANDS_SLOT):
        if counts[slot] != 1:
            raise PromptError(f"{where}: {slot} appears {counts[slot]} times, expected exactly once: {text!r}")
    rest = text.replace(HISTORY_SLOT, "").replace(CANDS_SLOT, "")
    stray = [c for c in rest if c in "[]"]
    if stray:
        raise PromptError(f"{where}: unpaired {''.join(stray)!r} outside the two slots: {text!r}")
    found = [s for s in _SPECIAL_STRINGS if s in text]
    if found:
        raise PromptError(f"{where}: special token string(s) {found} in the template: {text!r}")
    first, second = sorted((HISTORY_SLOT, CANDS_SLOT), key=text.index)
    i, j = text.index(first), text.index(second)
    return text[:i], first, text[i + len(first):j], second, text[j + len(second):]


def parse_templates(data: bytes, *, source: str) -> TemplateSet:
    """§8.6.2 steps 2-6 (no hash check; tests use it on modified bytes).

    UTF-8 strict decode, ``str.splitlines()``, ``strip()`` per line
    (``U data/data_interface.py:185-186``: the same lines for ``\\r\\n`` and ``\\n`` files),
    no empty line, then the slot rules of ``_split_template``.
    """
    if not isinstance(data, bytes):
        raise PromptError(f"{source}: template data must be bytes, got {type(data).__name__}")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PromptError(f"{source}: not UTF-8: {exc}") from exc
    lines = [line.strip() for line in text.splitlines()]      # [코드] U data/data_interface.py:185-186
    if not lines:
        raise PromptError(f"{source}: no template lines (the released code would keep an empty list, "
                          f"U data/data_interface.py:189-190)")
    parts = []
    for number, line in enumerate(lines, start=1):
        if not line:
            raise PromptError(f"{source} line {number}: empty line")
        parts.append(_split_template(line, f"{source} line {number}"))
    return TemplateSet(templates=tuple(lines), parts=tuple(parts))


def load_templates(path: Path) -> TemplateSet:
    """§8.6.2 steps 0-1, then ``parse_templates``. Call as ``load_templates(repo_root / cfg.prompt.template_path)``.

    A missing file is an error (not an empty list, ``U data/data_interface.py:189-190``).
    Before reading, ``run.content_sha256(path, text=True)`` must equal
    ``run.EXPECTED_INPUTS[TEMPLATE_FILE][0]``, so paths that skip the M0 environment
    check (CPU tests, smoke) read the same content.
    """
    path = Path(path)
    if not path.is_file():
        raise PromptError(f"{path}: template file not found")
    expected = run.EXPECTED_INPUTS[TEMPLATE_FILE][0]
    got = run.content_sha256(path, text=True)
    if got != expected:
        raise PromptError(f"{path}: template file sha256 (CRLF->LF) {got} != expected {expected} "
                          f"(run.EXPECTED_INPUTS[{TEMPLATE_FILE!r}])")
    return parse_templates(path.read_bytes(), source=str(path))


def check_catalog_titles(catalog: ItemCatalog) -> None:
    """Titles must not contain ``[``, ``]``, ``<s>``, ``</s>``, ``<unk>`` (§8.6.2).

    Such a title would create or break a slot or a special token. The empty title
    (item 266), titles with a trailing space and non-ASCII titles stay as they are (D20).
    """
    bad = [(i, t) for i, t in enumerate(catalog.titles) if any(s in t for s in _TITLE_FORBIDDEN)]
    if bad:
        raise PromptError(f"{len(bad)} catalog title(s) contain one of {list(_TITLE_FORBIDDEN)}: "
                          f"{[f'item {i}: {t!r}' for i, t in bad[:10]]}")


# ----------------------------------------------------------------------------
# Template choice (§8.6.3, D15)
# ----------------------------------------------------------------------------
def _check_epoch(sample: Sample, epoch: Optional[int]) -> None:
    """Training samples need epoch = int >= 0 (bool rejected); evaluation samples epoch None."""
    if sample.split == "train":
        if epoch is None:
            raise PromptError(f"{sample.sample_id}: a train sample needs the epoch (training key "
                              f"('template', 'train', row, epoch)), got None")
        if not _is_int(epoch) or epoch < 0:
            raise PromptError(f"{sample.sample_id}: epoch must be an int >= 0 (bool excluded), got {epoch!r}")
    elif epoch is not None:
        raise PromptError(f"{sample.sample_id}: an evaluation sample takes epoch None (fixed key "
                          f"('template', {sample.split!r}, row)), got {epoch!r}")


def choose_template(cfg: PromptConfig, seed: int, n_templates: int, sample: Sample,
                    epoch: Optional[int]) -> int:
    """Template index of one sample (D15, §8.6.3); the same checks for ``random`` and ``fixed``."""
    if not _is_int(n_templates) or n_templates < 1:
        raise PromptError(f"n_templates must be an int >= 1, got {n_templates!r}")
    _check_epoch(sample, epoch)
    if cfg.template_choice == "random":
        if cfg.fixed_template_index is not None:
            raise PromptError(f"prompt.template_choice random needs fixed_template_index null, got "
                              f"{cfg.fixed_template_index!r}")
        if sample.split == "train":
            key: tuple[Union[str, int], ...] = ("template", "train", sample.row, epoch)
        else:
            key = ("template", sample.split, sample.row)
        # one Generator per sample, one integers() draw (§8.6.3); [LLaRA §5.1.2 p.7] "randomly sampled"
        k = int(seeding.np_rng(seed, *key).integers(0, n_templates))
    elif cfg.template_choice == "fixed":
        k = cfg.fixed_template_index
        if not _is_int(k):
            raise PromptError(f"prompt.template_choice fixed needs an int fixed_template_index, got {k!r}")
    else:
        raise PromptError(f"prompt.template_choice must be one of {TEMPLATE_CHOICES}, got {cfg.template_choice!r}")
    if not 0 <= k < n_templates:
        raise PromptError(f"{sample.sample_id}: template index {k} is outside 0..{n_templates - 1}")
    return k


# ----------------------------------------------------------------------------
# Prompt text (§8.6.4)
# ----------------------------------------------------------------------------
def render_prompt(templates: TemplateSet, template_id: int, history_titles: Sequence[str],
                  cand_titles: Sequence[str], hybrid: bool) -> str:
    """Fill the two slots of template ``template_id`` in one step (no contract checks; see ``build_prompt``).

    hybrid: ``"title [HistoryEmb]"`` and ``"title [CansEmb]"``; text-only: ``"title [PH]"`` for both,
    items joined by ``", "`` (``U data/data_interface.py:43-48, :56-63``).
    """
    if not isinstance(hybrid, bool):
        raise PromptError(f"hybrid must be bool, got {hybrid!r}")
    if not _is_int(template_id) or not 0 <= template_id < len(templates.parts):
        raise PromptError(f"template index {template_id!r} is outside 0..{len(templates.parts) - 1}")
    tag_h, tag_c = (HISTORY_TAG, CANS_TAG) if hybrid else (PH_TAG, PH_TAG)
    values = {HISTORY_SLOT: ITEM_SEP.join(t + tag_h for t in history_titles),
              CANDS_SLOT: ITEM_SEP.join(t + tag_c for t in cand_titles)}
    before, first, middle, second, after = templates.parts[template_id]
    return before + values[first] + middle + values[second] + after


def build_prompt(sample: Sample, cands: tuple[int, ...], catalog: ItemCatalog, templates: TemplateSet,
                 template_id: int, hybrid: bool) -> str:
    """Prompt of one sample after the boundary checks of §8.6.4.

    ``3 <= len(history) <= 10`` (§3.1 ``Sample``) and the §3.1 candidate contract (20
    distinct Python ints 0-1681, the target exactly once, no history item) are checked
    again, so a wrong candidate tuple fails here and never reaches tokenization.
    History titles are in ``Sample.history`` order (oldest first,
    ``U data/movielens_data.py:89-91``), candidate titles in the M2 order.
    """
    where = sample.sample_id
    if not MIN_LEN_SEQ <= len(sample.history) <= HISTORY_LEN:
        raise PromptError(f"{where}: history length {len(sample.history)} is outside [{MIN_LEN_SEQ}, {HISTORY_LEN}]")
    try:
        check_candidates(cands, sample, num_items=NUM_ITEMS, num_candidates=NUM_CANDIDATES)
    except CandidateError as exc:
        raise PromptError(f"{where}: candidates break the §3.1 contract: {exc}") from exc
    titles = catalog.titles
    return render_prompt(templates, template_id, [titles[i] for i in sample.history],
                         [titles[c] for c in cands], hybrid)


def eval_is_hybrid(cfg: PromptConfig) -> bool:
    """Evaluation prompt format (D4): ``hybrid`` (paper_spec) or ``text_only`` (E3)."""
    if cfg.eval_prompt == "hybrid":
        return True
    if cfg.eval_prompt == "text_only":
        return False
    raise PromptError(f"prompt.eval_prompt must be one of {EVAL_PROMPTS}, got {cfg.eval_prompt!r}")


# ----------------------------------------------------------------------------
# Curriculum (§8.6.5, D3)
# ----------------------------------------------------------------------------
def curriculum_hybrid(mode: str, seed: int, tau: int, total: int) -> bool:
    """Whether micro-batch ``tau`` (1..total, over the whole training) uses the hybrid prompt.

    ``linear`` (D3, [LLaRA 간접 §4.3 p.6 식 9-10]): ``u < tau / total`` with
    ``u = seeding.uniform(seed, "curriculum", tau)``; at ``tau == total`` the probability is
    1.0 > u. ``all_hybrid``: always, no draw. ``reverse`` (2024-08 direction,
    ``U@0a69c15 data/data_interface.py:40``): ``u < 1 - tau / total``. ``u`` and the ratio stay
    Python float64 (§8.6.7).
    """
    if mode not in CURRICULA:
        raise PromptError(f"unknown curriculum {mode!r}, expected one of {CURRICULA}")
    if not _is_int(total) or total < 1:
        raise PromptError(f"curriculum total T must be an int >= 1, got {total!r}")
    if not _is_int(tau) or not 1 <= tau <= total:
        raise PromptError(f"curriculum tau must be an int in 1..{total}, got {tau!r}")
    if mode == "all_hybrid":
        return True
    u = seeding.uniform(seed, "curriculum", tau)
    if mode == "linear":
        return u < tau / total
    return u < 1 - tau / total


# ----------------------------------------------------------------------------
# Batch plan (§8.6.6)
# ----------------------------------------------------------------------------
class PlanEntry(NamedTuple):
    """One micro-batch of the plan: the 3-tuple ``(tau, sample_indices, hybrid)`` of §3.1."""
    tau: int
    sample_indices: tuple[int, ...]
    hybrid: bool


@dataclass(frozen=True)
class BatchPlan:
    """The micro-batches of one epoch (§3.1, §8.6.6), made in the main process.

    ``epoch`` counts from 0, ``tau = epoch * batches_per_epoch + b + 1``, ``total`` is
    T = epochs x batches_per_epoch. ``sample_indices`` are positions in
    ``LoadedData.train`` (row order). ``len``, iteration and indexing delegate to
    ``entries`` (M10 cuts accumulation windows with slices).
    """
    epoch: int
    batches_per_epoch: int
    total: int
    curriculum: str
    entries: tuple[PlanEntry, ...]

    def __post_init__(self) -> None:
        nb = self.batches_per_epoch
        if not _is_int(self.epoch) or self.epoch < 0:
            raise PromptError(f"BatchPlan.epoch must be an int >= 0, got {self.epoch!r}")
        if not _is_int(nb) or nb < 1:
            raise PromptError(f"BatchPlan.batches_per_epoch must be an int >= 1, got {nb!r}")
        if not _is_int(self.total) or self.total % nb != 0 or self.total < (self.epoch + 1) * nb:
            raise PromptError(f"BatchPlan.total {self.total!r} is not epochs x {nb} with epoch {self.epoch} "
                              f"inside the training")
        if self.curriculum not in CURRICULA:
            raise PromptError(f"BatchPlan.curriculum must be one of {CURRICULA}, got {self.curriculum!r}")
        if not isinstance(self.entries, tuple) or len(self.entries) != nb:
            raise PromptError(f"BatchPlan needs a tuple of {nb} entries, got {type(self.entries).__name__} "
                              f"of {len(self.entries)}")
        size = None
        seen: set[int] = set()
        for b, entry in enumerate(self.entries):
            if not isinstance(entry, PlanEntry):
                raise PromptError(f"BatchPlan entry {b} is not a PlanEntry: {entry!r}")
            if entry.tau != self.epoch * nb + b + 1 or not _is_int(entry.tau):
                raise PromptError(f"BatchPlan entry {b}: tau {entry.tau!r} != {self.epoch * nb + b + 1}")
            if not isinstance(entry.hybrid, bool):
                raise PromptError(f"BatchPlan entry {b} (tau {entry.tau}): hybrid must be bool")
            idx = entry.sample_indices
            if not isinstance(idx, tuple) or not idx or not all(_is_int(j) and j >= 0 for j in idx):
                raise PromptError(f"BatchPlan entry {b} (tau {entry.tau}): sample_indices must be a non-empty "
                                  f"tuple of ints >= 0, got {idx!r}")
            if size is None:
                size = len(idx)
            elif len(idx) != size:
                raise PromptError(f"BatchPlan entry {b} (tau {entry.tau}): {len(idx)} samples, entry 0 has {size}")
            if seen.intersection(idx) or len(set(idx)) != len(idx):
                raise PromptError(f"BatchPlan entry {b} (tau {entry.tau}): a sample index repeats within the epoch")
            seen.update(idx)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PlanEntry]:
        return iter(self.entries)

    def __getitem__(self, i: Union[int, slice]) -> Union[PlanEntry, tuple[PlanEntry, ...]]:
        return self.entries[i]

    def digest(self) -> str:
        """sha256 of ``"tau:hybrid:i1,i2,..."`` lines (§8.6.6): one value to compare local and VESSL plans."""
        text = "\n".join(f"{e.tau}:{int(e.hybrid)}:{','.join(map(str, e.sample_indices))}" for e in self.entries)
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def hybrid_count(self) -> int:
        """Number of hybrid micro-batches in this epoch."""
        return sum(1 for e in self.entries if e.hybrid)


def batches_per_epoch(num_samples: int, micro_batch_size: int) -> int:
    """NB = N // B (drop_last, ``U data/data_interface.py:142``); N >= B >= 1 (§8.6.6 step 1)."""
    if not _is_int(micro_batch_size) or micro_batch_size < 1:
        raise PromptError(f"micro-batch size must be an int >= 1, got {micro_batch_size!r}")
    if not _is_int(num_samples) or num_samples < micro_batch_size:
        raise PromptError(f"train 표본 {num_samples}개 < micro-batch {micro_batch_size}: no full micro-batch")
    return num_samples // micro_batch_size


def make_batch_plan(*, seed: int, epoch: int, epochs: int, num_samples: int, micro_batch_size: int,
                    curriculum: str) -> BatchPlan:
    """Pure function of (seed, epoch, epochs, N, B, curriculum) (§8.6.6).

    ``perm = torch.randperm(N, generator=torch_generator(seed, "train_order", epoch))`` is
    the only draw of that generator (``torch2.0 utils/data/sampler.py:132`` orders a
    shuffled DataLoader the same way); micro-batch ``b`` is ``perm[b*B:(b+1)*B]`` with
    ``tau = epoch * NB + b + 1``; the last ``N - NB*B`` positions are not used this epoch.
    """
    if curriculum not in CURRICULA:
        raise PromptError(f"unknown curriculum {curriculum!r}, expected one of {CURRICULA}")
    nb = batches_per_epoch(num_samples, micro_batch_size)
    if not _is_int(epochs) or epochs < 1:
        raise PromptError(f"epochs must be an int >= 1, got {epochs!r}")
    if not _is_int(epoch) or not 0 <= epoch < epochs:
        raise PromptError(f"epoch must be an int in 0..{epochs - 1} (epochs count from 0), got {epoch!r}")
    total = epochs * nb
    generator = seeding.torch_generator(seed, "train_order", epoch)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    size = micro_batch_size
    entries = []
    for b in range(nb):
        tau = epoch * nb + b + 1
        entries.append(PlanEntry(tau=tau, sample_indices=tuple(perm[b * size:(b + 1) * size]),
                                 hybrid=curriculum_hybrid(curriculum, seed, tau, total)))
    return BatchPlan(epoch=epoch, batches_per_epoch=nb, total=total, curriculum=curriculum, entries=tuple(entries))


def build_batch_plan(samples: Sequence[Sample], epoch: int, cfg: "Config") -> BatchPlan:
    """M10's call (§14.6.4): ``make_batch_plan`` with ``run.seed``, ``train.epochs``,
    ``train.micro_batch_size``, ``prompt.train_curriculum`` and N = ``len(samples)``.

    ``samples`` must be ``LoadedData.train`` (full or smoke): train samples in strictly
    ascending row order, because ``sample_indices`` are positions in that tuple.
    """
    previous = -1
    for position, s in enumerate(samples):
        if not isinstance(s, Sample) or s.split != "train":
            raise PromptError(f"build_batch_plan needs LoadedData.train; position {position} holds "
                              f"{getattr(s, 'sample_id', s)!r}, not a train sample")
        if s.row <= previous:
            raise PromptError(f"build_batch_plan needs train samples in strictly ascending row order; "
                              f"row {s.row} at position {position} follows row {previous}")
        previous = s.row
    return make_batch_plan(seed=cfg.run.seed, epoch=epoch, epochs=cfg.train.epochs, num_samples=len(samples),
                           micro_batch_size=cfg.train.micro_batch_size, curriculum=cfg.prompt.train_curriculum)


__all__ = [
    "PromptError", "HISTORY_SLOT", "CANDS_SLOT", "HISTORY_TAG", "CANS_TAG", "PH_TAG", "ITEM_SEP", "NUM_CANDIDATES",
    "CURRICULA", "TEMPLATE_CHOICES", "EVAL_PROMPTS", "TEMPLATE_FILE", "TemplateSet", "load_templates",
    "parse_templates", "check_catalog_titles", "choose_template", "render_prompt", "build_prompt", "eval_is_hybrid",
    "curriculum_hybrid", "PlanEntry", "BatchPlan", "batches_per_epoch", "make_batch_plan", "build_batch_plan",
]
