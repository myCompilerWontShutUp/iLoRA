"""M11 생성과 평가 (DESIGN.md §15; 15.6.1 공개 API, 15.6.2-15.6.14, 결정 D4, D9, D10, D11, D15-D17, D20, D22).

The model (M9) generates for the whole validation or test split in row order, with the fixed
candidates (D10, key ``("candidates", split, row)``), the fixed templates (D15, key
``("template", split, row)``), the hybrid prompt (D4; ``text_only`` for E3), left padding (D22),
``eval.batch_size`` and no drop_last (D17):

- ``generate_records``: the loader of M5, a check of every batch against the keyed values
  (15.6.2 step 5), greedy generation inside the M0 autocast of ``train.precision`` (D9), a check of
  every ``GenerationOutput`` (15.6.3), ``trim_generated`` + ``batch_decode`` + ``first_line`` +
  ``judge`` per sample ([코드] U model/model_interface.py:89-90, :165, :468-492), the M2 flags and the
  omega used in generation. The whole pass runs under ``torch.random.fork_rng`` with the model mode
  restored, and the M7 recorder collects omega and z at the one collection point [SPEC §4], which is
  compared bit for bit with the per-batch outputs.
- ``evaluate_split``: D17 metrics (ValidRatio, HR@1 = correct/valid, CombinedMetric = correct/total),
  the auxiliary drop_last protocols of the test split (N = 92 and 88, 15.6.6), the supplementary
  statistics (Wilson 95%, flags, generation diagnostics, 15.6.7), the optional batch-size check
  (15.6.10) and the files of ``eval/{ckpt_tag}/{split}/`` written once at the end through M12
  (``save_npz``, ``write_eval``; ``metrics.json`` last marks completion, 15.6.8).
- ``run_eval_command``: ``main.py eval`` (15.6.11): a re-evaluation of a training run's checkpoints,
  read only through M10 ``load_model_state``, with the reproduction check written to
  ``eval/summary.json``.

M12 (``records``) and M10 (``train``) are imported inside the functions that use them: ``records``
imports this module's ``EvalRecord`` inside its own functions, and ``train`` calls ``evaluate_split``
(no import cycle). Errors are ``EvalError`` with the split, batch number and sample_id; errors of
M2, M4, M5, M7, M9 and M12 propagate unchanged (15.6.13). Nothing is regenerated, skipped or
estimated after an error.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping, Optional, Sequence

import numpy as np
import torch

from . import run
from .batching import EvalBatch, EvalBatches, TokenizerBundle, make_loader
from .candidates import FLAG_NAMES, NUM_CANDIDATES, candidate_flags, candidates_for, match_norm
from .data import EXPECTED_SAMPLES, ItemCatalog, LoadedData, Sample
from .precision import autocast_for
from .prompts import TemplateSet, choose_template, eval_is_hybrid, load_templates

if TYPE_CHECKING:   # type names only (15.6.1): M7 recorder output, M0 config
    from .router import GateRecord
    from .sections import Config

logger = logging.getLogger("ilora.evaluate")


class EvalError(RuntimeError):
    """An evaluation step breaks the M11 rules (15.6.13); the message names split, batch and sample_id."""


EVAL_SPLITS: Final = ("val", "test")
FULL_SPLIT_SIZE: Final = {split: EXPECTED_SAMPLES[split] for split in EVAL_SPLITS}   # D17, M1 5.6.2: 94, 95
# Auxiliary drop_last protocols of the test split (15.6.6): [코드] U main.py:82 (batch 4),
# U train_movielens.sh:5 (batch 8), U data/data_interface.py:158 (drop_last=True).
AUX_DROP_LAST_BATCH_SIZES: Final = (4, 8)
WILSON_Z: Final = statistics.NormalDist().inv_cdf(0.975)   # 1.9599639845400536, supplementary only (15.6.7)
CKPT_TAG_RE: Final = re.compile(r"epoch(0|[1-9][0-9]*)|init")
FIRST_LINE_RULE: Final = 'strip().split("\\n")[0]'        # [코드] U model/model_interface.py:90, :165, :200
SMOKE_SPLIT_INDEX: Final = {"val": 1, "test": 2}          # data.smoke_users = (train, val, test), D14
BATCH_INVARIANCE_FILE: Final = "batch_invariance.json"    # 15.6.10 (eval.batch_invariance_check only)
SUMMARY_FILE: Final = "summary.json"                      # eval/summary.json of main.py eval (15.6.11)
AUTOCAST_DTYPE: Final = {"fp16_amp": "float16", "bf16_amp": "bfloat16", "fp32": None}   # D9, M0 precision tools
# main.py eval (15.6.11 step 2): the only config differences allowed against the training run.
EVAL_ALLOWED_SECTIONS: Final = ("eval", "records")
EVAL_ALLOWED_KEYS: Final = ("prompt.eval_prompt", "batching.num_workers", "data.subset", "data.smoke_users",
                            "data.train_rows")
# Allowed differences that can change the evaluation result (15.6.10): compare only when all are equal.
REPRODUCTION_KEYS: Final = ("eval.batch_size", "prompt.eval_prompt", "data.subset", "data.smoke_users")
_SELECTION_KEYS: Final = {"best_val": "best_val_epoch", "last": "last_epoch"}   # M10 eval/selection.json (14.6.7)


# ----------------------------------------------------------------------------
# Contract values (15.6.1, 3.1 EvalRecord, SplitMetrics, EvalResult)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Judgement:
    """One judgement ([코드] U model/model_interface.py:468-492)."""
    matched: tuple[int, ...]          # candidate ids whose normalized title is in the first line (candidate order)
    valid: bool                       # len(matched) == 1
    correct: bool                     # valid and that candidate's normalized title == the normalized target title


@dataclass(frozen=True)
class EvalRecord:
    """One evaluated sample (3.1 contract). Field order = ``records.jsonl`` key order (15.6.8).

    ``raw_text`` is the decoded text before ``strip``, ``gates`` the omega row used in generation (K floats,
    written as ``gate_0..gate_{K-1}``), ``flags`` the M2 ``candidate_flags`` in ``FLAG_NAMES`` order and
    ``gen_token_ids`` the new tokens up to and including the first EOS. No checks here: ``generate_records``
    builds records only after every check, and M12 ``write_eval`` checks them again before writing.
    """
    sample_id: str
    split: str
    user_id: int
    template_id: int
    hybrid: bool
    cand_ids: tuple[int, ...]
    cand_titles: tuple[str, ...]
    target_id: int
    target_title: str
    raw_text: str
    first_line: str
    matched: tuple[int, ...]
    valid: bool
    correct: bool
    gates: tuple[float, ...]
    flags: Mapping[str, bool]
    gen_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SplitMetrics:
    """D17 metrics of one split (field names = M12 ``metrics.json`` keys)."""
    split: str
    n_total: int
    n_valid: int
    n_correct: int
    valid_ratio: float                # n_valid / n_total (ValidRatio)
    hr_at_1: float                    # n_correct / n_valid, 0.0 when n_valid = 0 (HR@1, code ``hr``)
    combined_metric: float            # n_correct / n_total (CombinedMetric, code ``metric``)


@dataclass(frozen=True, eq=False)
class EvalResult:
    """Result of one (checkpoint, split) evaluation (M10 14.6.7). ``eq=False``: compare arrays with numpy."""
    split: str
    ckpt_tag: str
    metrics: SplitMetrics
    aux: Mapping[str, SplitMetrics]   # test: drop_last_bs4, drop_last_bs8; val: {}
    records: tuple[EvalRecord, ...]   # row order
    omega: np.ndarray                 # [N, K] float32
    z: np.ndarray                     # [N, d] float32
    out_dir: Path
    seconds: float


@dataclass(frozen=True, eq=False)
class GeneratedSplit:
    """Module-internal bundle of one generation pass (15.6.1): records, omega, z, diagnostics, call counts.

    Nothing is written to files. ``gate_record`` is the M7 ``GateRecord`` when the recorder was on.
    """
    split: str
    batch_size: int
    records: tuple[EvalRecord, ...]
    omega: np.ndarray                 # [N, K] float32, the per-batch GenerationOutput.omega concatenated
    z: np.ndarray                     # [N, d] float32
    n_batches: int                    # generate calls
    gate_calls: int                   # increase of Router.gate_calls (dynamic: n_batches, else 0)
    diagnostics: Mapping[str, int]    # 15.6.7 generation diagnostics
    gate_record: Optional["GateRecord"]
    seconds: float


def _is_int(value: object) -> bool:
    """Python or numpy integer, bool excluded."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_split(split: object) -> str:
    if split not in EVAL_SPLITS:
        raise EvalError(f"evaluation split must be one of {EVAL_SPLITS}, got {split!r}")
    return str(split)


def _check_tag(ckpt_tag: object) -> str:
    if not isinstance(ckpt_tag, str) or not CKPT_TAG_RE.fullmatch(ckpt_tag):
        raise EvalError(f"ckpt_tag must be epoch<e> (e from 0, no leading zero) or init, got {ckpt_tag!r}")
    return ckpt_tag


def _tag_epoch(ckpt_tag: str) -> Optional[int]:
    """``epoch{e}`` -> e, ``init`` -> None (``metrics.json`` ``epoch``, 15.6.9)."""
    return None if _check_tag(ckpt_tag) == "init" else int(ckpt_tag[len("epoch"):])


# ----------------------------------------------------------------------------
# First line, judgement, trimming and decoding (15.6.3-15.6.5)
# ----------------------------------------------------------------------------
def first_line(raw_text: str) -> str:
    """``raw_text.strip().split("\\n")[0]`` ([코드] U model/model_interface.py:90, :165, :200).

    Splits on ``"\\n"`` only (not ``splitlines``: ``\\r``, ``\\x85`` and ``\\u2028`` do not split); a trailing
    ``\\r`` is removed later by ``match_norm``. The second ``strip`` of the released code has no effect.
    """
    if not isinstance(raw_text, str):
        raise EvalError(f"raw_text must be str, got {type(raw_text).__name__}")
    return raw_text.strip().split("\n")[0]


def judge(first_line: str, cand_ids: Sequence[int], cand_titles: Sequence[str], target_title: str) -> Judgement:
    """Judgement of one first line ([코드] U model/model_interface.py:476-486, unchanged; D20).

    Every text is normalized by M2 ``match_norm`` (``strip().lower().strip()``, the one definition shared
    with the flags). Candidates whose normalized title is a substring of the normalized first line are
    collected in candidate order (two candidates with the same normalized title both match); exactly one
    match is valid, and a valid match is correct when its normalized title equals the normalized target
    title. ``[결정]`` input checks: 20 candidate ids and 20 titles, ids are integers without repeats
    (``EvalError`` otherwise). Empty, short and duplicated titles are not repaired (flags, D20).
    """
    if not isinstance(first_line, str) or not isinstance(target_title, str):
        raise EvalError(f"first_line and target_title must be str, got {type(first_line).__name__} and "
                        f"{type(target_title).__name__}")
    if isinstance(cand_ids, (str, bytes)) or isinstance(cand_titles, (str, bytes)):
        raise EvalError("cand_ids and cand_titles must be sequences, not one string")
    ids, titles = tuple(cand_ids), tuple(cand_titles)
    if len(ids) != NUM_CANDIDATES or len(titles) != NUM_CANDIDATES:
        raise EvalError(f"judge needs {NUM_CANDIDATES} candidate ids and titles (D10), got {len(ids)} ids and "
                        f"{len(titles)} titles")
    if not all(_is_int(c) for c in ids):
        raise EvalError(f"candidate ids must be integers, got {ids}")
    if not all(isinstance(t, str) for t in titles):
        raise EvalError("candidate titles must be str")
    ids = tuple(int(c) for c in ids)
    if len(set(ids)) != len(ids):
        repeated = sorted({c for c in ids if ids.count(c) > 1})
        raise EvalError(f"candidate ids repeat {repeated} (3.1 후보: no duplicates)")
    generated = match_norm(first_line)                                  # U :476
    real = match_norm(target_title)                                     # U :477
    hits = [(c, norm) for c, t in zip(ids, titles) if (norm := match_norm(t)) in generated]   # U :478-482
    valid = len(hits) == 1                                              # U :483
    correct = valid and hits[0][1] == real                              # U :485
    return Judgement(matched=tuple(c for c, _ in hits), valid=valid, correct=correct)


def trim_generated(row: Sequence[int], *, eos_id: int) -> tuple[int, ...]:
    """New tokens up to and including the first EOS; the ``[PAD]`` after it is dropped ``[결정]``.

    Without an EOS the whole row (then T = ``gen_max_new_tokens``). Only the pad length after EOS depends on
    the batch, so the result is the generation of one sample (15.6.3).
    """
    if not _is_int(eos_id):
        raise EvalError(f"eos_id must be an integer, got {eos_id!r}")
    if isinstance(row, (str, bytes)):
        raise EvalError("a generated row must be a sequence of token ids")
    values = list(row)
    if not all(_is_int(t) for t in values):
        raise EvalError(f"generated token ids must be integers, got {values[:8]}...")
    values = [int(t) for t in values]
    if int(eos_id) in values:
        return tuple(values[:values.index(int(eos_id)) + 1])
    return tuple(values)


def decode_generated(bundle: TokenizerBundle, rows: Sequence[Sequence[int]]) -> list[str]:
    """``batch_decode(rows, skip_special_tokens=True, clean_up_tokenization_spaces=False)`` ([코드] U :89).

    The 4.28.0 slow tokenizer drops ``all_special_ids`` (unk 0, bos 1, eos 2, 32000-32004) first and joins the
    rest with ``sp_model.decode`` (T4.28 tokenization_utils.py:883-913, :921-931,
    models/llama/tokenization_llama.py:132-149). The text is returned before ``strip`` (``raw_text``, 15.6.4).
    """
    texts = bundle.tokenizer.batch_decode([list(r) for r in rows], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
    if len(texts) != len(rows) or not all(isinstance(t, str) for t in texts):
        raise EvalError(f"batch_decode returned {len(texts)} texts for {len(rows)} rows")
    return list(texts)


# ----------------------------------------------------------------------------
# Metrics (15.6.6, D17) and supplementary statistics (15.6.7)
# ----------------------------------------------------------------------------
def compute_metrics(records: Sequence[EvalRecord], split: str) -> SplitMetrics:
    """D17 from integer counts: ``valid_ratio = v/n``, ``hr_at_1 = c/v`` (0.0 when v = 0, [코드] U :487-491),
    ``combined_metric = c/n`` ``[결정]`` (one division of the integer ratio, not the float product ``hr x
    valid_ratio`` of U :181). n = 0, a record of another split or ``correct`` without ``valid`` -> ``EvalError``.
    """
    split = _check_split(split)
    records = tuple(records)
    n = len(records)
    if n == 0:
        raise EvalError(f"{split}: no records to compute metrics from")
    for r in records:
        if r.split != split:
            raise EvalError(f"{split}: record {r.sample_id} has split {r.split!r}")
        if bool(r.correct) and not bool(r.valid):
            raise EvalError(f"{split}: record {r.sample_id} is correct but not valid")
    v = sum(1 for r in records if bool(r.valid))
    c = sum(1 for r in records if bool(r.correct))
    return SplitMetrics(split=split, n_total=n, n_valid=v, n_correct=c, valid_ratio=v / n,
                        hr_at_1=c / v if v else 0.0, combined_metric=c / n)


def aux_protocol_metrics(records: Sequence[EvalRecord], split: str) -> dict[str, SplitMetrics]:
    """Auxiliary protocols of the test split ``[결정]`` (D17 variant): ``drop_last_bs{b}`` = the first
    floor(N/b)*b records in row order (test 95 -> 92 and 88, smoke 10 -> 8 and 8), the sample sets of the
    released test loader (``shuffle=False, drop_last=True``, U data/data_interface.py:152-160) with batch 4
    (U main.py:82) and 8 (U train_movielens.sh:5). Validation: ``{}`` (its loader keeps every sample,
    U data/data_interface.py:145-150). Computed from the same records, nothing is generated again.
    """
    split = _check_split(split)
    records = tuple(records)
    if split == "val":
        return {}
    out: dict[str, SplitMetrics] = {}
    for b in AUX_DROP_LAST_BATCH_SIZES:
        keep = (len(records) // b) * b
        if keep == 0:
            raise EvalError(f"{split}: drop_last_bs{b} needs at least {b} records, got {len(records)}")
        out[f"drop_last_bs{b}"] = compute_metrics(records[:keep], split)
    return out


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> Optional[tuple[float, float]]:
    """Wilson score interval ``[결정]`` (supplementary only, 15.6.7), clipped to [0, 1]; None when n = 0.

    p = k/n, center = (p + z^2/2n) / (1 + z^2/n), half = z / (1 + z^2/n) * sqrt(p(1 - p)/n + z^2/4n^2).
    """
    if not _is_int(k) or not _is_int(n) or int(n) < 0 or not 0 <= int(k) <= int(n):
        raise EvalError(f"wilson_interval needs integers 0 <= k <= n, got k={k!r}, n={n!r}")
    if not isinstance(z, (int, float)) or isinstance(z, bool) or not math.isfinite(z) or z <= 0:
        raise EvalError(f"z must be a finite number > 0, got {z!r}")
    k, n = int(k), int(n)
    if n == 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    # k = 0 (k = n) makes center - half (center + half) exactly 0 (1); keep the exact value, not the rounding residue
    low = 0.0 if k == 0 else max(0.0, center - half)
    high = 1.0 if k == n else min(1.0, center + half)
    return low, high


def eval_dir(run_dir: Path, ckpt_tag: str, split: str) -> Path:
    """``<run_dir>/eval/{ckpt_tag}/{split}/`` (15.2; ``split`` is the contract value ``val``/``test``)."""
    return Path(run_dir) / "eval" / _check_tag(ckpt_tag) / _check_split(split)


# ----------------------------------------------------------------------------
# Expected values, batch and output checks (15.6.2 step 3 and 5, 15.6.3)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Expected:
    """Keyed values of one sample (15.6.2 step 3)."""
    sample: Sample
    cands: tuple[int, ...]
    template_id: int
    flags: Mapping[str, bool]


def _expected_rows(samples: Sequence[Sample], cfg: "Config", catalog: ItemCatalog,
                   n_templates: int) -> list[_Expected]:
    """Candidates (D10, M2 6.6.2), template (D15, M4 8.6.3) and flags (M2 6.6.4) of every sample, main process."""
    out = []
    for s in samples:
        cands = candidates_for(s, main_seed=cfg.run.seed, epoch=None, cfg=cfg.candidates)
        k = choose_template(cfg.prompt, cfg.run.seed, n_templates, s, None)
        out.append(_Expected(sample=s, cands=cands, template_id=k, flags=candidate_flags(cands, s.target, catalog)))
    return out


def check_batch(batch: EvalBatch, index: int, *, start: int, batch_size: int, expected: Sequence[_Expected],
                hybrid: bool, catalog: ItemCatalog, split: str) -> int:
    """15.6.2 step 5 ``[결정]``: batch ``index`` must hold rows ``[start, min(start + batch_size, N))`` in row order
    with the keyed candidates, templates, targets and titles and the evaluation format. Returns B.

    M5 builds the batch with the same functions; the check guarantees at the evaluation boundary that the
    recorded values are the D10/D15 key values (catches a changed order or a wrong epoch key).
    """
    where = f"{split} batch {index}"
    if not isinstance(batch, EvalBatch):
        raise EvalError(f"{where}: expected an EvalBatch, got {type(batch).__name__}")
    group = list(expected[start:start + batch_size])
    want_ids = [e.sample.sample_id for e in group]
    got_ids = list(batch.sample_ids)
    if got_ids != want_ids:
        pos = next((i for i, (a, b) in enumerate(zip(got_ids, want_ids)) if a != b), min(len(got_ids), len(want_ids)))
        have = got_ids[pos] if pos < len(got_ids) else None
        want = want_ids[pos] if pos < len(want_ids) else None
        raise EvalError(f"{where}: sample_ids differ from the row order at position {pos} (sample_id {have!r}, "
                        f"expected {want!r}; {len(got_ids)} rows, expected {len(want_ids)})")
    b = len(group)
    if batch.hybrid is not hybrid:
        raise EvalError(f"{where} (sample_id {want_ids[0]!r}): hybrid {batch.hybrid!r}, expected {hybrid} "
                        f"(prompt.eval_prompt, D4)")
    shapes = {"cand_ids": (b, NUM_CANDIDATES), "target_ids": (b,)}
    for name, shape in shapes.items():
        value = getattr(batch, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise EvalError(f"{where}: {name} must have shape {list(shape)}")
    if len(batch.template_ids) != b or len(batch.cand_titles) != b or len(batch.target_titles) != b:
        raise EvalError(f"{where}: template_ids, cand_titles and target_titles need {b} rows")
    cand_rows, targets = batch.cand_ids.tolist(), batch.target_ids.tolist()
    titles = catalog.titles
    for r, e in enumerate(group):
        sid = e.sample.sample_id
        if cand_rows[r] != list(e.cands):
            raise EvalError(f"{where} row {r} (sample_id {sid!r}): cand_ids {cand_rows[r]} != keyed candidates "
                            f"{list(e.cands)} (D10)")
        if batch.template_ids[r] != e.template_id:
            raise EvalError(f"{where} row {r} (sample_id {sid!r}): template {batch.template_ids[r]!r} != keyed "
                            f"template {e.template_id} (D15)")
        if targets[r] != e.sample.target:
            raise EvalError(f"{where} row {r} (sample_id {sid!r}): target id {targets[r]} != {e.sample.target}")
        if list(batch.cand_titles[r]) != [titles[c] for c in e.cands]:
            raise EvalError(f"{where} row {r} (sample_id {sid!r}): candidate titles differ from the catalog titles")
        if batch.target_titles[r] != titles[e.sample.target]:
            raise EvalError(f"{where} row {r} (sample_id {sid!r}): target title {batch.target_titles[r]!r} != "
                            f"catalog title {titles[e.sample.target]!r}")
    return b


def check_output(out: Any, *, batch_size: int, num_experts: int, z_dim: int, max_new_tokens: int, eos_id: int,
                 pad_id: int, rowsum_atol: float, sample_ids: Optional[Sequence[str]] = None) -> None:
    """The ``GenerationOutput`` format of M9 checked again at the evaluation boundary ``[결정]`` (15.6.3).

    ``new_token_ids``: CPU int64 [B, T] with 1 <= T <= ``max_new_tokens``, only ``[PAD]`` after the first EOS
    (greedy pads finished rows, T4.28 generation/utils.py:2285-2288) and an EOS in every row when
    T < ``max_new_tokens`` (generation stops when all rows are finished, :2300-2305). omega and z: CPU tensors
    that pass M12 ``validate_gate`` (float32 [B, K], finite, >= 0, row sums) and ``validate_z`` (float32 [B, d]);
    their ``RecordsError`` propagates unchanged (15.6.13). Other failures raise ``EvalError``.
    """
    from .records import validate_gate, validate_z

    shown = list(sample_ids) if sample_ids is not None else None
    ids = getattr(out, "new_token_ids", None)
    if not isinstance(ids, torch.Tensor):
        raise EvalError(f"new_token_ids must be a tensor, got {type(ids).__name__}")
    if ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.dim() != 2:
        raise EvalError(f"new_token_ids must be a CPU int64 [B, T] tensor, got {ids.device} {ids.dtype} "
                        f"{list(ids.shape)}")
    if ids.shape[0] != batch_size:
        raise EvalError(f"new_token_ids has {ids.shape[0]} rows for a batch of {batch_size}")
    steps = int(ids.shape[1])
    if not 1 <= steps <= max_new_tokens:
        raise EvalError(f"new_token_ids has T = {steps} new tokens, expected 1..{max_new_tokens}")
    is_eos = ids == eos_id
    seen = is_eos.long().cumsum(dim=1) - is_eos.long()         # EOS count strictly before each position
    after = (seen > 0) & (ids != pad_id)
    if bool(after.any()):
        row = int(after.any(dim=1).nonzero()[0, 0])
        raise EvalError(f"row {row} (sample_id {shown[row] if shown else row!r}) has a token other than [PAD] "
                        f"{pad_id} after its first EOS: {ids[row].tolist()}")
    if steps < max_new_tokens and not bool(is_eos.any(dim=1).all()):
        row = int((~is_eos.any(dim=1)).nonzero()[0, 0])
        raise EvalError(f"T = {steps} < {max_new_tokens} but row {row} (sample_id "
                        f"{shown[row] if shown else row!r}) has no EOS")
    for name in ("omega", "z"):
        value = getattr(out, name, None)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise EvalError(f"GenerationOutput.{name} must be a CPU tensor, got "
                            f"{value.device if isinstance(value, torch.Tensor) else type(value).__name__}")
    validate_gate(out.omega, batch=batch_size, num_experts=num_experts, rowsum_atol=rowsum_atol, sample_ids=shown)
    validate_z(out.z, batch=batch_size, dim=z_dim, sample_ids=shown)


def _build_record(e: _Expected, *, catalog: ItemCatalog, cand_titles: Sequence[str], target_title: str,
                  template_id: int, hybrid: bool, raw_text: str, gates: Sequence[float],
                  gen_token_ids: tuple[int, ...]) -> EvalRecord:
    """One ``EvalRecord`` after the record checks of 15.6.5 ``[결정]``: the target is a candidate, the titles are
    the catalog titles (``data.title_mode``), and for a valid judgement string-correct <=> ``matched[0] ==
    target_id`` (always true when the target is a candidate: a second candidate with the target's normalized
    title would also match, so a failure means a normalization error)."""
    s = e.sample
    if s.target not in e.cands:
        raise EvalError(f"{s.sample_id}: target {s.target} is not among the candidates {e.cands} (3.1 후보)")
    titles = tuple(cand_titles)
    if titles != tuple(catalog.titles[c] for c in e.cands):
        raise EvalError(f"{s.sample_id}: candidate titles differ from the catalog titles of the candidates")
    if target_title != catalog.titles[s.target]:
        raise EvalError(f"{s.sample_id}: target title {target_title!r} != catalog title {catalog.titles[s.target]!r}")
    line = first_line(raw_text)
    verdict = judge(line, e.cands, titles, target_title)
    if verdict.valid and verdict.correct != (verdict.matched[0] == s.target):
        raise EvalError(f"{s.sample_id}: string judgement correct={verdict.correct} but matched[0] = "
                        f"{verdict.matched[0]} vs target {s.target} (normalization error, 15.6.5)")
    return EvalRecord(sample_id=s.sample_id, split=s.split, user_id=s.user_id, template_id=int(template_id),
                      hybrid=bool(hybrid), cand_ids=tuple(e.cands), cand_titles=titles, target_id=s.target,
                      target_title=target_title, raw_text=raw_text, first_line=line, matched=verdict.matched,
                      valid=verdict.valid, correct=verdict.correct, gates=tuple(float(g) for g in gates),
                      flags={name: bool(e.flags[name]) for name in FLAG_NAMES}, gen_token_ids=gen_token_ids)


# ----------------------------------------------------------------------------
# Generation pass (15.6.3)
# ----------------------------------------------------------------------------
DIAGNOSTICS: Final = ("n_no_eos", "n_empty_first_line", "n_blank_leading_segment", "n_cut_at_max_tokens",
                      "n_no_match", "n_multi_match")   # 15.6.7 generation diagnostics (sample counts)


def _check_gen_config(model: Any, cfg: "Config", eos_id: int, pad_id: int) -> None:
    """15.6.3 ``[결정]``: greedy (do_sample False, num_beams 1), the ``model.gen_*`` values, one EOS id (2) and
    ``[PAD]`` (32000); generation ends only at EOS or the maximum length (no ``max_time``, no newline stop)."""
    gc, mc = model.gen_config, cfg.model
    want = {"do_sample": False, "num_beams": 1, "max_new_tokens": mc.gen_max_new_tokens,
            "min_new_tokens": mc.gen_min_new_tokens, "repetition_penalty": mc.gen_repetition_penalty,
            "pad_token_id": pad_id, "max_time": None}
    problems = [f"{key} {getattr(gc, key, None)!r} != {value!r}" for key, value in want.items()
                if getattr(gc, key, None) != value]
    eos = getattr(gc, "eos_token_id", None)
    if not _is_int(eos) or int(eos) != eos_id:
        problems.append(f"eos_token_id {eos!r} must be the one id {eos_id}")
    if mc.gen_strategy != "greedy":
        problems.append(f"model.gen_strategy {mc.gen_strategy!r} is not greedy")
    if problems:
        raise EvalError("model.gen_config does not match the evaluation settings (15.6.3): " + "; ".join(problems))


def _check_cudnn() -> None:
    """15.6.10 ``[결정]``: cuDNN autotuning may pick other algorithms per run (SASRec Conv1d uses cuDNN, M3)."""
    if torch.backends.cudnn.benchmark:
        raise EvalError("torch.backends.cudnn.benchmark is True; evaluation must be deterministic (1.3, D16)")


def _fork_devices(device: torch.device) -> list[int]:
    """``fork_rng`` devices: the CUDA index of the model (cuda:0 -> [0]) or none on CPU (15.6.13)."""
    if device.type == "cuda":
        return [device.index if device.index is not None else torch.cuda.current_device()]
    return []


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """float32 tensors equal bit for bit (``torch.equal`` would treat -0.0 and 0.0 as equal)."""
    return (a.dtype == b.dtype == torch.float32 and tuple(a.shape) == tuple(b.shape)
            and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def _check_gate_record(rec: "GateRecord", *, split: str, expected_ids: Sequence[str], omega: torch.Tensor,
                       z: torch.Tensor, gate_calls: int, routing_mode: str, num_experts: int) -> None:
    """One collection point [SPEC §4] (15.6.3): the recorder's omega and z are the per-batch outputs bit for bit."""
    problems = []
    if rec.split != split or tuple(rec.sample_ids) != tuple(expected_ids):
        problems.append(f"recorded split {rec.split!r} / {len(rec.sample_ids)} sample ids differ from the "
                        f"{len(expected_ids)} evaluated")
    if rec.routing_mode != routing_mode or rec.num_experts != num_experts:
        problems.append(f"recorded routing {rec.routing_mode!r} K={rec.num_experts}, expected {routing_mode!r} "
                        f"K={num_experts}")
    if not _same_bits(rec.omega, omega):
        problems.append("recorded omega differs from the GenerationOutput omega")
    if not _same_bits(rec.z, z):
        problems.append("recorded z differs from the GenerationOutput z")
    if rec.gate_calls != gate_calls:
        problems.append(f"recorded gate_calls {rec.gate_calls} != {gate_calls}")
    if problems:
        raise EvalError(f"{split}: gate record check failed (one collection point, [SPEC §4]): " + "; ".join(problems))


def _count_diagnostics(counts: Counter, rec: EvalRecord, *, eos_id: int, max_new_tokens: int) -> None:
    """15.6.7 generation diagnostics of one record (recorded only, no judgement)."""
    no_eos = eos_id not in rec.gen_token_ids
    counts["n_no_eos"] += int(no_eos)
    counts["n_empty_first_line"] += int(rec.first_line == "")
    counts["n_blank_leading_segment"] += int("\n" in rec.raw_text and rec.raw_text.split("\n")[0].strip() == "")
    counts["n_cut_at_max_tokens"] += int(no_eos and len(rec.gen_token_ids) == max_new_tokens and "\n" not in rec.raw_text)
    counts["n_no_match"] += int(len(rec.matched) == 0)
    counts["n_multi_match"] += int(len(rec.matched) >= 2)


def generate_records(model: Any, bundle: TokenizerBundle, data: LoadedData, split: str, cfg: "Config", *,
                     batch_size: int, templates: TemplateSet, record_gates: bool) -> GeneratedSplit:
    """Generate and judge every sample of ``split`` in row order (15.6.2 steps 2-5, 15.6.3-15.6.5). No files.

    ``EvalBatches`` + ``make_loader`` (M5 9.6.8) in groups ``[i, min(i + batch_size, N))``, each batch checked
    against the keyed values (``check_batch``), ``model.generate`` inside ``autocast_for(train.precision)`` (D9),
    ``check_output``, ``trim_generated``, ``decode_generated``, ``first_line``, ``judge``. The model is put in eval
    mode and its mode restored; the pass runs under ``torch.random.fork_rng`` (the DataLoader iterator draws
    its base seed from the global CPU generator, torch2.0 utils/data/dataloader.py:605). With ``record_gates``
    the M7 recorder collects omega and z and is compared bit for bit with the outputs; on any exception the
    recording is discarded (``discard_recording``) before the exception propagates. ``Router.gate_calls``
    must grow by the number of generate calls (dynamic) or not at all (cluster_hard, none).
    """
    split = _check_split(split)
    if not _is_int(batch_size) or int(batch_size) < 1:
        raise EvalError(f"batch_size must be an int >= 1, got {batch_size!r}")
    if not isinstance(record_gates, bool):
        raise EvalError(f"record_gates must be bool, got {record_gates!r}")
    samples = data.samples(split)
    if not samples:
        raise EvalError(f"{split}: no samples")
    catalog, ids = data.catalog, bundle.ids
    k = cfg.lora.num_experts
    router = model.omega_source
    if getattr(router, "num_experts", None) != k or getattr(model, "num_experts", k) != k:
        raise EvalError(f"the model has K = {getattr(router, 'num_experts', None)} experts, lora.num_experts = {k}")
    _check_gen_config(model, cfg, ids.eos, ids.pad)
    _check_cudnn()
    hybrid = eval_is_hybrid(cfg.prompt)                                          # D4 (text_only: E3)
    expected = _expected_rows(samples, cfg, catalog, len(templates.templates))
    expected_ids = [s.sample_id for s in samples]
    dev, z_dim, max_new = model.device, int(model.sasrec.hidden_size), cfg.model.gen_max_new_tokens
    cand_fn = functools.partial(candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)   # CandidateFn (M5 9.6.11)
    dataset = EvalBatches(samples, int(batch_size), catalog, templates, cand_fn, bundle, cfg.prompt, cfg.batching,
                          cfg.run.seed)
    records: list[EvalRecord] = []
    omegas: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    counts: Counter = Counter({name: 0 for name in DIAGNOSTICS})
    n_batches, gate_record = 0, None
    was_training = model.training
    started = time.perf_counter()
    recording = False
    model.eval()                                                                 # LoRA dropout off; SASRec eval (D12)
    try:
        with torch.random.fork_rng(devices=_fork_devices(dev)):
            if record_gates:
                router.start_recording(split)
                recording = True
            calls0 = router.gate_calls
            start = 0
            for index, batch in enumerate(make_loader(dataset, cfg.batching.num_workers, pin_memory=dev.type == "cuda")):
                b = check_batch(batch, index, start=start, batch_size=int(batch_size), expected=expected,
                                hybrid=hybrid, catalog=catalog, split=split)
                with autocast_for(cfg.train.precision, dev):                     # M0 precision tools (D9)
                    out = model.generate(batch)                                  # M9 13.6.6
                n_batches += 1
                check_output(out, batch_size=b, num_experts=k, z_dim=z_dim, max_new_tokens=max_new, eos_id=ids.eos,
                             pad_id=ids.pad, rowsum_atol=cfg.records.gate_rowsum_atol, sample_ids=batch.sample_ids)
                rows = [trim_generated(row, eos_id=ids.eos) for row in out.new_token_ids.tolist()]
                texts = decode_generated(bundle, rows)
                gates = out.omega.tolist()                                       # float32 values as Python floats
                for r in range(b):
                    rec = _build_record(expected[start + r], catalog=catalog, cand_titles=batch.cand_titles[r],
                                        target_title=batch.target_titles[r], template_id=batch.template_ids[r],
                                        hybrid=batch.hybrid, raw_text=texts[r], gates=gates[r], gen_token_ids=rows[r])
                    records.append(rec)
                    _count_diagnostics(counts, rec, eos_id=ids.eos, max_new_tokens=max_new)
                omegas.append(out.omega)
                zs.append(out.z)
                start += b
            if start != len(samples):
                raise EvalError(f"{split}: the loader gave {start} samples, expected {len(samples)} (no drop_last, D17)")
            omega, z = torch.cat(omegas), torch.cat(zs)
            gate_calls = router.gate_calls - calls0
            want_calls = n_batches if router.routing_mode == "dynamic" else 0
            if record_gates:
                gate_record = router.stop_recording(expected_ids)
                recording = False
                _check_gate_record(gate_record, split=split, expected_ids=expected_ids, omega=omega, z=z,
                                   gate_calls=gate_calls, routing_mode=router.routing_mode, num_experts=k)
            if gate_calls != want_calls:
                raise EvalError(f"{split}: Router.gate_calls grew by {gate_calls}, expected {want_calls} "
                                f"({router.routing_mode}, {n_batches} generate calls; M7 11.6.7)")
    except BaseException:
        if recording:
            router.discard_recording()                                           # M7 11.6.5
        raise
    finally:
        model.train(was_training)
    if counts["n_no_match"] + counts["n_multi_match"] + sum(r.valid for r in records) != len(records):
        raise EvalError(f"{split}: valid count != N - n_no_match - n_multi_match ({dict(counts)})")
    return GeneratedSplit(split=split, batch_size=int(batch_size), records=tuple(records),
                          omega=omega.numpy(), z=z.numpy(), n_batches=n_batches, gate_calls=gate_calls,
                          diagnostics={name: int(counts[name]) for name in DIAGNOSTICS}, gate_record=gate_record,
                          seconds=time.perf_counter() - started)


# ----------------------------------------------------------------------------
# Record comparison (15.6.10)
# ----------------------------------------------------------------------------
def _first_difference(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """First generation step where two token sequences differ (None when equal)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_records(a: Sequence[EvalRecord], b: Sequence[EvalRecord]) -> dict:
    """Sample-by-sample comparison of two evaluations of the same samples (15.6.10).

    Counts of equal ``gen_token_ids``, ``first_line``, judgement (``valid``, ``correct``), ``gate_*`` values (bit for
    bit) and whole records, the largest |omega difference|, and one entry per sample whose generation result
    differs (tokens, first line or judgement: sample_id, first differing step, both first lines, both judgements).
    Gate values that differ only in the last bits are counted, not listed. Different sample ids -> ``EvalError``.
    """
    a, b = tuple(a), tuple(b)
    ids_a, ids_b = [r.sample_id for r in a], [r.sample_id for r in b]
    if ids_a != ids_b:
        raise EvalError(f"records of different samples cannot be compared ({len(ids_a)} vs {len(ids_b)} records)")
    n = len(a)
    counts = {"records_equal": 0, "tokens_equal": 0, "first_line_equal": 0, "judgement_equal": 0, "gates_equal": 0}
    max_omega = 0.0
    differences = []
    names = [f.name for f in dataclasses.fields(EvalRecord)]
    for ra, rb in zip(a, b):
        same_tokens = tuple(ra.gen_token_ids) == tuple(rb.gen_token_ids)
        same_line = ra.first_line == rb.first_line
        judgement_a, judgement_b = [bool(ra.valid), bool(ra.correct)], [bool(rb.valid), bool(rb.correct)]
        same_record = all(getattr(ra, name) == getattr(rb, name) for name in names)
        if len(ra.gates) != len(rb.gates):
            raise EvalError(f"{ra.sample_id}: K differs ({len(ra.gates)} vs {len(rb.gates)} gates)")
        omega_diff = max((abs(float(x) - float(y)) for x, y in zip(ra.gates, rb.gates)), default=0.0)
        max_omega = max(max_omega, omega_diff)
        counts["records_equal"] += int(same_record)
        counts["tokens_equal"] += int(same_tokens)
        counts["first_line_equal"] += int(same_line)
        counts["judgement_equal"] += int(judgement_a == judgement_b)
        counts["gates_equal"] += int(tuple(ra.gates) == tuple(rb.gates))
        if not (same_tokens and same_line and judgement_a == judgement_b):
            differences.append({"sample_id": ra.sample_id,
                                "first_diff_step": _first_difference(ra.gen_token_ids, rb.gen_token_ids),
                                "first_line_a": ra.first_line, "first_line_b": rb.first_line,
                                "judgement_a": judgement_a, "judgement_b": judgement_b,
                                "tokens_equal": same_tokens, "max_abs_diff_omega": omega_diff})
    return {"n": n, **counts, "ratios": {key: (value / n if n else None) for key, value in counts.items()},
            "max_abs_diff_omega": max_omega, "differences": differences}


# ----------------------------------------------------------------------------
# Files (15.6.8, 15.6.9)
# ----------------------------------------------------------------------------
def _jsonable(obj: object) -> object:
    """Plain JSON values: dataclasses, mappings, sequences, numpy values, paths."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, np.bool_):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return str(obj)


def _write_json(path: Path, obj: object) -> str:
    """UTF-8 JSON with LF, fixed key order, temporary name + ``os.replace``; never overwrites; returns sha256."""
    path = Path(path)
    if path.exists():
        raise EvalError(f"{path} exists; evaluation files are never overwritten (4.6.3)")
    data = _jsonable(obj)
    hits = run.find_secrets(data)
    if hits:
        raise run.SecretLeakError(f"refusing to write token-like values at {hits} to {path}")
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    os.replace(tmp, path)
    return run.content_sha256(path)


def _expected_size(data: LoadedData, split: str) -> int:
    """15.6.2 step 1: full -> val 94 / test 95 (D17), smoke -> the chosen user count of that split (D14)."""
    if data.subset == "full":
        return FULL_SPLIT_SIZE[split]
    if data.subset == "smoke" and data.smoke_users is not None:
        return int(data.smoke_users[SMOKE_SPLIT_INDEX[split]])
    raise EvalError(f"data.subset {data.subset!r} with smoke_users {data.smoke_users!r} has no expected size")


def _prepare_out_dir(run_dir: Path, ckpt_tag: str, split: str) -> Path:
    """15.6.8 step 1: a completed folder (``metrics.json``) is an error; an incomplete one is kept as
    ``{split}.incomplete-<UTC>`` (M10 14.6.10 rule); then a new empty folder."""
    out = eval_dir(run_dir, ckpt_tag, split)
    if out.exists():
        if (out / "metrics.json").exists():
            raise EvalError(f"{out} holds a completed evaluation (metrics.json); it is never overwritten (15.6.8)")
        stamp = run.utc_stamp()
        moved, n = out.with_name(f"{split}.incomplete-{stamp}"), 1
        while moved.exists():
            moved, n = out.with_name(f"{split}.incomplete-{stamp}-{n}"), n + 1
        out.rename(moved)
        logger.warning("incomplete evaluation folder %s kept as %s", out, moved.name)
    out.mkdir(parents=True)
    return out


def _run_meta(run_dir: Path) -> dict:
    """The M0 ``meta.json`` of the run folder (empty when absent, e.g. a bare test folder)."""
    path = Path(run_dir) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _candidates_sha256(records: Sequence[EvalRecord]) -> str:
    """M2 6.6.5 serialization ``json.dumps([[row, list(cands)], ...], separators=(",", ":"))``, sha256."""
    rows = [[int(r.sample_id.rsplit(":", 1)[1]), list(r.cand_ids)] for r in records]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def _supplementary(gen: GeneratedSplit, metrics: SplitMetrics) -> dict:
    """``supplementary`` of ``metrics.json`` (15.6.7): Wilson 95%, flag counts and the flag-free metrics, the
    generation diagnostics. Not paper metrics; never used for selection or the main comparison columns."""
    records = gen.records
    n, v, c = metrics.n_total, metrics.n_valid, metrics.n_correct
    flag_free = [r for r in records if not any(r.flags.values())]
    return {
        "wilson95": {"z": WILSON_Z, "valid_ratio": wilson_interval(v, n), "hr_at_1": wilson_interval(c, v),
                     "combined_metric": wilson_interval(c, n)},
        "flags": {"counts": {name: sum(1 for r in records if r.flags[name]) for name in FLAG_NAMES},
                  "any": n - len(flag_free),
                  "flag_free": compute_metrics(flag_free, metrics.split) if flag_free else None},
        "generation": dict(gen.diagnostics),
    }


def _metrics_meta(model: Any, bundle: TokenizerBundle, data: LoadedData, cfg: "Config", run_dir: Path,
                  gen: GeneratedSplit, templates: TemplateSet, *, ckpt_meta: Optional[Mapping[str, object]],
                  times: Mapping[str, object], max_memory_gib: Optional[float]) -> dict:
    """``meta`` of ``metrics.json``: the M12 16.6.4 list (from M0 ``meta.json`` and ``Config``, [SPEC §3]) and the
    M11 evaluation, generation and input-fingerprint keys of 15.6.9 (``n_batches``, ``gate_calls`` increase)."""
    import transformers

    run_meta = _run_meta(run_dir)
    git = (run_meta.get("code") or {}).get("git") or {}
    llm_path = run_meta.get("llm_path")
    dev, mc, ids = model.device, cfg.model, bundle.ids
    counts = Counter(r.template_id for r in gen.records)
    return {
        "git_commit": git.get("commit"), "git_dirty": git.get("dirty"), "seed": cfg.run.seed, "dataset": "movielens",
        "routing_mode": cfg.router.routing_mode, "num_experts": cfg.lora.num_experts, "lora_r": cfg.lora.r,
        "epochs": cfg.train.epochs, "lr_peak": cfg.train.lr_peak, "micro_batch_size": cfg.train.micro_batch_size,
        "grad_accum": cfg.train.grad_accum, "num_candidates": cfg.candidates.num_candidates,
        "backbone": mc.backbone, "llm_path_basename": Path(llm_path).name if llm_path else None,
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__, "peft": "not_used",
        # evaluation (15.6.9); torch_num_threads: CPU records are bitwise equal only at the same count (15.6.10)
        "torch_num_threads": torch.get_num_threads(),
        "ckpt_meta": dict(ckpt_meta) if ckpt_meta is not None else None, "data_subset": data.subset,
        "smoke_users": list(data.smoke_users) if data.smoke_users is not None else None,
        "title_mode": data.title_mode, "eval_prompt": cfg.prompt.eval_prompt,
        "template_choice": cfg.prompt.template_choice, "eval_batch_size": cfg.eval.batch_size,
        "n_batches": gen.n_batches, "num_workers": cfg.batching.num_workers, "precision": cfg.train.precision,
        "autocast_dtype": AUTOCAST_DTYPE[cfg.train.precision], "device": str(dev), **dict(times),
        "max_memory_allocated_gib": max_memory_gib,
        # generation and decoding ([코드] U model/model_interface.py:70-90, M9 model.gen_*)
        "gen_strategy": mc.gen_strategy, "max_new_tokens": mc.gen_max_new_tokens,
        "min_new_tokens": mc.gen_min_new_tokens, "repetition_penalty": mc.gen_repetition_penalty,
        "pad_token_id": ids.pad, "eos_token_id": ids.eos, "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False, "first_line_rule": FIRST_LINE_RULE,
        # input fingerprints
        "tokenizer_sha256": bundle.model_sha256, "candidates_sha256": _candidates_sha256(gen.records),
        "template_counts": [counts.get(k, 0) for k in range(len(templates.templates))],
        "gate_calls": gen.gate_calls, "sasrec_state_digest": model.sasrec.state_digest,
    }


def _npz_meta(model: Any, cfg: "Config", ckpt_tag: str, ckpt_meta: Optional[Mapping[str, object]]) -> dict:
    """``meta_json`` of ``representations.npz`` (15.6.8 step 2). The UTC time is left out so that the same
    checkpoint and settings give the same file (15.6.10, 15.8 #11); it is in ``metrics.json`` meta."""
    return {"source": "evaluate", "routing_mode": cfg.router.routing_mode, "num_experts": cfg.lora.num_experts,
            "z_dim": int(model.sasrec.hidden_size), "ckpt_tag": ckpt_tag,
            "ckpt_meta": dict(ckpt_meta) if ckpt_meta is not None else None,
            "sasrec_state_digest": model.sasrec.state_digest, "device": str(model.device),
            "eval_batch_size": cfg.eval.batch_size, "torch_version": torch.__version__}


def _batch_invariance(main: GeneratedSplit, single: GeneratedSplit, split: str) -> dict:
    """``batch_invariance.json`` (15.6.10): the main batch size against batch 1, counts and ratios of equal
    tokens, first lines and judgements, the differing samples, batch-1 metrics and max |omega|, |z| differences."""
    comparison = compare_records(main.records, single.records)
    return {"split": split, "batch_sizes": [main.batch_size, single.batch_size], "n": comparison["n"],
            **{key: comparison[key] for key in ("records_equal", "tokens_equal", "first_line_equal",
                                               "judgement_equal", "gates_equal", "ratios", "differences")},
            "metrics_batch1": compute_metrics(single.records, split),
            "aux_protocols_batch1": aux_protocol_metrics(single.records, split),
            "max_abs_diff_omega": float(np.abs(main.omega.astype(np.float64) - single.omega.astype(np.float64)).max()),
            "max_abs_diff_z": float(np.abs(main.z.astype(np.float64) - single.z.astype(np.float64)).max()),
            "n_batches_batch1": single.n_batches, "gate_calls_batch1": single.gate_calls,
            "seconds_batch1": single.seconds}


def evaluate_split(model: Any, bundle: TokenizerBundle, data: LoadedData, split: str, cfg: "Config", run_dir: Path,
                   *, ckpt_tag: str, ckpt_meta: Optional[Mapping[str, object]] = None,
                   templates: Optional[TemplateSet] = None) -> EvalResult:
    """Evaluate the whole ``split`` and write ``eval/{ckpt_tag}/{split}/`` once at the end (15.6.2, 15.6.8).

    1. checks: split, ``ckpt_tag``, N = 94/95 for full data (D17) or the smoke user count (D14),
       ``cudnn.benchmark`` off; the folder is prepared first (completed -> ``EvalError``, incomplete -> kept as
       ``{split}.incomplete-<UTC>``); 2. ``generate_records`` with ``eval.batch_size`` and the recorder;
    3. D17 metrics, auxiliary protocols, supplementary statistics; with ``eval.batch_invariance_check`` a second
    pass with batch 1 (recorder off, main results unchanged); 4. ``representations.npz`` (M12 ``save_npz``),
    ``batch_invariance.json``, then M12 ``write_eval`` (``records.jsonl``, ``metrics.json`` last). M10 calls it as
    ``evaluate_split(model, bundle, data, "val", cfg, run_dir, ckpt_tag=f"epoch{e}", templates=templates)``.
    """
    from .records import REPRESENTATIONS_FILE, RepresentationTable, save_npz, write_eval

    started_utc, t0 = run.utc_now(), time.perf_counter()
    split, ckpt_tag = _check_split(split), _check_tag(ckpt_tag)
    samples = data.samples(split)
    want = _expected_size(data, split)
    if len(samples) != want:
        raise EvalError(f"{split} has {len(samples)} samples, expected {want} for data.subset {data.subset} "
                        f"(D17: the whole split, D14 smoke)")
    _check_cudnn()
    if ckpt_meta is not None and not isinstance(ckpt_meta, Mapping):
        raise EvalError(f"ckpt_meta must be a mapping or None, got {type(ckpt_meta).__name__}")
    if templates is None:
        templates = load_templates(run.resolve_path(cfg.prompt.template_path))      # M4 8.6.2, REPO_ROOT relative
    dev = model.device
    out_dir = _prepare_out_dir(run_dir, ckpt_tag, split)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)        # evaluation peak apart from the M10 step peaks (15.6.9)
    logger.info("evaluate %s %s: N %d, batch size %d, %d batches, prompt %s", ckpt_tag, split, len(samples),
                cfg.eval.batch_size, math.ceil(len(samples) / cfg.eval.batch_size), cfg.prompt.eval_prompt)
    gen = generate_records(model, bundle, data, split, cfg, batch_size=cfg.eval.batch_size, templates=templates,
                           record_gates=True)
    metrics = compute_metrics(gen.records, split)
    aux = aux_protocol_metrics(gen.records, split)
    supplementary = _supplementary(gen, metrics)
    invariance, invariance_s = None, None
    if cfg.eval.batch_invariance_check:
        single = generate_records(model, bundle, data, split, cfg, batch_size=1, templates=templates,
                                  record_gates=False)
        invariance, invariance_s = _batch_invariance(gen, single, split), single.seconds
    max_memory = None
    if dev.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated(dev) / 2**30
        torch.cuda.reset_peak_memory_stats(dev)
    seconds = time.perf_counter() - t0
    times = {"started_utc": started_utc.isoformat(), "ended_utc": run.utc_now().isoformat(), "seconds": seconds,
             "generation_seconds": gen.seconds, "batch_invariance_seconds": invariance_s}
    meta = _metrics_meta(model, bundle, data, cfg, run_dir, gen, templates, ckpt_meta=ckpt_meta, times=times,
                         max_memory_gib=max_memory)
    expected_ids = [s.sample_id for s in samples]
    table = RepresentationTable(split=split, sample_ids=tuple(expected_ids),
                                rows=np.array([s.row for s in samples], dtype=np.int64),
                                user_ids=np.array([s.user_id for s in samples], dtype=np.int64), z=gen.z,
                                omega=gen.omega, argmax=np.argmax(gen.omega, axis=1).astype(np.int64),
                                meta=_npz_meta(model, cfg, ckpt_tag, ckpt_meta))
    save_npz(out_dir / REPRESENTATIONS_FILE, table)
    if invariance is not None:
        _write_json(out_dir / BATCH_INVARIANCE_FILE, invariance)
    metrics_json = {**dataclasses.asdict(metrics), "epoch": _tag_epoch(ckpt_tag), "ckpt_tag": ckpt_tag,
                    "aux_protocols": aux, "supplementary": supplementary, "meta": meta}
    write_eval(out_dir, gen.records, metrics_json, expected_ids=expected_ids, num_experts=cfg.lora.num_experts,
               rowsum_atol=cfg.records.gate_rowsum_atol)
    logger.info("evaluated %s %s: valid %d, correct %d of %d; ValidRatio %.4f, HR@1 %.4f, CombinedMetric %.4f; "
                "aux %s; n_no_eos %d, n_blank_leading_segment %d; %.1f s", ckpt_tag, split, metrics.n_valid,
                metrics.n_correct, metrics.n_total, metrics.valid_ratio, metrics.hr_at_1, metrics.combined_metric,
                {name: (m.n_total, m.n_valid, m.n_correct) for name, m in aux.items()},
                gen.diagnostics["n_no_eos"], gen.diagnostics["n_blank_leading_segment"], seconds)
    return EvalResult(split=split, ckpt_tag=ckpt_tag, metrics=metrics, aux=aux, records=gen.records,
                      omega=gen.omega, z=gen.z, out_dir=out_dir, seconds=seconds)


# ----------------------------------------------------------------------------
# Reading a completed evaluation (15.6.11, M10 resume 14.6.7)
# ----------------------------------------------------------------------------
def load_eval_result(out_dir: Path, *, num_experts: int) -> EvalResult:
    """``EvalResult`` of a completed ``eval/{ckpt_tag}/{split}/`` folder (``metrics.json`` present), checked.

    Records (M12 ``read_eval_records``) and the NPZ (``load_npz``) must hold the same samples in the same order
    with ``gate_*`` = NPZ omega bit for bit; every record is judged again (``first_line``, ``judge``, the target
    equivalence) and the metrics and auxiliary protocols recomputed from the records must equal ``metrics.json``.
    A removed line or a changed ``valid`` fails. Used by M10 on resume instead of generating again.
    """
    from .records import (METRICS_FILE, RECORDS_FILE, REPRESENTATIONS_FILE, SPLIT_METRICS_FIELDS, load_npz,
                          read_eval_records)

    out_dir = Path(out_dir)
    split, ckpt_tag = _check_split(out_dir.name), _check_tag(out_dir.parent.name)
    if not _is_int(num_experts) or int(num_experts) < 1:
        raise EvalError(f"num_experts must be an int >= 1, got {num_experts!r}")
    if not (out_dir / METRICS_FILE).is_file():
        raise EvalError(f"{out_dir}: no metrics.json, the evaluation is not complete (15.6.8)")
    metrics_json = json.loads((out_dir / METRICS_FILE).read_text(encoding="utf-8"))
    records = read_eval_records(out_dir / RECORDS_FILE)
    table = load_npz(out_dir / REPRESENTATIONS_FILE)
    where = str(out_dir)
    if metrics_json.get("split") != split or table.split != split:
        raise EvalError(f"{where}: split {metrics_json.get('split')!r} / NPZ {table.split!r} != folder {split!r}")
    if metrics_json.get("ckpt_tag", ckpt_tag) != ckpt_tag or metrics_json.get("epoch") != _tag_epoch(ckpt_tag):
        raise EvalError(f"{where}: ckpt_tag/epoch {metrics_json.get('ckpt_tag')!r}/{metrics_json.get('epoch')!r} "
                        f"do not match the folder {ckpt_tag!r}")
    if table.omega is None or table.omega.shape[1] != int(num_experts):
        raise EvalError(f"{where}: NPZ omega has K = {None if table.omega is None else table.omega.shape[1]}, "
                        f"expected {num_experts}")
    if tuple(r.sample_id for r in records) != table.sample_ids:
        raise EvalError(f"{where}: records.jsonl holds {len(records)} samples that differ from the {len(table)} "
                        "NPZ samples (a line removed or reordered)")
    for i, r in enumerate(records):
        if r.split != split or len(r.gates) != int(num_experts):
            raise EvalError(f"{where}: {r.sample_id} has split {r.split!r} and {len(r.gates)} gates")
        verdict = judge(r.first_line, r.cand_ids, r.cand_titles, r.target_title)
        if r.first_line != first_line(r.raw_text) or (verdict.matched, verdict.valid, verdict.correct) != \
                (tuple(r.matched), bool(r.valid), bool(r.correct)):
            raise EvalError(f"{where}: {r.sample_id}: first line or judgement differs from the recorded "
                            f"(matched, valid, correct) = {(tuple(r.matched), r.valid, r.correct)}")
        if r.target_id not in r.cand_ids or (verdict.valid and verdict.correct != (verdict.matched[0] == r.target_id)):
            raise EvalError(f"{where}: {r.sample_id}: target {r.target_id} breaks the candidate/judgement rule")
        gates = np.asarray([float(g) for g in r.gates], dtype=np.float64)
        if not np.array_equal(gates.astype(np.float32).astype(np.float64), gates) or \
                not np.array_equal(gates.astype(np.float32).view(np.uint32), table.omega[i].view(np.uint32)):
            raise EvalError(f"{where}: {r.sample_id}: gate values differ from the NPZ omega bit for bit")
    metrics = compute_metrics(records, split)
    aux = aux_protocol_metrics(records, split)
    stored = {name: metrics_json.get(name) for name in SPLIT_METRICS_FIELDS}
    if stored != dataclasses.asdict(metrics):
        raise EvalError(f"{where}: metrics.json {stored} != recomputed {dataclasses.asdict(metrics)}")
    stored_aux = metrics_json.get("aux_protocols", {})
    if stored_aux != {name: dataclasses.asdict(m) for name, m in aux.items()}:
        raise EvalError(f"{where}: aux_protocols differ from the values recomputed from the records")
    seconds = (metrics_json.get("meta") or {}).get("seconds")
    return EvalResult(split=split, ckpt_tag=ckpt_tag, metrics=metrics, aux=aux, records=tuple(records),
                      omega=table.omega, z=table.z, out_dir=out_dir,
                      seconds=float(seconds) if isinstance(seconds, (int, float)) else 0.0)


# ----------------------------------------------------------------------------
# main.py eval (15.6.11) [결정]
# ----------------------------------------------------------------------------
def eval_config_differences(source: "Config", current: "Config") -> tuple[list[str], list[str]]:
    """``(forbidden, allowed)`` keys ``section.key: a != b`` that differ between the training run and this config.

    Allowed (15.6.11 step 2): the ``eval`` and ``records`` sections, ``prompt.eval_prompt`` (E3),
    ``batching.num_workers``, ``data.subset``, ``data.smoke_users``, ``data.train_rows``. Everything else,
    ``run.seed`` included, must be equal.
    """
    from .config import config_to_dict

    a, b = config_to_dict(source), config_to_dict(current)
    forbidden, allowed = [], []
    for section in a:
        for key in sorted(set(a[section]) | set(b.get(section, {}))):
            x, y = a[section].get(key), b.get(section, {}).get(key)
            if x != y:
                name = f"{section}.{key}"
                text = f"{name}: {x!r} != {y!r}"
                (allowed if section in EVAL_ALLOWED_SECTIONS or name in EVAL_ALLOWED_KEYS else forbidden).append(text)
    return forbidden, allowed


def _input_hash_problems(source_meta: Mapping[str, Any], *, cfg: "Config", repo_root: Path,
                         llm_path: Optional[Path], tokenizer_dir: Optional[Path]) -> list[str]:
    """15.6.11 step 2: the training run's input hashes (data, original SASRec, prompt: ``inputs``; converted
    SASRec and LLaMA ``config.json``: ``assets``, M0 4.6.3) must equal the current ones."""
    from .commands import asset_record

    problems = []
    source_inputs = source_meta.get("inputs")
    for path, sha in run.input_hashes(repo_root).items():
        got = source_inputs.get(path) if isinstance(source_inputs, Mapping) else None
        if got != sha:
            problems.append(f"inputs[{path}]: training run {got} != current {sha}")
    source_assets = source_meta.get("assets") if isinstance(source_meta.get("assets"), Mapping) else {}
    current = asset_record(cfg, repo_root=repo_root, llm_path=llm_path, tokenizer_dir=tokenizer_dir)
    for name, fields in (("sasrec_state", ("sha256",)), ("llama_config", ("sha256", "torch_dtype"))):
        a, b = source_assets.get(name) or {}, current.get(name) or {}
        for field in fields:
            if a.get(field) != b.get(field):
                problems.append(f"assets.{name}.{field}: training run {a.get(field)} != current {b.get(field)}")
    return problems


def resolve_eval_checkpoints(source: Path, tags: Sequence[str]) -> list[tuple[str, Optional[int], tuple[str, ...]]]:
    """``eval.checkpoints`` -> ``[(ckpt_tag, epoch, tags)]`` (15.6.11 step 3): ``best_val``/``last`` through the
    training run's ``eval/selection.json`` (``best_val_epoch``/``last_epoch``, M10 14.6.7), ``epoch<k>`` as given,
    ``init`` (no checkpoint, the seed-built model) first, then epochs ascending, each epoch once; every epoch
    needs its ``COMPLETE`` marker."""
    source = Path(source)
    selection: Optional[dict] = None
    epochs: dict[int, list[str]] = {}
    init = []
    for tag in tags:
        if tag == "init":
            init.append(tag)
            continue
        if tag in _SELECTION_KEYS:
            if selection is None:
                path = source / "eval" / "selection.json"
                if not path.is_file():
                    raise EvalError(f"{path} not found: best_val/last need a finished training run (M10 14.6.7)")
                selection = json.loads(path.read_text(encoding="utf-8"))
            epoch = selection.get(_SELECTION_KEYS[tag])
            if not _is_int(epoch) or int(epoch) < 0:
                raise EvalError(f"selection.json {_SELECTION_KEYS[tag]} = {epoch!r} is not an epoch number")
        else:
            match = re.fullmatch(r"epoch(0|[1-9][0-9]*)", str(tag))
            if not match:
                raise EvalError(f"eval.checkpoints has {tag!r}; allowed best_val, last, epoch<k>, init")
            epoch = int(match.group(1))
        epochs.setdefault(int(epoch), []).append(tag)
    out: list[tuple[str, Optional[int], tuple[str, ...]]] = [("init", None, tuple(init))] if init else []
    for epoch in sorted(epochs):
        ckpt = source / "checkpoints" / f"epoch{epoch}"
        if not (ckpt / "COMPLETE").is_file():
            raise EvalError(f"{ckpt}: no COMPLETE marker; only completed checkpoints are evaluated (15.6.11 step 3)")
        out.append((f"epoch{epoch}", epoch, tuple(epochs[epoch])))
    return out


def _cluster_hard_gate(cfg: "Config", data: LoadedData, repo_root: Path) -> tuple[Any, Any]:
    """cluster_hard only: M15 ``load_cluster_artifacts`` -> ``parse_expert_mapping`` -> ``ClusterHardGate`` in the
    order of M10 14.6.4 step 0.2 (19.6.7); ``expected_ids`` = every train, validation and test sample id."""
    from .cluster import ClusterHardGate, load_cluster_artifacts, parse_expert_mapping
    from .data import SPLITS

    expected_ids = [s.sample_id for split in SPLITS for s in data.samples(split)]
    state = run.resolve_path(cfg.sasrec.state_path, repo_root)
    artifacts = load_cluster_artifacts(run.resolve_path(cfg.cluster.artifact_dir, repo_root),
                                       expected_ids=expected_ids, sasrec_sha256=run.content_sha256(state),
                                       data_sha256=run.input_hashes(repo_root), ccfg=cfg.cluster)
    mapping = parse_expert_mapping(cfg.cluster.expert_mapping, n_clusters=artifacts.n_clusters,
                                   num_experts=cfg.lora.num_experts)
    return artifacts, ClusterHardGate(artifacts, mapping, cfg.lora.num_experts)


def _reproduction(source: Path, ckpt_tag: str, split: str, result: EvalResult, source_cfg: "Config",
                  cfg: "Config") -> dict:
    """Reproduction check of one (ckpt_tag, split) against the training run's records (15.6.10) ``[결정]``.

    ``absent``: the training run has no completed folder; ``not_comparable``: an allowed difference that can
    change the result (``REPRODUCTION_KEYS``) differs, with the keys; otherwise ``identical`` when the two
    ``records.jsonl`` sha256 are equal, else ``mismatch`` with ``compare_records`` (samples, first step).
    """
    from .config import config_to_dict
    from .records import METRICS_FILE, RECORDS_FILE, read_eval_records

    src = eval_dir(source, ckpt_tag, split)
    if not (src / METRICS_FILE).is_file():
        return {"status": "absent", "source_dir": str(src)}
    a, b = config_to_dict(source_cfg), config_to_dict(cfg)
    differing = [key for key in REPRODUCTION_KEYS
                 if a[key.split(".")[0]][key.split(".")[1]] != b[key.split(".")[0]][key.split(".")[1]]]
    if differing:
        return {"status": "not_comparable", "differing_keys": differing, "source_dir": str(src)}
    sha_source = run.content_sha256(src / RECORDS_FILE)
    sha_eval = run.content_sha256(result.out_dir / RECORDS_FILE)
    try:
        comparison: dict = compare_records(read_eval_records(src / RECORDS_FILE), result.records)
    except EvalError as exc:
        comparison = {"error": str(exc)}
    status = "identical" if sha_source == sha_eval else "mismatch"
    if status == "mismatch":
        logger.warning("reproduction mismatch for %s %s: %s", ckpt_tag, split,
                       [(d["sample_id"], d["first_diff_step"]) for d in comparison.get("differences", [])][:10])
    return {"status": status, "source_dir": str(src), "records_sha256": {"source": sha_source, "eval": sha_eval},
            "comparison": comparison}


def run_eval_command(cfg: "Config", run_dir: Path, *, repo_root: Path, llm_path: Optional[Path]) -> dict:
    """``main.py eval --set eval.source_run=<training run>`` (15.6.11), inside the M0 eval run folder.

    M0 ``run_command`` has made the run folder, checked the environment and run ``candidates.self_check()``.
    1. the training run's ``config.yaml`` may differ only in the allowed keys (``eval_config_differences``,
       ``EvalError`` listing the others) and its ``meta.json`` input hashes must equal the current ones;
    2. ``eval.checkpoints`` -> epochs (``resolve_eval_checkpoints``); test only with ``best_val``/``last`` (D11,
       ``ConfigError``); 3. data, templates, tokenizer, then the model once (cluster_hard: the M15 order
       artifacts -> mapping -> ``ClusterHardGate`` -> ``build_model`` -> ``verify_live_encoder``; dynamic and none
       never call M15 [SPEC §22]); 4. ``init`` first with the seed-built model, then each epoch ascending through
       M10 ``load_model_state`` (the only checkpoint reader, 14.6.6) and ``evaluate_split`` per split;
    5. ``eval/summary.json``: per (ckpt_tag, split) metrics, auxiliary protocols, file sha256 and the
       reproduction check. Returns the summary.
    """
    from .batching import load_tokenizer
    from .commands import tokenizer_dir_for
    from .config import ConfigError, load_config
    from .data import load_dataset
    from .model import build_model
    from .records import METRICS_FILE, RECORDS_FILE, REPRESENTATIONS_FILE

    run_dir, repo_root = Path(run_dir), Path(repo_root)
    ec = cfg.eval
    if ec.source_run is None:
        raise EvalError("eval needs --set eval.source_run=<training run folder> (15.6.12)")
    if "test" in ec.splits and any(tag not in _SELECTION_KEYS for tag in ec.checkpoints):
        raise ConfigError(f"with test in eval.splits only best_val/last are allowed (D11), got {list(ec.checkpoints)}")
    source = run.resolve_path(ec.source_run, repo_root)
    if not (source / "config.yaml").is_file():
        raise EvalError(f"{source}: no config.yaml, not a training run folder")
    source_cfg, _ = load_config(source / "config.yaml")
    forbidden, allowed = eval_config_differences(source_cfg, cfg)
    if forbidden:
        raise EvalError("the eval config differs from the training run outside the allowed keys (15.6.11): "
                        + "; ".join(forbidden))
    tokenizer_dir = tokenizer_dir_for(cfg, llm_path)
    if tokenizer_dir is None:
        raise EvalError("tokenizer folder not available (llama2_7b: LLM_PATH, tiny_random: ILORA_TEST_TOKENIZER)")
    source_meta = run.read_meta(source) if (source / "meta.json").is_file() else {}
    problems = _input_hash_problems(source_meta, cfg=cfg, repo_root=repo_root, llm_path=llm_path,
                                    tokenizer_dir=tokenizer_dir)
    if problems:
        raise EvalError(f"input hashes of {source} differ from the current inputs: " + "; ".join(problems))
    checkpoints = resolve_eval_checkpoints(source, ec.checkpoints)
    logger.info("eval of %s: checkpoints %s, splits %s, allowed differences %s", source,
                [(tag, list(names)) for tag, _, names in checkpoints], list(ec.splits), allowed)
    data = load_dataset(cfg, repo_root)
    templates = load_templates(run.resolve_path(cfg.prompt.template_path, repo_root))
    bundle = load_tokenizer(tokenizer_dir, cfg.batching.tokenizer_sha256)
    artifacts, hard_gate, live = None, None, None
    if cfg.router.routing_mode == "cluster_hard":
        artifacts, hard_gate = _cluster_hard_gate(cfg, data, repo_root)
    model = build_model(cfg, bundle, data.catalog, llm_path=llm_path, hard_gate=hard_gate)
    if artifacts is not None:
        from .cluster import verify_live_encoder

        live = verify_live_encoder(artifacts, model.sasrec, data.train, device=model.device,
                                   n=cfg.cluster.live_check_n, atol=cfg.cluster.live_check_atol)
    listed, results = [], {}
    for ckpt_tag, epoch, names in checkpoints:
        ckpt_dir, trainable_sha256 = None, None
        if epoch is not None:
            from .train import load_model_state       # M10 14.6.6: markers, sha256, generated_frozen.* bit check

            ckpt_dir = source / "checkpoints" / ckpt_tag
            trainable_sha256 = load_model_state(ckpt_dir, model)
        ckpt_meta = {"source_run": str(source), "checkpoint_dir": str(ckpt_dir) if ckpt_dir is not None else None,
                     "epoch": epoch, "tags": list(names), "trainable_sha256": trainable_sha256}
        listed.append({"ckpt_tag": ckpt_tag, **ckpt_meta})
        for split in ec.splits:
            result = evaluate_split(model, bundle, data, split, cfg, run_dir, ckpt_tag=ckpt_tag, ckpt_meta=ckpt_meta,
                                    templates=templates)
            files = {name: run.content_sha256(result.out_dir / name)
                     for name in (RECORDS_FILE, METRICS_FILE, REPRESENTATIONS_FILE, BATCH_INVARIANCE_FILE)
                     if (result.out_dir / name).is_file()}
            reproduction = _reproduction(source, ckpt_tag, split, result, source_cfg, cfg)
            results[f"{ckpt_tag}/{split}"] = {"ckpt_tag": ckpt_tag, "split": split, "tags": list(names),
                                              "metrics": result.metrics, "aux_protocols": result.aux,
                                              "files": files, "reproduction": reproduction}
            logger.info("eval %s %s: reproduction %s", ckpt_tag, split, reproduction["status"])
    summary = {"source_run": str(source), "routing_mode": cfg.router.routing_mode, "splits": list(ec.splits),
               "allowed_differences": allowed, "reproduction_keys": list(REPRODUCTION_KEYS),
               "checkpoints": listed, "cluster_live_check": live, "results": results,
               "created_utc": run.utc_now().isoformat()}
    _write_json(run_dir / "eval" / SUMMARY_FILE, summary)
    return _jsonable(summary)


__all__ = [
    "EvalError", "EVAL_SPLITS", "FULL_SPLIT_SIZE", "AUX_DROP_LAST_BATCH_SIZES", "WILSON_Z", "CKPT_TAG_RE",
    "FIRST_LINE_RULE", "BATCH_INVARIANCE_FILE", "SUMMARY_FILE", "EVAL_ALLOWED_SECTIONS", "EVAL_ALLOWED_KEYS",
    "REPRODUCTION_KEYS", "DIAGNOSTICS", "Judgement", "EvalRecord", "SplitMetrics", "EvalResult", "GeneratedSplit",
    "first_line", "judge", "trim_generated", "decode_generated", "compute_metrics", "aux_protocol_metrics",
    "wilson_interval", "eval_dir", "check_batch", "check_output", "generate_records", "evaluate_split",
    "compare_records", "load_eval_result", "eval_config_differences", "resolve_eval_checkpoints", "run_eval_command",
]
