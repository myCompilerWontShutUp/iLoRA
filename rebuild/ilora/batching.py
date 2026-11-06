"""M5 tokenization과 batch 구성 (DESIGN.md §9, §3.1 ``TokenIds``·``TokenizerBundle``·``TrainBatch``·``EvalBatch``·loader).

Tokenizer (§9.6.2, D13): the slow ``LlamaTokenizer`` of transformers 4.28.0 with
sentencepiece 0.1.97, ``tokenizer.model`` checked by sha256, then ``[PAD]`` and the
four extra tokens added in the released order (ids 32000-32004, length 32,005) and
two fingerprints of the 4.28 segment tokenization checked. Token ids are used only
through ``TokenIds``.

Encoding (§9.6.3-§9.6.4, D18): prompt and answer are tokenized separately with
``add_special_tokens=False`` (special tokens split the text into segments, each
segment gets the SentencePiece dummy prefix, and str-registered special tokens
absorb the spaces around them, ``T4.28 tokenization_utils.py:516-547``).

- training ``answer_only`` (paper_spec): ``[BOS] + p + a``, labels ``-100`` x (1 + len(p)) + a
- training ``released_pair`` (``variant_released_label``): ``[BOS] + p + [BOS] + a``, the
  second ``[BOS]`` also a label (what the 4.28.0 pair encoding and the token_type_ids mask
  of ``U model/model_interface.py:38-42`` produce)
- evaluation: ``[BOS] + p``

with ``a`` = target title + ``"\\n"``. No truncation (``U data/data_interface.py:79``):
a sample longer than ``batching.max_seq_len`` is an error. Every sample passes the
placeholder count check (§9.6.5) before padding.

Batches (§9.6.6, D22): training right padding, evaluation left padding, all tensors
CPU ``int64``; ``hist_ids``/``len_seq`` come from the shared ``data.pad_histories``.

Loaders (§9.6.7-§9.6.8): one dataset item is one whole batch (``batch_size=None``),
the training items follow the M4 ``BatchPlan`` made in the main process, and
candidates, templates and curriculum draws are keyed (M2, M4), so the batches do not
depend on the number of workers. Datasets and the ``CandidateFn`` pickle for Windows
spawn workers (``LlamaTokenizer`` reloads its SentencePiece model,
``T4.28 models/llama/tokenization_llama.py:98-106``).
"""
from __future__ import annotations

import dataclasses
import functools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterator, Optional, Protocol, Sequence, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer

from . import run
from .candidates import NUM_CANDIDATES, candidates_for
from .data import HISTORY_LEN, PAD_ID, ItemCatalog, LoadedData, Sample, pad_histories
from .prompts import (BatchPlan, TemplateSet, build_prompt, check_catalog_titles, choose_template,
                      eval_is_hybrid)
from .sections import BatchingConfig, PromptConfig

if TYPE_CHECKING:
    from .sections import Config


class BatchingError(ValueError):
    """Tokenizer, encoding, count, length or loader problem (§9.6.10). Raised at once, never repaired."""


# ----------------------------------------------------------------------------
# Constants (§9.6.1, §9.6.9: code constants, not config keys)
# ----------------------------------------------------------------------------
PAD_TOKEN: Final = "[PAD]"                                       # [코드] U model/model_interface.py:315
EXTRA_TOKENS: Final = ("[PH]", "[HistoryEmb]", "[CansEmb]", "[ItemEmb]")   # [코드] :317, order fixed
TERMINATOR: Final = "\n"                                          # [코드] U data/data_interface.py:21, :72
HIST_LEN: Final = HISTORY_LEN                                     # 10 (§3.1 hist_ids [B, 10])
NUM_CANDS: Final = NUM_CANDIDATES                                 # 20 (§3.1 cand_ids [B, 20])
PAD_ITEM: Final = PAD_ID                                          # 1682, [코드] U data/movielens_data.py:18
LABEL_MODES: Final = ("answer_only", "released_pair")             # D18 and variant_released_label
IGNORE_INDEX: Final = -100                                        # label of positions outside the answer
EVAL_SPLITS: Final = ("val", "test")

BASE_VOCAB: Final = 32000                                         # LLaMA SentencePiece pieces
VOCAB_SIZE: Final = BASE_VOCAB + 1 + len(EXTRA_TOKENS)            # 32,005 with [ItemEmb] kept (§9.6.2)
# Fingerprints of the 4.28 slow tokenization (§9.6.2 step 6, measured with the test tokenizer).
FINGERPRINTS: Final = (
    ("Star Wars\n", (7828, 9149, 13)),                                         # ▁Star ▁Wars <0x0A>
    ("watched Toy Story [HistoryEmb], Heat [HistoryEmb] in the previous",
     (20654, 29411, 13740, 32002, 1919, 940, 271, 32002, 297, 278, 3517)),       # ▁, after [HistoryEmb]
)


def _is_int(value: object) -> bool:
    """Python int, bool excluded."""
    return isinstance(value, int) and not isinstance(value, bool)


# ----------------------------------------------------------------------------
# Tokenizer (§9.6.2)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenIds:
    """Special token ids (§3.1). The only source of token ids for M9, M10, M11, M13."""
    bos: int
    eos: int
    unk: int
    newline: int                                                  # piece <0x0A>
    pad: int
    ph: int
    history_emb: int
    cans_emb: int
    item_emb: int
    base_vocab: int
    vocab_size: int
    new_token_ids: tuple[int, ...]                                # (pad, ph, history_emb, cans_emb, item_emb)


@dataclass(frozen=True)
class TokenizerBundle:
    """The loaded slow tokenizer, its ``TokenIds`` and the sha256 of ``tokenizer.model``."""
    tokenizer: LlamaTokenizer
    ids: TokenIds
    model_sha256: str


def load_tokenizer(tokenizer_dir: Path, expected_sha256: str) -> TokenizerBundle:
    """Load and check the tokenizer (§9.6.2 steps 1-6); every mismatch is a ``BatchingError``.

    ``tokenizer_dir`` is ``LLM_PATH`` for runs and ``ILORA_TEST_TOKENIZER`` for CPU tests
    (§9.2); ``expected_sha256`` is ``batching.tokenizer_sha256``.
    """
    import sentencepiece
    import transformers

    # 1. library pins: the tokenization depends on both (D13; sentencepiece does the split)
    want_tf, want_sp = run.REQUIRED_TRANSFORMERS, run.REQUIRED_LIBRARIES["sentencepiece"]
    if transformers.__version__ != want_tf:
        raise BatchingError(f"transformers {transformers.__version__} != {want_tf} (D13: 4.28.0 slow tokenizer)")
    if getattr(sentencepiece, "__version__", None) != want_sp:
        raise BatchingError(f"sentencepiece {getattr(sentencepiece, '__version__', None)} != {want_sp} "
                            f"(the slow tokenizer split is sp_model.encode)")
    # 2. tokenizer.model content
    tokenizer_dir = Path(tokenizer_dir)
    model_file = tokenizer_dir / "tokenizer.model"
    if not model_file.is_file():
        raise BatchingError(f"{model_file}: tokenizer.model not found")
    got = run.content_sha256(model_file)
    if got != expected_sha256:
        raise BatchingError(f"{model_file}: sha256 {got} != batching.tokenizer_sha256 {expected_sha256}")
    # 3. slow tokenizer, [코드] U model/model_interface.py:313
    tok = LlamaTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
    problems = []
    if type(tok) is not LlamaTokenizer or tok.is_fast:
        problems.append(f"not the slow LlamaTokenizer ({type(tok).__name__}, is_fast={tok.is_fast})")
    if tok.vocab_size != BASE_VOCAB:
        problems.append(f"base vocabulary {tok.vocab_size} != {BASE_VOCAB}")
    if (tok.bos_token_id, tok.eos_token_id, tok.unk_token_id) != (1, 2, 0):
        problems.append(f"bos/eos/unk ids {(tok.bos_token_id, tok.eos_token_id, tok.unk_token_id)} != (1, 2, 0)")
    if tok.add_bos_token is not True or tok.add_eos_token is not False:
        problems.append(f"add_bos_token/add_eos_token {(tok.add_bos_token, tok.add_eos_token)} != (True, False)")
    if tok.sp_model_kwargs != {}:
        problems.append(f"sp_model_kwargs {tok.sp_model_kwargs!r} != {{}} (no subword sampling)")
    if problems:
        raise BatchingError(f"{tokenizer_dir}: " + "; ".join(problems))
    # 4. special tokens in the released order; ids follow len(self) + i (T4.28 tokenization_utils.py:428)
    # [코드] U model/model_interface.py:315, :317 (L model/model_interface.py:203, :205); the pad = eos line
    # :314 before them is overwritten by :315 and has no effect (§9.4)
    added_pad = tok.add_special_tokens({"pad_token": PAD_TOKEN})
    added_extra = tok.add_special_tokens({"additional_special_tokens": list(EXTRA_TOKENS)})
    if (added_pad, added_extra) != (1, len(EXTRA_TOKENS)):
        raise BatchingError(f"{tokenizer_dir}: add_special_tokens added {(added_pad, added_extra)} tokens, "
                            f"expected (1, {len(EXTRA_TOKENS)})")
    # 5. TokenIds
    names = (PAD_TOKEN, *EXTRA_TOKENS)
    new_ids = tuple(tok.convert_tokens_to_ids(name) for name in names)
    ids = TokenIds(bos=tok.bos_token_id, eos=tok.eos_token_id, unk=tok.unk_token_id,
                   newline=tok.convert_tokens_to_ids("<0x0A>"), pad=new_ids[0], ph=new_ids[1],
                   history_emb=new_ids[2], cans_emb=new_ids[3], item_emb=new_ids[4],
                   base_vocab=tok.vocab_size, vocab_size=len(tok), new_token_ids=new_ids)
    bundle = TokenizerBundle(tokenizer=tok, ids=ids, model_sha256=got)
    if new_ids != tuple(range(BASE_VOCAB, VOCAB_SIZE)) or len(tok) != VOCAB_SIZE or tok.pad_token_id != ids.pad:
        problems.append(f"special ids {dict(zip(names, new_ids))}, length {len(tok)}, pad {tok.pad_token_id}; "
                        f"expected 32000-32004 and length {VOCAB_SIZE}")
    if ids.newline != 13:
        problems.append(f"newline piece <0x0A> has id {ids.newline}, expected 13")
    for name, tid in zip(names, new_ids):
        if encode_text(bundle, name) != [tid]:
            problems.append(f"{name} encodes to {encode_text(bundle, name)}, expected [{tid}]")
    # 6. fingerprints of the 4.28 behaviour (4.31+ with legacy false drops the ▁ after special tokens)
    for text, expected in FINGERPRINTS:
        if tuple(encode_text(bundle, text)) != expected:
            problems.append(f"fingerprint {text!r} -> {encode_text(bundle, text)}, expected {list(expected)}")
    if problems:
        raise BatchingError(f"{tokenizer_dir}: " + "; ".join(problems))
    return bundle


# ----------------------------------------------------------------------------
# Encoding (§9.6.3-§9.6.5)
# ----------------------------------------------------------------------------
def encode_text(bundle: TokenizerBundle, text: str) -> list[int]:
    """``tok(text, add_special_tokens=False)["input_ids"]`` (= ``convert_tokens_to_ids(tokenize(text))``,
    ``T4.28 tokenization_utils.py:614-617, :649``)."""
    if not isinstance(text, str):
        raise BatchingError(f"text must be str, got {type(text).__name__}")
    return list(bundle.tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_train(bundle: TokenizerBundle, prompt: str, answer: str,
                 label_mode: str) -> tuple[list[int], list[int]]:
    """(input_ids, labels) of one training sample (D18); ``answer`` is the target title without ``"\\n"``.

    ``p = encode_text(prompt)``, ``a = encode_text(answer + "\\n")``. ``answer_only``:
    ``[BOS] + p + a`` and labels ``-100`` x (1 + len(p)) + a. ``released_pair``:
    ``[BOS] + p + [BOS] + a`` and labels ``-100`` x (1 + len(p)) + [BOS] + a.
    """
    if label_mode not in LABEL_MODES:
        raise BatchingError(f"unknown batching.label_mode {label_mode!r}, expected one of {LABEL_MODES}")
    p = encode_text(bundle, prompt)
    a = encode_text(bundle, answer + TERMINATOR)          # [코드] U data/data_interface.py:72
    bos = bundle.ids.bos
    if label_mode == "answer_only":
        return [bos] + p + a, [IGNORE_INDEX] * (1 + len(p)) + a
    return [bos] + p + [bos] + a, [IGNORE_INDEX] * (1 + len(p)) + [bos] + a


def encode_eval(bundle: TokenizerBundle, prompt: str) -> list[int]:
    """``[BOS] + encode_text(prompt)`` (= ``add_special_tokens=True``, ``U data/data_interface.py:94-100``)."""
    return [bundle.ids.bos] + encode_text(bundle, prompt)


def check_token_counts(bundle: TokenizerBundle, input_ids: Sequence[int], *, hybrid: bool, len_seq: int,
                       prompt_len: int, n_bos: int, max_seq_len: int, where: str) -> None:
    """Placeholder and special-token counts of one unpadded sample (§9.6.5).

    hybrid: ``[HistoryEmb]`` = len_seq, ``[CansEmb]`` = 20, ``[PH]`` = 0; text-only:
    ``[PH]`` = len_seq + 20 and the other two 0. Always 0 ``[ItemEmb]``, ``[PAD]``, unk, eos;
    ``[BOS]`` at position 0 only (``n_bos`` 1: answer_only and evaluation) or at 0 and
    ``1 + prompt_len`` (``n_bos`` 2: released_pair). The answer part after the prompt,
    when there is one, ends with the newline id. ``len(input_ids) <= max_seq_len``
    (no truncation). M9 relies on this: the k-th ``[HistoryEmb]`` is ``hist_ids[b, k]`` and
    the k-th ``[CansEmb]`` is ``cand_ids[b, k]``.
    """
    ids = bundle.ids
    if not isinstance(hybrid, bool):
        raise BatchingError(f"{where}: hybrid must be bool, got {hybrid!r}")
    fmt = "hybrid" if hybrid else "text-only"
    head = f"{where} ({fmt})"
    if not _is_int(len_seq) or not 1 <= len_seq <= HIST_LEN:
        raise BatchingError(f"{head}: history length {len_seq!r} is outside 1..{HIST_LEN}")
    if n_bos not in (1, 2) or isinstance(n_bos, bool):
        raise BatchingError(f"{head}: n_bos must be 1 or 2, got {n_bos!r}")
    if not _is_int(max_seq_len) or max_seq_len < 1:
        raise BatchingError(f"{head}: max_seq_len must be an int >= 1, got {max_seq_len!r}")
    seq = list(input_ids)
    if not _is_int(prompt_len) or prompt_len < 0 or n_bos + prompt_len > len(seq):
        raise BatchingError(f"{head}: prompt_len {prompt_len!r} does not fit {len(seq)} tokens with {n_bos} BOS")
    counts = Counter(seq)
    if hybrid:
        expected = {"[HistoryEmb]": (ids.history_emb, len_seq), "[CansEmb]": (ids.cans_emb, NUM_CANDS),
                    "[PH]": (ids.ph, 0)}
    else:
        expected = {"[HistoryEmb]": (ids.history_emb, 0), "[CansEmb]": (ids.cans_emb, 0),
                    "[PH]": (ids.ph, len_seq + NUM_CANDS)}
    expected.update({"[ItemEmb]": (ids.item_emb, 0), "[PAD]": (ids.pad, 0), "<unk>": (ids.unk, 0),
                     "</s>": (ids.eos, 0), "<s>": (ids.bos, n_bos)})
    problems = [f"{name} x{counts[tid]} (expected {want})" for name, (tid, want) in expected.items()
                if counts[tid] != want]
    bos_at = [i for i, t in enumerate(seq) if t == ids.bos]
    want_bos = [0] if n_bos == 1 else [0, 1 + prompt_len]
    if bos_at != want_bos:
        problems.append(f"[BOS] at positions {bos_at[:5]} (expected {want_bos})")
    answer = seq[n_bos + prompt_len:]
    if answer and answer[-1] != ids.newline:
        problems.append(f"answer ends with id {answer[-1]}, not the newline id {ids.newline}")
    if len(seq) > max_seq_len:
        problems.append(f"{len(seq)} tokens > batching.max_seq_len {max_seq_len} (not truncated)")
    if problems:
        raise BatchingError(f"{head}: token check failed: " + "; ".join(problems))


def _prompt_len_from_labels(labels: Sequence[int], where: str) -> int:
    """len(p) of an ``encode_train`` result: the leading ``-100`` labels are ``[BOS] + p``."""
    for i, label in enumerate(labels):
        if label != IGNORE_INDEX:
            if i < 1:
                raise BatchingError(f"{where}: the first label must be {IGNORE_INDEX} ([BOS] of the prompt)")
            return i - 1
    raise BatchingError(f"{where}: no answer token has a label")


def _check_train_labels(bundle: TokenizerBundle, input_ids: Sequence[int], labels: Sequence[int], *,
                        label_mode: str, prompt_len: int, where: str) -> None:
    """§9.6.5 training part: label positions = len(a) (+1 for released_pair), equal to the input
    tokens there, all other positions -100, and the last label is the newline id."""
    n_bos = 2 if label_mode == "released_pair" else 1
    start = 1 + prompt_len
    problems = []
    if len(labels) != len(input_ids):
        problems.append(f"{len(labels)} labels for {len(input_ids)} tokens")
    n_answer = len(input_ids) - n_bos - prompt_len
    label_part = list(labels[start:])
    if n_answer < 1:
        problems.append("no answer tokens")
    if any(x != IGNORE_INDEX for x in labels[:start]):
        problems.append("a prompt position has a label")
    if len(label_part) != n_answer + n_bos - 1 or label_part != list(input_ids[start:]):
        problems.append(f"label positions {len(label_part)} do not equal the {n_answer} answer tokens"
                        f"{' + [BOS]' if n_bos == 2 else ''}")
    if not label_part or label_part[-1] != bundle.ids.newline:
        problems.append(f"the last label is not the newline id {bundle.ids.newline}")
    if problems:
        raise BatchingError(f"{where} ({label_mode}): label check failed: " + "; ".join(problems))


# ----------------------------------------------------------------------------
# Batches (§9.6.6, D22)
# ----------------------------------------------------------------------------
_TRAIN_TENSORS: Final = ("input_ids", "attention_mask", "labels", "hist_ids", "len_seq", "cand_ids")
_EVAL_TENSORS: Final = ("input_ids", "attention_mask", "hist_ids", "len_seq", "cand_ids", "target_ids")


@dataclass(frozen=True)
class TrainBatch:
    """§3.1 ``TrainBatch``: CPU int64 tensors, right padding (pad 32000, label -100)."""
    input_ids: Tensor                     # [B, L]
    attention_mask: Tensor                # [B, L] 1...1 0...0
    labels: Tensor                        # [B, L] -100 outside the answer (released_pair: second BOS too)
    hist_ids: Tensor                      # [B, 10] right padding 1682
    len_seq: Tensor                       # [B]
    cand_ids: Tensor                      # [B, 20] M2 order
    hybrid: bool                          # one format for the whole micro-batch (D3)
    tau: int
    sample_ids: list[str]
    template_ids: list[int]

    def pin_memory(self) -> "TrainBatch":
        """Copy with every tensor pinned (``torch2.0 utils/data/_utils/pin_memory.py:74-75`` calls this)."""
        return dataclasses.replace(self, **{n: getattr(self, n).pin_memory() for n in _TRAIN_TENSORS})

    def to(self, device: Union[torch.device, str]) -> "TrainBatch":
        """Copy with every tensor on ``device`` (``non_blocking=True``)."""
        return dataclasses.replace(self, **{n: getattr(self, n).to(device, non_blocking=True)
                                            for n in _TRAIN_TENSORS})


@dataclass(frozen=True)
class EvalBatch:
    """§3.1 ``EvalBatch``: CPU int64 tensors, left padding; titles for the M11 judgement."""
    input_ids: Tensor                     # [B, L] [PAD]...[PAD] [BOS] p
    attention_mask: Tensor                # [B, L] 0...0 1...1
    hist_ids: Tensor                      # [B, 10]
    len_seq: Tensor                       # [B]
    cand_ids: Tensor                      # [B, 20]
    target_ids: Tensor                    # [B]
    cand_titles: list[list[str]]          # the prompt strings, M2 order
    target_titles: list[str]
    hybrid: bool
    sample_ids: list[str]
    template_ids: list[int]

    def pin_memory(self) -> "EvalBatch":
        return dataclasses.replace(self, **{n: getattr(self, n).pin_memory() for n in _EVAL_TENSORS})

    def to(self, device: Union[torch.device, str]) -> "EvalBatch":
        return dataclasses.replace(self, **{n: getattr(self, n).to(device, non_blocking=True)
                                            for n in _EVAL_TENSORS})


def _batch_common(samples: Sequence[Sample], cands: Sequence[tuple[int, ...]], template_ids: Sequence[int],
                  encoded: Sequence[object], kind: str) -> tuple[Tensor, Tensor, Tensor]:
    """Shared checks and the hist_ids / len_seq / cand_ids tensors of one batch."""
    n = len(samples)
    if n == 0:
        raise BatchingError(f"{kind}: empty batch")
    if not len(cands) == len(template_ids) == len(encoded) == n:
        raise BatchingError(f"{kind}: {n} samples but {len(cands)} candidate tuples, {len(template_ids)} "
                            f"template ids and {len(encoded)} encodings")
    for s, c, k in zip(samples, cands, template_ids):
        if not isinstance(s, Sample):
            raise BatchingError(f"{kind}: {s!r} is not a Sample")
        if len(s.history) > HIST_LEN:
            raise BatchingError(f"{kind}: {s.sample_id} history length {len(s.history)} > {HIST_LEN}")
        if not isinstance(c, tuple) or len(c) != NUM_CANDS or not all(_is_int(x) for x in c):
            raise BatchingError(f"{kind}: {s.sample_id} candidates must be a tuple of {NUM_CANDS} ints, got {c!r}")
        if not _is_int(k) or k < 0:
            raise BatchingError(f"{kind}: {s.sample_id} template id must be an int >= 0, got {k!r}")
    hist_ids, len_seq = pad_histories([s.history for s in samples], PAD_ITEM, HIST_LEN)   # M1 §5.6.13
    cand_ids = torch.tensor([list(c) for c in cands], dtype=torch.int64)
    return hist_ids, len_seq, cand_ids


def collate_train(samples: Sequence[Sample], cands: Sequence[tuple[int, ...]], template_ids: Sequence[int],
                  encoded: Sequence[tuple[list[int], list[int]]], bundle: TokenizerBundle, *,
                  hybrid: bool, tau: int) -> TrainBatch:
    """Right-padded ``TrainBatch`` from per-sample ``(input_ids, labels)`` (§9.6.6, ``U model/model_interface.py:316``).

    ``input_ids = seq + [pad] * (L - n)``, ``attention_mask = [1] * n + [0] * (L - n)``,
    ``labels = lab + [-100] * (L - n)`` with ``L`` = the longest sample; real tokens keep
    positions 0..n-1, so no position_ids are needed.
    """
    hist_ids, len_seq, cand_ids = _batch_common(samples, cands, template_ids, encoded, "collate_train")
    if not isinstance(hybrid, bool):
        raise BatchingError(f"collate_train: hybrid must be bool, got {hybrid!r}")
    if not _is_int(tau) or tau < 1:
        raise BatchingError(f"collate_train: tau must be an int >= 1, got {tau!r}")
    for s, pair in zip(samples, encoded):
        if not (isinstance(pair, tuple) and len(pair) == 2 and len(pair[0]) == len(pair[1]) and len(pair[0]) > 0):
            raise BatchingError(f"collate_train: {s.sample_id} needs (input_ids, labels) of the same non-zero length")
    pad = bundle.ids.pad
    width = max(len(seq) for seq, _ in encoded)
    input_ids = [list(seq) + [pad] * (width - len(seq)) for seq, _ in encoded]
    mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq, _ in encoded]
    labels = [list(lab) + [IGNORE_INDEX] * (width - len(lab)) for _, lab in encoded]
    return TrainBatch(input_ids=torch.tensor(input_ids, dtype=torch.int64),
                      attention_mask=torch.tensor(mask, dtype=torch.int64),
                      labels=torch.tensor(labels, dtype=torch.int64),
                      hist_ids=hist_ids, len_seq=len_seq, cand_ids=cand_ids, hybrid=hybrid, tau=tau,
                      sample_ids=[s.sample_id for s in samples], template_ids=[int(k) for k in template_ids])


def collate_eval(samples: Sequence[Sample], cands: Sequence[tuple[int, ...]], template_ids: Sequence[int],
                 encoded: Sequence[list[int]], catalog: ItemCatalog, bundle: TokenizerBundle, *,
                 hybrid: bool) -> EvalBatch:
    """Left-padded ``EvalBatch`` from per-sample ``[BOS] + p`` (§9.6.6, D22).

    ``input_ids = [pad] * (L - n) + seq``, ``attention_mask = [0] * (L - n) + [1] * n``
    (the HF ``padding_side="left"`` result). Also used by the M13 overfit check on train
    samples (§17.6.8), so the split is not restricted here.
    """
    hist_ids, len_seq, cand_ids = _batch_common(samples, cands, template_ids, encoded, "collate_eval")
    if not isinstance(hybrid, bool):
        raise BatchingError(f"collate_eval: hybrid must be bool, got {hybrid!r}")
    if any(len(seq) == 0 for seq in encoded):
        raise BatchingError("collate_eval: empty encoding")
    pad = bundle.ids.pad
    width = max(len(seq) for seq in encoded)
    input_ids = [[pad] * (width - len(seq)) + list(seq) for seq in encoded]
    mask = [[0] * (width - len(seq)) + [1] * len(seq) for seq in encoded]
    titles = catalog.titles
    return EvalBatch(input_ids=torch.tensor(input_ids, dtype=torch.int64),
                     attention_mask=torch.tensor(mask, dtype=torch.int64),
                     hist_ids=hist_ids, len_seq=len_seq, cand_ids=cand_ids,
                     target_ids=torch.tensor([s.target for s in samples], dtype=torch.int64),
                     cand_titles=[[titles[c] for c in cs] for cs in cands],
                     target_titles=[titles[s.target] for s in samples], hybrid=hybrid,
                     sample_ids=[s.sample_id for s in samples], template_ids=[int(k) for k in template_ids])


# ----------------------------------------------------------------------------
# Datasets and loaders (§9.6.7, §9.6.8)
# ----------------------------------------------------------------------------
class CandidateFn(Protocol):
    """M2 §6.6.9: ``functools.partial(candidates_for, main_seed=..., cfg=...)``; ``epoch`` keyword-only."""

    def __call__(self, sample: Sample, *, epoch: Optional[int]) -> tuple[int, ...]: ...


def _identity(item: object) -> object:
    """collate_fn for ``batch_size=None`` (module level so that spawn workers can unpickle it)."""
    return item


class PlannedTrainBatches(Dataset):
    """Item ``i`` = the ``TrainBatch`` of ``plan.entries[i]`` (§9.6.7). Workers only tokenize the plan.

    Per sample, in ``sample_indices`` order: candidates ``candidates(s, epoch=plan.epoch)``
    (key ``("candidates", "train", row, epoch)``), template ``choose_template(..., plan.epoch)``,
    ``build_prompt`` with the plan's ``hybrid``, ``encode_train``, then the count and label checks.
    """

    def __init__(self, plan: BatchPlan, samples: Sequence[Sample], catalog: ItemCatalog,
                 templates: TemplateSet, candidates: CandidateFn, bundle: TokenizerBundle,
                 prompt_cfg: PromptConfig, batching_cfg: BatchingConfig, seed: int) -> None:
        if batching_cfg.label_mode not in LABEL_MODES:
            raise BatchingError(f"unknown batching.label_mode {batching_cfg.label_mode!r}, expected one of "
                                f"{LABEL_MODES}")
        check_catalog_titles(catalog)                       # M4 §8.6.2, once per dataset
        self.plan = plan
        self.samples = tuple(samples)
        bad = sorted({j for e in plan.entries for j in e.sample_indices if not 0 <= j < len(self.samples)})
        if bad:
            raise BatchingError(f"plan indices {bad[:10]} are outside the {len(self.samples)} train samples")
        self.catalog = catalog
        self.templates = templates
        self.candidates = candidates
        self.bundle = bundle
        self.prompt_cfg = prompt_cfg
        self.label_mode = batching_cfg.label_mode
        self.max_seq_len = batching_cfg.max_seq_len
        self.seed = seed

    def __len__(self) -> int:
        return len(self.plan)                               # SequentialSampler uses range(len(dataset))

    def __getitem__(self, index: int) -> TrainBatch:
        tau, sample_indices, hybrid = self.plan.entries[index]
        epoch = self.plan.epoch
        n_templates = len(self.templates.templates)
        n_bos = 2 if self.label_mode == "released_pair" else 1
        chosen, cand_list, template_ids, encoded = [], [], [], []
        for j in sample_indices:
            s = self.samples[j]
            where = f"{s.sample_id} tau {tau}"
            cands = self.candidates(s, epoch=epoch)
            k = choose_template(self.prompt_cfg, self.seed, n_templates, s, epoch)
            prompt = build_prompt(s, cands, self.catalog, self.templates, k, hybrid)
            input_ids, labels = encode_train(self.bundle, prompt, self.catalog.titles[s.target], self.label_mode)
            prompt_len = _prompt_len_from_labels(labels, where)
            check_token_counts(self.bundle, input_ids, hybrid=hybrid, len_seq=len(s.history),
                               prompt_len=prompt_len, n_bos=n_bos, max_seq_len=self.max_seq_len, where=where)
            _check_train_labels(self.bundle, input_ids, labels, label_mode=self.label_mode,
                                prompt_len=prompt_len, where=where)
            chosen.append(s)
            cand_list.append(cands)
            template_ids.append(k)
            encoded.append((input_ids, labels))
        return collate_train(chosen, cand_list, template_ids, encoded, self.bundle, hybrid=hybrid, tau=tau)


class EvalBatches(Dataset):
    """Item ``i`` = the ``EvalBatch`` of samples ``[i*bs, min((i+1)*bs, N))`` in row order (§9.6.8).

    No drop_last (D17). Candidates ``candidates(s, epoch=None)`` (key ``("candidates", split,
    row)``), template ``choose_template(..., None)``, format ``eval_is_hybrid`` (D4),
    ``encode_eval`` and the count check. Only one evaluation split per dataset.
    """

    def __init__(self, samples: Sequence[Sample], batch_size: int, catalog: ItemCatalog,
                 templates: TemplateSet, candidates: CandidateFn, bundle: TokenizerBundle,
                 prompt_cfg: PromptConfig, batching_cfg: BatchingConfig, seed: int) -> None:
        if not _is_int(batch_size) or batch_size < 1:
            raise BatchingError(f"eval.batch_size must be an int >= 1, got {batch_size!r}")
        samples = tuple(samples)
        if not samples:
            raise BatchingError("no evaluation samples")
        splits = sorted({getattr(s, "split", None) for s in samples}, key=str)
        if len(splits) != 1 or splits[0] not in EVAL_SPLITS or not all(isinstance(s, Sample) for s in samples):
            raise BatchingError(f"evaluation samples must all be Samples of one split in {EVAL_SPLITS}, got "
                                f"splits {splits}")
        check_catalog_titles(catalog)                       # M4 §8.6.2, once per dataset
        self.samples = samples
        self.batch_size = batch_size
        self.groups = tuple(range(i, min(i + batch_size, len(samples))) for i in range(0, len(samples), batch_size))
        self.catalog = catalog
        self.templates = templates
        self.candidates = candidates
        self.bundle = bundle
        self.prompt_cfg = prompt_cfg
        self.hybrid = eval_is_hybrid(prompt_cfg)
        self.max_seq_len = batching_cfg.max_seq_len
        self.seed = seed

    def __len__(self) -> int:
        return len(self.groups)                             # ceil(N / batch_size)

    def __getitem__(self, index: int) -> EvalBatch:
        n_templates = len(self.templates.templates)
        chosen, cand_list, template_ids, encoded = [], [], [], []
        for j in self.groups[index]:
            s = self.samples[j]
            cands = self.candidates(s, epoch=None)
            k = choose_template(self.prompt_cfg, self.seed, n_templates, s, None)
            prompt = build_prompt(s, cands, self.catalog, self.templates, k, self.hybrid)
            input_ids = encode_eval(self.bundle, prompt)
            check_token_counts(self.bundle, input_ids, hybrid=self.hybrid, len_seq=len(s.history),
                               prompt_len=len(input_ids) - 1, n_bos=1, max_seq_len=self.max_seq_len,
                               where=s.sample_id)
            chosen.append(s)
            cand_list.append(cands)
            template_ids.append(k)
            encoded.append(input_ids)
        return collate_eval(chosen, cand_list, template_ids, encoded, self.catalog, self.bundle, hybrid=self.hybrid)


def make_loader(dataset: Union[PlannedTrainBatches, EvalBatches], num_workers: int,
                pin_memory: bool) -> DataLoader:
    """``DataLoader(batch_size=None, shuffle=False, sampler=None, collate_fn=_identity, drop_last=False)``.

    ``batch_size=None`` turns automatic batching off and passes each dataset item through
    (``torch2.0 utils/data/dataloader.py:339-343``, ``_utils/fetch.py:52-54``); the
    SequentialSampler keeps the item order and several workers return items in index
    order (``dataloader.py:1299-1346``). No generator is passed: the iterator draws its
    base seed from the global CPU generator (``dataloader.py:605``, M10 §14.6.5).
    """
    if not _is_int(num_workers) or num_workers < 0:
        raise BatchingError(f"batching.num_workers must be an int >= 0, got {num_workers!r}")
    if not isinstance(pin_memory, bool):
        raise BatchingError(f"pin_memory must be bool, got {pin_memory!r}")
    return DataLoader(dataset, batch_size=None, shuffle=False, sampler=None, num_workers=num_workers,
                      collate_fn=_identity, pin_memory=pin_memory, drop_last=False)


def _candidate_fn(cfg: "Config") -> CandidateFn:
    """The picklable ``CandidateFn`` of M2 §6.6.9 (module-level function + frozen config)."""
    return functools.partial(candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)


def make_train_loader(plan: BatchPlan, data: LoadedData, cfg: "Config", epoch: int, *,
                      bundle: TokenizerBundle, templates: TemplateSet) -> Iterator[TrainBatch]:
    """``TrainBatch`` iterator of one epoch in plan order (§9.6.1, M10 §14.6.4 step 3).

    Checks ``plan.epoch == epoch``, every micro-batch has ``train.micro_batch_size`` samples
    and every index is a position in ``data.train``. Workers: ``batching.num_workers``;
    pinned memory only when ``model.device`` is ``cuda:0``.
    """
    if not _is_int(epoch) or plan.epoch != epoch:
        raise BatchingError(f"plan of epoch {plan.epoch} given for epoch {epoch!r}")
    size = cfg.train.micro_batch_size
    if not plan.entries:
        raise BatchingError(f"plan of epoch {plan.epoch} has no micro-batches")
    wrong = [e.tau for e in plan.entries if len(e.sample_indices) != size]
    if wrong:
        raise BatchingError(f"plan micro-batches at tau {wrong[:5]} do not have train.micro_batch_size = {size} "
                            f"samples (entry 0 has {len(plan.entries[0].sample_indices)})")
    n = len(data.train)
    outside = sorted({j for e in plan.entries for j in e.sample_indices if not 0 <= j < n})
    if outside:
        raise BatchingError(f"plan indices {outside[:10]} are outside len(data.train) = {n}")
    dataset = PlannedTrainBatches(plan, data.train, data.catalog, templates, _candidate_fn(cfg), bundle,
                                  cfg.prompt, cfg.batching, cfg.run.seed)
    return iter(make_loader(dataset, cfg.batching.num_workers, cfg.model.device == "cuda:0"))


def make_eval_loader(data: LoadedData, split: str, cfg: "Config", *, bundle: TokenizerBundle,
                     templates: TemplateSet) -> Iterator[EvalBatch]:
    """``EvalBatch`` iterator over the whole ``split`` (val or test) in row order, ``eval.batch_size`` (§9.6.8)."""
    if split not in EVAL_SPLITS:
        raise BatchingError(f"evaluation split must be one of {EVAL_SPLITS}, got {split!r}")
    dataset = EvalBatches(data.samples(split), cfg.eval.batch_size, data.catalog, templates, _candidate_fn(cfg),
                          bundle, cfg.prompt, cfg.batching, cfg.run.seed)
    return iter(make_loader(dataset, cfg.batching.num_workers, cfg.model.device == "cuda:0"))


__all__ = [
    "BatchingError", "PAD_TOKEN", "EXTRA_TOKENS", "TERMINATOR", "HIST_LEN", "NUM_CANDS", "PAD_ITEM", "LABEL_MODES",
    "IGNORE_INDEX", "EVAL_SPLITS", "BASE_VOCAB", "VOCAB_SIZE", "FINGERPRINTS", "TokenIds", "TokenizerBundle",
    "load_tokenizer", "encode_text", "encode_train", "encode_eval", "check_token_counts", "TrainBatch", "EvalBatch",
    "collate_train", "collate_eval", "CandidateFn", "PlannedTrainBatches", "EvalBatches", "make_loader",
    "make_train_loader", "make_eval_loader",
]
