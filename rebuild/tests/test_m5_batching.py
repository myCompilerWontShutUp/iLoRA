"""M5 tokenization과 batch 구성 시험 (DESIGN.md §9.8 #1-#10; §17.6.9 M5 표).

The worker-count part of #8 is ``test_workers.py``; #11 and #12 are ``vessl/test_vessl_env.py``;
the M9 re-check of ``bundle.ids`` in #1 is ``test_m9_model.py::test_build_rechecks_token_ids``.
Tests that need the tokenizer use the ``tokenizer_dir``/``bundle`` fixtures (``ILORA_TEST_TOKENIZER``,
skipped with "test tokenizer not available" otherwise). Golden values are the §9.6.2-§9.6.9 numbers
(test tokenizer sha256 9e556afd...d347, seed 1234).
"""
from __future__ import annotations

import ast
import functools
import json
import pickle
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaTokenizer

import refs_upstream
from ilora import batching, prompts
from ilora.batching import (IGNORE_INDEX, BatchingError, EvalBatch, EvalBatches, PlannedTrainBatches, TrainBatch,
                            check_token_counts, collate_eval, collate_train, encode_eval, encode_text, encode_train,
                            load_tokenizer, make_eval_loader, make_loader, make_train_loader)
from ilora.candidates import candidates_for
from ilora.data import ItemCatalog, Sample
from ilora.prompts import PromptError, build_batch_plan, build_prompt, choose_template, render_prompt

TOKENIZER_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
SPECIAL_IDS = {"[PAD]": 32000, "[PH]": 32001, "[HistoryEmb]": 32002, "[CansEmb]": 32003, "[ItemEmb]": 32004}
FINGERPRINT_1 = ("Star Wars\n", [7828, 9149, 13])
FINGERPRINT_2 = ("watched Toy Story [HistoryEmb], Heat [HistoryEmb] in the previous",
                 [20654, 29411, 13740, 32002, 1919, 940, 271, 32002, 297, 278, 3517])
N_CODE_CASES = 94 + 95 + 3000                    # §9.6.3 measurement, x 2 formats = 6,378
CASE_LENGTHS = (163, 314, 246)                   # answer_only lengths of the 6,378 (min, max, median)
WORST_CASE = {0: 631, 1: 632, 2: 639}            # §9.6.9 constructed worst case per template (answer_only)
TRAIN_E0_LENGTHS = (153, 334, 245)               # §9.6.3, all 66,126 train samples, epoch 0 keys
EVAL_LENGTHS = {"val": (211, 290, 241.5), "test": (204, 279, 240)}   # §9.6.4


# ----------------------------------------------------------------------------
# Helpers and module fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def code_tok(tokenizer_dir: Path) -> Any:
    """The released tokenizer set-up, ``U model/model_interface.py:313-317`` (pad = eos, then [PAD], right
    padding, then the four extra tokens). Reference only, never used by ``ilora``."""
    tok = LlamaTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)   # :313
    tok.pad_token = tok.eos_token                                               # :314 (overwritten next line)
    tok.add_special_tokens({"pad_token": "[PAD]"})                              # :315
    tok.padding_side = "right"                                                  # :316
    tok.add_special_tokens({"additional_special_tokens": ["[PH]", "[HistoryEmb]", "[CansEmb]", "[ItemEmb]"]})  # :317
    return tok


@pytest.fixture(scope="module")
def code_cases(full_data: Any, paper_cfg: Any) -> list:
    """(sample, candidates, template id) of validation 94 + test 95 (evaluation keys) and the first 3,000 train
    samples (epoch 0 keys), in that order (§9.6.3 measurement)."""
    seed, pc, cc = paper_cfg.run.seed, paper_cfg.prompt, paper_cfg.candidates
    cases = [(s, candidates_for(s, main_seed=seed, epoch=None, cfg=cc), choose_template(pc, seed, 3, s, None))
             for split in ("val", "test") for s in full_data.samples(split)]
    cases += [(s, candidates_for(s, main_seed=seed, epoch=0, cfg=cc), choose_template(pc, seed, 3, s, 0))
              for s in full_data.train[:3000]]
    assert len(cases) == N_CODE_CASES
    return cases


def _summary(values: list) -> tuple:
    return min(values), max(values), statistics.median(values)


# The files the slow LlamaTokenizer of transformers 4.28.0 reads (vocab file, tokenizer config, special tokens, added
# tokens when present). On VESSL ILORA_TEST_TOKENIZER is LLM_PATH (§17.6.1), which also holds about 12.6 GiB of
# weight shards: a test copies these files only, never the whole folder.
TOKENIZER_FILES = ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


def copy_tokenizer_files(source: Path, folder: Path) -> Path:
    """A new ``folder`` with the ``TOKENIZER_FILES`` of ``source`` that exist (``shutil.copy2``)."""
    folder.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, folder / name)
    return folder


def folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def _eval_parts(s: Sample, cfg: Any) -> tuple:
    seed = cfg.run.seed
    return (candidates_for(s, main_seed=seed, epoch=None, cfg=cfg.candidates),
            choose_template(cfg.prompt, seed, 3, s, None))


def _assert_batches_equal(a: Any, b: Any) -> None:
    assert type(a) is type(b)
    for name in a.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), name
        else:
            assert x == y, name


# ----------------------------------------------------------------------------
# M5-1 tokenizer
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-1")
def test_load_tokenizer_ids(bundle, tokenizer_dir, paper_cfg):
    tok, ids = bundle.tokenizer, bundle.ids
    assert paper_cfg.batching.tokenizer_sha256 == TOKENIZER_SHA256 == bundle.model_sha256
    assert type(tok) is LlamaTokenizer and not tok.is_fast and tok.sp_model_kwargs == {}
    assert len(tok) == 32005 == ids.vocab_size and tok.vocab_size == 32000 == ids.base_vocab
    assert (ids.bos, ids.eos, ids.unk, ids.newline) == (1, 2, 0, 13)
    assert (ids.pad, ids.ph, ids.history_emb, ids.cans_emb, ids.item_emb) == tuple(SPECIAL_IDS.values())
    assert ids.new_token_ids == (32000, 32001, 32002, 32003, 32004) and tok.pad_token_id == 32000
    for name, tid in SPECIAL_IDS.items():
        assert encode_text(bundle, name) == [tid] and tok.convert_tokens_to_ids(name) == tid
    assert tok.convert_tokens_to_ids("<0x0A>") == 13
    # fingerprints (§9.6.2 step 6)
    assert encode_text(bundle, FINGERPRINT_1[0]) == FINGERPRINT_1[1]
    assert tok.convert_ids_to_tokens(FINGERPRINT_1[1]) == ["▁Star", "▁Wars", "<0x0A>"]
    assert encode_text(bundle, FINGERPRINT_2[0]) == FINGERPRINT_2[1]
    assert tok.convert_ids_to_tokens([1919, 940, 271]) == ["▁,", "▁He", "at"]
    # str-registered special tokens absorb the spaces around them (T4.28 tokenization_utils.py:532-537)
    assert encode_text(bundle, "Toy Story [HistoryEmb] , Heat") == encode_text(bundle, "Toy Story[HistoryEmb], Heat")
    assert encode_text(bundle, "a  [PH]  b") == encode_text(bundle, "a[PH]b")
    assert encode_text(bundle, "a [PH] b") != encode_text(bundle, "a b")
    # every segment gets the dummy prefix: a separately encoded answer starts like a word after a space
    assert encode_text(bundle, "Heat\n")[0] == encode_text(bundle, "x Heat")[1]
    # what M9 build step 1 re-checks on bundle.ids (§13.6.3 step 1; the M9 test is test_m9_model.py)
    assert len(tok) == ids.vocab_size == 32005
    assert ids.new_token_ids == tuple(range(32000, 32005)) and min(ids.new_token_ids) >= ids.base_vocab
    assert (ids.bos, ids.eos) == (1, 2) and ids.pad != ids.eos
    # a second load gives the same ids; the bundle is independent of the session fixture
    again = load_tokenizer(tokenizer_dir, TOKENIZER_SHA256)
    assert again.ids == ids and again.tokenizer is not tok


@pytest.mark.criterion("M5-1")
@pytest.mark.parametrize("case", ["sha", "fast", "transformers_ver", "sentencepiece_ver", "sp_model_kwargs"])
def test_load_tokenizer_rejects(case, tokenizer_dir, tmp_path, monkeypatch):
    import sentencepiece
    import transformers

    folder, expected, pattern = tokenizer_dir, TOKENIZER_SHA256, None
    if case == "sha":
        expected, pattern = "0" * 64, f"sha256 {TOKENIZER_SHA256} != batching.tokenizer_sha256 {'0' * 64}"
    elif case == "fast":
        monkeypatch.setattr(LlamaTokenizer, "is_fast", property(lambda self: True))
        pattern = "not the slow LlamaTokenizer"
    elif case == "transformers_ver":
        monkeypatch.setattr(transformers, "__version__", "4.29.0")
        pattern = "transformers 4.29.0 != 4.28.0"
    elif case == "sentencepiece_ver":
        monkeypatch.setattr(sentencepiece, "__version__", "0.1.99")
        pattern = "sentencepiece 0.1.99 != 0.1.97"
    else:
        folder = copy_tokenizer_files(tokenizer_dir, tmp_path / "tokenizer")
        assert folder_bytes(folder) < 2**20                     # the tokenizer files only, also when it is LLM_PATH
        config_path = folder / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["sp_model_kwargs"] = {"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        pattern = "sp_model_kwargs"
    with pytest.raises(BatchingError, match=pattern):
        load_tokenizer(folder, expected)
    if case == "sha":
        with pytest.raises(BatchingError, match="not found"):
            load_tokenizer(tmp_path / "nowhere", TOKENIZER_SHA256)


@pytest.mark.criterion("M5-1")
def test_tokenizer_copy_leaves_weights_behind(tokenizer_dir, tmp_path):
    """``copy_tokenizer_files`` from a folder laid out like LLM_PATH on VESSL (tokenizer files next to config.json and
    a weight shard, here 2 MiB of zeros) copies the tokenizer files only, under 1 MiB, and ``load_tokenizer`` loads
    the copy with the same ids (§9.6.2)."""
    llama = copy_tokenizer_files(tokenizer_dir, tmp_path / "Llama-2-7b-hf")
    (llama / "config.json").write_text(json.dumps({"torch_dtype": "float16"}), encoding="utf-8")
    (llama / "model-00001-of-00002.safetensors").write_bytes(bytes(2 * 2**20))
    folder = copy_tokenizer_files(llama, tmp_path / "tokenizer")
    assert sorted(p.name for p in folder.iterdir()) == sorted(p.name for p in tokenizer_dir.iterdir()
                                                                if p.name in TOKENIZER_FILES)
    assert folder_bytes(folder) < 2**20 < folder_bytes(llama)
    assert load_tokenizer(folder, TOKENIZER_SHA256).ids == load_tokenizer(tokenizer_dir, TOKENIZER_SHA256).ids


# ----------------------------------------------------------------------------
# M5-2, M5-3 equality with the released encoding (6,378 cases)
# ----------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.criterion("M5-2")
def test_matches_code_pair_encoding(bundle, code_tok, code_cases, full_data, templates):
    titles = full_data.catalog.titles
    total = ok_answer = ok_released = 0
    lengths = {True: [], False: []}
    for hybrid in (True, False):
        for start in range(0, len(code_cases), 8):
            group = code_cases[start:start + 8]
            texts = [build_prompt(s, c, full_data.catalog, templates, k, hybrid) for s, c, k in group]
            answers = [titles[s.target] for s, _, _ in group]
            code_ids, code_mask, code_labels = refs_upstream.code_pair_input_and_labels(code_tok, texts, answers)
            if start < 94 and hybrid:   # validation batches: 4.28.0 token_type_ids are one column longer
                pair = code_tok([[p, a + "\n"] for p, a in zip(texts, answers)], return_tensors="pt",
                                padding="longest", truncation=False, add_special_tokens=True,
                                return_token_type_ids=True)
                assert pair.token_type_ids.shape[1] == pair.input_ids.shape[1] + 1
                if start == 0:
                    assert tuple(pair.input_ids.shape) == (8, 257) and tuple(pair.token_type_ids.shape) == (8, 258)
            for j, (text, answer) in enumerate(zip(texts, answers)):
                total += 1
                n = int(code_mask[j].sum())
                ids_c, lab_c = code_ids[j, :n].tolist(), code_labels[j, :n].tolist()
                k2 = 1 + len(encode_text(bundle, text))          # position of the second [BOS] in the code pair
                assert ids_c[0] == ids_c[k2] == bundle.ids.bos
                ids_r, lab_r = encode_train(bundle, text, answer, "released_pair")
                ids_a, lab_a = encode_train(bundle, text, answer, "answer_only")
                ok_released += (ids_r, lab_r) == (ids_c, lab_c)
                ok_answer += (ids_a, lab_a) == (ids_c[:k2] + ids_c[k2 + 1:], lab_c[:k2] + lab_c[k2 + 1:])
                s = group[j][0]
                for ids, n_bos in ((ids_a, 1), (ids_r, 2)):
                    check_token_counts(bundle, ids, hybrid=hybrid, len_seq=len(s.history), prompt_len=k2 - 1,
                                       n_bos=n_bos, max_seq_len=1024, where=s.sample_id)
                lengths[hybrid].append(len(ids_a))
    assert (total, ok_answer, ok_released) == (6378, 6378, 6378)
    assert lengths[True] == lengths[False]                       # every tag is one token
    assert _summary(lengths[True]) == CASE_LENGTHS


@pytest.mark.slow
@pytest.mark.criterion("M5-3")
def test_eval_encoding_matches_code(bundle, code_tok, code_cases, full_data, templates):
    total = same = 0
    for hybrid in (True, False):
        for start in range(0, len(code_cases), 8):
            group = code_cases[start:start + 8]
            texts = [build_prompt(s, c, full_data.catalog, templates, k, hybrid) for s, c, k in group]
            assert code_tok.padding_side == "right"
            out = code_tok(texts, return_tensors="pt", padding="longest", truncation=False, add_special_tokens=True,
                           return_token_type_ids=True)                      # U data/data_interface.py:94-100
            for j, text in enumerate(texts):
                total += 1
                n = int(out.attention_mask[j].sum())
                same += out.input_ids[j, :n].tolist() == encode_eval(bundle, text)
    assert (total, same) == (6378, 6378)


# ----------------------------------------------------------------------------
# M5-4 count checks
# ----------------------------------------------------------------------------
def _count_case(case: str, bundle: Any, full_data: Any, templates: Any, cfg: Any) -> tuple:
    """(input_ids, check kwargs) of one case; the error cases are expected to fail ``check_token_counts``."""
    s = full_data.val[2]
    cands, k = _eval_parts(s, cfg)
    catalog = full_data.catalog
    titles = catalog.titles
    hist = [titles[i] for i in s.history]
    cand = [titles[c] for c in cands]
    kw = dict(hybrid=True, len_seq=len(s.history), n_bos=1, max_seq_len=1024, where=s.sample_id)
    if case in ("hybrid", "text_only"):
        hybrid = case == "hybrid"
        ids = encode_eval(bundle, build_prompt(s, cands, catalog, templates, k, hybrid))
        kw["hybrid"] = hybrid
    elif case in ("answer_only", "released_pair", "released_wrong_prompt_len"):
        text = build_prompt(s, cands, catalog, templates, k, True)
        ids, _ = encode_train(bundle, text, titles[s.target], "answer_only" if case == "answer_only" else "released_pair")
        p_len = len(encode_text(bundle, text))
        kw.update(n_bos=1 if case == "answer_only" else 2,
                  prompt_len=p_len + (1 if case == "released_wrong_prompt_len" else 0))
        return ids, kw
    elif case == "title_with_history_tag":
        changed = list(titles)
        changed[s.history[0]] += " [HistoryEmb]"
        bad_catalog = ItemCatalog(titles=tuple(changed), title_mode="code", num_items=1682, pad_id=1682)
        with pytest.raises(PromptError):
            prompts.check_catalog_titles(bad_catalog)      # the datasets refuse such a catalog up front
        ids = encode_eval(bundle, build_prompt(s, cands, bad_catalog, templates, k, True))
    elif case == "candidates_19":
        ids = encode_eval(bundle, render_prompt(templates, k, hist, cand[:19], True))
    elif case == "item_emb_in_prompt":
        ids = encode_eval(bundle, render_prompt(templates, k, hist, cand, True) + " [ItemEmb]")
    elif case == "pad_in_prompt":
        ids = encode_eval(bundle, render_prompt(templates, k, hist, cand, True) + " [PAD]")
    elif case == "eos_in_prompt":
        ids = encode_eval(bundle, render_prompt(templates, k, hist, cand, True) + " </s>")
    elif case == "text_only_with_hybrid_counts":
        ids = encode_eval(bundle, render_prompt(templates, k, hist, cand, False))
    elif case == "history_11":
        ids = encode_eval(bundle, render_prompt(templates, k, hist, cand, True))
        kw["len_seq"] = 11
    else:
        raise AssertionError(case)
    kw["prompt_len"] = len(ids) - 1
    return ids, kw


COUNT_OK = ["hybrid", "text_only", "answer_only", "released_pair"]
COUNT_BAD = ["title_with_history_tag", "candidates_19", "item_emb_in_prompt", "pad_in_prompt", "eos_in_prompt",
             "text_only_with_hybrid_counts", "history_11", "released_wrong_prompt_len"]


@pytest.mark.criterion("M5-4")
@pytest.mark.parametrize("case", COUNT_OK + COUNT_BAD)
def test_token_count_checks(case, bundle, full_data, templates, paper_cfg):
    ids, kw = _count_case(case, bundle, full_data, templates, paper_cfg)
    counts = Counter(ids)
    b = bundle.ids
    if case in COUNT_OK:
        check_token_counts(bundle, ids, **kw)
        n = kw["len_seq"]
        if case == "text_only":
            assert (counts[b.history_emb], counts[b.cans_emb], counts[b.ph]) == (0, 0, n + 20)
        else:
            assert (counts[b.history_emb], counts[b.cans_emb], counts[b.ph]) == (n, 20, 0)
        assert counts[b.bos] == kw["n_bos"] and counts[b.item_emb] == counts[b.pad] == counts[b.eos] == 0
    else:
        with pytest.raises(BatchingError, match=kw["where"]):
            check_token_counts(bundle, ids, **kw)


@pytest.mark.criterion("M5-4")
def test_dataset_runs_the_checks(bundle, full_data, templates, paper_cfg, smoke_data, cpu_cfg):
    """The datasets run the checks for every sample: a catalog with a tag in a title and a too small
    max_seq_len stop the batch with an error instead of producing it."""
    changed = list(full_data.catalog.titles)
    changed[full_data.val[0].history[0]] += " [HistoryEmb]"
    bad_catalog = ItemCatalog(titles=tuple(changed), title_mode="code", num_items=1682, pad_id=1682)
    cand_fn = functools.partial(candidates_for, main_seed=1234, cfg=paper_cfg.candidates)
    with pytest.raises(PromptError, match="catalog title"):
        EvalBatches(full_data.val, 8, bad_catalog, templates, cand_fn, bundle, paper_cfg.prompt,
                    paper_cfg.batching, 1234)
    plan = build_batch_plan(smoke_data.train, 0, cpu_cfg)
    with pytest.raises(PromptError, match="catalog title"):
        PlannedTrainBatches(plan, smoke_data.train, bad_catalog, templates, cand_fn, bundle, cpu_cfg.prompt,
                            cpu_cfg.batching, 1234)


# ----------------------------------------------------------------------------
# M5-5 answers
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-5")
def test_answer_tokens_all_titles(bundle, full_data):
    titles = full_data.catalog.titles
    special = set(bundle.ids.new_token_ids) | {bundle.ids.bos, bundle.ids.eos, bundle.ids.unk}
    lengths = []
    for i, title in enumerate(titles):
        a = encode_text(bundle, title + batching.TERMINATOR)
        assert a[-1] == 13 and not special & set(a), (i, title, a)
        lengths.append(len(a))
    assert (min(lengths), max(lengths)) == (2, 23)
    assert [i for i, n in enumerate(lengths) if n == 23] == [1241, 1331]
    assert encode_text(bundle, titles[266] + "\n") == [29871, 13] and titles[266] == ""   # ▁ <0x0A>
    # the answer of encode_train is exactly these tokens, after the prompt
    ids, labels = encode_train(bundle, "Pick one:", titles[1241], "answer_only")
    a = encode_text(bundle, titles[1241] + "\n")
    assert ids[-len(a):] == a and [x for x in labels if x != IGNORE_INDEX] == a


# ----------------------------------------------------------------------------
# M5-6 padding
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-6")
def test_train_and_eval_padding(bundle, code_tok, full_data, templates, paper_cfg):
    seed, catalog = paper_cfg.run.seed, full_data.catalog
    samples = list(full_data.train[:8])
    cands = [candidates_for(s, main_seed=seed, epoch=0, cfg=paper_cfg.candidates) for s in samples]
    tids = [choose_template(paper_cfg.prompt, seed, 3, s, 0) for s in samples]
    texts = [build_prompt(s, c, catalog, templates, k, True) for s, c, k in zip(samples, cands, tids)]
    for mode in ("answer_only", "released_pair"):
        encoded = [encode_train(bundle, t, catalog.titles[s.target], mode) for t, s in zip(texts, samples)]
        batch = collate_train(samples, cands, tids, encoded, bundle, hybrid=True, tau=7)
        width = max(len(x) for x, _ in encoded)
        assert len({len(x) for x, _ in encoded}) > 1                 # real padding happens
        assert tuple(batch.input_ids.shape) == tuple(batch.labels.shape) == (8, width)
        for b, ((ids, lab), text) in enumerate(zip(encoded, texts)):
            n = len(ids)
            assert batch.input_ids[b].tolist() == ids + [32000] * (width - n)
            assert batch.attention_mask[b].tolist() == [1] * n + [0] * (width - n)
            p = encode_text(bundle, text)
            a = encode_text(bundle, catalog.titles[samples[b].target] + "\n")
            head = [IGNORE_INDEX] * (1 + len(p))
            answer = a if mode == "answer_only" else [1] + a
            assert batch.labels[b].tolist() == head + answer + [IGNORE_INDEX] * (width - n)
        assert (batch.hybrid, batch.tau) == (True, 7)
        assert batch.sample_ids == [s.sample_id for s in samples] and batch.template_ids == tids
    # hist_ids / len_seq / cand_ids
    assert batch.hist_ids.shape == (8, 10) and batch.cand_ids.shape == (8, 20) and batch.len_seq.shape == (8,)
    for b, s in enumerate(samples):
        assert batch.hist_ids[b].tolist() == list(s.history) + [1682] * (10 - len(s.history))
        assert batch.len_seq[b].item() == len(s.history) and batch.cand_ids[b].tolist() == list(cands[b])
    assert any(len(s.history) < 10 for s in samples)                 # the first windows are short
    for t in (batch.input_ids, batch.attention_mask, batch.labels, batch.hist_ids, batch.len_seq, batch.cand_ids):
        assert t.dtype == torch.int64 and t.device.type == "cpu"
    moved = batch.to(torch.device("cpu"))
    _assert_batches_equal(moved, batch)
    if torch.cuda.is_available():                                    # pinning needs CUDA
        _assert_batches_equal(batch.pin_memory(), batch)
    # evaluation: left padding equal to the HF padding_side="left" result
    val = list(full_data.val[:8])
    vparts = [_eval_parts(s, paper_cfg) for s in val]
    vtexts = [build_prompt(s, c, catalog, templates, k, True) for s, (c, k) in zip(val, vparts)]
    vencoded = [encode_eval(bundle, t) for t in vtexts]
    ebatch = collate_eval(val, [c for c, _ in vparts], [k for _, k in vparts], vencoded, catalog, bundle, hybrid=True)
    code_tok.padding_side = "left"
    try:
        hf = code_tok(vtexts, return_tensors="pt", padding="longest", add_special_tokens=True)
    finally:
        code_tok.padding_side = "right"
    assert len({len(x) for x in vencoded}) > 1
    assert torch.equal(ebatch.input_ids, hf.input_ids) and torch.equal(ebatch.attention_mask, hf.attention_mask)
    assert ebatch.target_ids.tolist() == [s.target for s in val]
    assert ebatch.cand_titles == [[catalog.titles[c] for c in cs] for cs, _ in vparts]
    assert ebatch.target_titles == [catalog.titles[s.target] for s in val]
    assert ebatch.hist_ids.shape == (8, 10) and ebatch.cand_ids.shape == (8, 20)
    for t in (ebatch.input_ids, ebatch.attention_mask, ebatch.hist_ids, ebatch.len_seq, ebatch.cand_ids,
              ebatch.target_ids):
        assert t.dtype == torch.int64
    _assert_batches_equal(ebatch.to("cpu"), ebatch)
    # collate errors
    with pytest.raises(BatchingError, match="empty batch"):
        collate_train([], [], [], [], bundle, hybrid=True, tau=1)
    with pytest.raises(BatchingError, match="tau"):
        collate_train(samples, cands, tids, encoded, bundle, hybrid=True, tau=0)
    with pytest.raises(BatchingError, match="candidates"):
        collate_train(samples, [c[:19] for c in cands], tids, encoded, bundle, hybrid=True, tau=1)
    with pytest.raises(BatchingError, match="same non-zero length"):
        collate_train(samples, cands, tids, [(x, y[:-1]) for x, y in encoded], bundle, hybrid=True, tau=1)
    with pytest.raises(BatchingError, match="hybrid"):
        collate_eval(val, [c for c, _ in vparts], [k for _, k in vparts], vencoded, catalog, bundle, hybrid=None)


# ----------------------------------------------------------------------------
# M5-7 lengths
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-7")
def test_constructed_worst_case(bundle, full_data, templates, paper_cfg):
    """§9.6.9: the 29 titles with the most ", title" tokens plus a 23-token answer item as target."""
    titles = full_data.catalog.titles
    piece = [len(encode_text(bundle, ", " + t)) for t in titles]
    order = sorted(range(len(titles)), key=lambda i: -piece[i])        # stable, like numpy argsort(kind="stable")
    answer_len = [len(encode_text(bundle, t + "\n")) for t in titles]
    longest = [i for i, n in enumerate(answer_len) if n == max(answer_len)]
    assert longest == [1241, 1331]
    max_len = paper_cfg.batching.max_seq_len
    assert max_len == 1024
    for k, expected in WORST_CASE.items():
        for target in longest:
            chosen = [i for i in order if i != target][:29]
            sample = Sample(split="train", row=0, user_id=0, history=tuple(chosen[:10]), target=target)
            cands = tuple(chosen[10:29] + [target])
            for hybrid in (True, False):
                text = build_prompt(sample, cands, full_data.catalog, templates, k, hybrid)
                ids_a, _ = encode_train(bundle, text, titles[target], "answer_only")
                ids_r, _ = encode_train(bundle, text, titles[target], "released_pair")
                assert (len(ids_a), len(ids_r)) == (expected, expected + 1), (k, target, hybrid)
                p_len = len(encode_text(bundle, text))
                check_token_counts(bundle, ids_r, hybrid=hybrid, len_seq=10, prompt_len=p_len, n_bos=2,
                                   max_seq_len=max_len, where="worst case")
    assert max(WORST_CASE.values()) + 1 + paper_cfg.model.gen_max_new_tokens == 704 <= max_len   # + 64 generated


@pytest.mark.slow
@pytest.mark.criterion("M5-7")
def test_real_max_lengths(bundle, full_data, templates, paper_cfg):
    seed, pc, cc, catalog = paper_cfg.run.seed, paper_cfg.prompt, paper_cfg.candidates, full_data.catalog
    train = []
    for s in full_data.train:
        text = build_prompt(s, candidates_for(s, main_seed=seed, epoch=0, cfg=cc), catalog, templates,
                            choose_template(pc, seed, 3, s, 0), True)
        train.append(len(encode_train(bundle, text, catalog.titles[s.target], "answer_only")[0]))
    assert len(train) == 66126 and _summary(train) == TRAIN_E0_LENGTHS
    for split, expected in EVAL_LENGTHS.items():
        lengths = []
        for s in full_data.samples(split):
            cands, k = _eval_parts(s, paper_cfg)
            lengths.append(len(encode_eval(bundle, build_prompt(s, cands, catalog, templates, k, True))))
        assert _summary(lengths) == expected, split
    assert max(train) <= paper_cfg.batching.max_seq_len


@pytest.mark.criterion("M5-7")
def test_too_long_rejected(bundle, full_data, templates, paper_cfg, smoke_data, cpu_cfg, profile_cfg):
    s = full_data.val[0]
    cands, k = _eval_parts(s, paper_cfg)
    text = build_prompt(s, cands, full_data.catalog, templates, k, True)
    ids = encode_eval(bundle, text)
    filler = encode_text(bundle, "the")
    assert filler == [278]                                              # ▁the
    long_ids = ids[:2] + filler * (1025 - len(ids)) + ids[2:]          # same counts, 1,025 tokens
    assert len(long_ids) == 1025
    kw = dict(hybrid=True, len_seq=len(s.history), n_bos=1, max_seq_len=1024, where=s.sample_id)
    with pytest.raises(BatchingError, match="1025 tokens > batching.max_seq_len 1024"):
        check_token_counts(bundle, long_ids, prompt_len=len(long_ids) - 1, **kw)
    check_token_counts(bundle, long_ids[:-1], prompt_len=len(long_ids) - 2, **kw)   # 1,024 passes
    # the dataset stops instead of truncating
    plan = build_batch_plan(smoke_data.train, 0, cpu_cfg)
    tight = profile_cfg("smoke_cpu", "batching.max_seq_len=150")
    cand_fn = functools.partial(candidates_for, main_seed=1234, cfg=tight.candidates)
    ds = PlannedTrainBatches(plan, smoke_data.train, smoke_data.catalog, templates, cand_fn, bundle, tight.prompt,
                             tight.batching, tight.run.seed)
    with pytest.raises(BatchingError, match="batching.max_seq_len 150"):
        ds[0]
    eds = EvalBatches(full_data.val, 8, full_data.catalog, templates, cand_fn, bundle, tight.prompt, tight.batching,
                      1234)
    with pytest.raises(BatchingError, match="batching.max_seq_len 150"):
        eds[0]


# ----------------------------------------------------------------------------
# M5-8 pickling and the CandidateFn call form (worker counts: test_workers.py)
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-8")
def test_dataset_pickle_round_trip(bundle, smoke_data, full_data, templates, cpu_cfg, paper_cfg):
    cand_fn = functools.partial(candidates_for, main_seed=cpu_cfg.run.seed, cfg=cpu_cfg.candidates)
    plan = build_batch_plan(smoke_data.train, 0, cpu_cfg)
    ds = PlannedTrainBatches(plan, smoke_data.train, smoke_data.catalog, templates, cand_fn, bundle, cpu_cfg.prompt,
                             cpu_cfg.batching, cpu_cfg.run.seed)
    eds = EvalBatches(full_data.val, 8, full_data.catalog, templates, cand_fn, bundle, paper_cfg.prompt,
                      paper_cfg.batching, paper_cfg.run.seed)
    for original in (ds, eds):
        restored = pickle.loads(pickle.dumps(original))
        assert restored.bundle.tokenizer is not original.bundle.tokenizer
        assert restored.bundle.tokenizer.sp_model is not None     # reloaded (T4.28 tokenization_llama.py:103-106)
        assert restored.bundle.ids == original.bundle.ids
        for text in (FINGERPRINT_1[0], FINGERPRINT_2[0], "a [PH] b [CansEmb], c [ItemEmb] [PAD]"):
            assert encode_text(restored.bundle, text) == encode_text(original.bundle, text)
        assert len(restored) == len(original)
        for i in (0, len(original) - 1):
            _assert_batches_equal(restored[i], original[i])
    bundle_copy = pickle.loads(pickle.dumps(bundle))
    assert encode_text(bundle_copy, FINGERPRINT_2[0]) == FINGERPRINT_2[1]


class _KeywordSpy:
    """Picklable ``CandidateFn`` that records how it is called (``epoch`` keyword-only, like ``candidates_for``)."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.epochs: list = []

    def __call__(self, sample: Sample, *, epoch: Any) -> tuple:
        self.epochs.append(epoch)
        return self.fn(sample, epoch=epoch)


@pytest.mark.criterion("M5-8")
def test_candidate_fn_keyword_only(bundle, smoke_data, full_data, templates, cpu_cfg, paper_cfg):
    fn = functools.partial(candidates_for, main_seed=cpu_cfg.run.seed, cfg=cpu_cfg.candidates)
    s = smoke_data.train[0]
    assert fn(s, epoch=0) == candidates_for(s, main_seed=1234, epoch=0, cfg=cpu_cfg.candidates)
    with pytest.raises(TypeError):
        fn(s, 0)
    made = batching._candidate_fn(cpu_cfg)
    assert made.func is candidates_for and made.args == () and made.keywords == {"main_seed": 1234,
                                                                                  "cfg": cpu_cfg.candidates}
    # no call in batching.py passes the epoch positionally (static check of the source)
    tree = ast.parse(Path(batching.__file__).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "candidates"]
    assert len(calls) == 2
    for call in calls:
        assert len(call.args) == 1 and [k.arg for k in call.keywords] == ["epoch"]
    # and at run time: training key epoch = plan.epoch, evaluation key epoch = None
    plan = build_batch_plan(smoke_data.train, 1, cpu_cfg)
    spy = _KeywordSpy(fn)
    PlannedTrainBatches(plan, smoke_data.train, smoke_data.catalog, templates, spy, bundle, cpu_cfg.prompt,
                        cpu_cfg.batching, 1234)[3]
    eval_spy = _KeywordSpy(fn)
    EvalBatches(full_data.test, 8, full_data.catalog, templates, eval_spy, bundle, paper_cfg.prompt,
                paper_cfg.batching, 1234)[0]
    assert spy.epochs == [1] * cpu_cfg.train.micro_batch_size and eval_spy.epochs == [None] * 8


# ----------------------------------------------------------------------------
# M5-9 evaluation loader
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-9")
def test_eval_loader_test_split(bundle, full_data, templates, profile_cfg):
    cfg = profile_cfg("paper_spec", "batching.num_workers=0")
    assert cfg.eval.batch_size == 8
    batches = list(make_eval_loader(full_data, "test", cfg, bundle=bundle, templates=templates))
    assert len(batches) == 12 and [len(b.sample_ids) for b in batches] == [8] * 11 + [7]
    assert all(isinstance(b, EvalBatch) and b.hybrid is True for b in batches)
    ids = [x for b in batches for x in b.sample_ids]
    assert ids == [s.sample_id for s in full_data.test] and len(set(ids)) == 95     # row order, no gaps or repeats
    rows = [int(x.split(":")[1]) for x in ids]
    assert rows == sorted(rows)
    flat = [(s, b, i) for b in batches for i, s in enumerate(b.sample_ids)]
    by_id = {s.sample_id: s for s in full_data.test}
    for sample_id, batch, i in flat:
        s = by_id[sample_id]
        cands, k = _eval_parts(s, cfg)
        assert batch.cand_ids[i].tolist() == list(cands) and batch.template_ids[i] == k
        assert batch.target_ids[i].item() == s.target and batch.target_titles[i] == full_data.catalog.titles[s.target]
    val = list(make_eval_loader(full_data, "val", cfg, bundle=bundle, templates=templates))
    assert [len(b.sample_ids) for b in val] == [8] * 11 + [6]
    text_cfg = profile_cfg("variant_text_only_eval", "batching.num_workers=0")
    first = next(make_eval_loader(full_data, "test", text_cfg, bundle=bundle, templates=templates))
    assert first.hybrid is False and (first.input_ids == bundle.ids.ph).any()
    assert not (first.input_ids == bundle.ids.history_emb).any()
    with pytest.raises(BatchingError, match="split"):
        make_eval_loader(full_data, "train", cfg, bundle=bundle, templates=templates)
    cand_fn = functools.partial(candidates_for, main_seed=1234, cfg=cfg.candidates)
    for samples, pattern in [(full_data.val[:3] + full_data.test[:3], "one split"),
                             (full_data.train[:5], "one split"), ((), "no evaluation samples")]:
        with pytest.raises(BatchingError, match=pattern):
            EvalBatches(samples, 8, full_data.catalog, templates, cand_fn, bundle, cfg.prompt, cfg.batching, 1234)
    for size in (0, -1, True):
        with pytest.raises(BatchingError, match="eval.batch_size"):
            EvalBatches(full_data.test, size, full_data.catalog, templates, cand_fn, bundle, cfg.prompt,
                        cfg.batching, 1234)
    with pytest.raises(BatchingError, match="num_workers"):
        make_loader(EvalBatches(full_data.test, 8, full_data.catalog, templates, cand_fn, bundle, cfg.prompt,
                                cfg.batching, 1234), -1, False)


# ----------------------------------------------------------------------------
# M5-10 train loader
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M5-10")
def test_make_train_loader_checks(bundle, smoke_data, templates, cpu_cfg, profile_cfg):
    assert cpu_cfg.batching.num_workers == 0 and cpu_cfg.train.micro_batch_size == 4
    plan = build_batch_plan(smoke_data.train, 0, cpu_cfg)
    assert len(plan) == 39 and len(smoke_data.train) == 156                       # §17.6.6
    with pytest.raises(BatchingError, match="epoch 0 given for epoch 1"):
        make_train_loader(plan, smoke_data, cpu_cfg, 1, bundle=bundle, templates=templates)
    with pytest.raises(BatchingError, match="micro_batch_size = 2"):
        make_train_loader(plan, smoke_data, profile_cfg("smoke_cpu", "train.micro_batch_size=2"), 0,
                          bundle=bundle, templates=templates)
    wide = prompts.make_batch_plan(seed=1234, epoch=0, epochs=2, num_samples=200, micro_batch_size=4,
                                   curriculum="linear")
    with pytest.raises(BatchingError, match="outside len"):
        make_train_loader(wide, smoke_data, cpu_cfg, 0, bundle=bundle, templates=templates)
    iterator = make_train_loader(plan, smoke_data, cpu_cfg, 0, bundle=bundle, templates=templates)
    batches = list(iterator)
    assert len(batches) == len(plan) == 39
    with pytest.raises(StopIteration):
        next(iterator)
    for batch, entry in zip(batches, plan):
        assert isinstance(batch, TrainBatch)
        assert (batch.tau, batch.hybrid) == (entry.tau, entry.hybrid)
        assert batch.sample_ids == [smoke_data.train[j].sample_id for j in entry.sample_indices]
        assert ((batch.labels != IGNORE_INDEX).sum(1) > 0).all()                 # every row has answer labels
        hyb = (batch.input_ids == bundle.ids.history_emb).sum().item()
        assert (hyb > 0) == entry.hybrid
    assert [b.tau for b in batches] == list(range(1, 40)) and sum(b.hybrid for b in batches) == plan.hybrid_count()
    # the same plan with released_pair labels: two [BOS] per row, the second one a label
    released = profile_cfg("smoke_cpu", "batching.label_mode=released_pair")
    first = next(make_train_loader(plan, smoke_data, released, 0, bundle=bundle, templates=templates))
    assert ((first.input_ids == 1).sum(1) == 2).all() and ((first.labels == 1).sum(1) == 1).all()
