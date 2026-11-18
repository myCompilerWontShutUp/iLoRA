"""M4 prompt·M5 token 시험: Steam, LastFM (DESIGN.md §23.2, §23.8 #8, #9; D28, D35, D37, D39).

Templates: ``prompt/game.txt`` and ``prompt/artist.txt`` pass the hash check of the M0 input table and parse into three
templates with the history slot first (D37). Titles: the slot-string rule of D39 accepts the bracketed titles of the
two catalogs and the MovieLens rule still refuses them. Token checks: with the real tokenizer (``bundle``, skipped only
with ``SKIP_NO_TOKENIZER``) every validation and test prompt, prompts built around the bracketed titles and the
constructed longest prompt (the §9.6.9 method as ``multi/sasrec_prompts/worst_case.py`` ran it) pass
``check_token_counts`` within ``batching.max_seq_len``; M5 batches pad with the dataset's ``num_items`` (§23.5).
Tests that read the real frames or tokenize many prompts carry ``slow`` (§23.8).
"""
from __future__ import annotations

import dataclasses
import functools
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ilora import data, prompts, run, seeding
from ilora.batching import (EvalBatches, PlannedTrainBatches, check_token_counts, encode_eval, encode_text,
                            encode_train)
from ilora.candidates import candidates_for
from ilora.config import load_config
from ilora.data import ItemCatalog, Sample
from ilora.datasets import MOVIELENS, SPECS
from ilora.prompts import (PromptError, build_batch_plan, build_prompt, check_catalog_titles, choose_template,
                           load_templates, parse_templates)

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
PROFILE_DIR = REBUILD_DIR / "configs" / "datasets"
NAMES = ("steam", "lastfm")
EVAL_SPLITS = ("val", "test")

# §8.6.2 step 6 (measured): the stripped lines of U prompt/game.txt:1-3 and U prompt/artist.txt:1-3 (D37: the "10 music
# artist names" of artist.txt line 2 stays as it is).
TEMPLATES = {
    "steam": (
        "This user has played [HistoryHere] in the previous. Please predict the next game this user will play. "
        "The game title candidates are [CansHere]. Choose only one game from the candidates. The answer is",
        "This user has played [HistoryHere] in the previous. Given the following 20 game titles, [CansHere], "
        "recommend one game for this user to play next. The game title you recommend is",
        "The visit history of this user is [HistoryHere]. Recommend a next game for this user to play from the "
        "following game title set, [CansHere]. The recommendation should contain one game title only. "
        "The recommendation is",
    ),
    "lastfm": (
        "This user has listened to [HistoryHere] in the previous. Please predict the next music artist this user will "
        "listen to. The artist name candidates are [CansHere]. Choose only one artist from the candidates. "
        "The answer is",
        "This user has listened to [HistoryHere] in the previous. Given the following 10 music artist names, "
        "[CansHere], recommend one artist for this user to listen to next. The artist name you recommend is",
        "The listening history of this user is [HistoryHere]. Recommend the next music artist for this user to listen "
        "to from the following artist name set, [CansHere]. The recommendation should contain one artist's name "
        "only. The recommendation is",
    ),
}
TEMPLATE_SHA256 = {"steam": "cdc7f7160d5e6beac2c31449540b55525cc069c5aa70277c0ca307a544f4b7b3",      # §23.2 템플릿 sha256
                   "lastfm": "0b85e0e091281f96460942e177273757a540a7f3b3743b500f53ea37e4288094"}    # (text)
# §23.2 대괄호 제목 (D39): the seven bracketed titles of the two catalogs.
BRACKET_TITLES = {
    "steam": {2224: "[the Sequence]", 2592: "Corrosion: Cold Winter Waiting [Enhanced Edition]"},
    "lastfm": {678: "[:SITD:]", 690: "Nurzery [Rhymes]", 1016: "[unknown]", 2985: "Dungeon Elite [**]",
               3568: "Superchic[k]"},
}
# D39: the strings Steam and LastFM titles may not hold (the slots and tags of M4 and M5, the special token strings).
ID2NAME_FORBIDDEN = ("[HistoryHere]", "[CansHere]", "[HistoryEmb]", "[CansEmb]", "[PH]", "[ItemEmb]", "[PAD]",
                     "<s>", "</s>", "<unk>")
# §23.2 평가 prompt 최대 길이, the §9.6.4 method (multi/sasrec_prompts/token_lengths_full.py): [BOS] + prompt of the
# chosen template, hybrid, over every sample of the split.
EVAL_MAX = {"steam": {"val": 363, "test": 392}, "lastfm": {"val": 241, "test": 233}}
# §23.2 구성 최장 prompt / + 응답 / + 생성 64 (multi/sasrec_prompts/worst_case.py, per template in worst_case.json).
WORST_CASE = {
    "steam": SimpleNamespace(per_template={0: 872, 1: 870, 2: 875}, target=1232, answer=53, totals=(875, 928, 939)),
    "lastfm": SimpleNamespace(per_template={0: 434, 1: 432, 2: 438}, target=1995, answer=17, totals=(438, 455, 502)),
}


@functools.lru_cache(maxsize=None)
def _released(name: str) -> SimpleNamespace:
    """``load_dataset`` of ``<name>_paper_spec`` on the released files and the templates of the dataset."""
    spec = SPECS[name]
    cfg, _ = load_config(PROFILE_DIR / f"{name}_paper_spec.yaml")
    return SimpleNamespace(name=name, spec=spec, cfg=cfg, full=data.load_dataset(cfg, REPO_ROOT),
                           templates=load_templates(REPO_ROOT / spec.template_file, spec=spec))


@pytest.fixture(scope="module", params=NAMES)
def released(request: pytest.FixtureRequest) -> SimpleNamespace:
    return _released(request.param)


@functools.lru_cache(maxsize=None)
def _catalog(name: str) -> ItemCatalog:
    """The real ``code`` catalog read from the title file only (no frame parsing, no tokenizer)."""
    spec = SPECS[name]
    raw = data.read_raw_titles(REPO_ROOT / spec.data_dir / spec.item_file, spec=spec)
    return data.build_catalog(raw, "code", spec=spec)


def _filler_catalog(num_items: int, special: dict[int, str]) -> ItemCatalog:
    """Fixed filler titles with some titles replaced; the replaced ones are the only bracketed or forbidden strings."""
    titles = [f"filler title {i:04d}" for i in range(num_items)]
    for i, title in special.items():
        titles[i] = title
    return ItemCatalog(titles=tuple(titles), title_mode="code", num_items=num_items, pad_id=num_items)


# ----------------------------------------------------------------------------
# Templates (§8.6.2, §23.8 #8; D37)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_templates_parse_into_three_with_the_history_slot_first(name):
    """D37: the hash-checked load gives the three stripped lines, each with the history slot before the candidates."""
    spec = SPECS[name]
    loaded = load_templates(REPO_ROOT / spec.template_file, spec=spec)
    assert loaded.templates == TEMPLATES[name] and len(loaded.parts) == 3
    for text, part in zip(loaded.templates, loaded.parts):
        assert part[1] == prompts.HISTORY_SLOT and part[3] == prompts.CANDS_SLOT
        assert "".join(part) == text
        assert text.count(prompts.HISTORY_SLOT) == text.count(prompts.CANDS_SLOT) == 1
    expected = TEMPLATE_SHA256[name]
    assert run.content_sha256(REPO_ROOT / spec.template_file, text=True) == expected
    assert run.expected_inputs(name)[spec.template_file] == (expected, True)


@pytest.mark.parametrize("name", NAMES)
def test_load_templates_rejects_a_modified_copy(name, tmp_path):
    """§8.6.2 steps 0-1: the hash check names the dataset's table and only the hash check rejects a one-byte change (the
    content still parses); a missing file is an error; LF and CRLF copies have the same text hash and load the same."""
    spec = SPECS[name]
    expected = TEMPLATE_SHA256[name]
    lf = (REPO_ROOT / spec.template_file).read_bytes().replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert lf != crlf and b"\r" not in lf
    file_name = Path(spec.template_file).name
    for label, content in (("lf", lf), ("crlf", crlf)):
        path = tmp_path / label / file_name
        path.parent.mkdir()
        path.write_bytes(content)
        assert run.content_sha256(path, text=True) == expected
        assert load_templates(path, spec=spec).templates == TEMPLATES[name]
    changed = bytes([lf[0] ^ 0x20]) + lf[1:]                         # one byte: "T" -> "t"
    path = tmp_path / "modified" / file_name
    path.parent.mkdir()
    path.write_bytes(changed)
    got = run.content_sha256(path, text=True)
    assert got != expected
    table = re.escape(f"run.expected_inputs({name!r})")
    with pytest.raises(PromptError, match=f"{got} != expected {expected} \\({table}"):
        load_templates(path, spec=spec)
    assert parse_templates(changed, source="changed").templates[0].startswith("this user")
    with pytest.raises(PromptError, match="not found"):
        load_templates(tmp_path / "missing" / file_name, spec=spec)


@pytest.mark.parametrize("name", NAMES)
def test_load_templates_checks_the_file_of_the_dataset(name):
    """§23.4: the file is compared with the entry of the ``spec`` it is loaded for: the default ``spec`` is MovieLens,
    and neither the MovieLens file nor the file of the other dataset passes."""
    spec = SPECS[name]
    path = REPO_ROOT / spec.template_file
    movielens_sha = run.expected_inputs("movielens")[MOVIELENS.template_file][0]
    with pytest.raises(PromptError, match=f"!= expected {movielens_sha}"):
        load_templates(path)
    with pytest.raises(PromptError, match=f"!= expected {TEMPLATE_SHA256[name]}"):
        load_templates(REPO_ROOT / MOVIELENS.template_file, spec=spec)
    other = SPECS["lastfm" if name == "steam" else "steam"]
    with pytest.raises(PromptError, match=f"!= expected {TEMPLATE_SHA256[other.name]}"):
        load_templates(path, spec=other)


# ----------------------------------------------------------------------------
# Title rules (§8.6.2, §23.8 #8; D39)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_bracket_titles_follow_the_dataset_rule(name):
    """D39: the real catalog passes the slot-string rule; the MovieLens rule (also the default) refuses it because of
    the bracketed titles of §23.2, exactly those."""
    spec, catalog = SPECS[name], _catalog(name)
    expected = BRACKET_TITLES[name]
    assert {i: catalog.titles[i] for i in expected} == expected
    assert [i for i, t in enumerate(catalog.titles) if "[" in t or "]" in t] == sorted(expected)
    assert check_catalog_titles(catalog, spec.title_forbidden) is None
    with pytest.raises(PromptError, match=f"{len(expected)} catalog title") as info:
        check_catalog_titles(catalog, MOVIELENS.title_forbidden)
    assert all(f"item {i}:" in str(info.value) for i in expected)
    with pytest.raises(PromptError, match=f"{len(expected)} catalog title"):
        check_catalog_titles(catalog)                                 # the default argument is the MovieLens rule


@pytest.mark.parametrize("name,item", [(n, i) for n in NAMES for i in BRACKET_TITLES[n]])
def test_each_bracket_title_is_accepted_alone_and_refused_by_the_movielens_rule(name, item):
    """D39: one bracketed title in a catalog with no other: the new rule keeps it, the MovieLens rule refuses it."""
    spec = SPECS[name]
    catalog = _filler_catalog(spec.num_items, {5: BRACKET_TITLES[name][item]})
    assert check_catalog_titles(catalog, spec.title_forbidden) is None
    with pytest.raises(PromptError, match="1 catalog title"):
        check_catalog_titles(catalog, MOVIELENS.title_forbidden)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", ID2NAME_FORBIDDEN)
def test_slot_and_special_strings_stay_forbidden(name, bad):
    """D39: the ten slot and special token strings stay forbidden in titles, whatever surrounds them."""
    spec = SPECS[name]
    for title in (f"A {bad} B", f"[{bad}]", bad):
        catalog = _filler_catalog(spec.num_items, {7: title})
        with pytest.raises(PromptError, match="item 7:"):
            check_catalog_titles(catalog, spec.title_forbidden)
    assert check_catalog_titles(_filler_catalog(spec.num_items, {7: "A B"}), spec.title_forbidden) is None


# ----------------------------------------------------------------------------
# Real tokenizer: every title, every evaluation prompt, the bracketed titles and the longest prompt (§23.8 #9)
# ----------------------------------------------------------------------------
def _sample(name: str, history: tuple[int, ...], target: int) -> Sample:
    """A constructed train ``Sample`` of the dataset (D28: its ``num_items``, so ids above 1681 are valid)."""
    return Sample(split="train", row=0, user_id=0, history=tuple(history), target=target,
                  num_items=SPECS[name].num_items)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_every_title_encodes_without_special_tokens(name, bundle):
    """§9.6.5 for any item of the catalog, as a history item, a candidate or a target: no title tokenizes to a special
    id or to nothing, and ``title + "\\n"`` (the answer, D18) ends with the newline id."""
    ids = bundle.ids
    special = {ids.bos, ids.eos, ids.unk, ids.pad, ids.ph, ids.history_emb, ids.cans_emb, ids.item_emb}
    catalog = _catalog(name)
    assert len(catalog.titles) == SPECS[name].num_items
    for i, title in enumerate(catalog.titles):
        tokens = encode_text(bundle, title)
        assert tokens and not set(tokens) & special, (i, title)
        assert encode_text(bundle, title + "\n")[-1] == ids.newline, (i, title)


@pytest.mark.slow
def test_every_evaluation_prompt_passes_the_token_checks(released, bundle):
    """§23.8 #9: every validation and test sample, all three templates and both formats, through ``build_prompt`` and
    ``check_token_counts`` (§9.6.5). The chosen template and the hybrid format give the §9.6.4 lengths of §23.2."""
    spec, cfg, full, templates = released.spec, released.cfg, released.full, released.templates
    seed, max_len = cfg.run.seed, cfg.batching.max_seq_len
    assert max_len == 1024
    for split in EVAL_SPLITS:
        samples = full.samples(split)
        assert len(samples) == spec.expected_samples[split]
        chosen_lengths = []
        for s in samples:
            cands = candidates_for(s, main_seed=seed, epoch=None, cfg=cfg.candidates)
            chosen = choose_template(cfg.prompt, seed, 3, s, None)      # D15, D35: the key has no dataset name
            assert chosen == int(seeding.np_rng(seed, "template", split, s.row).integers(0, 3)), s.sample_id
            for k in range(3):
                lengths = []
                for hybrid in (True, False):
                    ids = encode_eval(bundle, build_prompt(s, cands, full.catalog, templates, k, hybrid))
                    check_token_counts(bundle, ids, hybrid=hybrid, len_seq=len(s.history), prompt_len=len(ids) - 1,
                                       n_bos=1, max_seq_len=max_len, where=f"{s.sample_id} template {k}")
                    lengths.append(len(ids))
                assert lengths[0] == lengths[1], (s.sample_id, k)        # every tag is one token
                if k == chosen:
                    chosen_lengths.append(lengths[0])
        assert max(chosen_lengths) == EVAL_MAX[released.name][split], split


def _check_every_format(sample: Sample, cands: tuple[int, ...], catalog: ItemCatalog, templates, bundle, max_len: int,
                        where: str) -> None:
    """``build_prompt`` of every template and format, then the evaluation and both training encodings through
    ``check_token_counts`` (§9.6.3-§9.6.5, D18)."""
    title = catalog.titles[sample.target]
    for k in range(len(templates.templates)):
        for hybrid in (True, False):
            here = f"{where} template {k} {'hybrid' if hybrid else 'text-only'}"
            text = build_prompt(sample, cands, catalog, templates, k, hybrid)
            p_len = len(encode_text(bundle, text))
            kw = dict(hybrid=hybrid, len_seq=len(sample.history), max_seq_len=max_len, where=here)
            ids = encode_eval(bundle, text)
            assert len(ids) == 1 + p_len, here
            check_token_counts(bundle, ids, prompt_len=p_len, n_bos=1, **kw)
            ids_a, _ = encode_train(bundle, text, title, "answer_only")
            check_token_counts(bundle, ids_a, prompt_len=p_len, n_bos=1, **kw)
            ids_r, _ = encode_train(bundle, text, title, "released_pair")
            check_token_counts(bundle, ids_r, prompt_len=p_len, n_bos=2, **kw)


def _bracket_cases(name: str) -> list[tuple[str, Sample, tuple[int, ...]]]:
    """Constructed ``(where, sample, candidates)``: each bracketed title in a history and as the target (a candidate
    and the answer), and all of them together in one prompt. Valid samples of the §3.1 candidate contract."""
    brackets = sorted(BRACKET_TITLES[name])
    fill = [i for i in range(SPECS[name].num_items) if i not in BRACKET_TITLES[name]][:30]
    cases = []
    for b in brackets:
        cases.append((f"history {b}", _sample(name, (b, *fill[:2]), fill[2]), (*fill[3:22], fill[2])))
        cases.append((f"target {b}", _sample(name, tuple(fill[:3]), b), (b, *fill[3:22])))
    head, tail = brackets[:3], brackets[3:]
    history = (*head, *fill[:3 - len(head)])
    target = tail[0] if tail else next(i for i in fill if i not in history)
    negatives = [*tail[1:], *(i for i in fill if i not in history and i != target)][:19]
    cases.append(("all bracket titles", _sample(name, history, target), (*negatives, target)))
    return cases


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_bracket_title_prompts_pass_the_token_checks(name, bundle):
    """D39 (§23.2, multi/sasrec_prompts/extra_checks.py, here through ``build_prompt``): prompts holding each bracketed
    title, alone and all together, pass ``check_token_counts`` in every template, format and encoding; the title is kept
    as it is, right before its tag."""
    spec = SPECS[name]
    catalog = _catalog(name)
    templates = load_templates(REPO_ROOT / spec.template_file, spec=spec)
    cfg, _ = load_config(PROFILE_DIR / f"{name}_paper_spec.yaml")
    assert catalog.num_items == spec.num_items
    for where, sample, cands in _bracket_cases(name):
        _check_every_format(sample, cands, catalog, templates, bundle, cfg.batching.max_seq_len, where)
        hybrid_text = build_prompt(sample, cands, catalog, templates, 0, True)
        text_only = build_prompt(sample, cands, catalog, templates, 0, False)
        for item, title in BRACKET_TITLES[name].items():
            if item in sample.history:
                assert title + prompts.HISTORY_TAG in hybrid_text and title + prompts.PH_TAG in text_only, where
            if item in cands:
                assert title + prompts.CANS_TAG in hybrid_text and title + prompts.PH_TAG in text_only, where


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_constructed_longest_prompt_fits(name, bundle):
    """§23.8 #9, the §9.6.9 method as ``multi/sasrec_prompts/worst_case.py`` ran it: the 30 titles with the most
    ``", title"`` tokens, the one with the longest answer as the target, 40 shuffled orderings (``random.Random(1234)``,
    the generator of that measurement) and the plain one, every template and format. The result is the §23.2 row and is
    within ``batching.max_seq_len`` with the answer and the 64 generated tokens."""
    spec, design = SPECS[name], WORST_CASE[name]
    catalog = _catalog(name)
    templates = load_templates(REPO_ROOT / spec.template_file, spec=spec)
    cfg, _ = load_config(PROFILE_DIR / f"{name}_paper_spec.yaml")
    max_len = cfg.batching.max_seq_len
    titles = catalog.titles
    pieces = [len(encode_text(bundle, ", " + t)) for t in titles]
    answers = [len(encode_text(bundle, t + "\n")) for t in titles]
    top30 = sorted(range(len(titles)), key=lambda i: (-pieces[i], i))[:30]
    target = max(top30, key=lambda i: (answers[i], -top30.index(i)))
    assert (target, answers[target]) == (design.target, design.answer)
    others = [i for i in top30 if i != target]
    rng = random.Random(1234)
    orderings = []
    for _ in range(40):
        perm = others[:]
        rng.shuffle(perm)
        cands = perm[10:29] + [target]
        rng.shuffle(cands)
        orderings.append((perm[:10], cands))
    orderings.append((others[:10], others[10:29] + [target]))
    longest = {k: 0 for k in range(len(templates.templates))}
    for history, cands in orderings:
        sample = _sample(name, tuple(history), target)
        for k in longest:
            for hybrid in (True, False):
                text = build_prompt(sample, tuple(cands), catalog, templates, k, hybrid)
                ids = encode_eval(bundle, text)
                ids_a, _ = encode_train(bundle, text, titles[target], "answer_only")
                assert len(ids_a) == len(ids) + answers[target]
                for encoded in (ids, ids_a):
                    check_token_counts(bundle, encoded, hybrid=hybrid, len_seq=10, prompt_len=len(ids) - 1, n_bos=1,
                                       max_seq_len=max_len, where=f"longest prompt template {k}")
                longest[k] = max(longest[k], len(ids))
    assert longest == design.per_template
    prompt_max = max(longest.values())
    totals = (prompt_max, prompt_max + answers[target], prompt_max + cfg.model.gen_max_new_tokens)
    assert totals == design.totals
    assert max(totals) <= max_len == 1024


# ----------------------------------------------------------------------------
# M5 batches (§23.5: the padding id and the title rule of the dataset)
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_m5_batches_follow_the_dataset(released, bundle):
    """§23.5 M5, D28, D39: ``EvalBatches`` and ``PlannedTrainBatches`` accept the catalog under the dataset of
    ``prompt.template_path`` and refuse it under ``prompt/movie.txt``; ``hist_ids`` is padded with the ``num_items`` of
    the samples (§5.6.13) and the candidates stay below it."""
    spec, cfg, full, templates = released.spec, released.cfg, released.full, released.templates
    num_items, seed = spec.num_items, cfg.run.seed
    cand_fn = functools.partial(candidates_for, main_seed=seed, cfg=cfg.candidates)
    short_val = tuple(s for s in full.val if len(s.history) < 10)[:8]              # short histories show the padding
    short_train = tuple(s for s in full.train if len(s.history) < 10)[:64]
    assert len(short_val) == 8 and len(short_train) == 64
    plan = build_batch_plan(short_train, 0, cfg)
    eval_set = EvalBatches(short_val, len(short_val), full.catalog, templates, cand_fn, bundle, cfg.prompt,
                           cfg.batching, seed)
    train_set = PlannedTrainBatches(plan, short_train, full.catalog, templates, cand_fn, bundle, cfg.prompt,
                                    cfg.batching, seed)
    first_train = [short_train[i] for i in plan.entries[0].sample_indices]
    for batch, samples in ((eval_set[0], short_val), (train_set[0], first_train)):
        assert tuple(batch.hist_ids.shape) == (len(samples), 10)
        assert tuple(batch.cand_ids.shape) == (len(samples), 20) and int(batch.cand_ids.max()) < num_items
        assert batch.sample_ids == [s.sample_id for s in samples]
        for j, s in enumerate(samples):
            n = len(s.history)
            assert batch.hist_ids[j, :n].tolist() == list(s.history), s.sample_id
            assert batch.hist_ids[j, n:].tolist() == [num_items] * (10 - n) and n < 10, s.sample_id
            assert int(batch.len_seq[j]) == n, s.sample_id
    wrong = dataclasses.replace(cfg.prompt, template_path=MOVIELENS.template_file)
    with pytest.raises(PromptError, match="catalog title"):
        EvalBatches(short_val, len(short_val), full.catalog, templates, cand_fn, bundle, wrong, cfg.batching, seed)
    with pytest.raises(PromptError, match="catalog title"):
        PlannedTrainBatches(plan, short_train, full.catalog, templates, cand_fn, bundle, wrong, cfg.batching, seed)
