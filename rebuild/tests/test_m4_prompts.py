"""M4 prompt와 curriculum 시험 (DESIGN.md §8.8 #1-#7, #9; §17.6.9 M4 표).

#8 (worker counts) is ``test_workers.py::test_train_batches_identical_across_workers``. #9 is the
VESSL CPU bundle re-running the #6 and #7 tests on Linux (they carry ``criterion("M4-9")``).
Golden values are the §8.6.2-§8.6.6 numbers (seed 1234, numpy 1.24.2, torch 2.0.0 CPU).
"""
from __future__ import annotations

import copy
import dataclasses
import functools
from collections import Counter

import pytest

import refs_upstream
from ilora import prompts, run, seeding
from ilora.candidates import candidates_for
from ilora.config import ConfigError
from ilora.data import ItemCatalog
from ilora.prompts import (BatchPlan, PlanEntry, PromptError, TemplateSet, build_batch_plan, build_prompt,
                           check_catalog_titles, choose_template, curriculum_hybrid, eval_is_hybrid,
                           load_templates, make_batch_plan, parse_templates, render_prompt)
from ilora.sections import PromptConfig

# §8.6.2 step 6 (measured): the three stripped lines of prompt/movie.txt
TEMPLATES = (
    "This user has watched [HistoryHere] in the previous. Please predict the next movie this user will watch. "
    "Choose the answer from the following 20 movie titles: [CansHere]. Answer:",
    "This user has watched [HistoryHere] in the previous. Given the following 20 movie titles: [CansHere], "
    "recommend one movie for this user to watch next. The movie title you recommend is:",
    "The visit history of this user is: [HistoryHere]. Recommend a next movie for this user to watch from the "
    "following movie title set: [CansHere]. The recommendation should contain one movie title only. "
    "Recommendation:",
)
TEMPLATE_SHA256 = "957fc1f27bb930ca5acbe1b545186ecc04fd93ea2c5307644c706ad44395ecd3"
# §8.6.3 template counts (seed 1234)
TRAIN_E0_COUNTS = [22033, 22031, 22062]
VAL_COUNTS = [30, 33, 31]
TEST_COUNTS = [22, 36, 37]
SAME_TEMPLATE_E0_E1 = 0.3351
# §8.6.5 curriculum (seed 1234, paper_spec)
NB, EPOCHS = 8265, 5
T = EPOCHS * NB                                    # 41,325
CURRICULUM_COUNTS = {"linear": [846, 2550, 4076, 5766, 7436], "reverse": [7442, 5764, 4100, 2522, 823],
                     "all_hybrid": [NB] * EPOCHS}
EXPECTED_SUMS = [826.6, 2479.6, 4132.6, 5785.6, 7438.6]   # sum of tau/T per epoch (M10 §14.6.4 log values)
FIRST_HYBRID_TAU = 432
U_TAU_1, U_TAU_T = 0.260466, 0.615392
# §8.6.6 batch plan (seed 1234, N 66,126, B 8, E 5, linear)
FIRST_BATCH = (57248, 62110, 9506, 49288, 37323, 59238, 25930, 18480)
DIGESTS = (
    "206691cdaee8a823141bb6e99c5429643148e4f43edd4a8e0c136231a385a589",
    "8ba8ff1c0cb65cadbe43d95c846328443eafd3e60fe20044acc0f6469b7e5d5e",
    "b470b7450f1c8f87074b4b883932b0f7eb474211daad820fd073e84096a644dc",
    "c5fb0b531aa98d4572b07587ea83c080ec61eb17ce2260fa2c9f7047ed6a2ff8",
    "945a6f7b8500cef6b852b7b85e879456bf14b9ee526f4c4101e87e0af1d9ec5f",
)
N_TRAIN, MICRO = 66126, 8


def _lf_bytes(repo_root) -> bytes:
    return (repo_root / prompts.TEMPLATE_FILE).read_bytes().replace(b"\r\n", b"\n")


def _prompt_cfg(choice: str = "random", index=None, **kw) -> PromptConfig:
    values = dict(template_path=prompts.TEMPLATE_FILE, template_choice=choice, fixed_template_index=index,
                  train_curriculum="linear", eval_prompt="hybrid")
    values.update(kw)
    return PromptConfig(**values)


@functools.lru_cache(maxsize=None)
def _curriculum(mode: str) -> tuple[bool, ...]:
    """curriculum_hybrid for tau = 1..T (cached across the parametrized cases)."""
    return tuple(curriculum_hybrid(mode, 1234, tau, T) for tau in range(1, T + 1))


# ----------------------------------------------------------------------------
# M4-1 templates
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M4-1")
def test_templates_exact(repo_root, templates):
    loaded = load_templates(repo_root / "prompt/movie.txt")
    assert loaded.templates == TEMPLATES == templates.templates
    assert loaded == templates
    # five parts, history slot first in all three, rebuilt exactly
    for text, part in zip(loaded.templates, loaded.parts):
        assert part[1] == prompts.HISTORY_SLOT and part[3] == prompts.CANDS_SLOT
        assert "".join(part) == text
    assert loaded.parts[0][0] == "This user has watched " and loaded.parts[0][4] == ". Answer:"
    assert loaded.parts[2][2] == (". Recommend a next movie for this user to watch from the following movie "
                                  "title set: ")
    # the released reading (splitlines + strip, U data/data_interface.py:185-186) drops the line-end spaces
    raw_lines = _lf_bytes(repo_root).decode("utf-8").split("\n")
    assert [line.endswith(" ") for line in raw_lines] == [True, True, False]
    assert [t.split()[-1] for t in loaded.templates] == ["Answer:", "is:", "Recommendation:"]
    assert prompts.TEMPLATE_FILE in run.EXPECTED_INPUTS
    assert run.EXPECTED_INPUTS[prompts.TEMPLATE_FILE] == (TEMPLATE_SHA256, True)


@pytest.mark.criterion("M4-1")
def test_templates_crlf_invariance(repo_root, tmp_path):
    lf = _lf_bytes(repo_root)
    crlf = lf.replace(b"\n", b"\r\n")
    assert lf != crlf and b"\r" not in lf
    (tmp_path / "lf").mkdir()
    (tmp_path / "crlf").mkdir()
    lf_path, crlf_path = tmp_path / "lf" / "movie.txt", tmp_path / "crlf" / "movie.txt"
    lf_path.write_bytes(lf)
    crlf_path.write_bytes(crlf)
    assert run.content_sha256(lf_path, text=True) == run.content_sha256(crlf_path, text=True) == TEMPLATE_SHA256
    assert run.content_sha256(lf_path) != run.content_sha256(crlf_path)
    a, b = load_templates(lf_path), load_templates(crlf_path)
    assert a == b and a.templates == TEMPLATES
    assert parse_templates(crlf, source="crlf") == parse_templates(lf, source="lf") == a


# ----------------------------------------------------------------------------
# M4-2 template validation
# ----------------------------------------------------------------------------
def _with_line(lines: list[str], index: int, new: str) -> list[str]:
    out = list(lines)
    out[index] = new
    return out


PARSE_CASES = {
    "target_here": lambda ls: (_with_line(ls, 0, ls[0] + " [TargetHere]"), "line 1"),
    "no_cans": lambda ls: (_with_line(ls, 1, ls[1].replace("[CansHere]", "the titles")), "line 2"),
    "history_twice": lambda ls: (_with_line(ls, 2, ls[2] + " [HistoryHere]"), "line 3"),
    "ph": lambda ls: (_with_line(ls, 0, ls[0].replace("Answer:", "[PH] Answer:")), "line 1"),
    "empty_line": lambda ls: ([ls[0], "", ls[1], ls[2]], "line 2"),
    "unpaired_bracket": lambda ls: (_with_line(ls, 1, ls[1] + " ["), "line 2"),
    "bos_string": lambda ls: (_with_line(ls, 2, ls[2] + " <s>"), "line 3"),
}


@pytest.mark.criterion("M4-2")
@pytest.mark.parametrize("case", list(PARSE_CASES))
def test_parse_templates_rejects(case, repo_root):
    lines = _lf_bytes(repo_root).decode("utf-8").split("\n")
    assert parse_templates("\n".join(lines).encode("utf-8"), source="ok").templates == TEMPLATES
    bad_lines, where = PARSE_CASES[case](lines)
    with pytest.raises(PromptError, match=f"bad {where}"):
        parse_templates("\n".join(bad_lines).encode("utf-8"), source="bad")


@pytest.mark.criterion("M4-2")
def test_parse_templates_other_rejects():
    """No lines, non-UTF-8 bytes, other bracket groups and </s>, <unk> strings are also rejected."""
    good = TEMPLATES[0]
    for data, pattern in [(b"", "no template lines"), (b"\n\n", "empty line"),
                          (good.encode("utf-8") + b"\xff", "not UTF-8"),
                          (good.replace("Answer:", "[HistoryEmb] Answer:").encode(), "not allowed"),
                          (good.replace("Answer:", "[ItemEmb] Answer:").encode(), "not allowed"),
                          (good.replace("Answer:", "</s>").encode(), "special token"),
                          (good.replace("Answer:", "<unk>").encode(), "special token"),
                          (good.replace("[CansHere]", "[CansHere]]").encode(), "unpaired")]:
        with pytest.raises(PromptError, match=pattern):
            parse_templates(data, source="t")
    with pytest.raises(PromptError):
        parse_templates(good, source="str is not bytes")              # type: ignore[arg-type]
    # a template with the candidates first keeps its slot order
    swapped = parse_templates(b"Pick from [CansHere] after [HistoryHere] now", source="t")
    assert swapped.parts[0] == ("Pick from ", "[CansHere]", " after ", "[HistoryHere]", " now")
    assert render_prompt(swapped, 0, ["A", "B", "C"], ["X"], True) == \
        "Pick from X [CansEmb] after A [HistoryEmb], B [HistoryEmb], C [HistoryEmb] now"
    with pytest.raises(PromptError):
        TemplateSet(templates=("a [HistoryHere] b [CansHere]",), parts=(("a ", "[HistoryHere]", " b ", "[CansHere]",
                                                                         "x"),))


@pytest.mark.criterion("M4-2")
def test_load_templates_missing_or_modified(repo_root, tmp_path):
    with pytest.raises(PromptError, match="not found"):
        load_templates(tmp_path / "missing" / "movie.txt")
    raw = (repo_root / "prompt/movie.txt").read_bytes()
    changed = bytes([raw[0] ^ 0x20]) + raw[1:]                     # one byte: "T" -> "t"
    path = tmp_path / "movie.txt"
    path.write_bytes(changed)
    got = run.content_sha256(path, text=True)
    assert got != TEMPLATE_SHA256
    with pytest.raises(PromptError, match=f"{got} != expected {TEMPLATE_SHA256}"):
        load_templates(path)
    # the content is still parseable: only the hash check rejects it
    assert parse_templates(changed, source="changed").templates[0].startswith("this user")


@pytest.mark.criterion("M4-2")
def test_template_path_choice(profile_cfg):
    cfg = profile_cfg("paper_spec")
    assert cfg.prompt.template_path == prompts.TEMPLATE_FILE
    field = next(f for f in dataclasses.fields(PromptConfig) if f.name == "template_path")
    assert field.metadata["choices"] == (prompts.TEMPLATE_FILE,)
    with pytest.raises(ConfigError, match="prompt.template_path"):
        profile_cfg("paper_spec", "prompt.template_path=prompt/game.txt")


@pytest.mark.criterion("M4-2")
def test_check_catalog_titles(full_data):
    catalog = full_data.catalog
    check_catalog_titles(catalog)                                   # 0 of 1,682 (§8.6.2)
    assert catalog.titles[266] == "" and catalog.titles[1127].endswith(" ")   # allowed as they are (D20)
    for bad in ("Toy Story [HistoryEmb]", "Heat]", "[Heat", "A <s> B", "End</s>", "<unk>"):
        titles = list(catalog.titles)
        titles[5] = bad
        broken = ItemCatalog(titles=tuple(titles), title_mode="code", num_items=1682, pad_id=1682)
        with pytest.raises(PromptError, match="item 5"):
            check_catalog_titles(broken)


# ----------------------------------------------------------------------------
# M4-3 rendering equals the released replace chain
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M4-3")
def test_render_matches_code_chain(full_data, templates, paper_cfg):
    titles = full_data.catalog.titles
    n = 0
    for s in (*full_data.val, *full_data.test):
        cands = candidates_for(s, main_seed=paper_cfg.run.seed, epoch=None, cfg=paper_cfg.candidates)
        hist = [titles[i] for i in s.history]
        cand = [titles[c] for c in cands]
        for k in range(len(templates.templates)):
            for hybrid in (True, False):
                ours = render_prompt(templates, k, hist, cand, hybrid)
                assert ours == refs_upstream.code_prompt(templates.templates[k], hist, cand, hybrid), (s.sample_id, k)
                assert build_prompt(s, cands, full_data.catalog, templates, k, hybrid) == ours
                if hybrid:
                    assert (ours.count("[HistoryEmb]"), ours.count("[CansEmb]"), ours.count("[PH]")) == \
                        (len(s.history), 20, 0)
                else:
                    assert (ours.count("[HistoryEmb]"), ours.count("[CansEmb]"), ours.count("[PH]")) == \
                        (0, 0, len(s.history) + 20)
                assert "[HistoryHere]" not in ours and "[CansHere]" not in ours
                n += 1
    assert n == (94 + 95) * 3 * 2 == 1134
    # history order is Sample.history (oldest first); candidates keep the M2 order
    s = full_data.val[0]
    cands = candidates_for(s, main_seed=1234, epoch=None, cfg=paper_cfg.candidates)
    text = build_prompt(s, cands, full_data.catalog, templates, 0, True)
    hist_part = text[len(templates.parts[0][0]):text.index(templates.parts[0][2])]
    assert hist_part == ", ".join(titles[i] + " [HistoryEmb]" for i in s.history)
    with pytest.raises(PromptError):
        render_prompt(templates, 3, ["a"], ["b"], True)
    with pytest.raises(PromptError):
        render_prompt(templates, 0, ["a"], ["b"], 1)                # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# M4-4 build_prompt boundary checks
# ----------------------------------------------------------------------------
def _outside(cands: tuple[int, ...], history: tuple[int, ...]) -> int:
    return next(i for i in range(1682) if i not in cands and i not in history)


def _replace_negative(cands: tuple[int, ...], target: int, new: int) -> tuple[int, ...]:
    pos = next(i for i, c in enumerate(cands) if c != target)
    return cands[:pos] + (new,) + cands[pos + 1:]


def _drop_negative(cands: tuple[int, ...], target: int) -> tuple[int, ...]:
    pos = next(i for i, c in enumerate(cands) if c != target)
    return cands[:pos] + cands[pos + 1:]


BUILD_CASES = {
    "19_candidates": lambda c, s: _drop_negative(c, s.target),
    "21_candidates": lambda c, s: c + (_outside(c, s.history),),
    "duplicate": lambda c, s: _replace_negative(c, s.target, next(x for x in reversed(c) if x != s.target)),
    "no_target": lambda c, s: tuple(_outside(c, s.history) if x == s.target else x for x in c),
    "target_twice": lambda c, s: _replace_negative(c, s.target, s.target),
    "history_overlap": lambda c, s: _replace_negative(c, s.target, s.history[0]),
    "id_1682": lambda c, s: _replace_negative(c, s.target, 1682),
}


@pytest.mark.criterion("M4-4")
@pytest.mark.parametrize("case", list(BUILD_CASES))
def test_build_prompt_rejects(case, full_data, templates, paper_cfg):
    s = full_data.val[3]
    good = candidates_for(s, main_seed=paper_cfg.run.seed, epoch=None, cfg=paper_cfg.candidates)
    build_prompt(s, good, full_data.catalog, templates, 0, True)
    bad = BUILD_CASES[case](good, s)
    assert bad != good
    with pytest.raises(PromptError, match=s.sample_id):
        build_prompt(s, bad, full_data.catalog, templates, 0, True)


@pytest.mark.criterion("M4-4")
def test_build_prompt_rejects_history_length(full_data, templates, paper_cfg):
    """3 <= len(history) <= 10 is re-checked (M1 Sample already enforces it; bypassed here on a copy)."""
    s = full_data.val[0]
    cands = candidates_for(s, main_seed=1234, epoch=None, cfg=paper_cfg.candidates)
    for history in (s.history + (_outside(cands, s.history),), s.history[:2]):
        broken = copy.copy(s)
        object.__setattr__(broken, "history", history)
        with pytest.raises(PromptError, match="history length"):
            build_prompt(broken, cands, full_data.catalog, templates, 0, True)
    with pytest.raises(PromptError, match="candidates"):
        build_prompt(s, list(cands), full_data.catalog, templates, 0, True)   # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# M4-5 template choice
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M4-5")
def test_template_counts_seed1234(full_data, paper_cfg, profile_cfg):
    pc, seed = paper_cfg.prompt, paper_cfg.run.seed
    assert pc.template_choice == "random" and seed == 1234
    e0 = [choose_template(pc, seed, 3, s, 0) for s in full_data.train]
    e1 = [choose_template(pc, seed, 3, s, 1) for s in full_data.train]
    assert len(e0) == N_TRAIN
    assert [Counter(e0)[k] for k in range(3)] == TRAIN_E0_COUNTS
    assert round(sum(a == b for a, b in zip(e0, e1)) / N_TRAIN, 4) == SAME_TEMPLATE_E0_E1
    for split, expected in (("val", VAL_COUNTS), ("test", TEST_COUNTS)):
        got = Counter(choose_template(pc, seed, 3, s, None) for s in full_data.samples(split))
        assert [got[k] for k in range(3)] == expected, split
    # key draw: one integers(0, n) from np_rng(seed, "template", ...)
    s_train, s_val = full_data.train[10], full_data.val[5]
    assert e0[10] == int(seeding.np_rng(seed, "template", "train", s_train.row, 0).integers(0, 3))
    assert choose_template(pc, seed, 3, s_val, None) == \
        int(seeding.np_rng(seed, "template", "val", s_val.row).integers(0, 3))
    # same input, same result (no state); order of calls does not matter
    again = [choose_template(pc, seed, 3, s, 0) for s in reversed(full_data.train[:500])]
    assert again[::-1] == e0[:500]
    # variant_fixed_template: template 0 everywhere, no draw
    fixed = profile_cfg("variant_fixed_template").prompt
    assert (fixed.template_choice, fixed.fixed_template_index) == ("fixed", 0)
    assert {choose_template(fixed, seed, 3, s, 0) for s in full_data.train[:2000]} == {0}
    assert {choose_template(fixed, seed, 3, s, None) for s in (*full_data.val, *full_data.test)} == {0}


@pytest.mark.criterion("M4-5")
def test_template_choice_errors(full_data):
    s_train, s_val, s_test = full_data.train[0], full_data.val[0], full_data.test[0]
    random_cfg, fixed_cfg = _prompt_cfg(), _prompt_cfg("fixed", 0)
    for cfg in (random_cfg, fixed_cfg):                          # the same checks in both modes
        with pytest.raises(PromptError, match="needs the epoch"):
            choose_template(cfg, 1234, 3, s_train, None)
        for split_sample in (s_val, s_test):
            with pytest.raises(PromptError, match="epoch None"):
                choose_template(cfg, 1234, 3, split_sample, 0)
        for epoch in (True, -1, 1.0):
            with pytest.raises(PromptError, match="epoch"):
                choose_template(cfg, 1234, 3, s_train, epoch)    # type: ignore[arg-type]
        for n in (0, True, 3.0):
            with pytest.raises(PromptError, match="n_templates"):
                choose_template(cfg, 1234, n, s_train, 0)        # type: ignore[arg-type]
    with pytest.raises(PromptError, match="outside 0..2"):
        choose_template(_prompt_cfg("fixed", 3), 1234, 3, s_train, 0)
    with pytest.raises(PromptError, match="fixed_template_index null"):
        choose_template(_prompt_cfg("random", 0), 1234, 3, s_val, None)
    with pytest.raises(PromptError, match="int fixed_template_index"):
        choose_template(_prompt_cfg("fixed", None), 1234, 3, s_val, None)
    with pytest.raises(PromptError, match="template_choice"):
        choose_template(_prompt_cfg("first"), 1234, 3, s_val, None)
    assert choose_template(_prompt_cfg("fixed", 2), 1234, 3, s_train, 4) == 2
    assert eval_is_hybrid(_prompt_cfg()) is True
    assert eval_is_hybrid(_prompt_cfg(eval_prompt="text_only")) is False
    with pytest.raises(PromptError):
        eval_is_hybrid(_prompt_cfg(eval_prompt="mixed"))


# ----------------------------------------------------------------------------
# M4-6 curriculum (M4-9: re-run on VESSL Linux)
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M4-6", "M4-9")
@pytest.mark.parametrize("mode", ["linear", "reverse", "all_hybrid"])
def test_curriculum_counts_seed1234(mode):
    assert prompts.CURRICULA == ("linear", "all_hybrid", "reverse")
    draws = _curriculum(mode)
    per_epoch = [sum(draws[e * NB:(e + 1) * NB]) for e in range(EPOCHS)]
    assert per_epoch == CURRICULUM_COUNTS[mode]
    if mode == "linear":
        assert sum(per_epoch) == 20674
        assert draws.index(True) + 1 == FIRST_HYBRID_TAU
        assert [round(sum(tau / T for tau in range(e * NB + 1, (e + 1) * NB + 1)), 1) for e in range(EPOCHS)] \
            == EXPECTED_SUMS
    if mode == "reverse":
        assert sum(per_epoch) == 20651


@pytest.mark.criterion("M4-6")
def test_curriculum_boundaries():
    u1, u_last = seeding.uniform(1234, "curriculum", 1), seeding.uniform(1234, "curriculum", T)
    assert (round(u1, 6), round(u_last, 6)) == (U_TAU_1, U_TAU_T)
    assert curriculum_hybrid("linear", 1234, 1, T) is False          # u 0.260466 >= 1/T
    assert curriculum_hybrid("reverse", 1234, 1, T) is True
    assert curriculum_hybrid("linear", 1234, T, T) is True           # p(T) = 1 > u (D3)
    assert curriculum_hybrid("reverse", 1234, T, T) is False         # 1 - T/T = 0
    assert curriculum_hybrid("all_hybrid", 1234, 1, T) is True
    # keyed by tau: the same answer in any order, independent of earlier calls
    taus = [T, 7, 432, 1, 20000]
    assert [curriculum_hybrid("linear", 1234, t, T) for t in taus] == \
        [curriculum_hybrid("linear", 1234, t, T) for t in reversed(taus)][::-1]
    assert curriculum_hybrid("linear", 1234, 432, T) is True and curriculum_hybrid("linear", 1234, 431, T) is False
    for tau in (0, T + 1, -1, True, 1.0):
        with pytest.raises(PromptError, match="tau"):
            curriculum_hybrid("linear", 1234, tau, T)                # type: ignore[arg-type]
    for total in (0, True, -5):
        with pytest.raises(PromptError, match="total"):
            curriculum_hybrid("linear", 1234, 1, total)              # type: ignore[arg-type]
    for mode in ("step", "LINEAR", ""):
        with pytest.raises(PromptError, match="curriculum"):
            curriculum_hybrid(mode, 1234, 1, T)


# ----------------------------------------------------------------------------
# M4-7 batch plan (M4-9: re-run on VESSL Linux)
# ----------------------------------------------------------------------------
@pytest.mark.criterion("M4-7", "M4-9")
def test_batch_plan_golden():
    plans = [make_batch_plan(seed=1234, epoch=e, epochs=EPOCHS, num_samples=N_TRAIN, micro_batch_size=MICRO,
                             curriculum="linear") for e in range(EPOCHS)]
    assert prompts.batches_per_epoch(N_TRAIN, MICRO) == NB
    for e, plan in enumerate(plans):
        assert (plan.epoch, plan.batches_per_epoch, plan.total, plan.curriculum) == (e, NB, T, "linear")
        assert len(plan) == NB
        assert [x.tau for x in plan] == list(range(e * NB + 1, (e + 1) * NB + 1))
        assert all(isinstance(x, PlanEntry) and isinstance(x.hybrid, bool) and len(x.sample_indices) == MICRO
                   for x in plan)
        used = [j for x in plan for j in x.sample_indices]
        assert len(set(used)) == len(used) == 66120 and all(0 <= j < N_TRAIN for j in used)   # 6 dropped
        assert plan.digest() == DIGESTS[e]
        assert plan.hybrid_count() == CURRICULUM_COUNTS["linear"][e]
        assert [x.hybrid for x in plan] == list(_curriculum("linear")[e * NB:(e + 1) * NB])
    assert plans[0][0].sample_indices == FIRST_BATCH
    assert (plans[0][0].tau, plans[4][0].tau, plans[4][-1].tau) == (1, 33061, 41325)
    again = make_batch_plan(seed=1234, epoch=2, epochs=EPOCHS, num_samples=N_TRAIN, micro_batch_size=MICRO,
                            curriculum="linear")
    assert again == plans[2] and again.digest() == DIGESTS[2]
    # the order is randperm of torch_generator(seed, "train_order", epoch); the curriculum does not change it
    import torch
    perm = torch.randperm(N_TRAIN, generator=seeding.torch_generator(1234, "train_order", 3)).tolist()
    assert [j for x in plans[3] for j in x.sample_indices] == perm[:NB * MICRO]
    other = make_batch_plan(seed=1234, epoch=3, epochs=EPOCHS, num_samples=N_TRAIN, micro_batch_size=MICRO,
                            curriculum="reverse")
    assert [x.sample_indices for x in other] == [x.sample_indices for x in plans[3]]
    assert other.hybrid_count() == CURRICULUM_COUNTS["reverse"][3]
    # errors
    with pytest.raises(PromptError, match="train 표본 7개 < micro-batch 8"):
        make_batch_plan(seed=1234, epoch=0, epochs=EPOCHS, num_samples=7, micro_batch_size=8, curriculum="linear")
    for epoch in (EPOCHS, -1, True):
        with pytest.raises(PromptError, match="epoch"):
            make_batch_plan(seed=1234, epoch=epoch, epochs=EPOCHS, num_samples=N_TRAIN, micro_batch_size=MICRO,
                            curriculum="linear")
    with pytest.raises(PromptError, match="micro-batch size"):
        make_batch_plan(seed=1234, epoch=0, epochs=EPOCHS, num_samples=N_TRAIN, micro_batch_size=0,
                        curriculum="linear")
    with pytest.raises(PromptError, match="curriculum"):
        make_batch_plan(seed=1234, epoch=0, epochs=EPOCHS, num_samples=N_TRAIN, micro_batch_size=MICRO,
                        curriculum="cosine")
    # a hand-made plan must be consistent
    good = plans[0]
    with pytest.raises(PromptError, match="tau"):
        BatchPlan(epoch=1, batches_per_epoch=NB, total=T, curriculum="linear", entries=good.entries)
    with pytest.raises(PromptError, match="entries"):
        BatchPlan(epoch=0, batches_per_epoch=NB, total=T, curriculum="linear", entries=good.entries[:-1])
    small = make_batch_plan(seed=1234, epoch=0, epochs=1, num_samples=12, micro_batch_size=4, curriculum="linear")
    repeated = (small[0], small[1]._replace(sample_indices=small[0].sample_indices), small[2])
    with pytest.raises(PromptError, match="repeats"):
        BatchPlan(epoch=0, batches_per_epoch=3, total=3, curriculum="linear", entries=repeated)


@pytest.mark.criterion("M4-7")
def test_build_batch_plan_wrapper(full_data, paper_cfg, profile_cfg):
    assert (paper_cfg.run.seed, paper_cfg.train.epochs, paper_cfg.train.micro_batch_size,
            paper_cfg.prompt.train_curriculum) == (1234, EPOCHS, MICRO, "linear")
    for e in (0, 4):
        plan = build_batch_plan(full_data.train, e, paper_cfg)
        assert plan.digest() == DIGESTS[e]
        assert len(plan) == NB == len(plan.entries)
        assert list(plan) == list(plan.entries)
        assert plan[0] == plan.entries[0] and plan[-1] == plan.entries[-1]
        assert plan[16:32] == plan.entries[16:32] and isinstance(plan[16:32], tuple)
        assert plan[NB - 9:] == plan.entries[NB - 9:]                # M10 tail window (D24)
    assert build_batch_plan(full_data.train, 0, profile_cfg("variant_all_hybrid")).hybrid_count() == NB
    assert build_batch_plan(full_data.train, 0, profile_cfg("variant_reverse_curriculum")).hybrid_count() == 7442
    with pytest.raises(PromptError, match="epoch"):
        build_batch_plan(full_data.train, EPOCHS, paper_cfg)
    mixed = full_data.train[:100] + (full_data.val[0],) + full_data.train[100:200]
    with pytest.raises(PromptError, match="not a train sample"):
        build_batch_plan(mixed, 0, paper_cfg)
    with pytest.raises(PromptError, match="ascending"):
        build_batch_plan(tuple(reversed(full_data.train[:200])), 0, paper_cfg)
    with pytest.raises(PromptError, match="ascending"):
        build_batch_plan(full_data.train[:100] + full_data.train[99:200], 0, paper_cfg)
