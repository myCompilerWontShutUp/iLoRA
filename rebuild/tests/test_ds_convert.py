"""M3 변환 도구 시험: Steam, LastFM (DESIGN.md §23.5 M3, §23.8 #10, §7.6.1, §7.6.2; D28, D36, D42).

``tools/convert_sasrec.py --dataset <d>`` runs once per dataset into a tmp folder (a subprocess, as the
``converted_sasrec`` fixture of §17.6.2); without git or the full history the tests skip with ``SKIP_NO_GIT_HISTORY``
(``git_objects_dir`` of conftest). The §23.2 numbers are written out here so that the tool's table is checked against
them: the digest, 20 kept float32 tensors with ``item_embeddings [item_num + 1, 64]``, ``s_fc`` dropped, the manifest
counts, and the encoder built from the produced files through ``from_config`` with the released catalog equals the
original-code reference bit for bit on CPU (``verify_against_reference``). The table of the tool and the int-row
reading of step 9 are tested on synthetic frames without git; the MovieLens default path stays in the unchanged
``test_m3_convert.py``. Tests that run the tool carry ``slow`` themselves (§17.6.3: pytest 9 ignores marks on fixtures).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from ilora import data, run, sasrec
from ilora.config import load_config
from ilora.data import HISTORY_LEN, ItemCatalog, pad_histories
from ilora.datasets import SPECS
from ilora.sasrec import SASRecArch, SASRecEncoder, SASRecError

# torch 2.0.0's own loader warns that TypedStorage is deprecated whenever a file is read
pytestmark = pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
TOOL_PATH = REBUILD_DIR / "tools" / "convert_sasrec.py"
PROFILE_DIR = REBUILD_DIR / "configs" / "datasets"
NAMES = ("steam", "lastfm")
SPLITS = (("train", "train_data.df"), ("val", "Val_data.df"), ("test", "Test_data.df"))
# §7.6.2: 129,664 values for MovieLens, of which the item embedding [1683, 64] depends on the dataset
OTHER_TENSOR_VALUES = 129_664 - 1683 * 64

# §23.2 (입력 해시와 기대값 표, 아이템 수, 표본 수; multi/sasrec_prompts/convert_both.py): pickle blob and sha256, state
# digest, item count and the sample counts per split (D36, D42).
DESIGN = {
    "steam": SimpleNamespace(
        pickle="rec_model/steam.pt", blob="db78403205868f67c8214820ac9dd05c3f44205d",
        sha256="752c0cd11a8646adca10025b5746dfcb8968967a6104546f6f869a3ebf26609b",
        digest="d5d13fc0bc0f3c8ebc2f64e4247e9da3b486e6984c931bb8454ebbb17c8b3932",
        item_num=3581, samples={"train": 151048, "val": 1184, "test": 1186}),
    "lastfm": SimpleNamespace(
        pickle="rec_model/lastfm.pt", blob="4cad4b22cdcbf3e7cc959409af803f5d975a9941",
        sha256="6f94253543e0ca3b0bbd42b4f6b94f2c97c9fdfdd36c3b5ad3f6b576a806dd86",
        digest="6b0d161351901a272fe07ef4c2bf3ccc99e06a487bcaee1c9b88392f59435d85",
        item_num=4606, samples={"train": 47872, "val": 122, "test": 122}),
}


def _state_shapes(item_num: int) -> dict[str, tuple[int, ...]]:
    """§7.6.2: the 20 kept tensors; only ``item_embeddings`` depends on the dataset (``[item_num + 1, 64]``)."""
    return {
        "item_embeddings.weight": (item_num + 1, 64), "positional_embeddings.weight": (10, 64),
        "ln_1.weight": (64,), "ln_1.bias": (64,), "ln_2.weight": (64,), "ln_2.bias": (64,),
        "ln_3.weight": (64,), "ln_3.bias": (64,),
        "mh_attn.linear_q.weight": (64, 64), "mh_attn.linear_q.bias": (64,),
        "mh_attn.linear_k.weight": (64, 64), "mh_attn.linear_k.bias": (64,),
        "mh_attn.linear_v.weight": (64, 64), "mh_attn.linear_v.bias": (64,),
        "feed_forward.w_1.weight": (64, 64, 1), "feed_forward.w_1.bias": (64,),
        "feed_forward.w_2.weight": (64, 64, 1), "feed_forward.w_2.bias": (64,),
        "feed_forward.layer_norm.weight": (64,), "feed_forward.layer_norm.bias": (64,),
    }


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fresh import of ``rebuild/tools/convert_sasrec.py`` (constants can be monkeypatched per test)."""
    spec = importlib.util.spec_from_file_location("convert_sasrec_ds_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=NAMES)
def converted(request: pytest.FixtureRequest, git_objects_dir: Path,
              tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One run of ``tools/convert_sasrec.py --repo <repo_root> --dataset <d> --out <tmp>`` (§7.6.1, §23.6 DS1), in a
    subprocess like ``converted_sasrec``; ``git_objects_dir`` skips without git or full history. Tests mark slow."""
    name = request.param
    out_dir = tmp_path_factory.mktemp(f"converted_{name}") / f"sasrec_{name}"
    proc = subprocess.run([sys.executable, "-I", str(TOOL_PATH), "--repo", str(REPO_ROOT), "--dataset", name,
                           "--out", str(out_dir)], capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        pytest.fail(f"convert_sasrec.py --dataset {name} exited with {proc.returncode}\n{proc.stdout[-4000:]}\n"
                    f"{proc.stderr[-4000:]}", pytrace=False)
    return SimpleNamespace(name=name, design=DESIGN[name], out_dir=out_dir, stdout=proc.stdout,
                           state_path=out_dir / f"sasrec_{name}_state.pt",
                           reference_path=out_dir / f"sasrec_{name}_reference.pt",
                           manifest=json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=None)
def _released(name: str) -> SimpleNamespace:
    """``load_dataset`` of ``<name>_paper_spec`` on the released files (M1, sha256 of the four files first), once."""
    cfg, _ = load_config(PROFILE_DIR / f"{name}_paper_spec.yaml")
    return SimpleNamespace(cfg=cfg, full=data.load_dataset(cfg, REPO_ROOT))


# ----------------------------------------------------------------------------
# One real conversion per dataset (§23.8 #10, §7.6.2, §7.6.1 steps 7-12)
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_state_file(converted):
    """§7.6.2 with the sizes of the dataset: 20 float32 tensors, ``item_embeddings [item_num + 1, 64]``, ``s_fc`` dropped,
    the §23.2 digest and the pinned pickle as ``source``; the runtime reader accepts the file (§7.6.3)."""
    design = converted.design
    assert sorted(p.name for p in converted.out_dir.iterdir()) == sorted(
        ["manifest.json", f"sasrec_{converted.name}_reference.pt", f"sasrec_{converted.name}_state.pt"])
    obj = torch.load(converted.state_path, map_location="cpu", weights_only=True)
    assert set(obj) == set(sasrec.STATE_FILE_KEYS) and obj["format"] == sasrec.SASREC_STATE_FORMAT
    state = obj["state_dict"]
    assert {key: tuple(value.shape) for key, value in state.items()} == _state_shapes(design.item_num)
    assert len(state) == 20 and all(value.dtype == torch.float32 for value in state.values())
    assert sum(value.numel() for value in state.values()) == (design.item_num + 1) * 64 + OTHER_TENSOR_VALUES
    assert obj["dropped_keys"] == ["s_fc.bias", "s_fc.weight"] and not any("s_fc" in key for key in state)
    assert obj["digest"] == design.digest == sasrec.state_digest(state)
    arch = obj["arch"]
    assert arch == {"hidden_size": 64, "item_num": design.item_num, "state_size": 10, "num_heads": 1,
                    "layernorm_eps": 1e-5}
    assert all(type(arch[key]) is int for key in ("hidden_size", "item_num", "state_size", "num_heads"))
    assert type(arch["layernorm_eps"]) is float
    assert obj["source"] == {"path": design.pickle, "git_blob": design.blob, "sha256": design.sha256}
    loaded_arch, loaded, digest = sasrec.load_sasrec_state(converted.state_path, design.digest)   # the runtime reader
    assert loaded_arch == SASRecArch(64, design.item_num, 10, 1, 1e-5) and digest == design.digest
    assert all(torch.equal(loaded[key], state[key]) for key in state)
    assert f"wrote sasrec_{converted.name}_state.pt, sasrec_{converted.name}_reference.pt" in converted.stdout


@pytest.mark.slow
def test_manifest(converted):
    """§7.6.1 step 12 and the §23.2 counts: reference rows and item counts, dropped ``s_fc`` shapes, the 22 tensors of the
    original module, the 18 globals and the file hashes of the dataset (``run.expected_inputs``)."""
    design, manifest, name = converted.design, converted.manifest, converted.name
    total = sum(design.samples.values())
    assert manifest["reference"]["counts"] == design.samples and manifest["reference"]["total"] == total
    assert manifest["reference"]["z_shape"] == [total, 64]
    assert manifest["reference"]["item_emb_shape"] == [design.item_num, 64]
    state = manifest["state"]
    assert state["digest"] == design.digest and state["n_tensors"] == 20
    assert state["n_values"] == (design.item_num + 1) * 64 + OTHER_TENSOR_VALUES
    assert state["dropped"] == {"s_fc.bias": [design.item_num], "s_fc.weight": [design.item_num, 64]}
    assert manifest["arch"]["item_num"] == design.item_num
    observed = manifest["observed"]
    assert observed["attributes"] == {"device": "cuda:0", "hidden_size": {"type": "int", "value": 64},
                                      "item_num": {"type": "int", "value": design.item_num},
                                      "state_size": {"type": "int64", "value": 10}, "training": True}
    shapes = {s["name"]: s["shape"] for s in observed["state"]}
    assert len(shapes) == 22 == len(observed["state"]) and shapes["item_embeddings.weight"] == [design.item_num + 1, 64]
    assert shapes["s_fc.weight"] == [design.item_num, 64] and shapes["s_fc.bias"] == [design.item_num]
    assert manifest["pickle"]["path"] == design.pickle and manifest["pickle"]["git_blob"] == design.blob
    assert manifest["pickle"]["sha256"] == design.sha256 and len(manifest["pickle"]["globals"]) == 18
    assert manifest["pickle"]["size"] == (REPO_ROOT / design.pickle).stat().st_size
    table = run.expected_inputs(name)
    data_dir = SPECS[name].data_dir
    assert manifest["data"] == {f"{data_dir}/{file}": table[f"{data_dir}/{file}"][0] for _, file in SPLITS}
    assert manifest["outputs"] == {f"sasrec_{name}_state.pt": run.content_sha256(converted.state_path),
                                   f"sasrec_{name}_reference.pt": run.content_sha256(converted.reference_path)}
    check = manifest["self_check"]
    assert check["repeat"] == {"bitwise": True, "max_abs_diff": 0.0}
    assert check["padding_content"]["bitwise"] is True and check["padding_content"]["max_abs_diff"] == 0.0


@pytest.mark.slow
def test_reference_file(converted):
    """§7.6.1 step 12 and D42: every sample with ``len_seq >= 3`` in row order, right padded with the dataset's padding
    id; ``z`` and the item rows of the original code; the keys read by ``verify_against_reference``."""
    design = converted.design
    ref = torch.load(converted.reference_path, map_location="cpu", weights_only=True)
    assert set(ref) == set(sasrec.REFERENCE_FILE_KEYS) and ref["format"] == sasrec.SASREC_REFERENCE_FORMAT
    assert ref["state_digest"] == design.digest and (ref["chunk"], ref["num_threads"]) == (4096, 1)
    total = sum(design.samples.values())
    ids = ref["sample_ids"]
    assert len(ids) == total == len(set(ids))
    assert all(re.fullmatch(r"(train|val|test):(0|[1-9][0-9]*)", sid) for sid in ids)
    assert [sid.split(":")[0] for sid in ids] == [s for s, _ in SPLITS for _ in range(design.samples[s])]
    assert ref["hist_ids"].shape == (total, 10) and ref["hist_ids"].dtype == torch.int64
    assert ref["len_seq"].shape == (total,) and ref["len_seq"].dtype == torch.int64
    assert int(ref["len_seq"].min()) == 3 and int(ref["len_seq"].max()) == 10
    real = torch.arange(10)[None, :] < ref["len_seq"][:, None]
    assert bool(((ref["hist_ids"] != design.item_num) == real).all())       # right padding with the dataset's padding id
    assert int(ref["hist_ids"].max()) == design.item_num and int(ref["hist_ids"].min()) >= 0
    assert ref["z"].shape == (total, 64) and ref["z"].dtype == torch.float32 and bool(torch.isfinite(ref["z"]).all())
    assert torch.equal(ref["item_ids"], torch.arange(design.item_num, dtype=torch.int64))
    state = torch.load(converted.state_path, map_location="cpu", weights_only=True)["state_dict"]
    assert torch.equal(ref["item_emb"], state["item_embeddings.weight"][:design.item_num])   # cacu_x = lookup (U :219-222)


@pytest.mark.slow
def test_reference_rows_are_the_samples_of_m1(converted):
    """Step 9 reads the released frames with pandas, independent of M1 (§7.6.1): the reference ids and padded rows equal
    M1 ``load_dataset`` and ``pad_histories`` on the same files, so the reference covers exactly the run's samples (D42)."""
    full = _released(converted.name).full
    ref = torch.load(converted.reference_path, map_location="cpu", weights_only=True)
    samples = [sample for split, _ in SPLITS for sample in full.samples(split)]
    assert ref["sample_ids"] == [sample.sample_id for sample in samples]
    hist_ids, len_seq = pad_histories([sample.history for sample in samples], converted.design.item_num, HISTORY_LEN)
    assert torch.equal(hist_ids, ref["hist_ids"]) and torch.equal(len_seq, ref["len_seq"])


@pytest.mark.slow
def test_encoder_from_the_produced_files_equals_the_reference(converted):
    """§23.8 #10: the encoder built from the produced state with the catalog of the dataset (``from_config``, §7.6.6)
    equals the original-code reference on CPU bit for bit; a catalog of another dataset is refused."""
    design, name = converted.design, converted.name
    cfg, _ = load_config(PROFILE_DIR / f"{name}_paper_spec.yaml",
                         [f"sasrec.state_path={json.dumps(converted.state_path.as_posix())}",
                          f"sasrec.reference_path={json.dumps(converted.reference_path.as_posix())}"])
    assert cfg.sasrec.state_digest == design.digest and cfg.sasrec.source_sha256 == design.sha256     # 프로파일 (§23.6)
    catalog = _released(name).full.catalog
    assert catalog.num_items == catalog.pad_id == design.item_num
    encoder = SASRecEncoder.from_config(cfg, catalog, HISTORY_LEN)
    assert (encoder.item_num, encoder.state_digest) == (design.item_num, design.digest)
    report = sasrec.verify_against_reference(encoder, converted.reference_path, torch.device("cpu"))
    assert report["passed"] is True and report["bitwise"] is True
    assert report["max_abs_diff"] == 0.0 and report["tolerance"] == 0.0
    assert report["n"] == sum(design.samples.values()) and report["state_digest"] == design.digest
    assert {split: values["n"] for split, values in report["splits"].items()} == design.samples
    for split, values in report["splits"].items():
        assert values["bitwise"] and values["padding_matches"] and values["max_abs_diff"] == 0.0, split
    assert report["item_emb"]["bitwise"] and report["item_emb"]["n"] == design.item_num
    other = ItemCatalog(titles=tuple(f"item {i}" for i in range(1682)), title_mode="code", num_items=1682, pad_id=1682)
    with pytest.raises(SASRecError, match=f"catalog.num_items 1682 != SASRec item_num {design.item_num}"):
        SASRecEncoder.from_config(cfg, other, HISTORY_LEN)


@pytest.mark.slow
def test_step_7_compares_with_the_table_of_the_dataset(tool, converted):
    """§7.6.1 step 7: the observation of the real pickle passes for its own dataset and is refused for another one
    (``item_num`` and the shapes that follow it), also for the default MovieLens path."""
    observed = converted.manifest["observed"]
    tool.check_observed(observed, dataset=converted.name)
    other = "lastfm" if converted.name == "steam" else "steam"
    for wrong in (other, "movielens"):
        with pytest.raises(tool.ConversionError, match="attribute item_num") as caught:
            tool.check_observed(observed, dataset=wrong)
        assert "state tensors" in str(caught.value) and "item_embeddings" in str(caught.value)
    with pytest.raises(tool.ConversionError, match="attribute item_num"):
        tool.check_observed(observed)                                       # default = movielens (§23.5 M3)


# ----------------------------------------------------------------------------
# The per-dataset table and the command line (no git, no conversion)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_table_row_of_the_dataset(tool, name):
    """§23.5 M3: the row equals §23.2 and ``datasets.py`` (D28): pickle, blob, sha256 (= the M0 input table), data folder,
    sample counts, no ratings, ``SASRecArch`` with the dataset's ``item_num``, the ``s_fc`` shapes and the output names."""
    design, spec, pins = DESIGN[name], SPECS[name], tool.dataset_pins(name)
    assert (pins.pickle_path, pins.pickle_blob, pins.pickle_sha256) == (design.pickle, design.blob, design.sha256)
    assert pins.pickle_path == spec.sasrec_source and pins.data_dir == spec.data_dir
    assert run.expected_inputs(name)[design.pickle] == (design.sha256, False)
    assert dict(pins.expected_samples) == design.samples == dict(spec.expected_samples)
    assert pins.has_ratings is False and spec.has_ratings is False
    assert pins.arch == SASRecArch(64, design.item_num, 10, 1, 1e-5) and pins.arch.item_num == spec.num_items
    assert pins.attributes == {**tool.EXPECTED_ATTRIBUTES, "item_num": {"type": "int", "value": design.item_num}}
    assert dict(pins.dropped_shapes) == {"s_fc.bias": [design.item_num], "s_fc.weight": [design.item_num, 64]}
    assert pins.dropped_keys == ("s_fc.bias", "s_fc.weight")
    assert (pins.state_file, pins.reference_file) == (f"sasrec_{name}_state.pt", f"sasrec_{name}_reference.pt")


def test_movielens_row_is_the_module_constants(tool, monkeypatch):
    """§23.5 M3: ``movielens`` is the default dataset and its row is the module constants, read at call time, so the
    monkeypatches of tests/test_m3_convert.py (PICKLE_BLOB, PICKLE_SHA256) reach steps 3-9 as before."""
    pins = tool.dataset_pins()
    assert pins == tool.dataset_pins("movielens")
    assert (pins.pickle_path, pins.pickle_blob, pins.pickle_sha256, pins.data_dir) == (
        tool.PICKLE_PATH, tool.PICKLE_BLOB, tool.PICKLE_SHA256, tool.DATA_DIR)
    assert pins.expected_samples is tool.EXPECTED_SAMPLES and pins.arch is tool.EXPECTED_ARCH
    assert pins.attributes is tool.EXPECTED_ATTRIBUTES and pins.dropped_shapes is tool.DROPPED_SHAPES
    assert pins.dropped_keys == tool.DROPPED_KEYS and (pins.state_file, pins.reference_file) == (
        tool.STATE_FILE, tool.REFERENCE_FILE)
    assert pins.has_ratings is True
    monkeypatch.setattr(tool, "PICKLE_BLOB", "a" * 40)
    monkeypatch.setattr(tool, "PICKLE_SHA256", "b" * 64)
    assert (tool.dataset_pins().pickle_blob, tool.dataset_pins().pickle_sha256) == ("a" * 40, "b" * 64)
    with pytest.raises(tool.ConversionError, match="unknown dataset 'goodreads'"):
        tool.dataset_pins("goodreads")


def test_command_line_option(tool):
    """``--dataset`` (default movielens, choices of ``datasets.SPECS``); the MovieLens command line is unchanged."""
    parser = tool.build_parser()
    assert parser.parse_args(["--out", "folder"]).dataset == "movielens"
    assert parser.parse_args(["--dataset", "steam", "--out", "folder"]).dataset == "steam"
    choices = next(a for a in parser._actions if a.dest == "dataset").choices
    assert choices == ("movielens", "steam", "lastfm") == tuple(SPECS)
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "goodreads", "--out", "folder"])


def test_step_3_pins_the_pickle_of_the_dataset(tool):
    """§7.6.1 step 3: blob and sha256 of the dataset's pickle pass; a pickle of another dataset is refused."""
    for name, other in (("steam", "lastfm"), ("lastfm", "steam")):
        design = DESIGN[name]
        path = REPO_ROOT / design.pickle
        assert tool.check_pickle_file(path, dataset=name) == {"path": design.pickle, "git_blob": design.blob,
                                                              "sha256": design.sha256, "size": path.stat().st_size}
        with pytest.raises(tool.ConversionError, match=f"differ from the pinned blob {DESIGN[other].blob}"):
            tool.check_pickle_file(path, dataset=other)
        with pytest.raises(tool.ConversionError, match=f"differ from the pinned blob {tool.PICKLE_BLOB}"):
            tool.check_pickle_file(path)                                    # default = movielens


# ----------------------------------------------------------------------------
# Step 9 on synthetic frames (§7.6.1 step 9, §23.2 행 형식)
# ----------------------------------------------------------------------------
Rows = Mapping[str, Sequence[tuple[list, int]]]


def _seq(items: list[int], pad: int) -> list[int]:
    """A Steam/LastFM ``seq``: plain ids right-padded to 10 entries with the padding id."""
    return items + [pad] * (10 - len(items))


@pytest.fixture
def frames(tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, Rows], Path]:
    """Factory ``frames(dataset, rows) -> folder``: pickled frames of ``(seq, len_seq)`` rows in tmp_path/frames, accepted by
    step 9 of ``dataset``: their hashes replace the entries of the M0 table and the expected counts become the kept rows."""
    tables = {"steam": run.EXPECTED_INPUTS_STEAM, "lastfm": run.EXPECTED_INPUTS_LASTFM, "movielens": run.EXPECTED_INPUTS}
    original = tool.dataset_pins

    def make(dataset: str, rows: Rows) -> Path:
        folder = tmp_path / "frames"
        folder.mkdir()
        counts: dict[str, int] = {}
        for split, file_name in SPLITS:
            split_rows = rows[split]
            pd.DataFrame({"seq": [row[0] for row in split_rows], "next": [7] * len(split_rows),
                          "len_seq": [row[1] for row in split_rows]}).to_pickle(folder / file_name)
            key = f"{SPECS[dataset].data_dir}/{file_name}"
            monkeypatch.setitem(tables[dataset], key, (run.content_sha256(folder / file_name), False))
            counts[split] = sum(1 for _, length in split_rows if length >= 3)
        pins = dataclasses.replace(original(dataset), expected_samples=counts)
        monkeypatch.setattr(tool, "dataset_pins", lambda name="movielens": pins if name == dataset else original(name))
        return folder

    return make


def test_step_9_reads_plain_int_rows(tool, frames):
    """§23.2 행 형식 (no ratings): ``seq`` = 10 plain ids, rows with ``len_seq < 3`` left out, ``sample_id`` keeps the row of
    the file and the padding id is the dataset's (§7.6.1 step 9)."""
    pad = DESIGN["steam"].item_num
    folder = frames("steam", {"train": [(_seq([1, 2, 3], pad), 3), (_seq([4, 5], pad), 2), (list(range(10, 20)), 10)],
                              "val": [(_seq([pad - 1, np.int64(6), 7, 8], pad), 4)],       # numpy ints are integers too
                              "test": [(_seq([9, 10, 11], pad), 3), (_seq([12], pad), 1)]})
    out = tool.read_reference_samples(folder, dataset="steam")
    ids, hist, lens = out["train"]
    assert ids == ["train:0", "train:2"] and lens.tolist() == [3, 10] and lens.dtype == torch.int64
    assert hist.tolist() == [_seq([1, 2, 3], pad), list(range(10, 20))] and hist.dtype == torch.int64
    assert out["val"][0] == ["val:0"] and out["val"][1].tolist() == [_seq([pad - 1, 6, 7, 8], pad)]
    assert out["test"][0] == ["test:0"] and out["test"][2].tolist() == [3]


STEP_9_REFUSALS = {
    "pair_entries": ("expected a plain integer item id", [(1, 5), (2, 5), (3, 5)] + [(3581, 0)] * 7, 3),
    "bool_entry": ("expected a plain integer item id", [True, 2, 3] + [3581] * 7, 3),
    "movielens_padding": ("is not right-padded with 3581", [1, 2, 3] + [1682] * 7, 3),
    "item_above_the_catalog": ("is not right-padded with 3581", [3581, 2, 3] + [3581] * 7, 3),
    "short_seq": ("seq has 3 entries and len_seq 3, expected 10", [1, 2, 3], 3),
}


@pytest.mark.parametrize("case", sorted(STEP_9_REFUSALS))
def test_step_9_refuses_bad_rows(case, tool, frames):
    """A wrong layout of the int rows is a refusal naming the row (no silent reinterpretation): a rating pair or a bool in
    a layout without ratings, the MovieLens padding id, an id at or above the item count, a short ``seq``."""
    message, seq, length = STEP_9_REFUSALS[case]
    pad = DESIGN["steam"].item_num
    good = (_seq([1, 2, 3], pad), 3)
    folder = frames("steam", {"train": [good, (seq, length)], "val": [good], "test": [good]})
    with pytest.raises(tool.ConversionError, match=re.escape(message)) as caught:
        tool.read_reference_samples(folder, dataset="steam")
    assert "train row 1" in str(caught.value)


def test_step_9_refuses_wrong_counts_and_changed_files(tool, frames, monkeypatch):
    """The kept rows must equal the §23.2 counts, and a frame whose sha256 differs from ``run.expected_inputs`` is not
    unpickled (§7.6.1 step 9)."""
    pad = DESIGN["lastfm"].item_num
    good = (_seq([1, 2, 3], pad), 3)
    folder = frames("lastfm", {"train": [good], "val": [good], "test": [good]})
    pins = tool.dataset_pins("lastfm")
    monkeypatch.setattr(tool, "dataset_pins", lambda name="movielens": dataclasses.replace(
        pins, expected_samples={**pins.expected_samples, "val": 2}))
    with pytest.raises(tool.ConversionError, match="val: 1 samples with len_seq >= 3, expected 2"):
        tool.read_reference_samples(folder, dataset="lastfm")
    (folder / "Test_data.df").write_bytes(b"not a frame")
    monkeypatch.setattr(tool, "dataset_pins", lambda name="movielens": pins)
    with pytest.raises(tool.ConversionError, match=r"\(not unpickled\)"):
        tool.read_reference_samples(folder, dataset="lastfm")


def test_step_9_keeps_the_movielens_pair_rows(tool, frames):
    """§7.6.1 step 9 unchanged for the default dataset: item ids are the first element of ``(item, rating)`` pairs, the
    padding id is 1682 and the hashes come from ``run.EXPECTED_INPUTS`` (§23.5 M3)."""
    pair_seq = [(10, 4), (11, 3), (12, 5)] + [(1682, 0)] * 7
    short = [(1, 5), (2, 5)] + [(1682, 0)] * 8
    folder = frames("movielens", {"train": [(pair_seq, 3), (short, 2)], "val": [(pair_seq, 3)], "test": [(pair_seq, 3)]})
    out = tool.read_reference_samples(folder)
    assert out["train"][0] == ["train:0"] and out["train"][1].tolist() == [[10, 11, 12] + [1682] * 7]
    assert out["val"][2].tolist() == [3] and out["test"][0] == ["test:0"]


# ----------------------------------------------------------------------------
# encode_all pads with the item count of the checkpoint (§23.5 M3, §23.8)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("item_num", (3581, 4606))
def test_encode_all_uses_the_padding_of_the_dataset(item_num):
    """§23.5 M3: ``encode_all`` pads with the encoder's ``item_num`` (= padding id of the dataset), which ``forward`` demands:
    a row padded with the MovieLens id 1682 is refused (§7.6.4). Placeholder weights (§7.6.3, nothing is compared to them)."""
    encoder = SASRecEncoder(SASRecArch(64, item_num, 10, 1, 1e-5), "0" * 64)
    history = [item_num - 1, 7, 11]                                     # the last catalog id is a real item
    z = encoder.encode_all([history], chunk=2, num_threads=1)
    padded = torch.tensor([history + [item_num] * 7], dtype=torch.int64)
    assert z.shape == (1, 64) and torch.equal(z, encoder(padded, torch.tensor([3], dtype=torch.int64)))
    wrong = torch.tensor([history + [1682] * 7], dtype=torch.int64)
    with pytest.raises(ValueError, match=f"padding id {item_num}"):
        encoder(wrong, torch.tensor([3], dtype=torch.int64))
