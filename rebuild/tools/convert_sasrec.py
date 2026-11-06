"""SASRec pickle 변환 도구 (DESIGN.md §7.6.1, §7.6.2; M3).

    python -I rebuild/tools/convert_sasrec.py --repo <repository root> --out runs/rebuild/assets/sasrec

Converts the whole-module pickle ``rec_model/movielens.pt`` (git blob ``41c057dc``,
``U`` = ``L``) once into

- ``sasrec_movielens_state.pt``: a plain dict read with ``torch.load(weights_only=True)``
  (``format``, ``arch``, 20 float32 tensors, ``dropped_keys``, ``source``, ``digest``, §7.6.2),
- ``sasrec_movielens_reference.pt``: outputs of the ORIGINAL code, ``cacul_h`` for every
  sample with ``len_seq >= 3`` (66,315, per split in chunks of ``REF_CHUNK`` with
  ``REF_THREADS`` CPU thread) and ``cacu_x`` for every item (1,682),
- ``manifest.json``: provenance, the 18 pickle globals and the observed pickle state.

The original class code comes from pinned blobs of the upstream commit through git, never
from the working tree (whose ``SASRecModules_ori.py`` is the fork's bf16 copy, §7.5). The
pickle is scanned statically (nothing executed) before ``torch.load`` runs with an
allow-list ``find_class`` that maps ``__main__.SASRec`` to the original class. CPU only, no
random numbers. A non-empty output folder is refused (no overwrite, §4.6.3). Relative
``--out`` paths are taken from the repository root (M0 path rule, §4.6.2). ``state_digest``
and the file format constants are imported from ``ilora.sasrec`` (the runtime checks use the
same function); the reference itself is computed with the original code only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import pickle
import pickletools
import platform
import subprocess
import sys
import tempfile
import types
import warnings
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping, Optional, Sequence

REBUILD_DIR = Path(__file__).resolve().parents[1]
if str(REBUILD_DIR) not in sys.path:      # python -I puts neither the script folder nor the cwd on sys.path
    sys.path.insert(0, str(REBUILD_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from ilora import run  # noqa: E402
from ilora.sasrec import (  # noqa: E402
    REFERENCE_FILE_KEYS, SASREC_REFERENCE_FORMAT, SASREC_STATE_FORMAT, SASRecArch, expected_state_shapes,
    load_sasrec_state, state_digest,
)

# ----------------------------------------------------------------------------
# Pinned inputs (§7.6.1)
# ----------------------------------------------------------------------------
UPSTREAM_COMMIT = "28e47371192121b6ced44401d9e149fedfae13de"
ORIGINAL_BLOBS = {"SASRecModules_ori.py": "3ac3c975a58d43ee866b637ee5d4f99534974a7b",
                  "recommender/A_SASRec_final_bce_llm.py": "dc7c94999a4db5f1b97971a2913baae44687c124"}
PICKLE_BLOB = "41c057dc96df2cc6d432d7f839d98b57ac341b26"
PICKLE_SHA256 = "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9"
# Every global the pickle names (static scan, step 4). torch.FloatStorage is resolved by torch 2.0.0's
# UnpicklerWrapper before find_class, so the find_class allow-list is the other 17 (step 6).
EXPECTED_GLOBALS: frozenset[tuple[str, str]] = frozenset({
    ("__main__", "SASRec"),
    ("SASRecModules_ori", "MultiHeadAttention"),
    ("SASRecModules_ori", "PositionwiseFeedForward"),
    ("torch.nn.modules.sparse", "Embedding"),
    ("torch.nn.modules.normalization", "LayerNorm"),
    ("torch.nn.modules.linear", "Linear"),
    ("torch.nn.modules.conv", "Conv1d"),
    ("torch.nn.modules.dropout", "Dropout"),
    ("torch.nn.modules.activation", "Softmax"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch", "FloatStorage"),
    ("torch", "device"),
    ("collections", "OrderedDict"),
    ("__builtin__", "set"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"),
    ("_codecs", "encode"),
})
REF_CHUNK = 4096
REF_THREADS = 1

PICKLE_PATH = "rec_model/movielens.pt"                 # sasrec.source_path (§7.6.7)
DATA_DIR = "data/ref/movielens"                        # [코드] U train_movielens.sh:8
SPLIT_FILES = (("train", "train_data.df"), ("val", "Val_data.df"), ("test", "Test_data.df"))
ORIGINAL_MODULE = "A_SASRec_final_bce_llm"
ORIGINAL_SUBMODULE = "SASRecModules_ori"
FLOAT_STORAGE = ("torch", "FloatStorage")
STATE_FILE = "sasrec_movielens_state.pt"
REFERENCE_FILE = "sasrec_movielens_reference.pt"
MANIFEST_FILE = "manifest.json"
MIN_LEN_SEQ = 3                                        # [코드] U data/movielens_data.py:79
EXPECTED_SAMPLES = {"train": 66126, "val": 94, "test": 95}   # §7.6.1 step 9 (66,315)
PADDING_PROBE_ID = 5         # self-check: a real item id written into padded slots (step 11)

# Expected observations of the pickled module (step 7; §7.4, §7.8 #3)
EXPECTED_ARCH = SASRecArch(hidden_size=64, item_num=1682, state_size=10, num_heads=1, layernorm_eps=1e-5)
EXPECTED_MODEL_TYPE = f"{ORIGINAL_MODULE}.SASRec"
EXPECTED_ATTRIBUTES: dict[str, Any] = {
    "device": "cuda:0",                                # saved from cuda:0 (§7.5)
    "hidden_size": {"type": "int", "value": 64},
    "item_num": {"type": "int", "value": 1682},        # int(item_num), U recommender/...:152
    "state_size": {"type": "int64", "value": 10},      # numpy int64 in the pickle
    "training": True,                                  # saved in training mode (§7.5)
}
EXPECTED_SUBMODULES = {
    "dropout": "Dropout", "item_embeddings": "Embedding", "positional_embeddings": "Embedding",
    "emb_dropout": "Dropout", "ln_1": "LayerNorm", "ln_2": "LayerNorm", "ln_3": "LayerNorm",
    "mh_attn": "MultiHeadAttention", "mh_attn.linear_q": "Linear", "mh_attn.linear_k": "Linear",
    "mh_attn.linear_v": "Linear", "mh_attn.dropout": "Dropout", "mh_attn.softmax": "Softmax",
    "feed_forward": "PositionwiseFeedForward", "feed_forward.w_1": "Conv1d", "feed_forward.w_2": "Conv1d",
    "feed_forward.layer_norm": "LayerNorm", "feed_forward.dropout": "Dropout", "s_fc": "Linear",
}
DROPOUT_MODULES = ("dropout", "emb_dropout", "mh_attn.dropout", "feed_forward.dropout")
EXPECTED_DROPOUT_P = 0.1
LAYERNORM_MODULES = ("ln_1", "ln_2", "ln_3", "feed_forward.layer_norm")
EXPECTED_CONV1D = {"in_channels": 64, "out_channels": 64, "kernel_size": [1], "stride": [1], "padding": [0],
                   "dilation": [1], "groups": 1, "padding_mode": "zeros", "bias": True}
DROPPED_SHAPES = {"s_fc.bias": [1682], "s_fc.weight": [1682, 64]}   # next-item scores, unused by z (§7.5)
DROPPED_KEYS = tuple(sorted(DROPPED_SHAPES))


class ConversionError(RuntimeError):
    """The conversion is refused; the message names the cause (§7.8 #1)."""


def _say(message: str) -> None:
    print(f"convert_sasrec: {message}", flush=True)


# ----------------------------------------------------------------------------
# git and hashes
# ----------------------------------------------------------------------------
def git_blob_id(data: bytes) -> str:
    """git blob id of raw bytes (``sha1(b"blob <size>\\0" + data)``), independent of checkout filters."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=600)
    except FileNotFoundError as exc:
        raise ConversionError("git is not installed or not on PATH") from exc


def _git_text(repo: Path, *args: str) -> str:
    out = _git(repo, *args)
    if out.returncode != 0:
        stderr = out.stderr.decode("utf-8", "replace").strip()
        raise ConversionError(f"git {' '.join(args)} failed in {repo}: {stderr}")
    return out.stdout.decode("utf-8").strip()


def require_upstream_commit(repo: Path) -> None:
    """Step 1: the pinned upstream commit is in the repository's history."""
    out = _git(repo, "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}")
    if out.returncode != 0:
        stderr = out.stderr.decode("utf-8", "replace").strip()
        raise ConversionError(f"upstream commit {UPSTREAM_COMMIT} is not in {repo}: the full git history is "
                              f"needed (git fetch --unshallow) [{stderr}]")


def materialize_original_sources(repo: Path, dest: Path) -> dict[str, str]:
    """Step 2: write the pinned original files of ``UPSTREAM_COMMIT`` into ``dest``; returns path -> blob id.

    ``git rev-parse <commit>:<path>`` must give the pinned blob, ``git cat-file blob`` gives the
    bytes and their recomputed blob id must match. Files are written flat (module names) so that
    ``A_SASRec_final_bce_llm``'s ``from SASRecModules_ori import *`` finds its sibling. The
    working-tree copies are never read (§7.5: the fork's ``SASRecModules_ori.py`` differs).
    """
    written: dict[str, str] = {}
    for rel, blob in ORIGINAL_BLOBS.items():
        resolved = _git_text(repo, "rev-parse", f"{UPSTREAM_COMMIT}:{rel}")
        if resolved != blob:
            raise ConversionError(f"{rel} at upstream commit {UPSTREAM_COMMIT} is blob {resolved}, "
                                  f"expected the pinned blob {blob}")
        out = _git(repo, "cat-file", "blob", blob)
        if out.returncode != 0:
            raise ConversionError(f"git cat-file blob {blob} failed: {out.stderr.decode('utf-8', 'replace').strip()}")
        data = out.stdout
        if git_blob_id(data) != blob:
            raise ConversionError(f"bytes of {rel} hash to blob {git_blob_id(data)}, expected {blob}")
        target = Path(dest) / PurePosixPath(rel).name
        if target.exists():
            raise ConversionError(f"two pinned files share the module name {target.name}")
        target.write_bytes(data)
        written[rel] = blob
    return written


def check_pickle_file(pt_path: Path) -> dict[str, Any]:
    """Step 3: git blob id and SHA-256 of the pickle equal the pinned values."""
    if not pt_path.is_file():
        raise ConversionError(f"{pt_path} not found")
    raw = pt_path.read_bytes()
    blob, sha256 = git_blob_id(raw), hashlib.sha256(raw).hexdigest()
    if blob != PICKLE_BLOB or sha256 != PICKLE_SHA256:
        raise ConversionError(f"{pt_path}: git blob {blob} / sha256 {sha256} differ from the pinned "
                              f"blob {PICKLE_BLOB} / sha256 {PICKLE_SHA256}")
    return {"path": PICKLE_PATH, "git_blob": blob, "sha256": sha256, "size": len(raw)}


# ----------------------------------------------------------------------------
# Static pickle scan (step 4)
# ----------------------------------------------------------------------------
_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"}
_GET_OPS = {"GET", "BINGET", "LONG_BINGET"}
_PUT_OPS = {"PUT", "BINPUT", "LONG_BINPUT"}
_NEUTRAL_OPS = {"PROTO", "FRAME"}


def _data_pkl_record(zf: zipfile.ZipFile, pt_path: Path) -> str:
    records = [name for name in zf.namelist() if PurePosixPath(name).name == "data.pkl"]
    if len(records) != 1 or len(PurePosixPath(records[0]).parts) != 2:
        raise ConversionError(f"{pt_path}: expected exactly one '<record>/data.pkl' entry in the zip, got {records}")
    return records[0]


def pickle_record_name(pt_path: Path) -> str:
    """The zip record (folder) name of a ``torch.save`` file, e.g. ``SASRec_val_0.4206_...``."""
    with zipfile.ZipFile(pt_path) as zf:
        return PurePosixPath(_data_pkl_record(zf, pt_path)).parts[0]


def _globals_in_pickle(data: bytes) -> frozenset[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    pushes: list[Optional[str]] = []        # values pushed since the stack was last consumed (None = not a str)
    memo: dict[int, Optional[str]] = {}
    for op, arg, pos in pickletools.genops(data):
        name = op.name
        if name in ("GLOBAL", "INST"):
            module, _, attr = str(arg).partition(" ")
            found.add((module, attr))
            pushes = []
        elif name == "STACK_GLOBAL":
            if len(pushes) < 2 or not all(isinstance(s, str) for s in pushes[-2:]):
                raise ConversionError(f"STACK_GLOBAL at byte {pos} cannot be resolved statically; refused")
            found.add((pushes[-2], pushes[-1]))     # type: ignore[arg-type]
            pushes = []
        elif name in ("EXT1", "EXT2", "EXT4"):
            raise ConversionError(f"pickle uses the copyreg extension registry ({name}) at byte {pos}; refused")
        elif name in _STRING_OPS:
            pushes.append(arg if isinstance(arg, str) else None)
        elif name in _GET_OPS:
            pushes.append(memo.get(arg))
        elif name in _PUT_OPS:
            memo[arg] = pushes[-1] if pushes else None
        elif name == "MEMOIZE":
            memo[len(memo)] = pushes[-1] if pushes else None
        elif name not in _NEUTRAL_OPS:
            pushes = []
    return frozenset(found)


def scan_pickle_globals(pt_path: Path) -> frozenset[tuple[str, str]]:
    """Every global named by ``<record>/data.pkl``, read with ``pickletools.genops`` (nothing is executed)."""
    try:
        with zipfile.ZipFile(pt_path) as zf:
            data = zf.read(_data_pkl_record(zf, pt_path))
    except zipfile.BadZipFile as exc:
        raise ConversionError(f"{pt_path} is not a torch zip archive: {exc}") from exc
    return _globals_in_pickle(data)


def check_globals(found: frozenset[tuple[str, str]]) -> None:
    """The scanned globals equal ``EXPECTED_GLOBALS`` exactly."""
    extra, missing = sorted(found - EXPECTED_GLOBALS), sorted(EXPECTED_GLOBALS - found)
    if extra:
        raise ConversionError(f"the pickle names globals outside the allowed set: {extra} "
                              "(static scan; nothing was unpickled)")
    if missing:
        raise ConversionError(f"the pickle lacks expected globals: {missing}")


# ----------------------------------------------------------------------------
# Original code and model (steps 5-8)
# ----------------------------------------------------------------------------
def import_original_module(src_dir: Path) -> ModuleType:
    """Step 5: import ``A_SASRec_final_bce_llm`` from ``src_dir`` (first on ``sys.path`` during the import).

    Both original modules must come from ``src_dir`` (compared as resolved paths: ``__file__`` mixes
    ``/`` and ``\\`` on Windows), never from the repository root's fork copy.
    """
    src = Path(src_dir).resolve()
    for name in (ORIGINAL_MODULE, ORIGINAL_SUBMODULE):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(ORIGINAL_MODULE)
    finally:
        sys.path.remove(str(src))
    for name in (ORIGINAL_MODULE, ORIGINAL_SUBMODULE):
        file = getattr(sys.modules.get(name), "__file__", None)
        if file is None or Path(file).resolve().parent != src:
            raise ConversionError(f"{name} was imported from {file}, not from the pinned blobs in {src}")
    return module


def load_original_model(pt_path: Path, original_module: ModuleType) -> nn.Module:
    """Step 6: ``torch.load(weights_only=False)`` with an allow-list ``find_class``.

    Globals outside ``EXPECTED_GLOBALS`` minus ``torch.FloatStorage`` (17) are refused;
    ``__main__.SASRec`` maps to the original class and ``SASRecModules_ori`` names to the classes
    of the materialized blob.
    """
    allowed = EXPECTED_GLOBALS - {FLOAT_STORAGE}

    class AllowListUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f"global {module}.{name} is not in the allow-list")
            if (module, name) == ("__main__", "SASRec"):
                return original_module.SASRec
            if module == ORIGINAL_SUBMODULE:
                cls = getattr(original_module, name)
                if cls.__module__ != ORIGINAL_SUBMODULE:
                    raise pickle.UnpicklingError(f"{name} does not come from {ORIGINAL_SUBMODULE}")
                return cls
            return super().find_class(module, name)

    def load(file: Any, **kwargs: Any) -> Any:
        return AllowListUnpickler(file, **kwargs).load()

    pickle_module = types.ModuleType("convert_sasrec_pickle")
    pickle_module.Unpickler = AllowListUnpickler       # type: ignore[attr-defined]
    pickle_module.load = load                          # type: ignore[attr-defined]
    model = torch.load(pt_path, map_location="cpu", pickle_module=pickle_module, weights_only=False)
    if type(model) is not original_module.SASRec:
        raise ConversionError(f"the pickle holds {type(model)!r}, not the original SASRec class")
    return model


def _typed(value: Any) -> dict[str, Any]:
    return {"type": type(value).__name__, "value": int(value)}


def describe_original_model(model: nn.Module) -> dict:
    """Step 7: JSON values observed on the unpickled module (manifest ``observed``).

    Integer attributes carry their type name separately (``state_size`` is ``numpy.int64``,
    which ``json`` cannot write).
    """
    named = list(model.named_modules())
    return {
        "type": f"{type(model).__module__}.{type(model).__qualname__}",
        "attributes": {
            "device": str(model.device),
            "hidden_size": _typed(model.hidden_size),
            "item_num": _typed(model.item_num),
            "state_size": _typed(model.state_size),
            "training": bool(model.training),
        },
        "submodules": {name: type(m).__name__ for name, m in named if name},
        "modules_training": {name or "<root>": bool(m.training) for name, m in named},
        "dropout_p": {name: float(m.p) for name, m in named if isinstance(m, nn.Dropout)},
        "layernorm": {name: {"eps": float(m.eps), "elementwise_affine": bool(m.elementwise_affine),
                             "normalized_shape": [int(s) for s in m.normalized_shape]}
                      for name, m in named if isinstance(m, nn.LayerNorm)},
        "attention": {name: {"num_heads": int(m.num_heads), "hidden_size": int(m.hidden_size)}
                      for name, m in named if type(m).__name__ == "MultiHeadAttention"},
        "conv1d": {name: {"in_channels": int(m.in_channels), "out_channels": int(m.out_channels),
                          "kernel_size": list(m.kernel_size), "stride": list(m.stride),
                          "padding": list(m.padding), "dilation": list(m.dilation), "groups": int(m.groups),
                          "padding_mode": m.padding_mode, "bias": m.bias is not None}
                   for name, m in named if isinstance(m, nn.Conv1d)},
        "linear": {name: {"in_features": int(m.in_features), "out_features": int(m.out_features),
                          "bias": m.bias is not None} for name, m in named if isinstance(m, nn.Linear)},
        "embedding": {name: {"num_embeddings": _typed(m.num_embeddings), "embedding_dim": int(m.embedding_dim),
                             "padding_idx": m.padding_idx, "max_norm": m.max_norm}
                      for name, m in named if isinstance(m, nn.Embedding)},
        "softmax": {name: {"dim": m.dim} for name, m in named if isinstance(m, nn.Softmax)},
        "state": [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "", 1)}
                  for key, value in model.state_dict().items()],
        "buffers": [name for name, _ in model.named_buffers()],
    }


def check_observed(observed: Mapping[str, Any]) -> None:
    """Step 7: every observed value equals the design expectation (§7.4); otherwise refuse."""
    problems: list[str] = []

    def expect(label: str, got: Any, want: Any) -> None:
        if got != want:
            problems.append(f"{label}: observed {got!r}, expected {want!r}")

    expect("type", observed["type"], EXPECTED_MODEL_TYPE)
    for key, want in EXPECTED_ATTRIBUTES.items():
        expect(f"attribute {key}", observed["attributes"].get(key), want)
    expect("submodules", observed["submodules"], EXPECTED_SUBMODULES)
    expect("dropout p", observed["dropout_p"], {name: EXPECTED_DROPOUT_P for name in DROPOUT_MODULES})
    expect("layernorm", observed["layernorm"],
           {name: {"eps": EXPECTED_ARCH.layernorm_eps, "elementwise_affine": True,
                   "normalized_shape": [EXPECTED_ARCH.hidden_size]} for name in LAYERNORM_MODULES})
    expect("attention", observed["attention"],
           {"mh_attn": {"num_heads": EXPECTED_ARCH.num_heads, "hidden_size": EXPECTED_ARCH.hidden_size}})
    expect("conv1d", observed["conv1d"], {name: EXPECTED_CONV1D for name in ("feed_forward.w_1", "feed_forward.w_2")})
    h = EXPECTED_ARCH.hidden_size
    expect("linear", observed["linear"],
           {**{f"mh_attn.linear_{p}": {"in_features": h, "out_features": h, "bias": True} for p in "qkv"},
            "s_fc": {"in_features": h, "out_features": EXPECTED_ARCH.item_num, "bias": True}})
    expect("embedding", observed["embedding"], {
        "item_embeddings": {"num_embeddings": {"type": "int64", "value": EXPECTED_ARCH.item_num + 1},
                            "embedding_dim": h, "padding_idx": None, "max_norm": None},
        "positional_embeddings": {"num_embeddings": {"type": "int64", "value": EXPECTED_ARCH.state_size},
                                  "embedding_dim": h, "padding_idx": None, "max_norm": None}})
    expect("softmax", observed["softmax"], {"mh_attn.softmax": {"dim": -1}})
    shapes = {name: list(shape) for name, shape in expected_state_shapes(EXPECTED_ARCH).items()}
    shapes.update(DROPPED_SHAPES)
    expect("state tensors", {s["name"]: (s["shape"], s["dtype"]) for s in observed["state"]},
           {name: (shape, "float32") for name, shape in shapes.items()})
    expect("number of state tensors", len(observed["state"]), 22)
    expect("buffers", observed["buffers"], [])
    if problems:
        raise ConversionError("the unpickled SASRec differs from the design (7.4):\n  - " + "\n  - ".join(problems))


def arch_of(model: nn.Module) -> SASRecArch:
    """Sizes of the pickled module as Python numbers (§7.6.2: numpy int64 is not weights_only-safe)."""
    eps = {float(m.eps) for m in model.modules() if isinstance(m, nn.LayerNorm)}
    if len(eps) != 1:
        raise ConversionError(f"LayerNorm eps values differ: {sorted(eps)}")
    return SASRecArch(hidden_size=int(model.hidden_size), item_num=int(model.item_num),
                      state_size=int(model.state_size), num_heads=int(model.mh_attn.num_heads),
                      layernorm_eps=eps.pop())


# ----------------------------------------------------------------------------
# Reference samples and outputs (steps 9-11)
# ----------------------------------------------------------------------------
def read_reference_samples(data_dir: Path) -> dict[str, tuple[list[str], torch.Tensor, torch.Tensor]]:
    """Step 9: ``split -> (sample_ids, hist_ids [n, 10] int64, len_seq [n] int64)`` read with pandas.

    Independent of M1: the released ``.df`` files are hashed (``run.EXPECTED_INPUTS``) before
    they are unpickled; every row with ``len_seq >= 3`` in row order, ``sample_id = "{split}:{row}"``
    (``row`` = row of the file before the filter), item ids = first element of each ``seq`` pair,
    right padding with the padding id checked.
    """
    pad, length = EXPECTED_ARCH.item_num, EXPECTED_ARCH.state_size
    out: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]] = {}
    for split, name in SPLIT_FILES:
        rel = f"{DATA_DIR}/{name}"
        path = Path(data_dir) / name
        if not path.is_file():
            raise ConversionError(f"{path} not found")
        expected, is_text = run.EXPECTED_INPUTS[rel]
        got = run.content_sha256(path, is_text)
        if got != expected:
            raise ConversionError(f"{rel}: sha256 {got} != expected {expected} (not unpickled)")
        frame = pd.read_pickle(path)
        sample_ids: list[str] = []
        rows: list[list[int]] = []
        lengths: list[int] = []
        for row, (seq, len_seq) in enumerate(zip(frame["seq"].tolist(), frame["len_seq"].tolist())):
            n = int(len_seq)
            if n < MIN_LEN_SEQ:
                continue
            ids = [int(pair[0]) for pair in seq]
            if len(ids) != length or not n <= length:
                raise ConversionError(f"{split} row {row}: seq has {len(ids)} entries and len_seq {n}, expected {length}")
            if any(not 0 <= x < pad for x in ids[:n]) or any(x != pad for x in ids[n:]):
                raise ConversionError(f"{split} row {row}: {ids} with len_seq {n} is not right-padded with {pad}")
            sample_ids.append(f"{split}:{row}")
            rows.append(ids)
            lengths.append(n)
        if len(sample_ids) != EXPECTED_SAMPLES[split]:
            raise ConversionError(f"{split}: {len(sample_ids)} samples with len_seq >= {MIN_LEN_SEQ}, "
                                  f"expected {EXPECTED_SAMPLES[split]}")
        out[split] = (sample_ids, torch.tensor(rows, dtype=torch.int64), torch.tensor(lengths, dtype=torch.int64))
    return out


def compute_reference(model: nn.Module, hist_ids: torch.Tensor, len_seq: torch.Tensor) -> torch.Tensor:
    """Step 10: original ``cacul_h`` on one split, ``REF_CHUNK`` rows at a time with ``REF_THREADS`` CPU
    thread(s), no_grad -> ``z [n, hidden] float32`` (the ``[B, 1, h]`` output squeezed)."""
    training = [name or "<root>" for name, m in model.named_modules() if m.training]
    if training:
        raise ConversionError(f"the original model must be in eval mode (step 8), training: {training}")
    previous = torch.get_num_threads()
    parts = []
    try:
        torch.set_num_threads(REF_THREADS)
        with torch.no_grad():
            for start in range(0, hist_ids.shape[0], REF_CHUNK):
                out = model.cacul_h(hist_ids[start:start + REF_CHUNK], len_seq[start:start + REF_CHUNK])
                if out.dim() != 3 or out.shape[1] != 1:
                    raise ConversionError(f"cacul_h returned shape {list(out.shape)}, expected [B, 1, hidden]")
                parts.append(out.squeeze(1))
    finally:
        torch.set_num_threads(previous)
    z = torch.cat(parts).contiguous()
    if z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
        raise ConversionError(f"reference z must be finite float32, got {z.dtype}")
    return z


def self_check(model: nn.Module, samples: Mapping[str, tuple[list[str], torch.Tensor, torch.Tensor]],
               z_ref: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Step 11: the original code repeats bit for bit, and the pooled z ignores what padded slots hold."""
    repeat_diff, probe_diff, padded_rows = 0.0, 0.0, 0
    repeat_equal = probe_equal = True
    for split, (_, hist, lens) in samples.items():
        again = compute_reference(model, hist, lens)
        repeat_equal &= torch.equal(again, z_ref[split])
        repeat_diff = max(repeat_diff, float((again - z_ref[split]).abs().max()))
        probe = hist.clone()
        padded = torch.arange(hist.shape[1])[None, :] >= lens[:, None]
        probe[padded] = PADDING_PROBE_ID
        padded_rows += int(padded.any(dim=1).sum())
        changed = compute_reference(model, probe, lens)
        probe_equal &= torch.equal(changed, z_ref[split])
        probe_diff = max(probe_diff, float((changed - z_ref[split]).abs().max()))
    result = {"repeat": {"bitwise": bool(repeat_equal), "max_abs_diff": repeat_diff},
              "padding_content": {"bitwise": bool(probe_equal), "max_abs_diff": probe_diff,
                                  "rows_with_padding": padded_rows, "probe_id": PADDING_PROBE_ID}}
    if not (repeat_equal and probe_equal):
        raise ConversionError(f"the original code is not deterministic on this machine: {result}")
    return result


# ----------------------------------------------------------------------------
# Outputs (step 12)
# ----------------------------------------------------------------------------
def _ensure_empty_out(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ConversionError(f"output path {out_dir} exists and is not a folder")
        if any(out_dir.iterdir()):
            raise ConversionError(f"output folder {out_dir} exists and is not empty; refusing to overwrite (4.6.3)")


def write_outputs(out_dir: Path, state: dict, reference: dict, manifest: dict) -> None:
    """Step 12: state file, reference file, then ``manifest.json`` (with the two file sha256), each written
    under a temporary name and renamed. The two ``.pt`` files hold only tensors, str, int, float, list
    and dict, so ``torch.load(weights_only=True)`` reads them."""
    out_dir = Path(out_dir)
    _ensure_empty_out(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, obj in ((STATE_FILE, state), (REFERENCE_FILE, reference)):
        tmp = out_dir / f"{name}.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, out_dir / name)
        outputs[name] = run.content_sha256(out_dir / name)
    run.write_json_atomic(out_dir / MANIFEST_FILE, {**manifest, "outputs": outputs})


def _reread(out_dir: Path, digest: str, n_samples: int) -> None:
    """Both files load with weights_only=True and pass the runtime reader of ``ilora.sasrec``."""
    load_sasrec_state(out_dir / STATE_FILE, digest)
    ref = torch.load(out_dir / REFERENCE_FILE, map_location="cpu", weights_only=True)
    if (set(ref) != set(REFERENCE_FILE_KEYS) or ref["format"] != SASREC_REFERENCE_FORMAT
            or ref["state_digest"] != digest or len(ref["sample_ids"]) != n_samples):
        raise ConversionError(f"{out_dir / REFERENCE_FILE} does not read back as written")


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------
def convert(repo: Path, out_dir: Path) -> dict[str, Any]:
    """Steps 1-12 of §7.6.1; returns the manifest."""
    _ensure_empty_out(out_dir)
    require_upstream_commit(repo)                                                      # 1
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert_sasrec_src_", dir=out_dir.parent,
                                     ignore_cleanup_errors=True) as tmp:
        src_dir = Path(tmp)
        blobs = materialize_original_sources(repo, src_dir)                            # 2
        _say(f"original sources from {UPSTREAM_COMMIT[:7]}: {blobs}")
        pt_path = repo / PICKLE_PATH
        pickle_info = check_pickle_file(pt_path)                                       # 3
        found = scan_pickle_globals(pt_path)                                           # 4
        check_globals(found)
        pickle_info.update({"zip_record": pickle_record_name(pt_path),
                            "globals": [list(g) for g in sorted(found)],
                            "find_class_allow_list": [list(g) for g in sorted(EXPECTED_GLOBALS - {FLOAT_STORAGE})]})
        _say(f"pickle {PICKLE_PATH}: sha256 {pickle_info['sha256']}, {len(found)} globals as expected")
        original = import_original_module(src_dir)                                     # 5
        model = load_original_model(pt_path, original)                                 # 6
        observed = describe_original_model(model)                                      # 7
        check_observed(observed)
        arch = arch_of(model)
        if arch != EXPECTED_ARCH:
            raise ConversionError(f"architecture {arch} != expected {EXPECTED_ARCH}")
        model.device = torch.device("cpu")     # 8: cacul_h moves its input to self.device (U ...:203-204)
        model.eval()
        still_training = [name for name, m in model.named_modules() if m.training]
        if still_training:
            raise ConversionError(f"modules still in training mode after eval(): {still_training}")

        data_dir = repo / DATA_DIR
        samples = read_reference_samples(data_dir)                                     # 9
        _say("reference samples: " + ", ".join(f"{s} {len(v[0])}" for s, v in samples.items()))
        z_ref = {split: compute_reference(model, hist, lens) for split, (_, hist, lens) in samples.items()}   # 10
        item_ids = torch.arange(arch.item_num, dtype=torch.int64)
        with torch.no_grad():
            item_emb = model.cacu_x(item_ids).detach().contiguous()                    # U :219-222
        checks = self_check(model, samples, z_ref)                                     # 11
        _say(f"self check: {checks}")

        full_state = model.state_dict()
        dropped = sorted(key for key in full_state if key not in expected_state_shapes(arch))
        if tuple(dropped) != DROPPED_KEYS:
            raise ConversionError(f"dropped keys {dropped} != {list(DROPPED_KEYS)}")
        kept = {key: full_state[key].detach().to("cpu").clone().contiguous() for key in expected_state_shapes(arch)}
        digest = state_digest(kept)
        state = {"format": SASREC_STATE_FORMAT, "arch": arch.as_dict(), "state_dict": kept,
                 "dropped_keys": list(DROPPED_KEYS),
                 "source": {"path": PICKLE_PATH, "git_blob": pickle_info["git_blob"], "sha256": pickle_info["sha256"]},
                 "digest": digest}
        order = [split for split, _ in SPLIT_FILES]
        reference = {
            "format": SASREC_REFERENCE_FORMAT,
            "sample_ids": [sid for split in order for sid in samples[split][0]],
            "hist_ids": torch.cat([samples[split][1] for split in order]).contiguous(),
            "len_seq": torch.cat([samples[split][2] for split in order]).contiguous(),
            "z": torch.cat([z_ref[split] for split in order]).contiguous(),
            "item_ids": item_ids,
            "item_emb": item_emb,
            "chunk": REF_CHUNK,
            "num_threads": REF_THREADS,
            "state_digest": digest,
        }
        git = run.git_info(repo)
        manifest = {
            "tool": "rebuild/tools/convert_sasrec.py",
            "design": "rebuild/DESIGN.md 7.6.1-7.6.2",
            "created_utc": run.utc_now().isoformat(),
            "repository": {"path": str(repo), "head": git.get("commit"), "branch": git.get("branch"),
                           "dirty": git.get("dirty")},
            "upstream": {"commit": UPSTREAM_COMMIT, "original_blobs": blobs},
            "pickle": pickle_info,
            "observed": observed,
            "arch": arch.as_dict(),
            "state": {"format": SASREC_STATE_FORMAT, "digest": digest, "n_tensors": len(kept),
                      "n_values": sum(t.numel() for t in kept.values()), "dropped": dict(DROPPED_SHAPES)},
            "reference": {"format": SASREC_REFERENCE_FORMAT, "chunk": REF_CHUNK, "num_threads": REF_THREADS,
                          "counts": {split: len(samples[split][0]) for split in order},
                          "total": len(reference["sample_ids"]), "z_shape": list(reference["z"].shape),
                          "item_emb_shape": list(item_emb.shape)},
            "self_check": checks,
            "data": {f"{DATA_DIR}/{name}": run.content_sha256(data_dir / name) for _, name in SPLIT_FILES},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
        }
    write_outputs(out_dir, state, reference, manifest)                                 # 12
    _reread(out_dir, digest, len(reference["sample_ids"]))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_sasrec.py",
        description="Convert rec_model/movielens.pt once into the plain SASRec state, the original-code "
                    "reference outputs and manifest.json (rebuild/DESIGN.md 7.6.1).")
    parser.add_argument("--repo", type=Path, default=run.REPO_ROOT,
                        help="repository root with the full git history (default: the one holding this tool)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output folder, absolute or relative to --repo, e.g. runs/rebuild/assets/sasrec; "
                             "must not exist or be empty")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    out_dir = args.out if args.out.is_absolute() else repo / args.out
    try:
        with warnings.catch_warnings():
            # torch 2.0.0's loader touches TypedStorage and warns that it is deprecated (torch-internal)
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
            manifest = convert(repo, out_dir.resolve())
    except ConversionError as exc:
        print(f"convert_sasrec: refused: {exc}", file=sys.stderr, flush=True)
        return 1
    _say(f"wrote {STATE_FILE}, {REFERENCE_FILE} and {MANIFEST_FILE} to {out_dir}: state digest "
         f"{manifest['state']['digest']}, {manifest['reference']['total']} reference rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
