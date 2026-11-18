"""M3 변환 도구 시험 (DESIGN.md §7.8 #1-#4; §17.6.9 M3 표).

#1 runs ``tools/convert_sasrec.py``'s ``main`` in the test process against temporary git
repositories: an empty one (``git init``, so the upstream commit is missing) and ones that
share the real repository's objects through ``.git/objects/info/alternates`` (read-only; no
commit is created anywhere). Module constants are monkeypatched per case as §17.6.9 says.
#2-#4 read the output of one full conversion into a tmp folder (``converted_sasrec``, slow).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest
import torch

from ilora import run, sasrec

# torch 2.0.0's own loader warns that TypedStorage is deprecated whenever a file is read
pytestmark = pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")

REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REBUILD_DIR.parent
TOOL_PATH = REBUILD_DIR / "tools" / "convert_sasrec.py"
STATE_DIGEST = "6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae"   # §7.6.2, paper_spec
PICKLE_SHA256 = "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9"
PICKLE_BLOB = "41c057dc96df2cc6d432d7f839d98b57ac341b26"
# Fork baseline whose SASRecModules_ori.py is the bf16 copy (F 761dea8, blob e1e33498, §7.5): the blob case.
FORK_COMMIT = "6c7ebac"
FORK_MODULES_BLOB = "e1e3349807c467f8ae51824296f0d95a80748f22"
ORIGINAL_MODULES_BLOB = "3ac3c975a58d43ee866b637ee5d4f99534974a7b"

# §7.6.1 step 4: the 18 globals of the released pickle
EXPECTED_GLOBALS = {
    ("__main__", "SASRec"), ("SASRecModules_ori", "MultiHeadAttention"),
    ("SASRecModules_ori", "PositionwiseFeedForward"), ("torch.nn.modules.sparse", "Embedding"),
    ("torch.nn.modules.normalization", "LayerNorm"), ("torch.nn.modules.linear", "Linear"),
    ("torch.nn.modules.conv", "Conv1d"), ("torch.nn.modules.dropout", "Dropout"),
    ("torch.nn.modules.activation", "Softmax"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"), ("torch", "FloatStorage"), ("torch", "device"),
    ("collections", "OrderedDict"), ("__builtin__", "set"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"), ("_codecs", "encode"),
}
# §7.6.2: the 20 kept tensors
STATE_SHAPES = {
    "item_embeddings.weight": (1683, 64), "positional_embeddings.weight": (10, 64),
    "ln_1.weight": (64,), "ln_1.bias": (64,), "ln_2.weight": (64,), "ln_2.bias": (64,),
    "ln_3.weight": (64,), "ln_3.bias": (64,),
    "mh_attn.linear_q.weight": (64, 64), "mh_attn.linear_q.bias": (64,),
    "mh_attn.linear_k.weight": (64, 64), "mh_attn.linear_k.bias": (64,),
    "mh_attn.linear_v.weight": (64, 64), "mh_attn.linear_v.bias": (64,),
    "feed_forward.w_1.weight": (64, 64, 1), "feed_forward.w_1.bias": (64,),
    "feed_forward.w_2.weight": (64, 64, 1), "feed_forward.w_2.bias": (64,),
    "feed_forward.layer_norm.weight": (64,), "feed_forward.layer_norm.bias": (64,),
}
SAMPLE_COUNTS = {"train": 66126, "val": 94, "test": 95}


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch):
    """A fresh import of ``rebuild/tools/convert_sasrec.py`` (constants can be monkeypatched per test)."""
    spec = importlib.util.spec_from_file_location("convert_sasrec_under_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def temp_repo(tmp_path: Path, git_objects_dir: Path) -> Callable[[bool], Path]:
    """Factory ``temp_repo(with_history) -> Path``: ``git init`` in tmp_path (user git config not read); with
    history, the real repository's object store is attached as an alternate (read-only sharing, nothing
    committed). ``git_objects_dir`` skips without git or full history (conftest)."""
    config = tmp_path / "isolated.gitconfig"
    config.write_text("", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1"})

    def make(with_history: bool) -> Path:
        repo = tmp_path / "repo"
        subprocess.run(["git", "init", "-q", str(repo)], env=env, capture_output=True, check=True)
        if with_history:          # bytes: a text-mode write would end the line with CRLF on Windows
            (repo / ".git" / "objects" / "info" / "alternates").write_bytes(
                (git_objects_dir.as_posix() + "\n").encode("utf-8"))
        (repo / "rec_model").mkdir()
        return repo

    return make


class _MakesDirectory:
    """Unpickling this would call ``os.makedirs(path)``: the tool must refuse it without unpickling."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)

    def __reduce__(self):
        return (os.makedirs, (self.path,))


# ----------------------------------------------------------------------------------------------- #1
REFUSAL_CASES = ("blob", "pickle_hash", "foreign_global", "nonempty_out", "missing_commit")


@pytest.mark.criterion("M3-1")
@pytest.mark.parametrize("case", REFUSAL_CASES)
def test_tool_refuses(case, tool, temp_repo, tmp_path, monkeypatch, capsys):
    real_pickle = (REPO_ROOT / "rec_model" / "movielens.pt").read_bytes()
    out_dir = tmp_path / "assets" / "sasrec"
    marker = tmp_path / "unpickled_marker"
    repo = temp_repo(case != "missing_commit")
    pickle_path = repo / "rec_model" / "movielens.pt"
    if case == "blob":
        # an upstream commit whose SASRecModules_ori.py is not the pinned blob (the fork's bf16 copy)
        fork = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{FORK_COMMIT}^{{commit}}"],
                              capture_output=True, text=True, check=True).stdout.strip()
        monkeypatch.setattr(tool, "UPSTREAM_COMMIT", fork)
        pickle_path.write_bytes(real_pickle)
        cause = f"is blob {FORK_MODULES_BLOB}, expected the pinned blob {ORIGINAL_MODULES_BLOB}"
    elif case == "pickle_hash":
        changed = bytearray(real_pickle)
        changed[len(changed) // 2] ^= 0x01
        pickle_path.write_bytes(bytes(changed))
        cause = f"differ from the pinned blob {PICKLE_BLOB} / sha256 {PICKLE_SHA256}"
    elif case == "foreign_global":
        torch.save({"payload": _MakesDirectory(marker)}, pickle_path)       # a test pickle with os.makedirs
        raw = pickle_path.read_bytes()
        monkeypatch.setattr(tool, "PICKLE_SHA256", hashlib.sha256(raw).hexdigest())
        monkeypatch.setattr(tool, "PICKLE_BLOB", _blob_id(raw))
        cause = "globals outside the allowed set: [('os', 'makedirs')]"
    elif case == "nonempty_out":
        pickle_path.write_bytes(real_pickle)
        out_dir.mkdir(parents=True)
        (out_dir / "previous_run.txt").write_text("kept", encoding="utf-8")
        cause = "exists and is not empty; refusing to overwrite"
    else:
        pickle_path.write_bytes(real_pickle)
        cause = "the full git history is needed (git fetch --unshallow)"

    code = tool.main(["--repo", str(repo), "--out", str(out_dir)])
    err = capsys.readouterr().err
    assert code != 0
    assert "convert_sasrec: refused:" in err and cause in err, err
    assert not marker.exists(), "the foreign global was executed"
    left = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
    assert left == (["previous_run.txt"] if case == "nonempty_out" else []), left
    assert list(tmp_path.glob("assets/convert_sasrec_src_*")) == [], "temporary source folder left behind"


@pytest.mark.criterion("M3-1")
def test_static_scan_sees_protocol_4_globals(tool, tmp_path):
    """The scan resolves STACK_GLOBAL (protocol 4, memoized names) too, so a foreign global cannot hide there."""
    marker = tmp_path / "unpickled_marker"
    path = tmp_path / "protocol4.pt"
    torch.save({"payload": _MakesDirectory(marker), "weight": torch.zeros(2)}, path, pickle_protocol=4)
    found = tool.scan_pickle_globals(path)
    assert ("os", "makedirs") in found
    assert ("torch._utils", "_rebuild_tensor_v2") in found
    with pytest.raises(tool.ConversionError, match=r"\('os', 'makedirs'\)"):
        tool.check_globals(found)
    assert not marker.exists()
    assert tool.scan_pickle_globals(REPO_ROOT / "rec_model" / "movielens.pt") == EXPECTED_GLOBALS


# ----------------------------------------------------------------------------------------------- #2
@pytest.mark.slow
@pytest.mark.criterion("M3-2")
def test_state_file(converted_sasrec):
    path = converted_sasrec / "sasrec_movielens_state.pt"
    obj = torch.load(path, map_location="cpu", weights_only=True)
    assert set(obj) == {"format", "arch", "state_dict", "dropped_keys", "source", "digest"}
    assert obj["format"] == "ilora-rebuild-sasrec-v1"
    state = obj["state_dict"]
    assert {key: tuple(value.shape) for key, value in state.items()} == STATE_SHAPES
    assert all(value.dtype == torch.float32 for value in state.values())
    assert sum(value.numel() for value in state.values()) == 129_664
    assert obj["dropped_keys"] == ["s_fc.bias", "s_fc.weight"]
    manifest = json.loads((converted_sasrec / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"]["dropped"] == {"s_fc.weight": [1682, 64], "s_fc.bias": [1682]}
    assert obj["digest"] == STATE_DIGEST == sasrec.state_digest(state) == manifest["state"]["digest"]
    arch = obj["arch"]
    assert arch == {"hidden_size": 64, "item_num": 1682, "state_size": 10, "num_heads": 1, "layernorm_eps": 1e-5}
    assert all(type(arch[key]) is int for key in ("hidden_size", "item_num", "state_size", "num_heads"))
    assert type(arch["layernorm_eps"]) is float
    assert obj["source"] == {"path": "rec_model/movielens.pt", "git_blob": PICKLE_BLOB, "sha256": PICKLE_SHA256}
    loaded_arch, loaded, digest = sasrec.load_sasrec_state(path, STATE_DIGEST)       # the runtime reader
    assert loaded_arch == sasrec.SASRecArch(64, 1682, 10, 1, 1e-5) and digest == STATE_DIGEST
    assert all(torch.equal(loaded[key], state[key]) for key in STATE_SHAPES)
    assert manifest["outputs"]["sasrec_movielens_state.pt"] == run.content_sha256(path)


# ----------------------------------------------------------------------------------------------- #3
@pytest.mark.slow
@pytest.mark.criterion("M3-3")
def test_manifest_observed(converted_sasrec):
    manifest = json.loads((converted_sasrec / "manifest.json").read_text(encoding="utf-8"))
    observed = manifest["observed"]
    attributes = observed["attributes"]
    assert attributes["training"] is True
    assert attributes["device"] == "cuda:0"
    assert attributes["state_size"] == {"type": "int64", "value": 10}            # numpy int64 in the pickle
    assert attributes["item_num"]["value"] == 1682 and attributes["hidden_size"]["value"] == 64
    assert observed["dropout_p"] == {"dropout": 0.1, "emb_dropout": 0.1, "mh_attn.dropout": 0.1,
                                     "feed_forward.dropout": 0.1}
    assert {name: v["eps"] for name, v in observed["layernorm"].items()} == {
        "ln_1": 1e-5, "ln_2": 1e-5, "ln_3": 1e-5, "feed_forward.layer_norm": 1e-5}
    assert observed["attention"] == {"mh_attn": {"num_heads": 1, "hidden_size": 64}}
    assert len(observed["modules_training"]) == 20 and all(observed["modules_training"].values())
    assert len(observed["state"]) == 22
    assert {tuple(g) for g in manifest["pickle"]["globals"]} == EXPECTED_GLOBALS
    assert len(manifest["pickle"]["globals"]) == 18
    assert {tuple(g) for g in manifest["pickle"]["find_class_allow_list"]} == EXPECTED_GLOBALS - {("torch", "FloatStorage")}
    assert manifest["pickle"]["sha256"] == PICKLE_SHA256 and manifest["pickle"]["size"] == 973_163
    assert manifest["pickle"]["zip_record"].startswith("SASRec_val_0.4206_hr1_0.4222_")
    assert manifest["upstream"] == {"commit": "28e47371192121b6ced44401d9e149fedfae13de", "original_blobs": {
        "SASRecModules_ori.py": ORIGINAL_MODULES_BLOB,
        "recommender/A_SASRec_final_bce_llm.py": "dc7c94999a4db5f1b97971a2913baae44687c124"}}
    for rel, sha in manifest["data"].items():
        assert sha == run.EXPECTED_INPUTS[rel][0], rel


@pytest.mark.criterion("M3-3")
def test_observed_mismatch_refused(tool, sasrec_state_path):
    """Step 7 compares every observed value with the design: the asset's own observation passes, a
    single changed value (dropout p, numpy type of state_size, LayerNorm eps, a state shape) is refused."""
    manifest = json.loads((sasrec_state_path.parent / "manifest.json").read_text(encoding="utf-8"))
    observed = manifest["observed"]
    tool.check_observed(observed)
    changes = {
        "dropout p": lambda o: o["dropout_p"].update({"emb_dropout": 0.2}),
        "attribute state_size": lambda o: o["attributes"]["state_size"].update({"type": "int"}),
        "layernorm": lambda o: o["layernorm"]["ln_2"].update({"eps": 1e-6}),
        "state tensors": lambda o: o["state"][0].update({"shape": [1684, 64]}),
    }
    for label, change in changes.items():
        bad = json.loads(json.dumps(observed))
        change(bad)
        with pytest.raises(tool.ConversionError, match=label):
            tool.check_observed(bad)


# ----------------------------------------------------------------------------------------------- #4
@pytest.mark.slow
@pytest.mark.criterion("M3-4")
def test_reference_file(converted_sasrec):
    ref = torch.load(converted_sasrec / "sasrec_movielens_reference.pt", map_location="cpu", weights_only=True)
    assert set(ref) == set(sasrec.REFERENCE_FILE_KEYS)
    assert ref["format"] == sasrec.SASREC_REFERENCE_FORMAT and ref["state_digest"] == STATE_DIGEST
    ids = ref["sample_ids"]
    assert len(ids) == 66_315 and len(set(ids)) == 66_315
    pattern = re.compile(r"(train|val|test):(0|[1-9][0-9]*)")
    assert all(pattern.fullmatch(sid) for sid in ids)
    splits = [sid.split(":")[0] for sid in ids]
    assert {s: splits.count(s) for s in SAMPLE_COUNTS} == SAMPLE_COUNTS
    assert splits == ["train"] * 66_126 + ["val"] * 94 + ["test"] * 95
    for split in SAMPLE_COUNTS:                                             # rows ascending within a split
        rows = [int(sid.split(":")[1]) for sid in ids if sid.startswith(split + ":")]
        assert rows == sorted(rows) and len(set(rows)) == len(rows)
    assert ref["hist_ids"].shape == (66_315, 10) and ref["hist_ids"].dtype == torch.int64
    assert ref["len_seq"].shape == (66_315,) and ref["len_seq"].dtype == torch.int64
    assert int(ref["len_seq"].min()) == 3 and int(ref["len_seq"].max()) == 10
    real = torch.arange(10)[None, :] < ref["len_seq"][:, None]                # right padding with 1682
    assert bool(((ref["hist_ids"] != 1682) == real).all()) and int(ref["hist_ids"].max()) == 1682
    assert ref["z"].shape == (66_315, 64) and ref["z"].dtype == torch.float32
    assert bool(torch.isfinite(ref["z"]).all())
    assert torch.equal(ref["item_ids"], torch.arange(1682))
    assert ref["item_emb"].shape == (1682, 64) and ref["item_emb"].dtype == torch.float32
    state = torch.load(converted_sasrec / "sasrec_movielens_state.pt", map_location="cpu", weights_only=True)
    assert torch.equal(ref["item_emb"], state["state_dict"]["item_embeddings.weight"][:1682])   # cacu_x = lookup
    assert (ref["chunk"], ref["num_threads"]) == (4096, 1)
    manifest = json.loads((converted_sasrec / "manifest.json").read_text(encoding="utf-8"))
    check = manifest["self_check"]
    assert check["repeat"] == {"bitwise": True, "max_abs_diff": 0.0}
    assert check["padding_content"]["bitwise"] is True and check["padding_content"]["max_abs_diff"] == 0.0
    assert check["padding_content"]["rows_with_padding"] == int((ref["len_seq"] < 10).sum()) > 0
    assert manifest["reference"]["counts"] == SAMPLE_COUNTS
