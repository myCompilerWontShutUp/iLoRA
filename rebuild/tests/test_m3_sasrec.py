"""M3 SASRec 인코더 시험 (DESIGN.md §7.8 #5-#11; §17.6.5 SASRec 동등성, §17.6.9 M3 표).

#5-#10 use the converted asset (``sasrec_state_path``/``sasrec_reference_path``: the tool output in
``runs/rebuild/assets/sasrec`` or ``ILORA_SASREC_STATE``) and skip without it. #11 and the
state-file rejections build a synthetic state and reference in tmp_path from
``seeding.torch_seed_scope(0, "test", <name>)`` / ``seeding.np_rng(0, "test", <name>)`` (§17.6.2),
so they run without the asset. #1-#4 are ``test_m3_convert.py``, #12 ``vessl/test_vessl_sasrec.py``
and #13 ``vessl/test_run_outputs.py::test_e2_run_sasrec_digest`` (VESSL_REQUIRED).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import torch
from torch import nn

from ilora import commands, run, sasrec, seeding
from ilora.data import HISTORY_LEN, NUM_ITEMS, PAD_ID, ItemCatalog, pad_histories
from ilora.router import Router
from ilora.sasrec import SASRecArch, SASRecEncoder, SASRecError
from ilora.sections import RouterConfig

# torch 2.0.0's own loader warns that TypedStorage is deprecated whenever a file is read
pytestmark = pytest.mark.filterwarnings("ignore:TypedStorage is deprecated:UserWarning")

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIGEST = "6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae"   # §7.6.2, paper_spec
PICKLE_SHA256 = "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9"
PICKLE_BLOB = "41c057dc96df2cc6d432d7f839d98b57ac341b26"
SPLITS = ("train", "val", "test")
SPLIT_SIZES = {"train": 66_126, "val": 94, "test": 95}
ARCH = SASRecArch(hidden_size=64, item_num=1682, state_size=10, num_heads=1, layernorm_eps=1e-5)
MODULE_NAMES = {"", "item_embeddings", "positional_embeddings", "ln_1", "ln_2", "ln_3", "mh_attn",
                "mh_attn.linear_q", "mh_attn.linear_k", "mh_attn.linear_v", "feed_forward", "feed_forward.w_1",
                "feed_forward.w_2", "feed_forward.layer_norm"}


def _catalog(pad_id: int = PAD_ID) -> ItemCatalog:
    """An M1 ``ItemCatalog`` with placeholder titles (``from_config`` reads only pad_id and num_items)."""
    return ItemCatalog(titles=tuple(f"item {i}" for i in range(NUM_ITEMS)), title_mode="code",
                       num_items=NUM_ITEMS, pad_id=pad_id)


@pytest.fixture(scope="module")
def encoder(sasrec_state_path: Path, paper_cfg) -> SASRecEncoder:
    """``SASRecEncoder.from_config(paper_spec, catalog, 10)`` on the converted asset. Tests that change an
    encoder work on a copy."""
    return SASRecEncoder.from_config(paper_cfg, _catalog(), HISTORY_LEN)


@pytest.fixture(scope="module")
def reference(sasrec_reference_path: Path) -> dict:
    return torch.load(sasrec_reference_path, map_location="cpu", weights_only=True)


def _blocks(reference: dict) -> dict[str, slice]:
    """Rows of each split in the reference file (one contiguous block per split, train/val/test)."""
    splits = [sid.split(":", 1)[0] for sid in reference["sample_ids"]]
    out = {}
    for split in SPLITS:
        start = splits.index(split)
        out[split] = slice(start, start + splits.count(split))
    return out


def _independent_digest(state: dict[str, torch.Tensor]) -> str:
    """§7.6.2 written out again: sorted keys, ``key|dtype|shape\\n`` then little-endian raw bytes."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(f"{key}|{str(t.dtype)[len('torch.'):]}|{','.join(map(str, t.shape))}\n".encode())
        h.update(t.numpy().astype("<f4").tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- #2 (asset)
@pytest.mark.criterion("M3-2")
def test_state_digest_definition(sasrec_state_path):
    """The converted asset's content digest, recomputed independently, is the design value 6a78697b...1dae."""
    obj = torch.load(sasrec_state_path, map_location="cpu", weights_only=True)
    state = obj["state_dict"]
    assert _independent_digest(state) == sasrec.state_digest(state) == obj["digest"] == STATE_DIGEST
    renamed = dict(reversed(list(state.items())))                    # insertion order does not matter
    assert sasrec.state_digest(renamed) == STATE_DIGEST
    changed = dict(state)
    changed["ln_1.bias"] = state["ln_1.bias"].clone()
    changed["ln_1.bias"][0] += 1.0
    assert sasrec.state_digest(changed) != STATE_DIGEST


# ----------------------------------------------------------------------------------------------- #5
@pytest.mark.criterion("M3-5")
def test_matches_reference_bitwise(encoder, reference, torch_threads1):
    """Per split, the reference chunk (4,096) and 1 thread: z equals the original cacul_h bit for bit (§17.6.5)."""
    chunk = reference["chunk"]
    assert (chunk, reference["num_threads"]) == (4096, 1)
    assert reference["state_digest"] == encoder.state_digest == STATE_DIGEST
    for split, rows in _blocks(reference).items():
        hist, lens, z_ref = reference["hist_ids"][rows], reference["len_seq"][rows], reference["z"][rows]
        assert hist.shape[0] == SPLIT_SIZES[split]
        z = torch.cat([encoder(hist[i:i + chunk], lens[i:i + chunk]) for i in range(0, hist.shape[0], chunk)])
        assert z.dtype == torch.float32 and z.shape == z_ref.shape
        assert torch.equal(z, z_ref), f"{split}: max |dz| = {float((z - z_ref).abs().max())}"


@pytest.mark.criterion("M3-5")
def test_batch8_close(encoder, reference, torch_threads1, record_property):
    """Batches of 8 over all 66,315 rows: max |z - z_ref| <= 1e-5 (design prototype 7.63e-6, §7.6.5)."""
    hist, lens, z_ref = reference["hist_ids"], reference["len_seq"], reference["z"]
    z = torch.cat([encoder(hist[i:i + 8], lens[i:i + 8]) for i in range(0, hist.shape[0], 8)])
    diff = float((z - z_ref).abs().max())
    record_property("batch8_max_abs_diff", diff)
    assert diff <= 1e-5


@pytest.mark.criterion("M3-5")
def test_embed_items_bitwise(encoder, reference):
    items = encoder.embed_items(reference["item_ids"])
    assert items.dtype == torch.float32 and items.shape == (1682, 64) and not items.requires_grad
    assert torch.equal(items, reference["item_emb"])                 # cacu_x of the original code
    grid = reference["item_ids"][:20].reshape(4, 5)                  # any leading shape -> [..., 64]
    assert torch.equal(encoder.embed_items(grid), reference["item_emb"][:20].reshape(4, 5, 64))


# ----------------------------------------------------------------------------------------------- #6
@pytest.mark.criterion("M3-6")
def test_always_eval(encoder):
    enc = copy.deepcopy(encoder)
    parent = nn.ModuleDict({"sasrec": enc, "head": nn.Linear(64, 4)})
    parent.train()
    assert parent.training and parent["head"].training
    assert not any(m.training for m in enc.modules())                # a parent's .train() cannot switch it
    assert enc.train(True) is enc
    assert not any(m.training for m in enc.modules())
    enc.train()
    parent.eval()
    parent.train(True)
    assert not any(m.training for m in enc.modules())
    assert not any(isinstance(m, nn.Dropout) for m in enc.modules())  # D12 by structure
    assert {name for name, _ in enc.named_modules()} == MODULE_NAMES
    enc.ln_2.training = True                                         # forced flag: the forward refuses
    with pytest.raises(RuntimeError, match="training mode"):
        enc(torch.full((1, 10), PAD_ID).index_fill_(1, torch.arange(3), 7), torch.tensor([3]))


# ----------------------------------------------------------------------------------------------- #7
@pytest.mark.criterion("M3-7")
def test_frozen_and_gate_grad(encoder, reference):
    assert all(not p.requires_grad for p in encoder.parameters())
    hist, lens, ids = reference["hist_ids"][:32], reference["len_seq"][:32], reference["sample_ids"][:32]
    z = encoder(hist, lens)
    assert not z.requires_grad and z.grad_fn is None
    router = Router(RouterConfig(routing_mode="dynamic"), num_experts=4, z_dim=encoder.hidden_size,
                    main_seed=1234, hard_gate=None)
    omega = router(z, ids)
    with seeding.torch_seed_scope(0, "test", "m3_gate_loss"):
        weights = torch.randn(omega.shape)
    (omega * weights).sum().backward()
    grad = router.gate.proj.weight.grad
    assert grad is not None and grad.dtype == torch.float32 and bool((grad != 0).any())
    assert all(p.grad is None for p in encoder.parameters())         # gradient reaches W_g only
    enc = copy.deepcopy(encoder)
    enc.ln_3.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="requires grad"):
        enc(hist, lens)
    with pytest.raises(RuntimeError, match="requires grad"):
        enc.embed_items(reference["item_ids"][:3])


# ----------------------------------------------------------------------------------------------- #8
@pytest.mark.criterion("M3-8")
def test_fp32_under_autocast(encoder, reference):
    hist, lens = reference["hist_ids"][:256], reference["len_seq"][:256]
    outside = encoder(hist, lens)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = encoder(hist, lens)
        items_inside = encoder.embed_items(reference["item_ids"][:16])
        assert torch.is_autocast_cpu_enabled()                       # the caller's autocast comes back
        unprotected = encoder.mh_attn.linear_q(outside[:, None, :])  # an op outside the fp32 region
    assert inside.dtype == torch.float32 and torch.equal(inside, outside)
    assert items_inside.dtype == torch.float32 and torch.equal(items_inside, reference["item_emb"][:16])
    assert unprotected.dtype == torch.bfloat16
    half = copy.deepcopy(encoder).half()
    with pytest.raises(RuntimeError, match="float32"):
        half(hist, lens)
    with pytest.raises(RuntimeError, match="float32"):
        half.embed_items(reference["item_ids"][:3])


@pytest.mark.criterion("M3-8")
def test_strict_fp32_flags():
    """``strict_fp32``: autocast off inside; on CUDA both TF32 flags off inside and the caller's values after
    (the flag logic runs without a GPU; the CUDA numbers are M3-12)."""
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with sasrec.strict_fp32(torch.device("cuda")):
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with sasrec.strict_fp32(torch.device("cpu")):
                assert not torch.is_autocast_cpu_enabled()
                assert torch.backends.cudnn.allow_tf32                # CPU leaves the CUDA flags alone
            assert torch.is_autocast_cpu_enabled()
        with pytest.raises(RuntimeError), sasrec.strict_fp32("cuda"):
            torch.backends.cudnn.allow_tf32 = True
            raise RuntimeError("inside")
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
        with pytest.raises(ValueError, match="cpu or cuda"):
            sasrec.strict_fp32(torch.device("meta"))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


# ----------------------------------------------------------------------------------------------- #9
INPUT_CASES = ("float_hist_ids", "shape_b9", "len_seq_0", "len_seq_11", "left_padding", "pad_inside_real",
               "id_1683", "embed_items_pad")


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    rng = seeding.np_rng(0, "test", "m3_input_validation")
    histories = [[int(x) for x in rng.choice(NUM_ITEMS, size=n, replace=False)] for n in (3, 7, 10, 5)]
    return pad_histories(histories, PAD_ID, HISTORY_LEN)


@pytest.mark.criterion("M3-9")
@pytest.mark.parametrize("case", INPUT_CASES)
def test_input_validation(case, encoder):
    hist, lens = _valid_inputs()
    assert encoder(hist, lens).shape == (4, 64)                      # the unchanged inputs pass
    hist, lens = hist.clone(), lens.clone()
    call: Callable[[], object] = lambda: encoder(hist, lens)        # noqa: E731
    if case == "float_hist_ids":
        call, match = (lambda: encoder(hist.float(), lens)), "hist_ids must be int64"
    elif case == "shape_b9":
        call, match = (lambda: encoder(hist[:, :9], lens)), r"hist_ids must be \[B, 10\]"
    elif case == "len_seq_0":
        lens[0] = 0
        match = "len_seq 0 is outside"
    elif case == "len_seq_11":
        lens[0] = 11
        match = "len_seq 11 is outside"
    elif case == "left_padding":
        hist[0] = torch.cat([hist[0, 3:], hist[0, :3]])              # [1682 x 7, a, b, c] with len_seq 3
        match = "padding id 1682 inside the first len_seq = 3 positions"
    elif case == "pad_inside_real":
        hist[1, 2] = PAD_ID                                          # 1682 inside the 7 real positions
        match = "padding id 1682 inside the first len_seq = 7 positions"
    elif case == "id_1683":
        hist[2, 4] = PAD_ID + 1
        match = "item id outside 0-1681"
    else:
        call, match = (lambda: encoder.embed_items(torch.tensor([5, PAD_ID]))), "padding id 1682 is not an item"
    with pytest.raises(ValueError, match=match):
        call()


# ----------------------------------------------------------------------------------------------- #10
@pytest.mark.slow
@pytest.mark.criterion("M3-10")
def test_encode_all_matches_reference(encoder, reference, sasrec_reference_path):
    original = torch.get_num_threads()
    probe = 3 if original == 2 else 2                                # a value different from the file's 1
    torch.set_num_threads(probe)
    try:
        for split, rows in _blocks(reference).items():
            hist, lens = reference["hist_ids"][rows], reference["len_seq"][rows]
            histories = [row[:n] for row, n in zip(hist.tolist(), lens.tolist())]
            z = encoder.encode_all(histories, chunk=reference["chunk"], num_threads=reference["num_threads"])
            assert torch.get_num_threads() == probe                  # restored after the call
            assert z.device.type == "cpu" and torch.equal(z, reference["z"][rows]), split
        report = sasrec.verify_against_reference(encoder, sasrec_reference_path, torch.device("cpu"))
        assert torch.get_num_threads() == probe
    finally:
        torch.set_num_threads(original)
    assert report["passed"] and report["bitwise"] and report["max_abs_diff"] == 0.0
    assert report["item_emb"]["bitwise"] and report["n"] == 66_315
    assert {split: v["n"] for split, v in report["splits"].items()} == SPLIT_SIZES
    assert all(v["padding_matches"] for v in report["splits"].values())


@pytest.mark.slow
@pytest.mark.criterion("M3-10")
def test_pad_histories_matches_reference(full_data, reference):
    """M1 samples and ``pad_histories`` give the reference rows exactly (the tool read the files without M1)."""
    for split, rows in _blocks(reference).items():
        samples = full_data.samples(split)
        assert [s.sample_id for s in samples] == reference["sample_ids"][rows]
        hist, lens = pad_histories([s.history for s in samples], PAD_ID, HISTORY_LEN)
        assert torch.equal(hist, reference["hist_ids"][rows]), split
        assert torch.equal(lens, reference["len_seq"][rows]), split


def _export_z(samples, encoder: SASRecEncoder, cfg) -> tuple[torch.Tensor, list[str], str]:
    """z as M12 export makes it (§16.6.5 step 5): ``records.compute_representations`` once M12 exists,
    until then the same rule through the public M1/M3 API (consecutive batches of
    ``records.export_batch_size`` in row order, ``pad_histories``, ``encoder``, ``export_num_threads``)."""
    if importlib.util.find_spec("ilora.records") is not None:
        from ilora import records

        table = records.compute_representations(
            list(samples), encoder, None, num_experts=cfg.lora.num_experts,
            batch_size=cfg.records.export_batch_size, num_threads=cfg.records.export_num_threads,
            rowsum_atol=cfg.records.gate_rowsum_atol)
        return torch.from_numpy(np.asarray(table.z)), list(table.sample_ids), "records.compute_representations"
    size = cfg.records.export_batch_size
    previous = torch.get_num_threads()
    torch.set_num_threads(cfg.records.export_num_threads)
    try:
        parts = []
        for start in range(0, len(samples), size):
            batch = samples[start:start + size]
            hist, lens = pad_histories([s.history for s in batch], PAD_ID, HISTORY_LEN)
            parts.append(encoder(hist, lens))
    finally:
        torch.set_num_threads(previous)
    return torch.cat(parts), [s.sample_id for s in samples], "M12 rule via the encoder (records.py not built)"


@pytest.mark.slow
@pytest.mark.criterion("M3-10")
def test_export_z_close_to_reference(encoder, reference, full_data, paper_cfg, record_property):
    """Export z (1,024 per batch, 1 thread) against the reference: max difference <= 1e-5 (recorded; 0 locally)."""
    assert (paper_cfg.records.export_batch_size, paper_cfg.records.export_num_threads) == (1024, 1)
    for split, rows in _blocks(reference).items():
        z, ids, source = _export_z(full_data.samples(split), encoder, paper_cfg)
        assert ids == reference["sample_ids"][rows]
        diff = float((z - reference["z"][rows]).abs().max())
        record_property(f"export_{split}_max_abs_diff", diff)
        record_property("export_source", source)
        assert diff <= 1e-5, split


# ----------------------------------------------------------------------------------------------- #11
def _synthetic_state() -> dict[str, torch.Tensor]:
    with seeding.torch_seed_scope(0, "test", "m3_synthetic_state"):
        return {key: torch.randn(shape) * 0.5 for key, shape in sasrec.expected_state_shapes(ARCH).items()}


def _state_file(state: dict[str, torch.Tensor], **changes: object) -> dict:
    obj = {"format": sasrec.SASREC_STATE_FORMAT, "arch": ARCH.as_dict(), "state_dict": state,
           "dropped_keys": ["s_fc.bias", "s_fc.weight"],
           "source": {"path": "rec_model/movielens.pt", "git_blob": PICKLE_BLOB, "sha256": PICKLE_SHA256},
           "digest": sasrec.state_digest(state)}
    obj.update(changes)
    return obj


def _synthetic_assets(tmp_path: Path) -> tuple[Path, Path, str]:
    """A synthetic converted state and a reference made by the rebuild encoder itself (a check-env mock)."""
    state = _synthetic_state()
    state_path = tmp_path / "synthetic_state.pt"
    torch.save(_state_file(state), state_path)
    digest = sasrec.state_digest(state)
    encoder = SASRecEncoder(ARCH, digest)
    encoder.load_state_dict(state, strict=True)
    rng = seeding.np_rng(0, "test", "m3_synthetic_histories")
    histories = [[int(x) for x in rng.choice(NUM_ITEMS, size=int(n), replace=False)]
                 for n in rng.integers(3, HISTORY_LEN + 1, size=24)]
    hist, lens = pad_histories(histories, PAD_ID, HISTORY_LEN)
    reference = {"format": sasrec.SASREC_REFERENCE_FORMAT, "sample_ids": [f"val:{i}" for i in range(24)],
                 "hist_ids": hist, "len_seq": lens, "z": encoder.encode_all(histories),
                 "item_ids": torch.arange(NUM_ITEMS), "item_emb": state["item_embeddings.weight"][:NUM_ITEMS].clone(),
                 "chunk": sasrec.EXPORT_CHUNK, "num_threads": sasrec.EXPORT_THREADS, "state_digest": digest}
    reference_path = tmp_path / "synthetic_reference.pt"
    torch.save(reference, reference_path)
    return state_path, reference_path, digest


@pytest.mark.criterion("M3-11")
@pytest.mark.parametrize("case", ["digest", "source_sha256", "pad_id"])
def test_check_env_sasrec_failures(case, profile_cfg, tmp_path):
    """check-env's SASRec part (M0 ``commands.check_sasrec``) passes on the unchanged mock and fails for each change."""
    state_path, reference_path, digest = _synthetic_assets(tmp_path)
    paths = [f"sasrec.state_path={json.dumps(state_path.as_posix())}",
             f"sasrec.reference_path={json.dumps(reference_path.as_posix())}"]
    cfg = profile_cfg("smoke_cpu", *paths, f"sasrec.state_digest={json.dumps(digest)}")
    catalog = SimpleNamespace(pad_id=PAD_ID, num_items=NUM_ITEMS)
    result = commands.check_sasrec(cfg, catalog, repo_root=REPO_ROOT, device="cpu")
    assert result["state_digest"] == digest and result["source_sha256"] == PICKLE_SHA256
    assert result["reference"]["passed"] and result["reference"]["max_abs_diff"] == 0.0
    if case == "digest":
        other = profile_cfg("smoke_cpu", *paths, f"sasrec.state_digest={json.dumps('0' * 64)}")
        with pytest.raises(SASRecError, match=f"state digest {digest} != sasrec.state_digest 0{{64}}"):
            commands.check_sasrec(other, catalog, repo_root=REPO_ROOT, device="cpu")
    elif case == "source_sha256":
        changed = bytearray((REPO_ROOT / "rec_model" / "movielens.pt").read_bytes())
        changed[-1] ^= 0x01
        fake = tmp_path / "movielens.pt"
        fake.write_bytes(bytes(changed))
        other = profile_cfg("smoke_cpu", *paths, f"sasrec.state_digest={json.dumps(digest)}",
                            f"sasrec.source_path={json.dumps(fake.as_posix())}")
        with pytest.raises(run.EnvironmentCheckError, match=f"SASRec source sha256 [0-9a-f]{{64}} != "
                                                            f"sasrec.source_sha256 {PICKLE_SHA256}"):
            commands.check_sasrec(other, catalog, repo_root=REPO_ROOT, device="cpu")
    else:
        with pytest.raises(SASRecError, match="catalog.pad_id 1681 != SASRec item_num 1682"):
            commands.check_sasrec(cfg, SimpleNamespace(pad_id=1681, num_items=NUM_ITEMS), repo_root=REPO_ROOT,
                                  device="cpu")
        with pytest.raises(SASRecError, match="history length 9 != SASRec state_size 10"):
            SASRecEncoder.from_config(cfg, _catalog(), HISTORY_LEN - 1)


@pytest.mark.criterion("M3-11")
def test_verify_against_reference_rejects(tmp_path):
    """The reference comparison of check-env fails on one float32 step (CPU is bitwise), on a reference
    of another state and on a device other than the encoder's."""
    state_path, reference_path, digest = _synthetic_assets(tmp_path)
    arch, state, _ = sasrec.load_sasrec_state(state_path, digest)
    encoder = SASRecEncoder(arch, digest)
    encoder.load_state_dict(state, strict=True)
    report = sasrec.verify_against_reference(encoder, reference_path, "cpu")
    assert report["passed"] and report["bitwise"] and report["splits"]["val"]["n"] == 24
    ref = torch.load(reference_path, map_location="cpu", weights_only=True)
    z = ref["z"].clone()
    z[3, 5] = torch.nextafter(z[3, 5], torch.tensor(float("inf")))
    changed = tmp_path / "changed_reference.pt"
    torch.save({**ref, "z": z}, changed)
    with pytest.raises(SASRecError, match="differs from the reference outputs") as err:
        sasrec.verify_against_reference(encoder, changed, "cpu")
    assert err.value.report["splits"]["val"]["bitwise"] is False
    assert 0.0 < err.value.report["max_abs_diff"] < 1e-5
    torch.save({**ref, "state_digest": "0" * 64}, changed)
    with pytest.raises(SASRecError, match="reference made for SASRec state"):
        sasrec.verify_against_reference(encoder, changed, "cpu")
    with pytest.raises(ValueError, match="the encoder is on cpu"):
        sasrec.verify_against_reference(encoder, reference_path, "cuda")


STATE_FILE_CASES = ("missing_file", "format", "keys", "dtype", "shape", "arch_type", "file_digest", "expected_digest")


@pytest.mark.criterion("M3-11")
@pytest.mark.parametrize("case", STATE_FILE_CASES)
def test_load_sasrec_state_rejects(case, tmp_path):
    """The loader that check-env and build use stops on every state-file problem of §7.6.6."""
    state = _synthetic_state()
    digest = sasrec.state_digest(state)
    path = tmp_path / "state.pt"
    torch.save(_state_file(state), path)
    arch, loaded, got = sasrec.load_sasrec_state(path, digest)      # the unchanged file passes
    assert arch == ARCH and got == digest and set(loaded) == set(state)
    expected = digest
    if case == "missing_file":
        path = tmp_path / "absent.pt"
        match = "not found"
    elif case == "format":
        torch.save(_state_file(state, format="other-format"), path)
        match = "format 'other-format'"
    elif case == "keys":
        torch.save({**_state_file(state), "notes": "extra"}, path)
        match = "keys"
    elif case == "dtype":
        torch.save(_state_file({**state, "ln_1.bias": state["ln_1.bias"].double()}), path)
        match = r"ln_1.bias'\] is torch.float64"
    elif case == "shape":
        torch.save(_state_file({**state, "positional_embeddings.weight": state["positional_embeddings.weight"][:9]}),
                   path)
        match = r"has shape \(9, 64\), expected \(10, 64\)"
    elif case == "arch_type":
        torch.save(_state_file(state, arch={**ARCH.as_dict(), "hidden_size": 64.0}), path)
        match = "hidden_size must be a Python int"
    elif case == "file_digest":
        tampered = {**state, "ln_2.weight": state["ln_2.weight"] + 1.0}
        torch.save(_state_file(tampered, digest=digest), path)
        match = "recomputed state digest"
    else:
        expected = "f" * 64
        match = f"state digest {digest} != sasrec.state_digest {expected}"
    with pytest.raises(SASRecError, match=match):
        sasrec.load_sasrec_state(path, expected)
