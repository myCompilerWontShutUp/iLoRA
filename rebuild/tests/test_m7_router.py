"""M7 gate와 routing (DESIGN.md §11.8 #1-#11, #13).

#7 and #8 also have M10 tests (``test_m10_train.py::test_cluster_hard_training``,
``::test_dynamic_never_calls_cluster``), #12 is
``test_m11_evaluate.py::test_gate_record_single_collection_point`` and #14-#15 are
in ``tests/vessl/test_run_outputs.py`` (§17.6.9). z comes from M3 only in the
tests that need the converted SASRec asset; the others use synthetic z rows from
``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2) and the
``fake_hard_gate`` fixture for the M15 ``HardGate``.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ilora import seeding
from ilora.router import ROUTING_MODES, ROW_SUM_ATOL, GateRecord, LinearGate, Router, RouterError
from ilora.sections import RouterConfig

K, D = 4, 64                        # lora.num_experts (paper_spec), SASRec hidden_size
SEED = 1234                         # paper_spec run.seed
ROUTER_PY = Path(__file__).resolve().parents[1] / "ilora" / "router.py"
MAPPING_TEXT = "0:3,1:2,2:1,3:0"    # a permutation cluster -> expert mapping (accepted by M15, 19.8 #1)
MAPPING = (3, 2, 1, 0)              # expert of cluster c = MAPPING[c]


def make_router(mode: str = "dynamic", *, num_experts: int = K, z_dim: int = D, seed: int = SEED,
                hard_gate=None) -> Router:
    return Router(RouterConfig(routing_mode=mode), num_experts=num_experts, z_dim=z_dim, main_seed=seed,
                  hard_gate=hard_gate)


def synthetic_z(name: str, n: int, d: int = D) -> torch.Tensor:
    """float32 rows with norms near 24 (real z norms are 21-34, §11.6.6)."""
    with seeding.torch_seed_scope(0, "test", name):
        return torch.randn(n, d) * 3.0


def make_ids(split: str, n: int, start: int = 0) -> list[str]:
    return [f"{split}:{start + i}" for i in range(n)]


def gate_loss(omega: torch.Tensor) -> torch.Tensor:
    """A loss that depends on every omega entry (fixed synthetic weights)."""
    with seeding.torch_seed_scope(0, "test", "gate_loss_weights"):
        weights = torch.randn(omega.shape)
    return (omega * weights).sum()


# ----------------------------------------------------------------------------------------------- #1
@pytest.mark.criterion("M7-1")
def test_linear_gate_init():
    with torch.random.fork_rng(devices=[]):          # a bare LinearGate draws from the global RNG
        gate = LinearGate(D, K)
    assert [name for name, _ in gate.named_parameters()] == ["proj.weight"]
    assert tuple(gate.proj.weight.shape) == (K, D) and gate.proj.weight.dtype == torch.float32
    assert gate.proj.bias is None
    assert sum(p.numel() for p in gate.parameters()) == 256

    before = torch.get_rng_state()
    router = make_router()
    assert torch.equal(torch.get_rng_state(), before), "building the gate changed the global CPU RNG state"
    weight = router.gate.proj.weight
    assert weight.requires_grad and weight.device.type == "cpu" and weight.dtype == torch.float32
    assert float(weight.abs().max()) <= 1 / 8
    assert torch.equal(make_router().gate.proj.weight, weight)
    assert not torch.equal(make_router(seed=SEED + 1).gate.proj.weight, weight)
    # kaiming_uniform(a=sqrt(5)) on [K, 64], row-major, from derive_seed(seed, "init", "gate") (§11.6.2)
    with seeding.torch_seed_scope(SEED, "init", "gate"):
        expected = torch.empty(K, D)
        nn.init.kaiming_uniform_(expected, a=math.sqrt(5))
    assert torch.equal(weight, expected)


# ----------------------------------------------------------------------------------------------- #2
@pytest.mark.criterion("M7-2")
def test_dynamic_omega_matches_fp64():
    router = make_router()
    z = synthetic_z("dynamic_omega", 256)
    omega = router(z, make_ids("val", 256))
    assert omega.shape == (256, K) and omega.dtype == torch.float32 and omega.device == z.device
    assert bool((omega >= 0).all())
    assert float((omega.sum(dim=1) - 1.0).abs().max()) <= ROW_SUM_ATOL
    weight = router.gate.proj.weight.detach().double()
    expected = torch.softmax(z.double() @ weight.T, dim=-1)        # 식 11 in fp64
    assert float((omega.double() - expected).abs().max()) <= 1e-6
    assert omega.requires_grad
    assert router.gate_calls == 1


# ----------------------------------------------------------------------------------------------- #3
@pytest.mark.criterion("M7-3")
def test_omega_fp32_under_autocast():
    router = make_router()
    z = synthetic_z("autocast_z", 16)
    batch = make_ids("val", 16)
    outside = router(z, batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = router(z, batch)
        assert torch.is_autocast_cpu_enabled()                     # the caller's autocast state is restored
        unprotected = F.linear(z, router.gate.proj.weight)         # what a gate without the fp32 region would get
    assert inside.dtype == torch.float32
    assert torch.equal(inside, outside)
    assert unprotected.dtype == torch.bfloat16


# ----------------------------------------------------------------------------------------------- #4
@pytest.mark.criterion("M7-4")
def test_batch_independence():
    router = make_router()
    z = synthetic_z("batch_independence", 32)
    batch = make_ids("val", 32)
    together = router(z, batch)
    for i in (0, 7, 31):
        alone = router(z[i:i + 1], batch[i:i + 1])
        assert float((alone - together[i:i + 1]).abs().max()) <= 1e-6
    # other rows of the batch do not move row 0 (the released BatchNorm router would, §11.5)
    others = synthetic_z("batch_independence_others", 32)
    others[0] = z[0]
    mixed = router(others, batch)
    assert float((mixed[0] - together[0]).abs().max()) <= 1e-6


# ----------------------------------------------------------------------------------------------- #5
@pytest.mark.criterion("M7-5")
def test_gradient_only_to_wg(sasrec_state_path, cpu_cfg, smoke_data):
    """z from the real SASRec (M3): the gate loss reaches W_g and no SASRec parameter."""
    from ilora.data import HISTORY_LEN, pad_histories
    from ilora.sasrec import SASRecEncoder

    encoder = SASRecEncoder.from_config(cpu_cfg, smoke_data.catalog, history_len=HISTORY_LEN)
    samples = smoke_data.samples("train")[:8]
    hist_ids, len_seq = pad_histories([s.history for s in samples], smoke_data.catalog.pad_id, HISTORY_LEN)
    z = encoder(hist_ids, len_seq)
    assert z.dtype == torch.float32 and not z.requires_grad
    router = make_router(z_dim=encoder.hidden_size)
    omega = router(z, [s.sample_id for s in samples])
    gate_loss(omega).backward()
    grad = router.gate.proj.weight.grad
    assert grad is not None and grad.dtype == torch.float32 and bool((grad != 0).any())
    assert all(p.grad is None for p in encoder.parameters())


@pytest.mark.criterion("M7-5")
def test_gradient_reaches_wg_synthetic():
    """Same check with a frozen stand-in encoder (runs without the SASRec asset)."""
    with seeding.torch_seed_scope(0, "test", "frozen_encoder"):
        encoder = nn.Linear(D, D)
    encoder.requires_grad_(False)
    with torch.no_grad():                                          # z is a no_grad result, like M3 z
        z = encoder(synthetic_z("frozen_encoder_input", 6))
    router = make_router()
    omega = router(z, make_ids("train", 6))
    gate_loss(omega).backward()
    grad = router.gate.proj.weight.grad
    assert grad is not None and grad.dtype == torch.float32 and bool((grad != 0).any())
    assert z.grad is None
    assert all(p.grad is None for p in encoder.parameters())


# ----------------------------------------------------------------------------------------------- #6
def _real_cluster_hard_gate(request):
    """M15 ``ClusterHardGate`` on the smoke cluster artifacts; skips without the converted SASRec asset."""
    state_path = request.getfixturevalue("sasrec_state_path")
    cluster_dir = request.getfixturevalue("smoke_cluster_dir")
    cfg = request.getfixturevalue("cpu_cfg")
    data = request.getfixturevalue("smoke_data")
    repo_root = request.getfixturevalue("repo_root")
    from ilora import cluster, run

    expected = [s.sample_id for split in ("train", "val", "test") for s in data.samples(split)]
    artifacts = cluster.load_cluster_artifacts(cluster_dir, expected_ids=expected,
                                               sasrec_sha256=run.content_sha256(state_path),
                                               data_sha256=run.input_hashes(repo_root), ccfg=cfg.cluster)
    mapping = cluster.parse_expert_mapping(MAPPING_TEXT, n_clusters=artifacts.n_clusters, num_experts=K)
    assert tuple(mapping) == MAPPING
    hard_gate = cluster.ClusterHardGate(artifacts, mapping, K)
    batch = ([s for s in expected if s.startswith("train:")][:4] + [s for s in expected if s.startswith("val:")][:3]
             + [s for s in expected if s.startswith("test:")][:3])
    return hard_gate, dict(artifacts.assignments), batch


@pytest.mark.criterion("M7-6")
@pytest.mark.parametrize("source", ["fake", "real"])
def test_cluster_hard_omega(source, fake_hard_gate, request):
    if source == "fake":
        batch = make_ids("train", 6) + make_ids("val", 3, start=10) + make_ids("test", 3, start=20)
        clusters = {sid: i % K for i, sid in enumerate(batch)}
        hard_gate = fake_hard_gate({sid: MAPPING[c] for sid, c in clusters.items()}, K)
    else:
        hard_gate, clusters, batch = _real_cluster_hard_gate(request)
    router = make_router("cluster_hard", hard_gate=hard_gate)
    z = synthetic_z("cluster_hard_z", len(batch))
    omega = router(z, batch)
    assert omega.shape == (len(batch), K) and omega.dtype == torch.float32 and omega.device == z.device
    assert bool(((omega == 0) | (omega == 1)).all())
    assert torch.equal(omega.sum(dim=1), torch.ones(len(batch)))
    assert omega.argmax(dim=1).tolist() == [MAPPING[clusters[sid]] for sid in batch]
    assert router.gate_calls == 0
    assert hard_gate.lookups == len(batch)


@pytest.mark.criterion("M7-6")
@pytest.mark.parametrize("fault", ["two_hot", "half", "batch_axis", "dtype", "device"])
def test_hard_gate_output_checks(fault, fake_hard_gate):
    """Row sum 2, value 0.5, [B, 1, K], float64, another device: each is a RouterError (no squeeze, no cast)."""
    batch = make_ids("train", 5)
    table = {sid: i % K for i, sid in enumerate(batch)}
    z = synthetic_z("hard_gate_faults", 5)
    assert make_router("cluster_hard", hard_gate=fake_hard_gate(table, K))(z, batch).shape == (5, K)
    router = make_router("cluster_hard", hard_gate=fake_hard_gate(table, K, fault=fault))
    with pytest.raises(RouterError) as info:
        router(z, batch)
    if fault in ("two_hot", "half"):                  # value errors name the first bad row and its sample_id
        assert "row 0, sample_id 'train:0'" in str(info.value)


@pytest.mark.criterion("M7-6")
@pytest.mark.parametrize("fault", ["shape", "nan", "negative", "rowsum", "zeros"])
def test_hard_gate_other_faults(fault, fake_hard_gate):
    batch = make_ids("train", 3)
    router = make_router("cluster_hard", hard_gate=fake_hard_gate({s: 1 for s in batch}, K, fault=fault))
    with pytest.raises(RouterError):
        router(synthetic_z("hard_gate_other_faults", 3), batch)


@pytest.mark.criterion("M7-6")
def test_unknown_sample_id(fake_hard_gate):
    """An id without an assignment raises the HardGate's KeyError unchanged [SPEC §21]."""
    router = make_router("cluster_hard", hard_gate=fake_hard_gate({"train:0": 1}, K))
    with pytest.raises(KeyError) as info:
        router(synthetic_z("unknown_id", 2), ["train:0", "train:999"])
    assert not isinstance(info.value, RouterError)
    assert "train:999" in str(info.value)


# ----------------------------------------------------------------------------------------------- #7
@pytest.mark.criterion("M7-7")
def test_cluster_hard_has_no_gate(fake_hard_gate):
    train_ids = make_ids("train", 156)                  # smoke_cpu train window count (§17.6.6)
    hard_gate = fake_hard_gate({sid: i % K for i, sid in enumerate(train_ids)}, K)
    before = torch.get_rng_state()
    router = make_router("cluster_hard", hard_gate=hard_gate)
    assert torch.equal(torch.get_rng_state(), before)
    assert router.gate is None
    assert router.trainable_parameters() == []
    assert list(router.parameters()) == [] and dict(router.state_dict()) == {}
    z = synthetic_z("cluster_hard_epoch", 156)
    for start in range(0, 156, 4):                      # one synthetic epoch: 39 micro-batches of 4
        router(z[start:start + 4], train_ids[start:start + 4])
    assert router.gate_calls == 0
    assert hard_gate.lookups == 156
    dynamic = make_router()
    n_dynamic = sum(p.numel() for p in dynamic.trainable_parameters())
    n_cluster = sum(p.numel() for p in router.trainable_parameters())
    assert n_dynamic - n_cluster == 256                 # TrainableParams difference K x d [SPEC §11]


# ----------------------------------------------------------------------------------------------- #8
@pytest.mark.criterion("M7-8")
@pytest.mark.parametrize("mode,num_experts,with_hard_gate", [("dynamic", K, True), ("none", 1, True),
                                                             ("cluster_hard", K, False)],
                         ids=["dynamic_with_hard_gate", "none_with_hard_gate", "cluster_hard_without"])
def test_hard_gate_mode_mismatch(mode, num_experts, with_hard_gate, fake_hard_gate):
    hard_gate = fake_hard_gate({"train:0": 0}, num_experts) if with_hard_gate else None
    before = torch.get_rng_state()
    with pytest.raises(RouterError, match="hard_gate"):
        make_router(mode, num_experts=num_experts, hard_gate=hard_gate)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.criterion("M7-8")
def test_hard_gate_protocol_checked():
    class NoLookups:
        def __call__(self, sample_ids, device):
            return torch.zeros(len(sample_ids), K)

    class NotCallable:
        lookups = 0

    for bad in (NoLookups(), NotCallable()):
        with pytest.raises(RouterError, match="hard_gate"):
            make_router("cluster_hard", hard_gate=bad)


# ----------------------------------------------------------------------------------------------- #9
@pytest.mark.criterion("M7-9")
def test_none_mode():
    router = make_router("none", num_experts=1)
    z = synthetic_z("none_mode", 5)
    omega = router(z, make_ids("train", 5))
    assert omega.dtype == torch.float32 and torch.equal(omega, torch.ones(5, 1))
    assert router.gate is None and router.trainable_parameters() == []
    assert list(router.parameters()) == [] and dict(router.state_dict()) == {}
    assert router.gate_calls == 0
    assert router.provenance() == {"routing_mode": "none", "num_experts": 1, "z_dim": D, "gate_shape": None}


@pytest.mark.criterion("M7-9")
@pytest.mark.parametrize("mode,num_experts", [("none", 4), ("dynamic", 1), ("cluster_hard", 1)])
def test_mode_k_combinations(mode, num_experts, fake_hard_gate):
    hard_gate = fake_hard_gate({"train:0": 0}, num_experts) if mode == "cluster_hard" else None
    with pytest.raises(RouterError, match="num_experts"):
        make_router(mode, num_experts=num_experts, hard_gate=hard_gate)


@pytest.mark.criterion("M7-9")
def test_constructor_rejects_bad_values():
    assert ROUTING_MODES == ("dynamic", "cluster_hard", "none")
    with pytest.raises(RouterError, match="routing_mode"):
        make_router("hard")
    with pytest.raises(RouterError, match="routing_mode"):
        Router(None, num_experts=K, z_dim=D, main_seed=SEED, hard_gate=None)
    for kwargs in ({"num_experts": 0}, {"num_experts": True}, {"z_dim": 0}, {"z_dim": 64.0}):
        with pytest.raises(RouterError):
            make_router(**kwargs)
    with pytest.raises(RouterError, match="main_seed"):
        Router(RouterConfig(routing_mode="dynamic"), num_experts=K, z_dim=D, main_seed="1234", hard_gate=None)


# ----------------------------------------------------------------------------------------------- #10
@pytest.mark.criterion("M7-10")
@pytest.mark.parametrize("case", ["nan_z", "z_b1d", "ids_length", "fp16_z"])
def test_forward_input_checks(case):
    router = make_router()
    z = synthetic_z("forward_checks", 3)
    batch = make_ids("val", 3)
    if case == "nan_z":
        z[1, 5] = float("nan")
    elif case == "z_b1d":
        z = z.unsqueeze(1)                              # [B, 1, 64] is rejected, not squeezed
    elif case == "ids_length":
        batch = batch[:2]
    elif case == "fp16_z":
        z = z.half()
    with pytest.raises(RouterError):
        router(z, batch)
    assert router.gate_calls == 0                       # rejected before the gate runs


@pytest.mark.criterion("M7-10")
def test_nan_gate_weight_caught():
    router = make_router()
    with torch.no_grad():
        router.gate.proj.weight[2, 0] = float("nan")
    with pytest.raises(RouterError, match=r"row 0, sample_id 'val:0'"):
        router(synthetic_z("nan_gate", 3), make_ids("val", 3))


# ----------------------------------------------------------------------------------------------- #11
@pytest.mark.criterion("M7-11")
@pytest.mark.parametrize("case", ["grad_enabled", "double_start", "other_split", "duplicate_id",
                                  "expected_mismatch", "stop_without_start"])
def test_recorder_errors(case):
    router = make_router()
    z = synthetic_z("recorder_errors", 4)
    batch = make_ids("val", 4)
    if case == "stop_without_start":
        with pytest.raises(RouterError, match="without start_recording"):
            router.stop_recording(batch)
        return
    router.start_recording("val")
    if case == "grad_enabled":
        assert torch.is_grad_enabled()
        with pytest.raises(RouterError, match="no_grad"):
            router(z, batch)
        assert router.gate_calls == 0
    elif case == "double_start":
        with pytest.raises(RouterError, match="while recording"):
            router.start_recording("test")
    elif case == "other_split":
        with torch.no_grad(), pytest.raises(RouterError, match="does not belong"):
            router(z, make_ids("test", 4))
    elif case == "duplicate_id":
        with torch.no_grad():
            router(z[:2], batch[:2])
            with pytest.raises(RouterError, match="duplicate"):
                router(z[2:], [batch[2], batch[0]])
            with pytest.raises(RouterError, match="duplicate"):
                router(z[2:], [batch[3], batch[3]])
    elif case == "expected_mismatch":
        with torch.no_grad():
            router(z, batch)
        with pytest.raises(RouterError, match="first difference at position 0"):
            router.stop_recording(list(reversed(batch)))
        router.start_recording("val")                   # the failed stop ended the recording
        with torch.no_grad():
            router(z, batch)
        with pytest.raises(RouterError, match="recorded 4, expected 3"):
            router.stop_recording(batch[:3])
    router.discard_recording()
    router.start_recording("val")                       # usable again after every failure
    router.discard_recording()


@pytest.mark.criterion("M7-11")
@pytest.mark.parametrize("mode", ["dynamic", "cluster_hard", "none"])
def test_recorder_normal_path(mode, fake_hard_gate):
    k = 1 if mode == "none" else K
    val_ids, test_ids = make_ids("val", 10), make_ids("test", 2)
    hard_gate = None
    if mode == "cluster_hard":
        hard_gate = fake_hard_gate({sid: i % K for i, sid in enumerate(val_ids + test_ids)}, K)
    router = make_router(mode, num_experts=k, hard_gate=hard_gate)
    z = synthetic_z("recorder_normal", 10)

    # off by default: a training forward with gradients is allowed and nothing is kept
    router(z[:4], val_ids[:4])
    with pytest.raises(RouterError, match="without start_recording"):
        router.stop_recording([])

    calls_before = router.gate_calls
    router.start_recording("val")
    outputs = []
    with torch.no_grad():
        for start, end in ((0, 4), (4, 8), (8, 10)):     # three evaluation batches, the last one partial
            outputs.append(router(z[start:end], val_ids[start:end]))
    record = router.stop_recording(val_ids)
    assert isinstance(record, GateRecord)
    assert (record.split, record.routing_mode, record.num_experts) == ("val", mode, k)
    assert record.sample_ids == tuple(val_ids)
    assert record.omega.shape == (10, k) and record.z.shape == (10, D)
    for tensor in (record.omega, record.z):
        assert tensor.dtype == torch.float32 and tensor.device.type == "cpu" and not tensor.requires_grad
    assert torch.equal(record.omega, torch.cat(outputs))   # call order, bitwise
    assert torch.equal(record.z, z)
    assert record.gate_calls == (3 if mode == "dynamic" else 0)
    assert router.gate_calls - calls_before == record.gate_calls
    outputs[0].fill_(0.5)                                  # the record holds copies
    assert not torch.equal(record.omega[:4], outputs[0])

    # discard after a failed evaluation; then a new recording starts
    router.start_recording("test")
    with torch.no_grad():
        router(z[:2], test_ids)
    router.discard_recording()
    router.discard_recording()                             # no-op when not recording
    router.start_recording("test")
    empty = router.stop_recording([])
    assert empty.sample_ids == () and empty.omega.shape == (0, k) and empty.z.shape == (0, D)


@pytest.mark.criterion("M7-11")
def test_router_writes_no_files():
    """router.py has no file-writing call (open, np.save, torch.save, to_csv, ...) and no file-format imports."""
    tree = ast.parse(ROUTER_PY.read_text(encoding="utf-8"))
    forbidden_attrs = {"save", "savez", "savez_compressed", "to_csv", "write_text", "write_bytes", "dump", "open",
                       "write"}
    calls = []
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                calls.append(f"open at line {node.lineno}")
            elif isinstance(func, ast.Attribute) and func.attr in forbidden_attrs:
                calls.append(f".{func.attr} at line {node.lineno}")
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert calls == []
    assert not imported & {"json", "csv", "pickle", "numpy", "pandas", "io", "shutil"}


# ----------------------------------------------------------------------------------------------- #13
@pytest.mark.criterion("M7-13")
def test_state_dict_and_provenance(fake_hard_gate):
    dynamic = make_router()
    hard = make_router("cluster_hard", hard_gate=fake_hard_gate({"train:0": 0}, K))
    none = make_router("none", num_experts=1)
    assert list(dynamic.state_dict()) == ["gate.proj.weight"]
    assert dict(hard.state_dict()) == {} and dict(none.state_dict()) == {}
    assert dynamic.provenance() == {"routing_mode": "dynamic", "num_experts": K, "z_dim": D, "gate_shape": [K, D]}
    assert hard.provenance() == {"routing_mode": "cluster_hard", "num_experts": K, "z_dim": D, "gate_shape": None}
    for router in (dynamic, hard, none):
        assert json.loads(json.dumps(router.provenance())) == router.provenance()

    # reload into the same setting (M12 export, M9 checkpoint): the same omega
    trained = make_router(seed=SEED + 7)                    # stands for a trained gate
    reloaded = make_router()
    reloaded.load_state_dict(trained.state_dict(), strict=True)
    z, batch = synthetic_z("provenance_z", 5), make_ids("val", 5)
    assert reloaded.provenance() == trained.provenance()
    assert torch.equal(reloaded(z, batch), trained(z, batch))

    # another mode, K or z_dim: the provenance differs (M10 stops on it, §11.6.7) and strict loading fails
    others = {"num_experts_2": make_router(num_experts=2), "z_dim_32": make_router(z_dim=32),
              "cluster_hard": hard, "none": none}
    for name, other in others.items():
        assert other.provenance() != dynamic.provenance(), name
        with pytest.raises(RuntimeError):
            other.load_state_dict(dynamic.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        make_router().load_state_dict(hard.state_dict(), strict=True)
    # gate-less settings have empty state dicts: only the provenance tells them apart
    hard_k2 = make_router("cluster_hard", num_experts=2, hard_gate=fake_hard_gate({"train:0": 0}, 2))
    assert len({json.dumps(r.provenance(), sort_keys=True) for r in (hard, none, hard_k2)}) == 3
