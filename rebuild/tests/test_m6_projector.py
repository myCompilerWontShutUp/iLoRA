"""M6 projector (DESIGN.md §10.8 #1-#4).

#5 is ``test_m9_model.py::test_projector_only_for_hybrid``, #6-#8 are in
``test_m10_train.py``/``test_m9_model.py`` and #9 in ``tests/vessl`` (§17.6.9).
Synthetic inputs come from ``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ilora import seeding
from ilora.projector import Projector, build_projector, projector_param_count
from ilora.router import Router
from ilora.sections import RouterConfig

IN_DIM, OUT_DIM = 64, 4096          # SASRec hidden_size and Llama-2-7B hidden_size (§10.2)
SEED = 1234                         # paper_spec run.seed
EXPECTED_SHAPES = {
    "mlp_proj.0.weight": (4096, 64),
    "mlp_proj.0.bias": (4096,),
    "mlp_proj.2.weight": (4096, 4096),
    "mlp_proj.2.bias": (4096,),
}


@pytest.fixture(scope="module")
def projector() -> Projector:
    """``build_projector(64, 4096, 1234)``; tests that change a projector work on a copy."""
    return build_projector(IN_DIM, OUT_DIM, SEED)


def _item_repr(name: str, *shape: int) -> torch.Tensor:
    with seeding.torch_seed_scope(0, "test", name):
        return torch.randn(*shape)


@pytest.mark.criterion("M6-1")
def test_param_counts_and_shapes(projector):
    state = projector.state_dict()
    assert {key: tuple(value.shape) for key, value in state.items()} == EXPECTED_SHAPES
    n_params = sum(p.numel() for p in projector.parameters())
    assert n_params == 17_047_552 == projector_param_count(IN_DIM, OUT_DIM)
    for name, p in projector.named_parameters():
        assert p.dtype == torch.float32, name
        assert p.requires_grad, name
        assert p.device.type == "cpu", name
    assert (projector.in_dim, projector.out_dim) == (IN_DIM, OUT_DIM)
    # Linear -> GELU(erf) -> Linear, no dropout ([코드] U model/mlp_projector.py:6-10)
    layers = list(projector.mlp_proj)
    assert [type(m) for m in layers] == [nn.Linear, nn.GELU, nn.Linear]
    assert layers[1].approximate == "none"
    assert not any(isinstance(m, nn.Dropout) for m in projector.modules())
    assert projector_param_count(64, 64) == 8_320       # tiny LLaMA projector of M9 (13.8 #1)


@pytest.mark.criterion("M6-1")
@pytest.mark.parametrize("dims", [(0, 4096), (64, 0), (-1, 4096), (64.0, 4096), (True, 4096), (64, "4096")])
def test_build_rejects_bad_dims(dims):
    """§10.6.6: in_dim and out_dim must be positive ints (no hidden defaults)."""
    with pytest.raises(ValueError):
        build_projector(dims[0], dims[1], SEED)
    with pytest.raises(ValueError):
        projector_param_count(dims[0], dims[1])


@pytest.mark.criterion("M6-2")
def test_init_deterministic_and_isolated(projector):
    state = projector.state_dict()
    before = torch.get_rng_state()
    again = build_projector(IN_DIM, OUT_DIM, SEED)
    assert torch.equal(torch.get_rng_state(), before), "build_projector changed the global CPU RNG state"
    for key, value in again.state_dict().items():
        assert torch.equal(value, state[key]), key
    other = build_projector(IN_DIM, OUT_DIM, SEED + 1)
    for key, value in other.state_dict().items():
        assert not torch.equal(value, state[key]), key

    # Independent of other modules and of the global RNG: build the dynamic gate first and draw global numbers
    # in between (inside fork_rng so the test leaves the global state alone).
    with torch.random.fork_rng(devices=[]):
        Router(RouterConfig(routing_mode="dynamic"), num_experts=4, z_dim=IN_DIM, main_seed=SEED, hard_gate=None)
        torch.rand(1000)
        with_gate = build_projector(IN_DIM, OUT_DIM, SEED)
    for key, value in with_gate.state_dict().items():
        assert torch.equal(value, state[key]), key

    # Draw order of §10.6.2 on the CPU generator of derive_seed(seed, "init", "projector"):
    # weight 0, bias 0, weight 2, bias 2 (nn.Linear.reset_parameters).
    with seeding.torch_seed_scope(SEED, "init", "projector"):
        w0 = torch.empty(OUT_DIM, IN_DIM)
        nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
        b0 = torch.empty(OUT_DIM)
        nn.init.uniform_(b0, -1 / math.sqrt(IN_DIM), 1 / math.sqrt(IN_DIM))
        w2 = torch.empty(OUT_DIM, OUT_DIM)
        nn.init.kaiming_uniform_(w2, a=math.sqrt(5))
        b2 = torch.empty(OUT_DIM)
        nn.init.uniform_(b2, -1 / math.sqrt(OUT_DIM), 1 / math.sqrt(OUT_DIM))
    expected = {"mlp_proj.0.weight": w0, "mlp_proj.0.bias": b0, "mlp_proj.2.weight": w2, "mlp_proj.2.bias": b2}
    for key, value in expected.items():
        assert torch.equal(state[key], value), key


@pytest.mark.criterion("M6-3")
def test_init_ranges(projector):
    state = projector.state_dict()
    for key, bound in (("mlp_proj.0.weight", 1 / 8), ("mlp_proj.0.bias", 1 / 8),
                       ("mlp_proj.2.weight", 1 / 64), ("mlp_proj.2.bias", 1 / 64)):
        values = state[key]
        assert float(values.abs().max()) <= bound, key
        # uniform over the whole range: extreme values near the bound, std = bound / sqrt(3)
        assert float(values.abs().max()) > 0.99 * bound, key
        assert abs(float(values.std()) - bound / math.sqrt(3)) < 0.03 * bound, key
        assert abs(float(values.mean())) < 0.05 * bound, key


@pytest.mark.criterion("M6-4")
def test_forward_and_input_checks(projector):
    x = _item_repr("projector_input", 5, IN_DIM)
    y = projector(x)
    assert y.shape == (5, OUT_DIM) and y.dtype == torch.float32
    lin0, lin2 = projector.mlp_proj[0], projector.mlp_proj[2]
    expected = F.linear(F.gelu(F.linear(x, lin0.weight, lin0.bias)), lin2.weight, lin2.bias)
    assert torch.equal(y, expected)
    # more than two dimensions are allowed ("2차원 이상"), the last one is in_dim
    y3 = projector(x.view(5, 1, IN_DIM))
    assert y3.shape == (5, 1, OUT_DIM) and torch.equal(y3.view(5, OUT_DIM), y)

    bad_inputs = {
        "float16": x.half(),
        "int64": x.long(),
        "last_dim_63": _item_repr("projector_input_63", 5, 63),
        "empty": torch.empty(0, IN_DIM),
        "one_dim": x[0],
    }
    for name, bad in bad_inputs.items():
        with pytest.raises(ValueError, match="projector input"):
            projector(bad)

    half = copy.deepcopy(projector).half()
    with pytest.raises(RuntimeError, match="float32"):
        half(x)

    # CPU bf16 autocast (the CPU proxy of the GPU fp16 path, M6-9): the dtype follows autocast, values finite
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y_bf16 = projector(x)
    assert y_bf16.dtype == torch.bfloat16 and bool(torch.isfinite(y_bf16).all())
