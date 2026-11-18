"""M8 MoE-LoRA 층 (DESIGN.md §12.8 #1-#9; #10 also here as a layer-level check).

#10 is mapped to ``test_m9_model.py::test_gate_use_calls`` (§17.6.9); the test
here checks the same call counts on a bare 4.28.0 LLaMA without M9. #11-#12 are
in ``tests/vessl/test_vessl_model.py``. The upstream formula comes from
``refs_upstream.moelora_einsum`` (``U model/peft/tuners/moelora.py:750-783``).
Synthetic inputs come from ``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import refs_upstream
from ilora import router as router_module
from ilora import seeding
from ilora.moe_lora import (CANONICAL_TARGETS, GateContext, GateContextError, GateUse, MoELoRALinear,
                            attach_moe_lora, lora_named_parameters, lora_scaling)
from ilora.projector import projector_param_count
from ilora.router import ROW_SUM_ATOL, Router
from ilora.sections import LORA_TARGETS, LoraConfig, RouterConfig

R, ALPHA, K = 8, 32, 4                  # paper_spec lora.r, lora.alpha, lora.num_experts (D7)
D_IN, D_OUT = 16, 12                    # small layer for the numeric checks
PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def randn(name: str, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    with seeding.torch_seed_scope(0, "test", name):
        return torch.randn(*shape, dtype=dtype)


def softmax_omega(name: str, batch: int, k: int = K) -> torch.Tensor:
    return torch.softmax(randn(name, batch, k), dim=-1)          # float32 [B, K], rows sum to 1


def make_layer(*, k: int = K, dropout: float = 0.0, rule: str = "expert_rank", d_in: int = D_IN,
               d_out: int = D_OUT, random_b: bool = True) -> tuple[MoELoRALinear, GateContext]:
    """A wrapped Linear with A from reset_lora_parameters and (by default) a non-zero B."""
    ctx = GateContext(k)
    with seeding.torch_seed_scope(0, "test", "moe_lora_base"):
        base = nn.Linear(d_in, d_out, bias=False)
    layer = MoELoRALinear(base, r=R, num_experts=k, alpha=ALPHA, dropout=dropout, scaling_rule=rule,
                          gate_ctx=ctx, name="test.layer")
    with seeding.torch_seed_scope(0, "test", "moe_lora_init"):
        layer.reset_lora_parameters()
        if random_b:
            nn.init.normal_(layer.lora_B.weight, std=0.5)
    return layer, ctx


def lora_config(**changes) -> LoraConfig:
    values = dict(r=R, alpha=ALPHA, dropout=0.1, num_experts=K, scaling_rule="expert_rank",
                  target_modules=CANONICAL_TARGETS)
    values.update(changes)
    return LoraConfig(**values)


def tiny_llama():
    """CPU LLaMA with the 13.6.9 test shape (hidden 64, 2 layers, 4 heads, intermediate 128), fp32.

    Stands for M9 ``build_tiny_llama`` (the M8-7 fixture of §17.6.9) so that M8 is tested without M9.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=32000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, max_position_embeddings=1024, rms_norm_eps=1e-5,
                         initializer_range=0.02, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    with seeding.torch_seed_scope(0, "test", "tiny_llama"):
        model = LlamaForCausalLM(config)
    return model.requires_grad_(False)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max())


# ----------------------------------------------------------------------------------------------- #1
@pytest.mark.criterion("M8-1")
def test_canonical_targets_match_the_schema():
    """moe_lora.CANONICAL_TARGETS and sections.LORA_TARGETS are the same tuple (§12.6.1, D7)."""
    assert CANONICAL_TARGETS == LORA_TARGETS
    assert len(CANONICAL_TARGETS) * 32 == 224



@pytest.mark.criterion("M8-1")
def test_llama7b_lora_counts(llama7b_skeleton):
    skeleton = copy.deepcopy(llama7b_skeleton)      # the session fixture stays unwrapped for other tests
    ctx = GateContext(K)
    per_kind = dict.fromkeys(CANONICAL_TARGETS, 0)
    n_modules = 0
    for i, layer in enumerate(skeleton.model.layers):
        for target in CANONICAL_TARGETS:
            parent = getattr(layer, PARENT[target])
            wrapper = MoELoRALinear(getattr(parent, target), r=R, num_experts=K, alpha=ALPHA, dropout=0.1,
                                    scaling_rule="expert_rank", gate_ctx=ctx,
                                    name=f"model.layers.{i}.{PARENT[target]}.{target}")
            setattr(parent, target, wrapper)
            per_kind[target] += wrapper.lora_A.weight.numel() + wrapper.lora_B.weight.numel()
            n_modules += 1
            assert wrapper.base.weight.device.type == "meta" and not wrapper.base.weight.requires_grad
    assert n_modules == 224
    assert per_kind == {"q_proj": 2_097_152, "k_proj": 2_097_152, "v_proj": 2_097_152, "o_proj": 2_097_152,
                        "gate_proj": 3_866_624, "up_proj": 3_866_624, "down_proj": 3_866_624}
    named = lora_named_parameters(skeleton)
    assert len(named) == 448
    lora_total = sum(p.numel() for _, p in named)
    assert lora_total == 19_988_480
    for name, p in named:
        assert p.device.type == "cpu" and p.dtype == torch.float32 and p.requires_grad, name
    assert [name for name, _ in named[:14]] == [f"model.layers.0.{PARENT[t]}.{t}.lora_{ab}.weight"
                                                for t in CANONICAL_TARGETS for ab in "AB"]
    # E2 trainable parameters: LoRA + projector (M6) + gate (M7)
    gate_params = sum(p.numel() for p in Router(RouterConfig(routing_mode="dynamic"), num_experts=K, z_dim=64,
                                                  main_seed=1234, hard_gate=None).trainable_parameters())
    assert gate_params == 256
    total = lora_total + projector_param_count(64, 4096) + gate_params
    assert total == 37_036_288
    assert round(100 * gate_params / (total - gate_params), 6) == 0.000691      # < 1 % [논문 p.1 Abstract]
    assert lora_total == sum(R * (din + dout) for din, dout in
                             [(4096, 4096)] * 4 + [(4096, 11008)] * 2 + [(11008, 4096)]) * 32


# ----------------------------------------------------------------------------------------------- #2
@pytest.mark.criterion("M8-2")
@pytest.mark.parametrize("shape", [(3, 7), (2, 1)], ids=["3x7", "2x1"])
@pytest.mark.parametrize("dropout", [0.0, 0.1], ids=["p0", "p0.1"])
def test_matches_naive_and_upstream_fp64(dropout, shape):
    layer, ctx = make_layer(dropout=dropout)
    layer = layer.double().train()                       # base, A, B in fp64; omega stays float32 (contract)
    b, length = shape
    x0 = randn("fp64_x", b, length, D_IN, dtype=torch.float64)
    omega0 = softmax_omega("fp64_omega", b)
    upstream_grad = randn("fp64_grad", b, length, D_OUT, dtype=torch.float64)
    a0 = layer.lora_A.weight.detach().clone()
    b0 = layer.lora_B.weight.detach().clone()
    r_star, scaling = layer.r_star, layer.scaling

    x = x0.clone().requires_grad_(True)
    omega = omega0.clone().requires_grad_(True)
    with seeding.torch_seed_scope(0, "test", "dropout_mask"):    # the same mask for every implementation
        with ctx.use(omega):
            y = layer(x)
    (y * upstream_grad).sum().backward()
    ours = (y.detach(), x.grad, layer.lora_A.weight.grad, layer.lora_B.weight.grad, omega.grad)

    def naive(x, omega, a, b):
        """sum_k omega_k s B_k (A_k dropout(x)), one expert block at a time (식 12)."""
        xd = layer.lora_dropout(x)
        out = torch.zeros(x.shape[:-1] + (D_OUT,), dtype=x.dtype)
        for k in range(K):
            a_k, b_k = a[k * r_star:(k + 1) * r_star], b[:, k * r_star:(k + 1) * r_star]
            out = out + omega[:, k].view(-1, 1, 1) * scaling * F.linear(F.linear(xd, a_k), b_k)
        return out

    def upstream(x, omega, a, b):
        return refs_upstream.moelora_einsum(x, a, b, omega, scaling, R, K, dropout=layer.lora_dropout)

    for reference in (naive, upstream):
        x = x0.clone().requires_grad_(True)
        omega = omega0.clone().requires_grad_(True)
        a = a0.clone().requires_grad_(True)
        bb = b0.clone().requires_grad_(True)
        with seeding.torch_seed_scope(0, "test", "dropout_mask"):
            y_ref = layer.base(x) + reference(x, omega, a, bb)
        (y_ref * upstream_grad).sum().backward()
        ref = (y_ref.detach(), x.grad, a.grad, bb.grad, omega.grad)
        name = reference.__name__
        assert rel_err(ours[0], ref[0]) < 1e-12, f"{name}: output"
        for i, what in ((1, "x"), (2, "A"), (3, "B")):
            assert rel_err(ours[i], ref[i]) < 1e-12, f"{name}: grad {what}"
        assert ours[4].dtype == torch.float32 and ref[4].dtype == torch.float32
        assert rel_err(ours[4], ref[4]) < 1e-6, f"{name}: grad omega"


# ----------------------------------------------------------------------------------------------- #3
@pytest.mark.criterion("M8-3")
@pytest.mark.parametrize("rule", ["expert_rank", "total_rank"])
def test_k1_equals_plain_lora(rule):
    layer, ctx = make_layer(k=1, dropout=0.1, rule=rule)
    layer.train()
    assert layer.scaling == 4.0 and layer.r_star == R
    x = randn("k1_x", 3, 5, D_IN)
    with seeding.torch_seed_scope(0, "test", "k1_dropout"):
        with ctx.use(torch.ones(3, 1)):
            y = layer(x)
    with seeding.torch_seed_scope(0, "test", "k1_dropout"):
        plain = layer.base(x) + 4 * layer.lora_B(layer.lora_A(layer.lora_dropout(x)))   # h + (alpha/r) B A x (식 3)
    assert torch.equal(y, plain)


# ----------------------------------------------------------------------------------------------- #4
@pytest.mark.criterion("M8-4")
def test_scaling_rules_uniform_omega():
    assert lora_scaling(ALPHA, R, K, "expert_rank") == 16.0
    assert lora_scaling(ALPHA, R, K, "total_rank") == 4.0
    for rule in ("expert_rank", "total_rank"):
        assert lora_scaling(ALPHA, R, 1, rule) == 4.0
    for rule, factor in (("expert_rank", 1.0), ("total_rank", 0.25)):
        layer, ctx = make_layer(rule=rule)
        layer = layer.double()
        assert layer.scaling == {"expert_rank": 16.0, "total_rank": 4.0}[rule]
        x = randn("uniform_x", 3, 5, D_IN, dtype=torch.float64)
        with ctx.use(torch.full((3, K), 0.25)):
            y = layer(x)
        base_out = layer.base(x)
        standard = (ALPHA / R) * layer.lora_B(layer.lora_A(x))       # standard LoRA with the same A and B
        assert rel_err(y, base_out + factor * standard) < 1e-12, rule
        assert rel_err(y - base_out, factor * standard) < 1e-12, rule


# ----------------------------------------------------------------------------------------------- #5
@pytest.mark.criterion("M8-5")
def test_no_expert_output_saved():
    b, length, d_in, d_out = 2, 5, 7, 6                 # distinct sizes: [B, L, K, d_out] = [2, 5, 4, 6]
    layer, ctx = make_layer(dropout=0.1, d_in=d_in, d_out=d_out)
    layer.train()
    x = randn("saved_x", b, length, d_in).requires_grad_(True)
    omega = softmax_omega("saved_omega", b).requires_grad_(True)
    saved: list[tuple[int, ...]] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with ctx.use(omega):
            y = layer(x)
    y.sum().backward()
    expert_shape = (b, length, K, d_out)
    assert saved, "the hook saw no saved tensor"
    assert expert_shape not in saved
    assert all(len(s) < 4 for s in saved), saved
    # control: the released formula keeps exactly that tensor for backward (§12.5)
    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        refs_upstream.moelora_einsum(x, layer.lora_A.weight, layer.lora_B.weight, omega, layer.scaling, R, K)
    assert expert_shape in saved


# ----------------------------------------------------------------------------------------------- #6
@pytest.mark.criterion("M8-6")
def test_zero_b_initial_state():
    layer, ctx = make_layer(dropout=0.1, random_b=False)     # reset_lora_parameters: B = 0
    layer.train()
    assert torch.count_nonzero(layer.lora_B.weight) == 0
    x = randn("zero_b_x", 3, 4, D_IN).requires_grad_(True)
    omega = softmax_omega("zero_b_omega", 3).requires_grad_(True)
    with ctx.use(omega):
        y = layer(x)
    assert torch.equal(y, layer.base(x))
    (y * randn("zero_b_grad", 3, 4, D_OUT)).sum().backward()
    assert torch.count_nonzero(omega.grad) == 0
    assert torch.count_nonzero(layer.lora_A.weight.grad) == 0
    assert torch.count_nonzero(layer.lora_B.weight.grad) > 0


# ----------------------------------------------------------------------------------------------- #7
@pytest.mark.criterion("M8-7")
def test_lora_init():
    cfg = lora_config()
    llm = tiny_llama()
    before = torch.get_rng_state()
    names = attach_moe_lora(llm, cfg, GateContext(K), main_seed=1234)
    assert torch.equal(torch.get_rng_state(), before), "attach_moe_lora changed the global CPU RNG state"
    assert names == [f"model.layers.{i}.{PARENT[t]}.{t}" for i in range(2) for t in CANONICAL_TARGETS]
    params = lora_named_parameters(llm)
    assert [name for name, _ in params] == [f"{n}.lora_{ab}.weight" for n in names for ab in "AB"]
    for name in names:
        module = llm.get_submodule(name)
        assert isinstance(module, MoELoRALinear) and module.name == name
        d_in = module.base.in_features
        assert d_in == (128 if name.endswith("down_proj") else 64)
        assert float(module.lora_A.weight.abs().max()) <= 1 / math.sqrt(d_in)
        assert float(module.lora_A.weight.abs().max()) > 0.9 / math.sqrt(d_in)
        assert torch.count_nonzero(module.lora_B.weight) == 0

    # canonical order q, k, v, o, gate, up, down per layer from derive_seed(1234, "init", "lora")
    with seeding.torch_seed_scope(1234, "init", "lora"):
        expected = []
        for name in names:
            weight = torch.empty(R, llm.get_submodule(name).base.in_features)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            expected.append(weight)
    for name, weight in zip(names, expected):
        assert torch.equal(llm.get_submodule(name).lora_A.weight, weight), name

    same = tiny_llama()
    attach_moe_lora(same, cfg, GateContext(K), main_seed=1234)
    for (name, p), (name2, q) in zip(params, lora_named_parameters(same)):
        assert name == name2 and torch.equal(p, q), name
    other = tiny_llama()
    attach_moe_lora(other, cfg, GateContext(K), main_seed=1235)
    for (name, p), (_, q) in zip(params, lora_named_parameters(other)):
        if ".lora_A." in name:
            assert not torch.equal(p, q), name


# ----------------------------------------------------------------------------------------------- #8
@pytest.mark.criterion("M8-8")
def test_bf16_autocast_proxy():
    ctx = GateContext(K)
    with seeding.torch_seed_scope(0, "test", "bf16_base"):
        base = nn.Linear(D_IN, D_OUT, bias=False).to(torch.bfloat16)
    layer = MoELoRALinear(base, r=R, num_experts=K, alpha=ALPHA, dropout=0.1, scaling_rule="expert_rank",
                          gate_ctx=ctx, name="test.bf16")
    with seeding.torch_seed_scope(0, "test", "bf16_init"):
        layer.reset_lora_parameters()
        nn.init.normal_(layer.lora_B.weight, std=0.5)
    layer.train()
    x = randn("bf16_x", 2, 5, D_IN).to(torch.bfloat16)
    omega = softmax_omega("bf16_omega", 2).requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with ctx.use(omega):
            y = layer(x)
    assert y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all())
    (y.float() * randn("bf16_grad", 2, 5, D_OUT)).sum().backward()
    assert layer.lora_A.weight.dtype == torch.float32 and layer.lora_A.weight.grad.dtype == torch.float32
    assert layer.lora_B.weight.grad.dtype == torch.float32
    assert omega.grad is not None and omega.grad.dtype == torch.float32
    assert not base.weight.requires_grad and base.weight.grad is None
    with pytest.raises(RuntimeError, match="autocast"):          # bf16 input outside autocast: a clear error
        with ctx.use(omega.detach()):
            layer(x)
    assert not ctx.is_set


# ----------------------------------------------------------------------------------------------- #9
ERROR_CASES = ["empty_context", "batch_mismatch", "nested_use", "nan_omega", "row_sum", "float64_omega",
               "device_mismatch", "r_not_divisible", "unknown_target"]


@pytest.mark.criterion("M8-9")
@pytest.mark.parametrize("case", ERROR_CASES)
def test_errors(case):
    layer, ctx = make_layer()
    x = randn("errors_x", 3, 2, D_IN)
    good = softmax_omega("errors_omega", 3)
    expected: type[Exception] = GateContextError
    if case == "empty_context":
        def run():
            layer(x)
    elif case == "batch_mismatch":
        def run():
            with ctx.use(softmax_omega("errors_omega2", 2)):
                layer(x)
    elif case == "nested_use":
        def run():
            with ctx.use(good):
                with ctx.use(good):
                    pass
    elif case in ("nan_omega", "row_sum", "float64_omega"):
        bad = good.clone()
        if case == "nan_omega":
            bad[1, 2] = float("nan")
        elif case == "row_sum":
            bad = torch.ones(3, K)
        else:
            bad = good.double()

        def run():
            with ctx.use(bad):
                layer(x)
    elif case == "device_mismatch":
        def run():
            with ctx.use(good):
                layer(torch.empty(3, 2, D_IN, device="meta"))
    elif case == "r_not_divisible":
        expected = ValueError

        def run():
            MoELoRALinear(nn.Linear(D_IN, D_OUT, bias=False), r=6, num_experts=K, alpha=ALPHA, dropout=0.0,
                          scaling_rule="expert_rank", gate_ctx=ctx, name="test.r6")
    else:
        expected = ValueError
        llm = tiny_llama()

        def run():
            attach_moe_lora(llm, lora_config(target_modules=("q_proj", "foo_proj")), ctx, main_seed=1234)
    with pytest.raises(expected):
        run()
    assert not ctx.is_set
    if case == "unknown_target":
        assert not any(isinstance(m, MoELoRALinear) for m in llm.modules())


@pytest.mark.criterion("M8-9")
def test_gate_context_rules():
    """The row-sum tolerance is the one constant of router.py (§11.7, §12.6.4)."""
    import ilora.moe_lora as moe_lora_module

    assert moe_lora_module.ROW_SUM_ATOL is router_module.ROW_SUM_ATOL == 1e-5
    ctx = GateContext(K)
    assert ctx.num_experts == K and not ctx.is_set
    with pytest.raises(AttributeError):
        ctx.num_experts = 2                                  # read only
    near = torch.full((2, K), 0.25)
    near[0, 3] += 0.9 * ROW_SUM_ATOL                         # row sum inside the shared tolerance
    with ctx.use(near) as use:
        assert isinstance(use, GateUse) and use.batch_size == 2 and use.calls == 0
        assert ctx.is_set and ctx.get() is near and use.calls == 1
    assert not ctx.is_set
    far = torch.full((2, K), 0.25)
    far[0, 3] += 2 * ROW_SUM_ATOL
    for bad in (far, torch.full((2, K + 1), 1 / (K + 1)), torch.full((2, 1, K), 0.25), torch.empty(0, K),
                torch.tensor([[1.5, -0.5, 0.0, 0.0]] * 2)):
        with pytest.raises(GateContextError):
            with ctx.use(bad):
                pass
        assert not ctx.is_set
    with pytest.raises(GateContextError, match="empty"):
        ctx.get()
    with pytest.raises(RuntimeError, match="boom"):          # an exception in the block still clears omega
        with ctx.use(near):
            raise RuntimeError("boom")
    assert not ctx.is_set
    for bad_k in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            GateContext(bad_k)


@pytest.mark.criterion("M8-9")
def test_constructor_and_attach_rejects(llama7b_skeleton):
    ctx = GateContext(K)
    base = nn.Linear(D_IN, D_OUT, bias=False)
    good = dict(r=R, num_experts=K, alpha=ALPHA, dropout=0.1, scaling_rule="expert_rank", gate_ctx=ctx, name="t")
    bad_kwargs = [dict(r=0), dict(r=-8), dict(r=8.0), dict(alpha=0), dict(alpha=-32), dict(dropout=1.0),
                  dict(dropout=-0.1), dict(dropout=True), dict(scaling_rule="rank"), dict(gate_ctx=GateContext(2)),
                  dict(name="")]
    for change in bad_kwargs:
        with pytest.raises(ValueError):
            MoELoRALinear(base, **{**good, **change})

    class LinearSubclass(nn.Linear):
        pass

    with pytest.raises(ValueError, match="exactly torch.nn.Linear"):
        MoELoRALinear(LinearSubclass(D_IN, D_OUT, bias=False), **good)
    wrapped = MoELoRALinear(base, **good)
    with pytest.raises(ValueError, match="already a MoELoRALinear"):
        MoELoRALinear(wrapped, **good)
    with pytest.raises(IndexError):
        wrapped.expert_slices(K)
    # attach: the LLM must be on CPU (the meta skeleton is not), K must match, targets must exist and be unique
    with pytest.raises(ValueError, match="on CPU"):
        attach_moe_lora(llama7b_skeleton, lora_config(), GateContext(K), main_seed=1234)
    llm = tiny_llama()
    with pytest.raises(ValueError, match="experts"):
        attach_moe_lora(llm, lora_config(), GateContext(2), main_seed=1234)
    with pytest.raises(ValueError, match="duplicates"):
        attach_moe_lora(llm, lora_config(target_modules=("q_proj", "q_proj")), GateContext(K), main_seed=1234)
    with pytest.raises(ValueError, match="duplicates"):
        attach_moe_lora(llm, lora_config(target_modules=()), GateContext(K), main_seed=1234)
    attach_moe_lora(llm, lora_config(target_modules=("q_proj",)), GateContext(K), main_seed=1234)
    with pytest.raises(ValueError, match="already a MoELoRALinear"):       # attaching twice
        attach_moe_lora(llm, lora_config(target_modules=("q_proj",)), GateContext(K), main_seed=1234)
    assert not any(isinstance(m, MoELoRALinear) for m in llama7b_skeleton.modules())


@pytest.mark.criterion("M8-9")
def test_expert_slices_are_the_expert_blocks():
    layer, ctx = make_layer()
    x = randn("slices_x", 2, 3, D_IN)
    for k in range(K):
        a_k, b_k = layer.expert_slices(k)
        assert a_k.shape == (layer.r_star, D_IN) and b_k.shape == (D_OUT, layer.r_star)
        one_hot = torch.zeros(2, K)
        one_hot[:, k] = 1.0
        with ctx.use(one_hot):
            y = layer(x)
        expected = layer.base(x) + layer.scaling * F.linear(F.linear(x, a_k), b_k)
        assert float((y - expected).abs().max()) <= 1e-5 * float(expected.abs().max())


# ----------------------------------------------------------------------------------------------- #10
@pytest.mark.criterion("M8-10")
def test_tiny_llama_forward_and_generate_calls():
    """Every wrapped layer reads omega once per forward: 14 calls per step on the 2-layer LLaMA, 14 x T for T steps."""
    llm = tiny_llama()
    ctx = GateContext(K)
    attach_moe_lora(llm, lora_config(), ctx, main_seed=1234)
    llm.eval()
    with seeding.torch_seed_scope(0, "test", "tiny_llama_ids"):
        input_ids = torch.randint(3, 32000, (2, 6))
    mask = torch.ones_like(input_ids)
    omega = softmax_omega("tiny_llama_omega", 2)
    with torch.no_grad():
        with ctx.use(omega) as use:
            hidden = llm.model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)
        assert use.calls == 14 and hidden.last_hidden_state.shape == (2, 6, 64)
        with ctx.use(omega) as use:
            seq = llm.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=5, min_new_tokens=5,
                               do_sample=False, num_beams=1, pad_token_id=0, eos_token_id=2, use_cache=True)
    steps = seq.shape[1] - input_ids.shape[1]
    assert steps == 5 and use.calls == 14 * steps
    assert not ctx.is_set
    with pytest.raises(GateContextError):
        with torch.no_grad():
            llm.model(input_ids=input_ids, attention_mask=mask, use_cache=False)


def host_reads(fn) -> int:
    """``aten::item`` calls while ``fn()`` runs (CPU profiler): each is a device-to-host read, on CUDA a stream
    synchronisation."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    return sum(event.count for event in prof.key_averages() if event.key == "aten::item")


@pytest.mark.criterion("M8-10")
@pytest.mark.parametrize("k", [K, 1], ids=["K4", "K1"])
def test_layer_call_reads_nothing_back(k):
    """A wrapped layer's forward and backward read no value back to the host: omega is checked once in
    ``GateContext.use`` (§12.6.4, outside the profiled calls), never per layer call (§12.7: not 224 layers x every
    generated token). ``repeat_interleave`` without ``output_size`` would read its repeats twice per call."""
    layer, ctx = make_layer(k=k, dropout=0.1)
    layer.train()
    x = randn("host_reads_x", 3, 5, D_IN)
    omega = softmax_omega("host_reads_omega", 3, k).requires_grad_(True)
    assert host_reads(lambda: float(x.sum())) == 1                  # control: the profiler sees a host read
    with ctx.use(omega):
        assert host_reads(lambda: layer(x)) == 0
        y = layer(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):            # the CPU stand-in of the fp16 autocast path
            assert host_reads(lambda: layer(x)) == 0
    assert host_reads(lambda: y.sum().backward()) == 0
    assert omega.grad is not None and layer.lora_A.weight.grad is not None
