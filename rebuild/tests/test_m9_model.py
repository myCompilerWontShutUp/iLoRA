"""M9 LLM 연결 (DESIGN.md §13.8 #1-#13; also M5-1 last clause, M6-5, M6-8, M8-3 model level, M8-10).

The model is the CPU test backbone (``model.backbone = tiny_random``: hidden 64, 2 layers, CPU fp32,
``("init", "tiny_llama")``) built by the ``tiny_model`` fixture with the real SASRec state, the test
tokenizer and batches made by M1-M5 from the ``smoke_cpu`` data (§13.6.9, §17.6.2). #14-#17 are VESSL
tests in ``tests/vessl/test_vessl_model.py``. The insertion reference is ``refs_upstream.wrap_emb_loop``
(``U model/model_interface.py:453-461``). Synthetic values come from
``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import refs_upstream
from ilora import seeding
from ilora.batching import (IGNORE_INDEX, EvalBatch, TrainBatch, collate_eval, collate_train, encode_eval,
                            encode_train)
from ilora.candidates import candidates_for
from ilora.moe_lora import GateContext, MoELoRALinear, lora_named_parameters
from ilora.prompts import build_prompt, choose_template

SEED = 1234                       # paper_spec / smoke_cpu run.seed
K = 4                             # lora.num_experts
TINY_LORA = 17_408                # 2 layers x (4 x 1,024 + 2 x 1,536 + 1,536)
TINY_PROJECTOR = 8_320            # projector_param_count(64, 64)
GATE = 256                        # W_g [4, 64]
CLUSTER_HARD = ("router.routing_mode=cluster_hard", "cluster.artifact_dir=unused_by_m9/cluster")
K1 = ("lora.num_experts=1", "router.routing_mode=none")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class Batches:
    """Real ``TrainBatch``/``EvalBatch`` of the smoke_cpu data, made with the public M2, M4 and M5 functions."""

    def __init__(self, cfg, data, bundle, templates) -> None:
        self.cfg, self.data, self.bundle, self.templates = cfg, data, bundle, templates

    def _prompt(self, sample, epoch, hybrid):
        cfg = self.cfg
        cands = candidates_for(sample, main_seed=cfg.run.seed, epoch=epoch, cfg=cfg.candidates)
        k = choose_template(cfg.prompt, cfg.run.seed, len(self.templates.templates), sample, epoch)
        return cands, k, build_prompt(sample, cands, self.data.catalog, self.templates, k, hybrid)

    def train(self, *, hybrid: bool, n: int = 4, start: int = 0, samples=None, epoch: int = 0) -> TrainBatch:
        samples = list(samples) if samples is not None else list(self.data.train[start:start + n])
        cands, tids, encoded = [], [], []
        for s in samples:
            c, k, prompt = self._prompt(s, epoch, hybrid)
            cands.append(c)
            tids.append(k)
            encoded.append(encode_train(self.bundle, prompt, self.data.catalog.titles[s.target],
                                        self.cfg.batching.label_mode))
        return collate_train(samples, cands, tids, encoded, self.bundle, hybrid=hybrid, tau=1)

    def eval(self, split: str = "val", *, hybrid: bool = True, n: int = 4, start: int = 0,
             samples=None) -> EvalBatch:
        samples = list(samples) if samples is not None else list(self.data.samples(split)[start:start + n])
        cands, tids, encoded = [], [], []
        for s in samples:
            c, k, prompt = self._prompt(s, None, hybrid)
            cands.append(c)
            tids.append(k)
            encoded.append(encode_eval(self.bundle, prompt))
        return collate_eval(samples, cands, tids, encoded, self.data.catalog, self.bundle, hybrid=hybrid)


@pytest.fixture
def batches(cpu_cfg, smoke_data, bundle, templates) -> Batches:
    return Batches(cpu_cfg, smoke_data, bundle, templates)


@pytest.fixture
def hard_gate_table(smoke_data):
    """sample_id -> cluster (= expert) for every smoke sample, for the ``fake_hard_gate`` of cluster_hard."""
    return {s.sample_id: s.row % K for split in ("train", "val", "test") for s in smoke_data.samples(split)}


@contextlib.contextmanager
def record_gate_uses(model):
    """Collect the ``GateUse`` of every ``model.gate_ctx.use`` block (calls = wrapped-layer reads of omega)."""
    uses = []
    original = model.gate_ctx.use

    @contextlib.contextmanager
    def spy(omega):
        with original(omega) as use:
            uses.append(use)
            yield use

    model.gate_ctx.use = spy          # instance attribute in front of the method
    try:
        yield uses
    finally:
        del model.gate_ctx.use


def model_logits(model, batch) -> torch.Tensor:
    """Full logits ``[B, L, V]`` of the model path (checks, z/omega, inputs_embeds, LLaMA inside the gate)."""
    model._check_batch(batch, "train" if isinstance(batch, TrainBatch) else "eval")
    on_device = batch.to(model.device)
    with torch.no_grad():
        z, omega = model._z_omega(on_device)
        emb = model._embed_inputs(on_device)
        with model.gate_ctx.use(omega):
            return model.llm(inputs_embeds=emb, attention_mask=on_device.attention_mask, use_cache=False,
                             return_dict=True).logits


def lora_tensors(model, which: str = "AB"):
    return [(n, p) for n, p in lora_named_parameters(model.llm) if n.endswith(tuple(f"lora_{w}.weight" for w in which))]


def randomize_lora_b(model, name: str = "m9_lora_b", std: float = 0.02) -> None:
    """Non-zero B so that the LoRA term and omega matter (B = 0 at init, §12.6.3)."""
    with torch.no_grad(), seeding.torch_seed_scope(0, "test", name):
        for _, p in lora_tensors(model, "B"):
            p.normal_(0.0, std)


def perturb_trainable(model, name: str, std: float = 0.01) -> None:
    """Seeded noise on every trainable parameter (the checkpoint tests need values different from init)."""
    with torch.no_grad(), seeding.torch_seed_scope(0, "test", name):
        for items in model.param_groups().values():
            for _, p in items:
                p.add_(torch.randn(p.shape, dtype=p.dtype) * std)


def grad_is_nonzero(p: torch.Tensor) -> bool:
    return p.grad is not None and bool((p.grad != 0).any())


class PlainLoRA(nn.Module):
    """Standard LoRA ``h = W x + (alpha / r) B A x`` [논문 p.3 식 3] with given A and B (eval, no dropout)."""

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, scaling: float) -> None:
        super().__init__()
        self.base = base
        self.A = nn.Parameter(A.detach().clone(), requires_grad=False)
        self.B = nn.Parameter(B.detach().clone(), requires_grad=False)
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


def plain_lora_copy(llm: nn.Module) -> nn.Module:
    """A copy of ``llm`` with every MoE-LoRA layer replaced by ``PlainLoRA`` holding the same A and B."""
    plain = copy.deepcopy(llm)
    for name, module in list(plain.named_modules()):
        if isinstance(module, MoELoRALinear):
            parent_name, _, attr = name.rpartition(".")
            setattr(plain.get_submodule(parent_name), attr,
                    PlainLoRA(module.base, module.lora_A.weight, module.lora_B.weight, module.scaling))
    return plain


# ----------------------------------------------------------------------------------------------- #1
@pytest.mark.criterion("M9-1")
@pytest.mark.parametrize("mode", ["dynamic", "cluster_hard", "none"])
def test_tiny_build(mode, tiny_model, fake_hard_gate, hard_gate_table):
    gate = None
    if mode == "dynamic":
        model = tiny_model()
    elif mode == "cluster_hard":
        gate = fake_hard_gate(hard_gate_table, K)
        model = tiny_model(*CLUSTER_HARD, hard_gate=gate)
    else:
        model = tiny_model(*K1)
    llm = model.llm
    wrapped = [n for n, m in llm.named_modules() if isinstance(m, MoELoRALinear)]
    assert model.n_wrapped == len(wrapped) == 14
    groups = model.param_groups()
    assert list(groups) == (["projector", "gate", "lora"] if mode == "dynamic" else ["projector", "lora"])
    params = dict(model.named_parameters())
    for group, items in groups.items():
        names = [n for n, _ in items]
        assert names == sorted(names), group
        for name, p in items:
            assert params[name] is p and p.requires_grad and p.dtype == torch.float32, name
    lora = groups["lora"]
    assert len(lora) == 28 and sum(p.numel() for _, p in lora) == TINY_LORA
    assert all(n.startswith("llm.model.layers.") and n.endswith(("lora_A.weight", "lora_B.weight")) for n, _ in lora)
    assert [n for n, _ in groups["projector"]] == ["projector.mlp_proj.0.bias", "projector.mlp_proj.0.weight",
                                                   "projector.mlp_proj.2.bias", "projector.mlp_proj.2.weight"]
    assert sum(p.numel() for _, p in groups["projector"]) == TINY_PROJECTOR
    if mode == "dynamic":
        assert [n for n, _ in groups["gate"]] == ["omega_source.gate.proj.weight"]
        assert groups["gate"][0][1].numel() == GATE
    grouped = {n for items in groups.values() for n, _ in items}
    assert sorted(n for n, p in params.items() if p.requires_grad) == sorted(grouped)     # nothing else trains
    assert not llm.get_input_embeddings().weight.requires_grad and not llm.lm_head.weight.requires_grad
    assert not llm.model.norm.weight.requires_grad
    assert not any(p.requires_grad for p in model.sasrec.parameters())
    assert llm.config.vocab_size == 32005
    assert tuple(llm.get_input_embeddings().weight.shape) == (32005, 64) == tuple(llm.lm_head.weight.shape)
    assert (llm.config.hidden_size, llm.config.num_hidden_layers, llm.config.num_attention_heads,
            llm.config.intermediate_size) == (64, 2, 4, 128)
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    report = model.parameter_report()
    assert (report["lora"], report["projector"], report["gate"]) == (TINY_LORA, TINY_PROJECTOR,
                                                                     GATE if mode == "dynamic" else 0)
    assert report["trainable"] == sum(p.numel() for items in groups.values() for _, p in items)
    assert (report["lora_tensors"], report["wrapped_modules"], report["new_token_rows"]) == (28, 14, 2 * 5 * 64)
    assert model.device == torch.device("cpu") and not model.training and model.sasrec_is_eval()
    assert model.omega_source.routing_mode == mode and model.num_experts == (1 if mode == "none" else K)
    gen = model.gen_config
    assert (gen.do_sample, gen.num_beams, gen.max_new_tokens, gen.min_new_tokens, gen.repetition_penalty) == (
        False, 1, 64, 1, 1.0)
    assert (gen.pad_token_id, gen.bos_token_id, gen.eos_token_id, gen.use_cache) == (32000, 1, 2, True)
    gen.max_new_tokens = 3                                     # gen_config is a copy: the model keeps 64
    assert model.gen_config.max_new_tokens == 64
    with pytest.raises(AttributeError):
        model.device = torch.device("cpu")                    # read-only
    if gate is not None:
        assert gate.lookups == 0                               # building reads no cluster assignment


# ---------------------------------------------------------------------------------- M5-1 last clause
class _LenOnly:
    """Stands for a tokenizer whose length is not 32,005 (M9 reads only ``len(tokenizer)``)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


@pytest.mark.criterion("M5-1")
@pytest.mark.parametrize("case", ["tokenizer_length", "vocab_size", "new_token_ids", "below_base_vocab", "bos", "eos",
                                  "pad_is_eos", "named_order"])
def test_build_rechecks_token_ids(case, bundle, cpu_cfg, smoke_data, monkeypatch):
    """build_model step 1 re-checks ``bundle.ids`` before loading anything (§13.6.2, M5 9.8 #1)."""
    from ilora import model as model_module

    assert model_module.check_token_ids(bundle) is bundle.ids
    ids = bundle.ids
    replace = dataclasses.replace
    bad = {
        "tokenizer_length": lambda: replace(bundle, tokenizer=_LenOnly(32004)),
        "vocab_size": lambda: replace(bundle, ids=replace(ids, vocab_size=32006)),
        "new_token_ids": lambda: replace(bundle, ids=replace(ids, new_token_ids=(32000, 32001, 32002, 32003, 32005))),
        "below_base_vocab": lambda: replace(bundle, ids=replace(ids, pad=31999,
                                                                 new_token_ids=(31999, 32001, 32002, 32003, 32004))),
        "bos": lambda: replace(bundle, ids=replace(ids, bos=3)),
        "eos": lambda: replace(bundle, ids=replace(ids, eos=5)),
        "pad_is_eos": lambda: replace(bundle, ids=replace(ids, pad=2)),
        "named_order": lambda: replace(bundle, ids=replace(ids, ph=32002, history_emb=32001)),
    }[case]()

    def must_not_run(*args, **kwargs):
        raise AssertionError("build_model went past step 1")

    monkeypatch.setattr(model_module, "build_tiny_llama", must_not_run)
    monkeypatch.setattr(model_module, "load_base_llm", must_not_run)
    with pytest.raises(model_module.ModelError, match="13.6.2 step 1"):
        model_module.build_model(cpu_cfg, bad, smoke_data.catalog, llm_path=None)


# ----------------------------------------------------------------------------------------------- #2
@pytest.mark.criterion("M9-2")
def test_new_token_rows(tiny_model):
    from ilora.model import GENERATED_FROZEN_KEYS, build_tiny_llama, resize_new_tokens

    a, b = tiny_model(), tiny_model()
    fa, fb = a.generated_frozen_state(), b.generated_frozen_state()
    assert tuple(fa) == GENERATED_FROZEN_KEYS
    assert fa["generated_frozen.token_ids"].tolist() == [32000, 32001, 32002, 32003, 32004]
    for key in fa:
        assert torch.equal(fa[key], fb[key]), key                       # same seed: bitwise
    assert fa["generated_frozen.embed_rows"].shape == (5, 64) == fa["generated_frozen.lm_head_rows"].shape
    base = build_tiny_llama(32000, SEED)                                # the LLaMA before the resize
    old_embed = base.get_input_embeddings().weight.detach().clone()
    old_head = base.lm_head.weight.detach().clone()
    embed, head = a.llm.get_input_embeddings().weight.detach(), a.llm.lm_head.weight.detach()
    assert torch.equal(embed[:32000], old_embed) and torch.equal(head[:32000], old_head)
    assert torch.equal(embed[32000:], fa["generated_frozen.embed_rows"])
    assert torch.equal(head[32000:], fa["generated_frozen.lm_head_rows"])
    assert a.llm.get_input_embeddings().padding_idx is None
    for rows in (embed[32000:], head[32000:]):                         # N(0, 0.02) of 4.28.0 _init_weights
        assert 0.015 < float(rows.std()) < 0.025 and abs(float(rows.mean())) < 0.01

    def resized(seed: int, dtype: torch.dtype = torch.float32):
        llm = copy.deepcopy(base).to(dtype)
        before = torch.get_rng_state()
        resize_new_tokens(llm, 32005, seed)
        assert torch.equal(torch.get_rng_state(), before)              # the seed scope restores the CPU RNG
        return llm.get_input_embeddings().weight.detach(), llm.lm_head.weight.detach()

    same_embed, same_head = resized(SEED)
    other_embed, other_head = resized(SEED + 1)
    assert torch.equal(same_embed, embed) and torch.equal(same_head, head)
    assert not torch.equal(other_embed[32000:], embed[32000:]) and not torch.equal(other_head[32000:], head[32000:])
    assert torch.equal(other_embed[:32000], old_embed) and torch.equal(other_head[:32000], old_head)
    # fp16 CPU model: the same rules (CPU fp16 normal_ works in torch 2.0.0, §13.6.2 step 3)
    e1, h1 = resized(SEED, torch.float16)
    e2, h2 = resized(SEED, torch.float16)
    e3, h3 = resized(SEED + 1, torch.float16)
    assert e1.dtype == h1.dtype == torch.float16
    assert torch.equal(e1[32000:], e2[32000:]) and torch.equal(h1[32000:], h2[32000:])
    assert not torch.equal(e1[32000:], e3[32000:]) and not torch.equal(h1[32000:], h3[32000:])
    assert torch.equal(e1[:32000], old_embed.half()) and torch.equal(h1[:32000], old_head.half())
    for rows in (e1[32000:], h1[32000:]):
        assert 0.015 < float(rows.float().std()) < 0.025


# ----------------------------------------------------------------------------------------------- #3
@pytest.mark.criterion("M9-3")
def test_initial_logits_equal_unwrapped(tiny_model, batches):
    from ilora.model import build_tiny_llama, resize_new_tokens

    model = tiny_model()
    assert all(bool((p == 0).all()) for _, p in lora_tensors(model, "B"))            # B = 0 at init
    unwrapped = build_tiny_llama(32000, SEED)
    resize_new_tokens(unwrapped, 32005, SEED)
    unwrapped.eval()
    for batch in (batches.train(hybrid=True), batches.train(hybrid=False), batches.eval("val")):
        wrapped_logits = model_logits(model, batch)
        on_device = batch.to(model.device)
        with torch.no_grad():
            emb = model._embed_inputs(on_device)
            plain_logits = unwrapped(inputs_embeds=emb, attention_mask=on_device.attention_mask, use_cache=False,
                                     return_dict=True).logits
        assert torch.equal(wrapped_logits, plain_logits)


# ----------------------------------------------------------------------------------------------- #4
@pytest.mark.criterion("M9-4")
def test_loss_equals_hf_loss(tiny_model, batches):
    """The label-position loss equals the HF internal loss (labels=) with the same dropout masks (< 1e-6)."""
    model = tiny_model()
    randomize_lora_b(model)
    for hybrid in (True, False):
        batch = batches.train(hybrid=hybrid)
        on_device = batch.to(model.device)
        for training in (True, False):
            model.train(training)
            with seeding.torch_seed_scope(0, "test", "m9_dropout"):
                out = model.train_forward(batch)
            with seeding.torch_seed_scope(0, "test", "m9_dropout"):
                z, omega = model._z_omega(on_device)
                emb = model._embed_inputs(on_device)
                with model.gate_ctx.use(omega):
                    hf = model.llm(inputs_embeds=emb, attention_mask=on_device.attention_mask,
                                   labels=on_device.labels, use_cache=False, return_dict=True).loss
            assert out.loss.dim() == 0 and out.loss.dtype == torch.float32 and out.loss.requires_grad
            assert out.n_label_tokens == int((batch.labels != IGNORE_INDEX).sum())
            assert abs(float(out.loss) - float(hf)) < 1e-6, (hybrid, training, float(out.loss), float(hf))
            assert out.omega.shape == (4, K) and not out.omega.requires_grad and torch.equal(out.omega, omega.detach())
            assert out.z.shape == (4, 64) and out.z.dtype == torch.float32 and not out.z.requires_grad
            if training:   # the dropout masks matter: another seed gives another loss
                with seeding.torch_seed_scope(0, "test", "m9_dropout_other"):
                    assert float(model.train_forward(batch).loss) != float(out.loss)
    model.eval()


# ----------------------------------------------------------------------------------------------- #5
@pytest.mark.criterion("M9-5")
def test_embed_matches_reference_wrap_emb(tiny_model, batches):
    """Hybrid inputs_embeds = the released per-row loop ``wrap_emb`` fed with the behaviour tokens (bitwise).

    The behaviour tokens of the reference are the projector outputs of the same item rows in the same order (one
    call on [Σ len_seq + 20B, 64]): CPU GEMM results depend on the number of rows, so only the same call gives the
    same bits. The released layout (projector on the padded [B, 10] history and the [B, 20] candidates,
    ``U model/model_interface.py:446-448``) gives the same values up to that rounding (checked with a bound).
    """
    model = tiny_model()
    ids = model.ids
    for batch in (batches.train(hybrid=True), batches.eval("val"), batches.eval("test", n=3)):
        model._check_batch(batch, "train" if isinstance(batch, TrainBatch) else "eval")
        on_device = batch.to(model.device)
        hist, cand, len_seq = on_device.hist_ids, on_device.cand_ids, on_device.len_seq
        b, n_hist, n_cand = hist.shape[0], hist.shape[1], cand.shape[1]
        with torch.no_grad():
            ours = model._embed_inputs(on_device)
            plain = model.llm.get_input_embeddings()(on_device.input_ids)
            # items row by row: every row's history (oldest first), then every row's candidates
            items = [int(x) for r in range(b) for x in hist[r, :int(len_seq[r])]] + cand.reshape(-1).tolist()
            tok = model.projector(model.sasrec.embed_items(torch.tensor(items, dtype=torch.int64)))
            his = torch.zeros(b, n_hist, tok.shape[1])
            cans = torch.zeros(b, n_cand, tok.shape[1])
            k = 0
            for r in range(b):
                for j in range(int(len_seq[r])):
                    his[r, j] = tok[k]
                    k += 1
            for r in range(b):
                for j in range(n_cand):
                    cans[r, j] = tok[k]
                    k += 1
            reference = refs_upstream.wrap_emb_loop(
                plain.clone(), on_device.input_ids, his, cans, len_seq, torch.full_like(len_seq, n_cand),
                his_token_id=ids.history_emb, cans_token_id=ids.cans_emb)
            # the released layout: cacu_x (item_embeddings lookup, padding row included) through the projector
            table = model.sasrec.item_embeddings.weight
            released = refs_upstream.wrap_emb_loop(
                plain.clone(), on_device.input_ids, model.projector(table[hist]), model.projector(table[cand]),
                len_seq, torch.full_like(len_seq, n_cand), his_token_id=ids.history_emb, cans_token_id=ids.cans_emb)
        assert torch.equal(ours, reference)
        assert float((ours - released).abs().max()) <= 1e-5
        placeholders = (on_device.input_ids == ids.history_emb) | (on_device.input_ids == ids.cans_emb)
        assert int(placeholders.sum()) == int(len_seq.sum()) + n_cand * b
        assert not torch.equal(ours[placeholders], plain[placeholders])           # the rows were replaced
        assert torch.equal(ours[~placeholders], plain[~placeholders])             # nothing else changed


@pytest.mark.criterion("M9-5")
def test_text_only_embeddings(tiny_model, batches):
    """Text-only: inputs_embeds = embed_tokens(input_ids) and the projector is never called."""
    model = tiny_model("model.gen_max_new_tokens=4")
    calls = []
    handle = model.projector.register_forward_hook(lambda module, inputs, output: calls.append(output.shape))
    try:
        for batch in (batches.train(hybrid=False), batches.eval("val", hybrid=False)):
            on_device = batch.to(model.device)
            with torch.no_grad():
                ours = model._embed_inputs(on_device)
                assert torch.equal(ours, model.llm.get_input_embeddings()(on_device.input_ids))
        model.train_forward(batches.train(hybrid=False))
        model.generate(batches.eval("val", hybrid=False, n=2))
        assert calls == []
        model.train_forward(batches.train(hybrid=True))
        assert len(calls) == 1
    finally:
        handle.remove()


# ---------------------------------------------------------------------------------------------- M6-5
@pytest.mark.criterion("M6-5")
def test_projector_only_for_hybrid(tiny_model, batches):
    """Text-only micro-batch: projector forward hook 0 calls and grad None; hybrid: projector gradients (D3, 식 7)."""
    model = tiny_model()
    model.train()
    calls = []
    handle = model.projector.register_forward_hook(lambda module, inputs, output: calls.append(output.shape))
    try:
        model.train_forward(batches.train(hybrid=False)).loss.backward()
        assert calls == []
        assert all(p.grad is None for p in model.projector.parameters())
        assert all(p.grad is not None for _, p in lora_tensors(model))
        model.zero_grad(set_to_none=True)
        model.train_forward(batches.train(hybrid=True)).loss.backward()
        assert len(calls) == 1 and calls[0][-1] == 64
        assert all(grad_is_nonzero(p) for p in model.projector.parameters())
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.eval()


# --------------------------------------------------------------------------------------------- M8-10
@pytest.mark.criterion("M8-10")
def test_gate_use_calls(tiny_model, batches):
    """Every wrapped layer reads omega once per forward: 14 per training forward, 14 x T for T generated tokens."""
    model = tiny_model("model.gen_max_new_tokens=6")
    q_proj = model.llm.model.layers[0].self_attn.q_proj
    shapes = []
    handle = q_proj.register_forward_pre_hook(lambda module, inputs: shapes.append(tuple(inputs[0].shape)))
    try:
        with record_gate_uses(model) as uses:
            model.train()
            model.train_forward(batches.train(hybrid=True))
            model.train_forward(batches.train(hybrid=False))
            model.eval()
            shapes.clear()
            batch = batches.eval("val")
            out = model.generate(batch)
    finally:
        handle.remove()
    steps = out.new_token_ids.shape[1]
    assert 1 <= steps <= 6
    assert [u.calls for u in uses] == [14, 14, 14 * steps]
    assert [u.batch_size for u in uses] == [4, 4, 4]
    width = batch.input_ids.shape[1]
    assert shapes == [(4, width, 64)] + [(4, 1, 64)] * (steps - 1)     # prompt step, then one token per step
    assert not model.gate_ctx.is_set


@pytest.mark.criterion("M8-10")
def test_host_reads_do_not_grow_with_layers(tiny_model, batches, monkeypatch):
    """The host reads (``aten::item``, a CUDA stream synchronisation each) of one training forward do not depend on
    the number of MoE-LoRA modules: the 2-layer tiny LLaMA (14 modules) and a 3-layer one (21) give the same count,
    so no wrapped layer reads a value back (§12.6.4, §12.7)."""
    from torch.profiler import ProfilerActivity, profile

    from ilora import model as model_module

    def reads(model) -> int:
        batch = batches.train(hybrid=True)
        model.train()
        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                model.train_forward(batch)
        finally:
            model.eval()
        return sum(event.count for event in prof.key_averages() if event.key == "aten::item")

    two = tiny_model()
    monkeypatch.setitem(model_module.TINY_LLAMA_SHAPE, "num_hidden_layers", 3)
    three = tiny_model()
    assert (two.n_wrapped, three.n_wrapped) == (14, 21)
    assert reads(two) == reads(three)


# ---------------------------------------------------------------------------- M8-3 (model, §17.6.5)
@pytest.mark.criterion("M8-3")
def test_k1_model_equals_plain_lora(tiny_model, batches):
    """K = 1, routing none: the model logits equal those of the same A and B in a plain LoRA wrapper (bitwise)."""
    model = tiny_model(*K1)
    randomize_lora_b(model, "m9_k1_b")
    scalings = {m.scaling for m in model.llm.modules() if isinstance(m, MoELoRALinear)}
    assert scalings == {4.0}                                     # alpha / r for K = 1 under both rules
    plain = plain_lora_copy(model.llm).eval()
    for batch in (batches.train(hybrid=True), batches.train(hybrid=False), batches.eval("val")):
        ours = model_logits(model, batch)
        on_device = batch.to(model.device)
        with torch.no_grad():
            z, omega = model._z_omega(on_device)
            assert torch.equal(omega, torch.ones_like(omega))
            emb = model._embed_inputs(on_device)
            reference = plain(inputs_embeds=emb, attention_mask=on_device.attention_mask, use_cache=False,
                              return_dict=True).logits
        assert torch.equal(ours, reference)


# ----------------------------------------------------------------------------------------------- #6
def _plain_position(batch, ids, row: int = 0) -> int:
    """A real prompt position of ``row`` holding an ordinary token (no BOS, no placeholder, no label)."""
    labels = batch.labels[row] if isinstance(batch, TrainBatch) else None
    for p in range(batch.input_ids.shape[1]):
        token = int(batch.input_ids[row, p])
        if (int(batch.attention_mask[row, p]) == 1 and token != ids.bos and token < ids.base_vocab
                and (labels is None or int(labels[p]) == IGNORE_INDEX)):
            return p
    raise AssertionError("no ordinary prompt token found")


def _with_token(batch, row: int, position: int, token: int):
    input_ids = batch.input_ids.clone()
    input_ids[row, position] = token
    return dataclasses.replace(batch, input_ids=input_ids)


def _shift_row(batch, *, to_left_padding: bool):
    """Move the padding of the row with the most padding to the other side (input_ids, mask, labels)."""
    n_real = batch.attention_mask.sum(dim=1)
    row = int(n_real.argmin())
    pad = batch.input_ids.shape[1] - int(n_real[row])
    assert pad > 0
    shift = pad if to_left_padding else -pad
    fields = {}
    for name in ("input_ids", "attention_mask") + (("labels",) if isinstance(batch, TrainBatch) else ()):
        value = getattr(batch, name).clone()
        value[row] = torch.roll(value[row], shift)
        fields[name] = value
    return dataclasses.replace(batch, **fields), row


BATCH_CASES = {   # case: (rule named in the error, kind of call)
    "missing_history_emb": ("placeholder counts", "train"),
    "extra_cans_emb": ("placeholder counts", "train"),
    "ph_in_hybrid": ("placeholder counts", "train"),
    "cans_in_text_only": ("placeholder counts", "train"),
    "item_emb_in_input": (r"no \[ItemEmb\]", "eval"),
    "hist_padding_not_1682": ("hist_ids", "train"),
    "left_padded_train_batch": ("right padding", "train"),
    "right_padded_eval_batch": ("left padding", "eval"),
    "labels_differ_from_input_ids": ("labels equal input_ids", "train"),
    "no_label_tokens": ("at least one label token", "train"),
}


@pytest.mark.criterion("M9-6")
@pytest.mark.parametrize("case", list(BATCH_CASES))
def test_batch_checks(case, tiny_model, batches):
    from ilora.model import ModelError

    model = tiny_model("model.gen_max_new_tokens=2")
    ids = model.ids
    rule, kind = BATCH_CASES[case]
    row = 0
    if case == "missing_history_emb":
        batch = batches.train(hybrid=True)
        first = int((batch.input_ids[0] == ids.history_emb).nonzero()[0, 0])
        bad = _with_token(batch, 0, first, int(batch.input_ids[0, _plain_position(batch, ids)]))
    elif case in ("extra_cans_emb", "ph_in_hybrid"):
        batch = batches.train(hybrid=True)
        bad = _with_token(batch, 0, _plain_position(batch, ids), ids.cans_emb if case == "extra_cans_emb" else ids.ph)
    elif case == "cans_in_text_only":
        batch = batches.train(hybrid=False)
        bad = _with_token(batch, 0, _plain_position(batch, ids), ids.cans_emb)
    elif case == "item_emb_in_input":
        batch = batches.eval("val")
        bad = _with_token(batch, 0, _plain_position(batch, ids), ids.item_emb)
    elif case == "hist_padding_not_1682":
        batch = batches.train(hybrid=True)
        assert int(batch.len_seq[0]) < 10
        hist = batch.hist_ids.clone()
        hist[0, int(batch.len_seq[0])] = 0
        bad = dataclasses.replace(batch, hist_ids=hist)
    elif case == "left_padded_train_batch":
        bad, row = _shift_row(batches.train(hybrid=True), to_left_padding=True)
    elif case == "right_padded_eval_batch":
        bad, row = _shift_row(batches.eval("val"), to_left_padding=False)
    elif case == "labels_differ_from_input_ids":
        batch = batches.train(hybrid=True)
        labels = batch.labels.clone()
        first = int((labels[0] != IGNORE_INDEX).nonzero()[0, 0])
        labels[0, first] = (int(labels[0, first]) + 1) % ids.base_vocab
        bad = dataclasses.replace(batch, labels=labels)
    else:   # no_label_tokens
        batch = batches.train(hybrid=True)
        labels = batch.labels.clone()
        labels[0] = IGNORE_INDEX
        bad = dataclasses.replace(batch, labels=labels)
    if kind == "train":
        model.train()
        call = model.train_forward
    else:
        model.eval()
        call = model.generate
    with pytest.raises(ModelError, match=rule) as info:
        call(bad)
    assert f"row {row}" in str(info.value) and bad.sample_ids[row] in str(info.value)
    assert not model.gate_ctx.is_set


@pytest.mark.criterion("M9-6")
def test_batch_type_and_shape_checks(tiny_model, batches):
    """A batch of the wrong kind, shape or dtype is rejected before any computation (§13.6.10)."""
    from ilora.model import ModelError

    model = tiny_model()
    train, evaluation = batches.train(hybrid=True), batches.eval("val")
    with pytest.raises(ModelError, match="TrainBatch"):
        model.train_forward(evaluation)
    with pytest.raises(ModelError, match="EvalBatch"):
        model.generate(train)
    with pytest.raises(ModelError, match="cand_ids"):
        model.train_forward(dataclasses.replace(train, cand_ids=train.cand_ids[:, :19]))
    with pytest.raises(ModelError, match="int64"):
        model.train_forward(dataclasses.replace(train, attention_mask=train.attention_mask.bool()))
    with pytest.raises(ModelError, match="sample_ids"):
        model.train_forward(dataclasses.replace(train, sample_ids=train.sample_ids[:3]))


# ----------------------------------------------------------------------------------------------- #7
@pytest.mark.criterion("M9-7")
def test_gradient_flow(tiny_model, batches):
    """Random B: LoRA A and B, projector (hybrid) and gate get non-zero gradients; base and SASRec none."""
    model = tiny_model()
    randomize_lora_b(model)
    model.train()
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and any(n.startswith("sasrec.") for n, _ in frozen) and any(n.startswith("llm.") for n, _ in frozen)
    gate = model.omega_source.gate.proj.weight
    try:
        model.train_forward(batches.train(hybrid=True)).loss.backward()
        for name, p in lora_tensors(model):
            assert grad_is_nonzero(p), name
        assert all(grad_is_nonzero(p) for p in model.projector.parameters())
        assert grad_is_nonzero(gate)
        assert all(p.grad is None for _, p in frozen)
        model.zero_grad(set_to_none=True)
        model.train_forward(batches.train(hybrid=False)).loss.backward()
        assert all(p.grad is None for p in model.projector.parameters())           # text-only: no projector
        assert all(grad_is_nonzero(p) for _, p in lora_tensors(model)) and grad_is_nonzero(gate)
        assert all(p.grad is None for _, p in frozen)
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


@pytest.mark.criterion("M9-7")
def test_z_ignores_target(tiny_model, batches, smoke_data):
    """z = SR-EMB(i<n) [논문 p.5 식 10]: a batch whose target changed has the same z (and omega)."""
    model = tiny_model("model.gen_max_new_tokens=2")
    sample = smoke_data.train[10]
    other = next(t for t in range(1682) if t != sample.target and t not in sample.history)
    changed = dataclasses.replace(sample, target=other)
    model.train()
    a = model.train_forward(batches.train(hybrid=True, samples=[sample]))
    b = model.train_forward(batches.train(hybrid=True, samples=[changed]))
    assert torch.equal(a.z, b.z) and torch.equal(a.omega, b.omega)
    model.eval()
    evaluation = batches.eval("val", n=2)
    swapped = dataclasses.replace(evaluation, target_ids=evaluation.target_ids.flip(0))
    assert torch.equal(model.generate(evaluation).z, model.generate(swapped).z)


# ----------------------------------------------------------------------------------------------- #8
@pytest.mark.criterion("M9-8")
def test_sasrec_stays_eval(tiny_model, batches):
    model = tiny_model()
    dropouts = [m.lora_dropout for m in model.llm.modules() if isinstance(m, MoELoRALinear)]
    model.train()
    assert model.training and model.llm.training and all(d.training for d in dropouts)
    assert not any(m.training for m in model.sasrec.modules()) and model.sasrec_is_eval()
    batch = batches.train(hybrid=True)
    z1, z2 = model.train_forward(batch).z, model.train_forward(batch).z
    assert torch.equal(z1, z2)
    model.sasrec.train(True)                                   # even a direct call keeps SASRec in eval (D12)
    assert model.sasrec_is_eval()
    model.eval()
    assert not model.training and model.sasrec_is_eval() and not any(d.training for d in dropouts)


# ----------------------------------------------------------------------------------------------- #9
@pytest.mark.criterion("M9-9")
def test_generate_contract(tiny_model, batches, monkeypatch):
    from ilora.model import ModelError

    model = tiny_model()                                       # gen_max_new_tokens 64 (paper_spec value)
    randomize_lora_b(model)
    ids = model.ids
    batch = batches.eval("val")
    raw = []
    original = model.llm.generate

    def spy(**kwargs):
        assert model.gate_ctx.is_set                           # one omega for every generation step
        assert set(kwargs) == {"inputs_embeds", "attention_mask", "generation_config"}
        out = original(**kwargs)
        raw.append(out)
        return out

    monkeypatch.setattr(model.llm, "generate", spy)
    with record_gate_uses(model) as uses:
        out = model.generate(batch)
    steps = out.new_token_ids.shape[1]
    assert out.new_token_ids.dtype == torch.int64 and out.new_token_ids.device.type == "cpu"
    assert out.new_token_ids.shape[0] == 4 and 1 <= steps <= 64
    assert bool((raw[0][:, 0] == ids.bos).all()) and torch.equal(raw[0][:, 1:], out.new_token_ids)
    assert len(uses) == 1 and uses[0].calls == 14 * steps
    assert out.omega.shape == (4, K) and out.z.shape == (4, 64)
    assert out.omega.dtype == out.z.dtype == torch.float32 and out.omega.device.type == out.z.device.type == "cpu"
    for row in out.new_token_ids.tolist():                     # [PAD] after the first EOS
        if ids.eos in row:
            assert all(t == ids.pad for t in row[row.index(ids.eos) + 1:])
    assert not model.gate_ctx.is_set
    again = model.generate(batch)
    assert torch.equal(again.new_token_ids, out.new_token_ids)
    assert torch.equal(again.omega, out.omega) and torch.equal(again.z, out.z)
    model.train()
    with pytest.raises(ModelError, match="eval mode"):
        model.generate(batch)
    model.eval()

    def failing(**kwargs):
        assert model.gate_ctx.is_set
        raise RuntimeError("generation failed")

    monkeypatch.setattr(model.llm, "generate", failing)
    with pytest.raises(RuntimeError, match="generation failed"):
        model.generate(batch)
    assert not model.gate_ctx.is_set                           # cleared also after an exception (M8)


@pytest.mark.criterion("M9-9")
def test_batch_generation_equals_single(tiny_model, batches, smoke_data):
    """Left-padded batches of 8 give the same new tokens as one sample at a time (CPU fp32, D22)."""
    model = tiny_model()
    randomize_lora_b(model)
    pad, eos = model.ids.pad, model.ids.eos
    for split in ("val", "test"):
        samples = list(smoke_data.samples(split))
        batched = []
        for start in range(0, len(samples), 8):
            batch = batches.eval(split, samples=samples[start:start + 8])
            batched += model.generate(batch).new_token_ids.tolist()
        assert len(batched) == len(samples) == 10
        for sample, row in zip(samples, batched):
            single = model.generate(batches.eval(split, samples=[sample])).new_token_ids[0].tolist()
            assert row[:len(single)] == single, sample.sample_id
            assert all(t == pad for t in row[len(single):]), sample.sample_id
            assert len(single) == 64 or single[-1] == eos


@pytest.mark.criterion("M9-9")
def test_use_cache_required(tiny_model, batches, monkeypatch):
    """A generation config with use_cache False fails the build check; 4.28.0 really needs the cache."""
    from ilora import model as model_module

    model = tiny_model("model.gen_max_new_tokens=3")
    no_cache = model.gen_config
    no_cache.use_cache = False
    on_device = batches.eval("val").to(model.device)
    with torch.no_grad():
        z, omega = model._z_omega(on_device)
        emb = model._embed_inputs(on_device)
        # without a cache 4.28.0 feeds the prompt embeddings again at step 2 (modeling_llama.py:742-745)
        with pytest.raises(RuntimeError, match="is invalid for input of size"), model.gate_ctx.use(omega):
            model.llm.generate(inputs_embeds=emb, attention_mask=on_device.attention_mask, generation_config=no_cache)
    original = model_module.generation_config_for

    def without_cache(mcfg, ids):
        config = original(mcfg, ids)
        config.use_cache = False
        return config

    monkeypatch.setattr(model_module, "generation_config_for", without_cache)
    with pytest.raises(model_module.ModelError, match="use_cache"):
        tiny_model()


# ---------------------------------------------------------------------------------------------- #10
@pytest.mark.criterion("M9-10")
def test_position_ids_left_padding(tiny_model, batches):
    """4.28.0 position_ids = cumsum(mask) - 1: real tokens of every left-padded row at 0, 1, 2, ... (D22)."""
    model = tiny_model("model.gen_max_new_tokens=3")
    batch = batches.eval("val")
    n_real = batch.attention_mask.sum(dim=1)
    assert len(set(n_real.tolist())) > 1                       # rows of different length
    on_device = batch.to(model.device)
    with torch.no_grad():
        emb = model._embed_inputs(on_device)
    placeholder = torch.full((4, 1), model.ids.bos, dtype=torch.int64)
    inputs = model.llm.prepare_inputs_for_generation(placeholder, past_key_values=None,
                                                     attention_mask=on_device.attention_mask, inputs_embeds=emb,
                                                     use_cache=True)
    assert "inputs_embeds" in inputs and "input_ids" not in inputs           # first step: embeddings only
    positions = inputs["position_ids"]
    for r in range(4):
        n = int(n_real[r])
        assert positions[r, -n:].tolist() == list(range(n))
        assert all(v == 1 for v in positions[r, :-n].tolist())              # padded positions (masked)
    seen = []

    def hook(module, args, kwargs):
        seen.append({k: None if kwargs.get(k) is None else kwargs[k].clone()
                     for k in ("position_ids", "input_ids", "inputs_embeds")})

    handle = model.llm.model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = model.generate(batch)
    finally:
        handle.remove()
    steps = out.new_token_ids.shape[1]
    assert len(seen) == steps
    assert seen[0]["input_ids"] is None and torch.equal(seen[0]["inputs_embeds"], emb)
    assert torch.equal(seen[0]["position_ids"], positions)
    for t, call in enumerate(seen[1:], start=1):              # cached steps: the last token at position n + t - 1
        assert call["inputs_embeds"] is None and tuple(call["input_ids"].shape) == (4, 1)
        assert call["position_ids"].squeeze(1).tolist() == (n_real + t - 1).tolist()


# ---------------------------------------------------------------------------------------------- #11
@pytest.mark.criterion("M9-11", "M6-8")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")      # torch 2.0.0 torch.load(weights_only=True)
def test_checkpoint_round_trip(tiny_model, batches, tmp_path):
    """trainable_state -> torch.save -> load_trainable_state into a new model: same logits and generations."""
    from ilora.model import GENERATED_FROZEN_KEYS, ModelError

    trained = tiny_model("model.gen_max_new_tokens=8")
    perturb_trainable(trained, "m9_checkpoint_noise")
    trained.train()                                              # M6-8: a text-only step leaves the projector
    trained.train_forward(batches.train(hybrid=False)).loss.backward()
    assert all(p.grad is None for p in trained.projector.parameters())
    trained.zero_grad(set_to_none=True)
    trained.eval()
    state = trained.trainable_state()
    names = [n for items in trained.param_groups().values() for n, _ in items]
    assert list(state) == names and len(state) == 33                         # LoRA 28, projector 4, gate 1
    assert sum(k.startswith("projector.") for k in state) == 4              # saved without a gradient
    params = dict(trained.named_parameters())
    for name, value in state.items():
        assert value.device.type == "cpu" and value.dtype == torch.float32 and value.is_contiguous()
        assert not value.requires_grad and value.data_ptr() != params[name].data_ptr()      # copies
    path = tmp_path / "trainable.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    fresh = tiny_model("model.gen_max_new_tokens=8")
    assert any(not torch.equal(p, params[n]) for n, p in fresh.named_parameters() if n in state)
    fresh.load_trainable_state(loaded)
    for name, p in fresh.named_parameters():                                 # trainable and frozen
        assert torch.equal(p, params[name]), name
    frozen_a, frozen_b = trained.generated_frozen_state(), fresh.generated_frozen_state()
    assert all(torch.equal(frozen_a[k], frozen_b[k]) for k in GENERATED_FROZEN_KEYS)
    for batch in (batches.train(hybrid=True), batches.train(hybrid=False), batches.eval("val")):
        assert torch.equal(model_logits(trained, batch), model_logits(fresh, batch))
    train_batch = batches.train(hybrid=True)
    assert torch.equal(trained.train_forward(train_batch).loss, fresh.train_forward(train_batch).loss)
    eval_batch = batches.eval("test")
    a, b = trained.generate(eval_batch), fresh.generate(eval_batch)
    assert torch.equal(a.new_token_ids, b.new_token_ids) and torch.equal(a.omega, b.omega) and torch.equal(a.z, b.z)

    before = fresh.trainable_state()
    first = names[0]
    rejected = {
        "missing keys": {k: v for k, v in loaded.items() if k != first},
        "unexpected keys": {**loaded, "llm.model.norm.weight": torch.ones(64)},
        "generated_frozen": {**loaded, **trained.generated_frozen_state()},
        r"\[63\]": {**loaded, first: loaded[first][:-1]},
        "float64": {**loaded, first: loaded[first].double()},
        "not a tensor": {**loaded, first: loaded[first].tolist()},
    }
    for message, bad in rejected.items():
        with pytest.raises(ModelError, match=message):
            fresh.load_trainable_state(bad)
        after = fresh.trainable_state()
        assert all(torch.equal(before[k], after[k]) for k in before), message      # nothing copied
    fresh.load_trainable_state({first: torch.zeros_like(loaded[first])}, strict=False)
    assert torch.equal(dict(fresh.named_parameters())[first], torch.zeros_like(loaded[first]))


@pytest.mark.criterion("M9-11")
def test_router_from_checkpoint_keys(tiny_model, batches, cpu_cfg, fake_hard_gate, hard_gate_table):
    """The ``omega_source.`` keys alone load into an LLM-free Router that gives the model's omega (M12 export)."""
    from ilora.model import OMEGA_SOURCE_PREFIX
    from ilora.router import Router

    model = tiny_model("model.gen_max_new_tokens=2")
    perturb_trainable(model, "m9_router_noise", std=0.05)
    state = model.trainable_state()
    gate_state = {k[len(OMEGA_SOURCE_PREFIX):]: v for k, v in state.items() if k.startswith(OMEGA_SOURCE_PREFIX)}
    assert list(gate_state) == ["gate.proj.weight"]
    router = Router(cpu_cfg.router, num_experts=K, z_dim=64, main_seed=SEED + 1, hard_gate=None)
    assert not torch.equal(router.gate.proj.weight, model.omega_source.gate.proj.weight)
    router.load_state_dict(gate_state, strict=True)
    batch = batches.eval("val")
    out = model.generate(batch)
    with torch.no_grad():
        z = model.sasrec(batch.hist_ids, batch.len_seq)
        assert torch.equal(z, out.z)
        assert torch.equal(router(z, batch.sample_ids), out.omega)
        assert torch.equal(router(z, batch.sample_ids), model.omega_source(z, batch.sample_ids))
    for sets, hard_gate in ((CLUSTER_HARD, fake_hard_gate(hard_gate_table, K)), (K1, None)):
        other = tiny_model(*sets, hard_gate=hard_gate).trainable_state()
        assert len(other) == 32 and not any(k.startswith(OMEGA_SOURCE_PREFIX) for k in other)


# ---------------------------------------------------------------------------------------------- #12
@pytest.mark.criterion("M9-12")
def test_bf16_autocast_proxy(tiny_model, batches):
    """CPU stand-in for the GPU fp16 path: bf16 base under bf16 autocast -> bf16 logits, fp32 loss and gradients."""
    model = tiny_model()
    randomize_lora_b(model)
    lora_ids = {id(p) for _, p in lora_named_parameters(model.llm)}
    with torch.no_grad():
        for p in model.llm.parameters():
            if id(p) not in lora_ids:
                p.data = p.data.to(torch.bfloat16)                             # base bf16, LoRA masters fp32
    dtypes = {"logits": [], "tokens": []}
    handles = [model.llm.lm_head.register_forward_hook(lambda m, i, o: dtypes["logits"].append(o.dtype)),
               model.projector.register_forward_hook(lambda m, i, o: dtypes["tokens"].append(o.dtype))]
    model.train()
    try:
        for hybrid in (True, False):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model.train_forward(batches.train(hybrid=hybrid))
            assert out.loss.dtype == torch.float32 and bool(torch.isfinite(out.loss))
            assert out.omega.dtype == out.z.dtype == torch.float32
            out.loss.backward()
            for group, items in model.param_groups().items():
                for name, p in items:
                    if group == "projector" and not hybrid:
                        assert p.grad is None, name
                    else:
                        assert p.grad is not None and p.grad.dtype == torch.float32, name
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()
        model.eval()
    assert dtypes == {"logits": [torch.bfloat16, torch.bfloat16], "tokens": [torch.bfloat16]}


# ---------------------------------------------------------------------------------------------- #13
@pytest.mark.criterion("M9-13")
def test_llama7b_parameter_counts(llama7b_skeleton, tiny_model):
    """Counts on the meta Llama-2-7B skeleton with the function parameter_report() uses (§13.8 #13)."""
    from ilora.model import LLAMA2_7B_SHAPE, check_llama_config, parameter_counts
    from ilora.projector import build_projector
    from ilora.router import Router
    from ilora.sections import RouterConfig

    skeleton = copy.deepcopy(llama7b_skeleton)               # the session fixture stays unchanged
    check_llama_config(skeleton.config, shape=LLAMA2_7B_SHAPE, base_vocab=32000)
    assert sum(p.numel() for p in skeleton.parameters()) == 6_738_415_616
    with torch.device("meta"):
        skeleton.resize_token_embeddings(32005)
    assert sum(p.numel() for p in skeleton.parameters()) == 6_738_456_576           # + 2 x 5 x 4,096
    skeleton.requires_grad_(False)
    ctx = GateContext(K)
    for i, layer in enumerate(skeleton.model.layers):
        for parent_name, targets in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                     ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            parent = getattr(layer, parent_name)
            for target in targets:
                setattr(parent, target, MoELoRALinear(getattr(parent, target), r=8, num_experts=K, alpha=32,
                                                      dropout=0.1, scaling_rule="expert_rank", gate_ctx=ctx,
                                                      name=f"model.layers.{i}.{parent_name}.{target}"))
    projector = build_projector(64, 4096, SEED)
    dynamic = Router(RouterConfig(routing_mode="dynamic"), num_experts=K, z_dim=64, main_seed=SEED, hard_gate=None)
    counts = parameter_counts(skeleton, projector, dynamic, base_vocab=32000)
    assert counts["base_before_resize"] == 6_738_415_616
    assert counts["base"] == 6_738_456_576 and counts["new_token_rows"] == 40_960
    assert (counts["lora"], counts["lora_tensors"], counts["wrapped_modules"]) == (19_988_480, 448, 224)
    assert (counts["projector"], counts["gate"], counts["trainable"]) == (17_047_552, 256, 37_036_288)
    none = Router(RouterConfig(routing_mode="none"), num_experts=1, z_dim=64, main_seed=SEED, hard_gate=None)
    assert parameter_counts(skeleton, projector, none, base_vocab=32000)["trainable"] == 37_036_032
    model = tiny_model()
    report = model.parameter_report()
    assert report == parameter_counts(model.llm, model.projector, model.omega_source, base_vocab=32000,
                                      sasrec=model.sasrec)
    assert report["base"] == sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)
    assert report["base_before_resize"] == report["base"] - 2 * 5 * 64
    assert report["sasrec"] == 129_664 and report["total"] == sum(p.numel() for p in model.parameters())
