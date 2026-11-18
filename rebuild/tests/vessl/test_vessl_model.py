"""VESSL 모델 시험 (DESIGN.md §17.6.9: M6-9, M8-11, M8-12 = M9-17, M9-14, M9-15, M9-16, M12-14).

VESSL_REQUIRED: skipped locally. Run on the A100 with ``ILORA_VESSL=1`` (§17.6.1).
Values the design does not judge (fp16 vs fp32 differences, memory, load time,
batch-vs-single agreement) are recorded in ``measurements`` and stay TBD
(§17.6.10); only the stated limits are asserted. The M12 model test of §17.6.9
(``test_activation_monitor_fp16``) also belongs in this file.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import time

import pytest
import torch
from torch import nn

from ilora import seeding

pytestmark = pytest.mark.vessl


def _randn(name: str, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    with seeding.torch_seed_scope(0, "test", name):
        return torch.randn(*shape, dtype=dtype)


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


@pytest.mark.criterion("M6-9")
def test_projector_fp16_autocast(llama_model, paper_cfg, measurements):
    """The model's projector under the training/evaluation fp16 autocast gives finite float16 tokens (§10.6.3)."""
    from ilora.precision import autocast_for

    assert paper_cfg.train.precision == "fp16_amp"
    device = llama_model.device
    item_ids = torch.arange(1682, dtype=torch.int64, device=device)       # every item id 0-1681
    with torch.no_grad():
        item_repr = llama_model.sasrec.embed_items(item_ids)              # [1682, 64] float32, autocast off
        with autocast_for(paper_cfg.train.precision, device):
            tokens = llama_model.projector(item_repr)
    assert item_repr.dtype == torch.float32
    assert tokens.dtype == torch.float16
    assert tuple(tokens.shape) == (1682, llama_model.llm.config.hidden_size)
    assert bool(torch.isfinite(tokens).all())
    for name, p in llama_model.projector.named_parameters():
        assert p.dtype == torch.float32, name                             # D9: fp32 masters
    measurements["projector_fp16_autocast"] = {"n_items": 1682, "max_abs": float(tokens.float().abs().max())}


@pytest.mark.criterion("M8-11")
@pytest.mark.parametrize("dims", [(16, 12), (4096, 4096)], ids=["m8_2_dims", "q_proj_dims"])
@pytest.mark.parametrize("shape", [(3, 7), (2, 1)], ids=["3x7", "2x1"])
def test_moe_lora_layer_fp16(shape, dims, measurements):
    """One wrapped layer with an fp16 base under fp16 autocast: fp16 output, fp32 A/B/omega gradients.

    Inputs as in M8-2 (shapes [3, 7, d] and [2, 1, d], K = 4, r = 8, alpha = 32, the same synthetic seeds) at the
    M8-2 width and at the q_proj width. The difference to the fp32 computation of the same weights is recorded.
    """
    from ilora.moe_lora import GateContext, MoELoRALinear
    from ilora.precision import autocast_for

    device = torch.device("cuda:0")
    d_in, d_out = dims
    b, length = shape
    ctx = GateContext(4)
    with seeding.torch_seed_scope(0, "test", "moe_lora_base"):
        base = nn.Linear(d_in, d_out, bias=False)
    layer = MoELoRALinear(base, r=8, num_experts=4, alpha=32, dropout=0.0, scaling_rule="expert_rank",
                          gate_ctx=ctx, name="vessl.layer")
    with seeding.torch_seed_scope(0, "test", "moe_lora_init"):
        layer.reset_lora_parameters()
        nn.init.normal_(layer.lora_B.weight, std=0.5 if d_in == 16 else 0.02)
    reference = copy.deepcopy(layer).to(device)          # fp32 base: the fp32 computation of the same weights
    layer.base.half()                                    # D9: fp16 base, fp32 LoRA masters
    layer = layer.to(device)
    x = _randn("fp64_x", b, length, d_in, dtype=torch.float64).float().to(device)
    omega0 = torch.softmax(_randn("fp64_omega", b, 4), dim=-1).to(device)
    upstream_grad = _randn("fp64_grad", b, length, d_out).to(device)

    omega = omega0.clone().requires_grad_(True)
    with autocast_for("fp16_amp", device):
        with ctx.use(omega):
            y = layer(x.half())
    assert y.dtype == torch.float16 and bool(torch.isfinite(y).all())
    (y.float() * upstream_grad).sum().backward()
    assert layer.lora_A.weight.grad.dtype == torch.float32
    assert layer.lora_B.weight.grad.dtype == torch.float32
    assert omega.grad is not None and omega.grad.dtype == torch.float32
    assert layer.base.weight.dtype == torch.float16 and layer.base.weight.grad is None

    omega_ref = omega0.clone().requires_grad_(True)
    with reference.gate_ctx.use(omega_ref):              # deepcopy gave the reference its own GateContext
        y_ref = reference(x)
    (y_ref * upstream_grad).sum().backward()
    measurements[f"moe_lora_fp16_{d_in}x{d_out}_{b}x{length}"] = {
        "output_max_abs_diff": float((y.float() - y_ref).abs().max()),
        "output_max_rel_diff": _max_rel(y.float(), y_ref),
        "grad_A_max_rel_diff": _max_rel(layer.lora_A.weight.grad, reference.lora_A.weight.grad),
        "grad_B_max_rel_diff": _max_rel(layer.lora_B.weight.grad, reference.lora_B.weight.grad),
        "grad_omega_max_rel_diff": _max_rel(omega.grad, omega_ref.grad),
    }


@pytest.mark.slow
@pytest.mark.criterion("M8-12", "M9-17")
def test_longest_batch_memory(llama_model, llama_bundle, full_data, paper_cfg, templates, measurements):
    """Forward + backward of the longest training batch of the 5-epoch plan stays below 76 GiB (§12.8 #12, §13.8 #17).

    The batch is found on CPU with M4 ``build_batch_plan`` and M5 ``make_train_loader`` (every epoch, every batch);
    its position and length are recorded. The design figures (13.6.7: about 12.1 GiB less than the fork's bf16-base
    run; fork rewrite model 60.7 GiB) are recorded next to the measurement as expectations, not results.
    """
    from ilora.batching import make_train_loader
    from ilora.precision import autocast_for
    from ilora.prompts import build_batch_plan

    train = full_data.samples("train")
    longest = None
    for epoch in range(paper_cfg.train.epochs):
        plan = build_batch_plan(train, epoch, paper_cfg)
        loader = make_train_loader(plan, full_data, paper_cfg, epoch, bundle=llama_bundle, templates=templates)
        for index, batch in enumerate(loader):
            length = int(batch.input_ids.shape[1])
            if longest is None or length > longest[0]:
                longest = (length, epoch, index, batch)
    assert longest is not None
    length, epoch, index, batch = longest

    device = llama_model.device
    was_training = llama_model.training
    llama_model.train()
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with autocast_for(paper_cfg.train.precision, device):
            out = llama_model.train_forward(batch)
        out.loss.backward()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        loss = float(out.loss)
    finally:
        llama_model.zero_grad(set_to_none=True)
        llama_model.train(was_training)
    measurements["longest_batch_memory"] = {
        "epoch": epoch, "batch_index": index, "seq_len": length, "batch_size": int(batch.input_ids.shape[0]),
        "hybrid": bool(batch.hybrid), "sample_ids": list(batch.sample_ids), "loss": loss,
        "max_memory_allocated_gib": peak, "max_memory_reserved_gib": reserved,
        "limit_gib": 76.0,
        "design_expected_reduction_vs_fork_bf16_base_gib": 12.1,
        "design_fork_rewrite_model_gib": 60.7,
    }
    assert peak < 76.0


# ----------------------------------------------------------------------------
# M9 on Llama-2-7B (§13.8 #14-#16)
# ----------------------------------------------------------------------------
def _batch(cfg, data, bundle, templates, split: str, n: int, *, start: int = 0, hybrid: bool = True):
    """Samples ``start .. start + n - 1`` of ``split`` as one batch made with the public M2, M4 and M5 functions.

    train: ``TrainBatch`` with the epoch-0 training keys; val/test: left-padded ``EvalBatch`` (fixed keys).
    """
    from ilora.batching import collate_eval, collate_train, encode_eval, encode_train
    from ilora.candidates import candidates_for
    from ilora.prompts import build_prompt, choose_template

    samples = list(data.samples(split)[start:start + n])
    epoch = 0 if split == "train" else None
    cands, tids, encoded = [], [], []
    for s in samples:
        c = candidates_for(s, main_seed=cfg.run.seed, epoch=epoch, cfg=cfg.candidates)
        k = choose_template(cfg.prompt, cfg.run.seed, len(templates.templates), s, epoch)
        prompt = build_prompt(s, c, data.catalog, templates, k, hybrid)
        cands.append(c)
        tids.append(k)
        if split == "train":
            encoded.append(encode_train(bundle, prompt, data.catalog.titles[s.target], cfg.batching.label_mode))
        else:
            encoded.append(encode_eval(bundle, prompt))
    if split == "train":
        return collate_train(samples, cands, tids, encoded, bundle, hybrid=hybrid, tau=1)
    return collate_eval(samples, cands, tids, encoded, data.catalog, bundle, hybrid=hybrid)


def _trim(row: list, eos: int) -> list:
    """New tokens up to and including the first EOS (the M11 rule, §15.6.4)."""
    return row[:row.index(eos) + 1] if eos in row else row


@pytest.mark.slow
@pytest.mark.criterion("M9-14")
def test_load_llama2_fp16(llm_path, llama_bundle, paper_cfg, measurements):
    """LLM_PATH Llama-2-7b-hf loads as fp16 within the allowed load info, rope_scaling None, 32,005 rows after
    the seeded resize; the new rows hash the same in two loads. Load time and peak host memory are recorded."""
    import resource

    from ilora import run
    from ilora.model import LLAMA2_7B_SHAPE, check_llama_config, load_base_llm, resize_new_tokens

    digests, seconds = [], []
    for _ in range(2):
        start = time.perf_counter()
        llm = load_base_llm(llm_path, torch.float16)                 # load info and dtypes checked inside
        seconds.append(time.perf_counter() - start)
        check_llama_config(llm.config, shape=LLAMA2_7B_SHAPE, base_vocab=llama_bundle.ids.base_vocab)
        assert getattr(llm.config, "rope_scaling", None) is None
        assert {p.dtype for p in llm.parameters()} == {torch.float16}
        resize_new_tokens(llm, len(llama_bundle.tokenizer), paper_cfg.run.seed)
        assert llm.config.vocab_size == 32005
        assert llm.get_input_embeddings().weight.shape[0] == 32005 == llm.lm_head.weight.shape[0]
        rows = torch.cat([llm.get_input_embeddings().weight[32000:], llm.lm_head.weight[32000:]]).detach()
        assert rows.dtype == torch.float16 and bool(torch.isfinite(rows).all())
        digests.append(hashlib.sha256(rows.contiguous().view(torch.int16).numpy().tobytes()).hexdigest())
        del llm, rows
        gc.collect()
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20        # Linux: KiB
    measurements["llama2_fp16_load"] = {
        "llm_path": str(llm_path), "config_sha256": run.content_sha256(llm_path / "config.json"),
        "load_seconds": seconds, "new_rows_sha256": digests, "max_rss_gib": peak_gib,
    }
    assert digests[0] == digests[1]


@pytest.mark.slow
@pytest.mark.criterion("M9-15")
def test_fp16_train_microbatch(llama_model, llama_bundle, full_data, paper_cfg, templates, measurements):
    """One hybrid micro-batch under fp16 autocast at the initial state (B = 0): fp16 logits, finite fp32 loss,
    zero omega (W_g) and lora_A gradients, non-zero lora_B gradients; the difference to the HF loss is recorded.

    The backward is not scaled: with B = 0 a scaled fp16 gradient that overflows would turn 0 x inf into NaN, and
    the zero checks are about the initial state, not about GradScaler (M10 handles skipped steps).
    """
    from ilora.moe_lora import lora_named_parameters
    from ilora.precision import autocast_for

    model = llama_model
    device = model.device
    batch = _batch(paper_cfg, full_data, llama_bundle, templates, "train", paper_cfg.train.micro_batch_size)
    lora = lora_named_parameters(model.llm)
    assert all(not bool(p.any()) for n, p in lora if n.endswith("lora_B.weight"))       # first step: B = 0
    logits_dtypes = []
    handle = model.llm.lm_head.register_forward_hook(lambda m, i, o: logits_dtypes.append(o.dtype))
    was_training = model.training
    try:
        model.train()
        assert model.sasrec_is_eval()
        with autocast_for(paper_cfg.train.precision, device):
            out = model.train_forward(batch)
        assert logits_dtypes == [torch.float16]
        assert out.loss.dtype == torch.float32 and bool(torch.isfinite(out.loss))
        out.loss.backward()
        gate = model.omega_source.gate.proj.weight
        assert gate.grad is not None and not bool(gate.grad.any())
        assert all(p.grad is not None and not bool(p.grad.any()) for n, p in lora if n.endswith("lora_A.weight"))
        b_nonzero = sum(bool(p.grad.any()) for n, p in lora if n.endswith("lora_B.weight"))
        assert b_nonzero > 0
        grads_finite = all(bool(torch.isfinite(p.grad).all()) for _, p in lora if p.grad is not None)
        model.zero_grad(set_to_none=True)
        model.eval()                                                  # no dropout for the loss comparison
        on_device = batch.to(device)
        with torch.no_grad(), autocast_for(paper_cfg.train.precision, device):
            ours = model.train_forward(batch).loss
            z, omega = model._z_omega(on_device)
            emb = model._embed_inputs(on_device)
            with model.gate_ctx.use(omega):
                hf = model.llm(inputs_embeds=emb, attention_mask=on_device.attention_mask, labels=on_device.labels,
                               use_cache=False, return_dict=True).loss
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    measurements["fp16_train_microbatch"] = {
        "sample_ids": list(batch.sample_ids), "seq_len": int(batch.input_ids.shape[1]),
        "n_label_tokens": out.n_label_tokens, "loss_train_mode": float(out.loss), "loss_eval_mode": float(ours),
        "hf_loss_eval_mode": float(hf), "abs_diff_direct_vs_hf": abs(float(ours) - float(hf)),
        "lora_B_tensors_with_nonzero_grad": b_nonzero, "lora_grads_finite": grads_finite,
    }


@pytest.mark.slow
@pytest.mark.criterion("M9-16")
def test_stock_generate_fp16(llama_model, llama_bundle, full_data, paper_cfg, templates, measurements,
                             monkeypatch):
    """The unpatched 4.28.0 generate(inputs_embeds, attention_mask) runs on the A100 under fp16 autocast: first
    returned column BOS, the same batch twice gives the same tokens; batch-vs-single agreement is recorded."""
    from ilora import run
    from ilora.precision import autocast_for

    assert run.transformers_integrity_problems() == []                    # stock files (D13)
    model = llama_model
    device = model.device
    eos = model.ids.eos
    batch = _batch(paper_cfg, full_data, llama_bundle, templates, "val", paper_cfg.eval.batch_size)
    raw = []
    original = model.llm.generate

    def spy(**kwargs):
        out = original(**kwargs)
        raw.append(out)
        return out

    monkeypatch.setattr(model.llm, "generate", spy)
    was_training = model.training
    model.eval()
    try:
        with autocast_for(paper_cfg.train.precision, device):
            first = model.generate(batch)
            second = model.generate(batch)
            singles = []
            for i in range(len(batch.sample_ids)):
                one = _batch(paper_cfg, full_data, llama_bundle, templates, "val", 1, start=i)
                assert one.sample_ids == [batch.sample_ids[i]]
                singles.append(_trim(model.generate(one).new_token_ids[0].tolist(), eos))
    finally:
        model.train(was_training)
    assert all(bool((seq[:, 0] == model.ids.bos).all()) for seq in raw)
    assert torch.equal(first.new_token_ids, second.new_token_ids)
    assert torch.equal(first.omega, second.omega) and torch.equal(first.z, second.z)
    batched = [_trim(row, eos) for row in first.new_token_ids.tolist()]
    matches = sum(a == b for a, b in zip(batched, singles))
    measurements["stock_generate_fp16"] = {
        "sample_ids": list(batch.sample_ids), "steps": int(first.new_token_ids.shape[1]),
        "batch_vs_single_identical": matches, "n": len(singles),
    }


# ----------------------------------------------------------------------------
# M12 activation probe on the A100 (§16.8 #14)
# ----------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.criterion("M12-14")
def test_activation_monitor_fp16(llama_model, llama_bundle, full_data, paper_cfg, templates, measurements):
    """``ActivationMaxMonitor`` on the 224 fp16 MoE-LoRA layers under fp16 autocast: the aminmax probe of attach and
    the hooks run on CUDA, the loss of one training micro-batch is bit for bit the same with and without hooks
    (same CPU/CUDA seeds for dropout), and the micro-batch time with and without hooks is recorded (TBD, §16.6.3)."""
    from ilora.moe_lora import MoELoRALinear
    from ilora.precision import autocast_for
    from ilora.records import ActivationMaxMonitor

    model = llama_model
    device = model.device
    batch = _batch(paper_cfg, full_data, llama_bundle, templates, "train", paper_cfg.train.micro_batch_size)

    def micro_batch(name: str) -> tuple[torch.Tensor, float]:
        model.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seeding.derive_seed(0, "test", name))       # LoRA dropout masks (CPU and CUDA)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with autocast_for(paper_cfg.train.precision, device):
                out = model.train_forward(batch)
            out.loss.backward()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
        loss = out.loss.detach().clone()
        model.zero_grad(set_to_none=True)
        return loss, seconds

    monitor = ActivationMaxMonitor(model, MoELoRALinear)
    was_training = model.training
    try:
        model.train()
        micro_batch("m12_fp16_warmup")                                     # kernels and allocator warmed up
        loss_off, seconds_off = micro_batch("m12_fp16_monitor")
        monitor.attach()                                                   # aminmax tried on fp16 CUDA first
        loss_on, seconds_on = micro_batch("m12_fp16_monitor")
        value, index = monitor.end_micro_batch(1)
        maxima, taus = monitor.module_maxima()
    finally:
        monitor.detach()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    assert len(monitor.module_names) == model.n_wrapped == 224
    assert value.device == device and value.dtype == torch.float32 and index.dtype == torch.int64
    assert torch.equal(loss_off, loss_on)
    assert bool((taus == 1).all())                                         # every layer ran in the micro-batch
    measurements["activation_monitor_fp16"] = {
        "sample_ids": list(batch.sample_ids), "loss": float(loss_on), "max_abs": float(value),
        "module": monitor.module_names[int(index)], "module_max_abs_top5": sorted(
            zip(maxima.tolist(), monitor.module_names), reverse=True)[:5],
        "seconds_without_hooks": seconds_off, "seconds_with_hooks": seconds_on,
        "seconds_difference": seconds_on - seconds_off,
    }
