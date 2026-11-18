"""M10 학습 루프 (DESIGN.md §14.8 #1-#14; also M6-6, M6-7, M6-8, M7-7, M7-8 and real-run parts of M11-16, M12-7).

CPU tests use the tiny model of §13.6.9 (``tiny_model``: hidden 64, 2 layers, fp32 CPU) with the real SASRec
state and the smoke_cpu data of §17.6.6 (156 train samples = 39 micro-batches of 4, windows of 4 and a last
window of 3, T = 20, w = 5, hybrid 10 / 28). Training runs go through M0 ``run_command`` into tmp folders
(never runs/rebuild/). Runs compared bit for bit use one CPU thread (§17.6.5). The fp16 GradScaler needs CUDA;
its loop logic (skip flags, lr and tau over skipped steps, the stop after 16 consecutive skips) runs here with
a stand-in scaler of the same interface (§14.8; the GPU checks are #15-#16 under tests/vessl/). The M11 eval
command and the M12 export run on these real checkpoints (the parts M11-16 and M12-7 left to M10).
``cpu_smoke_run`` (M13 smoke.py) is replaced by the same S1/S5/S7 runs made here with ``run_command``.
Synthetic values come from ``seeding.torch_seed_scope(0, "test", <name>)`` (§17.6.2).
"""
from __future__ import annotations

import contextlib
import copy
import csv
import dataclasses
import hashlib
import io
import itertools
import json
import math
import os
import random
import subprocess
import sys
import textwrap
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import refs_upstream
from ilora import run, seeding
from ilora import train as train_module
from ilora.train import SkipGuard, TrainingError

pytestmark = [pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated"),
              pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")]          # torch 2.0.0 torch.load

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
K = 4
NO_BI = "eval.batch_invariance_check=false"
REFERENCE_SETS = ("train.log_every_steps=3",)          # S1 settings; summary lines at s = 3, 6, 9, 10, 12, 15, 18, 20
ONE_EPOCH = ("train.epochs=1",)
CLUSTER_FUNCTIONS = ("load_cluster_artifacts", "parse_expert_mapping", "ClusterHardGate", "verify_live_encoder",
                     "run_cluster", "build_fit_set", "fit_kmeans", "predict")
TIME_COLUMNS = ("time_s", "train_s", "val_s", "save_s", "max_mem_gib", "max_mem_reserved_gib")
TINY_TRAINABLE = {"projector": 8_320, "gate": 256, "lora": 17_408}


# ----------------------------------------------------------------------------
# Configs and runs
# ----------------------------------------------------------------------------
def sasrec_sets() -> list[str]:
    """The SASRec ``--set`` texts of tests/conftest.py ``sasrec_overrides`` (none for the default asset path)."""
    value = os.environ.get("ILORA_SASREC_STATE")
    if not value:
        return []
    state = Path(value)
    return [f"sasrec.state_path={json.dumps(state.as_posix())}",
            f"sasrec.reference_path={json.dumps((state.parent / 'sasrec_movielens_reference.pt').as_posix())}"]


def smoke_cfg(*sets: str):
    """``smoke_cpu`` with ``sets`` (M0 ``load_config``) and the texts it was loaded with."""
    from ilora.config import load_config

    texts = [*sasrec_sets(), *sets]
    cfg, _ = load_config(CONFIG_DIR / "smoke_cpu.yaml", texts)
    return cfg, texts


def run_command(command: str, root: Path, name: str, *sets: str, tokenizer_dir: Path) -> Path:
    """M0 ``run_command`` of ``smoke_cpu`` + ``sets`` into ``root`` (a tmp folder); returns the run folder."""
    from ilora import commands

    cfg, texts = smoke_cfg(*sets)
    return commands.run_command(command, cfg, texts, profile=name, root=root, argv=["test_m10_train", name],
                                tokenizer_dir=tokenizer_dir)


def resume_run(run_dir: Path, tokenizer_dir: Path) -> Path:
    """``main.py train --resume <run_dir>``: the folder's config.yaml and profile (M0 ``load_run_config``)."""
    from ilora import commands

    cfg, profile = commands.load_run_config(run_dir)
    return commands.run_command("train", cfg, [], profile=profile, root=run_dir.parent, argv=["resume", str(run_dir)],
                                tokenizer_dir=tokenizer_dir, resume=run_dir)


def only_child(folder: Path) -> Path:
    (child,) = [p for p in folder.iterdir() if p.is_dir()]
    return child


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def drop_time(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{k: v for k, v in row.items() if k not in TIME_COLUMNS} for row in rows]


def torch_bytes(obj) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8))


@contextlib.contextmanager
def one_thread():
    """``torch.set_num_threads(1)`` for runs compared bit for bit (§17.6.5); restored afterwards."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


# ----------------------------------------------------------------------------
# Spies (call order of evaluate_split, checkpoint markers, M15 functions and build_model)
# ----------------------------------------------------------------------------
class CallLog:
    """Events of one training run: ``("evaluate_split", split, ckpt_tag, cpu_rng_unchanged)``,
    ``("marker", file, folder)``, ``("cluster", name)``, ``("build_model", hard_gate type)``."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def names(self, kind: str) -> list:
        return [e[1:] for e in self.events if e[0] == kind]


def install_spies(mp: pytest.MonkeyPatch, log: CallLog) -> None:
    from ilora import cluster, evaluate
    from ilora import model as model_module

    original_eval = evaluate.evaluate_split

    def eval_spy(model, bundle, data, split, cfg, run_dir, **kwargs):
        before = torch.get_rng_state().clone()
        result = original_eval(model, bundle, data, split, cfg, run_dir, **kwargs)
        log.events.append(("evaluate_split", split, kwargs["ckpt_tag"], torch.equal(before, torch.get_rng_state())))
        return result

    mp.setattr(evaluate, "evaluate_split", eval_spy)
    original_marker = train_module.write_marker

    def marker_spy(path, payload):
        sha = original_marker(path, payload)
        log.events.append(("marker", Path(path).name, Path(path).parent.name))
        return sha

    mp.setattr(train_module, "write_marker", marker_spy)
    for name in CLUSTER_FUNCTIONS:
        original = getattr(cluster, name)

        def cluster_spy(*args, __name=name, __original=original, **kwargs):
            log.events.append(("cluster", __name))
            return __original(*args, **kwargs)

        mp.setattr(cluster, name, cluster_spy)
    original_build = model_module.build_model

    def build_spy(*args, **kwargs):
        log.events.append(("build_model", type(kwargs.get("hard_gate")).__name__))
        return original_build(*args, **kwargs)

    mp.setattr(model_module, "build_model", build_spy)


# ----------------------------------------------------------------------------
# GradScaler stand-ins (the CUDA fp16 scaler cannot run on CPU)
# ----------------------------------------------------------------------------
class FakeScaler:
    """The ``torch.cuda.amp.GradScaler`` interface used by M10 with chosen skipped steps: a skipped step does not
    call ``optimizer.step`` and halves the scale (backoff 0.5), like an inf/NaN gradient on the GPU."""

    def __init__(self, skip_steps, init_scale: float = 65536.0) -> None:
        self.skip_steps, self.value = set(skip_steps), float(init_scale)
        self.index, self.found_inf, self.optimizer_steps = 0, False, 0

    def is_enabled(self) -> bool:
        return True

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        self.found_inf = self.index in self.skip_steps
        if not self.found_inf:
            optimizer.step()
            self.optimizer_steps += 1

    def update(self) -> None:
        if self.found_inf:
            self.value *= 0.5
        self.index += 1

    def get_scale(self) -> float:
        return self.value

    def state_dict(self) -> dict:
        return {"scale": self.value, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000,
                "_growth_tracker": self.index}

    def load_state_dict(self, state) -> None:
        self.value, self.index = float(state["scale"]), int(state["_growth_tracker"])


# ----------------------------------------------------------------------------
# Tiny-model helpers
# ----------------------------------------------------------------------------
def randomize_lora_b(model, name: str, std: float = 0.02) -> None:
    """Non-zero LoRA B so that omega and the LoRA term matter (B = 0 at init, §12.6.3)."""
    from ilora.moe_lora import lora_named_parameters

    with torch.no_grad(), seeding.torch_seed_scope(0, "test", name):
        for n, p in lora_named_parameters(model.llm):
            if n.endswith("lora_B.weight"):
                p.normal_(0.0, std)


def plan_with(cfg, data, flags=None):
    """Epoch-0 ``BatchPlan`` of the smoke data, with the ``hybrid`` flags replaced where ``flags`` gives one."""
    from ilora.prompts import build_batch_plan

    plan = build_batch_plan(data.train, 0, cfg)
    if flags:
        entries = tuple(e._replace(hybrid=flags.get(i, e.hybrid)) for i, e in enumerate(plan.entries))
        plan = dataclasses.replace(plan, entries=entries)
    return plan


def loader(plan, cfg, data, bundle, templates):
    from ilora.batching import make_train_loader

    return make_train_loader(plan, data, cfg, plan.epoch, bundle=bundle, templates=templates)


def trainer_parts(model, cfg):
    from ilora.precision import make_grad_scaler

    return (train_module.build_optimizer(model.param_groups(), cfg.train, model=model),
            make_grad_scaler(cfg.train, model.device), train_module.step_counts(156, cfg.train))


def checkpoint_state(cfg, model, sc, *, epoch: int = 0, step: int = 3, tau: int = 12) -> dict:
    """The run values of ``train.STATE_KEYS`` as ``train()`` passes them to ``save_checkpoint``."""
    prov = train_module.current_provenance(cfg, model)
    frozen = train_module.frozen_checksum(model)
    return {"epoch": epoch, "global_step": step, "tau_done": tau, "config_hash": prov["config_hash"],
            "input_hashes": prov["input_hashes"], "cluster": prov["cluster"],
            "sasrec_state_digest": model.sasrec.state_digest, "frozen_checksum": frozen,
            "frozen_checksum_at_save": dict(frozen),
            "counters": {"skipped_steps": 0, "nonfinite_loss": 0, "gate_calls": tau, "cluster_lookups": 0,
                         "train_s": 0.0, "skip_guard": {"consecutive": 0, "total": 0, "run_start": None,
                                                        "run_scale": None}},
            "epoch_stats": {}, "step_counts": dataclasses.asdict(sc)}


def model_logits(model, batch) -> torch.Tensor:
    """Full logits of the model path (checks, z/omega, inputs_embeds, LLaMA inside the gate context)."""
    model._check_batch(batch, "train")
    on_device = batch.to(model.device)
    with torch.no_grad():
        z, omega = model._z_omega(on_device)
        emb = model._embed_inputs(on_device)
        with model.gate_ctx.use(omega):
            return model.llm(inputs_embeds=emb, attention_mask=on_device.attention_mask, use_cache=False,
                             return_dict=True).logits


# ----------------------------------------------------------------------------------------------- #1
@pytest.mark.criterion("M10-1")
def test_step_counts(paper_cfg, cpu_cfg):
    """paper_spec: 8,265 micro-batches per epoch, tail 9, 517 steps, T 2,585, w 129, 41,325 micro-batches; the
    epoch tau and step ranges of §14.6.2; smoke_cpu 39 / 3 / 10 / 20 / 5 / 78 and the 1-epoch run T 10, w 2."""
    from ilora.prompts import make_batch_plan

    sc = train_module.step_counts(66_126, paper_cfg.train)
    assert (sc.micro_per_epoch, sc.tail, sc.steps_per_epoch, sc.total_steps, sc.warmup_steps, sc.total_micro) == \
        (8_265, 9, 517, 2_585, 129, 41_325)
    assert sc.window_sizes() == (16,) * 516 + (9,)
    for epoch in range(5):
        tau, step = 8_265 * epoch, 517 * epoch
        firsts = []
        for size in sc.window_sizes():
            firsts.append((step, tau + 1))
            tau, step = tau + size, step + 1
        assert firsts[0] == (517 * epoch, 8_265 * epoch + 1)
        assert (step - 1, tau) == (517 * epoch + 516, 8_265 * (epoch + 1))
    for epoch in (0, 4):                         # the M4 plan of the same epoch covers the same tau range
        plan = make_batch_plan(seed=1234, epoch=epoch, epochs=5, num_samples=66_126, micro_batch_size=8,
                               curriculum="linear")
        train_module.check_plan(plan, epoch, sc, tau_done=8_265 * epoch)
        assert (plan.entries[0].tau, plan.entries[-1].tau) == (8_265 * epoch + 1, 8_265 * (epoch + 1))
    smoke = train_module.step_counts(156, cpu_cfg.train)
    assert (smoke.micro_per_epoch, smoke.tail, smoke.steps_per_epoch, smoke.total_steps, smoke.warmup_steps,
            smoke.total_micro) == (39, 3, 10, 20, 5, 78)
    assert smoke.window_sizes() == (4,) * 9 + (3,)
    one = train_module.step_counts(156, dataclasses.replace(cpu_cfg.train, epochs=1))
    assert (one.total_steps, one.warmup_steps, one.total_micro) == (10, 2, 39)                    # 18.6.5
    with pytest.raises(TrainingError, match="micro_batch_size"):
        train_module.step_counts(3, cpu_cfg.train)
    with pytest.raises(TrainingError, match="warmup"):                      # T = 10, 10 // 20 = 0
        train_module.step_counts(156, dataclasses.replace(cpu_cfg.train, epochs=1, warmup_divisor=20))


# ----------------------------------------------------------------------------------------------- #2
def closed_form_lr(s: int, *, start: float, peak: float, floor: float, w: int, total: int) -> float:
    """§14.6.2 in another form: the warmup as an exact fraction of the float inputs, the cosine as cos^2."""
    if s < w:
        return float(Fraction(start) + (Fraction(peak) - Fraction(start)) * s / w)
    return floor + (peak - floor) * math.cos(math.pi * (s - w) / (total - w) / 2) ** 2


@pytest.mark.criterion("M10-2")
def test_lr_matches_upstream(paper_cfg):
    """lr_at at the §14.6.2 points (7 significant digits as printed there, 1e-12 against the closed form), equal to
    ``refs_upstream.upstream_lr`` (U optims.py:20-36, cur_epoch = s // 517, max_step 2,585) at every s, and inside
    [1e-6, 1e-4]."""
    tc = paper_cfg.train
    sc = train_module.step_counts(66_126, tc)
    printed = {0: 1e-6, 64: 5.011628e-05, 128: 9.923256e-05, 129: 1e-4, 516: 9.405772e-05, 1_292: 5.461093e-05,
               2_584: 1.000040e-06}
    for s, value in printed.items():
        got = train_module.lr_at(s, sc, tc)
        assert f"{got:.6e}" == f"{value:.6e}", s
    for s in range(sc.total_steps):
        got = train_module.lr_at(s, sc, tc)
        exact = closed_form_lr(s, start=tc.lr_warmup_start, peak=tc.lr_peak, floor=tc.lr_floor, w=129, total=2_585)
        assert abs(got - exact) <= 1e-12 * exact, s
        assert got == refs_upstream.upstream_lr(s, s // 517, 2_585, init_lr=1e-4, min_lr=1e-6, warmup_start_lr=1e-6,
                                                warmup_steps=2_585 // 20), s
        assert 1e-6 <= got <= 1e-4, s
    for bad in (-1, 2_585, 1.0, True):
        with pytest.raises(TrainingError):
            train_module.lr_at(bad, sc, tc)


# ----------------------------------------------------------------------------------------------- #3, M6-7
@pytest.mark.criterion("M10-3", "M6-7")
def test_optimizer_groups(tiny_model, fake_hard_gate, smoke_data, cpu_cfg, paper_cfg):
    """Adam (not AdamW) with groups (projector, gate, lora) in that order, weight decay (1e-5, 1e-5, 0), lr scale 1,
    betas (0.9, 0.999), eps 1e-8, amsgrad False; projector and gate share the LoRA lr at every step (M6-7);
    cluster_hard and K = 1 have (projector, lora) and 256 parameters fewer."""
    model = tiny_model()
    groups = model.param_groups()
    opt = train_module.build_optimizer(groups, cpu_cfg.train, model=model)
    assert type(opt) is torch.optim.Adam
    assert [pg["name"] for pg in opt.param_groups] == ["projector", "gate", "lora"]
    assert [pg["weight_decay"] for pg in opt.param_groups] == [1e-5, 1e-5, 0.0]
    for pg in opt.param_groups:
        assert (pg["betas"], pg["eps"], pg["amsgrad"]) == ((0.9, 0.999), 1e-8, False)
        assert [id(p) for p in pg["params"]] == [id(p) for _, p in groups[pg["name"]]]
    assert train_module.group_sizes(groups) == {**TINY_TRAINABLE, "total": 25_984}
    for tc, n in ((paper_cfg.train, 66_126), (cpu_cfg.train, 156)):
        sc = train_module.step_counts(n, tc)
        for s in range(sc.total_steps):
            lrs = train_module.group_lrs(s, sc, tc, train_module.GROUP_ORDER)
            assert lrs["projector"] == lrs["lora"] == lrs["gate"] == train_module.lr_at(s, sc, tc), s
    sc = train_module.step_counts(156, cpu_cfg.train)
    train_module.set_lrs(opt, train_module.group_lrs(7, sc, cpu_cfg.train, list(groups)))
    assert [pg["lr"] for pg in opt.param_groups] == [train_module.lr_at(7, sc, cpu_cfg.train)] * 3
    released = dataclasses.replace(cpu_cfg.train, lr_scale_gate=0.3)              # variant_released_gate_lr (D19)
    lrs = train_module.group_lrs(7, sc, released, train_module.GROUP_ORDER)
    assert lrs["gate"] == 0.3 * lrs["lora"] and lrs["projector"] == lrs["lora"]
    table = {s.sample_id: s.row % K for split in ("train", "val", "test") for s in smoke_data.samples(split)}
    hard = tiny_model("router.routing_mode=cluster_hard", "cluster.artifact_dir=unused_by_m10/cluster",
                      hard_gate=fake_hard_gate(table, K))
    k1 = tiny_model("lora.num_experts=1", "router.routing_mode=none")
    for other in (hard, k1):
        other_opt = train_module.build_optimizer(other.param_groups(), cpu_cfg.train, model=other)
        assert [pg["name"] for pg in other_opt.param_groups] == ["projector", "lora"]
        assert [pg["weight_decay"] for pg in other_opt.param_groups] == [1e-5, 0.0]
        assert train_module.group_sizes(other.param_groups())["total"] == 25_984 - 256


@pytest.mark.criterion("M10-3")
@pytest.mark.parametrize("case", ["fp16", "two_groups", "outside"])
def test_optimizer_rejects(case, tiny_model, cpu_cfg):
    """An fp16 parameter, a parameter in two groups and a trainable parameter outside the groups -> TrainingError."""
    model = tiny_model()
    groups = {g: list(items) for g, items in model.param_groups().items()}
    if case == "fp16":
        name, p = groups["projector"][0]
        groups["projector"][0] = (name, nn.Parameter(p.detach().half()))
        match = "float32"
    elif case == "two_groups":
        groups["projector"] = sorted(groups["projector"] + [groups["lora"][0]], key=lambda item: item[0])
        match = "two groups"
    else:
        model.llm.lm_head.weight.requires_grad_(True)
        match = "outside every group"
    with pytest.raises(TrainingError, match=match):
        train_module.build_optimizer(groups, cpu_cfg.train, model=model)


@pytest.mark.criterion("M10-3")
def test_group_sizes_llama7b_shape(llama7b_skeleton, paper_cfg):
    """Llama-2-7B shape (meta skeleton, hidden 4,096, intermediate 11,008, 32 layers): groups 17,047,552 / 256 /
    19,988,480 (total 37,036,288, without the gate 37,036,032), accepted by build_optimizer."""
    from ilora.moe_lora import GateContext, MoELoRALinear, lora_named_parameters
    from ilora.projector import build_projector
    from ilora.router import Router
    from ilora.sections import RouterConfig

    skeleton = copy.deepcopy(llama7b_skeleton)
    ctx = GateContext(K)
    for i, layer in enumerate(skeleton.model.layers):
        for parent_name, targets in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                     ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            parent = getattr(layer, parent_name)
            for target in targets:
                setattr(parent, target, MoELoRALinear(getattr(parent, target), r=8, num_experts=K, alpha=32,
                                                      dropout=0.1, scaling_rule="expert_rank", gate_ctx=ctx,
                                                      name=f"model.layers.{i}.{parent_name}.{target}"))
    projector = build_projector(64, 4096, 1234)
    gate = Router(RouterConfig(routing_mode="dynamic"), num_experts=K, z_dim=64, main_seed=1234, hard_gate=None)
    groups = {"projector": sorted((f"projector.{n}", p) for n, p in projector.named_parameters()),
              "gate": [("omega_source.gate.proj.weight", gate.gate.proj.weight)],
              "lora": sorted((f"llm.{n}", p) for n, p in lora_named_parameters(skeleton))}
    sizes = train_module.group_sizes(groups)
    assert sizes == {"projector": 17_047_552, "gate": 256, "lora": 19_988_480, "total": 37_036_288}
    assert len(groups["lora"]) == 448
    opt = train_module.build_optimizer(groups, paper_cfg.train)
    assert [sum(p.numel() for p in pg["params"]) for pg in opt.param_groups] == [17_047_552, 256, 19_988_480]
    no_gate = {g: groups[g] for g in ("projector", "lora")}
    assert train_module.group_sizes(no_gate)["total"] == 37_036_032


@pytest.mark.criterion("M10-3")
def test_mapping_config_rejected(profile_cfg):
    """Group values are scalar keys (§14.6.9): a mapping in the ``train`` section is a ConfigError of the M0 loader."""
    from ilora.config import ConfigError

    for text in ("train.lr_scale_projector={projector: 1.0}", "train.weight_decay_gate={gate: 1.0e-5}",
                 "train.lr_scale={projector: 1.0, gate: 0.3, lora: 1.0}", "train.adam_beta1=[0.9, 0.999]"):
        with pytest.raises(ConfigError):
            profile_cfg("paper_spec", text)


# ----------------------------------------------------------------------------------------------- #6
@pytest.mark.criterion("M10-6")
def test_skip_guard():
    """15 consecutive skips then a success pass; the 16th consecutive skip is a TrainingError naming the step range,
    the scale change and the last gradient norm."""
    guard = SkipGuard(16)
    scale = 65536.0
    for s in range(15):
        guard.update(s, True, scale, scale / 2)
        scale /= 2
    guard.update(15, False, scale, scale)
    assert (guard.consecutive, guard.total) == (0, 15)
    for s in range(16, 31):
        guard.update(s, True, scale, scale / 2)
        scale /= 2
    with pytest.raises(TrainingError, match=r"steps 16-31") as info:
        guard.update(31, True, scale, scale / 2, grad_norm=float("inf"))
    message = str(info.value)
    assert "16 consecutive" in message and "scale 2.0 -> " in message and "grad norm inf" in message
    for bad in (0, -1, 1.5, True):
        with pytest.raises(TrainingError):
            SkipGuard(bad)


# ----------------------------------------------------------------------------------------------- #9
@pytest.mark.criterion("M10-9")
@pytest.mark.parametrize("counts, best, last, order", [([40, 45, 45, 44, 43], 2, 4, (2, 4)),
                                                      ([40, 41, 42, 43, 45], 4, 4, (4,))])
def test_select_epochs(counts, best, last, order):
    """best = most validation answers correct (later epoch on ties), last = final epoch, test order ascending once."""
    from ilora.evaluate import SplitMetrics

    val = {e: SplitMetrics("val", 94, 90, c, 90 / 94, c / 90, c / 94) for e, c in enumerate(counts)}
    assert train_module.select_epochs(val) == (best, last)
    assert train_module.epochs_to_test(best, last) == order
    doc = train_module.selection_document(val, best, last)
    assert (doc["best_val_epoch"], doc["last_epoch"], doc["primary"]) == (best, last, "best_val")
    assert doc["test_dirs"] == {"best_val": f"eval/epoch{best}/test", "last": f"eval/epoch{last}/test"}
    assert [doc["val"][str(e)]["n_correct"] for e in range(5)] == counts
    with pytest.raises(TrainingError, match="N differs"):
        train_module.select_epochs({**val, 4: dataclasses.replace(val[4], n_total=95)})
    with pytest.raises(TrainingError):
        train_module.select_epochs({e: val[e] for e in (0, 2)})
    with pytest.raises(TrainingError):
        train_module.select_epochs({})


# ----------------------------------------------------------------------------------------------- #11
@pytest.mark.criterion("M10-11")
def test_sasrec_train_mode_detected():
    """A model whose train() leaves SASRec in train mode is refused (D12); a correct one passes."""
    class Mock:
        def __init__(self, keeps_eval: bool) -> None:
            self.keeps_eval, self.calls = keeps_eval, 0

        def train(self, mode: bool = True):
            self.calls += 1
            return self

        def sasrec_is_eval(self) -> bool:
            return self.keeps_eval

    good, bad = Mock(True), Mock(False)
    train_module.enter_train_mode(good)
    assert good.calls == 1
    with pytest.raises(TrainingError, match="SASRec"):
        train_module.enter_train_mode(bad)


# ----------------------------------------------------------------------------------------------- #4
def assert_same_gradients(got: dict, want: dict) -> int:
    """Gradient dicts (name -> tensor or None) equal within relative 1e-6 of the largest entry; returns the tensors
    compared."""
    checked = 0
    for name, b in want.items():
        a = got[name]
        assert (a is None) == (b is None), name
        if b is None:
            continue
        scale = float(b.abs().max())
        assert float((a - b).abs().max()) <= 1e-6 * max(scale, 1e-30), name
        checked += 1
    return checked


@pytest.mark.criterion("M10-4")
@pytest.mark.parametrize("n", [9, 16])
def test_accumulation_matches_mean_loss(n, tiny_model, paper_cfg, cpu_cfg, smoke_data, bundle, templates,
                                        torch_threads1):
    """D24: the gradient accumulated by ``accumulate_window`` over a window of n micro-batches with the paper_spec
    divisor ``train.grad_accum`` = 16 (``loss_i / 16`` each) equals the autograd gradient of (sum loss_i) / 16
    (relative 1e-6, the same dropout masks), for the full window (16) and the last window of an epoch (9) alike:
    Lightning 1.8.6 divides every micro-batch loss by accumulate_grad_batches (PL1.8.6
    loops/optimization/optimizer_loop.py:88, :416). A divisor below the window size is a TrainingError."""
    from ilora.precision import make_grad_scaler

    divisor = paper_cfg.train.grad_accum
    assert divisor == 16
    model = tiny_model()
    randomize_lora_b(model, f"m10_accum_{n}")
    plan = plan_with(cpu_cfg, smoke_data)
    entries = plan.entries[:n]
    batches = list(itertools.islice(loader(plan, cpu_cfg, smoke_data, bundle, templates), n))
    scaler = make_grad_scaler(cpu_cfg.train, model.device)
    params = [(name, p) for items in model.param_groups().values() for name, p in items]
    model.train()
    with torch.random.fork_rng(devices=[]):
        state = torch.get_rng_state()
        micros = train_module.accumulate_window(model, iter(batches), entries, smoke_data.train, precision="fp32",
                                                scaler=scaler, divisor=divisor)
        accumulated = {name: None if p.grad is None else p.grad.clone() for name, p in params}
        for _, p in params:
            p.grad = None
        torch.set_rng_state(state)                              # the same LoRA dropout masks
        losses = [model.train_forward(batch).loss for batch in batches]
        (torch.stack(losses).sum() / divisor).backward()
    assert [m.tau for m in micros] == [e.tau for e in entries] and len(micros) == n
    assert [float(m.loss) for m in micros] == [float(x.detach()) for x in losses]
    want = {name: p.grad for name, p in params}
    assert assert_same_gradients(accumulated, want) >= 28 + 1    # every LoRA tensor and the gate at least
    for bad in (n - 1, float(divisor), True):
        with pytest.raises(TrainingError, match="divisor"):
            train_module.accumulate_window(model, iter(batches), entries, smoke_data.train, precision="fp32",
                                           scaler=scaler, divisor=bad)


@pytest.mark.criterion("M10-4")
@pytest.mark.parametrize("n", [3, 4])
def test_train_window_divides_by_grad_accum(n, tiny_model, cpu_cfg, smoke_data, bundle, templates, torch_threads1,
                                            monkeypatch):
    """``train_window`` divides every window by ``train.grad_accum`` (4 in smoke_cpu, D24): the gradient it steps
    with (read by a spy on ``gradient_norms``, before ``scaler.step``) equals the autograd gradient of
    (sum loss_i) / 4 for a full window (4) and for the last window of an epoch (3), not (sum loss_i) / 3."""
    model = tiny_model()
    randomize_lora_b(model, f"m10_window_divisor_{n}")
    plan = plan_with(cpu_cfg, smoke_data)
    entries = plan.entries[:n]
    batches = list(itertools.islice(loader(plan, cpu_cfg, smoke_data, bundle, templates), n))
    opt, scaler, sc = trainer_parts(model, cpu_cfg)
    params = [(name, p) for items in model.param_groups().values() for name, p in items]
    seen: dict = {}
    original = train_module.gradient_norms

    def norms_spy(groups):
        seen.update({name: None if p.grad is None else p.grad.detach().clone() for name, p in params})
        return original(groups)

    monkeypatch.setattr(train_module, "gradient_norms", norms_spy)
    model.train()
    assert cpu_cfg.train.grad_accum == 4 and sc.window_sizes()[-1] == 3
    with torch.random.fork_rng(devices=[]):
        state = torch.get_rng_state()
        losses = [model.train_forward(batch).loss for batch in batches]
        (torch.stack(losses).sum() / cpu_cfg.train.grad_accum).backward()
        want = {name: None if p.grad is None else p.grad.detach().clone() for name, p in params}
        for _, p in params:
            p.grad = None
        torch.set_rng_state(state)                              # the same LoRA dropout masks
        result = train_module.train_window(model, opt, scaler, iter(batches), entries, smoke_data.train, step=0,
                                           sc=sc, tc=cpu_cfg.train)
    assert len(result.micros) == n and result.losses == [float(x.detach()) for x in losses]
    assert assert_same_gradients(seen, want) >= 28 + 1


# ----------------------------------------------------------------------------------------------- #5, M6-6
@pytest.mark.criterion("M10-5", "M6-6")
def test_text_only_window_leaves_projector(tiny_model, cpu_cfg, smoke_data, bundle, templates, torch_threads1):
    """After an all-text-only window the projector got no gradient and its values and Adam state are unchanged bit
    for bit while LoRA moved; one hybrid micro-batch gives it a gradient; a model whose projector gets a (zero)
    gradient in a text-only window breaks the presence rule (TrainingError)."""
    model = tiny_model()
    flags = {i: i < 4 or i == 10 for i in range(16)}            # windows: hybrid x4 | text x4 | one hybrid | text x4
    plan = plan_with(cpu_cfg, smoke_data, flags)
    batches = loader(plan, cpu_cfg, smoke_data, bundle, templates)
    opt, scaler, sc = trainer_parts(model, cpu_cfg)
    model.train()

    def window(i: int):
        return train_module.train_window(model, opt, scaler, batches, plan.entries[4 * i:4 * i + 4], smoke_data.train,
                                         step=i, sc=sc, tc=cpu_cfg.train)

    projector = model.param_groups()["projector"]
    lora = model.param_groups()["lora"]
    first = window(0)
    assert first.norms["projector"] is not None and all(opt.state[p]["step"].item() == 1 for _, p in projector)
    values = {name: p.detach().clone() for name, p in projector}
    states = {name: {k: v.clone() for k, v in opt.state[p].items()} for name, p in projector}
    lora_before = {name: p.detach().clone() for name, p in lora}
    second = window(1)
    assert second.norms["projector"] is None and second.norms["lora"] is not None
    for name, p in projector:
        assert p.grad is None and same_bits(p.detach(), values[name]), name
        assert set(opt.state[p]) == set(states[name])
        for key, value in states[name].items():
            assert same_bits(opt.state[p][key], value), (name, key)
    assert any(not same_bits(p.detach(), lora_before[name]) for name, p in lora)
    third = window(2)
    assert third.norms["projector"] is not None and all(opt.state[p]["step"].item() == 2 for _, p in projector)
    original = model.train_forward

    def leaky_forward(batch):                    # mock: the projector enters the text-only loss with weight 0
        out = original(batch)
        extra = sum(p.sum() for p in model.projector.parameters()) * 0.0
        return dataclasses.replace(out, loss=out.loss + extra)

    model.train_forward = leaky_forward
    with pytest.raises(TrainingError, match="text-only window"):
        window(3)


# ----------------------------------------------------------------------------------------------- M6-8, #7
@pytest.fixture(scope="module")
def saved_tiny(cpu_cfg, smoke_data, bundle, templates, sasrec_state_path, tmp_path_factory):
    """A tiny model after a hybrid window and a text-only window (Adam state for every group, the projector without
    a gradient in the last window), saved with ``save_checkpoint`` right after the text-only window. One thread."""
    from ilora.model import build_model

    with one_thread():
        model = build_model(cpu_cfg, bundle, smoke_data.catalog, llm_path=None)
        randomize_lora_b(model, "m10_saved_tiny")
        plan = plan_with(cpu_cfg, smoke_data, {i: i < 4 for i in range(8)})
        batches = loader(plan, cpu_cfg, smoke_data, bundle, templates)
        opt, scaler, sc = trainer_parts(model, cpu_cfg)
        model.train()
        results = [train_module.train_window(model, opt, scaler, batches, plan.entries[4 * i:4 * i + 4],
                                             smoke_data.train, step=i, sc=sc, tc=cpu_cfg.train) for i in range(2)]
        ckpt = tmp_path_factory.mktemp("m10_saved_tiny") / "epoch0"
        files = train_module.save_checkpoint(ckpt, model, opt, scaler, checkpoint_state(cpu_cfg, model, sc))
    return SimpleNamespace(model=model, opt=opt, scaler=scaler, sc=sc, ckpt=ckpt, files=files, results=results)


def copy_ckpt(saved, tmp_path: Path) -> Path:
    import shutil

    return Path(shutil.copytree(saved.ckpt, tmp_path / "epoch0"))


@pytest.mark.criterion("M6-8")
def test_checkpoint_keys_after_text_only_window(saved_tiny, tiny_model, cpu_cfg, smoke_data, bundle, templates,
                                                torch_threads1):
    """A checkpoint saved right after a text-only window still holds the 4 projector keys (no requires_grad
    filter, 36 keys in all); the reloaded model gives the same logits bit for bit."""
    assert saved_tiny.results[0].norms["projector"] is not None and saved_tiny.results[1].norms["projector"] is None
    state = train_module.read_trainable_state(saved_tiny.ckpt)
    projector_keys = sorted(k for k in state if k.startswith("projector."))
    assert projector_keys == [f"projector.mlp_proj.{i}.{w}" for i in (0, 2) for w in ("bias", "weight")]
    assert len(state) == 36 and sum(k.startswith("generated_frozen.") for k in state) == 3
    fresh = tiny_model()
    assert train_module.load_model_state(saved_tiny.ckpt, fresh) == saved_tiny.files["trainable.pt"]
    batch = next(iter(loader(plan_with(cpu_cfg, smoke_data, {i: True for i in range(39)}), cpu_cfg, smoke_data,
                             bundle, templates)))
    model = saved_tiny.model
    was_training = model.training
    try:
        model.eval()
        fresh.eval()
        assert same_bits(model_logits(model, batch), model_logits(fresh, batch))
    finally:
        model.train(was_training)


def assert_optimizer_equal(a: torch.optim.Optimizer, b: torch.optim.Optimizer) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    assert sorted(sa["state"]) == sorted(sb["state"])
    for index, values in sa["state"].items():
        assert set(values) == set(sb["state"][index])
        for key, value in values.items():
            assert same_bits(value, sb["state"][index][key]), (index, key)


@pytest.mark.criterion("M10-7")
def test_checkpoint_round_trip_state(saved_tiny, tiny_model, cpu_cfg, tmp_path, torch_threads1):
    """save_checkpoint -> load_checkpoint into a new model, optimizer and scaler: parameters, Adam state, GradScaler
    state and the CPU RNG state are equal bit for bit; markers, sha256 and weights_only loading."""
    model, opt, sc, files = saved_tiny.model, saved_tiny.opt, saved_tiny.sc, saved_tiny.files
    ckpt = copy_ckpt(saved_tiny, tmp_path)
    assert sorted(p.name for p in ckpt.iterdir()) == ["STATE_SAVED", "train_state.pt", "trainable.pt"]
    assert read_json(ckpt / "STATE_SAVED")["files"] == files
    assert all(run.content_sha256(ckpt / name) == sha for name, sha in files.items())
    state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
    assert state["format_version"] == train_module.TRAIN_STATE_FORMAT and state["scaler"] == {}
    assert state["group_param_names"] == train_module.trainable_names(model) and state["rng"]["cuda"] is None
    torch.rand(5)                                                   # move the global CPU generator
    assert not torch.equal(torch.get_rng_state(), state["rng"]["cpu"])
    fresh = tiny_model()
    fresh_opt, fresh_scaler, _ = trainer_parts(fresh, cpu_cfg)
    info = train_module.load_checkpoint(ckpt, fresh, fresh_opt, fresh_scaler, cpu_cfg)
    assert (info["epoch"], info["global_step"], info["tau_done"]) == (0, 3, 12)
    assert torch.equal(torch.get_rng_state(), state["rng"]["cpu"])
    for (name, a), (_, b) in zip(model.named_parameters(), fresh.named_parameters()):
        assert same_bits(a, b), name
    assert_optimizer_equal(opt, fresh_opt)
    for key, value in model.generated_frozen_state().items():
        assert same_bits(value, fresh.generated_frozen_state()[key])
    # a populated GradScaler state (the CUDA fp16 scaler) goes through the same path with the stand-in scaler
    stub = FakeScaler(skip_steps=(), init_scale=1024.0)
    stub.index = 7
    ckpt2 = tmp_path / "epoch_stub"
    train_module.save_checkpoint(ckpt2, model, opt, stub, checkpoint_state(cpu_cfg, model, sc))
    other_model = tiny_model()
    other_opt, _, _ = trainer_parts(other_model, cpu_cfg)
    other = FakeScaler(skip_steps=())
    train_module.load_checkpoint(ckpt2, other_model, other_opt, other, cpu_cfg)
    assert other.state_dict() == stub.state_dict()
    third = tiny_model()
    third_opt, disabled, _ = trainer_parts(third, cpu_cfg)
    with pytest.raises(TrainingError, match="GradScaler state"):          # enabled state into a disabled scaler
        train_module.load_checkpoint(ckpt2, third, third_opt, disabled, cpu_cfg)


def rewrite_checkpoint(ckpt: Path, *, trainable=None, train_state=None) -> None:
    """Replace trainable.pt and/or train_state.pt and keep every sha256 consistent (markers, trainable_sha256), so
    that the check under test, not the sha256 check, is the one that fails."""
    marker = read_json(ckpt / "STATE_SAVED")
    if trainable is not None:
        data = torch_bytes(trainable)
        (ckpt / "trainable.pt").write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        marker["files"]["trainable.pt"] = sha
        if train_state is None:
            train_state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
        train_state["trainable_sha256"] = sha
    if train_state is not None:
        data = torch_bytes(train_state)
        (ckpt / "train_state.pt").write_bytes(data)
        marker["files"]["train_state.pt"] = hashlib.sha256(data).hexdigest()
    (ckpt / "STATE_SAVED").write_text(json.dumps(marker), encoding="utf-8")


REJECT_CASES = ["missing_key", "extra_key", "shape_dtype", "generated_frozen", "config_hash", "group_order",
                "frozen_checksum", "sha256"]


@pytest.mark.criterion("M10-7")
@pytest.mark.parametrize("case", REJECT_CASES)
def test_checkpoint_load_rejects(case, saved_tiny, tiny_model, cpu_cfg, profile_cfg, tmp_path, torch_threads1):
    """Each mismatch is a TrainingError before anything is loaded: missing or extra key, dtype, one bit of a
    generated_frozen row, another config hash, another group order, a frozen-parameter checksum that differs from the
    rebuilt model (another base model or SASRec), a changed file (sha256). The model-only reader of the eval command
    and the test (load_model_state) refuses the same trainable.pt changes."""
    ckpt = copy_ckpt(saved_tiny, tmp_path)
    trainable = torch.load(ckpt / "trainable.pt", map_location="cpu", weights_only=True)
    cfg = cpu_cfg
    if case == "missing_key":
        del trainable[next(k for k in sorted(trainable) if ".lora_A." in k)]
        match = "missing keys"
    elif case == "extra_key":
        trainable["llm.model.layers.0.extra.weight"] = torch.zeros(2)
        match = "unexpected keys"
    elif case == "shape_dtype":
        trainable["projector.mlp_proj.0.weight"] = trainable["projector.mlp_proj.0.weight"].double()
        match = "float64"
    elif case == "generated_frozen":
        rows = trainable["generated_frozen.embed_rows"].clone()
        rows.view(torch.int32)[0, 0] ^= 1                                 # one bit of one new token row
        trainable["generated_frozen.embed_rows"] = rows
        match = "generated_frozen"
    elif case == "config_hash":
        cfg = profile_cfg("smoke_cpu", "train.log_every_steps=5")
        match = "config_hash"
    elif case == "group_order":
        state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
        lora = state["group_param_names"]["lora"]
        state["group_param_names"]["lora"] = [lora[1], lora[0], *lora[2:]]
        rewrite_checkpoint(ckpt, train_state=state)
        match = "group_param_names"
    elif case == "frozen_checksum":
        state = torch.load(ckpt / "train_state.pt", map_location="cpu", weights_only=True)
        name = sorted(state["frozen_checksum"])[0]
        state["frozen_checksum"][name] += 1.0
        rewrite_checkpoint(ckpt, train_state=state)
        match = "frozen parameter checksum differs"
    else:
        data = bytearray((ckpt / "trainable.pt").read_bytes())
        data[len(data) // 2] ^= 0x01
        (ckpt / "trainable.pt").write_bytes(bytes(data))
        match = "sha256"
    if case in ("missing_key", "extra_key", "shape_dtype", "generated_frozen"):
        rewrite_checkpoint(ckpt, trainable=trainable)
    fresh = tiny_model()
    fresh_opt, fresh_scaler, _ = trainer_parts(fresh, cfg)
    before = {name: p.detach().clone() for name, p in fresh.named_parameters()}
    with pytest.raises(TrainingError, match=match):
        train_module.load_checkpoint(ckpt, fresh, fresh_opt, fresh_scaler, cfg)
    assert all(same_bits(p.detach(), before[name]) for name, p in fresh.named_parameters())
    assert fresh_opt.state_dict()["state"] == {}
    if case in ("missing_key", "extra_key", "shape_dtype", "generated_frozen", "sha256"):
        with pytest.raises(TrainingError, match=match):
            train_module.load_model_state(ckpt, tiny_model())


@pytest.mark.criterion("M10-7")
def test_checkpoint_markers_and_readers(saved_tiny, tiny_model, cpu_cfg, tmp_path, torch_threads1):
    """read_trainable_state needs a marker (STATE_SAVED or COMPLETE) and the marker sha256; COMPLETE and STATE_SAVED
    must agree; save_checkpoint never overwrites; load_model_state returns the trainable.pt sha256."""
    model, opt, scaler, sc, files = saved_tiny.model, saved_tiny.opt, saved_tiny.scaler, saved_tiny.sc, saved_tiny.files
    ckpt = copy_ckpt(saved_tiny, tmp_path)
    with pytest.raises(TrainingError, match="never overwritten"):
        train_module.save_checkpoint(ckpt, model, opt, scaler, checkpoint_state(cpu_cfg, model, sc))
    with pytest.raises(TrainingError, match="misses"):
        train_module.save_checkpoint(tmp_path / "other", model, opt, scaler, {"epoch": 0})
    assert set(train_module.read_trainable_state(ckpt)) == set(model.trainable_state()) | {
        "generated_frozen.token_ids", "generated_frozen.embed_rows", "generated_frozen.lm_head_rows"}
    val_sha = train_module.write_marker(ckpt / "val_metrics.json", {"epoch": 0})
    train_module.write_marker(ckpt / "COMPLETE", {"epoch": 0, "files": {**files, "val_metrics.json": val_sha}})
    assert train_module.checkpoint_files(ckpt, need=("trainable.pt", "train_state.pt", "val_metrics.json")) == \
        {**files, "val_metrics.json": val_sha}
    assert train_module.load_model_state(ckpt, tiny_model()) == files["trainable.pt"]
    (ckpt / "STATE_SAVED").unlink()
    assert train_module.load_model_state(ckpt, tiny_model()) == files["trainable.pt"]     # COMPLETE alone suffices
    (ckpt / "COMPLETE").unlink()
    with pytest.raises(TrainingError, match="marker"):
        train_module.read_trainable_state(ckpt)
    bad = tmp_path / "bad"
    train_module.save_checkpoint(bad, model, opt, scaler, checkpoint_state(cpu_cfg, model, sc))
    (bad / "COMPLETE").write_text(json.dumps({"files": {"trainable.pt": "0" * 64}}), encoding="utf-8")
    with pytest.raises(TrainingError, match="disagree"):
        train_module.read_trainable_state(bad)


# ----------------------------------------------------------------------------------------------- runs with the stand-in GradScaler
SKIPPED = tuple(range(3, 18))                       # 15 consecutive skipped steps (train.max_consecutive_skips 16)
SKIP_RUN_SETS = (*ONE_EPOCH, "train.grad_accum=1", NO_BI)   # 39 steps of one micro-batch, T 39, w 9


@pytest.fixture(scope="module")
def skip_run(tokenizer_dir, sasrec_state_path, tmp_path_factory):
    """One dynamic epoch through ``run_command('train')`` with the stand-in GradScaler skipping steps 3-17, spies on
    evaluate_split, the markers, M15 and build_model, and the Python/numpy global RNG states before and after."""
    log, scalers = CallLog(), []

    def fake_make_grad_scaler(tc, device):
        scalers.append(FakeScaler(SKIPPED, init_scale=tc.scaler_init_scale))
        return scalers[-1]

    root = tmp_path_factory.mktemp("m10_skip_run")
    py_before, np_before = random.getstate(), np.random.get_state()
    with pytest.MonkeyPatch.context() as mp:
        install_spies(mp, log)
        mp.setattr(train_module, "make_grad_scaler", fake_make_grad_scaler)
        run_dir = run_command("train", root, "m10_skip", *SKIP_RUN_SETS, tokenizer_dir=tokenizer_dir)
    return SimpleNamespace(run_dir=run_dir, log=log, scaler=scalers[0], python=(py_before, random.getstate()),
                           numpy=(np_before, np.random.get_state()))


@pytest.mark.criterion("M10-12")
def test_global_rng_untouched(skip_run):
    """After a one-epoch smoke training the Python ``random`` and numpy global states equal the start (nothing uses
    them, §14.6.5); the torch CPU RNG state is the same before and after every M11 evaluation (fork_rng)."""
    assert skip_run.python[0] == skip_run.python[1]
    (name_a, keys_a, *rest_a), (name_b, keys_b, *rest_b) = skip_run.numpy
    assert name_a == name_b and np.array_equal(keys_a, keys_b) and rest_a == rest_b
    evaluations = skip_run.log.names("evaluate_split")
    assert [(split, tag) for split, tag, _ in evaluations] == [("val", "epoch0"), ("test", "epoch0")]
    assert all(unchanged for _, _, unchanged in evaluations)
    assert run.read_meta(skip_run.run_dir)["status"] == "completed"


@pytest.mark.criterion("M10-6", "M10-2")
def test_skipped_steps_keep_schedule(skip_run):
    """Skipped steps (stand-in for the CUDA fp16 GradScaler): every step row records ``skipped``, the scale halves on
    each skip, the lr follows lr_at(s) and tau continues through the 15 skipped steps, and the run completes; the
    optimizer stepped only on the other 24 steps (§14.6.2 [결정], §14.6.8)."""
    from ilora.config import load_config

    run_dir = skip_run.run_dir
    cfg, _ = load_config(run_dir / "config.yaml")
    sc = train_module.step_counts(156, cfg.train)
    assert (sc.total_steps, sc.warmup_steps) == (39, 9)
    rows = read_csv(run_dir / "records" / "train" / "optimizer_steps.csv")
    assert [int(r["step"]) for r in rows] == list(range(39))
    assert [int(r["step"]) for r in rows if r["skipped"] == "1"] == list(SKIPPED)
    scale = 65536.0
    for s, row in enumerate(rows):
        assert (int(row["tau_first"]), int(row["tau_last"]), int(row["n_micro"])) == (s + 1, s + 1, 1)
        assert float(row["lr_lora"]) == train_module.lr_at(s, sc, cfg.train) == float(row["lr_projector"])
        after = scale / 2 if s in SKIPPED else scale
        assert (float(row["scale_before"]), float(row["scale_after"])) == (scale, after), s
        scale = after
    assert scale == 65536.0 / 2 ** 15
    summary = read_json(run_dir / "records" / "train" / "run_summary.json")
    assert summary["status"] == "completed" and summary["skipped_steps"] == list(SKIPPED)
    assert summary["train_summary"]["skipped_steps"] == 15
    (epoch,) = read_csv(run_dir / "records" / "train" / "epochs.csv")
    assert int(epoch["n_skipped_steps"]) == 15
    assert skip_run.scaler.optimizer_steps == 39 - 15
    assert train_module.SUMMARY_PREFIX + " 39/39 epoch 0" in (run_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.criterion("M7-8")
def test_dynamic_never_calls_cluster(skip_run):
    """A dynamic training run calls no public M15 function (spies on ilora.cluster) and builds the model without a
    hard gate [SPEC §22]."""
    assert skip_run.log.names("cluster") == []
    assert skip_run.log.names("build_model") == [("NoneType",)]
    assert "cluster_live_check" not in run.read_meta(skip_run.run_dir)


@pytest.mark.criterion("M10-6")
def test_sixteen_consecutive_skips_stop(tokenizer_dir, sasrec_state_path, tmp_path, monkeypatch):
    """The 16th consecutive skipped step stops the run with a TrainingError naming steps 3-18; the step rows up to
    there (16 skipped) and run_summary.json (failed) are written by the recorder's close; no checkpoint."""
    monkeypatch.setattr(train_module, "make_grad_scaler", lambda tc, device: FakeScaler(range(3, 19)))
    root = tmp_path / "runs"
    with pytest.raises(TrainingError, match=r"16 consecutive GradScaler skips at steps 3-18"):
        run_command("train", root, "m10_skip16", *SKIP_RUN_SETS, tokenizer_dir=tokenizer_dir)
    run_dir = only_child(root)
    assert run.read_meta(run_dir)["status"] == "failed"
    rows = read_csv(run_dir / "records" / "train" / "optimizer_steps.csv")
    assert [int(r["step"]) for r in rows] == list(range(19))
    assert [int(r["step"]) for r in rows if r["skipped"] == "1"] == list(range(3, 19))
    summary = read_json(run_dir / "records" / "train" / "run_summary.json")
    assert summary["status"] == "failed" and summary["skipped_steps"] == list(range(3, 19))
    assert summary["micro_batches"] == 19 and summary["train_summary"]["status"] == "failed"
    assert list((run_dir / "checkpoints").iterdir()) == []


@pytest.mark.criterion("M10-7")
def test_frozen_parameter_change_stops_training(tokenizer_dir, sasrec_state_path, tmp_path, monkeypatch):
    """The epoch-end check of §14.6.4: a base LLaMA weight changed in place during epoch 0 stops the run with a
    TrainingError before the checkpoint is saved, so every saved ``frozen_checksum_at_save`` equals the start value
    that the smoke check ``frozen_sasrec`` compares it with (17.6.7 #2)."""
    original = train_module.train_window
    touched = []

    def train_window_then_touch(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        if not touched:
            with torch.no_grad():
                model.llm.model.layers[0].self_attn.q_proj.base.weight[0, 0] += 1e-3
            touched.append(result.step)
        return result

    monkeypatch.setattr(train_module, "train_window", train_window_then_touch)
    root = tmp_path / "runs"
    with pytest.raises(TrainingError, match="epoch 0: a frozen parameter changed"):
        run_command("train", root, "m10_frozen", *ONE_EPOCH, NO_BI, tokenizer_dir=tokenizer_dir)
    run_dir = only_child(root)
    assert touched == [0] and run.read_meta(run_dir)["status"] == "failed"
    assert list((run_dir / "checkpoints").iterdir()) == []


# ----------------------------------------------------------------------------------------------- S1: 2-epoch reference run
@pytest.fixture(scope="module")
def reference_run(tokenizer_dir, sasrec_state_path, tmp_path_factory):
    """smoke_cpu (S1 of §17.6.6: 2 epochs, dynamic) through ``run_command('train')`` with one CPU thread, the spies
    on evaluate_split, markers, M15 and build_model, and ``train.log_every_steps=3``."""
    log = CallLog()
    root = tmp_path_factory.mktemp("m10_reference")
    with pytest.MonkeyPatch.context() as mp, one_thread():
        install_spies(mp, log)
        run_dir = run_command("train", root, "m10_reference", *REFERENCE_SETS, tokenizer_dir=tokenizer_dir)
    return SimpleNamespace(run_dir=run_dir, log=log, sets=REFERENCE_SETS)


@pytest.mark.slow
@pytest.mark.criterion("M10-13")
def test_record_counts(reference_run, smoke_data):
    """S1 records: 78 micro rows, 20 step rows (4 x 9 + 3 per epoch), 2 epoch rows, hybrid 10 + 28 = 38 as planned,
    training gate calls 78 (evaluation excluded), columns = M12 definitions, run.log summary lines every 3 steps and
    at each epoch end."""
    import re

    from ilora import records
    from ilora.config import load_config
    from ilora.prompts import build_batch_plan

    run_dir = reference_run.run_dir
    folder = run_dir / "records" / "train"
    for name, fields in (("micro_batches.csv", records.MICRO_FIELDS), ("optimizer_steps.csv", records.step_fields(K)),
                         ("epochs.csv", records.EPOCH_FIELDS)):
        assert (folder / name).read_text(encoding="utf-8").splitlines()[0] == ",".join(fields)
    micro = records.read_rows(folder / "micro_batches.csv", records.MICRO_FIELDS)
    steps = records.read_rows(folder / "optimizer_steps.csv", records.step_fields(K))
    epochs = records.read_rows(folder / "epochs.csv", records.EPOCH_FIELDS)
    assert (len(micro), len(steps), len(epochs)) == (78, 20, 2)
    assert [r["tau"] for r in micro] == list(range(1, 79)) and [r["step"] for r in steps] == list(range(20))
    assert [r["n_micro"] for r in steps] == ([4] * 9 + [3]) * 2
    cfg, _ = load_config(run_dir / "config.yaml")
    for epoch in (0, 1):
        plan = build_batch_plan(smoke_data.train, epoch, cfg)
        rows = [r for r in micro if r["epoch"] == epoch]
        assert [r["hybrid"] for r in rows] == [e.hybrid for e in plan.entries]
        assert [r["curriculum_p"] for r in rows] == [e.tau / 78 for e in plan.entries]
        assert epochs[epoch]["plan_digest"] == plan.digest()
        assert epochs[epoch]["n_hybrid"] == plan.hybrid_count() == sum(r["n_hybrid"] for r in steps if r["epoch"] == epoch)
        assert epochs[epoch]["last_step"] == 10 * epoch + 9 and epochs[epoch]["tau_last"] == 39 * (epoch + 1)
    assert [e["n_hybrid"] for e in epochs] == [10, 28]
    assert all(r["gate_calls"] == r["n_micro"] and r["cluster_lookups"] == 0 for r in steps)
    assert sum(r["gate_calls"] for r in steps) == 78
    assert all(r["projector_has_grad"] == (r["n_hybrid"] > 0) for r in steps)
    assert all((r["grad_norm_projector"] is None) == (r["n_hybrid"] == 0) for r in steps)
    assert all(abs(sum(r[f"omega_mean_{k}"] for k in range(K)) - 1.0) <= 1e-6 for r in steps)
    assert all(r["act_max_module"] is not None and math.isfinite(r["act_max_abs"]) for r in micro)
    summary = read_json(folder / "run_summary.json")
    assert summary["status"] == "completed" and summary["gate_calls"] == 78
    assert (summary["micro_batches"], summary["optimizer_steps"], summary["epochs"]) == (78, 20, 2)
    assert summary["train_summary"]["trainable_params"] == {**TINY_TRAINABLE, "total": 25_984}
    for metrics in run_dir.glob("eval/*/*/metrics.json"):           # evaluation gate calls stay out of the sum
        assert read_json(metrics)["meta"]["gate_calls"] == 2
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    summaries = [int(m.group(1)) for m in re.finditer(r"train step (\d+)/20 epoch", log)]
    assert summaries == [3, 6, 9, 10, 12, 15, 18, 20]


@pytest.mark.slow
@pytest.mark.criterion("M10-10")
def test_test_after_last_complete(reference_run):
    """Order of evaluate_split calls and marker writes in the 2-epoch run: each validation after its epoch's
    STATE_SAVED and before its val_metrics.json/COMPLETE; test only after the last epoch's COMPLETE and selection.json,
    for best_val and last ascending (§14.6.7, D11)."""
    events = [e for e in reference_run.log.events if e[0] in ("evaluate_split", "marker")]
    key = [(e[0], e[1], e[2]) for e in events]
    expected_prefix = []
    for epoch in (0, 1):
        expected_prefix += [("marker", "STATE_SAVED", f"epoch{epoch}"), ("evaluate_split", "val", f"epoch{epoch}"),
                            ("marker", "val_metrics.json", f"epoch{epoch}"), ("marker", "COMPLETE", f"epoch{epoch}")]
    expected_prefix.append(("marker", "selection.json", "eval"))
    assert key[:len(expected_prefix)] == expected_prefix
    selection = read_json(reference_run.run_dir / "eval" / "selection.json")
    order = train_module.epochs_to_test(selection["best_val_epoch"], selection["last_epoch"])
    assert key[len(expected_prefix):] == [("evaluate_split", "test", f"epoch{e}") for e in order]
    last_complete = key.index(("marker", "COMPLETE", "epoch1"))
    assert all(i > last_complete for i, k in enumerate(key) if k[:2] == ("evaluate_split", "test"))


@pytest.mark.slow
@pytest.mark.criterion("M10-8")
def test_resume_equals_uninterrupted(reference_run, tokenizer_dir, sasrec_state_path, tmp_path, monkeypatch,
                                     torch_threads1):
    """2 epochs at once = epoch 0, a failure when epoch 1's plan is built, ``--resume`` that fails at epoch 1's
    validation (STATE_SAVED only), and ``--resume`` again from the validation: trainable.pt bit for bit, the three
    CSVs row for row (times and memory aside), the evaluation records; rows planted beyond the boundary move to
    ``*.incomplete-<UTC>.csv``."""
    from ilora import evaluate, prompts

    root = tmp_path / "runs"
    original_plan = prompts.build_batch_plan

    def failing_plan(samples, epoch, cfg):
        if epoch == 1:
            raise RuntimeError("injected failure at the start of epoch 1")
        return original_plan(samples, epoch, cfg)

    monkeypatch.setattr(prompts, "build_batch_plan", failing_plan)
    with pytest.raises(RuntimeError, match="injected failure at the start of epoch 1"):
        run_command("train", root, "m10_resume", *reference_run.sets, tokenizer_dir=tokenizer_dir)
    monkeypatch.setattr(prompts, "build_batch_plan", original_plan)
    run_dir = only_child(root)
    assert run.read_meta(run_dir)["status"] == "failed"
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["epoch0"]
    assert (run_dir / "checkpoints" / "epoch0" / "COMPLETE").is_file()
    reference = reference_run.run_dir
    planted = {"micro_batches.csv": (lambda r: r["tau"] in ("40", "41")), "optimizer_steps.csv": (lambda r: r["step"] == "10"),
               "epochs.csv": (lambda r: r["epoch"] == "1")}
    for name, pick in planted.items():                              # rows beyond the epoch-0 boundary
        lines = (reference / "records" / "train" / name).read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        extra = [line for line in lines[1:] if pick(dict(zip(header, next(csv.reader([line])))))]
        with open(run_dir / "records" / "train" / name, "a", encoding="utf-8", newline="") as f:
            f.write("".join(line + "\n" for line in extra))
    original_eval = evaluate.evaluate_split

    def failing_eval(model, bundle, data, split, cfg, run_dir_, **kwargs):
        if split == "val" and kwargs.get("ckpt_tag") == "epoch1":
            raise RuntimeError("injected failure in the validation of epoch 1")
        return original_eval(model, bundle, data, split, cfg, run_dir_, **kwargs)

    monkeypatch.setattr(evaluate, "evaluate_split", failing_eval)
    with pytest.raises(RuntimeError, match="injected failure in the validation of epoch 1"):
        resume_run(run_dir, tokenizer_dir)
    monkeypatch.setattr(evaluate, "evaluate_split", original_eval)
    epoch1 = run_dir / "checkpoints" / "epoch1"
    assert (epoch1 / "STATE_SAVED").is_file() and not (epoch1 / "COMPLETE").is_file()
    moved = {name: sorted((run_dir / "records" / "train").glob(f"{Path(name).stem}.incomplete-*.csv"))
             for name in planted}
    assert [len(read_csv(paths[0])) for paths in moved.values()] == [2, 1, 1]
    assert read_csv(moved["micro_batches.csv"][0])[0]["tau"] == "40"
    resume_run(run_dir, tokenizer_dir)                               # validation of epoch 1 onwards
    meta = run.read_meta(run_dir)
    assert meta["status"] == "completed" and len(meta["resumes"]) == 2
    assert [s["status"] for s in meta["train"]["segments"]] == ["failed", "failed", "completed"]
    for epoch in (0, 1):
        assert run.content_sha256(run_dir / "checkpoints" / f"epoch{epoch}" / "trainable.pt") == \
            run.content_sha256(reference / "checkpoints" / f"epoch{epoch}" / "trainable.pt")
    for name in planted:
        assert drop_time(read_csv(run_dir / "records" / "train" / name)) == \
            drop_time(read_csv(reference / "records" / "train" / name)), name
    for records_file in sorted(reference.glob("eval/*/*/records.jsonl")):
        rel = records_file.relative_to(reference)
        assert run.content_sha256(run_dir / rel) == run.content_sha256(records_file), rel
    assert read_json(run_dir / "eval" / "selection.json") == read_json(reference / "eval" / "selection.json")
    summary = read_json(run_dir / "records" / "train" / "run_summary.json")
    assert summary["status"] == "completed" and summary["train_summary"]["resumes"] == 2


@pytest.mark.criterion("M10-8")
def test_segment_rules():
    """M10 ``segment_problems`` on meta.json ``train.segments``: a segment left ``running`` by a killed process is a
    problem until a resume marks it ``interrupted`` (``ended_utc`` null: never measured, ``detected_utc`` set);
    completed and failed segments need ``ended_utc``; interrupted segments need a resume of a ``running`` folder; the
    last segment is completed."""
    t = "2026-11-06T00:00:00+00:00"
    killed = {"started_utc": t, "resume": False, "ended_utc": None, "status": "running"}
    marked = {**killed, "status": "interrupted", "detected_utc": t}
    done = {"started_utc": t, "resume": True, "ended_utc": t, "status": "completed"}
    failed = {"started_utc": t, "resume": False, "ended_utc": t, "status": "failed"}
    after_kill, after_failure = [{"previous_status": "running"}], [{"previous_status": "failed"}]
    problems = train_module.segment_problems
    assert problems([{**done, "resume": False}], []) == []
    assert problems([failed, done], after_failure) == []
    assert problems([marked, done], after_kill) == []
    assert problems([killed, done], after_kill) == ["segment 0 is running: never closed and not marked interrupted by "
                                                    "a resume"]
    assert problems([marked, done], after_failure) == ["1 interrupted segments but 0 resumes of a running folder"]
    assert problems([{**marked, "detected_utc": None}, done], after_kill)
    assert problems([{**failed, "ended_utc": None}, done], after_failure)
    assert problems([marked], after_kill) and problems([], [])


@pytest.mark.criterion("M10-8")
def test_stale_val_metrics_kept_with_counter(tmp_path):
    """A resumed STATE_SAVED-only epoch keeps the val_metrics.json it left behind as
    ``val_metrics.incomplete-<UTC>.json`` and adds ``-<n>`` when that name is taken (two resumes within one second),
    so an earlier kept file is never replaced (Linux ``rename`` would overwrite it). Nothing to keep gives None."""
    ckpt = tmp_path / "epoch1"
    ckpt.mkdir()
    keep, stamp = train_module._keep_stale_val_metrics, "20261106T000000Z"
    assert keep(ckpt, stamp) is None and not any(ckpt.iterdir())
    names = []
    for text in ("first", "second", "third"):
        (ckpt / train_module.VAL_METRICS_FILE).write_text(text, encoding="utf-8")
        names.append(keep(ckpt, stamp).name)
    assert names == [f"val_metrics.incomplete-{stamp}.json", f"val_metrics.incomplete-{stamp}-1.json",
                     f"val_metrics.incomplete-{stamp}-2.json"]
    assert [(ckpt / name).read_text(encoding="utf-8") for name in names] == ["first", "second", "third"]
    assert not (ckpt / train_module.VAL_METRICS_FILE).exists()


def wait_for_lock_release(run_dir: Path, timeout: float = 60.0) -> None:
    """The OS frees the run.lock of an ended process when its handles close (on Windows not always at once)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with run.hold_run_lock(run_dir, "test probe"):
                return
        except run.RunLockError:
            assert time.monotonic() < deadline, "run.lock still held after the process ended"
            time.sleep(0.1)


@pytest.mark.slow
@pytest.mark.criterion("M10-8")
def test_resume_after_a_killed_process(tokenizer_dir, sasrec_state_path, tmp_path):
    """A training process that ends without its except and finally blocks (``os._exit`` in epoch 1's validation, like
    SIGKILL or the OOM killer) leaves meta.json and its segment ``running`` (§4.8 #6) and frees run.lock. ``--resume``
    marks that segment ``interrupted`` (``ended_utc`` null, ``detected_utc`` set), finishes the run from the
    validation, and the segments pass M10 ``segment_problems`` (the E2 check of tests/vessl/test_run_outputs.py)."""
    _, texts = smoke_cfg(NO_BI)
    root = tmp_path / "runs"
    code = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        sys.path.insert(0, {str(REBUILD_DIR)!r})
        from ilora import commands, evaluate
        from ilora.config import load_config

        original = evaluate.evaluate_split

        def exit_in_epoch1_validation(model, bundle, data, split, cfg, run_dir, **kwargs):
            if split == "val" and kwargs.get("ckpt_tag") == "epoch1":
                os._exit(9)                          # no except, no finally: like SIGKILL
            return original(model, bundle, data, split, cfg, run_dir, **kwargs)

        evaluate.evaluate_split = exit_in_epoch1_validation
        texts = {texts!r}
        cfg, _ = load_config(Path({str(CONFIG_DIR / "smoke_cpu.yaml")!r}), texts)
        commands.run_command("train", cfg, texts, profile="m10_killed", root=Path({str(root)!r}), argv=["child"],
                             tokenizer_dir=Path({str(tokenizer_dir)!r}))
    """)
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 9, proc.stderr[-4000:]
    run_dir = only_child(root)
    meta = run.read_meta(run_dir)
    assert meta["status"] == "running" and meta["ended_utc"] is None
    assert [(s["status"], s["ended_utc"]) for s in meta["train"]["segments"]] == [("running", None)]
    assert (run_dir / "checkpoints" / "epoch1" / "STATE_SAVED").is_file()
    wait_for_lock_release(run_dir)
    resume_run(run_dir, tokenizer_dir)
    meta = run.read_meta(run_dir)
    assert meta["status"] == "completed"
    (resume,) = meta["resumes"]
    assert resume["previous_status"] == "running" and resume["start_epoch"] == 1
    segments = meta["train"]["segments"]
    assert [(s["status"], s["resume"]) for s in segments] == [("interrupted", False), ("completed", True)]
    assert segments[0]["ended_utc"] is None and segments[0]["detected_utc"] and segments[1]["ended_utc"]
    assert train_module.segment_problems(segments, meta["resumes"]) == []
    assert (run_dir / "checkpoints" / "epoch1" / "COMPLETE").is_file()


# ----------------------------------------------------------------------------------------------- S5: cluster_hard
def assignments_of(cluster_dir: Path) -> dict[str, int]:
    out = {}
    for name in ("train", "validation", "test"):
        for row in read_csv(cluster_dir / f"{name}_assignments.csv"):
            out[row["sample_id"]] = int(row["cluster_id"])
    return out


@pytest.fixture(scope="module")
def hard_run(tokenizer_dir, sasrec_state_path, smoke_cluster_dir, tmp_path_factory):
    """S5 of §17.6.6: cluster_hard for one epoch with a copy of the M15 smoke artifacts, spies on the M15 functions
    and build_model."""
    import shutil

    root = tmp_path_factory.mktemp("m10_hard")
    artifacts = Path(shutil.copytree(smoke_cluster_dir, root / "cluster_run" / "cluster"))
    sets = ("router.routing_mode=cluster_hard", f"cluster.artifact_dir={json.dumps(artifacts.as_posix())}", *ONE_EPOCH)
    log = CallLog()
    with pytest.MonkeyPatch.context() as mp, one_thread():
        install_spies(mp, log)
        run_dir = run_command("train", root / "runs", "m10_hard", *sets, tokenizer_dir=tokenizer_dir)
    return SimpleNamespace(run_dir=run_dir, log=log, sets=sets, artifacts=artifacts)


@pytest.mark.slow
@pytest.mark.criterion("M10-14", "M7-7")
def test_cluster_hard_training(hard_run):
    """S5: M15 is called in the order artifacts -> mapping -> ClusterHardGate -> build_model(hard_gate) -> live z
    check; no gate group (256 parameters fewer, 35 checkpoint keys); training gate calls 0 and cluster lookups 156
    (39 x 4); the live check passed (meta.json, run.log); evaluation gates are the one-hot of mapping[cluster]."""
    run_dir = hard_run.run_dir
    order = [e for e in hard_run.log.events if e[0] in ("cluster", "build_model")]
    assert order == [("cluster", "load_cluster_artifacts"), ("cluster", "parse_expert_mapping"),
                     ("cluster", "ClusterHardGate"), ("build_model", "ClusterHardGate"),
                     ("cluster", "verify_live_encoder")]
    summary = read_json(run_dir / "records" / "train" / "run_summary.json")
    train_summary = summary["train_summary"]
    assert train_summary["trainable_params"] == {"projector": 8_320, "lora": 17_408, "total": 25_984 - 256}
    assert train_summary["routing_mode"] == "cluster_hard" and train_summary["unused_experts"] == []
    assert (summary["gate_calls"], summary["cluster_lookups"]) == (0, 156)
    steps = read_csv(run_dir / "records" / "train" / "optimizer_steps.csv")
    assert all(r["lr_gate"] == "" and r["grad_norm_gate"] == "" for r in steps)
    assert [int(r["cluster_lookups"]) for r in steps] == [4 * int(r["n_micro"]) for r in steps]
    assert sum(int(r["gate_calls"]) for r in steps) == 0
    live = run.read_meta(run_dir)["cluster_live_check"]
    assert live["passed"] is True and live["n"] == 64 and live["max_abs_diff"] <= 1e-4
    assert "cluster live check passed: n=64" in (run_dir / "run.log").read_text(encoding="utf-8")
    state = train_module.read_trainable_state(run_dir / "checkpoints" / "epoch0")
    assert len(state) == 35 and not any(k.startswith("omega_source.") for k in state)
    assignment = assignments_of(hard_run.artifacts)
    for folder in sorted(run_dir.glob("eval/*/*/metrics.json")):
        assert read_json(folder)["meta"]["gate_calls"] == 0
        for line in read_jsonl(folder.parent / "records.jsonl"):
            expert = assignment[line["sample_id"]]                      # identity mapping 0:0,1:1,2:2,3:3
            assert [line[f"gate_{k}"] for k in range(K)] == [1.0 if k == expert else 0.0 for k in range(K)]
    provenance = torch.load(run_dir / "checkpoints" / "epoch0" / "train_state.pt", map_location="cpu",
                            weights_only=True)["cluster"]
    assert provenance["expert_mapping"] == "0:0,1:1,2:2,3:3" and provenance["fit_split"] == "train"
    assert provenance["manifest_sha256"] == run.content_sha256(hard_run.artifacts / "manifest.json")


@pytest.mark.slow
@pytest.mark.criterion("M10-14")
def test_cluster_artifact_tamper_fails_before_build(smoke_cluster_dir, tokenizer_dir, sasrec_state_path, tmp_path,
                                                    monkeypatch):
    """One changed byte of a listed artifact file stops cluster_hard training in load_cluster_artifacts (ClusterError
    naming the file) before build_model is called; the run is marked failed."""
    import shutil

    from ilora.cluster import ClusterError

    artifacts = Path(shutil.copytree(smoke_cluster_dir, tmp_path / "cluster_run" / "cluster"))
    target = artifacts / "train_assignments.csv"
    data = bytearray(target.read_bytes())
    data[-2] ^= 0x01
    target.write_bytes(bytes(data))
    log = CallLog()
    install_spies(monkeypatch, log)
    sets = ("router.routing_mode=cluster_hard", f"cluster.artifact_dir={json.dumps(artifacts.as_posix())}", *ONE_EPOCH)
    with pytest.raises(ClusterError, match="train_assignments.csv"):
        run_command("train", tmp_path / "runs", "m10_tamper", *sets, tokenizer_dir=tokenizer_dir)
    assert [e for e in log.events if e[0] in ("cluster", "build_model")] == [("cluster", "load_cluster_artifacts")]
    assert run.read_meta(only_child(tmp_path / "runs"))["status"] == "failed"


# ----------------------------------------------------------------------------------------------- S7: K = 1
@pytest.fixture(scope="module")
def k1_run(tokenizer_dir, sasrec_state_path, tmp_path_factory):
    """S7 of §17.6.6: K = 1 (routing none, the LLaRA-equivalent LoRA) for one epoch."""
    sets = ("lora.num_experts=1", "router.routing_mode=none", *ONE_EPOCH)
    run_dir = run_command("train", tmp_path_factory.mktemp("m10_k1"), "m10_k1", *sets, tokenizer_dir=tokenizer_dir)
    return SimpleNamespace(run_dir=run_dir, sets=sets)


def source_set(run_dir: Path, key: str) -> str:
    return f"{key}={json.dumps(run_dir.as_posix())}"


def flip_generated_frozen_bit(ckpt: Path) -> None:
    """One bit of ``generated_frozen.embed_rows`` with consistent sha256 in both markers (the bit check must fail)."""
    state = torch.load(ckpt / "trainable.pt", map_location="cpu", weights_only=True)
    rows = state["generated_frozen.embed_rows"].clone()
    rows.view(torch.int32)[0, 0] ^= 1
    state["generated_frozen.embed_rows"] = rows
    data = torch_bytes(state)
    (ckpt / "trainable.pt").write_bytes(data)
    for marker in ("STATE_SAVED", "COMPLETE"):
        doc = read_json(ckpt / marker)
        doc["files"]["trainable.pt"] = hashlib.sha256(data).hexdigest()
        (ckpt / marker).write_text(json.dumps(doc), encoding="utf-8")


EVAL_CASES = ["identical", "frozen_bit", "text_only", "cluster_hard"]


@pytest.mark.slow
@pytest.mark.criterion("M11-16")
@pytest.mark.parametrize("case", EVAL_CASES)
def test_eval_command_on_trained_run(case, request, tokenizer_dir, tmp_path, monkeypatch, torch_threads1):
    """``main.py eval`` on checkpoints written by M10 (no stand-ins): the best_val/last re-evaluation reproduces the
    training-time records (identical, S2) without any M15 call; one generated_frozen bit makes the real
    load_model_state fail (run failed, no metrics.json); text_only is not_comparable with hybrid false (S6); a
    cluster_hard run goes through the M15 order and gives one-hot gates with no gate call. The source runs and these
    evaluations use one CPU thread: the thread count is part of the CPU arithmetic (15.6.10)."""
    import shutil

    source = request.getfixturevalue("hard_run" if case == "cluster_hard" else "reference_run")
    src = source.run_dir
    selection = read_json(src / "eval" / "selection.json")
    best, last = selection["best_val_epoch"], selection["last_epoch"]
    root = tmp_path / "runs"
    log = CallLog()
    install_spies(monkeypatch, log)
    if case == "frozen_bit":
        src = Path(shutil.copytree(src, tmp_path / "source_copy"))
        flip_generated_frozen_bit(src / "checkpoints" / f"epoch{last}")
        with pytest.raises(TrainingError, match="generated_frozen"):
            run_command("eval", root, "m10_eval", *source.sets, source_set(src, "eval.source_run"), NO_BI,
                        "eval.checkpoints=[last]", "eval.splits=[val]", tokenizer_dir=tokenizer_dir)
        run_dir = only_child(root)
        assert run.read_meta(run_dir)["status"] == "failed"
        assert not list(run_dir.glob("eval/*/*/metrics.json"))
        return
    extra = ("prompt.eval_prompt=text_only", "eval.checkpoints=[last]", "eval.splits=[val]") if case == "text_only" else ()
    run_dir = run_command("eval", root, "m10_eval", *source.sets, source_set(src, "eval.source_run"), NO_BI, *extra,
                          tokenizer_dir=tokenizer_dir)
    summary = read_json(run_dir / "eval" / "summary.json")
    if case == "text_only":
        (item,) = summary["results"].values()
        assert (item["ckpt_tag"], item["split"]) == (f"epoch{last}", "val")
        assert item["reproduction"]["status"] == "not_comparable"
        assert item["reproduction"]["differing_keys"] == ["prompt.eval_prompt"]
        assert all(line["hybrid"] is False for line in read_jsonl(run_dir / "eval" / f"epoch{last}" / "val" /
                                                                    "records.jsonl"))
        return
    epochs = train_module.epochs_to_test(best, last)
    assert list(summary["results"]) == [f"epoch{e}/{s}" for e in epochs for s in ("val", "test")]
    assert [item["reproduction"]["status"] for item in summary["results"].values()] == ["identical"] * 2 * len(epochs)
    for key in summary["results"]:
        assert run.content_sha256(run_dir / "eval" / key / "records.jsonl") == \
            run.content_sha256(src / "eval" / key / "records.jsonl")
    for item in summary["checkpoints"]:
        assert item["trainable_sha256"] == run.content_sha256(src / "checkpoints" / item["ckpt_tag"] / "trainable.pt")
    order = [e for e in log.events if e[0] in ("cluster", "build_model")]
    if case == "identical":
        assert order == [("build_model", "NoneType")]                    # dynamic: no M15 call [SPEC §22]
        return
    assert order == [("cluster", "load_cluster_artifacts"), ("cluster", "parse_expert_mapping"),
                     ("cluster", "ClusterHardGate"), ("build_model", "ClusterHardGate"),
                     ("cluster", "verify_live_encoder")]
    assert summary["cluster_live_check"]["passed"] is True
    assignment = assignments_of(source.artifacts)
    for key in summary["results"]:
        assert read_json(run_dir / "eval" / key / "metrics.json")["meta"]["gate_calls"] == 0
        for line in read_jsonl(run_dir / "eval" / key / "records.jsonl"):
            expert = assignment[line["sample_id"]]
            assert [line[f"gate_{k}"] for k in range(K)] == [1.0 if k == expert else 0.0 for k in range(K)]


EXPORT_CASES = ["dynamic", "k1", "cluster_hard"]


@pytest.mark.slow
@pytest.mark.criterion("M12-7")
@pytest.mark.parametrize("case", EXPORT_CASES)
def test_export_on_trained_run(case, request, tokenizer_dir, tmp_path):
    """``main.py export`` of real M10 runs: crosscheck.json passed (max |d omega| <= 1e-5, |d z| recorded), NPZ
    156/10/10 rows, gates_{validation,test}.csv, analysis folders with figures for K >= 2 (none for K = 1),
    manifest sha256; export_router loads only the omega_source key of read_trainable_state (generated_frozen.*
    present but unused); K = 1 (S7) and cluster_hard (S5, one-hot = mapping[cluster]) through the same command."""
    from ilora import records
    from ilora.config import load_config

    source = request.getfixturevalue({"dynamic": "reference_run", "k1": "k1_run", "cluster_hard": "hard_run"}[case])
    src = source.run_dir
    run_dir = run_command("export", tmp_path / "runs", "m10_export", *source.sets,
                          source_set(src, "records.source_run"), tokenizer_dir=tokenizer_dir)
    selection = read_json(src / "eval" / "selection.json")
    epochs = train_module.epochs_to_test(selection["best_val_epoch"], selection["last_epoch"])
    assert sorted(p.name for p in (run_dir / "records" / "export").iterdir()) == [f"epoch{e}" for e in epochs]
    k = 1 if case == "k1" else K
    for epoch in epochs:
        out = run_dir / "records" / "export" / f"epoch{epoch}"
        cross = read_json(out / "crosscheck.json")
        for name in ("validation", "test"):
            entry = cross["splits"][name]
            assert entry["passed"] and entry["max_abs_diff_omega"] <= 1e-5 and entry["argmax_mismatches"] == 0
            assert isinstance(entry["max_abs_diff_z"], float) and math.isfinite(entry["max_abs_diff_z"])   # recorded
        tables = {name: records.load_npz(out / f"representations_{name}.npz") for name in ("train", "validation", "test")}
        assert [len(t) for t in tables.values()] == [156, 10, 10]
        assert all(t.omega.shape[1] == k for t in tables.values())
        assert (out / "gates_validation.csv").is_file() and (out / "gates_test.csv").is_file()
        assert not (out / "gates_train.csv").exists()
        manifest = read_json(out / "manifest.json")
        for name, sha in manifest["files"].items():
            assert run.content_sha256(out / name) == sha, name
        for rel, sha in manifest["analysis_files"].items():
            assert run.content_sha256(run_dir / rel) == sha, rel
        assert manifest["checkpoint"]["trainable_sha256"] == run.content_sha256(
            src / "checkpoints" / f"epoch{epoch}" / "trainable.pt")
        analysis = run_dir / "records" / "analysis" / f"epoch{epoch}"
        if k == 1:
            assert not analysis.exists() and manifest["analysis"] == "not run: gate analysis requires K >= 2"
            assert all(bool((t.omega == 1.0).all()) for t in tables.values())
        else:
            for split in ("test", "val", "train"):
                assert (analysis / split / "gate_summary.json").is_file()
                assert (analysis / split / "figures" / "gate_heatmap.png").is_file()
                assert (analysis / split / "figures" / "gate_ten_sequences.png").is_file() == (split == "test")
        if case == "cluster_hard":
            assignment = assignments_of(source.artifacts)
            for table in tables.values():
                expected = np.array([assignment[sid] for sid in table.sample_ids])
                assert np.array_equal(table.argmax, expected) and bool(((table.omega == 0) | (table.omega == 1)).all())
            assert manifest["cluster_live_check"]["passed"] is True
    cfg, _ = load_config(src / "config.yaml")
    state = train_module.read_trainable_state(src / "checkpoints" / f"epoch{epochs[-1]}")
    assert sum(key.startswith("generated_frozen.") for key in state) == 3
    if case == "dynamic":
        router = records.export_router(cfg, state, z_dim=64)
        assert list(router.state_dict()) == ["gate.proj.weight"]
        assert same_bits(router.state_dict()["gate.proj.weight"], state["omega_source.gate.proj.weight"])
        return
    assert not any(key.startswith("omega_source.") for key in state) and len(state) == 35
    if case == "k1":                                                 # S7 training records
        assert dict(records.export_router(cfg, state, z_dim=64).state_dict()) == {}
        steps = read_csv(src / "records" / "train" / "optimizer_steps.csv")
        assert all(r["omega_mean_0"] == "1.0" and r["omega_entropy_mean"] == "" for r in steps)
        assert read_json(src / "records" / "train" / "run_summary.json")["train_summary"]["trainable_params"] == \
            {"projector": 8_320, "lora": 17_408, "total": 25_728}
        for metrics in src.glob("eval/*/*/metrics.json"):
            assert all(line["gate_0"] == 1.0 for line in read_jsonl(metrics.parent / "records.jsonl"))
