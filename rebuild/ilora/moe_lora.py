"""M8 MoE-LoRA 층 (DESIGN.md §12; D7, D8, D9).

``MoELoRALinear`` wraps one frozen ``nn.Linear`` of LLaMA and computes

    h' = W x + sum_k omega_k * s * B_k A_k x        (식 12 with the D8 scaling s)

as ``W x + s * B((A dropout(x)) * expand(omega))``. The K experts are the row
blocks of ``A [r, d_in]`` and the column blocks of ``B [d_out, r]`` with rank
``r* = r / K`` (식 9), so the layer never builds or saves the ``[B, L, K, d_out]``
expert outputs of the released code (§12.5, §12.7). ``GateContext`` is the one
object per model that holds ``omega`` while a forward or a generation runs
(§12.6.4); every wrapped layer reads it and checks batch size and device.
``attach_moe_lora`` replaces the target layers of a CPU LLaMA and initialises the
LoRA tensors in a fixed canonical order under the ``("init", "lora")`` seed scope
(§12.6.3).
"""
from __future__ import annotations

import contextlib
import itertools
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

import torch
from torch import nn

from . import seeding
from .router import ROW_SUM_ATOL

if TYPE_CHECKING:   # type names only; importing transformers here is not needed at run time
    from transformers import LlamaForCausalLM

    from .sections import LoraConfig

# [코드] U model/model_interface.py:364-371, L model/model_interface.py:215-220 (D7). Initialisation order inside a
# layer (§12.6.3); sections.LORA_TARGETS is the same tuple (the schema does not import torch code).
CANONICAL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SCALING_RULES = ("expert_rank", "total_rank")                       # lora.scaling_rule (§12.6.7, D8)
# Parent module of each target in a 4.28.0 LlamaDecoderLayer (T4.28 models/llama/modeling_llama.py:255-256): attention
# projections under self_attn (:176-179), MLP projections under mlp (:151-153, registered gate, down, up).
_TARGET_PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
                  "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
_WRAPPED_NAME_RE = re.compile(r"(?:.*\.)?layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)")


class GateContextError(RuntimeError):
    """omega missing, invalid, nested, or not matching the layer input (§12.6.2, §12.6.4)."""


@dataclass
class GateUse:
    """Record of one ``GateContext.use`` block (§12.6.1)."""
    batch_size: int
    calls: int = 0          # wrapped-layer calls that read omega inside the block


class GateContext:
    """Holder of omega ``[B, K] float32`` shared by every MoE-LoRA layer of one model (§3.1, §12.6.4).

    omega exists only inside ``with gate_ctx.use(omega) as use:``; it is checked once there (shape, dtype,
    finite, >= 0, row sums within ``ROW_SUM_ATOL``) and cleared in ``finally`` even when the block raises.
    The computation graph of omega is kept, so the dynamic gate receives the gradients of all layers.
    """

    def __init__(self, num_experts: int) -> None:
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError(f"num_experts must be a positive int, got {num_experts!r}")
        self._num_experts = num_experts
        self._omega: Optional[torch.Tensor] = None
        self._use: Optional[GateUse] = None

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def is_set(self) -> bool:
        return self._omega is not None

    @contextlib.contextmanager
    def use(self, omega: torch.Tensor) -> Iterator[GateUse]:
        """Check omega, hold it for the block and clear it afterwards. A rejected omega is never stored."""
        self._check(omega)
        use = GateUse(batch_size=int(omega.shape[0]))
        self._omega, self._use = omega, use
        try:
            yield use
        finally:
            self._omega, self._use = None, None

    def get(self) -> torch.Tensor:
        """omega for a layer call (counts the call in ``GateUse.calls``); ``GateContextError`` outside ``use``."""
        if self._omega is None or self._use is None:
            raise GateContextError("GateContext is empty: MoE-LoRA layers run only inside GateContext.use(omega) "
                                   "(M9 sets omega for every forward and generation)")
        self._use.calls += 1
        return self._omega

    def _check(self, omega: torch.Tensor) -> None:
        if self._omega is not None:
            raise GateContextError("GateContext.use is already active (nested use is not allowed)")
        if not isinstance(omega, torch.Tensor):
            raise GateContextError(f"omega must be a torch.Tensor, got {type(omega).__name__}")
        if omega.dim() != 2 or omega.shape[0] < 1 or omega.shape[1] != self._num_experts:
            raise GateContextError(f"omega must be [B, {self._num_experts}] with B >= 1, got {tuple(omega.shape)}")
        if omega.dtype != torch.float32:
            raise GateContextError(f"omega must be float32, got {omega.dtype}")
        with torch.no_grad():
            values = omega.detach()
            ok = (torch.isfinite(values).all(dim=1) & (values >= 0).all(dim=1)
                  & ((values.sum(dim=1) - 1.0).abs() <= ROW_SUM_ATOL))
            if not bool(ok.all()):       # one synchronisation per use(), not per layer call (§12.6.4)
                row = int((~ok).nonzero()[0, 0])
                raise GateContextError(f"omega row {row} is not a probability vector (finite, >= 0, "
                                       f"|row sum - 1| <= {ROW_SUM_ATOL}): {values[row].tolist()}")


def lora_scaling(alpha: int, r: int, num_experts: int, rule: str) -> float:
    """Scaling ``s`` of one expert (D8, §20.1).

    ``expert_rank``: alpha / r* with r* = r / K (16 for paper_spec), ``[코드] U model/peft/tuners/moelora.py:618``.
    ``total_rank``: alpha / r (4, ``variant_total_rank_scaling``: 식 12 substituted into 식 3).
    Both rules give alpha / r for K = 1 (standard LoRA, LLaRA).
    """
    for name, value in (("r", r), ("num_experts", num_experts), ("alpha", alpha)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"lora {name} must be a positive int, got {value!r}")
    if r % num_experts != 0:
        raise ValueError(f"r must be divisible by num_experts (식 9 r* = r/K), got r={r}, num_experts={num_experts}")
    if rule == "expert_rank":
        return alpha / (r // num_experts)
    if rule == "total_rank":
        return alpha / r
    raise ValueError(f"scaling_rule must be one of {SCALING_RULES}, got {rule!r}")


def _autocast_enabled(device: torch.device) -> bool:
    # torch 2.0.0: CUDA autocast state is torch.is_autocast_enabled(), CPU is torch.is_autocast_cpu_enabled().
    if device.type == "cuda":
        return torch.is_autocast_enabled()
    if device.type == "cpu":
        return torch.is_autocast_cpu_enabled()
    return False


class MoELoRALinear(nn.Module):
    """One frozen LLaMA Linear plus K LoRA experts mixed by omega (§12.6.1, §12.6.2).

    ``lora_A.weight [r, d_in]`` and ``lora_B.weight [d_out, r]`` are float32 masters on CPU
    (``[코드] U model/peft/tuners/moelora.py:616-617``); expert k is ``A[k r* : (k+1) r*]`` and
    ``B[:, k r* : (k+1) r*]`` (``:750-753``, ``:774-776``). Building the layer uses no random numbers: the
    tensors are made with ``skip_init`` and filled with zeros; ``reset_lora_parameters`` sets the actual
    initial values (``attach_moe_lora`` calls it inside the seed scope).
    """

    base: nn.Linear
    lora_A: nn.Linear
    lora_B: nn.Linear
    lora_dropout: nn.Module
    r: int
    num_experts: int
    r_star: int
    scaling: float
    name: str

    def __init__(self, base: nn.Linear, *, r: int, num_experts: int, alpha: int, dropout: float,
                 scaling_rule: str, gate_ctx: GateContext, name: str) -> None:
        super().__init__()
        if isinstance(base, MoELoRALinear):
            raise ValueError(f"{name}: base is already a MoELoRALinear")
        if type(base) is not nn.Linear:
            raise ValueError(f"{name}: base must be exactly torch.nn.Linear (fan_in_fan_out layers are not "
                             f"supported), got {type(base).__name__}")
        scaling = lora_scaling(alpha, r, num_experts, scaling_rule)
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0):
            raise ValueError(f"{name}: dropout must be a number in [0, 1), got {dropout!r}")
        if not isinstance(gate_ctx, GateContext):
            raise ValueError(f"{name}: gate_ctx must be a GateContext, got {type(gate_ctx).__name__}")
        if gate_ctx.num_experts != num_experts:
            raise ValueError(f"{name}: gate_ctx has {gate_ctx.num_experts} experts, the layer {num_experts}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty str, got {name!r}")

        self.base = base
        for p in self.base.parameters():          # the base weight stays frozen (식 3 "W remains frozen", D9)
            p.requires_grad_(False)
        self.r = r
        self.num_experts = num_experts
        self.r_star = r // num_experts
        self.scaling = scaling
        self.name = name
        self.gate_ctx = gate_ctx                  # shared plain object, not a submodule
        # [코드] U model/peft/tuners/moelora.py:608-611: one dropout on x shared by all experts, Identity when p = 0.
        self.lora_dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        # [코드] U :616-617 (bias=False). skip_init: no random numbers are drawn while building (§12.6.3).
        self.lora_A = torch.nn.utils.skip_init(nn.Linear, base.in_features, r, bias=False,
                                               device="cpu", dtype=torch.float32)
        self.lora_B = torch.nn.utils.skip_init(nn.Linear, r, base.out_features, bias=False,
                                               device="cpu", dtype=torch.float32)
        with torch.no_grad():                     # deterministic content until reset_lora_parameters (no RNG use)
            self.lora_A.weight.zero_()
            self.lora_B.weight.zero_()

    def extra_repr(self) -> str:
        return (f"name={self.name}, r={self.r}, num_experts={self.num_experts}, r_star={self.r_star}, "
                f"scaling={self.scaling}")

    def reset_lora_parameters(self) -> None:
        """A ~ kaiming_uniform(a=sqrt(5)) = U(-1/sqrt(d_in), 1/sqrt(d_in)), B = 0, from the global CPU RNG.

        ``[코드] U model/peft/tuners/moelora.py:670-674`` (same as pip peft). B = 0 makes the first output equal
        to the base output and the first gradients of omega and A exactly zero (§12.6.3).
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def expert_slices(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(A_k [r*, d_in], B_k [d_out, r*])``: views of the master weights (autograd-tracked; detach to analyse)."""
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < self.num_experts:
            raise IndexError(f"{self.name}: expert index must be in [0, {self.num_experts}), got {k!r}")
        lo, hi = k * self.r_star, (k + 1) * self.r_star
        return self.lora_A.weight[lo:hi], self.lora_B.weight[:, lo:hi]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = self.gate_ctx.get()                                            # 1. GateContextError when empty
        if x.dim() < 2 or x.shape[0] != omega.shape[0] or omega.device != x.device:   # 2.
            raise GateContextError(f"{self.name}: input x {tuple(x.shape)} on {x.device} does not match omega "
                                   f"{tuple(omega.shape)} on {omega.device} (need x.dim() >= 2, the same batch "
                                   "size and the same device)")
        lora_dtype = self.lora_A.weight.dtype
        if x.dtype != lora_dtype and not _autocast_enabled(x.device):           # 3.
            raise RuntimeError(f"{self.name}: input dtype {x.dtype} differs from the LoRA dtype {lora_dtype} "
                               "outside autocast; fp16 inputs run only inside the autocast of the M0 precision "
                               f"tools (x {tuple(x.shape)}, omega {tuple(omega.shape)})")
        base_out = self.base(x)                                                # 4. frozen W x
        a = self.lora_A(self.lora_dropout(x))                                  # 5. [B, ..., r], one mask
        # 6. [B, r], column j = omega_{j // r*}: the values of repeat_interleave(r*, dim=1) (§12.6.2 step 6), built
        # with expand + reshape because torch 2.0.0 repeat_interleave without output_size reads the repeats back to
        # the host (2 .item() per call: a CUDA sync in every one of the 224 layers per forward and per generated
        # token, against the 12.7 rule of no per-layer synchronisation).
        w = omega.to(lora_dtype)[:, :, None].expand(-1, self.num_experts, self.r_star).reshape(omega.shape[0], self.r)
        w = w.view(w.shape[0], *([1] * (a.dim() - 2)), w.shape[1])             #    [B, 1, ..., 1, r]
        g = a.to(lora_dtype) * w                                               # 7. gate product in the LoRA dtype
        lora_out = self.lora_B(g)                                              # 8. = sum_k omega_k B_k A_k x
        return base_out + (lora_out * self.scaling).to(base_out.dtype)         # 9.


def _canonical_position(name: str) -> tuple[int, int]:
    """(layer index, CANONICAL_TARGETS index) of a wrapped module name ``...layers.<i>.<self_attn|mlp>.<target>``."""
    match = _WRAPPED_NAME_RE.fullmatch(name)
    if match is None or match.group(3) not in CANONICAL_TARGETS or _TARGET_PARENT[match.group(3)] != match.group(2):
        raise ValueError(f"MoE-LoRA module {name!r} is not at a LLaMA target position "
                         "(model.layers.<i>.<self_attn|mlp>.<target>)")
    return int(match.group(1)), CANONICAL_TARGETS.index(match.group(3))


def attach_moe_lora(llm: "LlamaForCausalLM", cfg: "LoraConfig", gate_ctx: GateContext, main_seed: int) -> list[str]:
    """Replace every target Linear of the CPU LLaMA ``llm`` by a ``MoELoRALinear`` and initialise the LoRA tensors.

    Targets are ``cfg.target_modules`` (⊆ ``CANONICAL_TARGETS``, D7) in every decoder layer. The initial values are
    drawn inside ``seeding.torch_seed_scope(main_seed, "init", "lora")`` in the canonical order layer 0..L-1 and,
    inside a layer, q, k, v, o, gate, up, down (not the HF registration order gate, down, up), so they do not depend
    on how the modules were created; the global CPU RNG state is unchanged afterwards (§12.6.3). Returns the
    replaced module names (relative to ``llm``) in that order. M9 moves the model to its device afterwards.
    """
    targets = tuple(cfg.target_modules)
    unknown = [t for t in targets if t not in CANONICAL_TARGETS]
    if unknown:
        raise ValueError(f"lora.target_modules {unknown} not in {list(CANONICAL_TARGETS)}")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError(f"lora.target_modules must be non-empty without duplicates, got {list(targets)}")
    if not isinstance(gate_ctx, GateContext) or gate_ctx.num_experts != cfg.num_experts:
        raise ValueError(f"gate_ctx must be a GateContext with lora.num_experts = {cfg.num_experts} experts")
    for tname, tensor in itertools.chain(llm.named_parameters(), llm.named_buffers()):
        if tensor.device.type != "cpu":
            raise ValueError(f"attach_moe_lora needs the LLM on CPU (M9 moves it after attaching); "
                             f"{tname} is on {tensor.device}")
    layers = getattr(getattr(llm, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("attach_moe_lora expects a LlamaForCausalLM (llm.model.layers)")

    found = []
    for i, layer in enumerate(layers):
        for target in CANONICAL_TARGETS:
            if target not in targets:
                continue
            parent_name = _TARGET_PARENT[target]
            module_name = f"model.layers.{i}.{parent_name}.{target}"
            parent = getattr(layer, parent_name, None)
            base = getattr(parent, target, None) if isinstance(parent, nn.Module) else None
            if not isinstance(base, nn.Module):
                raise ValueError(f"target {target} not found in layer {i} ({module_name})")
            found.append((module_name, parent, target, base))
    if len(found) != len(layers) * len(targets):
        raise ValueError(f"replaced {len(found)} modules, expected layers x targets = {len(layers)} x {len(targets)}")
    # Build every wrapper before replacing anything, so a rejected base leaves the model unchanged.
    wrapped = [(module_name, parent, target,
                MoELoRALinear(base, r=cfg.r, num_experts=cfg.num_experts, alpha=cfg.alpha, dropout=cfg.dropout,
                              scaling_rule=cfg.scaling_rule, gate_ctx=gate_ctx, name=module_name))
               for module_name, parent, target, base in found]
    for _, parent, target, wrapper in wrapped:
        setattr(parent, target, wrapper)
    with seeding.torch_seed_scope(main_seed, "init", "lora"):
        for _, _, _, wrapper in wrapped:
            wrapper.reset_lora_parameters()
    return [module_name for module_name, _, _, _ in wrapped]


def lora_named_parameters(llm: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """``(name, parameter)`` of every LoRA tensor in ``llm``, A before B, in the ``attach_moe_lora`` order.

    Names are relative to ``llm`` (``model.layers.0.self_attn.q_proj.lora_A.weight``); M9 prefixes them and sorts
    them by name for the optimizer group (§12.6.8).
    """
    wrapped = [(_canonical_position(name), name, module) for name, module in llm.named_modules()
               if isinstance(module, MoELoRALinear)]
    wrapped.sort(key=lambda item: item[0])
    out: list[tuple[str, nn.Parameter]] = []
    for _, name, module in wrapped:
        out.append((f"{name}.lora_A.weight", module.lora_A.weight))
        out.append((f"{name}.lora_B.weight", module.lora_B.weight))
    return out


__all__ = ["CANONICAL_TARGETS", "SCALING_RULES", "GateContextError", "GateUse", "GateContext", "lora_scaling",
           "MoELoRALinear", "attach_moe_lora", "lora_named_parameters"]
