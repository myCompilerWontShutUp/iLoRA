"""M6 projector (DESIGN.md §10; D3, D4, D9, D19).

The projector maps SASRec item representations ``[n, 64] float32`` (M3
``embed_items``) to behaviour tokens ``[n, hidden]`` in the LLM input-embedding
space (Proj_τ1, [논문 p.4 §2, 식 6]). M9 calls it once per hybrid batch, only
for the items that appear in the prompt, and inserts the tokens at the
``[HistoryEmb]``/``[CansEmb]`` positions; text-only batches never call it (D3).

Structure ``Linear(in, out) -> GELU -> Linear(out, out)`` with bias and no
dropout, parameter names ``mlp_proj.0.*`` and ``mlp_proj.2.*``
(``[코드] U model/mlp_projector.py:6-10``, ``L`` same blob ``07f66b36``;
two-layer perceptron ``[LLaRA 간접 §4.1 p.5 식 5]``). Parameters stay float32
(D9); the dtype of the output is decided by the caller's autocast. The widths
come from the SASRec state and the LLaMA config, not from the config file, so
the ``projector`` config section has no keys (§10.6.7).
"""
from __future__ import annotations

import torch
from torch import nn

from . import seeding


def _check_dims(in_dim: object, out_dim: object) -> None:
    """``in_dim`` and ``out_dim`` must be positive ints (§10.6.6)."""
    for name, value in (("in_dim", in_dim), ("out_dim", out_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"projector {name} must be a positive int, got {value!r}")


class Projector(nn.Module):
    """``Linear(in_dim, out_dim) -> GELU -> Linear(out_dim, out_dim)`` (§10.6.1). Build it with ``build_projector``."""

    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        _check_dims(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        # [코드] U model/mlp_projector.py:6-10: widths rec_size -> llm_size -> llm_size
        # (U model/model_interface.py:393), bias on, nn.GELU() default (erf, approximate="none"), no dropout,
        # default nn.Linear initialisation.
        # Both Linear layers are created on CPU in float32 so that the CPU generator of the seed scope fills them
        # whatever the default device is (§10.6.2).
        self.mlp_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, device="cpu", dtype=torch.float32),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, device="cpu", dtype=torch.float32),
        )

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"

    def forward(self, item_repr: torch.Tensor) -> torch.Tensor:
        """``[..., in_dim] float32`` -> ``[..., out_dim]`` (§10.6.3). No dtype conversion: autocast decides."""
        if not isinstance(item_repr, torch.Tensor):
            raise TypeError(f"projector input must be a torch.Tensor, got {type(item_repr).__name__}")
        if item_repr.dtype != torch.float32:
            raise ValueError(f"projector input must be float32 (M3 item representation), got {item_repr.dtype}")
        if item_repr.dim() < 2:
            raise ValueError(f"projector input must have at least 2 dimensions, got shape {tuple(item_repr.shape)}")
        if item_repr.shape[-1] != self.in_dim:
            raise ValueError(f"projector input last dimension must be in_dim = {self.in_dim}, "
                             f"got shape {tuple(item_repr.shape)}")
        if item_repr.numel() == 0:
            raise ValueError(f"projector input must have at least one element, got shape {tuple(item_repr.shape)}")
        bad = [(name, p.dtype) for name, p in self.named_parameters() if p.dtype != torch.float32]
        if bad:
            raise RuntimeError(f"projector parameters must stay float32 (D9), got {bad}")
        return self.mlp_proj(item_repr)


def build_projector(in_dim: int, out_dim: int, main_seed: int) -> Projector:
    """Projector with its initial values drawn from ``derive_seed(main_seed, "init", "projector")`` (§10.6.2).

    Draw order on the CPU generator (``nn.Linear.reset_parameters``, row-major): ``mlp_proj.0.weight``
    kaiming_uniform(a=sqrt(5)) = U(-1/sqrt(in), 1/sqrt(in)), ``mlp_proj.0.bias`` U(-1/sqrt(in), 1/sqrt(in)),
    ``mlp_proj.2.weight`` and ``mlp_proj.2.bias`` U(-1/sqrt(out), 1/sqrt(out)). The global CPU RNG state is
    restored afterwards and CUDA generators are not touched (``seeding.torch_seed_scope``), so the values do not
    depend on other modules or on whether a gate is built. M9 moves the module to the model device.
    """
    _check_dims(in_dim, out_dim)
    with seeding.torch_seed_scope(main_seed, "init", "projector"):
        return Projector(in_dim, out_dim)


def projector_param_count(in_dim: int, out_dim: int) -> int:
    """``in*out + out + out*out + out`` (17,047,552 for 64 -> 4096, §10.2)."""
    _check_dims(in_dim, out_dim)
    return in_dim * out_dim + out_dim + out_dim * out_dim + out_dim


__all__ = ["Projector", "build_projector", "projector_param_count"]
