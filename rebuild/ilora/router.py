"""M7 gate와 routing (DESIGN.md §11; D1, D7, D9, D19).

``Router`` is the only source of the expert weights ``omega [B, K] float32`` for
the three routing modes (§11.1, §11.6.1, §3.1 ``ω``):

- ``dynamic`` (E2, paper_spec): ``omega = Softmax(W_g z)``, one shared linear gate
  without bias (식 11, D1), computed in fp32 outside autocast with CUDA TF32 off
  (§11.6.3, §11.6.6);
- ``cluster_hard`` (E4): a one-hot ``omega`` looked up by ``sample_id`` through a
  ``HardGate`` (M15 ``ClusterHardGate``). No gate is built, called or trained
  [SPEC §9, §11];
- ``none`` (E1, K = 1): ``omega = 1``, no gate (the single LoRA of LLaRA).

M9 calls ``router(z, sample_ids)`` once per batch; that call is the single
collection point of gate values [SPEC §4]. During evaluation the recorder
(``start_recording`` / ``stop_recording``) keeps CPU copies of omega and z for
the M11 cross-check (§11.6.5). The Router never accumulates per-sample values
outside a recording and never writes files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from . import seeding
# The single definition of the fp32 region (§7.6.3, §11.6.3): autocast off and, on CUDA, TF32 off.
from .sasrec import strict_fp32

if TYPE_CHECKING:   # the router section type (M0 schema); only routing_mode is read
    from .sections import RouterConfig

# |row sum - 1| tolerance of omega (§11.7: fp32 softmax over K = 4 measured at most 2.4e-7, about 40x margin).
# The single definition: M8 GateContext imports it (§12.6.4); M12 records.gate_rowsum_atol has the same value.
ROW_SUM_ATOL = 1e-5
ROUTING_MODES = ("dynamic", "cluster_hard", "none")   # router.routing_mode (§11.6.8)
_AUTOCAST_DEVICE_TYPES = ("cpu", "cuda")


class RouterError(ValueError):
    """Invalid Router construction, input, output or recorder use (§11.6.2, §11.6.3, §11.6.5)."""


class HardGate(Protocol):
    """One-hot omega looked up by ``sample_id`` (§11.6.1).

    M15 ``ClusterHardGate`` satisfies it structurally; M7 does not import ``cluster.py``,
    so the dependency points from M15 to M7.
    """

    lookups: int   # cumulative number of looked-up samples (M10 logs the training increments)

    def __call__(self, sample_ids: Sequence[str], device: torch.device) -> torch.Tensor:  # [B, K] float32 one-hot
        ...


@dataclass(frozen=True)
class GateRecord:
    """omega and z of one evaluation, in call order (M7 -> M11, §11.6.5)."""
    split: str
    routing_mode: str
    num_experts: int
    sample_ids: tuple[str, ...]
    omega: torch.Tensor          # [N, K] float32 CPU
    z: torch.Tensor              # [N, d] float32 CPU
    gate_calls: int              # gate calls during the recording (0 for cluster_hard and none)


@dataclass
class _Recording:
    split: str
    calls_at_start: int
    sample_ids: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    omega: list[torch.Tensor] = field(default_factory=list)
    z: list[torch.Tensor] = field(default_factory=list)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RouterError(f"{name} must be a positive int, got {value!r}")
    return value


class LinearGate(nn.Module):
    """식 11 gate: ``omega = Softmax(W_g z)`` with ``W_g in R^{K x d}`` and no bias (D1).

    [논문 p.6 §3.2.2 식 11]; same shape as the unused ``Dense`` gate ``nn.Linear(64, num_moe, bias=False)``
    + Softmax ``[코드] U model/peft/tuners/gating.py:5-16``. Built on CPU in float32 with the default
    ``nn.Linear`` initialisation, kaiming_uniform(a=sqrt(5)) = U(-1/sqrt(d), 1/sqrt(d)); ``Router`` builds it
    inside the ``("init", "gate")`` seed scope (§11.6.2). The weight is ``proj.weight``.
    """

    def __init__(self, in_dim: int, num_experts: int) -> None:
        super().__init__()
        _positive_int("in_dim", in_dim)
        _positive_int("num_experts", num_experts)
        self.proj = nn.Linear(in_dim, num_experts, bias=False, device="cpu", dtype=torch.float32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = F.linear(z, self.proj.weight)          # [B, K]
        return torch.softmax(logits, dim=-1)


class Router(nn.Module):
    """The omega source of M9 (``OmegaSource``) for the three routing modes (§11.6.1).

    ``rcfg`` is the ``router`` section (``routing_mode``). ``num_experts`` is ``lora.num_experts`` (the only
    definition of K, §12.6.9), ``z_dim`` is ``SASRecEncoder.hidden_size``. The Router reads no other config.
    """

    routing_mode: str
    num_experts: int
    z_dim: int
    gate: Optional[LinearGate]
    hard_gate: Optional[HardGate]
    gate_calls: int

    def __init__(self, rcfg: "RouterConfig", *, num_experts: int, z_dim: int, main_seed: int,
                 hard_gate: Optional[HardGate]) -> None:
        super().__init__()
        mode = getattr(rcfg, "routing_mode", None)
        if mode not in ROUTING_MODES:
            raise RouterError(f"router.routing_mode must be one of {ROUTING_MODES}, got {mode!r}")
        _positive_int("num_experts", num_experts)
        _positive_int("z_dim", z_dim)
        if isinstance(main_seed, bool) or not isinstance(main_seed, int):
            raise RouterError(f"main_seed must be an int, got {main_seed!r}")
        # Mode <-> K (§11.6.2, §11.7): E1 has no gate parameter only if none <=> K == 1.
        if mode == "none" and num_experts != 1:
            raise RouterError(f"routing_mode none needs lora.num_experts == 1, got {num_experts}")
        if mode != "none" and num_experts < 2:
            raise RouterError(f"routing_mode {mode} needs lora.num_experts >= 2, got {num_experts} "
                              "(K = 1 is routing_mode none)")
        # Mode <-> hard_gate: the dynamic path never builds or reads cluster artifacts [SPEC §22].
        if mode == "cluster_hard":
            if hard_gate is None:
                raise RouterError("routing_mode cluster_hard needs a hard_gate (M15 ClusterHardGate)")
            if not callable(hard_gate):
                raise RouterError(f"hard_gate must be callable as hard_gate(sample_ids, device), "
                                  f"got {type(hard_gate).__name__}")
            lookups = getattr(hard_gate, "lookups", None)
            if isinstance(lookups, bool) or not isinstance(lookups, int):
                raise RouterError(f"hard_gate must have an int attribute lookups (HardGate protocol), got {lookups!r}")
        elif hard_gate is not None:
            raise RouterError(f"routing_mode {mode} takes no hard_gate, got {type(hard_gate).__name__}")

        self.routing_mode = mode
        self.num_experts = num_experts
        self.z_dim = z_dim
        self.gate_calls = 0
        # A plain attribute, never registered as a submodule: it is not part of state_dict (§11.6.1, §11.6.7).
        object.__setattr__(self, "hard_gate", hard_gate)
        if mode == "dynamic":
            # ("init", "gate") key of §4.6.4: CPU generator seeded and restored, CUDA untouched (§11.6.2).
            with seeding.torch_seed_scope(main_seed, "init", "gate"):
                self.gate = LinearGate(z_dim, num_experts)
        else:
            self.gate = None
        self._recording: Optional[_Recording] = None

    def extra_repr(self) -> str:
        return f"routing_mode={self.routing_mode}, num_experts={self.num_experts}, z_dim={self.z_dim}"

    # ------------------------------------------------------------------ forward (§11.6.3)
    def forward(self, z: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor:
        ids = self._check_input(z, sample_ids)
        recording = self._recording
        if recording is not None:
            self._check_recordable(recording, ids)
        if self.routing_mode == "dynamic":
            assert self.gate is not None
            # M3 sasrec.strict_fp32 (§11.6.3): omega must not depend on the caller's autocast or TF32 settings (§11.7)
            with strict_fp32(z.device):
                omega = self.gate(z)
            self.gate_calls += 1
        elif self.routing_mode == "cluster_hard":
            # z is checked but not used; an unknown sample_id raises M15's KeyError unchanged [SPEC §21].
            omega = self.hard_gate(list(ids), z.device)
        else:
            omega = torch.ones(z.shape[0], 1, dtype=torch.float32, device=z.device)
        self._check_output(omega, z, ids)
        if recording is not None:
            recording.sample_ids.extend(ids)
            recording.seen.update(ids)
            recording.omega.append(omega.detach().to(device="cpu", dtype=torch.float32, copy=True))
            recording.z.append(z.detach().to(device="cpu", dtype=torch.float32, copy=True))
        return omega

    def _check_input(self, z: torch.Tensor, sample_ids: Sequence[str]) -> tuple[str, ...]:
        if not isinstance(z, torch.Tensor):
            raise RouterError(f"z must be a torch.Tensor, got {type(z).__name__}")
        if z.dtype != torch.float32:
            raise RouterError(f"z must be float32 (M3 z contract), got {z.dtype}")
        if z.dim() != 2 or z.shape[1] != self.z_dim:
            raise RouterError(f"z must be [B, {self.z_dim}] (no squeeze), got {tuple(z.shape)}")
        if z.shape[0] < 1:
            raise RouterError(f"z must have at least one row, got {tuple(z.shape)}")
        if z.device.type not in _AUTOCAST_DEVICE_TYPES:
            raise RouterError(f"z must be on a cpu or cuda device, got {z.device}")
        if isinstance(sample_ids, (str, bytes)):
            raise RouterError(f"sample_ids must be a sequence of str, got one {type(sample_ids).__name__}")
        ids = tuple(sample_ids)
        if len(ids) != z.shape[0]:
            raise RouterError(f"len(sample_ids) = {len(ids)} != B = {z.shape[0]}")
        for sid in ids:
            if not isinstance(sid, str):
                raise RouterError(f"sample_ids must be str, got {sid!r}")
        finite = torch.isfinite(z).all(dim=1)
        if not bool(finite.all()):
            row = int((~finite).nonzero()[0, 0])
            raise RouterError(f"z has NaN/Inf at row {row} (sample_id {ids[row]!r})")
        return ids

    def _check_output(self, omega: torch.Tensor, z: torch.Tensor, ids: tuple[str, ...]) -> None:
        mode, batch, k = self.routing_mode, z.shape[0], self.num_experts
        if not isinstance(omega, torch.Tensor):
            raise RouterError(f"{mode} omega must be a torch.Tensor, got {type(omega).__name__}")
        if tuple(omega.shape) != (batch, k):
            raise RouterError(f"{mode} omega must be [B, K] = [{batch}, {k}], got {tuple(omega.shape)}")
        if omega.dtype != torch.float32:
            raise RouterError(f"{mode} omega must be float32, got {omega.dtype}")
        if omega.device != z.device:
            raise RouterError(f"{mode} omega must be on the device of z ({z.device}), got {omega.device}")
        with torch.no_grad():
            values = omega.detach()
            ok = (torch.isfinite(values).all(dim=1) & (values >= 0).all(dim=1)
                  & ((values.sum(dim=1) - 1.0).abs() <= ROW_SUM_ATOL))
            rule = f"finite, >= 0, |row sum - 1| <= {ROW_SUM_ATOL}"
            if mode == "cluster_hard":
                # One-hot: values 0 or 1 and exactly one 1 per row, so the row sum is exactly 1 [SPEC §11].
                ok &= ((values == 0) | (values == 1)).all(dim=1) & ((values == 1).sum(dim=1) == 1)
                rule += ", one-hot"
            bad = ~ok
            if bool(bad.any()):      # all value conditions in one synchronisation (§11.6.3)
                row = int(bad.nonzero()[0, 0])
                raise RouterError(f"{mode} omega breaks the omega contract ({rule}) at row {row}, "
                                  f"sample_id {ids[row]!r}: {values[row].tolist()}")

    # ------------------------------------------------------------------ optimizer and checkpoint (§11.6.7)
    def trainable_parameters(self) -> list[nn.Parameter]:
        """``[W_g]`` for dynamic, ``[]`` for cluster_hard and none (no gate group in the optimizer [SPEC §11])."""
        return [self.gate.proj.weight] if self.gate is not None else []

    def provenance(self) -> dict:
        """routing_mode, K, z_dim and gate weight shape (plain JSON values); M10 compares it on reload."""
        return {
            "routing_mode": self.routing_mode,
            "num_experts": self.num_experts,
            "z_dim": self.z_dim,
            "gate_shape": list(self.gate.proj.weight.shape) if self.gate is not None else None,
        }

    # ------------------------------------------------------------------ recorder (§11.6.5)
    def start_recording(self, split: str) -> None:
        """Record omega and z of the following no_grad forwards of ``split`` (evaluation only)."""
        if self._recording is not None:
            raise RouterError(f"start_recording({split!r}) while recording split {self._recording.split!r}")
        if not isinstance(split, str) or not split or ":" in split:
            raise RouterError(f"split must be a non-empty str without ':', got {split!r}")
        self._recording = _Recording(split=split, calls_at_start=self.gate_calls)

    def stop_recording(self, expected_sample_ids: Sequence[str]) -> GateRecord:
        """End the recording and return it; the recorded ids must equal ``expected_sample_ids`` in order.

        The recording ends in every case (also when the check fails), so ``discard_recording`` after a failed
        stop is a no-op.
        """
        recording = self._recording
        if recording is None:
            raise RouterError("stop_recording without start_recording")
        self._recording = None
        if isinstance(expected_sample_ids, (str, bytes)):
            raise RouterError("expected_sample_ids must be a sequence of str, got one str")
        expected = tuple(expected_sample_ids)
        got = tuple(recording.sample_ids)
        if got != expected:
            n = min(len(got), len(expected))
            pos = next((i for i in range(n) if got[i] != expected[i]), n)
            have = got[pos] if pos < len(got) else None
            want = expected[pos] if pos < len(expected) else None
            raise RouterError(f"recorded sample_ids of split {recording.split!r} differ from the expected ids "
                              f"(recorded {len(got)}, expected {len(expected)}); first difference at position "
                              f"{pos}: recorded {have!r}, expected {want!r}")
        omega = (torch.cat(recording.omega) if recording.omega
                 else torch.empty(0, self.num_experts, dtype=torch.float32))
        z = torch.cat(recording.z) if recording.z else torch.empty(0, self.z_dim, dtype=torch.float32)
        return GateRecord(split=recording.split, routing_mode=self.routing_mode, num_experts=self.num_experts,
                          sample_ids=got, omega=omega, z=z,
                          gate_calls=self.gate_calls - recording.calls_at_start)

    def discard_recording(self) -> None:
        """Drop an unfinished recording (M11 calls it when an evaluation fails). No-op when not recording."""
        self._recording = None

    def _check_recordable(self, recording: _Recording, ids: tuple[str, ...]) -> None:
        if torch.is_grad_enabled():
            raise RouterError("gate values are recorded only in no_grad evaluation; a forward with gradients "
                              f"enabled while recording split {recording.split!r} is a training step [SPEC §4]")
        prefix = f"{recording.split}:"
        in_batch: set[str] = set()
        for sid in ids:
            if not sid.startswith(prefix):
                raise RouterError(f"sample_id {sid!r} does not belong to the recorded split {recording.split!r}")
            if sid in recording.seen or sid in in_batch:
                raise RouterError(f"duplicate sample_id {sid!r} in the recording of split {recording.split!r}")
            in_batch.add(sid)


__all__ = ["ROW_SUM_ATOL", "ROUTING_MODES", "RouterError", "HardGate", "GateRecord", "LinearGate", "Router"]
