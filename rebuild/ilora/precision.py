"""M0 precision 도구 (DESIGN.md §3.1 precision 도구, §4.2, §14.6.10 계약 추가 제안 5, D9).

``autocast_for(precision, device)`` opens the autocast region of one forward
(M10 학습, M11 생성) and ``make_grad_scaler(tc, device)`` builds the GradScaler
from the ``train.scaler_*`` keys. Both refuse a combination that torch 2.0.0
would silently turn into "no autocast" or "no scaling": torch 2.0.0 disables
CPU fp16 autocast and a GradScaler without CUDA with only a warning
(``torch2.0 amp/autocast_mode.py``, ``torch2.0 cuda/amp/grad_scaler.py``).
"""
from __future__ import annotations

import contextlib
from typing import Any, ContextManager, Union

import torch

PRECISIONS = ("fp16_amp", "bf16_amp", "fp32")   # train.precision choices (§14.6.9)

DeviceLike = Union[torch.device, str]


def _as_device(device: DeviceLike) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _require_cuda(precision: str, dev: torch.device) -> None:
    if dev.type != "cuda":
        raise ValueError(f"precision {precision} needs a CUDA device, got {dev} (D9; fp32 is the CPU mode)")
    if not torch.cuda.is_available():
        raise ValueError(f"precision {precision} needs CUDA but torch.cuda.is_available() is False")


def autocast_for(precision: str, device: DeviceLike) -> ContextManager[Any]:
    """Autocast region for ``precision`` on ``device`` (D9).

    - ``fp16_amp``: ``torch.autocast("cuda", dtype=torch.float16)`` (paper_spec)
    - ``bf16_amp``: ``torch.autocast(device.type, dtype=torch.bfloat16)`` (D9 대안; CPU bf16 is the
      CPU test proxy of §13.6.9)
    - ``fp32``: no autocast (CPU smoke)
    """
    dev = _as_device(device)
    if precision == "fp16_amp":
        _require_cuda(precision, dev)
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16_amp":
        if dev.type == "cuda":
            _require_cuda(precision, dev)
        elif dev.type != "cpu":
            raise ValueError(f"precision bf16_amp supports cuda and cpu devices, got {dev}")
        return torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
    if precision == "fp32":
        return contextlib.nullcontext()
    raise ValueError(f"unknown precision {precision!r}, expected one of {PRECISIONS}")


def make_grad_scaler(tc: Any, device: DeviceLike) -> torch.cuda.amp.GradScaler:
    """GradScaler from the ``train`` section ``tc`` (D9, §14.6.3).

    Enabled only for ``fp16_amp``, with ``scaler_init_scale``, ``scaler_growth_factor``,
    ``scaler_backoff_factor`` and ``scaler_growth_interval``. For ``bf16_amp`` and ``fp32`` the
    scaler is disabled (``scale`` returns the loss, ``step`` calls ``optimizer.step``,
    ``get_scale`` is 1.0).
    """
    dev = _as_device(device)
    precision = tc.precision
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}, expected one of {PRECISIONS}")
    enabled = precision == "fp16_amp"
    if enabled:
        _require_cuda(precision, dev)
    return torch.cuda.amp.GradScaler(
        init_scale=float(tc.scaler_init_scale),
        growth_factor=float(tc.scaler_growth_factor),
        backoff_factor=float(tc.scaler_backoff_factor),
        growth_interval=int(tc.scaler_growth_interval),
        enabled=enabled,
    )


__all__ = ["PRECISIONS", "autocast_for", "make_grad_scaler"]
