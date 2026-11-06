"""M0 난수 (DESIGN.md §3.1, §4.6.4).

All randomness in the rebuild comes from ``derive_seed``. A random choice is
identified by its purpose and coordinates (for example
``("candidates", "val", row)``), never by DataLoader worker state, so results
do not depend on the number of workers or on batch composition.
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Iterator, Union

import numpy as np
import torch

Key = Union[str, int]

_MASK63 = (1 << 63) - 1


def _check_key(key: object) -> None:
    # bool is a subclass of int; reject it so that True and 1 cannot collide.
    if isinstance(key, bool) or not isinstance(key, (str, int)):
        raise TypeError(f"seed key must be str or int, got {type(key).__name__}: {key!r}")
    if isinstance(key, str) and "|" in key:
        raise ValueError(f"seed key must not contain '|': {key!r}")


def derive_seed(main_seed: int, *keys: Key) -> int:
    """Return a 63-bit seed for ``(main_seed, *keys)``.

    The seed is the first 8 bytes of SHA-256 over ``"{main_seed}|{key1}|..."``,
    masked to 63 bits (DESIGN.md §3.1).
    """
    _check_key(main_seed)
    if not isinstance(main_seed, int):
        raise TypeError(f"main_seed must be int, got {type(main_seed).__name__}")
    for key in keys:
        _check_key(key)
    text = "|".join([str(main_seed), *(str(k) for k in keys)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK63


def np_rng(main_seed: int, *keys: Key) -> np.random.Generator:
    """numpy Generator (PCG64) for one purpose."""
    return np.random.Generator(np.random.PCG64(derive_seed(main_seed, *keys)))


def torch_generator(main_seed: int, *keys: Key) -> torch.Generator:
    """CPU torch.Generator for one purpose (e.g. DataLoader order)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(derive_seed(main_seed, *keys))
    return gen


def uniform(main_seed: int, *keys: Key) -> float:
    """One draw from Uniform[0, 1) for one purpose."""
    return float(np_rng(main_seed, *keys).random())


@contextlib.contextmanager
def torch_seed_scope(main_seed: int, *keys: Key) -> Iterator[None]:
    """Seed the CPU torch RNG for a block and restore it afterwards.

    Used for module initialization: modules are created on CPU inside this
    scope, so the initial values are the same on CPU tests and GPU runs.
    Only the CPU generator is seeded and restored; CUDA generators are not
    touched (``torch.manual_seed`` would also re-seed every CUDA device).
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derive_seed(main_seed, *keys))
        yield


def seed_torch_global(main_seed: int, *keys: Key) -> int:
    """Seed the global torch RNG (CPU and CUDA). Returns the seed used."""
    seed = derive_seed(main_seed, *keys)
    torch.manual_seed(seed)
    return seed
