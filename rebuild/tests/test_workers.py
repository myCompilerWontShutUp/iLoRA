"""Worker-count independence (DESIGN.md §17.6.5 "worker 0/2/8 무관", §4.6.4, §6.8 #2).

Candidates, templates and curriculum draws are keyed by ``derive_seed`` coordinates,
never by DataLoader worker state, so the number of workers (0, 2, 8) must not
change any batch (M0-4, M2-2, M4-8, M5-8). ``test_candidates_identical_across_workers``
checks the M2 part: candidates computed inside spawn workers through the picklable
``CandidateFn`` (``functools.partial(candidates_for, main_seed=..., cfg=...)``,
§6.6.9) equal the main-process values. Each worker also consumes its global
``random``/numpy/torch generators by a batch-dependent amount first, as real
worker state would differ.

The batch order here is a plain ``train_order`` permutation (§3.2 step 1) cut into
micro-batches of ``train.micro_batch_size``; the full TrainBatch/EvalBatch tests of
§17.6.5 (``test_train_batches_identical_across_workers[0-2-8]``,
``test_eval_batches_identical_across_workers[0-2]``) use the M4 plan and M5 loaders.
"""
from __future__ import annotations

import functools
import itertools
import random
from typing import Any, Callable, Sequence

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from ilora import batching, candidates, prompts, seeding

N_BATCHES = 30                     # §17.6.5: first 30 batches of epoch 0 of the smoke plan


def _identity(item: Any) -> Any:
    """collate_fn for ``batch_size=None`` (module level so that spawn workers can unpickle it)."""
    return item


class CandidateBatches(Dataset):
    """One item = one micro-batch: ``[(sample_id, candidates), ...]`` computed in the worker."""

    def __init__(self, samples: Sequence[Any], batches: Sequence[Sequence[int]], cand_fn: Callable[..., Any],
                 epoch: int) -> None:
        self.samples = tuple(samples)
        self.batches = [tuple(b) for b in batches]
        self.cand_fn = cand_fn
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> list:
        random.random()                                    # disturb worker-global generators
        np.random.random(index % 5 + 1)
        torch.rand(index % 3 + 1)
        return [(self.samples[i].sample_id, self.cand_fn(self.samples[i], epoch=self.epoch))
                for i in self.batches[index]]


def _batches(n_samples: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    order = torch.randperm(n_samples, generator=seeding.torch_generator(seed, "train_order", epoch)).tolist()
    n_full = n_samples // batch_size                       # drop_last
    return [order[b * batch_size:(b + 1) * batch_size] for b in range(n_full)][:N_BATCHES]


@pytest.mark.slow
@pytest.mark.criterion("M0-4", "M2-2")
@pytest.mark.parametrize("num_workers", [0, 2, 8])
def test_candidates_identical_across_workers(num_workers, cpu_cfg, smoke_data):
    seed, epoch = cpu_cfg.run.seed, 0
    samples = smoke_data.train
    batches = _batches(len(samples), cpu_cfg.train.micro_batch_size, seed, epoch)
    assert len(batches) == N_BATCHES and len(samples) == 156
    cand_fn = functools.partial(candidates.candidates_for, main_seed=seed, cfg=cpu_cfg.candidates)
    expected = [[(samples[i].sample_id, candidates.candidates_for(samples[i], main_seed=seed, epoch=epoch,
                                                                  cfg=cpu_cfg.candidates)) for i in batch]
                for batch in batches]
    loader = DataLoader(CandidateBatches(samples, batches, cand_fn, epoch), batch_size=None, shuffle=False,
                        num_workers=num_workers, collate_fn=_identity,
                        generator=seeding.torch_generator(0, "test", "workers"))
    # with 0 workers __getitem__ runs here: keep this process's global generators as they were
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        got = [[(sample_id, tuple(cands)) for sample_id, cands in batch] for batch in loader]
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
    assert got == expected


# ----------------------------------------------------------------------------
# TrainBatch / EvalBatch through the M4 plan and the M5 loaders (§17.6.5)
# ----------------------------------------------------------------------------
def _assert_same_batch(a: Any, b: Any, where: str) -> None:
    """Every field equal: tensors by dtype, shape and value, lists and scalars by ``==``."""
    assert type(a) is type(b), where
    for name in a.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), f"{where}: {name}"
        else:
            assert x == y, f"{where}: {name}"


def _read_restoring_global_rng(make_iterator: Callable[[], Any], limit: int) -> list:
    """Up to ``limit`` items of a loader; this process's global random/numpy/torch states are restored
    afterwards (the DataLoader iterator draws its base seed from the global torch generator, M10 §14.6.5)."""
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        return list(itertools.islice(make_iterator(), limit))
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])


@pytest.mark.slow
@pytest.mark.criterion("M0-4", "M2-2", "M4-8", "M5-8")
@pytest.mark.parametrize("num_workers", [0, 2, 8])
def test_train_batches_identical_across_workers(num_workers, smoke_data, bundle, templates, profile_cfg):
    """smoke_cpu plan of epoch 0, first 30 TrainBatch through ``make_train_loader`` with 0/2/8 workers =
    the same batches built one by one in this process; tau, hybrid, sample ids follow the plan and the
    candidates and templates are the keyed values."""
    cfg = profile_cfg("smoke_cpu", f"batching.num_workers={num_workers}")
    plan = prompts.build_batch_plan(smoke_data.train, 0, cfg)
    assert len(plan) == 39 and len(smoke_data.train) == 156             # §17.6.6
    cand_fn = functools.partial(candidates.candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)
    dataset = batching.PlannedTrainBatches(plan, smoke_data.train, smoke_data.catalog, templates, cand_fn, bundle,
                                           cfg.prompt, cfg.batching, cfg.run.seed)
    expected = [dataset[i] for i in range(N_BATCHES)]
    got = _read_restoring_global_rng(
        lambda: batching.make_train_loader(plan, smoke_data, cfg, 0, bundle=bundle, templates=templates), N_BATCHES)
    assert len(got) == N_BATCHES
    for i, (a, b) in enumerate(zip(got, expected)):
        _assert_same_batch(a, b, f"batch {i} ({num_workers} workers)")
    assert [(b.tau, b.hybrid) for b in got] == [(e.tau, e.hybrid) for e in plan[:N_BATCHES]]
    assert any(b.hybrid for b in got) and not all(b.hybrid for b in got)
    for b, entry in zip(got, plan[:N_BATCHES]):
        chosen = [smoke_data.train[j] for j in entry.sample_indices]
        assert b.sample_ids == [s.sample_id for s in chosen]
        assert b.cand_ids.tolist() == [list(candidates.candidates_for(s, main_seed=1234, epoch=0, cfg=cfg.candidates))
                                       for s in chosen]
        assert b.template_ids == [prompts.choose_template(cfg.prompt, 1234, 3, s, 0) for s in chosen]


@pytest.mark.slow
@pytest.mark.criterion("M0-4", "M5-8")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_eval_batches_identical_across_workers(num_workers, full_data, bundle, templates, profile_cfg):
    """Whole validation split through ``make_eval_loader`` with 0/2 workers = the EvalBatches items built
    in this process, every field."""
    cfg = profile_cfg("paper_spec", f"batching.num_workers={num_workers}")
    cand_fn = functools.partial(candidates.candidates_for, main_seed=cfg.run.seed, cfg=cfg.candidates)
    dataset = batching.EvalBatches(full_data.val, cfg.eval.batch_size, full_data.catalog, templates, cand_fn, bundle,
                                   cfg.prompt, cfg.batching, cfg.run.seed)
    expected = [dataset[i] for i in range(len(dataset))]
    got = _read_restoring_global_rng(
        lambda: batching.make_eval_loader(full_data, "val", cfg, bundle=bundle, templates=templates), 1000)
    assert len(got) == len(expected) == 12
    for i, (a, b) in enumerate(zip(got, expected)):
        _assert_same_batch(a, b, f"batch {i} ({num_workers} workers)")
    assert [x for b in got for x in b.sample_ids] == [s.sample_id for s in full_data.val]
