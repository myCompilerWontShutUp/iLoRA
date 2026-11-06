"""M0 난수 (DESIGN.md §4.8 #4; worker 수 무관은 test_workers.py)."""
import numpy as np
import pytest
import torch

from ilora.seeding import derive_seed, np_rng, torch_generator, torch_seed_scope, uniform

pytestmark = pytest.mark.criterion("M0-4")


def test_derive_seed_is_deterministic_and_63_bit():
    a = derive_seed(1234, "candidates", "val", 5)
    assert a == derive_seed(1234, "candidates", "val", 5)
    assert 0 <= a < 2**63


def test_derive_seed_separates_purposes_and_coordinates():
    seeds = {
        derive_seed(1234, "candidates", "val", 5),
        derive_seed(1234, "candidates", "val", 6),
        derive_seed(1234, "candidates", "test", 5),
        derive_seed(1234, "template", "val", 5),
        derive_seed(1235, "candidates", "val", 5),
        derive_seed(1234, "candidates", "train", 5, 0),
        derive_seed(1234, "candidates", "train", 5, 1),
    }
    assert len(seeds) == 7


def test_derive_seed_matches_documented_formula():
    import hashlib

    digest = hashlib.sha256("1234|curriculum|7".encode()).digest()
    assert derive_seed(1234, "curriculum", 7) == int.from_bytes(digest[:8], "big") & (2**63 - 1)


@pytest.mark.parametrize("bad", [True, 1.0, None, ("a",), "a|b"])
def test_derive_seed_rejects_bad_keys(bad):
    with pytest.raises((TypeError, ValueError)):
        derive_seed(1234, bad)


def test_rng_helpers_are_reproducible():
    assert np.array_equal(np_rng(1, "x").permutation(10), np_rng(1, "x").permutation(10))
    assert torch.equal(torch.randperm(10, generator=torch_generator(1, "x")),
                       torch.randperm(10, generator=torch_generator(1, "x")))
    assert uniform(1, "x") == uniform(1, "x")
    assert 0.0 <= uniform(1, "x") < 1.0


def test_torch_seed_scope_restores_global_state():
    torch.manual_seed(0)
    expected_next = torch.rand(1)
    torch.manual_seed(0)
    with torch_seed_scope(1234, "init", "gate"):
        inside_a = torch.rand(4)
    after = torch.rand(1)
    assert torch.equal(after, expected_next)
    with torch_seed_scope(1234, "init", "gate"):
        inside_b = torch.rand(4)
    assert torch.equal(inside_a, inside_b)
