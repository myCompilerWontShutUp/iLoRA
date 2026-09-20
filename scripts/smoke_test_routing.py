"""Local, dependency-light validation for the routing/clustering code added on this branch.

Everything in this file runs on plain CPU torch/numpy/sklearn (already available locally, see
experiments/CODEBASE_ANALYSIS.md "Other structural facts"). It deliberately does NOT import
`model.peft.tuners.moelora` — that import pulls in the whole vendored PEFT package, which in
turn imports `accelerate`/`transformers` at package-init time
(`model/peft/__init__.py` -> `peft_model.py`), neither of which is installed in this local
Windows environment (nor should be, per spec section 13: no attempt to fully replicate the
VESSL CUDA env on Windows). Instead, section 2 below reproduces the exact tensor contract that
`moelora.py` `Linear.forward`/`calculate_B` uses (moelora.py lines 750-799), with plain torch,
so the gate shapes this branch produces are verified against the real broadcast semantics
without needing the transformers-dependent import chain. The actual end-to-end forward pass
through `MoeLoraModel` + LlamaForCausalLM is marked VESSL_REQUIRED at the bottom.

Exit code is 0 if every check passes, 1 otherwise.
"""
import os
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from model.routing_utils import (
    parse_cluster_expert_mapping,
    validate_expert_mapping,
    build_onehot_gate,
    load_assignment_csv,
    gate_diagnostics,
)

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()
        FAILURES.append(name)


# --- 1. routing_utils: mapping parser -------------------------------------------------------

def test_parse_mapping_valid():
    m = parse_cluster_expert_mapping("0:0,1:1,2:2,3:3")
    assert m == {0: 0, 1: 1, 2: 2, 3: 3}, m


def test_parse_mapping_reordered():
    m = parse_cluster_expert_mapping("2:1, 0:3,1:0")
    assert m == {2: 1, 0: 3, 1: 0}, m


def test_parse_mapping_rejects_duplicate():
    try:
        parse_cluster_expert_mapping("0:0,0:1")
        raise AssertionError("duplicate cluster id was not rejected")
    except ValueError:
        pass


def test_parse_mapping_rejects_non_int():
    for bad in ["a:0", "0:b", "0-0", "", "0:0,", ":"]:
        try:
            parse_cluster_expert_mapping(bad)
            raise AssertionError(f"malformed mapping {bad!r} was not rejected")
        except ValueError:
            pass


def test_validate_mapping_rejects_out_of_range_expert():
    try:
        validate_expert_mapping({0: 0, 1: 5}, num_clusters=2, num_moe=4)
        raise AssertionError("out-of-range expert id was not rejected")
    except ValueError:
        pass


def test_validate_mapping_rejects_out_of_range_cluster():
    try:
        validate_expert_mapping({0: 0, 5: 1}, num_clusters=2, num_moe=4)
        raise AssertionError("out-of-range cluster id was not rejected")
    except ValueError:
        pass


# --- 1b. routing_utils: one-hot gate construction -------------------------------------------

def test_onehot_gate_shape_and_property():
    gate = build_onehot_gate([0, 3, 1, 2], num_moe=4)
    assert gate.shape == (4, 1, 4), gate.shape
    row_sums = gate.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), row_sums
    assert bool((gate.sum(dim=-1).squeeze(-1) == 1).all())
    expected = [0, 3, 1, 2]
    assert gate.squeeze(1).argmax(dim=-1).tolist() == expected


def test_onehot_gate_rejects_out_of_range_expert_id():
    try:
        build_onehot_gate([0, 4], num_moe=4)
        raise AssertionError("out-of-range expert id was not rejected")
    except ValueError:
        pass
    try:
        build_onehot_gate([-1, 0], num_moe=4)
        raise AssertionError("negative expert id was not rejected")
    except ValueError:
        pass


# --- 1c. routing_utils: gate diagnostics (NaN/Inf/sum validation) --------------------------

def test_gate_diagnostics_clean():
    gate = torch.softmax(torch.randn(10, 1, 4), dim=-1)
    diag = gate_diagnostics(gate)
    assert diag["sum_ok"], diag
    assert not diag["has_nan"] and not diag["has_inf"], diag
    assert diag["num_rows_violating"] == 0, diag


def test_gate_diagnostics_catches_bad_sum():
    gate = torch.full((3, 1, 4), 0.5)
    diag = gate_diagnostics(gate)
    assert not diag["sum_ok"], diag
    assert diag["num_rows_violating"] == 3, diag


def test_gate_diagnostics_catches_nan_and_inf():
    gate = torch.tensor([[[0.25, 0.25, 0.25, 0.25]], [[float("nan"), 0.3, 0.3, 0.4]]])
    diag = gate_diagnostics(gate)
    assert diag["has_nan"], diag

    gate2 = torch.tensor([[[float("inf"), 0.0, 0.0, 0.0]]])
    diag2 = gate_diagnostics(gate2)
    assert diag2["has_inf"], diag2


# --- 1d. routing_utils: assignment CSV round trip -------------------------------------------

def test_assignment_csv_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test_assignments.csv")
        with open(path, "w") as f:
            f.write("sample_id,cluster_id\n0,2\n1,0\n5,3\n")
        assignments = load_assignment_csv(path)
        assert assignments == {0: 2, 1: 0, 5: 3}, assignments


def test_assignment_csv_missing_file_raises():
    try:
        load_assignment_csv(os.path.join(tempfile.gettempdir(), "definitely_missing_12345.csv"))
        raise AssertionError("missing assignment file was not rejected")
    except FileNotFoundError:
        pass


def test_assignment_csv_bad_header_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.csv")
        with open(path, "w") as f:
            f.write("id,cluster\n0,1\n")
        try:
            load_assignment_csv(path)
            raise AssertionError("bad header was not rejected")
        except ValueError:
            pass


# --- 2. moelora tensor contract, reproduced without importing the transformers-dependent module

def moelora_style_forward(x, lora_A_weight, lora_B_weight, gate, num_moe, scaling):
    """Reproduces model/peft/tuners/moelora.py Linear.forward + calculate_B (lines 750-799)
    exactly: A_out = lora_A(x).reshape(batch, seq, num_moe, r_per_expert); B_out via einsum
    against a reshaped lora_B weight; weighted sum over the expert axis using `gate`."""
    batch_size, seq_len, _ = x.size()
    A_out = (x @ lora_A_weight.T).reshape(batch_size, seq_len, num_moe, -1)
    r_per_expert = A_out.shape[-1]
    weight = lora_B_weight.t().reshape(num_moe, r_per_expert, -1)
    B_out = torch.einsum('ijkl, klm->ijkm', A_out, weight)
    Gate = gate.unsqueeze(-1)  # moelora.py:780 / the analogous line for a one-hot gate
    return (B_out * Gate).sum(dim=-2) * scaling


def test_moelora_contract_dynamic_shaped_gate():
    torch.manual_seed(0)
    in_features, out_features, r, num_moe = 16, 16, 8, 4
    x = torch.randn(2, 5, in_features)
    lora_A = torch.randn(r, in_features)
    lora_B = torch.randn(out_features, r)
    gate = torch.softmax(torch.randn(2, 1, num_moe), dim=-1)  # shape produced by NLPRecommendationRouter
    out = moelora_style_forward(x, lora_A, lora_B, gate, num_moe, scaling=1.0)
    assert out.shape == (2, 5, out_features), out.shape


def test_moelora_contract_onehot_gate_isolates_one_expert():
    torch.manual_seed(1)
    in_features, out_features, r, num_moe = 16, 16, 8, 4
    x = torch.randn(3, 4, in_features)
    lora_A = torch.randn(r, in_features)
    lora_B = torch.randn(out_features, r)

    onehot = build_onehot_gate([0, 2, 3], num_moe=num_moe)
    out_onehot = moelora_style_forward(x, lora_A, lora_B, onehot, num_moe, scaling=1.0)

    # Manually compute what "only expert k active" should look like for sample 0 (expert 0)
    # and cross-check against a dynamic gate that puts all its mass on the same expert.
    forced = torch.zeros(3, 1, num_moe)
    forced[0, 0, 0] = 1.0
    forced[1, 0, 2] = 1.0
    forced[2, 0, 3] = 1.0
    out_forced = moelora_style_forward(x, lora_A, lora_B, forced, num_moe, scaling=1.0)

    assert torch.allclose(out_onehot, out_forced, atol=1e-6), \
        "one-hot gate from build_onehot_gate did not isolate the intended expert"

    assert out_onehot.shape == (3, 4, out_features), out_onehot.shape


# --- 3. KMeans artifact save/load (synthetic data) ------------------------------------------

def test_kmeans_save_load_round_trip():
    from sklearn.cluster import KMeans
    import joblib

    rng = np.random.RandomState(0)
    X = np.concatenate([
        rng.normal(loc=[0, 0], scale=0.1, size=(20, 2)),
        rng.normal(loc=[5, 5], scale=0.1, size=(20, 2)),
        rng.normal(loc=[0, 5], scale=0.1, size=(20, 2)),
        rng.normal(loc=[5, 0], scale=0.1, size=(20, 2)),
    ])
    kmeans = KMeans(n_clusters=4, random_state=1234, n_init=10).fit(X)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "kmeans_k4_seed1234.joblib")
        joblib.dump(kmeans, path)
        reloaded = joblib.load(path)
        assert np.array_equal(reloaded.predict(X), kmeans.predict(X))
        assert reloaded.n_clusters == 4


def main():
    check("parse_cluster_expert_mapping: valid", test_parse_mapping_valid)
    check("parse_cluster_expert_mapping: reordered keys", test_parse_mapping_reordered)
    check("parse_cluster_expert_mapping: rejects duplicate cluster id", test_parse_mapping_rejects_duplicate)
    check("parse_cluster_expert_mapping: rejects malformed input", test_parse_mapping_rejects_non_int)
    check("validate_expert_mapping: rejects out-of-range expert id", test_validate_mapping_rejects_out_of_range_expert)
    check("validate_expert_mapping: rejects out-of-range cluster id", test_validate_mapping_rejects_out_of_range_cluster)
    check("build_onehot_gate: shape + one-hot property", test_onehot_gate_shape_and_property)
    check("build_onehot_gate: rejects out-of-range expert id", test_onehot_gate_rejects_out_of_range_expert_id)
    check("gate_diagnostics: clean gate passes", test_gate_diagnostics_clean)
    check("gate_diagnostics: catches bad row sum", test_gate_diagnostics_catches_bad_sum)
    check("gate_diagnostics: catches NaN/Inf", test_gate_diagnostics_catches_nan_and_inf)
    check("load_assignment_csv: round trip", test_assignment_csv_round_trip)
    check("load_assignment_csv: missing file raises", test_assignment_csv_missing_file_raises)
    check("load_assignment_csv: bad header raises", test_assignment_csv_bad_header_raises)
    check("moelora tensor contract: dynamic-shaped gate", test_moelora_contract_dynamic_shaped_gate)
    check("moelora tensor contract: one-hot gate isolates one expert", test_moelora_contract_onehot_gate_isolates_one_expert)
    check("KMeans artifact save/load round trip", test_kmeans_save_load_round_trip)

    print()
    print("VESSL_REQUIRED (not testable locally, no GPU / no Llama-2 weights / no transformers install):")
    print("  - actual model.peft.tuners.moelora.Linear.forward through a real MoeLoraModel")
    print("  - MInterface end-to-end forward()/generate() (needs LlamaForCausalLM)")
    print("  - --routing_mode cluster_hard full training run")
    print("  - gates_test.csv / gates_validation.csv produced by a real evaluation run")
    print()

    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("All local checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
