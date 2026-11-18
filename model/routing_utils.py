"""Pure-Python/torch helpers for cluster-hard routing and gate diagnostics.

Deliberately free of any dependency on LlamaForCausalLM / the PEFT wrapper / pytorch_lightning,
so this module can be imported and unit-tested (scripts/smoke_test_routing.py) on a machine that
has no GPU and no Llama-2 weights.
"""
import csv
import os

import torch


def parse_cluster_expert_mapping(mapping_str):
    """Parse "0:0,1:1,2:2,3:3" into {0:0, 1:1, 2:2, 3:3}.

    Raises ValueError on any malformed entry, non-integer key/value, or duplicate cluster id.
    """
    if not isinstance(mapping_str, str) or not mapping_str.strip():
        raise ValueError(f"cluster_expert_mapping must be a non-empty string, got {mapping_str!r}")

    mapping = {}
    for part in mapping_str.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"cluster_expert_mapping has an empty entry in {mapping_str!r}")
        if ":" not in part:
            raise ValueError(f"cluster_expert_mapping entry {part!r} is missing ':' (expected cluster:expert)")
        cluster_str, expert_str = part.split(":", 1)
        try:
            cluster_id = int(cluster_str.strip())
            expert_id = int(expert_str.strip())
        except ValueError as exc:
            raise ValueError(f"cluster_expert_mapping entry {part!r} is not 'int:int'") from exc
        if cluster_id in mapping:
            raise ValueError(f"cluster_expert_mapping has duplicate cluster id {cluster_id}")
        mapping[cluster_id] = expert_id
    return mapping


def validate_expert_mapping(mapping, num_clusters, num_moe):
    """Raise ValueError if any cluster id or expert id in `mapping` is out of range."""
    for cluster_id, expert_id in mapping.items():
        if not (0 <= cluster_id < num_clusters):
            raise ValueError(f"cluster id {cluster_id} in mapping is outside [0, {num_clusters})")
        if not (0 <= expert_id < num_moe):
            raise ValueError(f"expert id {expert_id} in mapping is outside [0, {num_moe})")


def build_onehot_gate(expert_ids, num_moe, device=None, dtype=None):
    """Build a one-hot gate tensor of shape [batch, 1, num_moe] from a list/tensor of expert ids.

    Mirrors the shape produced by NLPRecommendationRouter.forward (model/router/nlpr.py:80),
    which ends with `out.unsqueeze(1)` -> [batch, 1, num_experts].
    """
    if not torch.is_tensor(expert_ids):
        expert_ids = torch.tensor(list(expert_ids), dtype=torch.long)
    expert_ids = expert_ids.to(dtype=torch.long)

    if expert_ids.numel() > 0:
        min_id = int(expert_ids.min())
        max_id = int(expert_ids.max())
        if min_id < 0 or max_id >= num_moe:
            raise ValueError(f"expert id out of range [0, {num_moe}): min={min_id}, max={max_id}")

    batch_size = expert_ids.shape[0]
    gate = torch.zeros((batch_size, num_moe), dtype=dtype or torch.float32, device=device)
    gate.scatter_(1, expert_ids.to(device=device).unsqueeze(1), 1.0)
    return gate.unsqueeze(1)


def load_assignment_csv(path):
    """Load a {sample_id -> cluster_id} lookup table from a CSV with columns sample_id,cluster_id."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cluster assignment file not found: {path}. "
            "Run scripts/fit_kmeans.py before using --routing_mode cluster_hard."
        )
    assignments = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames or "cluster_id" not in reader.fieldnames:
            raise ValueError(f"{path} must have columns 'sample_id' and 'cluster_id', found {reader.fieldnames}")
        for row in reader:
            sample_id = int(row["sample_id"])
            if sample_id in assignments:
                raise ValueError(f"Duplicate sample_id {sample_id} in {path}")
            assignments[sample_id] = int(row["cluster_id"])
    if not assignments:
        raise ValueError(f"{path} contains no assignment rows")
    return assignments


def gate_diagnostics(gate, atol=1e-3):
    """Return a dict of validation stats for a [batch, ..., num_experts] gate tensor.

    Checks: per-row probabilities sum to ~1, no NaN, no Inf. Never raises; callers decide
    whether a violation is fatal.
    """
    gate = gate.detach()
    row_sums = gate.sum(dim=-1)
    has_nan = bool(torch.isnan(gate).any())
    has_inf = bool(torch.isinf(gate).any())
    sum_ok = bool(torch.all(torch.abs(row_sums - 1.0) <= atol)) if not (has_nan or has_inf) else False
    return {
        "sum_ok": sum_ok,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min_sum": float(row_sums.min()) if row_sums.numel() > 0 else None,
        "max_sum": float(row_sums.max()) if row_sums.numel() > 0 else None,
        "num_rows_violating": int((torch.abs(row_sums - 1.0) > atol).sum()) if not (has_nan or has_inf) else int(gate.shape[0]),
    }
