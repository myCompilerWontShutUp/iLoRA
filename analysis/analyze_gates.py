"""Gate diversity analysis for the iLoRA dynamic-routing baseline (spec section 5/6).

Reads results/<results_dir>/gates_<split>.csv (produced by MInterface._flush_gate_export when
run with --export_gates) and computes:
  1. mean expert utilization (mean gate_k per expert)
  2. argmax expert utilization (count + ratio per expert)
  3. per-sample normalized entropy H(p) = -sum(p_i log p_i) / log(K)  (mean/std/median/min/max)
  4. expert-wise gate standard deviation across samples
  5. Jensen-Shannon divergence of each sample's gate from the global mean gate
     (mean/std/median/min/max)
  6. one supplementary dispersion statistic: top1-minus-top2 gate margin (mean/std)
  7. a conservative, non-thresholded collapse-diagnostic write-up

Outputs (into --results_dir):
  gate_summary.json, gate_summary.csv, gate_interpretation.txt,
  figures/gate_heatmap.png, figures/expert_mean_utilization.png,
  figures/argmax_expert_distribution.png, figures/entropy_distribution.png

No numeric threshold anywhere in this file declares routing "collapsed" or "healthy" — spec
section 5 explicitly forbids an arbitrary boolean verdict; the interpretation file only reports
numbers plus conservative qualitative wording.
"""
import argparse
import json
import os

import sys

import numpy as np
import pandas as pd
import matplotlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from model.routing_utils import gate_diagnostics  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.style": "normal",
    "font.weight": "normal",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
})


def gate_columns(df):
    cols = sorted([c for c in df.columns if c.startswith("gate_")],
                  key=lambda c: int(c.split("_")[1]))
    return cols


def normalized_entropy(probs):
    k = probs.shape[1]
    eps = 1e-12
    h = -(probs * np.log(probs + eps)).sum(axis=1)
    return h / np.log(k)


def js_divergence(p, q):
    """Jensen-Shannon divergence (base e, range [0, ln 2]) between each row of p and the fixed
    vector q. `m` depends on the row (m_i = 0.5*(p_i+q)), so both KL terms must be summed per
    row (axis=1), not over the whole matrix."""
    eps = 1e-12
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(p / m)).sum(axis=1)
    kl_qm = (q * np.log(q / m)).sum(axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_gate_summary(df, gate_cols):
    gates = df[gate_cols].to_numpy(dtype=np.float64)
    n_samples, num_experts = gates.shape

    mean_utilization = gates.mean(axis=0)
    expert_std = gates.std(axis=0)

    argmax = df["argmax_expert"].to_numpy()
    argmax_counts = np.bincount(argmax, minlength=num_experts)
    argmax_ratio = argmax_counts / n_samples

    entropy = normalized_entropy(gates)

    global_mean_gate = mean_utilization
    js = js_divergence(gates, global_mean_gate)

    sorted_gates = np.sort(gates, axis=1)[:, ::-1]
    top1_minus_top2 = sorted_gates[:, 0] - sorted_gates[:, 1]

    summary = {
        "n_samples": int(n_samples),
        "num_experts": int(num_experts),
        "mean_expert_utilization": mean_utilization.tolist(),
        "expert_gate_std": expert_std.tolist(),
        "argmax_expert_count": argmax_counts.tolist(),
        "argmax_expert_ratio": argmax_ratio.tolist(),
        "entropy": {
            "mean": float(entropy.mean()), "std": float(entropy.std()),
            "median": float(np.median(entropy)), "min": float(entropy.min()), "max": float(entropy.max()),
        },
        "js_divergence_from_global_mean": {
            "mean": float(js.mean()), "std": float(js.std()),
            "median": float(np.median(js)), "min": float(js.min()), "max": float(js.max()),
        },
        "top1_minus_top2_margin": {
            "mean": float(top1_minus_top2.mean()), "std": float(top1_minus_top2.std()),
        },
    }

    diag = gate_diagnostics(torch.from_numpy(gates))
    summary["num_gate_rows_failing_validation"] = diag["num_rows_violating"]
    summary["gate_has_nan"] = diag["has_nan"]
    summary["gate_has_inf"] = diag["has_inf"]

    return summary, gates, entropy, argmax


def write_interpretation(summary, path):
    lines = []
    lines.append("Gate diversity report (numbers only; no fixed collapse threshold applied).")
    lines.append("")
    lines.append(f"n_samples = {summary['n_samples']}, num_experts = {summary['num_experts']}")
    lines.append(f"mean_expert_utilization = {['%.4f' % v for v in summary['mean_expert_utilization']]}")
    lines.append(f"argmax_expert_ratio = {['%.4f' % v for v in summary['argmax_expert_ratio']]}")
    lines.append(f"entropy mean/std = {summary['entropy']['mean']:.4f} / {summary['entropy']['std']:.4f}")
    lines.append(f"JS divergence from global mean, mean/std = "
                 f"{summary['js_divergence_from_global_mean']['mean']:.4f} / "
                 f"{summary['js_divergence_from_global_mean']['std']:.4f}")
    lines.append(f"top1-minus-top2 margin mean/std = "
                 f"{summary['top1_minus_top2_margin']['mean']:.4f} / "
                 f"{summary['top1_minus_top2_margin']['std']:.4f}")
    lines.append("")
    lines.append("Qualitative reading (conservative, not a threshold-based verdict):")
    ent_mean = summary["entropy"]["mean"]
    util_spread = float(np.max(summary["mean_expert_utilization"]) - np.min(summary["mean_expert_utilization"]))
    if ent_mean > 0.9 and util_spread < 0.1:
        lines.append("- Global expert utilization is close to uniform and per-sample entropy is high on "
                      "average, consistent with Case A in the spec (near-uniform gates, low apparent "
                      "instance-wise personalization) or Case C with high overlap; JS-divergence spread "
                      "above should be checked before concluding either way.")
    elif ent_mean < 0.5:
        lines.append("- Per-sample entropy is low on average, consistent with gates concentrating on "
                      "few experts per sample; whether this reflects one dominant expert globally (Case B) "
                      "or diverse per-sample concentration on different experts (Case C) should be read "
                      "off mean_expert_utilization and argmax_expert_ratio above, not off entropy alone.")
    else:
        lines.append("- Entropy and utilization values sit between the extremes described in the spec's "
                      "Case A/B/C; read the per-expert utilization, argmax ratio, and JS-divergence spread "
                      "above directly rather than inferring a single verdict from this text.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_gate_heatmap(gates, argmax, out_path, max_rows=500):
    n = gates.shape[0]
    order = np.argsort(argmax, kind="stable")
    gates_sorted = gates[order]
    if n > max_rows:
        idx = np.linspace(0, n - 1, max_rows).astype(int)
        gates_sorted = gates_sorted[idx]

    fig, ax = plt.subplots(figsize=(4.5, 6))
    im = ax.imshow(gates_sorted, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Sample (sorted by argmax expert)")
    ax.set_xticks(range(gates.shape[1]))
    ax.set_xticklabels([f"Expert {i+1}" for i in range(gates.shape[1])])
    ax.set_yticks([])
    ax.set_title("Gate weights per sample")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gate weight")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_expert_mean_utilization(mean_utilization, out_path):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    x = np.arange(len(mean_utilization))
    ax.bar(x, mean_utilization, color="0.4", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Expert {i+1}" for i in x])
    ax.set_ylabel("Mean gate weight")
    ax.set_title("Mean expert utilization")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_argmax_distribution(argmax_counts, out_path):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    x = np.arange(len(argmax_counts))
    ax.bar(x, argmax_counts, color="0.4", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Expert {i+1}" for i in x])
    ax.set_ylabel("Sample count")
    ax.set_title("Argmax expert distribution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_entropy_distribution(entropy, out_path):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.hist(entropy, bins=20, color="0.4", edgecolor="white")
    ax.set_xlabel("Normalized entropy")
    ax.set_ylabel("Sample count")
    ax.set_title("Per-sample gate entropy")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/baseline")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, f"gates_{args.split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run baseline evaluation with --export_gates first.")
    df = pd.read_csv(csv_path)
    gate_cols = gate_columns(df)

    summary, gates, entropy, argmax = compute_gate_summary(df, gate_cols)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    flat_row = {"n_samples": summary["n_samples"], "num_experts": summary["num_experts"]}
    for i, v in enumerate(summary["mean_expert_utilization"]):
        flat_row[f"mean_utilization_expert_{i}"] = v
    for i, v in enumerate(summary["argmax_expert_ratio"]):
        flat_row[f"argmax_ratio_expert_{i}"] = v
    for i, v in enumerate(summary["expert_gate_std"]):
        flat_row[f"expert_gate_std_{i}"] = v
    for k, v in summary["entropy"].items():
        flat_row[f"entropy_{k}"] = v
    for k, v in summary["js_divergence_from_global_mean"].items():
        flat_row[f"js_divergence_{k}"] = v
    for k, v in summary["top1_minus_top2_margin"].items():
        flat_row[f"top1_minus_top2_margin_{k}"] = v
    flat_row["num_gate_rows_failing_validation"] = summary["num_gate_rows_failing_validation"]
    flat_row["gate_has_nan"] = summary["gate_has_nan"]
    flat_row["gate_has_inf"] = summary["gate_has_inf"]
    pd.DataFrame([flat_row]).to_csv(os.path.join(args.results_dir, "gate_summary.csv"), index=False)

    write_interpretation(summary, os.path.join(args.results_dir, "gate_interpretation.txt"))

    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_gate_heatmap(gates, argmax, os.path.join(fig_dir, "gate_heatmap.png"))
    plot_expert_mean_utilization(np.array(summary["mean_expert_utilization"]),
                                  os.path.join(fig_dir, "expert_mean_utilization.png"))
    plot_argmax_distribution(np.array(summary["argmax_expert_count"]),
                              os.path.join(fig_dir, "argmax_expert_distribution.png"))
    plot_entropy_distribution(entropy, os.path.join(fig_dir, "entropy_distribution.png"))

    print(f"[analyze_gates] wrote gate_summary.json / .csv / gate_interpretation.txt to {args.results_dir}")
    print(f"[analyze_gates] wrote 4 figures to {fig_dir}")


if __name__ == "__main__":
    main()
