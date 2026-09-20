"""Clustering analysis / visualization for the SASRec-representation KMeans clusters
(spec section 7/8). This script does not fit KMeans (that is scripts/fit_kmeans.py) — it reads
the artifacts fit_kmeans.py already produced and cross-references them against the dynamic
baseline's gate export for interpretation.

Inputs:
  <clustering_dir>/{train,validation,test}_assignments.csv   (scripts/fit_kmeans.py)
  <representations_dir>/representations_{train,validation,test}.npz  (scripts/extract_representations.py)
  <artifacts_dir>/kmeans_k<K>_seed<S>.joblib                  (scripts/fit_kmeans.py)
  <baseline_results_dir>/gates_test.csv                       (MInterface, --export_gates)

Outputs (into <clustering_dir>):
  cluster_summary.json
  figures/sasrec_pca_clusters.png     (PCA is visualization-only, never the clustering input)
  figures/cluster_dynamic_gate_heatmap.png
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
})


def load_assignments(clustering_dir, split):
    path = os.path.join(clustering_dir, f"{split}_assignments.csv")
    return pd.read_csv(path)


def load_representations(representations_dir, split):
    path = os.path.join(representations_dir, f"representations_{split}.npz")
    data = np.load(path)
    return pd.DataFrame({"sample_id": data["sample_id"]}), data["sasrec"]


def find_kmeans_artifact(artifacts_dir):
    candidates = sorted(glob.glob(os.path.join(artifacts_dir, "kmeans_k*_seed*.joblib")))
    if not candidates:
        raise FileNotFoundError(f"No kmeans_k*_seed*.joblib found under {artifacts_dir}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clustering_dir", default="results/clustering")
    parser.add_argument("--representations_dir", default="results/baseline")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--baseline_results_dir", default="results/baseline")
    parser.add_argument("--silhouette_sample_size", type=int, default=5000,
                         help="Subsample size for silhouette_score on the train split "
                              "(full pairwise silhouette is O(n^2) and intractable at MovieLens "
                              "train-set scale). Documented in experiments/IMPLEMENTATION_NOTES.md.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    kmeans_path = find_kmeans_artifact(args.artifacts_dir)
    kmeans = joblib.load(kmeans_path)

    train_assign = load_assignments(args.clustering_dir, "train")
    val_assign = load_assignments(args.clustering_dir, "validation")
    test_assign = load_assignments(args.clustering_dir, "test")

    _, train_reps = load_representations(args.representations_dir, "train")

    n_clusters = kmeans.n_clusters
    train_sizes = train_assign["cluster_id"].value_counts().reindex(range(n_clusters), fill_value=0)
    val_sizes = val_assign["cluster_id"].value_counts().reindex(range(n_clusters), fill_value=0)
    test_sizes = test_assign["cluster_id"].value_counts().reindex(range(n_clusters), fill_value=0)

    n_train = len(train_assign)
    sample_size = min(args.silhouette_sample_size, n_train)
    if n_train >= 2 and train_reps.shape[0] == n_train:
        sil = float(silhouette_score(train_reps, train_assign["cluster_id"].to_numpy(),
                                      sample_size=sample_size, random_state=args.seed))
    else:
        sil = None

    summary = {
        "kmeans_artifact": os.path.basename(kmeans_path),
        "n_clusters": int(n_clusters),
        "train_cluster_size": train_sizes.tolist(),
        "train_cluster_ratio": (train_sizes / max(n_train, 1)).tolist(),
        "validation_cluster_size": val_sizes.tolist(),
        "test_cluster_size": test_sizes.tolist(),
        "silhouette_score_train": sil,
        "silhouette_sample_size": sample_size,
    }

    gates_test_path = os.path.join(args.baseline_results_dir, "gates_test.csv")
    if os.path.exists(gates_test_path):
        gates_df = pd.read_csv(gates_test_path)
        gate_cols = sorted([c for c in gates_df.columns if c.startswith("gate_")],
                            key=lambda c: int(c.split("_")[1]))
        merged = test_assign.merge(gates_df[["sample_id"] + gate_cols + ["argmax_expert"]],
                                    on="sample_id", how="inner")
        cluster_gate_mean = merged.groupby("cluster_id")[gate_cols].mean().reindex(range(n_clusters))
        summary["cluster_dynamic_gate_mean"] = cluster_gate_mean.fillna(0).values.tolist()

        contingency = pd.crosstab(merged["cluster_id"], merged["argmax_expert"])
        contingency = contingency.reindex(index=range(n_clusters), columns=range(len(gate_cols)), fill_value=0)
        summary["cluster_vs_dynamic_argmax_expert_contingency"] = contingency.values.tolist()
        summary["cluster_gate_merge_n_samples"] = int(len(merged))
    else:
        summary["cluster_dynamic_gate_mean"] = "NOT_RUN (gates_test.csv not found)"
        summary["cluster_vs_dynamic_argmax_expert_contingency"] = "NOT_RUN (gates_test.csv not found)"
        cluster_gate_mean = None

    os.makedirs(args.clustering_dir, exist_ok=True)
    with open(os.path.join(args.clustering_dir, "cluster_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig_dir = os.path.join(args.clustering_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    pca = PCA(n_components=2, random_state=args.seed)
    coords = pca.fit_transform(train_reps)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cluster_ids = train_assign["cluster_id"].to_numpy()
    max_points = 5000
    if len(coords) > max_points:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords_plot, clusters_plot = coords[idx], cluster_ids[idx]
    else:
        coords_plot, clusters_plot = coords, cluster_ids
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for c in range(n_clusters):
        mask = clusters_plot == c
        ax.scatter(coords_plot[mask, 0], coords_plot[mask, 1], s=6, alpha=0.5,
                   color=f"C{c}", marker=markers[c % len(markers)], label=f"Cluster {c}")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("SASRec representation clusters (PCA projection, visualization only)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sasrec_pca_clusters.png"))
    plt.close(fig)

    if cluster_gate_mean is not None:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        data = cluster_gate_mean.fillna(0).to_numpy()
        im = ax.imshow(data, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([f"Expert {i+1}" for i in range(data.shape[1])])
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([f"Cluster {i}" for i in range(data.shape[0])])
        ax.set_title("Mean dynamic gate per cluster (test split)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean gate weight")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "cluster_dynamic_gate_heatmap.png"))
        plt.close(fig)
    else:
        print("[cluster_sasrec] skipping cluster_dynamic_gate_heatmap.png: gates_test.csv not found")

    print(f"[cluster_sasrec] wrote cluster_summary.json to {args.clustering_dir}")
    print(f"[cluster_sasrec] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
