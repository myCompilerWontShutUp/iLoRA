"""Fit KMeans(k=4) on TRAIN SASRec representations only, then predict cluster ids for
validation/test. Never fits on validation or test data (spec section 0.8 / 10).

Inputs: representations_{train,validation,test}.npz produced by extract_representations.py.
Outputs:
    <artifacts_dir>/kmeans_k4_seed1234.joblib
    <output_dir>/train_assignments.csv
    <output_dir>/validation_assignments.csv
    <output_dir>/test_assignments.csv

Each assignments CSV has columns: sample_id,cluster_id.

PCA is not used here: this script clusters the raw SASRec representation, never a PCA
projection, per spec section 7 ("PCA는 시각화 용도로만 사용한다").
"""
import argparse
import csv
import os

import numpy as np
from sklearn.cluster import KMeans
import joblib


def load_representations(results_dir, split):
    path = os.path.join(results_dir, f"representations_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/extract_representations.py before scripts/fit_kmeans.py."
        )
    data = np.load(path)
    return data["sample_id"], data["sasrec"]


def write_assignments_csv(path, sample_ids, cluster_ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "cluster_id"])
        for sid, cid in zip(sample_ids, cluster_ids):
            writer.writerow([int(sid), int(cid)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/baseline",
                         help="Directory containing representations_{split}.npz from extract_representations.py")
    parser.add_argument("--output_dir", default="results/clustering")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_init", type=int, default=10,
                         help="Explicit n_init so behavior does not depend on sklearn's version default.")
    args = parser.parse_args()

    train_ids, train_reps = load_representations(args.results_dir, "train")
    val_ids, val_reps = load_representations(args.results_dir, "validation")
    test_ids, test_reps = load_representations(args.results_dir, "test")

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=args.n_init)
    train_clusters = kmeans.fit_predict(train_reps)  # fit uses TRAIN ONLY
    val_clusters = kmeans.predict(val_reps)
    test_clusters = kmeans.predict(test_reps)

    os.makedirs(args.artifacts_dir, exist_ok=True)
    artifact_path = os.path.join(args.artifacts_dir, f"kmeans_k{args.n_clusters}_seed{args.seed}.joblib")
    joblib.dump(kmeans, artifact_path)

    write_assignments_csv(os.path.join(args.output_dir, "train_assignments.csv"), train_ids, train_clusters)
    write_assignments_csv(os.path.join(args.output_dir, "validation_assignments.csv"), val_ids, val_clusters)
    write_assignments_csv(os.path.join(args.output_dir, "test_assignments.csv"), test_ids, test_clusters)

    print(f"[fit_kmeans] fit on {len(train_ids)} train samples, n_clusters={args.n_clusters}, seed={args.seed}")
    print(f"[fit_kmeans] artifact -> {artifact_path}")
    print(f"[fit_kmeans] assignments -> {args.output_dir}/{{train,validation,test}}_assignments.csv")
    counts = np.bincount(train_clusters, minlength=args.n_clusters)
    print(f"[fit_kmeans] train cluster sizes: {counts.tolist()}")


if __name__ == "__main__":
    main()
