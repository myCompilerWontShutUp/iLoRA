"""Generate results/today_summary.md from whatever real result files exist on disk.

Spec section 19: this script never invents a number. Any section whose backing file is
missing is written as NOT_RUN, not estimated or copied from the paper.
"""
import argparse
import json
import os

NOT_RUN = "NOT_RUN"


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_text(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def fmt(value, digits=4):
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def section_setup(baseline_metrics):
    lines = ["## 1. Experiment setup", ""]
    if baseline_metrics is None:
        lines.append(NOT_RUN)
        return lines
    for key in ["git_commit_hash", "seed", "dataset", "num_moe", "lora_r", "epochs",
                "learning_rate", "batch_size", "accumulate_grad_batches", "candidate_count",
                "llm_path_basename", "gpu_model", "torch_version", "transformers_version",
                "peft_version", "pytorch_lightning_version"]:
        lines.append(f"- {key}: {baseline_metrics.get(key, NOT_RUN)}")
    return lines


def section_baseline(baseline_metrics):
    lines = ["## 2. Baseline reproduction (iLoRA Dynamic)", ""]
    if baseline_metrics is None:
        lines.append(NOT_RUN)
        return lines
    lines.append(f"- ValidRatio (test_prediction_valid): {fmt(baseline_metrics.get('test_prediction_valid', NOT_RUN))}")
    lines.append(f"- HR@1 (test_hr): {fmt(baseline_metrics.get('test_hr', NOT_RUN))}")
    lines.append(f"- CombinedMetric (metric): {fmt(baseline_metrics.get('metric', NOT_RUN))}")
    lines.append(f"- TrainableParams: {baseline_metrics.get('trainable_params', NOT_RUN)}")
    lines.append(f"- ElapsedTime (seconds): {fmt(baseline_metrics.get('elapsed_seconds', NOT_RUN))}")
    return lines


def section_gate_diversity(gate_summary, interpretation_text):
    lines = ["## 3. Gate diversity (dynamic routing, test split)", ""]
    if gate_summary is None:
        lines.append(NOT_RUN)
        return lines
    lines.append(f"- n_samples: {gate_summary.get('n_samples', NOT_RUN)}")
    lines.append(f"- mean_expert_utilization: {gate_summary.get('mean_expert_utilization', NOT_RUN)}")
    lines.append(f"- argmax_expert_ratio: {gate_summary.get('argmax_expert_ratio', NOT_RUN)}")
    ent = gate_summary.get("entropy", {})
    lines.append(f"- entropy mean/std: {fmt(ent.get('mean', NOT_RUN))} / {fmt(ent.get('std', NOT_RUN))}")
    js = gate_summary.get("js_divergence_from_global_mean", {})
    lines.append(f"- JS divergence from global mean, mean/std: {fmt(js.get('mean', NOT_RUN))} / {fmt(js.get('std', NOT_RUN))}")
    if interpretation_text:
        lines.append("")
        lines.append("Qualitative reading (from gate_interpretation.txt):")
        lines.append("")
        lines.append("```")
        lines.append(interpretation_text.strip())
        lines.append("```")
    return lines


def section_clustering(cluster_summary):
    lines = ["## 4. SASRec clustering", ""]
    if cluster_summary is None:
        lines.append(NOT_RUN)
        return lines
    lines.append(f"- n_clusters: {cluster_summary.get('n_clusters', NOT_RUN)}")
    lines.append(f"- train_cluster_size: {cluster_summary.get('train_cluster_size', NOT_RUN)}")
    lines.append(f"- train_cluster_ratio: {cluster_summary.get('train_cluster_ratio', NOT_RUN)}")
    lines.append(f"- validation_cluster_size: {cluster_summary.get('validation_cluster_size', NOT_RUN)}")
    lines.append(f"- test_cluster_size: {cluster_summary.get('test_cluster_size', NOT_RUN)}")
    lines.append(f"- silhouette_score_train: {fmt(cluster_summary.get('silhouette_score_train', NOT_RUN))}")
    return lines


def section_cluster_hard(cluster_hard_metrics):
    lines = ["## 5. Cluster-hard routing", ""]
    if cluster_hard_metrics is None:
        lines.append(NOT_RUN)
        return lines
    lines.append(f"- ValidRatio (test_prediction_valid): {fmt(cluster_hard_metrics.get('test_prediction_valid', NOT_RUN))}")
    lines.append(f"- HR@1 (test_hr): {fmt(cluster_hard_metrics.get('test_hr', NOT_RUN))}")
    lines.append(f"- CombinedMetric (metric): {fmt(cluster_hard_metrics.get('metric', NOT_RUN))}")
    lines.append(f"- TrainableParams: {cluster_hard_metrics.get('trainable_params', NOT_RUN)}")
    lines.append(f"- ElapsedTime (seconds): {fmt(cluster_hard_metrics.get('elapsed_seconds', NOT_RUN))}")
    return lines


def section_comparison(comparison_md):
    lines = ["## 6. Performance comparison", ""]
    if comparison_md is None:
        lines.append(NOT_RUN)
        return lines
    lines.append(comparison_md.strip())
    return lines


def section_limitations():
    return [
        "## 7. Limitations",
        "",
        "- With lora_r=8 and num_moe=4, each expert has effective rank 2 "
        "(experiments/CODEBASE_ANALYSIS.md section T/U). Cluster-hard activates exactly one "
        "rank-2 expert per sample while dynamic mixes all four rank-2 experts with a soft gate, "
        "so any performance gap between the two conflates routing strategy with active-capacity "
        "(spec section 13). This was not worked around in the primary experiment.",
        "- Cluster assignment for cluster_hard is fixed and precomputed (never recomputed inside "
        "the training loop), per experiments/IMPLEMENTATION_NOTES.md section 4.",
        "- The paper-Appendix-vs-released-script hyperparameter comparison (spec section 1.V) "
        "was not verified in this pass (VESSL_REQUIRED / no network fetch performed); the "
        "released `train_movielens.sh` / `test_movielens.sh` values were used regardless.",
        "- silhouette_score on the train split is computed on a random subsample "
        "(see analysis/cluster_sasrec.py --silhouette_sample_size) because full pairwise "
        "silhouette at the MovieLens train-set scale is computationally intractable.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_results_dir", default="results/baseline")
    parser.add_argument("--cluster_hard_results_dir", default="results/cluster_hard")
    parser.add_argument("--clustering_dir", default="results/clustering")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    baseline_metrics = read_json(os.path.join(args.baseline_results_dir, "metrics.json"))
    cluster_hard_metrics = read_json(os.path.join(args.cluster_hard_results_dir, "metrics.json"))
    gate_summary = read_json(os.path.join(args.baseline_results_dir, "gate_summary.json"))
    interpretation_text = read_text(os.path.join(args.baseline_results_dir, "gate_interpretation.txt"))
    cluster_summary = read_json(os.path.join(args.clustering_dir, "cluster_summary.json"))
    comparison_md = read_text(os.path.join(args.results_dir, "final_comparison.md"))

    lines = ["# Today's experiment summary", "",
              "Generated by analysis/generate_summary.py from real result files only; "
              "any section backed by a missing file reads NOT_RUN.", ""]
    lines += section_setup(baseline_metrics) + [""]
    lines += section_baseline(baseline_metrics) + [""]
    lines += section_gate_diversity(gate_summary, interpretation_text) + [""]
    lines += section_clustering(cluster_summary) + [""]
    lines += section_cluster_hard(cluster_hard_metrics) + [""]
    lines += section_comparison(comparison_md) + [""]
    lines += section_limitations()

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "today_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[generate_summary] wrote {out_path}")


if __name__ == "__main__":
    main()
