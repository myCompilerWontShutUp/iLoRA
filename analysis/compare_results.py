"""Build the final baseline-vs-cluster_hard comparison table (spec section 12).

Reads results/baseline/metrics.json and results/cluster_hard/metrics.json (each written by
MInterface._write_metrics_json, see model/model_interface.py) and writes:
  results/final_comparison.csv
  results/final_comparison.md

Column mapping to the actual metric names in this codebase (see
experiments/CODEBASE_ANALYSIS.md section P/Q and experiments/IMPLEMENTATION_NOTES.md section 7):
  ValidRatio      <- metrics.json "test_prediction_valid"
  HR@1            <- metrics.json "test_hr"
  CombinedMetric  <- metrics.json "metric"  (= hr * valid_ratio)

If a metrics.json is missing, the corresponding row is filled with "NOT_RUN" rather than
inventing a number, per spec section 0.11.
"""
import argparse
import json
import os

import pandas as pd

NOT_RUN = "NOT_RUN"


def dataframe_to_markdown(df):
    """Minimal Markdown table writer so this script does not depend on the optional
    `tabulate` package (not in requirements.txt)."""
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator] + rows)

ROWS = [
    ("iLoRA Dynamic", "dynamic", "baseline"),
    ("Cluster Hard", "cluster_hard", "cluster_hard"),
]

COLUMNS = ["Method", "Routing", "ValidRatio", "HR@1", "CombinedMetric", "Seed", "Epoch",
           "num_moe", "lora_r", "TrainableParams", "ElapsedTime"]


def load_metrics(results_dir):
    path = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_row(method_name, routing_label, metrics):
    if metrics is None:
        return {
            "Method": method_name, "Routing": routing_label,
            "ValidRatio": NOT_RUN, "HR@1": NOT_RUN, "CombinedMetric": NOT_RUN,
            "Seed": NOT_RUN, "Epoch": NOT_RUN, "num_moe": NOT_RUN, "lora_r": NOT_RUN,
            "TrainableParams": NOT_RUN, "ElapsedTime": NOT_RUN,
        }
    return {
        "Method": method_name,
        "Routing": routing_label,
        "ValidRatio": metrics.get("test_prediction_valid", NOT_RUN),
        "HR@1": metrics.get("test_hr", NOT_RUN),
        "CombinedMetric": metrics.get("metric", NOT_RUN),
        "Seed": metrics.get("seed", NOT_RUN),
        "Epoch": metrics.get("epochs", NOT_RUN),
        "num_moe": metrics.get("num_moe", NOT_RUN),
        "lora_r": metrics.get("lora_r", NOT_RUN),
        "TrainableParams": metrics.get("trainable_params", NOT_RUN),
        "ElapsedTime": metrics.get("elapsed_seconds", NOT_RUN),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_results_dir", default="results/baseline")
    parser.add_argument("--cluster_hard_results_dir", default="results/cluster_hard")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    baseline_metrics = load_metrics(args.baseline_results_dir)
    cluster_hard_metrics = load_metrics(args.cluster_hard_results_dir)

    rows = [
        build_row("iLoRA Dynamic", "dynamic", baseline_metrics),
        build_row("Cluster Hard", "cluster_hard", cluster_hard_metrics),
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "final_comparison.csv")
    md_path = os.path.join(args.output_dir, "final_comparison.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Final comparison: iLoRA Dynamic vs. Cluster Hard\n\n")
        f.write(dataframe_to_markdown(df))
        f.write("\n")

    print(f"[compare_results] wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
