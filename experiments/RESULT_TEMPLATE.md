# Result template

This is the structure `analysis/generate_summary.py` writes to `results/today_summary.md`,
shown here with every value as `NOT_RUN` because no VESSL run has produced real numbers yet.
This file is not meant to be filled in by hand — it exists so the expected shape of the final
write-up is reviewable before the actual run, and so nobody mistakes a missing section in a real
`today_summary.md` for a bug rather than an unrun stage. Do not copy paper numbers into this file
or into `today_summary.md`; every value must come from an actual file on disk (spec section 0.11
/ section 19).

```
# Today's experiment summary

## 1. Experiment setup
- git_commit_hash, seed, dataset, num_moe, lora_r, epochs, learning_rate, batch_size,
  accumulate_grad_batches, candidate_count, llm_path_basename, gpu_model, torch_version,
  transformers_version, peft_version, pytorch_lightning_version
  <- results/baseline/metrics.json

## 2. Baseline reproduction (iLoRA Dynamic)
- ValidRatio, HR@1, CombinedMetric, TrainableParams, ElapsedTime
  <- results/baseline/metrics.json

## 3. Gate diversity (dynamic routing, test split)
- n_samples, mean_expert_utilization, argmax_expert_ratio, entropy mean/std,
  JS divergence from global mean mean/std, qualitative reading
  <- results/baseline/gate_summary.json, gate_interpretation.txt

## 4. SASRec clustering
- n_clusters, train/validation/test cluster size, train cluster ratio, silhouette_score_train
  <- results/clustering/cluster_summary.json

## 5. Cluster-hard routing
- ValidRatio, HR@1, CombinedMetric, TrainableParams, ElapsedTime
  <- results/cluster_hard/metrics.json

## 6. Performance comparison
- the full iLoRA Dynamic vs. Cluster Hard table
  <- results/final_comparison.md

## 7. Limitations
- the rank-splitting confounder (lora_r=8, num_moe=4 -> effective rank 2 per expert)
- fixed vs. recomputed cluster assignment rationale
- unverified paper-vs-released-script hyperparameter comparison
- silhouette subsampling
```

Run `bash run_all_vessl.sh` on the VESSL instance to produce the real `results/today_summary.md`.
