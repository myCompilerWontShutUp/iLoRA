# VESSL run guide

Assumes a VESSL Workspace with Linux and exactly one A100 80GB GPU has already been created
(this guide does not cover creating that workspace). Run everything below inside it.

```bash
git clone https://github.com/myCompilerWontShutUp/iLoRA.git
cd iLoRA
git checkout exp/gate-cluster
```
Expected: working tree on branch `exp/gate-cluster`, matching what `git status` /
`git branch --show-current` show.

```bash
export LLM_PATH=/actual/path/to/Llama-2-7b-hf
export CUDA_VISIBLE_DEVICES=0
```
Expected: no output. Both scripts below read these two variables.

```bash
bash setup_vessl.sh
```
Expected: dependencies installed from `requirements.txt`, the README's documented
`transformers` debug-file workaround checked (and applied only if actually needed, logged either
way), `checkpoints/`, `outputs/`, `results/{baseline,cluster_hard,clustering}`, `artifacts/`,
`logs/` created, installed package versions printed at the end.

```bash
bash scripts/preflight_vessl.sh
```
Expected:
```
GPU count 1
GPU model + VRAM printed
LLM_PATH exists
MovieLens dataset files present
Preflight passed. No training has started.
```
If GPU count is not exactly 1, or `LLM_PATH`/the dataset is missing, this exits non-zero and
does **not** start training.

```bash
bash run_all_vessl.sh
```
Expected: runs, in order, baseline training → baseline evaluation (with gate export) → SASRec
representation export → gate diversity analysis → KMeans fit (train-only) + validation/test
cluster prediction → clustering analysis → cluster-hard training (from scratch) → cluster-hard
evaluation → final comparison → `results/today_summary.md`. Each stage logs a
`[step/total] ...` line and writes stdout/stderr to `logs/`. Re-running is safe: a stage whose
output file already exists is skipped unless `FORCE=1` is set.

To run only one half of the pipeline:
```bash
bash run_baseline_movielens.sh        # dynamic-routing baseline only
bash run_cluster_hard_movielens.sh    # requires the baseline's representations_train.npz
```

## Where results land

- `results/baseline/metrics.json`, `gates_test.csv`, `gate_summary.json`,
  `figures/*.png` — baseline reproduction + gate diversity.
- `artifacts/kmeans_k4_seed1234.joblib`, `results/clustering/*` — SASRec clustering.
- `results/cluster_hard/metrics.json` — cluster-hard evaluation.
- `results/final_comparison.csv` / `.md`, `results/today_summary.md` — final write-up.

## If something fails partway

Re-run the same command; completed stages are skipped (see above). Set `FORCE=1` before a
command to force that pipeline's stages to redo their work even if outputs already exist. Check
`logs/` for the specific stage that failed — each script's `[n/total]` line shows how far it got.
