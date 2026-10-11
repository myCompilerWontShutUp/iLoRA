# VESSL run guide

## Workspace

- GPU: exactly one A100 80GB (preflight aborts on any other count).
- Container image: a **Managed** image. VESSL requires custom images to contain SSHD and
  Python; the official `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel` has Python but no SSHD, so
  it is not used. The Managed image's own Python/CUDA versions do not matter: everything runs in a
  dedicated conda env with Python 3.10 and the pinned `torch==2.0.0` (CUDA 11.7) wheel.
- Disk: about 100 GB. Keep the Llama-2 weights on persistent storage if the workspace offers it.
- Pausing or stopping the workspace recreates the container: everything outside a mounted volume
  (repository, conda env, weights, checkpoints, logs) is lost. Stop a run with
  `tmux kill-session -t ilora` instead, and copy `results/`, `artifacts/` and `logs/` out before
  pausing.

## Commands

```bash
git clone https://github.com/myCompilerWontShutUp/iLoRA.git
cd iLoRA && git checkout exp/gate-cluster
export LLM_PATH=/root/models/Llama-2-7b-hf   # where the weights are, or should be downloaded to
export HF_TOKEN=...                         # only if the weights are not there yet
bash bootstrap_vessl.sh
```

`bootstrap_vessl.sh` (log in `logs/bootstrap_*.log`), about 30 minutes plus the download:

1. installs `tmux` if missing;
2. creates conda env `ilora` (Python 3.10, conda-forge; installs Miniforge if there is no conda);
3. downloads `meta-llama/Llama-2-7b-hf` (json, safetensors, tokenizer) into `$LLM_PATH` if it is
   not there; the HF account needs access to that gated repo. The token is read from `HF_TOKEN`
   only and is not written anywhere in the repository;
4. runs `setup_vessl.sh`: pinned requirements, the bitsandbytes CUDA 11.7 library path, and the
   two transformers source patches from `debug/` (required: without `debug/utils.py` every
   validation/test `generate()` fails);
5. runs `scripts/preflight_vessl.sh` and `scripts/test_moelora_equivalence.py`;
6. GPU smoke test of the evaluation path on the untrained model: expects all 95 MovieLens test
   sequences evaluated and 95 rows in `gates_test.csv` (outputs under `benchmark_runs/`);
7. GPU training-memory benchmark (256 micro-batches): must finish without OOM; prints the highest
   `max_memory_allocated` (expected roughly 50-55 GiB) and a time estimate for the full run.

`SKIP_GPU_CHECKS=1 bash bootstrap_vessl.sh` stops after step 5.

Then start the experiment in tmux (the exact command is printed at the end of the bootstrap):

```bash
tmux new -s ilora
cd iLoRA && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ilora
export LLM_PATH=/root/models/Llama-2-7b-hf CUDA_VISIBLE_DEVICES=0
bash run_all_vessl.sh
```

Detach with `Ctrl-b d`, re-attach with `tmux attach -t ilora`. Re-running `run_all_vessl.sh`
skips finished stages; `FORCE=1` redoes them. After the MoE-LoRA memory fix the first VESSL run
trained at about 1.38 micro-batches/s (about 1 h 40 min per epoch, about 8.5 h per 5-epoch
training); the benchmark in step 7 prints the current estimate.

## If the training loss becomes NaN

The first full baseline run on VESSL (fp16 AMP, lr 8e-4, as in the released scripts) showed
`loss=nan` for the whole second epoch and 0 valid predictions at the first validation. Stop such a
run (`tmux kill-session -t ilora`): it does not recover. Then, in the conda env:

```bash
python scripts/diagnose_nan.py --llm_path "$LLM_PATH"                    # same config as the run
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --precision bf16   # candidate fix
```

Each run trains the baseline configuration with the full-run LR schedule until 48 micro-batches
after the first non-finite loss (or one epoch, about 2 h at most) and writes
`benchmark_runs/nan_diag/<config>/summary.json` plus per-micro-batch and per-optimizer-step CSVs.
The summary names the module that first produced inf/NaN, whether the parameters were still
finite, and whether GradScaler skipped every later step (fp16 overflow leaves the run stuck at
fixed parameters). `--precision bf16` matches the dtype the weights are loaded in and LLaRA's
default; whether the full run should switch is decided from these two summaries.

## Running baseline and cluster_hard in parallel (two workspaces, about one day instead of two)

cluster_hard does not depend on the baseline training: its clusters come from the frozen SASRec
representations, which `run_cluster_hard_movielens.sh` extracts itself. Total GPU hours are the
same; each workspace still uses exactly one GPU.

- Workspace A: `bash bootstrap_vessl.sh`, then in tmux `bash run_baseline_movielens.sh`.
- Workspace B (start after A's bootstrap passed): `SKIP_GPU_CHECKS=1 bash bootstrap_vessl.sh`,
  then in tmux `bash run_cluster_hard_movielens.sh`.
- When B finishes: `tar czf cluster_hard_results.tgz results/clustering results/cluster_hard artifacts logs`,
  copy the archive into the repository on A, `tar xzf cluster_hard_results.tgz`, and once A has
  finished too run `bash merge_parallel_results.sh` on A (CPU only; re-runs the clustering analysis
  with the baseline gates, then the final comparison and summary). Workspace B can be terminated
  as soon as the archive is copied.

A stage is skipped on re-run only if it really finished: training writes
`checkpoints/<run>/TRAINING_COMPLETE` when `main.py` exits successfully (`last.ckpt` alone is
not trusted, it also exists after a crash), and a freshly trained model is always re-evaluated.

## Where results land

- `results/baseline/metrics.json`, `gates_test.csv`, `gate_summary.json`, `figures/*.png`
- `artifacts/kmeans_k4_seed1234.joblib`, `results/clustering/*`
- `results/cluster_hard/metrics.json`
- `results/final_comparison.csv` / `.md`, `results/today_summary.md`

Copy `results/`, `artifacts/` and `logs/` out of the workspace before terminating it.
