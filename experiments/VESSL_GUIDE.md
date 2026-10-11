# VESSL run guide

## Workspace

- GPU: exactly one A100 80GB (preflight aborts on any other count).
- Container image: a **Managed** image. VESSL requires custom images to contain SSHD and
  Python; the official `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel` has Python but no SSHD, so
  it is not used. The Managed image's own Python/CUDA versions do not matter: everything runs in a
  dedicated conda env with Python 3.10 and the pinned `torch==2.0.0` (CUDA 11.7) wheel.
- Disk: about 100 GB. Keep the Llama-2 weights on persistent storage if the workspace offers it.

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
skips finished stages; `FORCE=1` redoes them. At 2.04 s per micro-batch (measured before the
MoE-LoRA memory fix) one 5-epoch training run is about 23 hours, so baseline + cluster_hard take
about two days; the benchmark in step 7 prints the current estimate.

## Where results land

- `results/baseline/metrics.json`, `gates_test.csv`, `gate_summary.json`, `figures/*.png`
- `artifacts/kmeans_k4_seed1234.joblib`, `results/clustering/*`
- `results/cluster_hard/metrics.json`
- `results/final_comparison.csv` / `.md`, `results/today_summary.md`

Copy `results/`, `artifacts/` and `logs/` out of the workspace before terminating it.
