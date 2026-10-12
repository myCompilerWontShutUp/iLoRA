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
4. runs `setup_vessl.sh`: pinned requirements, bitsandbytes on torch's CUDA 11.7 runtime (below),
   and the two transformers source patches from `debug/` (required: without `debug/utils.py`
   every validation/test `generate()` fails); then re-activates the env and runs
   `scripts/verify_bnb_cuda.py` again, so the activation hook alone is shown to be enough;
5. runs `scripts/preflight_vessl.sh` and `scripts/test_moelora_equivalence.py`;
6. GPU smoke test of the evaluation path on the untrained model: expects all 95 MovieLens test
   sequences evaluated and 95 rows in `gates_test.csv` (outputs under `benchmark_runs/`);
7. GPU training-memory benchmark (256 micro-batches): must finish without OOM; prints the highest
   `max_memory_allocated` (expected roughly 50-55 GiB) and a time estimate for the full run.

`SKIP_GPU_CHECKS=1 bash bootstrap_vessl.sh` stops after step 5.

### bitsandbytes and the image's CUDA 12.8

The image ships CUDA 12.8 in `/usr/local/cuda` (also on its `LD_LIBRARY_PATH`); torch 2.0.0+cu117
uses its own CUDA 11.7 runtime from the `nvidia-*` pip wheels. bitsandbytes 0.37.2 takes the first
`libcudart.so` it finds (`$CONDA_PREFIX/lib`, then the `LD_LIBRARY_PATH` directories, picked in
arbitrary order when several match, then `/usr/local/cuda/lib64`) and loads
`libbitsandbytes_cuda<that version>.so`. Picking the 12.8 runtime made it look for
`libbitsandbytes_cuda128.so`, which 0.37.2 does not ship, and the import failed.

`scripts/setup_bnb_cuda_runtime.sh` (called by `setup_vessl.sh`, safe to re-run) derives the
version from `torch.version.cuda`, links the wheel's `libcudart.so.11.0` as
`$CONDA_PREFIX/lib/libcudart.so` (checked first, so the system CUDA is never reached), and writes
`$CONDA_PREFIX/etc/conda/{activate,deactivate}.d/ilora_bnb_cuda.sh`, which put the `nvidia-*` lib
directories first on `LD_LIBRARY_PATH` on every `conda activate ilora` (new terminals and tmux
included) and restore it on deactivate. `$CONDA_PREFIX/lib` itself is not put on
`LD_LIBRARY_PATH`, so system tools such as tmux keep working. `scripts/verify_bnb_cuda.py` checks
torch 2.0.0 / CUDA 11.7, the GPU, the runtime bitsandbytes picks, and that
`libbitsandbytes_cuda117.so` is the binary actually loaded. Nothing is compiled; the pinned
torch 2.0.0+cu117 / bitsandbytes 0.37.2 stay as they are.

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
run (`tmux kill-session -t ilora`): it does not recover. Do not start another full run before the
cause is isolated; `main.py` and the run scripts stay unchanged until then.

`scripts/diagnose_nan.py` changes one variable at a time; everything else (MovieLens, seed 1234,
num_moe 4, lora_r 8, batch 8 x 16, LR schedule shape, MoE-LoRA memory fix, model, data order,
step limit) is identical between runs. In the conda env, one run after the other:

```bash
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision 16   2>&1 | tee logs/diag_A.log   # A
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision bf16 2>&1 | tee logs/diag_B.log   # B
python scripts/diagnose_nan.py --compare benchmark_runs/nan_diag/p16_lr0.0001 benchmark_runs/nan_diag/pbf16_lr0.0001
```

Each run trains until 48 micro-batches after the first non-finite loss, or for one epoch (about
2 h) if there is none, and writes `benchmark_runs/nan_diag/<config>/`: `summary.json`,
`micro_batches.csv`, `optimizer_steps.csv`, and for a NaN run `first_nan_batch.pt` /
`first_nan_trainable.pt`. Recorded: epoch / micro-batch / optimizer step of the first non-finite
loss, lr, loss, gradient norm, GradScaler scale and skipped steps (fp16), the module that first
produced inf/NaN, token lengths, gate finiteness, and trainable parameters finite before and after
each optimizer step. `--lr` changes only the peak; warmup start and floor stay 8e-6.

`--compare` checks that the two runs saw the same batches and differ only in precision, then
reads the outcome as: fp16 NaN / bf16 stable -> precision very likely; both stable -> lr 8e-4
alone or together with fp16; both NaN -> not precision alone; fp16 stable / bf16 NaN -> unexpected,
re-check the implementation. Optional follow-ups, not run automatically: C `--lr 8e-4 --precision 16`
and D `--lr 8e-4 --precision bf16`, compared the same way.

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
