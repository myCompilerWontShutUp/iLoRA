# VESSL run guide

> **A workspace Pause or Stop deletes the container filesystem.** On 2026-10-12 a Pause removed
> `/root/iLoRA`, the `ilora` conda env, `/root/models/Llama-2-7b-hf`, the tmux sessions, the
> training logs, the checkpoints and the NaN-diagnosis files. Unless they are on a persistent
> volume, nothing under `/root` survives. Before any long run read "Storage and Pause" below; before
> pausing run `bash scripts/archive_results.sh` and download the archive. Stop a run with
> `tmux kill-session -t ilora`, never with Pause.

## Workspace

- GPU: exactly one A100 80GB (preflight aborts on any other count).
- Container image: a **Managed** image. VESSL requires custom images to contain SSHD and
  Python; the official `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel` has Python but no SSHD, so
  it is not used. The Managed image's own Python/CUDA versions do not matter: everything runs in a
  dedicated conda env with Python 3.10 and the pinned `torch==2.0.0` (CUDA 11.7) wheel.
- Disk: about 100 GB free for the repository (preflight fails below 10 GiB, warns below 30 GiB).
- Storage: see "Storage and Pause". `scripts/env_check.py` says for the repository, the weights and
  the conda env whether they are on the container filesystem.

## Commands

```bash
git clone https://github.com/myCompilerWontShutUp/iLoRA.git
cd iLoRA && git checkout exp/gate-cluster
export LLM_PATH=/root/models/Llama-2-7b-hf   # where the weights are, or should be downloaded to
export HF_TOKEN=...                         # only if the weights are not there yet
bash bootstrap_vessl.sh
```

`bootstrap_vessl.sh` (log in `logs/bootstrap_*.log`), about 30 minutes plus the download. It can be
run again at any time; finished steps stay as they are.

0. storage: which of the repository / weights / env are on the container filesystem (warning);
1. installs `tmux` with apt if missing (`DEBIAN_FRONTEND=noninteractive`);
2. creates conda env `ilora` with Python 3.10 from conda-forge only (`-y --override-channels`, so the
   Anaconda defaults channel and its Terms of Service prompt are never involved; installs Miniforge
   if there is no conda), or reuses it. It stops if an existing `ilora` env has another Python, or if
   `python` resolves outside the env. `PYTHONNOUSERSITE=1` keeps `~/.local` packages out;
3. checks that `config.json`, `tokenizer.model` and every shard listed in the index are in
   `$LLM_PATH`; if not, downloads (or resumes) `meta-llama/Llama-2-7b-hf` with `HF_TOKEN` (gated
   repo; the token is read from the environment only and written nowhere);
4. runs `setup_vessl.sh`: pinned requirements with the env's own `python -m pip`, bitsandbytes on
   torch's CUDA 11.7 runtime (below), and the two transformers source patches from `debug/`
   (required: without `debug/utils.py` every validation/test `generate()` fails); then re-activates
   the env and runs `scripts/verify_bnb_cuda.py` again, so the activation hook alone is enough;
5. runs `scripts/preflight_vessl.sh` (which ends with `scripts/env_check.py`),
   `scripts/test_moelora_equivalence.py` and `scripts/test_measurement.py`;
6. GPU smoke test of the evaluation path on the untrained model: expects all 95 MovieLens test
   sequences evaluated and 95 rows in `gates_test.csv` (outputs under `benchmark_runs/`);
7. GPU training-memory benchmark (256 micro-batches): must finish without OOM; prints the highest
   `max_memory_allocated` (expected roughly 50-55 GiB) and a time estimate for the full run;
8. prints the activation command, the tmux commands and the Pause warning.

`SKIP_GPU_CHECKS=1 bash bootstrap_vessl.sh` skips steps 6-7.

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

## Environment check before training

`bash scripts/preflight_vessl.sh` runs at the start of `run_baseline_movielens.sh` and of
`run_cluster_hard_movielens.sh` and stops the script before any training if a check fails. Its last
part, `python scripts/env_check.py` (can be run on its own; report in `logs/env_check_<time>.json`),
prints one OK / WARN / FAIL line each for: git commit and branch; conda env `ilora`, the Python
executable inside it and Python 3.10; torch 2.0.0 and every package imported from the env (not
`~/.local`); `torch.version.cuda` 11.7; `torch.cuda.is_available()`; GPU name and VRAM (exactly one,
at least 75 GiB); the libcudart torch loaded; the bitsandbytes binary and libcudart bitsandbytes
actually loaded (`scripts/verify_bnb_cuda.py`); transformers 4.28.0 with both patches;
pytorch_lightning 1.8.6; `LLM_PATH` with every shard; the MovieLens files and `rec_model/movielens.pt`;
free disk; writable `logs log checkpoints outputs results runs benchmark_runs archives`; tmux
(starts and stops a detached session on a private socket, and does not load libtinfo from conda);
and the storage lines. FAIL lines name the fix.

## Environment problems seen on VESSL

| Problem | Cause | Handled by | Checked |
| --- | --- | --- | --- |
| Base Python 3.14 although the image name says py3.10 | the Managed image's base env | all work happens in conda env `ilora`, Python 3.10; bootstrap stops if an existing `ilora` has another Python or `python` resolves outside it | env_check (python executable, version, packages inside the env) |
| `conda create` stopped at the Anaconda Terms of Service | newer conda refuses the defaults channel until its ToS are accepted interactively | `conda create -y -q -c conda-forge --override-channels python=3.10`: defaults is never used, nothing to accept | `scripts/test_env_scripts.sh` (static); VESSL: a fresh workspace |
| bitsandbytes looked for `libbitsandbytes_cuda128.so` | it picked the image's CUDA 12.8 `libcudart.so` | `scripts/setup_bnb_cuda_runtime.sh` (section above), unchanged since it worked on VESSL | `verify_bnb_cuda.py` in setup, after re-activation, and in every preflight |
| tmux: `NCURSES6_TINFO_6.4.current not found` | an earlier manual fix put `$CONDA_PREFIX/lib` (with conda's `libtinfo.so.6`) on `LD_LIBRARY_PATH` | the activation hook adds only the `nvidia-*` wheel directories and restores `LD_LIBRARY_PATH` on deactivate; plain `tmux` works, `env -u LD_LIBRARY_PATH` is no longer needed | `test_env_scripts.sh`: 10 activate/deactivate cycles, no `libtinfo` directory on the path; env_check: tmux session start/stop and `ldd` |
| Everything lost on Pause | the container filesystem is recreated | warnings in bootstrap, run_all, env_check; `scripts/archive_results.sh` | env_check storage lines |

Only VESSL can confirm: that a fresh workspace's `conda create` really asks nothing; that the real
bitsandbytes loads `libbitsandbytes_cuda117.so` (verify_bnb_cuda.py prints it); and interactive
tmux attach / detach (`tmux new -s t`, `Ctrl-b d`, `tmux attach -t t`), which needs a terminal.

## Storage and Pause

1. Before a long run, look at the storage lines of `python scripts/env_check.py --storage-only`
   (also printed by bootstrap and run_all). "container filesystem" means a Pause/Stop deletes it.
   "other mounts" lists separate filesystems; `/opt/vessl` is VESSL's own mount, not a data volume.
2. If the workspace has a persistent volume (configured in VESSL when the workspace is created),
   clone the repository and put `LLM_PATH` there; whether a mount is persistent can only be seen in
   the VESSL workspace settings, not from inside the container.
3. Without one, treat the container as disposable: never Pause during a run, stop runs with
   `tmux kill-session -t ilora`, and archive before pausing.

### Before a Pause

```bash
bash scripts/archive_results.sh                      # logs log benchmark_runs results outputs artifacts runs
bash scripts/archive_results.sh --with-checkpoints   # plus checkpoints/ (LoRA weights, larger)
bash scripts/archive_results.sh --dry-run            # only list what would be packed
```

It writes `archives/ilora_results_<host>_<time>.tgz` and a manifest (git commit, folder sizes, file
count, sha256), uploads nothing, deletes nothing, and never overwrites an archive. Download it with
the JupyterLab file browser (`archives/`, right click, Download) or
`scp -P <port> root@<host>:/root/iLoRA/archives/<file>.tgz .`, and compare the sha256. `--dest DIR`
also copies it into an existing directory (for example a persistent mount you created yourself);
the script never picks a location on its own. The weights are not archived (bootstrap downloads them
again with `HF_TOKEN`).

## If the training loss becomes NaN

The first full baseline run on VESSL (fp16 AMP, lr 8e-4, as in the released scripts) showed
`loss=nan` for the whole second epoch and 0 valid predictions at the first validation. Stop such a
run (`tmux kill-session -t ilora`): it does not recover.

Outcome (IMPLEMENTATION_NOTES.md section 11i): at the paper's peak lr 1e-4, run A (fp16) trained a
full epoch and run B (bf16) passed the warmup peak without a non-finite loss, so the NaN needs
lr 8e-4. The run scripts now use `--lr 1e-4` with fp16; the released `train_movielens.sh` is
unchanged. The diagnostic below is kept for re-checking.

`scripts/diagnose_nan.py` changes one variable at a time; everything else (MovieLens, seed 1234,
num_moe 4, lora_r 8, batch 8 x 16, LR schedule shape, MoE-LoRA memory fix, model, data order,
step limit) is identical between runs. In the conda env, one run after the other:

```bash
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision 16   2>&1 | tee logs/diag_A.log   # A
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision bf16 2>&1 | tee logs/diag_B.log   # B
python scripts/diagnose_nan.py --compare runs/nan_diag_p16_lr0.0001_<time> runs/nan_diag_pbf16_lr0.0001_<time>
```

Each run trains until 48 micro-batches after the first non-finite loss, or for one epoch (about
2 h) if there is none, and writes a new run folder `runs/nan_diag_<config>_<time>/` (runs A and B
of 2026-10-12 used the earlier layout `benchmark_runs/nan_diag/<config>/`, which `--compare` still
reads): `summary.json`, `micro_batches.csv.gz`, `optimizer_steps.csv.gz`, the measurement files of
the next section, and for a NaN run `first_nan_batch.pt` / `first_nan_trainable.pt`. Recorded: epoch / micro-batch / optimizer step of the first non-finite
loss, lr, loss, gradient norm, GradScaler scale and skipped steps (fp16), the module that first
produced inf/NaN, token lengths, gate finiteness, and trainable parameters finite before and after
each optimizer step. `--lr` changes only the peak; warmup start and floor stay 8e-6.

bf16 needs the SASRec causal-mask change in `SASRecModules_ori.py` (torch 2.0 has no CUDA
bfloat16 `torch.tril`; the mask is now built as bool and cast, giving identical values in every
dtype). `python scripts/test_sasrec_mask_equivalence.py` checks that on the GPU before run B,
including that the fp16 path is bit-identical to the upstream code used for run A.

`--compare` checks that the two runs saw the same batches and differ only in precision, then
reads the outcome as: fp16 NaN / bf16 stable -> precision very likely; both stable -> lr 8e-4
alone or together with fp16; both NaN -> not precision alone; fp16 stable / bf16 NaN -> unexpected,
re-check the implementation. Optional follow-ups, not run automatically: C `--lr 8e-4 --precision 16`
and D `--lr 8e-4 --precision bf16`, compared the same way.

## Measurement run folders

`main.py --measure`, `scripts/diagnose_nan.py` and `scripts/measure_base_activation.py` write a new
folder `runs/<name>_<time>/` per run (IMPLEMENTATION_NOTES.md section 11j; the folder's README.txt
explains every column). The hooks only read tensors; without `--measure`, `main.py` is unchanged.

```bash
# once on a new workspace (GPU, about 25-40 min): bit-identity, overhead, short run, kill test
python scripts/verify_measurement_gpu.py --llm_path "$LLM_PATH"
python scripts/test_measurement.py                                   # CPU, about 1 min

# diagnostic run (fp16 / bf16) and the activation baseline before training
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision 16
python scripts/diagnose_nan.py --llm_path "$LLM_PATH" --lr 1e-4 --precision bf16
python scripts/measure_base_activation.py --llm_path "$LLM_PATH" --precision 16 --conditions base,lora     --num_micro_batches 200 --compare_with runs/nan_diag_p16_lr0.0001_<time>

# during a run, in another tmux window
python scripts/gpu_monitor.py --latest --interval 60
python scripts/plot_live.py runs/<run> --watch 300

# afterwards (CPU; also on a laptop after copying the folder)
python analysis/plot_measurement.py runs/<run> [--compare runs/<run2>] [--base runs/<base_run>]
python scripts/inspect_anomaly.py runs/<run> <event_id>
python scripts/finalize_run.py runs/<run>                            # only after a hard kill
```

Measuring the full baseline: add `--measure` (and e.g. `--stats_every 4` if the folder gets too
large; m_t and anomalies are still checked every micro-batch) to the training `main.py` call. The
overhead and the folder size per epoch are printed by `verify_measurement_gpu.py`. `runs/` is
git-ignored: download the folders before pausing the workspace.

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
