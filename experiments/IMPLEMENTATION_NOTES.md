# Implementation Notes (Phase 2 design decisions + running log)

This file records design decisions that are not fully dictated by
`experiments/EXPERIMENT_SPEC.md`, and any place where a discrepancy was found between the
released code and expectations. It is updated as implementation proceeds.

## 1. sample identity

MovieLens data has no `user_id` (see `CODEBASE_ANALYSIS.md` §C). `sample_id` is defined as the
positional row index into the per-split DataFrame used by `MovielensData` (0-based, per split).
This index is threaded end-to-end as a new `sample_idx` field:

- `data/movielens_data.py`: `MovielensData.__getitem__` adds `'sample_idx': i` to the returned
  sample dict, for every stage (train/val/test). This is a pure addition; no existing key is
  changed.
- `data/data_interface.py`: `TrainCollater.__call__` adds `"sample_idx": torch.stack([...])` to
  the batch dict in **both** the `train=True` and `train=False` branches.

`user_id` is recorded as the literal string `NOT_AVAILABLE` in every export that has a `user_id`
column, since the dataset does not provide one.

These two additions are the only changes to the original data pipeline. They do not alter any
existing field, tensor, or control flow used by the original dynamic-routing baseline — the new
key is simply unused by any code path unless it explicitly asks for it (`--routing_mode
cluster_hard` or `--export_gates`).

**Compatibility note**: `TrainCollater` in `data/data_interface.py` is shared by all three
datasets (MovieLens/Steam/LastFM). Since it now unconditionally reads `sample['sample_idx']`,
`data/steam_data.py` and `data/lastfm_data.py` also received the identical one-line addition
(`'sample_idx': i` in their `__getitem__`) so Steam/LastFM are not broken by a MovieLens-only
change to shared code, per spec §0.5. Today's experiment scope remains MovieLens-only; Steam/
LastFM are not otherwise touched or run.

## 2. Gate / representation export design

- Gate export is instrumented at the single real call site established in
  `CODEBASE_ANALYSIS.md` §H/I/J: `model_interface.py` `MInterface.generate()`, right after
  `gate_weights = self._resolve_gate(...)` (the line that used to be
  `gate_weights = self.router(user_embeds)`). `generate()` is called by both `validation_step`
  and `test_step`; a new `split` argument (`"validation"` / `"test"`) tells the exporter which
  file to write to, and lets `cluster_hard` mode look up the correct assignment table.
- Export is opt-in via `--export_gates` (default off) so that normal test/validation runs that
  do not care about gate diagnostics pay no extra cost. `run_baseline_movielens.sh` turns it on
  for the baseline evaluation step only.
- Rows are accumulated in memory (`self._gate_export_rows`, reset in
  `on_validation_epoch_start` / `on_test_epoch_start`) and written once, in
  `on_validation_epoch_end` / `on_test_epoch_end`, as `gates_validation.csv` / `gates_test.csv`
  under `--results_dir`. No per-step CSV I/O, per spec §4.
- Every stored tensor is `.detach().float().cpu()` before leaving the model, per spec §4.
- `sasrec_0..sasrec_{D-1}` columns use `D = user_embeds.shape[-1]` read at runtime (confirmed to
  be 64 for MovieLens in `CODEBASE_ANALYSIS.md`, but the code never hardcodes it).
- Per-row validation (`model/routing_utils.gate_diagnostics`): gate probability sum within
  `1e-3` of 1, no NaN, no Inf. Violations are **not** fatal (a long VESSL run should not be
  aborted over a diagnostic row) — they are counted and the count is written into
  `gate_summary.json`'s `"num_gate_rows_failing_validation"` field by `analyze_gates.py`, and a
  warning is printed at export time.
- Train-split SASRec representations do **not** go through this path (training never calls
  `generate()`). They are produced by the fully decoupled `scripts/extract_representations.py`
  (see §3), which is the same script used for the validation/test representations that feed
  KMeans `predict()`, for consistency of provenance across all three splits.

## 3. Why representation extraction is a separate, LLaMA-free script

`CODEBASE_ANALYSIS.md` §F/G/R establishes that `encode_users` only calls the frozen `SASRec.
cacul_h`, which needs nothing but the SASRec checkpoint and the raw `(item_id, rating)` sequence
— no LLaMA, no GPU, no MoE-LoRA machinery. `scripts/extract_representations.py` therefore
imports `MovielensData` and `SASRec` directly and computes representations for train/val/test
with `rec_model.eval()` explicitly forced (see §4 below for why this matters), independent of
whether any LLaMA-based baseline or cluster-hard run has happened. This is also what makes it
possible to test this script for real, locally, on Windows/CPU, against the real shipped
`rec_model/movielens.pt` and real MovieLens data (verified in `CODEBASE_ANALYSIS.md`).

## 4. Cluster assignment must be fixed, not recomputed on the fly

`CODEBASE_ANALYSIS.md` §R/S surfaces a real subtlety: `load_rec_model()` calls `rec_model.eval()`
once at startup, but PyTorch Lightning calls `model.train()` at the start of every training
epoch, which recurses into `self.rec_model` and would silently re-enable its `Dropout` layers
during training — even though its weights are frozen. That is pre-existing upstream behavior and
this experiment does **not** touch it for the `dynamic` routing path (baseline preservation is
non-negotiable, spec §0.1).

For `cluster_hard`, however, spec §10 explicitly requires a reproducible, epoch-stable cluster
assignment rather than one that could flicker if it were recomputed via `kmeans.predict()` on a
representation that might carry train-time dropout noise. The resolution used here:

- Cluster assignment is computed **once**, offline, by `scripts/fit_kmeans.py`, using
  representations produced by `scripts/extract_representations.py` (which forces
  `rec_model.eval()` itself, independent of any Lightning trainer state).
- The result is persisted as plain lookup tables:
  `results/clustering/{train,validation,test}_assignments.csv` (`sample_id -> cluster_id`).
- At both training time (`forward()`, always split="train") and evaluation time
  (`generate()`, split="validation"/"test"), `cluster_hard` mode looks up the sample's cluster
  by `sample_idx` in the appropriate fixed table and converts it to a one-hot gate via
  `--cluster_expert_mapping`. **No KMeans inference happens inside the training/eval loop at
  all** — this sidesteps the dropout-in-"eval"-model subtlety entirely, and makes cluster-hard
  training strictly reproducible given the same assignment CSVs.
- `--cluster_model_path` (the saved `KMeans` joblib artifact) is loaded only for a sanity
  assertion (that its `n_clusters` is consistent with the mapping) and is otherwise used by the
  analysis/visualization scripts (silhouette score, PCA projection), not by the training loop.

## 5. Router exclusion for cluster_hard (spec §11)

`configure_optimizers` builds exactly 3 param groups, and `LinearWarmupCosineLRScheduler`
(`optims.py`) indexes `init_lr_list`/`min_lr_list`/`warmup_start_lr_list` positionally against
`optimizer.param_groups` by index. Removing the router group when `routing_mode == cluster_hard`
would require also resizing those three lists — a larger, riskier change than necessary for what
the spec actually asks for. Instead:

- When `routing_mode == 'cluster_hard'`, every parameter of `self.router` has `requires_grad`
  set to `False` in `__init__`, right after construction.
- `self.router` is never called in the cluster_hard forward path (`_resolve_gate` branches
  before touching `self.router`), so it also never receives a gradient from autograd — the
  `requires_grad = False` is redundant with that but makes the "no update" property explicit,
  inspectable (`all(not p.requires_grad for p in self.router.parameters())`), and immune to
  future refactors of the forward path.
- `self._router_call_count` is incremented only in the `dynamic` branch of `_resolve_gate` and
  logged at the end of each validation/test epoch, so "was the neural router actually called
  this epoch" is directly observable in the logs for both modes (should be 0 for cluster_hard,
  equal to the number of batches for dynamic).
- `PeftModel`/`MoeLoraModel`/`mark_only_lora_as_trainable` are untouched — they only ever see
  `self.llama_model`, never `self.router`, so nothing here affects LoRA/gating trainability.

## 6. routing_mode plumbing keeps the dynamic path byte-for-byte identical

`forward()` and `generate()` previously called `self.router(user_embeds)` directly. Both now
call `self._resolve_gate(user_embeds, batch, split)`, whose `dynamic` branch is exactly
`return self.router(user_embeds)` — the same object, same call, same tensor in, same tensor out.
`--routing_mode` defaults to `dynamic`, so running with no new flags at all reproduces the
original code path exactly (spec §0 / §22 requirement). All `cluster_hard`-only state
(`_cluster_mapping`, `_cluster_assignments`) is `None` unless `routing_mode == 'cluster_hard'`,
and no cluster file is opened, read, or referenced otherwise.

## 7. Metric name mapping (spec §12 columns vs. actual code, from CODEBASE_ANALYSIS.md §P/Q)

| spec column     | actual PyTorch Lightning metric name |
|-----------------|---------------------------------------|
| ValidRatio       | `test_prediction_valid`              |
| HR@1             | `test_hr`                             |
| CombinedMetric   | `metric` (`= hr * valid_ratio`)       |

There is no other metric defined anywhere in this codebase; nothing is invented.

## 8. Hyperparameter source of truth

Per spec §0.7, the **released MovieLens shell scripts** (`train_movielens.sh`,
`test_movielens.sh`) are the source of truth for baseline hyperparameters, not the argparse
defaults in `main.py` and not (yet, pending a manual read of the paper) the paper Appendix.
`run_baseline_movielens.sh` and `run_cluster_hard_movielens.sh` pass exactly the same values as
the released scripts (`batch_size=8`, `accumulate_grad_batches=16`, `cans_num=20`,
`max_epochs=5`, `lr=8e-4`, `lr_warmup_start_lr=8e-6`, `lr_decay_min_lr=8e-6`, `router=share`,
`gating=Dense`) plus `--seed 1234` (spec §0.6) and the new routing/export flags. The
paper-vs-released-script hyperparameter diff (spec §1.V) is left as `VESSL_REQUIRED` /
unverified in this pass — no network fetch of the paper was performed; this does not block
implementation since the released-script values are used regardless of the outcome of that
comparison.

**Superseded for the peak lr, see section 11i:** the run scripts now use the paper's `lr=1e-4`;
every other value above is unchanged.

## 9. Known, documented (not fixed) confounder

Per `CODEBASE_ANALYSIS.md` §T/U and spec §13: with `lora_r=8, num_moe=4`, each expert has
effective rank 2. `cluster_hard`'s one-hot gate activates one rank-2 expert per sample, while
`dynamic` mixes all four rank-2 experts with a soft gate. Any performance difference between the
two conflates routing strategy with active-capacity. This is recorded here and in
`results/today_summary.md`'s Limitations section; it is explicitly **not** worked around in the
primary experiment (no rank or parameter-budget changes), per spec.

## 10. Dependency / environment notes for `setup_vessl.sh`

Pinned versions from `requirements.txt`: `torch==2.0.0`, `transformers==4.28.0`, `peft==0.3.0`
(not actually imported — PEFT is vendored under `model/peft`, see `CODEBASE_ANALYSIS.md`
"Other structural facts"), `pytorch_lightning==1.8.6`, `bitsandbytes==0.37.2`. `setup_vessl.sh`
installs from `requirements.txt` as-is first and only applies the README's documented
`debug/utils.py` / `debug/modeling_llama.py` replacement if the installed `transformers` files
differ from what the debug files expect (detected by hash comparison against the debug files
actually shipped in this repo, not guessed) — this keeps the workaround minimal and traceable
instead of always overwriting installed package files. Any such replacement, and the exact
`torch`/`transformers`/`peft`/`pytorch_lightning`/`bitsandbytes` versions actually resolved on
the VESSL box, are appended to this file's "VESSL run log" section (§12) the first time
`setup_vessl.sh` is actually executed there — not invented ahead of time.

## 11. New CLI arguments added to `main.py`

All default to values that reproduce the original baseline unmodified:

- `--routing_mode {dynamic,cluster_hard}` (default `dynamic`)
- `--cluster_model_path` (default `None`; required by `cluster_hard`, used only for a sanity
  assertion, see §4)
- `--cluster_assignment_dir` (default `results/clustering`; used only by `cluster_hard`)
- `--cluster_expert_mapping` (default `"0:0,1:1,2:2,3:3"`; used only by `cluster_hard`)
- `--export_gates` (store_true, default `False`)
- `--results_dir` (default `results/baseline`; `run_cluster_hard_movielens.sh` passes
  `results/cluster_hard`)

## 11b. Pre-existing issue found during Phase 7 validation (not fixed, out of scope)

`python -m compileall` on the full repo surfaces a pre-existing syntax error in
`model/peft/tuners/test_moelora.py` (line 57: `unittest.main()import torch`, concatenated
without a newline) and two harmless `SyntaxWarning: invalid escape sequence '\.'` in
`model/peft/tuners/lora.py:212` and `model/peft/tuners/moelora.py:255` (regex strings missing an
`r` prefix). None of these three lines were touched by this branch, and none of them are on any
path this experiment exercises (`test_moelora.py` is an unused/uncalled leftover test file, and
the two escape-sequence warnings are in a `re.match` pattern that is only reached for models with
a `layers_pattern` config, which this experiment never sets). Per spec §0.2 (no unrelated
refactors) and §23 (only implement what today's experiment needs), these are left as-is and
recorded here rather than fixed.

## 11c. Preflight code review before real A100 usage (2026-09-21)

A pre-spend review of the actual exp/gate-cluster implementation (not a new phase of work)
checked determinism, fixed-vs-recomputed cluster assignment, pipeline ordering, and the
setup/preflight scripts' ability to catch environment problems before GPU time is spent. Two
real gaps were found and fixed with minimal, targeted changes:

- **`scripts/preflight_vessl.sh` did not hard-fail on a broken torch/CUDA install.** It printed
  `torch.cuda.is_available()` and `torch.cuda.device_count()` but never checked them —
  `requirements.txt` pins `torch==2.0.0` with no CUDA-build pin, so a driver/runtime mismatch on
  the actual VESSL image could pass `nvidia-smi`'s GPU-count check (a driver-level check) while
  torch itself still cannot see the GPU, and this would previously have gone undetected until
  training was already underway. Fixed: preflight now explicitly asserts
  `torch.cuda.is_available()` is `True` and `torch.cuda.device_count() == 1`, exiting non-zero
  before training if not. Verified locally that this check correctly fails (exit code 1) on this
  CPU-only machine, where `torch.cuda.is_available()` is `False`.
- **`setup_vessl.sh` had no import-time smoke test.** `model/peft/import_utils.py`'s
  `is_bnb_available()` only checks `importlib.util.find_spec("bitsandbytes")` (is the package on
  disk?), not whether it actually imports — `model/peft/tuners/moelora.py` does
  `if is_bnb_available(): import bitsandbytes` at module load time, and a bitsandbytes native
  extension that fails to load (a known failure mode for the old pinned `bitsandbytes==0.37.2`
  against a mismatched CUDA toolkit) would crash `import main` — and therefore every single
  training/eval command — even though this experiment never uses 8-bit/4-bit loading and gains
  nothing from bitsandbytes actually working. Fixed: `setup_vessl.sh` now does
  `import transformers; import pytorch_lightning; import model.peft` right after
  `pip install -r requirements.txt` and aborts with a clear message if any of the three fail,
  catching this (and any transformers-debug-file-driven import break) once, cheaply, at setup
  time instead of repeatedly inside `run_all_vessl.sh`.

Both fixes were verified locally (embedded Python blocks syntax-checked by extraction and
`compile()`; the new preflight CUDA gate and the new setup import-smoke-test were each run
directly and confirmed to fail with a clear message and exit code 1 in this GPU-less/deps-less
local environment, which is the correct behavior here). `scripts/smoke_test_routing.py`'s full
17-check suite was re-run and still passes.

Everything else reviewed came back clean, matching the design already recorded above, and was
left unmodified per the reviewer's instruction not to expand scope:
- `scripts/extract_representations.py` forces `rec_model.eval()` unconditionally (line 38) and
  never calls `.train()`; empirically verified locally by running it twice, independently, end to
  end against the real `rec_model/movielens.pt` checkpoint and the real MovieLens
  validation/test splits — the two runs' `sasrec` arrays were bit-for-bit identical
  (`max_abs_diff == 0.0`), not merely within floating-point tolerance.
- `MInterface._resolve_gate`'s `cluster_hard` branch only does dict lookups into
  `self._cluster_assignments` (loaded once, at `__init__`, from the fixed assignment CSVs); it
  never calls `self.router(...)` or any KMeans `.predict()` — confirmed by re-reading
  `model/model_interface.py` lines 91-115.
- `scripts/fit_kmeans.py` calls `kmeans.fit_predict(train_reps)` on the train split only and
  `kmeans.predict(...)` (no further fitting) on validation/test — confirmed by re-reading lines
  56-63.
- `run_baseline_movielens.sh` step 5 (representation export) and `run_cluster_hard_movielens.sh`
  steps 1-3 (representation check, KMeans fit, assignment-file existence check) run, and are
  ordered, strictly before any cluster_hard training command; `run_all_vessl.sh` calls the
  baseline script before the cluster-hard script. `MInterface._setup_routing_mode` also
  independently fails fast (`FileNotFoundError`) if an assignment CSV is missing, so an ordering
  mistake would surface immediately rather than silently.
- `LLM_PATH` is required to be an existing local filesystem path
  (`scripts/preflight_vessl.sh`'s `[[ ! -e "$LLM_PATH" ]]` check) and is passed straight into
  `LlamaTokenizer.from_pretrained` / `LlamaForCausalLM.from_pretrained`, both of which accept a
  local Hugging-Face-format directory directly.

## 11d. Benchmark/dry-run script and VESSL-verified setup fixes (2026-09-21)

After the preflight review in §11c passed on the real VESSL A100 instance (`torch 2.0.0+cu117`,
`transformers 4.28.0`, `pytorch_lightning 1.8.6`, conda env `ilora` / Python 3.10.21), three real
environment problems were hit and fixed manually on the VESSL box before any training was
started:

1. The base/system Python on the VESSL image is 3.14, which cannot install `requirements.txt`
   (pins like `torch==2.0.0` predate 3.14 wheel availability). Fixed manually with a dedicated
   `conda create -n ilora python=3.10`.
2. `requirements.txt` was missing `safetensors`, so the import smoke test added in §11c failed.
   Fixed manually with `pip install safetensors==0.3.1`.
3. `bitsandbytes==0.37.2` probed the system CUDA (12.8) instead of the `torch==2.0.0+cu117` build
   actually installed, and failed to find a matching `libbitsandbytes_cuda128.so`. Fixed manually
   by symlinking the pip-installed CUDA 11.7 runtime's `libcudart.so.11.0` to `libcudart.so` and
   pointing `LD_LIBRARY_PATH` at the pip-installed `nvidia-*` lib directories, after which
   bitsandbytes correctly detected CUDA 11.7 and loaded `libbitsandbytes_cuda117.so`.

`setup_vessl.sh` was updated to automate exactly these three verified fixes (no broader
environment handling was added):
- A Python-version hard gate (`python -c 'import sys; print(...)'` compared against `3.10`)
  right after the OS check, before `pip install` is attempted, so the failure mode in (1) is a
  clear, early, actionable message instead of a confusing pip resolution failure.
- `safetensors==0.3.1` added to `requirements.txt` directly, so (2) does not recur.
- A new step, gated on `CONDA_PREFIX` being set *and* `torch.version.cuda == "11.7"` (so it only
  ever fires in exactly the verified scenario), that creates the `libcudart.so` symlink if
  missing and writes the `LD_LIBRARY_PATH` export into
  `$CONDA_PREFIX/etc/conda/activate.d/ilora_bnb_cuda117.sh` (the standard conda mechanism for
  environment variables that should be active whenever that env is active) — this makes the fix
  from (3) persist across every future `conda activate ilora` and therefore across every
  subsequent `bash scripts/preflight_vessl.sh` / `bash run_all_vessl.sh` invocation, without
  needing to touch those other scripts at all.

`scripts/benchmark_baseline.py` was added: a standalone dry-run that loads the model once and
runs exactly 256 real training micro-batches under the released baseline's exact configuration
(`routing_mode=dynamic`, `batch_size=8`, `accumulate_grad_batches=16`, `precision=16`, `seed=1234`,
`num_moe=4`, `lora_r=8`), with validation/test/checkpointing/gate-export/analysis all disabled
(`limit_val_batches=0`, `enable_checkpointing=False`, `callbacks=[]`, `logger=False`) and all
output confined to `benchmark_runs/baseline_dry_run/` (gitignored), never touching
`checkpoints/baseline`, `outputs/baseline`, `results/baseline`, or `logs/`. It prints
model-loading time, 256-batch elapsed time, sec/batch, and the actual optimizer-step count as
`[MEASURED]`, and a linear extrapolation to the real baseline's full 5-epoch run and to "baseline
+ cluster_hard at the same speed" as clearly separate `[ESTIMATED]` lines — this exists so the
real per-batch speed on the actual rented A100 is known before committing to the two full
5-epoch training runs `run_all_vessl.sh` would otherwise start blind.

Building this script required duplicating a subset of `main.py`'s argparse defaults (main.py has
no importable entry point and was intentionally left unmodified). Cross-referencing every
`self.hparams.X` access in `model/model_interface.py` and every `kwargs['X']` access in
`data/data_interface.py` against the benchmark script's constructed args caught one real bug
before it could waste GPU time: `args.loss = 'lm'` was missing, which `configure_loss` (called
from every `training_step`) would have hit as an `AttributeError` on the very first batch. Fixed
by adding it. `capture_dir` is referenced by `save_gradients_to_file`, but that method is dead
code (never called from anywhere, confirmed by `grep`), so it was correctly left out.

## 11e. Fix: benchmark_baseline.py needs main.py's pickle-compatibility imports too (2026-09-21)

Running `scripts/benchmark_baseline.py` for real on the VESSL A100 loaded Llama-2-7B successfully
but then failed in `MInterface.load_rec_model()`'s `torch.load('./rec_model/movielens.pt', ...)`
with `AttributeError: module '__main__' has no attribute 'SASRec'`.

Root cause: `rec_model/movielens.pt` was pickled at a time when the `SASRec` class's
`__module__` was `'__main__'` (i.e., whatever script defined/trained it was executed directly,
so Python recorded its classes as belonging to `__main__` at pickle time — this is baked into
the pickle bytes and cannot be changed by how anyone re-imports the class later). `torch.load`'s
unpickler resolves class references by looking up `sys.modules['__main__'].<ClassName>`, so
`SASRec` (and by the same mechanism, `Caser`/`GRU`) must be bound as attributes of whichever
module is actually running as `__main__` at load time. `main.py` gets this for free because it
does `from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU` and
`from SASRecModules_ori import *` at its own top level and is itself invoked as `__main__`.
`scripts/extract_representations.py` already had the same property incidentally (it also imports
`SASRec` at its own top level to call `cacul_h` directly), which is why it loaded the checkpoint
successfully both on VESSL and in every local test in this file's earlier sections.
`scripts/benchmark_baseline.py` did not import these names into its own top level, so when it
runs as `__main__`, they weren't there.

Fix (verified locally, see below): added
`from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU` and
`from SASRecModules_ori import *` at the top of `scripts/benchmark_baseline.py`, in the same
combination and order as `main.py`. `main.py`, `model/model_interface.py`, and
`rec_model/movielens.pt` itself were not touched.

Verified locally without needing a GPU: reproduced the exact `AttributeError: Can't get
attribute 'SASRec' on <module '__main__' ...>` by running a standalone script that calls
`torch.load('rec_model/movielens.pt', ...)` without importing `SASRec`, then confirmed that
adding the same two import lines used above (as `__main__`-level imports) makes the identical
`torch.load` call succeed, returning a real `SASRec` instance. `scripts/smoke_test_routing.py`'s
full 17-check suite still passes.

## 11f. Paper-aligned-lr OOM check + per-batch memory diagnostics (2026-09-21)

`scripts/benchmark_baseline.py` gained a `--lr` flag (default `8e-4`, unchanged from the released
`train_movielens.sh` value) and a `BatchMemoryDiagnostics` `pl.Callback` that logs, at the start
of every micro-batch, the batch's token length (max = common padded length, mean = actual
non-padded length per sample) and `torch.cuda.memory_allocated/reserved/max_memory_allocated/
max_memory_reserved`. `trainer.fit()` is wrapped in a `try/except RuntimeError` that recognizes a
CUDA OOM message specifically, prints the last logged batch's diagnostics plus the raw CUDA error
(which already contains PyTorch's own allocated/reserved/free breakdown), and exits cleanly
instead of a raw traceback. This exists to check whether the OOM found in the forensic audit
(see the audit report delivered in conversation, not checked into this file) reproduces at
`--lr 1e-4` (the value stated in the paper's Appendix E for MovieLens, vs. the released script's
`8e-4`) — since learning rate does not change activation/gradient tensor sizes, a reproduction at
the same batch/memory point under a different lr would further confirm the OOM is a property of
the released hyperparameters (batch size, sequence length, model size) and not lr-dependent, or
anything introduced by this branch's own changes (already ruled out separately, see §6 of the
forensic audit). No default in `main.py` or any `train_*.sh` script was changed; `--lr` on this
one throwaway script is purely additive and opt-in.

## 11g. Dataset alignment with the paper, and OOM root cause (2026-10-11)

### What the paper actually specifies for the data

iLoRA (arXiv:2408.10159v4, §4): "We request the dataset from LLaRA [9] and maintain exactly the
same experimental settings as described in the original paper. Our study builds upon the main
table from the LLaRA paper, using the results reported in the LLaRA paper directly." The data
definition therefore lives in LLaRA (arXiv:2312.02445, §5.1.1 and Table 1): MovieLens100K; Steam
filtered to users with >= 20 reviews then 1/3 of users and games; chronological 8:1:1
train/val/test split; last 10 interactions kept as history (padded); 20 candidates per sequence
including the correct next item; dataset statistics #Sequence 943 / 11,938 / 1,220 and
#Item 1,682 / 3,581 / 4,606 (MovieLens / Steam / LastFM).

### Provenance check of `data/ref` against LLaRA's official release

LLaRA's README points to https://huggingface.co/datasets/joyliao7777/LLaRA. Comparing git blob
ids / sha256 of the committed files with that release:

| file | vs LLaRA release |
|---|---|
| lastfm: train, Val, Test, id2name | identical |
| movielens: Val, Test, u.item | identical |
| movielens: train_data.df | different bytes, but `seq`, `len_seq`, `next` and the index are identical row-for-row (68,388 rows); iLoRA's copy only adds an int `cluster` column (4 clusters) that no code reads |
| steam: train, Test, id2name | identical to LLaRA's 2024-05-04 Steam update |
| steam: Val_data.df | **missing from iLoRA** (without it Steam training cannot start: `DInterface` loads the val split eagerly) |

Against LLaRA Table 1, MovieLens (754 train users + 94 val + 95 test = 943, 1,682 items) and LastFM
(976 + 122 + 122 = 1,220, 4,606 items) match exactly, with an 8:1:1 split. Steam val/test are
1,194 each (= 10% of 11,938).

### Changes applied to match the paper (= LLaRA's protocol and LLaRA's own code)

- `data/lastfm_data.py`: `negative_sampling` restored to LLaRA's
  `random.sample(canset, cans_num-1) + [next_item]` + shuffle. iLoRA had changed it to
  `random.sample(canset, 1)`, which never contains the ground truth (verified 0/122 on the real
  test set), so HR@1 was structurally 0. Default `data_dir` restored from `lastfm_ctr` to
  `lastfm`. Now 20 candidates with the ground truth in 122/122 val and test sequences.
- `data/data_interface.py`: removed `drop_last=True` from `test_dataloader` (not in LLaRA). With
  batch_size 8 it silently dropped 7 of 95 MovieLens test sequences; the paper's protocol is
  "across all sequences".
- `data/ref/steam/Val_data.df`: added from the LLaRA release (git blob
  `358a8631f8d76a6c67670c4ec0268c16c106bf59`, matching the HF-reported oid and the same release
  as the committed Steam train/test).
- `metrics.json` now also records `test_total_num`, `test_valid_num`, `test_correct_num` and
  `test_hr_over_total` (counts recovered exactly from `calculate_hr1`'s ratios; verified on 2.2M
  integer triples). The original `test_hr` (correct/valid) is unchanged.

Left unchanged deliberately: the `len_seq >= 3` filter (present in LLaRA's code too; it removes no
MovieLens/LastFM val/test rows), and `prompt/artist.txt`'s "10 music artist names" (identical in
LLaRA's prompt; the paper shows no prompt text).

### What the paper's own numbers imply about evaluation

Searching integer (N, valid, correct) with N <= the full test size:
- iLoRA MovieLens 0.9891 / 0.5275 has exactly one solution: N=92, valid=91, correct=48, with
  HR = correct/valid. 92 = 95 with `drop_last` at batch size 4.
- LLaRA MovieLens 0.9684 / 0.4421 (and TALLRec 0.9263 / 0.3895, LLaRA-GRU4Rec LastFM) have
  exactly one solution each: the full test set with HR = correct/total.
So iLoRA Table 1 places an HR computed as correct/valid on 92 sequences next to LLaRA HRs computed
as correct/total on all 95. Under one convention iLoRA's MovieLens HR would be 48/92 = 0.5217
(correct/total) — this is an inference from the reported digits, not something the paper states.

Corrections to the earlier forensic audit: (1) the "five random seeds" sentence (iLoRA App. E,
LLaRA §5.1.2) sits in the conventional-baseline paragraph, so it does not establish that iLoRA's
own rows are 5-seed averages (the MovieLens row is consistent with a single run); (2) LLaRA's
paper reports LLaRA (SASRec) LastFM HR@1 = 0.4508, so in the inconsistent LastFM/LLaRA cell it is
the 8.51% Imp. that disagrees, not the HR; (3) iLoRA's "ChatRec" row is numerically LLaRA's
"GPT-4" row.

### OOM root cause (training, MovieLens, A100 80GB)

- Real Llama-2 tokenizer on the TrainCollater prompts: padded batch length L has mean 272,
  p90 288, p97 ~296, max 322.
- The failed request reads "Tried to allocate 148.00 MiB". torch 2.0's caching allocator prints
  the request rounded up to a multiple of 2 MiB (`get_allocation_size`, kRoundLarge), so the real
  request was in (146, 148] MiB. That matches a fp32 `[batch=8, L, num_moe=4, 4096]` tensor
  (`B_out * Gate` of a 4096-wide MoE-LoRA module; under fp16 autocast `B_out` itself is fp16 and
  the product with the fp32 gate is fp32) for L = 293-296, but also e.g. fp16 logits
  `[8, L, 32005]` for L = 299-303. The message names no tensor, so this identification is an
  inference from size only (corrected 2026-10-23; an earlier version said "exactly one fp32
  tensor" and "no other tensor shape"). The evidence that the per-expert outputs caused the OOM is
  the memory measurement below and the fix removing the OOM, not this one number.
- Upstream `moelora.Linear.forward` builds `B_out = einsum(A_out, B)` of shape
  `[B, L, num_moe, out_features]` in every target module of every layer and keeps it for
  backward (needed for the gradient w.r.t. the gate). Measured with saved-tensor hooks on the
  real upstream class: 803 KiB/token/layer saved by the LoRA branch vs 139 KiB/token/layer for an
  algebraically identical form, lora_B((A_out * Gate).reshape(B, L, r)). At B=8, L=296, 32
  layers that is roughly 25 GiB of fp16 activations on the GPU.
- Typical batches (L ~ 272) fit just under 80 GB; a p97 batch (L ~ 296) does not. Over 37 batches
  the chance of meeting one is ~70%, matching an OOM at batch 37.

Fix status: APPLIED to `model/peft/tuners/moelora.py` (`Linear.forward` only):
- the LoRA branch computes `lora_B((A_out * Gate).reshape(B, L, r)) * scaling` instead of
  `einsum` -> `B_out` -> `(B_out * Gate).sum(-2) * scaling`; the now-unused `Linear.calculate_B`
  is removed (the 8-bit class keeps its own copy, untouched);
- the `except Exception` that turns any error into `torch.zeros_like(x)` now re-raises CUDA OOM,
  so an OOM can no longer silently zero a projection and keep training; every other exception
  keeps the upstream behavior.
`scripts/test_moelora_equivalence.py` loads the upstream file from git (28e4737) next to the
working-tree file and checks, with identical weights: outputs and gradients w.r.t. x, gate,
lora_A, lora_B agree (float64 ~1e-16, float32 ~1e-7, also with lora_dropout=0.1 and a
single-token generation step); saved-for-backward bytes drop by exactly the upstream B_out
tensor; an OOM propagates while other errors still return zeros.

### Will it fit on the A100 80GB? (measured, not estimated from formulas)

The repo's `debug/modeling_llama.py` decoder layer with real dimensions and the real MoE-LoRA
modules was run on CPU with the GPU's dtype flow (bf16 base weights, fp16 autocast; CUDA
autocast's fp32 cross_entropy and its non-intervention in `torch.cat` reproduced), and peak
memory was tracked op-by-op with `torch.distributed._tools.mem_tracker.MemTracker` for
K in {1,2,3} layers and batch 1/2 at L in {296, 322, 350}. A linear model in layers and batch
size predicts all independent K=1 runs exactly, giving per layer: 0.378 GiB constant (the fp16
autocast copies of the frozen bf16 weights, = 202.4M params x 2 B) plus, per sample,
0.213 GiB (upstream) vs 0.119 GiB (rewrite) of activations at L=296 — the difference is exactly
the fp16 B_out tensor. Static memory (exact parameter counts: bf16 base, fp32 LoRA/projector/
router, fp32 gradients, Adam states, rotary caches) is 13.23 GiB. Calibrating to the observed
OOM (36 batches passed, failed at L~296 with 76.12 GiB allocated; the model over-predicts
activations by 9.4%) and using the full 5-epoch length distribution (41,325 simulated batches
with fresh negatives each epoch; padded L mean 273, p99.9 325, max 347):

| | typical batch (L=272) | longest batch of a 5-epoch run (L=347) | batches over the ~76.1 GiB budget |
|---|---|---|---|
| upstream moelora | 71.7 GiB | 86.2 GiB | 2,221 / 41,325 |
| rewritten moelora (repo) | 51.7 GiB | 60.7 GiB (64.2 uncalibrated) | 0 / 41,325 |

Re-running the same measurement with the repo's working-tree moelora.py reproduces the
pre-application numbers to four decimals. Validation/test generation (KV cache ~1.7 GB) and
cluster_hard training (the one-hot gate needs no gradient, so B_out was never kept there) are
below the training peak. Final confirmation is a GPU run of `scripts/benchmark_baseline.py`,
whose per-batch log should show roughly 50-55 GiB at L ~ 270-300.

## 11h. Fresh-workspace bootstrap (2026-10-11)

- `setup_vessl.sh` step 7 now applies `debug/utils.py` and `debug/modeling_llama.py` to the
  installed transformers 4.28.0 (originals kept once as `*.orig`, idempotent, verified by the
  `user_embeds`/`gate_weights` allow-list and a patched import), replacing the earlier
  detect-and-log behavior of §10. The forensic audit showed both files are required: the earlier
  VESSL session applied only `modeling_llama.py`, which is enough for the training-only benchmark
  but would have failed at the first validation `generate()`.
- `bootstrap_vessl.sh` prepares a fresh workspace end to end (conda env, optional Llama-2 download
  from `HF_TOKEN`, setup, preflight, CPU regression test, GPU evaluation smoke test on the test
  split, GPU training-memory benchmark) and prints, but does not start, the full run.
- Image: VESSL requires SSHD in custom images; `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel` has
  none (its official Dockerfile stage installs only ca-certificates/libjpeg/libpng/gcc), so a
  Managed image + the conda env is used, as in the earlier working session.

## 11i. Peak learning rate: the paper's 1e-4 instead of the released 8e-4 (2026-10-12)

The released `train_movielens.sh` uses `--lr 8e-4` (warmup start and floor 8e-6); the paper's
Appendix E states a maximum learning rate of 1e-4 for MovieLens and Steam (2e-4 for LastFM), with
warmup, cosine annealing and mixed precision. The paper gives no warmup start, floor, batch size,
epoch count or LoRA rank, so those keep the released values.

Evidence on the VESSL A100 (MovieLens, seed 1234, batch 8 x 16, fp16 AMP unless noted):

| Run | Peak lr | Result |
| --- | --- | --- |
| first full baseline run (`main.py`) | 8e-4 | 0 valid predictions at the first validation, `loss=nan` for the whole second epoch |
| `diagnose_nan.py` A | 1e-4 | one epoch (517 optimizer steps), no non-finite loss; loss 5.08 -> 0.32; 4 GradScaler skips (steps 1, 2, 77, 98); largest fp16/bf16 module output 22,400 (`layers.30.mlp.down_proj`) |
| `diagnose_nan.py` B, bf16 | 1e-4 | stopped by hand at step 195, past the warmup peak (step 129), no non-finite loss; same batches as A |

The NaN therefore needs lr 8e-4; whether fp16 is also needed (runs C/D) was not tested, since it
does not matter at 1e-4. `run_baseline_movielens.sh`, `run_cluster_hard_movielens.sh` and the
default of `scripts/benchmark_baseline.py` now use `--lr 1e-4`; warmup start and floor stay 8e-6
(the values A and B ran with), precision stays fp16 (the released setting; the paper says
half-precision). The released `train_movielens.sh` / `test_movielens.sh` are left unchanged.
Results produced with these scripts are a paper-hyperparameter reproduction on the released code,
not a run of the released script as published.

## 11j. Measurement, raw-data storage and anomaly tracking (2026-10-23)

Package `measurement/` plus scripts; a run folder's README.txt lists every column.

- **Run folders.** Every measured run writes a new `runs/<name>_<YYYYmmdd-HHMMSS>/` (never reused;
  `--out_dir` must be new or empty): `meta.json` (git commit and `git status`, command line, config,
  seed, precision, max lr, torch / CUDA / GPU, start and end time, every threshold and sample size,
  TensorBoard folder, paths of the later-analysis inputs `gates_test.csv`, representation npz,
  assignment CSVs and `metrics.json`), `micro_batches.csv.gz`, `optimizer_steps.csv.gz`,
  `activation_stats.csv.gz`, `anomalies.csv`, `anomaly_dumps/`, `anomaly_report.md`, `README.txt`,
  and `gpu.csv` when `scripts/gpu_monitor.py` runs. Gzip CSVs are flushed every optimizer step and
  every 16 micro-batches; `measurement.io.read_frame` reads a file cut off by a kill up to its last
  flush. `runs/` is git-ignored; copy folders out before pausing a VESSL workspace.
- **Hooks (`measurement/activation.py`).** Forward hooks on every module whose output (first tensor of
  a tuple) is fp16/bf16, the set behind the earlier `max_abs_lowp_out`; a module called several
  times in one forward is recorded per call (`name#2`, ...). Each call is reduced on the GPU: max
  |value| and its position (found among the 32 rows with the largest row maxima, which contain the
  32 largest elements), the 32 largest values, the output row holding the maximum, the non-finite
  count; on recorded micro-batches (`--stats_every`, default every one, 0 = never) also mean |value|
  and std over all elements (float32) and median / p90 / p99 / p999 of |value| from a
  100,000-element random sample (`--quantile_sample`; exact when the output is smaller). The sample
  uses the monitor's own `torch.Generator` (`--sample_seed`), so the global RNG that dropout uses is
  untouched. Results are stacked and moved to the CPU once per micro-batch. The hooks read detached
  tensors and return nothing.
- **m_t columns.** `micro_batches.csv.gz` keeps the earlier diagnose_nan.py columns and adds m_t with
  its module, layer, module type, sample, dataset sample id, token position / id / string, feature
  dim, sign, first-token and known-massive-dim (1415 / 2533 of a 4096-wide output) flags, and the
  non-finite element count. Token fields are filled only for LLaMA outputs shaped
  `[batch, L, features]` (SASRec and projector outputs are not token-aligned).
- **Anomaly tracker (`measurement/anomaly.py`).** Per module and micro-batch, always (not subject to
  `--stats_every`): `nonfinite` and `over_fp16_max` (> 65,504) give critical; `high`
  (>= 0.5 x 65,504) gives high; `abs` (>= 16,384) and `ratio` (>= 4 x the module's median over the
  previous 200 micro-batches, once 20 exist) give warn; `record` (new per-module maximum, only after
  100 micro-batches, since at the start every module sets a record each time) gives info. All
  thresholds are flags. `warn` is a severity added here (the task listed info / high / critical).
  The fp16 rules are applied to bf16 runs too, labeled by the `precision` column. Dumps
  (`<event_id>.npz` + `.txt`): the token vector at the event, the 32 largest values with (sample,
  token, feature), the sample's input ids and decoded prompt, the max |value| of the other modules
  of the same layer; at most 200 dumps, high and critical always dumped. Terminal lines for warn
  and above (`--print_level`); info events go to anomalies.csv only.
- **Where it runs.** `scripts/diagnose_nan.py` now builds on `measurement.callback.MeasurementCallback`;
  its earlier columns, `--compare`, origin module, first-NaN dumps and early stop are unchanged,
  its output moved from `benchmark_runs/nan_diag/<tag>/` to a run folder, and `--compare` still
  reads the earlier folders. `--max_micro_batches` was added. `main.py --measure` (with the same
  flags) attaches the callback to the real training. The measurement flags are parsed by a separate
  parser and removed before `main.py`'s own parser, so without them `args`, the model hparams and
  the callbacks are exactly as before. TensorBoard logging (L1, L2) is unchanged; its folder is
  recorded in meta.json.
- **Baseline activation.** `scripts/measure_base_activation.py`: forward passes only, on the first N
  (200) micro-batches of the same seeded training dataloader, with (a) LoRA disabled and (b) the
  step-0 MoE-LoRA model (or `--ckpt_path`), fp16 or bf16. `lora_B` starts at zero, so (b) equals (a)
  on the base path. `compare_base_vs_trained.md` (also `scripts/compare_base_vs_trained.py`)
  compares m_t and per-module maxima with a trained run and checks that the batch order matched.
- **Tools.** `scripts/gpu_monitor.py` (L3, separate process, stops when the run ends),
  `scripts/plot_live.py --watch` (L4, warns on non-finite losses and critical events),
  `analysis/plot_measurement.py` (figures from the folder alone), `scripts/inspect_anomaly.py`,
  `scripts/finalize_run.py` (README and report after a hard kill).
- **Checks.** `scripts/test_measurement.py` (CPU, synthetic data): a planted value is found at the
  exact (sample, token, feature) with the right rules and dump; sampled quantiles within 0.5% of the
  exact ones (exact for small outputs); loss, gradients and RNG state bit-identical with and without
  hooks (float32, bf16); a killed writer leaves every flushed row; a toy folder yields every file,
  figure and report. `scripts/verify_measurement_gpu.py` (VESSL): the same bit-identity on the real
  model in fp16 and bf16 (with an off-vs-off run that shows whether the GPU run is deterministic at
  all), hook overhead, run-folder size per epoch, a 50-micro-batch run with figures, a SIGKILL test.
- **Open item: gates of all users.** Gates are exported only from `generate()` on validation / test
  (`MInterface._record_gate_export`, 95 MovieLens test sequences). The gate depends only on the
  SASRec user representation and the router (`gate = router(encode_users(seq, len_seq))`), so the
  gate of every train / validation / test sequence needs only `rec_model` and the trained router
  weights from a checkpoint, no LLaMA: a CPU script that loads `router.*` from `last.ckpt`, runs
  `rec_model.cacul_h` over each split and writes `gates_<split>.csv` in the existing format.

## 12. VESSL run log

Not run yet as of this writing (implementation phase, local Windows environment only). This
section will be filled with actual dates, commands, dependency versions, and elapsed times once
someone executes `run_all_vessl.sh` on the real VESSL A100 instance. Nothing here is a
projection or estimate.
