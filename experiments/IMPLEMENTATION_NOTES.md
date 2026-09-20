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

## 12. VESSL run log

Not run yet as of this writing (implementation phase, local Windows environment only). This
section will be filled with actual dates, commands, dependency versions, and elapsed times once
someone executes `run_all_vessl.sh` on the real VESSL A100 instance. Nothing here is a
projection or estimate.
