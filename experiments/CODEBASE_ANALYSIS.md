# Codebase Analysis (Phase 1)

All findings below are from direct reading of the actual code at the commit checked out on
`exp/gate-cluster` (fork of `AkaliKong/iLoRA`). File paths and line numbers refer to the
repository root. No behavior is assumed; every claim below was verified by reading the source
or, where marked `[VERIFIED LOCALLY]`, by running real code against the shipped MovieLens data
and the shipped `rec_model/movielens.pt` checkpoint on CPU (Windows, torch 2.6 cpu build).

## A/B. MovieLens train / test entry point

- Single entry point for both: [main.py](../main.py). `--mode train` vs `--mode test` selects
  `trainer.fit(...)` or `trainer.test(...)` (main.py:67-70).
- MovieLens is invoked via [train_movielens.sh](../train_movielens.sh) and
  [test_movielens.sh](../test_movielens.sh), which both pass `--router share --gating Dense
  --dataset movielens_data --data_dir data/ref/movielens --cans_num 20 --rec_embed SASRec
  --llm_tuning moelora --rec_model_path ./rec_model/movielens.pt`.
- **Important**: `--router share` is the value actually used by the released MovieLens scripts,
  even though the CLI default in main.py:138 is `unshare`. The `unshare` code path in
  [model/model_interface.py](../model/model_interface.py) `generate()` calls `self.router(...)`
  without `self.router` ever having been constructed (it is only built when
  `router == 'share'`, model_interface.py:30-31) — `unshare` is dead/broken in this codebase.
  All experiment work in this repo assumes `--router share` (the only functional configuration),
  matching the released scripts.

## C. Dataset split structure

- [data/movielens_data.py](../data/movielens_data.py) `MovielensData` reads three pickled
  pandas DataFrames per split: `train_data.df`, `Val_data.df`, `Test_data.df`
  (data/movielens_data.py:52-58).
- `[VERIFIED LOCALLY]` Actual columns of `Test_data.df` (and by construction the other splits):
  `['seq', 'len_seq', 'next']`. `seq` is a list of `(item_id, rating)` tuples (padded to length
  10), `next` is a `(item_id, rating)` tuple. **There is no `user_id` column anywhere in the
  MovieLens data** — sessions are anonymous. Rows with `len_seq < 3` are dropped
  (data/movielens_data.py:79).
- `__getitem__(self, i)` (data/movielens_data.py:25) uses the positional row index `i` into the
  per-split DataFrame. This row index is the only stable, reproducible sample identifier
  available and is what this experiment uses as `sample_id`. `user_id` is recorded as
  `NOT_AVAILABLE` in all exports, per spec.

## D/E. Batch structure and sample identity

- Collation is [data/data_interface.py](../data/data_interface.py) `TrainCollater.__call__`
  (data_interface.py:30). It builds a `dict` with keys `tokens, seq, cans, len_seq, len_cans,
  item_id[, flag | correct_answer, cans_name]`. **No sample index is currently threaded through
  the collated batch.** To attach `sample_id` to gate/representation exports we add a new
  `sample_idx` field end-to-end: `MovielensData.__getitem__` (return dict) →
  `TrainCollater.__call__` (both the `train=True` and `train=False` branches) → batch dict key
  `"sample_idx"`. This is purely additive (one new key), so it cannot change any existing
  dynamic-routing numerics.

## F/G. SASRec representation function and dimension

- `MInterface.encode_users` (model/model_interface.py:422-429) calls
  `self.rec_model.cacul_h(seq, len_seq)` for `rec_embed == "SASRec"`.
- `SASRec.cacul_h` ([recommender/A_SASRec_final_bce_llm.py:202-217](../recommender/A_SASRec_final_bce_llm.py#L202-L217))
  returns `state_hidden = extract_axis_1(ff_out, len_states - 1)`, shape `[batch, 1, hidden_size]`
  (the `extract_axis_1` helper unsqueezes to a length-1 sequence axis,
  recommender/A_SASRec_final_bce_llm.py:13-18).
- `[VERIFIED LOCALLY]` Loaded `rec_model/movielens.pt` on CPU
  (`torch.load(..., map_location="cpu", weights_only=False)`): it is a real `SASRec` instance
  with `item_embeddings: Embedding(1683, 64)`, `positional_embeddings: Embedding(10, 64)`,
  i.e. **hidden_size = 64**, max sequence length (`state_size`) = 10, `item_num` = 1682. This
  matches `--rec_size 64` (main.py:107) and the router's hardcoded `input_size=64`
  (model/router/nlpr.py:83-84: `build_router(...) = NLPRecommendationRouter(..., input_size=64,
  num_experts=4)`). The experiment code does not hardcode 64; it reads the dimension from the
  tensor shape at runtime, per spec.
- `MInterface.encode_users` (model_interface.py:428) actually **returns the raw SASRec
  representation `user_rec_embs`**, not the LLM-space projection `user_txt_embs` that it also
  computes — this is what all downstream code calls "`user_embeds`". So `user_embeds` in
  `forward`/`generate` is the raw, frozen SASRec output, shape `[batch, 1, 64]`.

## H/I/J/K. Router location, call sites, gate computation, gate tensor shape

- Router class: `NLPRecommendationRouter` in [model/router/nlpr.py](../model/router/nlpr.py)
  (a small ResNet-style CNN over the length-1 "sequence" axis, ending in
  `GateFunction` = `Linear(64,4)` + softmax, nlpr.py:39-45, 61-62). Constructed once via
  `build_router()` (nlpr.py:83-84, hardcodes `num_experts=4` regardless of `--num_moe`) and
  assigned to `self.router` in `MInterface.__init__` (model_interface.py:30-31), only when
  `router == 'share'`.
- Call sites: `MInterface.forward` (model_interface.py:48) and `MInterface.generate`
  (model_interface.py:73, inside the `router == 'share'` branch that returns before the
  unreachable `unshare` code below it). **`generate()` is used by both `validation_step` and
  `test_step`** (model_interface.py:159-160, 194-195), so line 73 is the single real call site
  during evaluation. This is the collection point this experiment instruments — nowhere else.
- Gate tensor shape at that call site: `NLPRecommendationRouter.forward` ends with
  `out.unsqueeze(1)` (nlpr.py:80), so `gate_weights.shape == [batch, 1, 4]`. Values sum to 1 over
  the last dim (softmax, nlpr.py:45).

## L/M. How gate_weights reaches the MoE-LoRA layers; sharing mechanism

- `gate_weights` is passed as a kwarg into `self.llama_model(..., gate_weights=gate_weights)`
  (model_interface.py:56, 87). The PEFT wrapper `MoeLoraModel.forward`/`generate`
  ([model/peft/tuners/moelora.py:195-217](../model/peft/tuners/moelora.py#L195-L217)) does
  `self.gate_weights.clear(); self.gate_weights.extend([kwargs['gate_weights']])`, i.e. it
  stores the *same* one gate tensor for the whole forward/generate call in a single-element
  Python `list` that lives on the `MoeLoraModel` instance.
- Every target `Linear` (moelora.py `Linear.__init__`, moelora.py:715-716) was constructed with
  `gate_weights=self.gate_weights` — the *same list object* — so every LoRA-augmented linear
  layer in every transformer block reads `self.gate_weights[0]` in its own `forward`
  (moelora.py:759, 780). That is the "shared router" mechanism: one gate vector, computed once
  per forward pass from `user_embeds`, is broadcast unchanged to all target `Linear` layers via a
  shared mutable list reference — there is no per-layer routing weight in the active code path.
  (Each `MoeLoraLayer` also owns a per-layer `self.gating[adapter_name]` submodule, created in
  `update_layer`, moelora.py:604-606, but it is **not called** in the live `Linear.forward` —
  the call is commented out, moelora.py:779 — so it is vestigial/unused when `router == 'share'`.)

## N/O. Checkpoint save / load

- Save: `MInterface.on_save_checkpoint` (model_interface.py:296-309). With `--save part`
  (the default), it drops `optimizer_states` and any `state_dict` key whose parameter has
  `requires_grad == False`. Because `mark_only_lora_as_trainable` (moelora.py:554-557) leaves
  only `lora_`- and `gating`-named params trainable, plus the projector and `self.router`
  (added explicitly to the optimizer, model_interface.py:227-236), the checkpoint keeps: LoRA
  A/B weights, the per-layer `gating` submodules (unused but still trainable/saved), the MLP
  projector, and the shared `router` weights. The frozen 7B LLaMA backbone and the frozen
  SASRec model are **not** saved.
- Load: `main.py:42-46`, `torch.load(ckpt_path, map_location='cpu')` then
  `model.load_state_dict(ckpt['state_dict'], strict=False)` (non-strict, since the checkpoint is
  a partial state dict).

## P/Q. Evaluation metric

- `MInterface.calculate_hr1` (model_interface.py:468-492): for each generated sample, checks
  which of the `cans` (candidate item title strings) are literal substrings of the generated
  text. If **exactly one** candidate matches, the sample counts toward `valid_num`; if that one
  match equals the ground truth `real`, it also counts toward `correct_num`.
  `valid_ratio = valid_num/total_num`; `hr1 = correct_num/valid_num` (0 if `valid_num==0`).
  Logged metric names (model_interface.py:182-184, 218-220): `val_prediction_valid`, `val_hr`,
  `metric` (train-time validation), and `test_prediction_valid`, `test_hr`, `metric` (test time),
  where `metric = hr * valid_ratio` in both cases. **These are the only metric names that exist
  in this codebase** — "ValidRatio" / "HR@1" in the spec's comparison table map to
  `test_prediction_valid` and `test_hr` respectively; `CombinedMetric` maps to `metric`. There is
  no separate "HR@1" computed independently of the valid-generation filter.

## R/S. Is SASRec frozen or trained jointly?

- **Frozen (Case A).** `MInterface.load_rec_model` (model_interface.py:406-412) calls
  `self.rec_model.eval()` and sets `requires_grad = False` on every parameter. `self.rec_model`
  is never added to any optimizer param group (model_interface.py:227-236 lists only
  `projector`, `router`, and non-`gating` LLaMA params).
- **Caveat found during analysis (not previously documented anywhere in the repo):** calling
  `self.rec_model.eval()` once at load time does not permanently pin it in eval mode. PyTorch
  Lightning's `Trainer.fit` calls `model.train()` at the start of every training epoch, which
  recurses into **all** submodules, including `self.rec_model`, and would silently flip its
  `Dropout` layers (`dropout=0.1` throughout `SASRec`, recommender/A_SASRec_final_bce_llm.py)
  back into train mode even though its weights never update. This means, in the *unmodified
  baseline*, the SASRec representation used during a training epoch can be stochastic (dropout
  noise) even though the weights are fixed — this is pre-existing behavior in the upstream code,
  not something introduced here, and this experiment does not change it for the `dynamic` path
  (see next section for why this matters for `cluster_hard`).

## T/U. LoRA rank per expert; effective meaning of lora_r=8, num_moe=4

- `MoeLoraLayer.update_layer` (moelora.py:596-621) builds **one** `lora_A: Linear(in, r)` and
  **one** `lora_B: Linear(r, out)` for the whole adapter, with `r = lora_r = 8`. It does **not**
  allocate separate A/B matrices per expert.
- The expert split happens only in `Linear.forward` (moelora.py:756-799):
  `A_out = lora_A(x).reshape(batch, seq, num_moe, -1)` — the single rank-8 output is reshaped
  into `num_moe=4` chunks of size `8/4 = 2`. `calculate_B` (moelora.py:750-753) reshapes
  `lora_B.weight` the same way and does a per-chunk matmul. So **each of the 4 experts has
  effective rank 2**, not rank 8. `self.scaling[adapter] = lora_alpha / (r // num_moe)`
  (moelora.py:618) already reflects this (scaling uses the per-expert rank, 2, not 8).
- Consequence (this is exactly the confounder the spec asks to document, not fix): with
  `lora_r=8, num_moe=4`, **dynamic routing** mixes 4 rank-2 experts with a learned soft gate,
  while **cluster-hard routing with a one-hot gate activates exactly one rank-2 expert** per
  sample. Any accuracy gap between the two cannot be cleanly separated from "one active rank-2
  expert vs. a soft mixture of four rank-2 experts" (i.e., active-capacity confounder). Per spec
  §13 this is recorded as a limitation and **not** worked around today (no rank/parameter budget
  changes in the primary experiment).

## V. train_movielens.sh vs. paper Appendix hyperparameters

Not verified against the paper's Appendix in this pass (no network fetch performed as part of
static local analysis). Recorded as `VESSL_REQUIRED` / follow-up in
`experiments/IMPLEMENTATION_NOTES.md`; per spec §7 the released shell-script values are used
regardless of any discrepancy, so this does not block implementation. Values actually used
(from `train_movielens.sh` / `test_movielens.sh`, taking precedence over the argparse defaults
in main.py where they differ): `batch_size=8`, `accumulate_grad_batches=16`, `cans_num=20`,
`max_epochs=5`, `lr=8e-4`, `lr_warmup_start_lr=8e-6`, `lr_decay_min_lr=8e-6`, `router=share`,
`gating=Dense`, `llm_tuning=moelora`, `rec_embed=SASRec`. Argparse-only defaults kept as-is:
`lora_r=8`, `lora_alpha=32`, `lora_dropout=0.1`, `num_moe=4`, `seed=1234`.

## W. README debug workarounds

[README.md](../README.md) documents two known environment workarounds unrelated to this
experiment's code changes: if `transformers/generation/utils.py` errors, replace it with
[debug/utils.py](../debug/utils.py); if `transformers/models/llama/modeling_llama.py` errors,
replace it with [debug/modeling_llama.py](../debug/modeling_llama.py). These are patches to the
*installed* `transformers` package (version pinned to `4.28.0` in requirements.txt), not to this
repository's own code. `setup_vessl.sh` checks whether these replacements are needed and applies
them only if the installed `transformers` files trigger the documented error, recording the
decision in `experiments/IMPLEMENTATION_NOTES.md` rather than silently patching.

## Other structural facts relevant to later phases

- **PEFT is vendored, not pip-installed**: `model/model_interface.py:17` imports
  `from .peft import ...` — this repo ships its own copy of `peft` under
  [model/peft/](../model/peft/) (based on peft 0.3.0 vintage, modified to add `MoeLoraConfig` /
  `MoeLoraModel`). The pip `peft==0.3.0` in requirements.txt is not actually used by
  `model_interface.py`.
- **Gating choices**: `--gating` selects among `Dense/topK/MLP/Drop/MLP_noise/Noise` in
  [model/peft/tuners/gating.py](../model/peft/tuners/gating.py) — but as established above
  (H/I/J), these per-layer `gating` submodules are dead code in the live `router=share` forward
  path; only `--gating Dense` is used by the released scripts and this experiment does not
  exercise the other choices.
- `[VERIFIED LOCALLY]` Local Windows environment has `Python 3.13.2`, `pandas 2.2.3`,
  `numpy 2.2.3`, `scikit-learn 1.6.1`, `matplotlib 3.10.1`, `torch 2.6.0+cpu` available (none of
  these were installed for this task; they were already present). This is materially better
  than "no dependencies" — it means `rec_model/movielens.pt` (a real, frozen, pretrained
  `SASRec` checkpoint, 952 KB) and the real MovieLens session data
  (`data/ref/movielens/{train,Val,Test}_data.df`) can be loaded and run **for real** on CPU
  locally (`torch.load(..., weights_only=False)` is required due to the newer local torch
  defaulting `weights_only=True`). This was verified directly: loading `rec_model/movielens.pt`
  reproduces the exact architecture described above. Consequently `scripts/extract_representations.py`
  (SASRec-only, no LLaMA involved) is designed to be genuinely tested end-to-end locally against
  real data, not just synthetic tensors — everything that touches the 7B LLaMA model or GPU
  training remains `VESSL_REQUIRED`.
- `rec_model/movielens.pt` vs `rec_model/SASRec_movielens.pt`: the scripts use
  `./rec_model/movielens.pt` (952 KB); there is also a separate `SASRec_movielens.pt` file in the
  same directory. This experiment does not change which checkpoint the released scripts point
  to.
- No `user_id`/no existing gate-export/no clustering code exists anywhere in the current
  repository — everything in Phases 3-8 is new code, added alongside the untouched original
  files.
