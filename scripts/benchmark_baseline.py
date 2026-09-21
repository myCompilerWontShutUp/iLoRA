"""One-time dry-run to measure real per-batch training speed and GPU memory behavior on the
actual VESSL A100 before committing to the full run_all_vessl.sh pipeline (two full 5-epoch
trainings).

Runs the SAME model/training configuration as the released MovieLens baseline
(run_baseline_movielens.sh / train_movielens.sh): routing_mode=dynamic, batch_size=8,
accumulate_grad_batches=16 (effective batch size 128), precision=16, seed=1234, num_moe=4,
lora_r=8 -- but executes only a fixed number of real forward/backward micro-batches (default
256) and nothing else:

- no validation, no test, no gate export, no analysis (limit_val_batches=0, and validation_step /
  test_step / gate export are simply never invoked)
- no checkpoint is ever written (enable_checkpointing=False, no callbacks besides the
  diagnostics callback below)
- all output goes under --benchmark_dir (default benchmark_runs/baseline_dry_run), never under
  checkpoints/baseline, outputs/baseline, results/baseline, or logs/

`--lr` defaults to 8e-4, matching the released train_movielens.sh exactly (unchanged default --
this script's baseline behavior is untouched). Pass `--lr 1e-4` to instead match the value the
paper's Appendix E states for MovieLens (the released script's own default remains 8e-4; this
flag never changes it, it only lets this one throwaway run use a different value). Every other
hyperparameter matches the released code (paper gives no number for lora_r, so the released
default of 8 is kept, per experiments/CODEBASE_ANALYSIS.md/IMPLEMENTATION_NOTES.md).

A `BatchMemoryDiagnostics` callback logs, at the start of every micro-batch: batch index, input
token length (max = the batch's common padded length, mean = average non-padded length per
sample via the attention mask), and `torch.cuda.memory_allocated/reserved/max_memory_allocated/
max_memory_reserved`. If CUDA raises an out-of-memory error, it is caught, the last logged
batch's diagnostics plus the raw CUDA error message (which itself reports allocated/reserved/free
at the point of failure) are printed, and the script exits cleanly instead of crashing with a
raw traceback -- this is what lets one run be compared directly against another (e.g. a prior
run at lr=8e-4 vs. this run at lr=1e-4) by batch index and memory reached, not just "it OOM'd".

Every number printed at the end is explicitly labeled MEASURED or ESTIMATED. Nothing here is
written to any file under results/ -- this is a console-only, throwaway diagnostic run, and it
does not change anything about how run_baseline_movielens.sh / run_cluster_hard_movielens.sh
actually train, nor any hyperparameter default in main.py / train_movielens.sh.
"""
import argparse
import os
import sys
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.model_interface import MInterface
from data.data_interface import DInterface
# Pickle compatibility, exactly mirroring main.py's top-level imports: rec_model/movielens.pt was
# pickled with these classes resolvable as `__main__.SASRec` etc. (the module that happens to be
# `__main__` when torch.load unpickles it). main.py gets this "for free" because it imports these
# names at its own top level and is itself invoked as `__main__`; this script needs the identical
# import here for the same reason, since it is a different `__main__` module. Without this,
# MInterface.load_rec_model()'s torch.load(rec_model_path) fails with
# AttributeError: module '__main__' has no attribute 'SASRec'. See
# experiments/CODEBASE_ANALYSIS.md and experiments/IMPLEMENTATION_NOTES.md for why this script
# duplicates a subset of main.py's imports instead of importing from main.py itself.
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU  # noqa: F401
from SASRecModules_ori import *  # noqa: F401,F403

GiB = 1024 ** 3


class BatchMemoryDiagnostics(pl.Callback):
    """Logs per-micro-batch token length and CUDA memory stats, purely for OOM diagnosis.

    Read-only: never touches gradients, the optimizer, or any tensor value, so it cannot change
    training numerics. `on_train_batch_start` fires before that batch's forward/backward, so if
    CUDA OOMs during batch N, `self.last` still holds batch N's own pre-batch snapshot -- exactly
    the "OOM 직전 batch" diagnostics requested.
    """

    def __init__(self):
        self.last = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        input_ids = batch["tokens"].input_ids
        attention_mask = batch["tokens"].attention_mask
        token_len_max = int(input_ids.shape[1])  # padding="longest" -> this batch's common padded length
        token_len_mean = float(attention_mask.sum(dim=1).float().mean().item())  # actual non-pad length

        stats = {
            "batch_number": batch_idx + 1,  # 1-indexed, matches "37/256" style reporting
            "token_len_max": token_len_max,
            "token_len_mean": token_len_mean,
        }
        if torch.cuda.is_available():
            stats.update({
                "memory_allocated_gib": torch.cuda.memory_allocated() / GiB,
                "memory_reserved_gib": torch.cuda.memory_reserved() / GiB,
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / GiB,
                "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / GiB,
            })
        self.last = stats
        if torch.cuda.is_available():
            print(f"[batch {stats['batch_number']}] token_len max={token_len_max} "
                  f"mean={token_len_mean:.1f} | allocated={stats['memory_allocated_gib']:.2f}GiB "
                  f"reserved={stats['memory_reserved_gib']:.2f}GiB "
                  f"max_allocated={stats['max_memory_allocated_gib']:.2f}GiB "
                  f"max_reserved={stats['max_memory_reserved_gib']:.2f}GiB")
        else:
            print(f"[batch {stats['batch_number']}] token_len max={token_len_max} mean={token_len_mean:.1f} "
                  "(no CUDA device visible, memory stats unavailable)")


def build_args(cli_args):
    """Mirror main.py's argparse defaults for exactly the hyperparameters the released
    MovieLens baseline uses. This duplicates a subset of main.py's parser (rather than
    importing from it) because main.py has no importable entry point and is intentionally not
    modified as part of adding this benchmark."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llm_path', required=True)
    p.add_argument('--rec_model_path', default='./rec_model/movielens.pt')
    p.add_argument('--data_dir', default='data/ref/movielens')
    p.add_argument('--prompt_path', default='./prompt/movie.txt')
    p.add_argument('--num_train_batches', type=int, default=256,
                    help='Number of real forward/backward micro-batches to run (not optimizer steps).')
    p.add_argument('--benchmark_dir', default='benchmark_runs/baseline_dry_run',
                    help='Dedicated, throwaway directory for this benchmark run only.')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=8e-4,
                    help='Peak learning rate. Default 8e-4 matches the released train_movielens.sh '
                         'exactly (unchanged). Pass 1e-4 to instead match the paper Appendix E value '
                         'for MovieLens; this only affects this throwaway run, never the released '
                         'script or main.py default.')
    args = p.parse_args(cli_args)

    # Everything below matches run_baseline_movielens.sh / train_movielens.sh exactly (except
    # --lr, which is intentionally CLI-overridable above; its default is still the released
    # value). Do not change the rest to "tune" the benchmark -- that would defeat the point of
    # measuring the actual baseline's own speed/memory behavior.
    args.dataset = 'movielens_data'
    args.router = 'share'
    args.gating = 'Dense'
    args.routing_mode = 'dynamic'
    args.batch_size = 8
    args.accumulate_grad_batches = 16
    args.cans_num = 20
    args.rec_embed = 'SASRec'
    args.llm_tuning = 'moelora'
    args.loss = 'lm'
    args.lr_warmup_start_lr = 8e-6
    args.lr_decay_min_lr = 8e-6
    args.max_epochs = 5  # matches the real baseline; only shapes the LR schedule here, see below
    args.seed = 1234
    args.num_moe = 4
    args.lora_r = 8
    args.lora_alpha = 32
    args.lora_dropout = 0.1
    args.peft_dir = None
    args.peft_config = None
    args.model_name = 'mlp_projector'
    args.rec_size = 64
    args.weight_decay = 1e-5
    args.lr_scheduler = 'cosine'
    args.no_augment = False
    args.padding_item_id = 1682
    args.save = 'part'
    args.export_gates = False
    args.results_dir = os.path.join(args.benchmark_dir, 'results')
    args.cluster_model_path = None
    args.cluster_assignment_dir = 'results/clustering'
    args.cluster_expert_mapping = '0:0,1:1,2:2,3:3'
    args.ckpt_dir = os.path.join(args.benchmark_dir, 'checkpoints')
    args.output_dir = os.path.join(args.benchmark_dir, 'outputs')
    args.log_dir = 'benchmark_dry_run_logs'
    args.ckpt_path = None
    args.local_rank = 3
    args.if_rand = False
    return args


def main():
    torch.multiprocessing.set_start_method('spawn')
    args = build_args(sys.argv[1:])

    os.makedirs(args.benchmark_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=== benchmark_baseline.py: dry-run speed / memory measurement ===")
    print("This does NOT touch checkpoints/baseline, outputs/baseline, results/baseline, or logs/.")
    print(f"Benchmark-only directory: {args.benchmark_dir}")
    print(f"Micro-batches to run: {args.num_train_batches} (no validation, no test, no checkpoint save)")
    print(f"lr={args.lr} (released train_movielens.sh default is 8e-4; paper Appendix E states 1e-4)")
    print("")
    print("Reference point from a prior released-code-lr (8e-4) run on this same VESSL A100:")
    print("  37/256 batches completed, ~2.04 sec/batch, OOM with 76.12 GiB allocated / "
          "78.61 GiB reserved (79.25 GiB total capacity), failed to allocate an additional 148 MiB.")
    print("")

    pl.seed_everything(args.seed)

    t0 = time.time()
    model = MInterface(**vars(args))
    model_load_seconds = time.time() - t0
    print(f"[MEASURED] model loading time: {model_load_seconds:.1f}s")

    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))
    n_train_samples = len(data_module.trainset)
    print(f"[MEASURED] train split size: {n_train_samples} samples")

    steps_per_epoch = n_train_samples // args.batch_size  # drop_last=True in train_dataloader
    full_run_max_steps = max(steps_per_epoch * args.max_epochs // args.accumulate_grad_batches, 1)

    diagnostics = BatchMemoryDiagnostics()
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        amp_backend='native',
        max_epochs=1,
        max_steps=full_run_max_steps,
        limit_train_batches=args.num_train_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        callbacks=[diagnostics],
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    t1 = time.time()
    oom_error = None
    try:
        trainer.fit(model=model, datamodule=data_module)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom_error = e
        else:
            raise
    train_seconds = time.time() - t1
    optimizer_steps = trainer.global_step

    print("")
    print("=== Results ===")
    print(f"[MEASURED] model loading time: {model_load_seconds:.1f}s")

    if oom_error is not None:
        last = diagnostics.last or {}
        print(f"[MEASURED] CUDA OUT OF MEMORY at batch {last.get('batch_number', '?')}/"
              f"{args.num_train_batches} after {train_seconds:.1f}s "
              f"({optimizer_steps} optimizer step(s) completed before the OOM'ing batch).")
        print(f"[MEASURED] OOM'ing batch's token length: max={last.get('token_len_max', '?')} "
              f"mean={last.get('token_len_mean', float('nan')):.1f}" if 'token_len_mean' in last
              else "[MEASURED] OOM'ing batch's token length: unavailable")
        if 'memory_allocated_gib' in last:
            print(f"[MEASURED] memory just before the OOM'ing batch started: "
                  f"allocated={last['memory_allocated_gib']:.2f}GiB "
                  f"reserved={last['memory_reserved_gib']:.2f}GiB "
                  f"max_allocated={last['max_memory_allocated_gib']:.2f}GiB "
                  f"max_reserved={last['max_memory_reserved_gib']:.2f}GiB")
        print("[MEASURED] raw CUDA error (contains PyTorch's own allocated/reserved/free breakdown "
              f"at the moment of failure):\n{oom_error}")
        print("")
        print("=== Comparison ===")
        print(f"  this run   (lr={args.lr}): OOM at batch {last.get('batch_number', '?')}/{args.num_train_batches}")
        print("  prior run  (lr=8e-4):      OOM at batch 37/256, ~2.04 sec/batch, "
              "76.12 GiB allocated / 78.61 GiB reserved")
        print("  If both runs OOM at a similar batch/memory point despite the different learning "
              "rate, that is evidence the OOM is independent of lr (as expected -- lr only scales "
              "the optimizer update, it does not change activation or gradient tensor sizes) and "
              "is instead driven by batch composition/model size/sequence length, matching "
              "experiments/IMPLEMENTATION_NOTES.md's OOM analysis.")
        return

    sec_per_batch = train_seconds / args.num_train_batches
    print(f"[MEASURED] {args.num_train_batches} training batches completed with NO OOM, "
          f"elapsed time: {train_seconds:.1f}s")
    print(f"[MEASURED] sec/batch: {sec_per_batch:.4f}s")
    print(f"[MEASURED] optimizer steps processed: {optimizer_steps}")
    print("  (note: sec/batch includes one-time DataLoader worker startup inside this run, "
          "so it is a slightly conservative/high estimate of steady-state speed.)")
    if diagnostics.last and 'memory_allocated_gib' in diagnostics.last:
        print(f"[MEASURED] memory at the last completed batch: "
              f"allocated={diagnostics.last['memory_allocated_gib']:.2f}GiB "
              f"reserved={diagnostics.last['memory_reserved_gib']:.2f}GiB "
              f"max_allocated={diagnostics.last['max_memory_allocated_gib']:.2f}GiB "
              f"max_reserved={diagnostics.last['max_memory_reserved_gib']:.2f}GiB")
    print("")
    print("=== Comparison ===")
    print(f"  this run   (lr={args.lr}): completed all {args.num_train_batches} batches, no OOM")
    print("  prior run  (lr=8e-4):      OOM at batch 37/256, ~2.04 sec/batch, "
          "76.12 GiB allocated / 78.61 GiB reserved")
    print("  If this run completes cleanly while the lr=8e-4 run OOM'd, that would suggest the "
          "OOM is sensitive to something that happened to differ between the two runs' batch "
          "compositions (dataloader shuffling is seeded, so with the same seed and batch_size "
          "the batch contents should be identical regardless of lr -- an unexpected difference "
          "here would itself be worth investigating).")

    full_micro_batches = steps_per_epoch * 5
    estimated_baseline_seconds = full_micro_batches * sec_per_batch
    estimated_baseline_hours = estimated_baseline_seconds / 3600
    print(f"[ESTIMATED] baseline total training time for {n_train_samples} train samples, "
          f"batch_size={args.batch_size}, 5 epochs ({full_micro_batches} micro-batches): "
          f"{estimated_baseline_seconds:.0f}s (~{estimated_baseline_hours:.2f}h). "
          "Linear extrapolation from the measured sec/batch above only; does NOT include "
          "validation/test/gate-export/analysis time for the real baseline run.")

    estimated_total_seconds = estimated_baseline_seconds * 2
    estimated_total_hours = estimated_total_seconds / 3600
    print(f"[ESTIMATED] full experiment (baseline + cluster_hard), assuming cluster_hard trains "
          f"at the same measured speed: {estimated_total_seconds:.0f}s (~{estimated_total_hours:.2f}h). "
          "This is a simple 2x of the baseline estimate above, not a separate measurement of "
          "cluster_hard's actual speed.")


if __name__ == "__main__":
    main()
