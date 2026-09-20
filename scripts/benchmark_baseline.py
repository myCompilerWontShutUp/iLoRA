"""One-time dry-run to measure real per-batch training speed on the actual VESSL A100 before
committing to the full run_all_vessl.sh pipeline (two full 5-epoch trainings).

Runs the SAME model/training configuration as the released MovieLens baseline
(run_baseline_movielens.sh / train_movielens.sh): routing_mode=dynamic, batch_size=8,
accumulate_grad_batches=16, precision=16, seed=1234, num_moe=4, lora_r=8 -- but executes only a
fixed number of real forward/backward micro-batches (default 256) and nothing else:

- no validation, no test, no gate export, no analysis (limit_val_batches=0, and validation_step /
  test_step / gate export are simply never invoked)
- no checkpoint is ever written (enable_checkpointing=False, no callbacks)
- all output goes under --benchmark_dir (default benchmark_runs/baseline_dry_run), never under
  checkpoints/baseline, outputs/baseline, results/baseline, or logs/

Every number printed at the end is explicitly labeled MEASURED or ESTIMATED. Nothing here is
written to any file under results/ -- this is a console-only, throwaway diagnostic run, and it
does not change anything about how run_baseline_movielens.sh / run_cluster_hard_movielens.sh
actually train.
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
    args = p.parse_args(cli_args)

    # Everything below matches run_baseline_movielens.sh / train_movielens.sh exactly. Do not
    # change these to "tune" the benchmark -- that would defeat the point of measuring the
    # actual baseline's own speed.
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
    args.lr = 8e-4
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

    print("=== benchmark_baseline.py: dry-run speed measurement ===")
    print("This does NOT touch checkpoints/baseline, outputs/baseline, results/baseline, or logs/.")
    print(f"Benchmark-only directory: {args.benchmark_dir}")
    print(f"Micro-batches to run: {args.num_train_batches} (no validation, no test, no checkpoint save)")
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
        callbacks=[],
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    t1 = time.time()
    trainer.fit(model=model, datamodule=data_module)
    train_seconds = time.time() - t1
    optimizer_steps = trainer.global_step

    sec_per_batch = train_seconds / args.num_train_batches

    print("")
    print("=== Results ===")
    print(f"[MEASURED] model loading time: {model_load_seconds:.1f}s")
    print(f"[MEASURED] {args.num_train_batches} training batches elapsed time: {train_seconds:.1f}s")
    print(f"[MEASURED] sec/batch: {sec_per_batch:.4f}s")
    print(f"[MEASURED] optimizer steps processed: {optimizer_steps}")
    print("  (note: sec/batch includes one-time DataLoader worker startup inside this run, "
          "so it is a slightly conservative/high estimate of steady-state speed.)")

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
