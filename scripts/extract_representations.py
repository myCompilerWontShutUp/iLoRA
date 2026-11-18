"""Extract frozen SASRec user representations for MovieLens train/validation/test splits.

This script deliberately never imports LlamaForCausalLM, the PEFT wrapper, or
pytorch_lightning: `SASRec.cacul_h` is a frozen, pretrained function of `(seq, len_seq)` alone
(see experiments/CODEBASE_ANALYSIS.md sections F/G/R), so representations can be computed
standalone, on CPU, without any of the 7B-model machinery. That is also what lets this script be
tested for real, locally, against the real shipped `rec_model/movielens.pt` checkpoint and real
MovieLens data (see experiments/IMPLEMENTATION_NOTES.md section 3).

`rec_model.eval()` is forced unconditionally here (this script never touches a
pytorch_lightning Trainer, so there is no risk of a hidden `.train()` call flipping SASRec's
dropout back on) — the resulting representations are the fixed, reproducible reference used for
KMeans fitting and cluster-hard routing (experiments/IMPLEMENTATION_NOTES.md section 4).

Output: <output_dir>/representations_<split>.npz with keys:
    sample_id: int64 array [N]   (row index into the split's DataFrame, see CODEBASE_ANALYSIS.md section C)
    sasrec:    float32 array [N, D]
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from data.movielens_data import MovielensData
from recommender.A_SASRec_final_bce_llm import SASRec  # noqa: F401 (needed for torch.load unpickling)

SPLIT_TO_STAGE = {"train": "train", "validation": "val", "test": "test"}


def load_rec_model(rec_model_path, device):
    rec_model = torch.load(rec_model_path, map_location=device, weights_only=False)
    rec_model.eval()
    for p in rec_model.parameters():
        p.requires_grad = False
    rec_model.device = device
    return rec_model


def extract_split(rec_model, data_dir, stage, cans_num, batch_size, device):
    dataset = MovielensData(data_dir=data_dir, stage=stage, cans_num=cans_num)
    n = len(dataset)
    sample_ids = []
    reps = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = [dataset[i] for i in range(start, end)]
            seq = torch.stack([torch.tensor(s["seq"]) for s in batch], dim=0).to(device)
            len_seq = torch.stack([torch.tensor(s["len_seq"]) for s in batch], dim=0).to(device)
            state_hidden = rec_model.cacul_h(seq, len_seq)  # [batch, 1, D]
            rep = state_hidden.squeeze(1).float().cpu().numpy()
            reps.append(rep)
            sample_ids.extend(s["sample_idx"] for s in batch)
    reps = np.concatenate(reps, axis=0) if reps else np.zeros((0, 0), dtype=np.float32)
    return np.array(sample_ids, dtype=np.int64), reps.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/ref/movielens")
    parser.add_argument("--rec_model_path", default="rec_model/movielens.pt")
    parser.add_argument("--output_dir", default="results/baseline")
    parser.add_argument("--cans_num", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    rec_model = load_rec_model(args.rec_model_path, device)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        stage = SPLIT_TO_STAGE[split]
        sample_ids, reps = extract_split(rec_model, args.data_dir, stage, args.cans_num, args.batch_size, device)
        out_path = os.path.join(args.output_dir, f"representations_{split}.npz")
        np.savez(out_path, sample_id=sample_ids, sasrec=reps)
        print(f"[extract_representations] {split}: {reps.shape[0]} samples, dim={reps.shape[1]} -> {out_path}")


if __name__ == "__main__":
    main()
