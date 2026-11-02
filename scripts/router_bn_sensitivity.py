"""Router gates with and without the BatchNorm running statistics that checkpoints do not contain (CPU).

MInterface.on_save_checkpoint drops every tensor that is not a trainable parameter, so the 19 BatchNorm
layers of the shared router run with their initial statistics (mean 0, variance 1) in every
`--mode test` evaluation and every gate export made from a saved checkpoint, while in the training
process (training batches and the in-process validation) they use batch or running statistics.

This script loads only the router parameters of a checkpoint and the frozen SASRec model (no LLaMA,
nothing trained, the checkpoint is only read) and computes the gates of the validation and test users:
  initial            the statistics a --mode test run has
  recomputed         re-estimated from the training sequences with eval_protocol_variants.recompute_router_bn
                     (train mode, batches of 8, cumulative average; fp32 on CPU)
  train_batch8       the router in train mode on consecutive batches of 8, i.e. batch statistics as in
                     fine-tuning, where a user's gate also depends on the other users of the batch
Gate summaries use analysis/analyze_gates.compute_gate_summary, the same function as the baseline report.

    python scripts/router_bn_sensitivity.py --ckpt "checkpoints/baseline/epoch=01-metric=0.521.ckpt"
    python scripts/router_bn_sensitivity.py --ckpt checkpoints/baseline/last.ckpt --gates_csv results/baseline/gates_test.csv
"""
import argparse
import copy
import json
import os
import sys

from eval_protocol_variants import recompute_router_bn  # also puts the repository root first on sys.path
import eval_prompt_semantics as base

import numpy as np
import pandas as pd
import torch

from analysis.analyze_gates import compute_gate_summary
from data.movielens_data import MovielensData
from model.router.nlpr import build_router
from recommender.A_SASRec_final_bce_llm import SASRec  # noqa: F401 (torch.load unpickles the SASRec object)


def load_router(ckpt):
    state = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']  # tensors and plain containers only
    router = build_router()
    params = {k[len('router.'):]: v for k, v in state.items() if k.startswith('router.')}
    missing, unexpected = router.load_state_dict(params, strict=False)
    buffers = [k for k in missing if k.rsplit('.', 1)[-1] in ('running_mean', 'running_var', 'num_batches_tracked')]
    if unexpected or len(buffers) != len(missing):
        raise SystemExit(f'router keys do not match: missing {missing}, unexpected {unexpected}')
    return router.eval(), {'router_parameters_loaded': len(params), 'batchnorm_buffers_missing': len(buffers)}


def load_rec_model(path):
    rec_model = torch.load(path, map_location='cpu')
    for p in rec_model.parameters():
        p.requires_grad = False
    rec_model.device = torch.device('cpu')
    return rec_model.eval()


def sequences(stage):
    data = MovielensData(data_dir=os.path.join(base.REPO, 'data/ref/movielens'), stage=stage, cans_num=20).session_data
    return torch.tensor(list(data['seq'])), torch.tensor(list(data['len_seq']))


def gates(router, users, train_batch=None):
    with torch.no_grad():
        if train_batch is None:
            return router.eval()(users).squeeze(1).double().numpy()
        r = copy.deepcopy(router).train()
        n = len(users) - len(users) % train_batch
        return np.concatenate([r(users[i:i + train_batch]).squeeze(1).double().numpy() for i in range(0, n, train_batch)])


def summarize(g, reference=None):
    frame = pd.DataFrame(g, columns=[f'gate_{k}' for k in range(g.shape[1])])
    frame['argmax_expert'] = g.argmax(axis=1)
    summary = compute_gate_summary(frame, list(frame.columns[:-1]))[0]
    if reference is not None:
        ref = reference[:len(g)]
        summary['vs_initial'] = {'mean_abs_gate_difference': float(np.abs(g - ref).mean()),
                                 'max_abs_gate_difference': float(np.abs(g - ref).max()),
                                 'argmax_agreement': float((g.argmax(1) == ref.argmax(1)).mean())}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--rec_model_path', default='rec_model/movielens.pt')
    parser.add_argument('--gates_csv', help='exported gates_test.csv of the same checkpoint, compared with "initial"')
    parser.add_argument('--out_dir', help='default runs/router_bn_sensitivity/<checkpoint>')
    args = parser.parse_args()

    ckpt = base.repo_path(args.ckpt)
    out_dir = base.repo_path(args.out_dir or os.path.join(
        'runs', 'router_bn_sensitivity', os.path.splitext(os.path.basename(ckpt))[0]))
    if base.protected_by(out_dir):
        raise SystemExit(f'{out_dir} is inside {base.protected_by(out_dir)}/, which this script never writes to')
    if os.path.exists(out_dir):
        raise SystemExit(f'{out_dir} exists; result folders are never overwritten')
    digest = base.sha256(ckpt)
    router, loaded = load_router(ckpt)
    rec_model = load_rec_model(base.repo_path(args.rec_model_path))
    users = {}
    with torch.no_grad():
        for split, stage in (('validation', 'val'), ('test', 'test')):
            seq, lens = sequences(stage)
            users[split] = rec_model.cacul_h(seq, lens)  # SASRec in eval mode, as in validation/test

    report = {'checkpoint': {'path': os.path.relpath(ckpt, base.REPO), 'sha256': digest, **loaded},
              'training_performed': False, 'precision': 'fp32 on CPU (the run evaluated under fp16 autocast)',
              'splits': {}}
    initial = {s: gates(router, u) for s, u in users.items()}
    batch8 = {s: gates(router, u, train_batch=8) for s, u in users.items()}
    train_seq, train_len = sequences('train')
    report['recompute'] = recompute_router_bn(router, rec_model, train_seq, train_len, batch_size=8, seed=0)
    recomputed = {s: gates(router, u) for s, u in users.items()}
    for split in users:
        report['splits'][split] = {'initial': summarize(initial[split]),
                                   'recomputed': summarize(recomputed[split], initial[split]),
                                   'train_batch8': summarize(batch8[split], initial[split])}
    if args.gates_csv:
        exported = pd.read_csv(base.repo_path(args.gates_csv))
        exported = exported.sort_values('sample_id')[[f'gate_{k}' for k in range(4)]].to_numpy()
        report['exported_gates_vs_initial'] = {
            'file': args.gates_csv, 'rows': len(exported),
            'max_abs_difference': float(np.abs(exported - initial['test'][:len(exported)]).max())}
    if base.sha256(ckpt) != digest:
        raise SystemExit('checkpoint changed while reading it')
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'router_bn_sensitivity.json'), 'w') as f:
        json.dump(report, f, indent=2)

    for split, variants in report['splits'].items():
        for name, s in variants.items():
            extra = s.get('vs_initial')
            print(f"{split:10s} {name:12s} entropy {s['entropy']['mean']:.4f}  margin {s['top1_minus_top2_margin']['mean']:.4f}  "
                  f"JS {s['js_divergence_from_global_mean']['mean']:.6f}  argmax {s['argmax_expert_count']}"
                  + (f"  |dgate| {extra['mean_abs_gate_difference']:.4f} argmax-agree {extra['argmax_agreement']:.2f}" if extra else ''))
    if 'exported_gates_vs_initial' in report:
        print(f"exported gates vs initial (CPU fp32): max |diff| {report['exported_gates_vs_initial']['max_abs_difference']:.2e}")
    print(f"wrote {os.path.join(out_dir, 'router_bn_sensitivity.json')}")


if __name__ == '__main__':
    main()
