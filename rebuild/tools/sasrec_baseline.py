"""Stage S0 of the Steam ladder (DESIGN.md §24.3, D45): the frozen SASRec alone, without the LLM.

    python -I rebuild/tools/sasrec_baseline.py --config rebuild/configs/datasets/steam_paper_spec.yaml

For every validation and test sample of the profile's dataset, the 20 fixed evaluation candidates (D10, the lists
every later stage is evaluated on) are ranked by the released SASRec's own output layer and the top one is compared
with the target:

    score(item) = s_fc.weight[item] . z + s_fc.bias[item]

``z`` comes from the converted frozen encoder (M3 ``SASRecEncoder.encode_all``, the SR-EMB of every later stage).
``s_fc`` is the output layer the released checkpoint was trained with ([코드] U recommender/A_SASRec_final_bce_llm.py:
``supervised_output = self.s_fc(state_hidden)``); the conversion drops it (§7.6.2), so it is read here from the
released pickle. The pickle must have the sha256 pinned by the profile (``sasrec.source_sha256``) and is read with an
allow-list unpickler: tensors and stub modules only, no class of the pickle is imported and no code from it runs.

Writes ``runs/rebuild/<dataset>_sasrec_baseline_<UTC>/`` with ``meta.json`` and ``baseline.json`` and prints one line
per split. CPU only. No paper number is written; the reference values are in rebuild/STEAM_LADDER.md.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Optional, Sequence

REBUILD_DIR = Path(__file__).resolve().parents[1]
if str(REBUILD_DIR) not in sys.path:
    sys.path.insert(0, str(REBUILD_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from ilora import run  # noqa: E402  (needs the sys.path entry above)
from ilora.candidates import candidates_for  # noqa: E402
from ilora.config import ConfigError, config_hash, load_config  # noqa: E402
from ilora.data import HISTORY_LEN, load_dataset  # noqa: E402
from ilora.datasets import spec_of  # noqa: E402
from ilora.sasrec import SASRecEncoder  # noqa: E402

COMMAND = "sasrec-baseline"
RESULT_FILE = "baseline.json"
WILSON_Z = 1.959963984540054
OUTPUT_LAYER = "s_fc"                         # [코드] U recommender/A_SASRec_final_bce_llm.py (SASRec.forward)

# The 18 globals the released SASRec pickles name (tools/convert_sasrec.py EXPECTED_GLOBALS, §7.6.1). The three
# project classes become plain nn.Module stubs: only their stored tensors are read.
_STUB_CLASSES = {("__main__", "SASRec"), ("SASRecModules_ori", "MultiHeadAttention"),
                 ("SASRecModules_ori", "PositionwiseFeedForward")}
_ALLOWED = {("torch.nn.modules.sparse", "Embedding"), ("torch.nn.modules.normalization", "LayerNorm"),
            ("torch.nn.modules.linear", "Linear"), ("torch.nn.modules.conv", "Conv1d"),
            ("torch.nn.modules.dropout", "Dropout"), ("torch.nn.modules.activation", "Softmax"),
            ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
            ("torch", "FloatStorage"), ("torch", "device"), ("collections", "OrderedDict"),
            ("__builtin__", "set"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("_codecs", "encode")}


class BaselineError(RuntimeError):
    """The SASRec baseline cannot be computed as specified (§24.3)."""


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _STUB_CLASSES:
            return nn.Module
        if (module, name) == ("__builtin__", "set"):
            return set
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global {module}.{name} is not in the allow-list")


def load_output_layer(pt_path: Path, expected_sha256: str, num_items: int) -> tuple[np.ndarray, np.ndarray, str]:
    """``s_fc`` weight ``[num_items, d]`` and bias ``[num_items]`` (float64) of the released pickle, and its sha256."""
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        raise BaselineError(f"{pt_path} not found")
    sha256 = hashlib.sha256(pt_path.read_bytes()).hexdigest()
    if sha256 != expected_sha256:
        raise BaselineError(f"{pt_path}: sha256 {sha256} differs from the profile's sasrec.source_sha256 "
                            f"{expected_sha256}")
    pickle_module = types.ModuleType("sasrec_baseline_pickle")
    pickle_module.Unpickler = _StubUnpickler                                   # type: ignore[attr-defined]
    pickle_module.load = lambda file, **kwargs: _StubUnpickler(file, **kwargs).load()   # type: ignore[attr-defined]
    model = torch.load(pt_path, map_location="cpu", pickle_module=pickle_module, weights_only=False)
    layer = getattr(model, "_modules", {}).get(OUTPUT_LAYER)
    if layer is None or not hasattr(layer, "weight") or not hasattr(layer, "bias"):
        raise BaselineError(f"{pt_path}: no {OUTPUT_LAYER} layer with weight and bias")
    weight = layer.weight.detach().to(torch.float64).numpy()
    bias = layer.bias.detach().to(torch.float64).numpy()
    if weight.ndim != 2 or weight.shape[0] != num_items or bias.shape != (num_items,):
        raise BaselineError(f"{pt_path}: {OUTPUT_LAYER} shapes {weight.shape} / {bias.shape} do not fit "
                            f"{num_items} items")
    if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
        raise BaselineError(f"{pt_path}: {OUTPUT_LAYER} has non-finite values")
    return weight, bias, sha256


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + WILSON_Z * WILSON_Z / n
    centre = (p + WILSON_Z * WILSON_Z / (2 * n)) / d
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + WILSON_Z * WILSON_Z / (4 * n * n)) / d
    return centre - half, centre + half


def sasrec_baseline(cfg: Any, repo_root: Path = run.REPO_ROOT) -> dict:
    """HitRatio@1 of the frozen SASRec on the fixed candidates of the validation and test split (§24.3).

    A sample is a hit when its target has the strictly highest score among its 20 candidates. ``drop_last`` is the
    same count on the first ``n - n % eval.batch_size`` samples (the evaluation loader of the released code drops
    the last partial batch, D17 and D41).
    """
    repo_root = Path(repo_root)
    spec = spec_of(cfg)
    data = load_dataset(cfg, repo_root)
    weight, bias, sha256 = load_output_layer(run.resolve_path(cfg.sasrec.source_path, repo_root),
                                             cfg.sasrec.source_sha256, data.catalog.num_items)
    encoder = SASRecEncoder.from_config(cfg, data.catalog, HISTORY_LEN)
    if weight.shape[1] != encoder.arch.hidden_size:
        raise BaselineError(f"{OUTPUT_LAYER} width {weight.shape[1]} != encoder hidden size {encoder.arch.hidden_size}")
    batch = cfg.eval.batch_size
    splits = {}
    for split in ("val", "test"):
        samples = data.samples(split)
        z = encoder.encode_all([s.history for s in samples]).to(torch.float64).numpy()
        hits, ranks, margins = [], [], []
        for sample, row in zip(samples, z):
            cands = np.asarray(candidates_for(sample, main_seed=cfg.run.seed, epoch=None, cfg=cfg.candidates))
            scores = weight[cands] @ row + bias[cands]
            at = int(np.flatnonzero(cands == sample.target)[0])
            others = np.delete(scores, at)
            ranks.append(int((others > scores[at]).sum()) + 1)
            hits.append(bool(scores[at] > others.max()))
            margins.append(float(abs(scores[at] - others.max())))
        n, k = len(hits), int(sum(hits))
        low, high = wilson_interval(k, n)
        m = n - n % batch
        splits[split] = {"n": n, "correct": k, "hr_at_1": k / n, "wilson95": [low, high],
                         "drop_last": {"batch_size": batch, "n": m, "correct": int(sum(hits[:m])),
                                       "hr_at_1": sum(hits[:m]) / m},
                         "mean_rank_of_target": float(np.mean(ranks)),
                         "min_margin_target_vs_best_other": float(min(margins))}
    return {"dataset": spec.name, "seed": cfg.run.seed, "num_candidates": cfg.candidates.num_candidates,
            "chance_level": 1 / cfg.candidates.num_candidates, "scoring": f"{OUTPUT_LAYER}(z) on the candidates",
            "source_pickle": {"path": cfg.sasrec.source_path, "sha256": sha256},
            "encoder_state_digest": encoder.state_digest, "splits": splits}


def run_baseline(config_path: Path, sets: Sequence[str] = (), *, root: Path = run.RUNS_ROOT,
                 repo_root: Path = run.REPO_ROOT, argv: Optional[Sequence[str]] = None) -> Path:
    """Write ``<root>/<dataset>_sasrec_baseline_<UTC>/`` (``meta.json``, ``baseline.json``) and return it."""
    repo_root = Path(repo_root)
    cfg, _ = load_config(Path(config_path), list(sets))
    spec = spec_of(cfg)
    profile = f"{spec.name}_sasrec_baseline"
    run_dir = run.create_run_dir(profile, root=Path(root))
    run.start_meta(run_dir, profile=profile, argv=list(sys.argv if argv is None else argv),
                   config_hash=config_hash(cfg), overrides={}, seed=cfg.run.seed,
                   extra={"command": COMMAND, "profile_path": str(Path(config_path).resolve()),
                          "inputs": run.input_hashes(repo_root, spec.name)})
    with run.tracked_run(run_dir):
        result = sasrec_baseline(cfg, repo_root)
        run.write_json_atomic(run_dir / RESULT_FILE, result)
    return run_dir


def format_lines(result: dict) -> list[str]:
    lines = [f"dataset {result['dataset']}, seed {result['seed']}, {result['scoring']}, "
             f"chance {result['chance_level']:.4f}"]
    for split, r in result["splits"].items():
        d = r["drop_last"]
        lines.append(f"{split}: correct {r['correct']}/{r['n']} = {r['hr_at_1']:.4f} "
                     f"(95% {r['wilson95'][0]:.4f}-{r['wilson95'][1]:.4f}); first {d['n']} "
                     f"(drop_last, batch {d['batch_size']}): {d['correct']}/{d['n']} = {d['hr_at_1']:.4f}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sasrec_baseline.py",
                                     description="HitRatio@1 of the frozen SASRec alone on the fixed evaluation "
                                                 "candidates (stage S0 of the Steam ladder, DESIGN 24.3).")
    parser.add_argument("--config", type=Path, required=True, metavar="PROFILE_YAML",
                        help="profile file, e.g. rebuild/configs/datasets/steam_paper_spec.yaml")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override one key with a YAML value; repeatable")
    parser.add_argument("--root", type=Path, default=None, metavar="DIR",
                        help="folder for the run folder (default runs/rebuild)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    full_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    root = run.resolve_path(args.root) if args.root else run.RUNS_ROOT
    try:
        out = run_baseline(args.config, args.sets, root=root, argv=full_argv)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except BaselineError as exc:
        print(f"baseline error: {exc}", file=sys.stderr)
        return 1
    import json

    result = json.loads((out / RESULT_FILE).read_text(encoding="utf-8"))
    for line in format_lines(result):
        print(line)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
