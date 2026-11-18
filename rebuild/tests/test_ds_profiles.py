"""데이터셋 프로파일 (DESIGN.md §23.6, D27, D43): 18개 파일, 부모와 다른 키, 교차 검사, step·hybrid 수 (§23.2)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from ilora import run
from ilora.config import ConfigError, config_hash, config_to_dict, dump_config, load_config
from ilora.datasets import SPECS, spec_of
from ilora.prompts import make_batch_plan
from ilora.sections import Config
from ilora.train import TrainingError, step_counts

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
PROFILE_DIR = CONFIG_DIR / "datasets"
ARTIFACT = "runs/rebuild/cluster_run_for_tests/cluster"      # any path (cluster.artifact_dir comes from --set only)

# §23.2 hash table: original pickle sha256 and converted state digest of the SASRec checkpoints (D36).
PICKLE_SHA256 = {"movielens": "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9",
                 "steam": "752c0cd11a8646adca10025b5746dfcb8968967a6104546f6f869a3ebf26609b",
                 "lastfm": "6f94253543e0ca3b0bbd42b4f6b94f2c97c9fdfdd36c3b5ad3f6b576a806dd86"}
STATE_DIGEST = {"steam": "d5d13fc0bc0f3c8ebc2f64e4247e9da3b486e6984c931bb8454ebbb17c8b3932",
                "lastfm": "6b0d161351901a272fe07ef4c2bf3ccc99e06a487bcaee1c9b88392f59435d85"}
TEMPLATE = {"steam": "prompt/game.txt", "lastfm": "prompt/artist.txt"}
# D29 variants (U train_steam.sh:18-20, U train_lastfm.sh:18-20): lr_peak, lr_warmup_start, lr_floor
RELEASED_LR = {"steam": (5.0e-4, 5.0e-6, 5.0e-6), "lastfm": (7.0e-4, 7.0e-6, 7.0e-6)}
# §23.6 data.train_rows (multi/verify/check_numbers2.py)
OVERFIT_ROWS = {"steam": (39, 94, 129, 133, 210, 230, 316, 329, 344, 601, 748, 749, 762, 795, 812, 939),
                "lastfm": (61, 94, 168, 181, 199, 433, 525, 546, 564, 591, 617, 628, 643, 657, 680, 732)}
# §23.6 keys that are written in smoke_cpu.yaml, smoke_gpu_100.yaml and overfit_gpu.yaml.
SMOKE_CPU_KEYS = {"data.subset": "smoke", "data.smoke_users": (2, 10, 10), "batching.num_workers": 0,
                  "model.backbone": "tiny_random", "model.torch_dtype": "float32", "model.device": "cpu",
                  "train.epochs": 2, "train.micro_batch_size": 4, "train.grad_accum": 4, "train.warmup_divisor": 4,
                  "train.precision": "fp32", "eval.batch_invariance_check": True, "records.flush_every_steps": 5,
                  "cluster.live_check_n": 64}
SMOKE_GPU_KEYS = {"data.subset": "smoke", "train.epochs": 2, "eval.batch_invariance_check": True}
OVERFIT_KEYS = {"data.subset": "smoke", "data.smoke_users": (16, 10, 10), "prompt.train_curriculum": "all_hybrid",
                "train.epochs": 25, "train.micro_batch_size": 4, "train.grad_accum": 1}


def _table(d: str) -> dict[str, tuple[str, dict]]:
    """§23.6 for dataset ``d``: profile stem -> (``extends``, the keys and values that its file sets)."""
    paper = {"sasrec.source_path": f"rec_model/{d}.pt", "sasrec.source_sha256": PICKLE_SHA256[d],
             "sasrec.state_path": f"runs/rebuild/assets/sasrec_{d}/sasrec_{d}_state.pt",
             "sasrec.state_digest": STATE_DIGEST[d],
             "sasrec.reference_path": f"runs/rebuild/assets/sasrec_{d}/sasrec_{d}_reference.pt",
             "prompt.template_path": TEMPLATE[d]}
    if d == "steam":                                         # D31, D32
        paper.update({"train.micro_batch_size": 4, "train.grad_accum": 32, "eval.batch_size": 4})
    else:                                                    # D29, D30
        paper.update({"train.lr_peak": 2.0e-4, "train.lr_warmup_start": 2.0e-6, "train.lr_floor": 2.0e-6})
    base = f"{d}_paper_spec.yaml"
    peak, start, floor = RELEASED_LR[d]
    smoke_cpu = dict(SMOKE_CPU_KEYS)
    smoke_gpu = dict(SMOKE_GPU_KEYS)
    if d == "lastfm":                                        # §23.6: 41 train samples
        smoke_cpu.update({"train.warmup_divisor": 3, "cluster.live_check_n": 32})
    else:                                                    # §23.6: 1,385 train samples, T = 22 and 11
        smoke_gpu.update({"train.warmup_divisor": 10})
    return {
        f"{d}_paper_spec": ("../paper_spec.yaml", paper),
        f"{d}_llara_baseline": (base, {"lora.num_experts": 1, "router.routing_mode": "none"}),
        f"{d}_variant_text_only_eval": (base, {"prompt.eval_prompt": "text_only"}),
        f"{d}_cluster_hard": (base, {"router.routing_mode": "cluster_hard"}),
        f"{d}_variant_released_lr": (base, {"train.lr_peak": peak, "train.lr_warmup_start": start,
                                            "train.lr_floor": floor}),
        f"{d}_variant_bf16": (base, {"model.torch_dtype": "bfloat16", "train.precision": "bf16_amp"}),
        f"{d}_smoke_cpu": (base, smoke_cpu),
        f"{d}_smoke_gpu_100": (base, smoke_gpu),
        f"{d}_overfit_gpu": (base, {**OVERFIT_KEYS, "data.train_rows": OVERFIT_ROWS[d]}),
    }


TABLE = {**_table("steam"), **_table("lastfm")}
NAMES = tuple(TABLE)
# Keys that the files write like smoke_cpu.yaml and overfit_gpu.yaml although their value equals the Steam parent's (4).
SAME_AS_PARENT = {"steam_smoke_cpu": {"train.micro_batch_size"}, "steam_overfit_gpu": {"train.micro_batch_size"}}


def _sets(name: str) -> list[str]:
    return [f"cluster.artifact_dir={ARTIFACT}"] if name.endswith("_cluster_hard") else []


def _load(name: str, *sets: str):
    cfg, _ = load_config(PROFILE_DIR / f"{name}.yaml", [*_sets(name), *sets])
    return cfg


def _flat(cfg) -> dict:
    return {f"{s.name}.{k.name}": getattr(getattr(cfg, s.name), k.name)
            for s in dataclasses.fields(cfg) for k in dataclasses.fields(getattr(cfg, s.name))}


def test_files_format_and_unique_stems():
    """§23.6, D43: exactly the 18 files, ASCII only, first line a comment, every stem unique among all profiles."""
    stems = sorted(p.stem for p in PROFILE_DIR.iterdir())
    assert stems == sorted(TABLE) and len(stems) == 18
    assert all(p.suffix == ".yaml" for p in PROFILE_DIR.iterdir())
    for path in PROFILE_DIR.iterdir():
        raw = path.read_bytes()
        assert all(b < 128 for b in raw), path.name
        assert raw.splitlines()[0].startswith(b"#"), path.name
    assert len({*stems, *(p.stem for p in CONFIG_DIR.glob("*.yaml"))}) == len(stems) + 17


@pytest.mark.parametrize("name", NAMES)
def test_profile_loads_and_round_trips(name, tmp_path):
    """Every profile loads (the two cluster_hard ones only with cluster.artifact_dir, §23.6) and saving then
    reloading gives the same dict and hash; every value agrees with the dataset that the file name says (D27)."""
    if name.endswith("_cluster_hard"):
        with pytest.raises(ConfigError, match="artifact_dir"):
            load_config(PROFILE_DIR / f"{name}.yaml")
    cfg = _load(name)
    assert isinstance(cfg, Config)
    cfg.validate()
    d = name.split("_", 1)[0]
    spec = spec_of(cfg)
    assert spec is SPECS[d]
    assert cfg.prompt.template_path == spec.template_file == TEMPLATE[d]
    assert cfg.sasrec.source_path == spec.sasrec_source == f"rec_model/{d}.pt"
    assert cfg.sasrec.source_sha256 == run.expected_inputs(d)[spec.sasrec_source][0]
    assert cfg.data.title_mode in spec.title_modes
    path = tmp_path / "config.yaml"
    dump_config(cfg, path)
    again, applied = load_config(path)
    assert applied == {} and config_to_dict(again) == config_to_dict(cfg)
    assert config_hash(again) == config_hash(cfg) and again == cfg


@pytest.mark.parametrize("name", NAMES)
def test_resolved_differences_equal_the_table(name):
    """§23.6: the file extends its parent and sets exactly the keys of the table; the keys of the table take the
    table values and no other key differs from the parent after resolution."""
    parent, expected = TABLE[name]
    raw = yaml.safe_load((PROFILE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    assert raw.get("extends") == parent, name
    assert {f"{s}.{k}" for s, keys in raw.items() if s != "extends" for k in keys} == set(expected), name
    flat = _flat(_load(name))
    base = _flat(load_config(PROFILE_DIR / parent)[0])       # the parent without the --set of the cluster_hard files
    values = dict(expected, **({"cluster.artifact_dir": ARTIFACT} if name.endswith("_cluster_hard") else {}))
    for key, value in values.items():
        assert flat[key] == value and type(flat[key]) is type(value), (name, key)
    differing = {key for key in base if flat[key] != base[key]}
    assert differing <= set(values), (name, differing - set(values))
    assert set(values) - differing == SAME_AS_PARENT.get(name, set()), name


# (profile under configs/, --set texts, the name that the ConfigError carries) (§23.4 Config.validate cross checks)
REJECTED = [
    ("datasets/steam_paper_spec", ("prompt.template_path=prompt/movie.txt",), "prompt.template_path"),
    ("datasets/steam_paper_spec", ("prompt.template_path=prompt/artist.txt",), "prompt.template_path"),
    ("datasets/lastfm_paper_spec", ("prompt.template_path=prompt/game.txt",), "prompt.template_path"),
    ("datasets/lastfm_paper_spec", ("prompt.template_path=prompt/movie.txt",), "prompt.template_path"),
    ("paper_spec", ("prompt.template_path=prompt/artist.txt",), "prompt.template_path"),
    ("paper_spec", ("prompt.template_path=prompt/read.txt",), "prompt.template_path"),        # no dataset has it
    ("datasets/steam_paper_spec", (f'sasrec.source_sha256="{PICKLE_SHA256["movielens"]}"',), "prompt.template_path"),
    ("datasets/lastfm_paper_spec", (f'sasrec.source_sha256="{PICKLE_SHA256["steam"]}"',), "prompt.template_path"),
    ("paper_spec", (f'sasrec.source_sha256="{PICKLE_SHA256["lastfm"]}"',), "prompt.template_path"),
    ("paper_spec", ('sasrec.source_sha256="' + "a" * 64 + '"',), "sasrec.source_sha256"),     # no dataset's pickle
    ("datasets/steam_paper_spec", ("data.title_mode=fixed",), "data.title_mode"),             # D38
    ("datasets/lastfm_paper_spec", ("data.title_mode=fixed",), "data.title_mode"),            # D38
]


@pytest.mark.parametrize("profile,sets,name", REJECTED, ids=[f"{p}:{s[0][:40]}" for p, s, _ in REJECTED])
def test_cross_checks_reject_a_mismatch(profile, sets, name):
    """D27: template and SASRec pickle must belong to one dataset; Steam and LastFM allow only title_mode code."""
    with pytest.raises(ConfigError, match=name):
        load_config(CONFIG_DIR / f"{profile}.yaml", list(sets))


def test_cross_checks_on_a_built_config():
    """``Config.validate`` itself (§23.4) on configs that never went through ``load_config``; MovieLens keeps
    ``data.title_mode`` ``fixed`` (D20) and a matching pair passes."""
    cfg = _load("steam_paper_spec")
    cfg.validate()
    for path in ("prompt/movie.txt", "prompt/read.txt"):      # another dataset's template, then no dataset's
        broken = dataclasses.replace(cfg, prompt=dataclasses.replace(cfg.prompt, template_path=path))
        with pytest.raises(ConfigError, match="prompt.template_path"):
            broken.validate()
    fixed = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, title_mode="fixed"))
    with pytest.raises(ConfigError, match="data.title_mode"):
        fixed.validate()
    movielens_fixed, _ = load_config(CONFIG_DIR / "variant_fixed_titles.yaml")
    movielens_fixed.validate()
    assert movielens_fixed.data.title_mode == "fixed" and spec_of(movielens_fixed).name == "movielens"


# §23.2 (M10 14.6.2, M4 8.6.5, seed 1234): config file, N, NB, dropped samples, tail (D24), steps per epoch, T, w,
# T_mb, hybrid micro-batches per epoch, their sum.
LENGTHS = {
    "paper_spec": (CONFIG_DIR / "paper_spec.yaml", 66126, 8265, 6, 9, 517, 2585, 129, 41325,
                   (846, 2550, 4076, 5766, 7436), 20674),
    "steam_paper_spec": (PROFILE_DIR / "steam_paper_spec.yaml", 151048, 37762, 0, 2, 1181, 5905, 295, 188810,
                         (3781, 11484, 18910, 26406, 34093), 94674),
    "lastfm_paper_spec": (PROFILE_DIR / "lastfm_paper_spec.yaml", 47872, 5984, 0, 0, 374, 1870, 93, 29920,
                          (586, 1820, 2979, 4195, 5373), 14953),
}


@pytest.mark.parametrize("name", list(LENGTHS))
def test_step_counts_and_hybrid_counts_equal_the_table(name):
    """§23.2 row "epoch당 micro-batch" to "hybrid micro-batch": ``step_counts`` and ``make_batch_plan`` with the
    numbers of the design and the values of the profile (D24 tail, D3 curriculum over all epochs)."""
    path, n, nb, dropped, tail, per_epoch, total, warmup, micro, hybrid, hybrid_sum = LENGTHS[name]
    cfg, _ = load_config(path)
    sc = step_counts(n, cfg.train)
    assert (sc.n_samples, sc.micro_per_epoch, n - nb * cfg.train.micro_batch_size, sc.tail, sc.steps_per_epoch,
            sc.total_steps, sc.warmup_steps, sc.total_micro) == (n, nb, dropped, tail, per_epoch, total, warmup, micro)
    assert sum(sc.window_sizes()) == nb and len(sc.window_sizes()) == per_epoch
    plans = [make_batch_plan(seed=cfg.run.seed, epoch=e, epochs=cfg.train.epochs, num_samples=n,
                             micro_batch_size=cfg.train.micro_batch_size, curriculum=cfg.prompt.train_curriculum)
             for e in range(cfg.train.epochs)]
    assert all(plan.batches_per_epoch == nb and plan.total == micro for plan in plans)
    assert tuple(plan.hybrid_count() for plan in plans) == hybrid and sum(hybrid) == hybrid_sum


# §23.2, §23.6: train samples (the smoke selection of seed 1234, multi/verify/check_numbers2.py) and (T, w) of every
# step: the profile's train.epochs and, for smoke_cpu and smoke_gpu_100, train.epochs=1 (E4 and E1, smoke.py _E4_SETS
# and _E1_SETS). multi/critic/c7_steps.py computes the same step_counts.
SMOKE_RUNS = {
    "steam_smoke_cpu": (80, {2: (10, 2), 1: (5, 1)}),
    "lastfm_smoke_cpu": (41, {2: (6, 2), 1: (3, 1)}),
    "steam_smoke_gpu_100": (1385, {2: (22, 2), 1: (11, 1)}),
    "lastfm_smoke_gpu_100": (3626, {2: (58, 2), 1: (29, 1)}),
    "steam_overfit_gpu": (16, {25: (100, 5)}),
    "lastfm_overfit_gpu": (16, {25: (100, 5)}),
}


@pytest.mark.parametrize("name", list(SMOKE_RUNS))
def test_smoke_profiles_fit_their_data(name):
    """§23.6: every step of the smoke plan has a warmup of at least one step at the real train sample count, and
    ``verify_live_encoder`` of the cluster and E4 steps finds ``cluster.live_check_n`` train samples."""
    samples, runs = SMOKE_RUNS[name]
    cfg = _load(name)
    assert cfg.train.epochs == max(runs), name
    for epochs, (total, warmup) in runs.items():
        sc = step_counts(samples, dataclasses.replace(cfg.train, epochs=epochs))
        assert (sc.epochs, sc.total_steps, sc.warmup_steps) == (epochs, total, warmup), (name, epochs)
    if "_smoke_" in name:
        assert cfg.cluster.live_check_n <= samples, name


def test_values_of_the_movielens_profiles_do_not_fit():
    """§23.6 "smoke 값의 이유": the checks above catch a plain copy of the MovieLens smoke values (w = 0 at
    train.epochs=1, T = 3 and 11, multi/critic/c7_steps.py), which is why the two profiles differ."""
    plain_cpu, _ = load_config(CONFIG_DIR / "smoke_cpu.yaml")
    plain_gpu, _ = load_config(CONFIG_DIR / "smoke_gpu_100.yaml")
    lastfm_cpu, steam_gpu = _load("lastfm_smoke_cpu"), _load("steam_smoke_gpu_100")
    with pytest.raises(TrainingError, match="warmup_divisor"):
        step_counts(41, dataclasses.replace(lastfm_cpu.train, warmup_divisor=plain_cpu.train.warmup_divisor, epochs=1))
    assert plain_cpu.cluster.live_check_n > 41                # and the cluster check would refuse 64 of 41
    with pytest.raises(TrainingError, match="warmup_divisor"):
        step_counts(1385, dataclasses.replace(steam_gpu.train, warmup_divisor=plain_gpu.train.warmup_divisor, epochs=1))
