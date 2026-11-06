"""M14 프로파일 파일 (DESIGN.md §18.6.2-§18.6.5, §18.8 #1-#3)."""
from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Optional

import pytest
import yaml

from ilora.config import ConfigError, iter_items, load_config
from ilora.sections import Config

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
ARTIFACT = "runs/rebuild/cluster_run_for_tests/cluster"      # any path (§18.8 #1)
NEEDS_ARTIFACT_DIR = ("cluster_hard", "variant_user_level_cluster")
OPT_INT, OPT_STR, INTS, STRS = Optional[int], Optional[str], tuple[int, ...], tuple[str, ...]

# DESIGN.md §18.6.2: key -> (type, paper_spec value, label with backticks removed). 87 keys.
KEY_TABLE = {
    "run.seed": (int, 1234, "[SPEC §0.6] (D2, 계보 [코드] U main.py:84)"),
    "data.title_mode": (str, "code", "[코드] U data/movielens_data.py:61-66, :74 (D20)"),
    "data.subset": (str, "full", "[결정] D14"),
    "data.smoke_users": (INTS, (80, 10, 10), "[결정] D14"),
    "data.train_rows": (INTS, (), "[결정]"),
    "candidates.num_candidates": (int, 20, "[논문 App B p.16] [LLaRA §5.1.3 p.7] [코드] U main.py:122 (D10)"),
    "sasrec.source_path": (str, "rec_model/movielens.pt", "[코드] U·L 같은 blob 41c057dc"),
    "sasrec.source_sha256": (str, "6928a06778d0e5b64ead0641501f73f13beb2c61ca2d881ee15582630d101df9", "[코드] (계산값)"),
    "sasrec.state_path": (str, "runs/rebuild/assets/sasrec/sasrec_movielens_state.pt", "[결정]"),
    "sasrec.state_digest": (str, "6a78697b92ce8994250842dc780eac723285e5f51741264c09706e0057c51dae",
                            "[결정] (prototype 계산값)"),
    "sasrec.reference_path": (str, "runs/rebuild/assets/sasrec/sasrec_movielens_reference.pt", "[결정]"),
    "prompt.template_path": (str, "prompt/movie.txt", "[코드]"),
    "prompt.template_choice": (str, "random", "[LLaRA §5.1.2 p.7] (D15)"),
    "prompt.fixed_template_index": (OPT_INT, None, "[결정]"),
    "prompt.train_curriculum": (str, "linear", "[LLaRA 간접 §4.3 p.6 식 9-11] (D3)"),
    "prompt.eval_prompt": (str, "hybrid", "[논문 p.4 식 6-7 (암시)] (D4)"),
    "batching.label_mode": (str, "answer_only", "[결정] D18"),
    "batching.num_workers": (int, 8, "[코드] U main.py:83"),
    "batching.max_seq_len": (int, 1024, "[결정]"),
    "batching.tokenizer_sha256": (str, "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347", "[결정]"),
    "router.routing_mode": (str, "dynamic", "[논문 p.6 식 11] (D1), [SPEC §9]"),
    "lora.r": (int, 8, "[코드] D7"),
    "lora.alpha": (int, 32, "[코드] D7"),
    "lora.dropout": (float, 0.1, "[코드] D7"),
    "lora.num_experts": (int, 4, "[논문 p.9 §4.3.2]"),
    "lora.scaling_rule": (str, "expert_rank", "[결정] D8"),
    "lora.target_modules": (STRS, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                            "[코드] D7"),
    "model.backbone": (str, "llama2_7b", "[논문 App B p.16], base는 [코드] U README.md:13"),
    "model.torch_dtype": (str, "float16", "[결정] D9"),
    "model.device": (str, "cuda:0", "[SPEC §0.12]"),
    "model.gen_strategy": (str, "greedy", "[코드] U model/model_interface.py:70, L model/model_interface.py:43"),
    "model.gen_max_new_tokens": (int, 64, "[코드] U …:70, :80"),
    "model.gen_min_new_tokens": (int, 1, "[코드] U …:70, :81"),
    "model.gen_repetition_penalty": (float, 1.0, "[코드] U …:70, :83"),
    "train.epochs": (int, 5, "[v1 App E p.16] [LLaRA §5.1.2 p.7] D11"),
    "train.micro_batch_size": (int, 8, "[결정] 값은 [코드] U train_movielens.sh:5"),
    "train.grad_accum": (int, 16, "[결정] 값은 [코드] U train_movielens.sh:6, 곱 128 [v1 App E p.16] [LLaRA §5.1.2 p.7]"),
    "train.lr_peak": (float, 1.0e-4, "[논문 App E p.17] D5"),
    "train.lr_warmup_start": (float, 1.0e-6, "[v1 App E p.16] [LLaRA §5.1.2 p.7] D6"),
    "train.lr_floor": (float, 1.0e-6, "[코드] U main.py:90 D6"),
    "train.warmup_divisor": (int, 20, "[코드] U model/model_interface.py:248 D6"),
    "train.lr_scale_projector": (float, 1.0, "[논문 App E p.17] D19"),
    "train.lr_scale_gate": (float, 1.0, "[논문 p.5 §3.2 (2)] (암시) D19"),
    "train.lr_scale_lora": (float, 1.0, "[논문 App E p.17] D19"),
    "train.weight_decay_projector": (float, 1.0e-5,
                                     "[v1 App E p.16] 값, 적용 범위 [코드] U model/model_interface.py:228 D19"),
    "train.weight_decay_gate": (float, 1.0e-5, "[v1 App E p.16] 값, 적용 범위 [코드] U model/model_interface.py:230 D19"),
    "train.weight_decay_lora": (float, 0.0, "[코드] U model/model_interface.py:232 D19"),
    "train.adam_beta1": (float, 0.9, "[코드] torch 2.0.0 기본값, U·L 기본값 사용"),
    "train.adam_beta2": (float, 0.999, "[코드] 같음"),
    "train.adam_eps": (float, 1.0e-8, "[코드] 같음"),
    "train.precision": (str, "fp16_amp", "[논문 App E p.17] D9"),
    "train.scaler_init_scale": (float, 65536.0, "[코드] torch 2.0.0 GradScaler 기본값"),
    "train.scaler_growth_factor": (float, 2.0, "[코드] 같음"),
    "train.scaler_backoff_factor": (float, 0.5, "[코드] 같음"),
    "train.scaler_growth_interval": (int, 2000, "[코드] 같음"),
    "train.max_consecutive_skips": (int, 16, "[결정]"),
    "train.log_every_steps": (int, 10, "[결정]"),
    "eval.batch_size": (int, 8, "[결정] (계보 [코드] U train_movielens.sh:5)"),
    "eval.batch_invariance_check": (bool, False, "[결정]"),
    "eval.source_run": (OPT_STR, None, "[결정]"),
    "eval.checkpoints": (STRS, ("best_val", "last"), "[결정] D11"),
    "eval.splits": (STRS, ("val", "test"), "[결정] D11·D17"),
    "records.flush_every_steps": (int, 20, "[결정]"),
    "records.activation_max": (bool, True, "[결정]"),
    "records.gate_rowsum_atol": (float, 1.0e-5, "[결정]"),
    "records.gate_crosscheck_atol": (float, 1.0e-5, "[결정]"),
    "records.source_run": (OPT_STR, None, "[결정]"),
    "records.export_checkpoints": (STRS, ("best_val", "last"), "[결정] D11"),
    "records.export_num_threads": (int, 1, "[결정]"),
    "records.export_batch_size": (int, 1024, "[결정]"),
    "records.analysis_splits": (STRS, ("test", "val", "train"), "[결정]"),
    "records.figure_formats": (STRS, ("png",), "[SPEC §6]"),
    "cluster.unit": (str, "sequence", "[결정] D25"),
    "cluster.n_clusters": (int, 4, "[SPEC §7]"),
    "cluster.random_state": (int, 1234, "[SPEC §7]"),
    "cluster.n_init": (int, 10, "[SPEC §7] 값 [결정]"),
    "cluster.init": (str, "k-means++", "[결정]"),
    "cluster.algorithm": (str, "lloyd", "[결정]"),
    "cluster.max_iter": (int, 300, "[결정]"),
    "cluster.tol": (float, 1.0e-4, "[결정]"),
    "cluster.fit_num_threads": (int, 1, "[결정]"),
    "cluster.expert_mapping": (str, "0:0,1:1,2:2,3:3", "[SPEC §9]"),
    "cluster.artifact_dir": (OPT_STR, None, "[결정]"),
    "cluster.dynamic_export_dir": (OPT_STR, None, "[결정]"),
    "cluster.z_crosscheck_atol": (float, 1.0e-5, "[결정]"),
    "cluster.live_check_n": (int, 256, "[결정]"),
    "cluster.live_check_atol": (float, 1.0e-4, "[결정]"),
}

# Choice sets of the §18.6.2 type column for single-valued keys.
SCALAR_CHOICES = {
    "data.title_mode": ("code", "fixed"),
    "data.subset": ("full", "smoke"),
    "candidates.num_candidates": (20,),
    "prompt.template_path": ("prompt/movie.txt",),
    "prompt.template_choice": ("random", "fixed"),
    "prompt.train_curriculum": ("linear", "all_hybrid", "reverse"),
    "prompt.eval_prompt": ("hybrid", "text_only"),
    "batching.label_mode": ("answer_only", "released_pair"),
    "router.routing_mode": ("dynamic", "cluster_hard", "none"),
    "lora.scaling_rule": ("expert_rank", "total_rank"),
    "model.backbone": ("llama2_7b", "tiny_random"),
    "model.torch_dtype": ("float16", "bfloat16", "float32"),
    "model.device": ("cuda:0", "cpu"),
    "model.gen_strategy": ("greedy",),
    "train.precision": ("fp16_amp", "bf16_amp", "fp32"),
    "cluster.unit": ("sequence", "user"),
    "cluster.init": ("k-means++",),
    "cluster.algorithm": ("lloyd",),
}

OVERFIT_ROWS = (71, 196, 377, 554, 797, 1001, 1129, 1234, 1595, 1651, 1905, 2070, 2311, 2693, 2765, 2874)

# §18.6.4-§18.6.5: profile -> (extends, keys that differ from paper_spec after resolution, with values).
# cluster.artifact_dir of the two cluster_hard profiles comes from --set, never from the file.
OVERLAYS = {
    "paper_spec": (None, {}),
    "llara_baseline": ("paper_spec.yaml", {"lora.num_experts": 1, "router.routing_mode": "none"}),
    "cluster_hard": ("paper_spec.yaml", {"router.routing_mode": "cluster_hard", "cluster.artifact_dir": ARTIFACT}),
    "variant_all_hybrid": ("paper_spec.yaml", {"prompt.train_curriculum": "all_hybrid"}),
    "variant_reverse_curriculum": ("paper_spec.yaml", {"prompt.train_curriculum": "reverse"}),
    "variant_text_only_eval": ("paper_spec.yaml", {"prompt.eval_prompt": "text_only"}),
    "variant_released_lr": ("paper_spec.yaml", {"train.lr_peak": 8.0e-4, "train.lr_warmup_start": 8.0e-6,
                                                "train.lr_floor": 8.0e-6}),
    "variant_released_gate_lr": ("paper_spec.yaml", {"train.lr_scale_gate": 0.3}),
    "variant_released_label": ("paper_spec.yaml", {"batching.label_mode": "released_pair"}),
    "variant_total_rank_scaling": ("paper_spec.yaml", {"lora.scaling_rule": "total_rank"}),
    "variant_fixed_template": ("paper_spec.yaml", {"prompt.template_choice": "fixed",
                                                   "prompt.fixed_template_index": 0}),
    "variant_fixed_titles": ("paper_spec.yaml", {"data.title_mode": "fixed"}),
    "variant_user_level_cluster": ("cluster_hard.yaml", {"router.routing_mode": "cluster_hard",
                                                         "cluster.artifact_dir": ARTIFACT, "cluster.unit": "user"}),
    "variant_bf16": ("paper_spec.yaml", {"model.torch_dtype": "bfloat16", "train.precision": "bf16_amp"}),
    "smoke_cpu": ("paper_spec.yaml", {
        "data.subset": "smoke", "data.smoke_users": (2, 10, 10), "batching.num_workers": 0,
        "model.backbone": "tiny_random", "model.torch_dtype": "float32", "model.device": "cpu",
        "train.epochs": 2, "train.micro_batch_size": 4, "train.grad_accum": 4, "train.warmup_divisor": 4,
        "train.precision": "fp32", "eval.batch_invariance_check": True, "records.flush_every_steps": 5,
        "cluster.live_check_n": 64}),
    "smoke_gpu_100": ("paper_spec.yaml", {"data.subset": "smoke", "train.epochs": 2,
                                          "eval.batch_invariance_check": True}),
    "overfit_gpu": ("paper_spec.yaml", {
        "data.subset": "smoke", "data.smoke_users": (16, 10, 10), "data.train_rows": OVERFIT_ROWS,
        "prompt.train_curriculum": "all_hybrid", "train.epochs": 25, "train.micro_batch_size": 4,
        "train.grad_accum": 1}),
}
PROFILES = tuple(OVERLAYS)


def _load(name: str, *sets: str):
    cfg, _ = load_config(CONFIG_DIR / f"{name}.yaml", list(sets))
    return cfg


def _flat(cfg) -> dict:
    out = {}
    for section in dataclasses.fields(cfg):
        obj = getattr(cfg, section.name)
        for f in dataclasses.fields(obj):
            out[f"{section.name}.{f.name}"] = getattr(obj, f.name)
    return out


def _sets(name: str) -> tuple[str, ...]:
    return (f"cluster.artifact_dir={ARTIFACT}",) if name in NEEDS_ARTIFACT_DIR else ()


@pytest.mark.criterion("M14-1")
@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads(name):
    """Every profile loads with M0 load_config; the cluster_hard ones only with cluster.artifact_dir."""
    if name in NEEDS_ARTIFACT_DIR:
        with pytest.raises(ConfigError, match="artifact_dir"):
            _load(name)
    cfg = _load(name, *_sets(name))
    assert isinstance(cfg, Config)
    cfg.validate()


@pytest.mark.criterion("M14-1")
def test_profile_files():
    """17 files exactly, comments and values in ASCII only (§18.6.3), every key spelled out in paper_spec."""
    assert sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) == sorted(PROFILES)
    for path in CONFIG_DIR.glob("*.yaml"):
        assert all(b < 128 for b in path.read_bytes()), path.name
    raw = yaml.safe_load((CONFIG_DIR / "paper_spec.yaml").read_text(encoding="utf-8"))
    assert "extends" not in raw
    flat = {f"{s}.{k}" for s, keys in raw.items() for k in keys}
    assert flat == set(KEY_TABLE)
    assert raw["projector"] == {}


@pytest.mark.criterion("M14-2")
def test_resolved_differences():
    """Resolved overlays differ from paper_spec exactly in the keys of the §18.6.4-§18.6.5 tables."""
    base = _flat(_load("paper_spec"))
    for name, (parent, expected) in OVERLAYS.items():
        flat = _flat(_load(name, *_sets(name)))
        diff = {key for key in base if flat[key] != base[key]}
        assert diff == set(expected), name
        for key, value in expected.items():
            assert flat[key] == value, (name, key)
        raw = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        assert raw.get("extends") == parent, name
        if parent is None:
            continue
        file_keys = {f"{s}.{k}" for s, keys in raw.items() if s != "extends" for k in keys}
        own = set(expected) - {"cluster.artifact_dir"}
        if name == "variant_user_level_cluster":
            own -= {"router.routing_mode"}           # inherited from cluster_hard.yaml
        assert file_keys == own, name


@pytest.mark.criterion("M14-3")
def test_paper_spec_values_and_labels():
    """paper_spec values, types and labels equal the §18.6.2 table (document <-> code)."""
    assert len(KEY_TABLE) == 87
    flat = _flat(_load("paper_spec"))
    assert set(flat) == set(KEY_TABLE)
    for key, (_, value, _) in KEY_TABLE.items():
        assert flat[key] == value, key
        assert type(flat[key]) is type(value), key
    labels = {f"{s}.{k}": label for s, k, label, _ in iter_items()}
    assert set(labels) == set(KEY_TABLE)
    for key, (_, _, label) in KEY_TABLE.items():
        assert labels[key] == label, key
    hints = typing.get_type_hints(Config)
    for key, (tp, _, _) in KEY_TABLE.items():
        section, name = key.split(".")
        cls = hints[section]
        assert typing.get_type_hints(cls)[name] == tp, key
        field = next(f for f in dataclasses.fields(cls) if f.name == name)
        assert field.metadata.get("choices") == SCALAR_CHOICES.get(key), key
