"""M0 실제 설정 schema·프로파일·main.py (DESIGN.md §4.8 #1-#2, §17.6.9 M0 표, §17.8 #5)."""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import typing
from pathlib import Path
from typing import Optional

import pytest

from ilora import sections
from ilora.config import LABEL_RE, ConfigError, config_hash, config_to_dict, dump_config, iter_items, load_config
from ilora.sections import Config

REBUILD_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REBUILD_DIR / "configs"
PROFILES = (
    "paper_spec", "llara_baseline", "cluster_hard",
    "variant_all_hybrid", "variant_reverse_curriculum", "variant_text_only_eval", "variant_released_lr",
    "variant_released_gate_lr", "variant_released_label", "variant_total_rank_scaling", "variant_fixed_template",
    "variant_fixed_titles", "variant_user_level_cluster", "variant_bf16",
    "smoke_cpu", "smoke_gpu_100", "overfit_gpu",
)
NEEDS_ARTIFACT_DIR = ("cluster_hard", "variant_user_level_cluster")
ANY_ARTIFACT_DIR = "cluster.artifact_dir=runs/rebuild/cluster_run_for_tests/cluster"
SUPPORTED_TYPES = (int, float, bool, str, Optional[int], Optional[str], tuple[int, ...], tuple[str, ...])


def _sets_for(name: str) -> tuple[str, ...]:
    return (ANY_ARTIFACT_DIR,) if name in NEEDS_ARTIFACT_DIR else ()


@pytest.mark.criterion("M0-2")
def test_sections_follow_contract():
    """14 sections in the §3.1 order, frozen dataclasses, 87 keys, projector empty."""
    names = tuple(f.name for f in dataclasses.fields(Config))
    assert names == sections.SECTION_NAMES
    assert len(names) == 14
    hints = typing.get_type_hints(Config)
    total = 0
    for name in names:
        cls = hints[name]
        assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, name
        assert callable(getattr(cls, "validate", None)), name
        total += len(dataclasses.fields(cls))
    assert Config.__dataclass_params__.frozen
    assert total == 87
    assert dataclasses.fields(sections.ProjectorConfig) == ()


@pytest.mark.criterion("M0-2")
def test_every_key_has_label():
    """Every declared key has an evidence label (§0.1) and a description."""
    items = list(iter_items())
    assert len(items) == 87
    assert len({(s, k) for s, k, _, _ in items}) == 87
    for section, key, label, doc in items:
        assert section in sections.SECTION_NAMES
        assert label and LABEL_RE.match(label), f"{section}.{key}: {label!r}"
        assert doc and doc.strip(), f"{section}.{key} has no description"


@pytest.mark.criterion("M0-1")
def test_key_types_supported():
    """Every key uses a type that the strict loader supports; choices only on scalar keys."""
    hints = typing.get_type_hints(Config)
    for name in sections.SECTION_NAMES:
        cls = hints[name]
        field_types = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            tp = field_types[f.name]
            assert tp in SUPPORTED_TYPES, f"{name}.{f.name}: {tp}"
            choices = f.metadata.get("choices")
            if choices is not None:
                assert tp in (int, str), f"{name}.{f.name}: choices on {tp}"
                assert all(isinstance(c, tp) for c in choices)


@pytest.mark.criterion("M0-1")
@pytest.mark.parametrize("name", PROFILES)
def test_profile_round_trip(name, profile_cfg, tmp_path):
    """Every profile loads, validates, and saving then reloading gives the same dict and hash."""
    cfg = profile_cfg(name, *_sets_for(name))
    cfg.validate()
    path = tmp_path / "config.yaml"
    dump_config(cfg, path)
    again, applied = load_config(path)
    assert applied == {}
    assert config_to_dict(again) == config_to_dict(cfg)
    assert config_hash(again) == config_hash(cfg)
    assert again == cfg


@pytest.mark.criterion("M0-1")
def test_profile_list_is_complete():
    assert sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) == sorted(PROFILES)


REJECTED = [
    # (profile, --set texts): every case must raise ConfigError
    ("paper_spec", ("data.smoke_users=[80,10]",)),
    ("paper_spec", ("data.smoke_users=[0,10,10]",)),
    ("paper_spec", ("data.train_rows=[1,2]",)),                       # train_rows needs subset smoke
    ("paper_spec", ("data.subset=smoke", "data.train_rows=[2,1]")),
    ("paper_spec", ("data.subset=smoke", "data.train_rows=[1,1]")),
    ("paper_spec", ("data.subset=smoke", "data.train_rows=[-1,2]")),
    ("paper_spec", ("data.subset=smoke", "data.train_rows=[" + ",".join(str(i) for i in range(65)) + "]")),
    ("paper_spec", ("data.title_mode=raw",)),
    ("paper_spec", ("candidates.num_candidates=21",)),
    ("paper_spec", ('sasrec.source_sha256="' + "a" * 64 + '"',)),
    ("paper_spec", ('sasrec.state_digest="abc"',)),
    ("paper_spec", ("prompt.template_path=prompt/game.txt",)),
    ("paper_spec", ("prompt.template_choice=fixed",)),
    ("paper_spec", ("prompt.fixed_template_index=0",)),
    ("paper_spec", ("prompt.template_choice=fixed", "prompt.fixed_template_index=-1")),
    ("paper_spec", ("prompt.train_curriculum=step",)),
    ("paper_spec", ("batching.num_workers=-1",)),
    ("paper_spec", ("batching.max_seq_len=0",)),
    ("paper_spec", ('batching.tokenizer_sha256="ABC"',)),
    ("paper_spec", ("lora.r=6",)),
    ("paper_spec", ("lora.r=0",)),
    ("paper_spec", ("lora.r=8.0",)),
    ("paper_spec", ("lora.alpha=0",)),
    ("paper_spec", ("lora.dropout=1.0",)),
    ("paper_spec", ("lora.target_modules=[]",)),
    ("paper_spec", ("lora.target_modules=[q_proj,q_proj]",)),
    ("paper_spec", ("lora.target_modules=[q_proj,lm_head]",)),
    ("paper_spec", ("lora.scaling_rule=alpha",)),
    ("paper_spec", ("model.device=cpu",)),
    ("paper_spec", ("model.torch_dtype=float32",)),
    ("paper_spec", ("model.backbone=tiny_random",)),
    ("paper_spec", ("model.gen_min_new_tokens=65",)),
    ("paper_spec", ("model.gen_strategy=beam",)),
    ("paper_spec", ("train.adam_beta1=1.0",)),
    ("paper_spec", ("train.adam_beta2=-0.1",)),
    ("paper_spec", ("train.lr_warmup_start=2.0e-4",)),
    ("paper_spec", ("train.lr_floor=2.0e-4",)),
    ("paper_spec", ("train.lr_peak=0.0",)),
    ("paper_spec", ("train.lr_peak=.inf",)),
    ("paper_spec", ("train.epochs=0",)),
    ("paper_spec", ("train.scaler_growth_factor=1.0",)),
    ("paper_spec", ("train.scaler_backoff_factor=1.0",)),
    ("paper_spec", ("train.lr_scale_projector={projector: 1.0}",)),   # M10 §14.6.9: no mapping values
    ("paper_spec", ("train.precision=fp32",)),                        # needs model.torch_dtype float32
    ("paper_spec", ("train.precision=bf16_amp",)),                    # needs model.torch_dtype bfloat16
    ("paper_spec", ("eval.batch_size=0",)),
    ("paper_spec", ("eval.splits=[train]",)),
    ("paper_spec", ("eval.splits=[]",)),
    ("paper_spec", ("eval.splits=[val,val]",)),
    ("paper_spec", ("eval.checkpoints=[epoch3]",)),                   # test only with best_val/last (D11)
    ("paper_spec", ("eval.checkpoints=[epoch03]", "eval.splits=[val]")),
    ("paper_spec", ("eval.checkpoints=[best]", "eval.splits=[val]")),
    ("paper_spec", ('eval.source_run=""',)),
    ("paper_spec", ("records.export_checkpoints=[init]",)),
    ("paper_spec", ("records.analysis_splits=[dev]",)),
    ("paper_spec", ("records.figure_formats=[svg]",)),
    ("paper_spec", ("records.gate_rowsum_atol=0.0",)),
    ("paper_spec", ("cluster.n_clusters=1",)),
    ("paper_spec", ("cluster.init=random",)),
    ("paper_spec", ("cluster.expert_mapping=1:1",)),                  # YAML 1.1 reads 1:1 as int 61
    ("paper_spec", ("cluster.artifact_dir=x/cluster",)),              # dynamic must not name artifacts
    ("paper_spec", ("router.routing_mode=cluster_hard",)),            # needs cluster.artifact_dir
    ("paper_spec", ("router.routing_mode=none",)),                    # none <=> num_experts == 1
    ("paper_spec", ("lora.num_experts=1",)),                          # dynamic with K = 1
    ("paper_spec", ('run.seed="1234"',)),
    ("smoke_cpu", ("model.device=cuda:0",)),
    ("smoke_cpu", ("train.precision=fp16_amp",)),
    ("cluster_hard", ()),
    ("variant_user_level_cluster", ()),
]


@pytest.mark.criterion("M0-1")
@pytest.mark.parametrize("profile,sets", REJECTED, ids=[f"{p}:{'|'.join(s) or '-'}" for p, s in REJECTED])
def test_section_rules_reject(profile, sets):
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / f"{profile}.yaml", list(sets))


ACCEPTED = [
    ("paper_spec", ("prompt.template_choice=fixed", "prompt.fixed_template_index=2")),
    ("paper_spec", ('cluster.expert_mapping="0:3,1:2,2:1,3:0"',)),
    ("paper_spec", ("cluster.expert_mapping=0:3,1:2,2:1,3:0",)),
    ("paper_spec", ("eval.checkpoints=[epoch0,epoch12,init]", "eval.splits=[val]")),
    ("paper_spec", ("data.subset=smoke", "data.train_rows=[0,3,64]")),
    ("paper_spec", ("lora.target_modules=[down_proj,q_proj]",)),
    ("llara_baseline", ()),
    ("variant_bf16", ()),
    ("smoke_cpu", ("router.routing_mode=cluster_hard", "cluster.artifact_dir=x/cluster", "train.epochs=1")),
    ("smoke_cpu", ("lora.num_experts=1", "router.routing_mode=none", "train.epochs=1")),
]


@pytest.mark.criterion("M0-1")
@pytest.mark.parametrize("profile,sets", ACCEPTED, ids=[f"{p}:{'|'.join(s) or '-'}" for p, s in ACCEPTED])
def test_valid_combinations_load(profile, sets):
    cfg, applied = load_config(CONFIG_DIR / f"{profile}.yaml", list(sets))
    assert set(applied) == {s.split("=", 1)[0] for s in sets}
    cfg.validate()


@pytest.mark.criterion("M0-1")
def test_override_values_keep_their_type():
    cfg, applied = load_config(CONFIG_DIR / "paper_spec.yaml",
                               ["train.lr_peak=8.0e-4", "train.lr_floor=1e-6", 'cluster.expert_mapping="0:3,1:2,2:1,3:0"'])
    assert cfg.train.lr_peak == pytest.approx(8.0e-4) and isinstance(cfg.train.lr_peak, float)
    assert cfg.train.lr_floor == pytest.approx(1e-6) and isinstance(cfg.train.lr_floor, float)
    assert cfg.cluster.expert_mapping == "0:3,1:2,2:1,3:0"
    assert cfg.lora.target_modules == tuple(sections.LORA_TARGETS)
    assert applied["train.lr_peak"] == pytest.approx(8.0e-4)


def _run_main(args, cwd):
    return subprocess.run([sys.executable, "-I", str(REBUILD_DIR / "main.py"), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


@pytest.mark.criterion("M13-5")
def test_main_help(tmp_path):
    """``python -I rebuild/main.py --help`` exits 0 from any folder and shows the six commands."""
    proc = _run_main(["--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for command in ("train", "eval", "export", "cluster", "smoke", "check-env"):
        assert command in proc.stdout
    assert "--config" in proc.stdout and "--set" in proc.stdout and "--resume" in proc.stdout


@pytest.mark.criterion("M0-1")
def test_main_rejects_bad_arguments_before_any_run(tmp_path):
    """Config errors and wrong argument combinations stop before a run folder is made."""
    bad_value = _run_main(["check-env", "--config", str(CONFIG_DIR / "paper_spec.yaml"), "--set", "lora.r=8.0"],
                          tmp_path)
    assert bad_value.returncode == 2 and "config error" in bad_value.stderr
    unknown = _run_main(["check-env", "--config", str(CONFIG_DIR / "paper_spec.yaml"), "--set", "train.lr=1"],
                        tmp_path)
    assert unknown.returncode == 2 and "unknown key" in unknown.stderr
    resume_eval = _run_main(["eval", "--resume", str(tmp_path)], tmp_path)
    assert resume_eval.returncode == 2 and "--resume is only for train" in resume_eval.stderr
    resume_with_config = _run_main(["train", "--resume", str(tmp_path), "--config",
                                    str(CONFIG_DIR / "paper_spec.yaml")], tmp_path)
    assert resume_with_config.returncode == 2
    no_config = _run_main(["train"], tmp_path)
    assert no_config.returncode == 2 and "--config is required" in no_config.stderr
    eval_without_source = _run_main(["eval", "--config", str(CONFIG_DIR / "paper_spec.yaml")], tmp_path)
    assert eval_without_source.returncode == 2 and "eval.source_run" in eval_without_source.stderr
    unknown_command = _run_main(["fit", "--config", str(CONFIG_DIR / "paper_spec.yaml")], tmp_path)
    assert unknown_command.returncode == 2
