"""M0 설정 스키마 (DESIGN.md §4.6.2, §3.1 ``Config``, 전체 키 표 §18.6.2).

One frozen dataclass per config section, in the section order of DESIGN.md
§3.1. Every key is declared with ``config.item(label, doc, choices=...)``:
the label is the evidence label of the key copied from DESIGN.md §18.6.2
(backticks removed), the type and the choices come from the module tables
(§5.6.8 M1, §6.6.7 M2, §7.6.7 M3, §8.6.8 M4, §9.6.9 M5, §10.6.7 M6,
§11.6.8 M7, §12.6.7 M8, §13.6.11 M9, §14.6.9 M10, §15.6.12 M11,
§16.6.9 M12, §19.6.10 M15). There are 87 keys; ``projector`` has none.

``validate()`` of a section implements the rules its chapter lists and raises
``ConfigError``; ``Config.validate()`` adds the cross-section rules of the
§3.1 ``Config`` row. ``config.build_config`` calls both.

List-valued keys whose elements are restricted ("⊆ {...}") are checked in
``validate()`` because ``config.item(choices=...)`` compares the whole value.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from typing import Any, Iterable, Optional, Sequence

from .config import ConfigError, item
from .run import EXPECTED_INPUTS

# ----------------------------------------------------------------------------
# Shared constants (values fixed by the design, not config keys)
# ----------------------------------------------------------------------------
SECTION_NAMES = ("run", "data", "candidates", "sasrec", "prompt", "batching", "projector", "router",
                 "lora", "model", "train", "eval", "records", "cluster")   # §3.1 Config, §4.6.2

# [코드] U model/model_interface.py:369, L model/model_interface.py:220 (D7); must equal
# moe_lora.CANONICAL_TARGETS (§12.6.1). Kept here so the schema does not import torch code.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EVAL_SPLITS = ("val", "test")                                   # §15.6.12
EVAL_FIXED_CHECKPOINTS = ("best_val", "last", "init")           # §15.6.12, plus epoch<k>
EVAL_CHECKPOINT_RE = re.compile(r"epoch(0|[1-9][0-9]*)")
EXPORT_CHECKPOINTS = ("best_val", "last")                       # §16.6.9 (D11)
ANALYSIS_SPLITS = ("test", "val", "train")                      # §16.6.9
FIGURE_FORMATS = ("png", "pdf")                                 # §16.6.9 [SPEC §6]
TRAIN_ROWS_MAX = 64                                             # §5.6.7 (1-64 rows)
# D9 and §13.6.11: autocast dtype and base dtype must be the same.
DTYPE_FOR_PRECISION = {"fp16_amp": "float16", "bf16_amp": "bfloat16", "fp32": "float32"}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


# ----------------------------------------------------------------------------
# Small validation helpers (all raise ConfigError naming section.key)
# ----------------------------------------------------------------------------
def _fail(where: str, message: str) -> None:
    raise ConfigError(f"{where}: {message}")


def _at_least(where: str, value: int, low: int) -> None:
    if value < low:
        _fail(where, f"must be >= {low}, got {value!r}")


def _number(where: str, value: float, *, low: Optional[float] = None, low_open: bool = False,
            high: Optional[float] = None, high_open: bool = False) -> None:
    if not math.isfinite(value):
        _fail(where, f"must be finite, got {value!r}")
    if low is not None and (value <= low if low_open else value < low):
        _fail(where, f"must be {'>' if low_open else '>='} {low}, got {value!r}")
    if high is not None and (value >= high if high_open else value > high):
        _fail(where, f"must be {'<' if high_open else '<='} {high}, got {value!r}")


def _sha256(where: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        _fail(where, f"must be 64 lowercase hex characters (a quoted sha256), got {value!r}")


def _optional_path(where: str, value: Optional[str]) -> None:
    if value is not None and not value.strip():
        _fail(where, "must be null or a non-empty path")


def _subset(where: str, values: Sequence[Any], allowed: Iterable[Any]) -> None:
    """Non-empty list without duplicates whose elements are all in ``allowed``."""
    allowed = tuple(allowed)
    if not values:
        _fail(where, "must not be empty")
    if len(set(values)) != len(values):
        _fail(where, f"must not repeat values, got {list(values)}")
    bad = [v for v in values if v not in allowed]
    if bad:
        _fail(where, f"{bad} not in {list(allowed)}")


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """``run`` 구역 (M0 §4.6.2): ``seed`` 하나."""
    seed: int = item("[SPEC §0.6] (D2, 계보 [코드] U main.py:84)",
                     "모든 derive_seed의 main seed (단일 실행, D2)")

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class DataConfig:
    """``data`` 구역 (M1 §5.6.8)."""
    title_mode: str = item("[코드] U data/movielens_data.py:61-66, :74 (D20)",
                           "제목 정규화. fixed는 variant_fixed_titles (§5.6.6)", choices=("code", "fixed"))
    subset: str = item("[결정] D14", "smoke면 §5.6.7로 사용자 선택", choices=("full", "smoke"))
    smoke_users: tuple[int, ...] = item("[결정] D14",
                                        "(train, val, test) smoke 사용자 수. full이면 범위 검사만 함")
    train_rows: tuple[int, ...] = item("[결정]",
                                       "비어 있으면 미사용. 비어 있지 않으면 smoke train을 이 행들의 표본으로 "
                                       "고정 (§5.6.7, overfit_gpu만)")

    def validate(self) -> None:
        # §5.6.8 validate(): len(smoke_users) == 3, each >= 1
        if len(self.smoke_users) != 3:
            _fail("data.smoke_users", f"must have 3 values (train, val, test), got {list(self.smoke_users)}")
        for value in self.smoke_users:
            _at_least("data.smoke_users", value, 1)
        # §5.6.8: non-empty train_rows -> subset smoke, 1-64 rows, strictly ascending, >= 0
        if self.train_rows:
            if self.subset != "smoke":
                _fail("data.train_rows", f"needs data.subset == smoke, got {self.subset!r}")
            if len(self.train_rows) > TRAIN_ROWS_MAX:
                _fail("data.train_rows", f"at most {TRAIN_ROWS_MAX} rows, got {len(self.train_rows)}")
            if self.train_rows[0] < 0:
                _fail("data.train_rows", f"rows must be >= 0, got {self.train_rows[0]}")
            for a, b in zip(self.train_rows, self.train_rows[1:]):
                if not a < b:
                    _fail("data.train_rows", f"must be strictly ascending, got {a} before {b}")


@dataclass(frozen=True)
class CandidatesConfig:
    """``candidates`` 구역 (M2 §6.6.7)."""
    num_candidates: int = item("[논문 App B p.16] [LLaRA §5.1.3 p.7] [코드] U main.py:122 (D10)",
                               "정답 포함 후보 수. cand_ids [B, 20]과 템플릿 문구 '20 movie titles'에 묶임",
                               choices=(20,))

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class SasrecConfig:
    """``sasrec`` 구역 (M3 §7.6.7). 경로는 절대 경로 또는 저장소 루트 기준 (run.resolve_path)."""
    source_path: str = item("[코드] U·L 같은 blob 41c057dc", "원본 pickle. 변환 도구 입력, meta.json 해시 기록")
    source_sha256: str = item("[코드] (계산값)",
                              "원본 pickle sha256. M0 EXPECTED_INPUTS['rec_model/movielens.pt']와 같아야 함")
    state_path: str = item("[결정]", "변환 state 파일 (§7.6.1 도구 출력)")
    state_digest: str = item("[결정] (prototype 계산값)", "변환 state 내용 digest (§7.6.2)")
    reference_path: str = item("[결정]", "원본 코드 기준 출력 (check-env, M13)")

    def validate(self) -> None:
        _sha256("sasrec.source_sha256", self.source_sha256)
        _sha256("sasrec.state_digest", self.state_digest)
        # §7.6.7: must equal the M0 expected hash of the original pickle (else a config error)
        expected = EXPECTED_INPUTS["rec_model/movielens.pt"][0]
        if self.source_sha256 != expected:
            _fail("sasrec.source_sha256", f"{self.source_sha256} != EXPECTED_INPUTS value {expected}")
        for name in ("source_path", "state_path", "reference_path"):
            if not getattr(self, name).strip():
                _fail(f"sasrec.{name}", "must be a non-empty path")


@dataclass(frozen=True)
class PromptConfig:
    """``prompt`` 구역 (M4 §8.6.8)."""
    template_path: str = item("[코드]", "템플릿 파일 (저장소 루트 기준). 다른 값은 M0 해시 검사와 어긋나 거부",
                              choices=("prompt/movie.txt",))
    template_choice: str = item("[LLaRA §5.1.2 p.7] (D15)", "D15. fixed는 variant_fixed_template",
                                choices=("random", "fixed"))
    fixed_template_index: Optional[int] = item("[결정]", "fixed면 0 이상 정수 (상한은 M4가 템플릿 수로 확인), "
                                                       "random이면 null")
    train_curriculum: str = item("[LLaRA 간접 §4.3 p.6 식 9-11] (D3)", "D3. all_hybrid·reverse는 변형",
                                 choices=("linear", "all_hybrid", "reverse"))
    eval_prompt: str = item("[논문 p.4 식 6-7 (암시)] (D4)", "D4. text_only는 E3", choices=("hybrid", "text_only"))

    def validate(self) -> None:
        # §8.6.8: template_choice and fixed_template_index must agree
        if self.template_choice == "fixed":
            if self.fixed_template_index is None:
                _fail("prompt.fixed_template_index", "must be an integer when template_choice is fixed")
            _at_least("prompt.fixed_template_index", self.fixed_template_index, 0)
        elif self.fixed_template_index is not None:
            _fail("prompt.fixed_template_index",
                  f"must be null when template_choice is random, got {self.fixed_template_index!r}")


@dataclass(frozen=True)
class BatchingConfig:
    """``batching`` 구역 (M5 §9.6.9)."""
    label_mode: str = item("[결정] D18", "D18. released_pair = variant_released_label",
                           choices=("answer_only", "released_pair"))
    num_workers: int = item("[코드] U main.py:83", "DataLoader worker 수 (0 이상). 결과와 무관 (§4.6.4)")
    max_seq_len: int = item("[결정]", "표본 토큰 수 상한 (1 이상). 넘으면 오류, 자르지 않음")
    tokenizer_sha256: str = item("[결정]", "tokenizer.model 내용 sha256")

    def validate(self) -> None:
        _at_least("batching.num_workers", self.num_workers, 0)
        _at_least("batching.max_seq_len", self.max_seq_len, 1)
        _sha256("batching.tokenizer_sha256", self.tokenizer_sha256)


@dataclass(frozen=True)
class ProjectorConfig:
    """``projector`` 구역 (M6 §10.6.7): 키 없음, 프로파일에는 ``projector: {}``."""

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class RouterConfig:
    """``router`` 구역 (M7 §11.6.8). K는 ``lora.num_experts``."""
    routing_mode: str = item("[논문 p.6 식 11] (D1), [SPEC §9]", "ω 공급 방식. E1 프로파일 none, E4 cluster_hard",
                             choices=("dynamic", "cluster_hard", "none"))

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class LoraConfig:
    """``lora`` 구역 (M8 §12.6.7)."""
    r: int = item("[코드] D7", "전체 rank, U main.py:128")
    alpha: int = item("[코드] D7", "LoRA alpha, U main.py:129")
    dropout: float = item("[코드] D7", "A 앞 입력 dropout, U main.py:130")
    num_experts: int = item("[논문 p.9 §4.3.2]", "K. r % K == 0 (식 9). K의 유일한 정의, E1 프로파일은 1")
    scaling_rule: str = item("[결정] D8", "expert_rank: alpha/r* (= 16), total_rank: alpha/r (= 4)",
                             choices=("expert_rank", "total_rank"))
    target_modules: tuple[str, ...] = item("[코드] D7", "대상 Linear ⊆ CANONICAL_TARGETS. 빈 목록·중복·목록 밖 거부")

    def validate(self) -> None:
        # §12.6.6 / §12.6.7
        _at_least("lora.r", self.r, 1)
        _at_least("lora.alpha", self.alpha, 1)
        _at_least("lora.num_experts", self.num_experts, 1)
        if self.r % self.num_experts != 0:
            _fail("lora.r", f"r % num_experts must be 0 (식 9), got r={self.r}, num_experts={self.num_experts}")
        _number("lora.dropout", self.dropout, low=0.0, high=1.0, high_open=True)
        _subset("lora.target_modules", self.target_modules, LORA_TARGETS)


@dataclass(frozen=True)
class ModelConfig:
    """``model`` 구역 (M9 §13.6.11)."""
    backbone: str = item("[논문 App B p.16], base는 [코드] U README.md:13", "tiny_random은 CPU smoke·test 전용",
                         choices=("llama2_7b", "tiny_random"))
    torch_dtype: str = item("[결정] D9", "base 로드 dtype. bfloat16은 D9 대안 프로파일만, float32는 tiny_random만",
                            choices=("float16", "bfloat16", "float32"))
    device: str = item("[SPEC §0.12]", "cpu는 tiny_random만", choices=("cuda:0", "cpu"))
    gen_strategy: str = item("[코드] U model/model_interface.py:70, L model/model_interface.py:43",
                             "do_sample False, num_beams 1", choices=("greedy",))
    gen_max_new_tokens: int = item("[코드] U …:70, :80", "생성 최대 새 토큰 수")
    gen_min_new_tokens: int = item("[코드] U …:70, :81", "생성 최소 새 토큰 수")
    gen_repetition_penalty: float = item("[코드] U …:70, :83", "1.0이면 processor 없음")

    def validate(self) -> None:
        # §13.6.11 validate: tiny_random <=> (cpu, float32); llama2_7b => cuda:0 and float16/bfloat16
        tiny = self.backbone == "tiny_random"
        if tiny != (self.device == "cpu" and self.torch_dtype == "float32"):
            _fail("model.backbone", f"tiny_random <=> (device cpu, torch_dtype float32); got backbone="
                                    f"{self.backbone}, device={self.device}, torch_dtype={self.torch_dtype}")
        if self.backbone == "llama2_7b" and (self.device != "cuda:0"
                                             or self.torch_dtype not in ("float16", "bfloat16")):
            _fail("model.backbone", f"llama2_7b needs device cuda:0 and torch_dtype float16/bfloat16; got "
                                    f"device={self.device}, torch_dtype={self.torch_dtype}")
        _at_least("model.gen_max_new_tokens", self.gen_max_new_tokens, 1)
        _at_least("model.gen_min_new_tokens", self.gen_min_new_tokens, 0)
        if self.gen_min_new_tokens > self.gen_max_new_tokens:
            _fail("model.gen_min_new_tokens",
                  f"must be <= gen_max_new_tokens ({self.gen_min_new_tokens} > {self.gen_max_new_tokens})")
        _number("model.gen_repetition_penalty", self.gen_repetition_penalty, low=0.0, low_open=True)


@dataclass(frozen=True)
class TrainConfig:
    """``train`` 구역 (M10 §14.6.9). 그룹 값은 mapping이 아니라 그룹마다 스칼라 키."""
    epochs: int = item("[v1 App E p.16] [LLaRA §5.1.2 p.7] D11", "epoch 수")
    micro_batch_size: int = item("[결정] 값은 [코드] U train_movielens.sh:5", "micro-batch 크기, drop_last")
    grad_accum: int = item("[결정] 값은 [코드] U train_movielens.sh:6, 곱 128 [v1 App E p.16] [LLaRA §5.1.2 p.7]",
                           "창 크기 A (accumulation)")
    lr_peak: float = item("[논문 App E p.17] D5", "최대 lr")
    lr_warmup_start: float = item("[v1 App E p.16] [LLaRA §5.1.2 p.7] D6", "warmup 시작 lr")
    lr_floor: float = item("[코드] U main.py:90 D6", "cosine 끝값")
    warmup_divisor: int = item("[코드] U model/model_interface.py:248 D6", "w = T // 값")
    lr_scale_projector: float = item("[논문 App E p.17] D19", "projector 그룹 lr 배수")
    lr_scale_gate: float = item("[논문 p.5 §3.2 (2)] (암시) D19", "gate 그룹 lr 배수. variant_released_gate_lr는 0.3")
    lr_scale_lora: float = item("[논문 App E p.17] D19", "LoRA 그룹 lr 배수")
    weight_decay_projector: float = item("[v1 App E p.16] 값, 적용 범위 [코드] U model/model_interface.py:228 D19",
                                         "projector 그룹 Adam coupled L2")
    weight_decay_gate: float = item("[v1 App E p.16] 값, 적용 범위 [코드] U model/model_interface.py:230 D19",
                                    "gate 그룹 Adam coupled L2")
    weight_decay_lora: float = item("[코드] U model/model_interface.py:232 D19", "LoRA 그룹 Adam coupled L2")
    adam_beta1: float = item("[코드] torch 2.0.0 기본값, U·L 기본값 사용", "Adam beta1")
    adam_beta2: float = item("[코드] 같음", "Adam beta2 (torch 2.0.0 기본값, U·L 기본값 사용)")
    adam_eps: float = item("[코드] 같음", "Adam eps (torch 2.0.0 기본값, U·L 기본값 사용)")
    precision: str = item("[논문 App E p.17] D9", "fp16_amp (D9), bf16_amp (대안), fp32 (CPU smoke 전용)",
                          choices=("fp16_amp", "bf16_amp", "fp32"))
    scaler_init_scale: float = item("[코드] torch 2.0.0 GradScaler 기본값", "GradScaler init_scale")
    scaler_growth_factor: float = item("[코드] 같음", "GradScaler growth_factor (torch 2.0.0 기본값)")
    scaler_backoff_factor: float = item("[코드] 같음", "GradScaler backoff_factor (torch 2.0.0 기본값)")
    scaler_growth_interval: int = item("[코드] 같음", "GradScaler growth_interval (torch 2.0.0 기본값)")
    max_consecutive_skips: int = item("[결정]", "연속 skip 한계 (§14.6.8)")
    log_every_steps: int = item("[결정]", "run.log 요약 간격 (optimizer step)")

    def validate(self) -> None:
        # §14.6.9 type column and validate()
        for name in ("epochs", "micro_batch_size", "grad_accum", "warmup_divisor", "scaler_growth_interval",
                     "max_consecutive_skips", "log_every_steps"):
            _at_least(f"train.{name}", getattr(self, name), 1)
        for name in ("lr_peak", "lr_warmup_start", "lr_scale_projector", "lr_scale_gate", "lr_scale_lora",
                     "adam_eps", "scaler_init_scale"):
            _number(f"train.{name}", getattr(self, name), low=0.0, low_open=True)
        for name in ("lr_floor", "weight_decay_projector", "weight_decay_gate", "weight_decay_lora"):
            _number(f"train.{name}", getattr(self, name), low=0.0)
        for name in ("adam_beta1", "adam_beta2"):
            _number(f"train.{name}", getattr(self, name), low=0.0, high=1.0, high_open=True)
        _number("train.scaler_growth_factor", self.scaler_growth_factor, low=1.0, low_open=True)
        _number("train.scaler_backoff_factor", self.scaler_backoff_factor, low=0.0, low_open=True,
                high=1.0, high_open=True)
        if self.lr_warmup_start > self.lr_peak:
            _fail("train.lr_warmup_start", f"must be <= lr_peak ({self.lr_warmup_start} > {self.lr_peak})")
        if self.lr_floor > self.lr_peak:
            _fail("train.lr_floor", f"must be <= lr_peak ({self.lr_floor} > {self.lr_peak})")


@dataclass(frozen=True)
class EvalConfig:
    """``eval`` 구역 (M11 §15.6.12)."""
    batch_size: int = item("[결정] (계보 [코드] U train_movielens.sh:5)",
                           "평가 batch 크기. drop_last 없음. 모든 실험 프로파일이 같은 값")
    batch_invariance_check: bool = item("[결정]", "true면 batch 1로 다시 생성해 비교 (§15.6.10)")
    source_run: Optional[str] = item("[결정]", "eval 명령이 읽을 학습 실행 폴더. eval 명령에서 필수")
    checkpoints: tuple[str, ...] = item("[결정] D11", "eval 명령 전용. 원소는 best_val, last, epoch<k>, init")
    splits: tuple[str, ...] = item("[결정] D11·D17", "eval 명령 전용. ⊆ {val, test}")

    def validate(self) -> None:
        # §15.6.12 validate()
        _at_least("eval.batch_size", self.batch_size, 1)
        _optional_path("eval.source_run", self.source_run)
        if not self.checkpoints:
            _fail("eval.checkpoints", "must not be empty")
        if len(set(self.checkpoints)) != len(self.checkpoints):
            _fail("eval.checkpoints", f"must not repeat values, got {list(self.checkpoints)}")
        for tag in self.checkpoints:
            if tag not in EVAL_FIXED_CHECKPOINTS and not EVAL_CHECKPOINT_RE.fullmatch(tag):
                _fail("eval.checkpoints", f"{tag!r} is not best_val, last, init or epoch<k>")
        _subset("eval.splits", self.splits, EVAL_SPLITS)
        # D11: test is evaluated only with the checkpoints chosen before looking at test
        if "test" in self.splits:
            other = [t for t in self.checkpoints if t not in EXPORT_CHECKPOINTS]
            if other:
                _fail("eval.checkpoints", f"with test in eval.splits only best_val/last are allowed, got {other}")


@dataclass(frozen=True)
class RecordsConfig:
    """``records`` 구역 (M12 §16.6.9)."""
    flush_every_steps: int = item("[결정]", "학습 기록을 optimizer step N번마다 파일에 씀")
    activation_max: bool = item("[결정]", "MoE-LoRA 층 출력의 micro-batch별 최대 |값| 계측")
    gate_rowsum_atol: float = item("[결정]", "ω 행 합 허용오차")
    gate_crosscheck_atol: float = item("[결정]", "export ω와 EvalRecord ω의 최대 차")
    source_run: Optional[str] = item("[결정]", "export가 읽을 학습 실행 폴더. export에서는 필수 (--set)")
    export_checkpoints: tuple[str, ...] = item("[결정] D11", "export·분석할 checkpoint ⊆ {best_val, last}")
    export_num_threads: int = item("[결정]", "export CPU thread 수 (결정적 z)")
    export_batch_size: int = item("[결정]", "export batch 크기")
    analysis_splits: tuple[str, ...] = item("[결정]", "gate 분석 split ⊆ {test, val, train}")
    figure_formats: tuple[str, ...] = item("[SPEC §6]", "그림 형식 ⊆ {png, pdf}")

    def validate(self) -> None:
        _at_least("records.flush_every_steps", self.flush_every_steps, 1)
        _number("records.gate_rowsum_atol", self.gate_rowsum_atol, low=0.0, low_open=True)
        _number("records.gate_crosscheck_atol", self.gate_crosscheck_atol, low=0.0, low_open=True)
        _optional_path("records.source_run", self.source_run)
        _subset("records.export_checkpoints", self.export_checkpoints, EXPORT_CHECKPOINTS)
        _at_least("records.export_num_threads", self.export_num_threads, 1)
        _at_least("records.export_batch_size", self.export_batch_size, 1)
        _subset("records.analysis_splits", self.analysis_splits, ANALYSIS_SPLITS)
        _subset("records.figure_formats", self.figure_formats, FIGURE_FORMATS)


@dataclass(frozen=True)
class ClusterConfig:
    """``cluster`` 구역 (M15 §19.6.10). paper_spec(dynamic)은 읽지 않지만 모든 키를 적음."""
    unit: str = item("[결정] D25", "군집 단위. user는 variant_user_level_cluster", choices=("sequence", "user"))
    n_clusters: int = item("[SPEC §7]", "군집 수 (2 이상)")
    random_state: int = item("[SPEC §7]", "KMeans 난수. derive_seed 예외 (§4.6.4)")
    n_init: int = item("[SPEC §7] 값 [결정]", "KMeans n_init (scikit-learn 1.2.2 기본값 명시)")
    init: str = item("[결정]", "KMeans init (1.2.2 기본값 명시)", choices=("k-means++",))
    algorithm: str = item("[결정]", "KMeans algorithm (1.2.2 기본값 명시)", choices=("lloyd",))
    max_iter: int = item("[결정]", "KMeans max_iter (1.2.2 기본값 명시)")
    tol: float = item("[결정]", "KMeans tol (1.2.2 기본값 명시, 차원별 분산 평균에 곱하는 상대값)")
    fit_num_threads: int = item("[결정]", "fit 중 OpenMP thread 수 (결정성)")
    expert_mapping: str = item("[SPEC §9]", "군집 -> expert. 형식 검사는 M15 parse_expert_mapping (§19.6.7)")
    artifact_dir: Optional[str] = item("[결정]", "cluster_hard 학습이 읽을 cluster/ 폴더. cluster_hard에서는 --set 필수")
    dynamic_export_dir: Optional[str] = item("[결정]", "dynamic 비교용 E2 export 폴더 (records/export/epoch{a})")
    z_crosscheck_atol: float = item("[결정]", "E2 export z와 군집 z의 최대 차")
    live_check_n: int = item("[결정]", "학습 전 live z 대조 표본 수")
    live_check_atol: float = item("[결정]", "live(학습 장치) z와 저장(CPU) z의 최대 차")

    def validate(self) -> None:
        _at_least("cluster.n_clusters", self.n_clusters, 2)
        _at_least("cluster.n_init", self.n_init, 1)
        _at_least("cluster.max_iter", self.max_iter, 1)
        _number("cluster.tol", self.tol, low=0.0, low_open=True)
        _at_least("cluster.fit_num_threads", self.fit_num_threads, 1)
        if not self.expert_mapping:
            _fail("cluster.expert_mapping", "must not be empty")
        _optional_path("cluster.artifact_dir", self.artifact_dir)
        _optional_path("cluster.dynamic_export_dir", self.dynamic_export_dir)
        _number("cluster.z_crosscheck_atol", self.z_crosscheck_atol, low=0.0, low_open=True)
        _at_least("cluster.live_check_n", self.live_check_n, 1)
        _number("cluster.live_check_atol", self.live_check_atol, low=0.0, low_open=True)


# ----------------------------------------------------------------------------
# The whole configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """확정된 설정 (§3.1 ``Config``). 다른 모듈은 이 객체만 받음."""
    run: RunConfig
    data: DataConfig
    candidates: CandidatesConfig
    sasrec: SasrecConfig
    prompt: PromptConfig
    batching: BatchingConfig
    projector: ProjectorConfig
    router: RouterConfig
    lora: LoraConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    records: RecordsConfig
    cluster: ClusterConfig

    def validate(self) -> None:
        """Section rules, then the cross-section rules of the §3.1 ``Config`` row."""
        for f in fields(self):
            getattr(self, f.name).validate()
        mode = self.router.routing_mode
        # §11.6.8: cluster_hard <=> cluster.artifact_dir given
        if (mode == "cluster_hard") != (self.cluster.artifact_dir is not None):
            _fail("cluster.artifact_dir", "router.routing_mode == cluster_hard <=> cluster.artifact_dir is set "
                                          f"(routing_mode={mode}, artifact_dir={self.cluster.artifact_dir!r}); "
                                          "cluster_hard takes --set cluster.artifact_dir=<cluster run>/cluster")
        # §11.6.8: none <=> lora.num_experts == 1
        if (mode == "none") != (self.lora.num_experts == 1):
            _fail("router.routing_mode", "routing_mode == none <=> lora.num_experts == 1 "
                                         f"(routing_mode={mode}, num_experts={self.lora.num_experts})")
        # §13.6.11, D9: train.precision <-> model.torch_dtype
        expected_dtype = DTYPE_FOR_PRECISION[self.train.precision]
        if self.model.torch_dtype != expected_dtype:
            _fail("train.precision", f"{self.train.precision} needs model.torch_dtype {expected_dtype}, "
                                     f"got {self.model.torch_dtype}")
        # §3.1: tiny_random <=> (cpu, float32) (also in ModelConfig.validate)
        tiny = self.model.backbone == "tiny_random"
        if tiny != (self.model.device == "cpu" and self.model.torch_dtype == "float32"):
            _fail("model.backbone", "tiny_random <=> (device cpu, torch_dtype float32)")
        # §3.1 section rule: non-empty train_rows -> subset smoke (also in DataConfig.validate)
        if self.data.train_rows and self.data.subset != "smoke":
            _fail("data.train_rows", "needs data.subset == smoke")


__all__ = [
    "Config", "RunConfig", "DataConfig", "CandidatesConfig", "SasrecConfig", "PromptConfig", "BatchingConfig",
    "ProjectorConfig", "RouterConfig", "LoraConfig", "ModelConfig", "TrainConfig", "EvalConfig", "RecordsConfig",
    "ClusterConfig", "SECTION_NAMES", "LORA_TARGETS", "EVAL_SPLITS", "EVAL_FIXED_CHECKPOINTS", "EVAL_CHECKPOINT_RE",
    "EXPORT_CHECKPOINTS", "ANALYSIS_SPLITS", "FIGURE_FORMATS", "TRAIN_ROWS_MAX", "DTYPE_FOR_PRECISION",
]
