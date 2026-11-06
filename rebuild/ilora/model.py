"""M9 LLM 연결 (DESIGN.md §13; D4, D9, D12, D13, D22, D23).

``build_model`` loads Llama-2-7B in fp16 (or the CPU test backbone ``tiny_random``),
adds the five token rows of M5 (``[PAD]``, ``[PH]``, ``[HistoryEmb]``, ``[CansEmb]``,
``[ItemEmb]``) inside a seed scope, freezes every LLaMA weight, wraps the 224 target
Linear layers with M8 MoE-LoRA and joins SASRec (M3), the projector (M6) and the
omega source (M7 ``Router``, which holds the M15 ``ClusterHardGate`` for cluster_hard)
into one ``ILoRAModel`` (§13.6.1, §13.6.2).

- ``train_forward(TrainBatch)``: z and omega once per batch (fp32, autocast off,
  §13.6.4), ``inputs_embeds`` with the behaviour tokens written at the
  ``[HistoryEmb]``/``[CansEmb]`` positions (hybrid only, §13.6.3), the LLaMA body
  inside ``GateContext.use(omega)`` and the mean cross entropy of the answer tokens
  computed from the label positions only (§13.6.5).
- ``generate(EvalBatch)``: the stock transformers 4.28.0 ``generate`` with
  ``inputs_embeds`` and the left-padded ``attention_mask`` (mask-based position_ids,
  D22), the same omega for every generation step (식 12, §13.6.6).
- ``param_groups``/``trainable_state``/``load_trainable_state``/
  ``generated_frozen_state``/``parameter_report`` for M10-M12 (§13.6.7, §13.6.8).

Every check raises ``ModelError`` with the cause; batch errors name the row and the
``sample_id`` (§13.6.10). Nothing here patches transformers (D13).
"""
from __future__ import annotations

import copy
import itertools
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Protocol, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from . import seeding
from .batching import (BASE_VOCAB, HIST_LEN, IGNORE_INDEX, NUM_CANDS, VOCAB_SIZE, EvalBatch, TokenIds,
                       TokenizerBundle, TrainBatch)
from .data import MIN_LEN_SEQ
from .moe_lora import GateContext, MoELoRALinear, attach_moe_lora, lora_named_parameters
from .projector import build_projector
from .router import HardGate, Router
from .sasrec import SASRecEncoder

if TYPE_CHECKING:
    from .data import ItemCatalog
    from .sections import Config, ModelConfig

log = logging.getLogger("ilora.model")


class ModelError(RuntimeError):
    """Tokenizer ids, LLaMA load, build, batch or generation check failed (§13.6.10). Never repaired."""


# ----------------------------------------------------------------------------
# Constants (§13.6.2, §13.6.9, §13.6.11)
# ----------------------------------------------------------------------------
BACKBONES = ("llama2_7b", "tiny_random")                     # model.backbone
TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}   # D9
PARAM_GROUPS = ("projector", "gate", "lora")                 # D19 groups, M10 GROUP_ORDER
OMEGA_SOURCE_PREFIX = "omega_source."                        # checkpoint key prefix of the gate (§13.6.8)
GENERATED_FROZEN_PREFIX = "generated_frozen."                # new token rows saved with the checkpoint (§13.6.8)
GENERATED_FROZEN_KEYS = ("generated_frozen.token_ids", "generated_frozen.embed_rows",
                         "generated_frozen.lm_head_rows")
# [논문 App B p.16] Llama2-7B; shapes of the public Llama-2-7b-hf config.json (§13.6.2 step 2).
LLAMA2_7B_SHAPE = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
                   "intermediate_size": 11008}
# [결정] CPU test backbone (§13.6.9): test constants, no experimental meaning.
TINY_LLAMA_SHAPE = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 128}
TINY_MAX_POSITIONS = 1024
# §13.6.2 step 2: the only key allowed to be missing is the rotary buffer computed at init (T4.28
# models/llama/modeling_llama.py LlamaRotaryEmbedding registers inv_freq).
_ALLOWED_MISSING_RE = re.compile(r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq")


# ----------------------------------------------------------------------------
# Contract values (§13.6.1, §3.1 ILoRAModel)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainForward:
    """Result of one training micro-batch (M10 §14.6.10 name)."""
    loss: Tensor                 # [] float32, attached to the graph: mean CE over the answer tokens
    n_label_tokens: int          # number of answer tokens in the micro-batch
    omega: Tensor                # [B, K] float32, detached, model device
    z: Tensor                    # [B, 64] float32, no grad, model device


@dataclass(frozen=True)
class GenerationOutput:
    """Result of one evaluation batch (M11 §15.6.1)."""
    new_token_ids: Tensor        # [B, T] int64 CPU, 1 <= T <= gen_max_new_tokens, BOS placeholder removed
    omega: Tensor                # [B, K] float32 CPU
    z: Tensor                    # [B, 64] float32 CPU


class SeqEncoder(Protocol):
    """What M9 uses of the M3 ``SASRecEncoder`` (§7.6.3 names)."""
    hidden_size: int
    state_digest: str

    def __call__(self, hist_ids: Tensor, len_seq: Tensor) -> Tensor: ...      # z [B, 64] fp32

    def embed_items(self, item_ids: Tensor) -> Tensor: ...                     # [n, 64] fp32

    def train(self, mode: bool = True) -> nn.Module: ...                       # stays eval (D12)


class OmegaSource(Protocol):
    """What M9 uses of the M7 ``Router`` (dynamic, cluster_hard and none, §11.6.1)."""
    num_experts: int

    def __call__(self, z: Tensor, sample_ids: Sequence[str]) -> Tensor: ...    # [B, K] fp32


def torch_dtype_of(name: str) -> torch.dtype:
    """``model.torch_dtype`` text -> torch dtype (D9)."""
    if name not in TORCH_DTYPES:
        raise ModelError(f"model.torch_dtype must be one of {sorted(TORCH_DTYPES)}, got {name!r}")
    return TORCH_DTYPES[name]


def check_token_ids(bundle: TokenizerBundle) -> TokenIds:
    """§13.6.2 step 1: re-check the ids that M5 ``load_tokenizer`` already checked (§9.6.2).

    ``len(tokenizer) == ids.vocab_size == 32005``, ``new_token_ids == (32000, ..., 32004)`` all
    ``>= base_vocab`` (32000), bos 1, eos 2, pad != eos, and the five named ids in the released order
    ``[PAD], [PH], [HistoryEmb], [CansEmb], [ItemEmb]`` ([코드] U model/model_interface.py:313-319).
    ``[ItemEmb]`` stays in the vocabulary only; a batch that contains it is rejected (§13.6.3).
    """
    ids = getattr(bundle, "ids", None)
    if not isinstance(ids, TokenIds):
        raise ModelError(f"bundle.ids must be the M5 TokenIds, got {type(ids).__name__}")
    problems = []
    try:
        length = len(bundle.tokenizer)
    except TypeError as exc:
        raise ModelError(f"len(bundle.tokenizer) failed: {exc}") from exc
    if not length == ids.vocab_size == VOCAB_SIZE:
        problems.append(f"len(tokenizer) {length}, ids.vocab_size {ids.vocab_size}, expected both {VOCAB_SIZE}")
    if ids.base_vocab != BASE_VOCAB:
        problems.append(f"ids.base_vocab {ids.base_vocab} != {BASE_VOCAB}")
    new_ids = tuple(ids.new_token_ids)
    if new_ids != tuple(range(BASE_VOCAB, VOCAB_SIZE)):
        problems.append(f"new_token_ids {new_ids} != {tuple(range(BASE_VOCAB, VOCAB_SIZE))}")
    if any(t < ids.base_vocab for t in new_ids):
        problems.append(f"new_token_ids {new_ids} must all be >= base_vocab {ids.base_vocab}")
    named = (ids.pad, ids.ph, ids.history_emb, ids.cans_emb, ids.item_emb)
    if named != new_ids:
        problems.append(f"(pad, ph, history_emb, cans_emb, item_emb) = {named} != new_token_ids {new_ids}")
    if (ids.bos, ids.eos) != (1, 2):
        problems.append(f"(bos, eos) = {(ids.bos, ids.eos)} != (1, 2)")
    if ids.pad == ids.eos:
        problems.append(f"pad id {ids.pad} equals eos id {ids.eos}")
    if problems:
        raise ModelError("tokenizer ids do not match the M5 contract (13.6.2 step 1): " + "; ".join(problems))
    return ids


def generation_config_for(mcfg: "ModelConfig", ids: TokenIds) -> GenerationConfig:
    """Greedy ``GenerationConfig`` from the ``model`` section (§13.6.6, §13.6.11).

    [코드] U model/model_interface.py:70-88, L model/model_interface.py:43-57: do_sample False, num_beams 1,
    max/min new tokens, repetition penalty, pad = ``[PAD]``. ``use_cache=True`` is fixed, not a key: 4.28.0
    feeds ``inputs_embeds`` again at every step without a cache (T4.28 models/llama/modeling_llama.py:742-745).
    Passing this object to ``generate`` makes the model folder's ``generation_config.json`` irrelevant
    (T4.28 generation/utils.py:1212-1229).
    """
    if mcfg.gen_strategy != "greedy":
        raise ModelError(f"model.gen_strategy must be greedy, got {mcfg.gen_strategy!r}")
    return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=mcfg.gen_max_new_tokens,
                            min_new_tokens=mcfg.gen_min_new_tokens,
                            repetition_penalty=mcfg.gen_repetition_penalty, pad_token_id=ids.pad,
                            bos_token_id=ids.bos, eos_token_id=ids.eos, use_cache=True)


def check_generation_config(gen_config: GenerationConfig, mcfg: "ModelConfig", ids: TokenIds) -> None:
    """The generation settings the build and ``generate`` rely on (§13.6.2 step 8, §13.6.6)."""
    want = {"do_sample": False, "num_beams": 1, "max_new_tokens": mcfg.gen_max_new_tokens,
            "min_new_tokens": mcfg.gen_min_new_tokens, "repetition_penalty": mcfg.gen_repetition_penalty,
            "pad_token_id": ids.pad, "bos_token_id": ids.bos, "eos_token_id": ids.eos}
    problems = [f"{key} {getattr(gen_config, key, None)!r} != {value!r}" for key, value in want.items()
                if getattr(gen_config, key, None) != value]
    if getattr(gen_config, "use_cache", None) is not True:
        problems.append(f"use_cache {getattr(gen_config, 'use_cache', None)!r} is not True (4.28.0 inputs_embeds "
                        "generation fails at the second step without a cache, 13.6.6)")
    if problems:
        raise ModelError("generation config check failed: " + "; ".join(problems))


# ----------------------------------------------------------------------------
# LLaMA backbone (§13.6.2 steps 2-4, §13.6.9)
# ----------------------------------------------------------------------------
def load_base_llm(path: Path, torch_dtype: torch.dtype) -> LlamaForCausalLM:
    """``LlamaForCausalLM.from_pretrained(path, torch_dtype=..., output_loading_info=True)`` on CPU, checked.

    No ``device_map`` (§13.6.2 step 2). ``unexpected_keys``, ``mismatched_keys`` and ``error_msgs`` must be
    empty and ``missing_keys`` a subset of ``model.layers.*.self_attn.rotary_emb.inv_freq`` (computed at init);
    every parameter has ``torch_dtype`` (D9: the fp16 checkpoint is loaded as fp16, not bf16 as in
    [코드] U model/model_interface.py:318). The config shape is checked by ``check_llama_config``.
    """
    path = Path(path)
    if not path.is_dir():
        raise ModelError(f"LLaMA folder {path} does not exist (LLM_PATH)")
    if torch_dtype not in TORCH_DTYPES.values():
        raise ModelError(f"torch_dtype must be one of {list(TORCH_DTYPES.values())}, got {torch_dtype!r}")
    llm, info = LlamaForCausalLM.from_pretrained(str(path), torch_dtype=torch_dtype, output_loading_info=True)
    problems = []
    if type(llm) is not LlamaForCausalLM:
        problems.append(f"loaded {type(llm).__name__}, not LlamaForCausalLM")
    for key in ("unexpected_keys", "mismatched_keys", "error_msgs"):
        values = list(info.get(key) or [])
        if values:
            problems.append(f"{key} {values[:10]} (expected none)")
    missing = sorted(k for k in (info.get("missing_keys") or []) if not _ALLOWED_MISSING_RE.fullmatch(k))
    if missing:
        problems.append(f"missing_keys {missing[:10]} (only rotary_emb.inv_freq may be missing)")
    if getattr(llm, "hf_device_map", None) is not None:
        problems.append("the model was dispatched with a device_map")
    bad_dtype = sorted({f"{name}: {p.dtype}" for name, p in llm.named_parameters() if p.dtype != torch_dtype})
    if bad_dtype:
        problems.append(f"parameters not in {torch_dtype}: {bad_dtype[:5]}")
    off_cpu = sorted(name for name, p in llm.named_parameters() if p.device.type != "cpu")
    if off_cpu:
        problems.append(f"parameters not on CPU: {off_cpu[:5]}")
    if problems:
        raise ModelError(f"{path}: LLaMA load check failed (13.6.2 step 2): " + "; ".join(problems))
    return llm


def check_llama_config(config: LlamaConfig, *, shape: Mapping[str, int], base_vocab: int) -> None:
    """§13.6.2 step 2 config check: the ``shape`` values, ``vocab_size == base_vocab``,
    ``tie_word_embeddings`` False, ``rope_scaling`` absent or None and ``num_key_value_heads`` (when present)
    equal to the head count. 4.28.0 ``LlamaConfig`` reads neither of the last two keys, so other values would
    silently give another model.
    """
    problems = [f"{key} {getattr(config, key, None)!r} != {value}" for key, value in shape.items()
                if getattr(config, key, None) != value]
    if getattr(config, "vocab_size", None) != base_vocab:
        problems.append(f"vocab_size {getattr(config, 'vocab_size', None)!r} != tokenizer base vocabulary {base_vocab}")
    if getattr(config, "tie_word_embeddings", None) is not False:
        problems.append(f"tie_word_embeddings {getattr(config, 'tie_word_embeddings', None)!r} is not False")
    if getattr(config, "rope_scaling", None) is not None:
        problems.append(f"rope_scaling {config.rope_scaling!r} is not None (not supported by 4.28.0)")
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is not None and kv_heads != getattr(config, "num_attention_heads", None):
        problems.append(f"num_key_value_heads {kv_heads!r} != num_attention_heads "
                        f"{getattr(config, 'num_attention_heads', None)!r} (4.28.0 has no grouped-query attention)")
    if problems:
        raise ModelError("LLaMA config check failed (13.6.2 step 2): " + "; ".join(problems))


def tiny_llama_config(vocab_size: int) -> LlamaConfig:
    """``LlamaConfig`` of the CPU test backbone (§13.6.9): hidden 64, intermediate 128, 2 layers, 4 heads."""
    return LlamaConfig(vocab_size=vocab_size, hidden_size=TINY_LLAMA_SHAPE["hidden_size"],
                       intermediate_size=TINY_LLAMA_SHAPE["intermediate_size"],
                       num_hidden_layers=TINY_LLAMA_SHAPE["num_hidden_layers"],
                       num_attention_heads=TINY_LLAMA_SHAPE["num_attention_heads"],
                       max_position_embeddings=TINY_MAX_POSITIONS, rms_norm_eps=1e-5, initializer_range=0.02,
                       bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)


def build_tiny_llama(vocab_size: int, main_seed: int) -> LlamaForCausalLM:
    """Random CPU fp32 LLaMA of the test shape (§13.6.9), loaded back through ``load_base_llm``.

    The weights are drawn inside ``seeding.torch_seed_scope(main_seed, "init", "tiny_llama")``; the model is
    saved with ``save_pretrained`` to a temporary folder and read again, so the test path passes the same
    load check, resize, freeze and wrapping as Llama-2-7B. CPU only: torch 2.0.0 has no CPU fp16 matmul.
    """
    for name, value in (("vocab_size", vocab_size), ("main_seed", main_seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelError(f"{name} must be an int, got {value!r}")
    config = tiny_llama_config(vocab_size)
    with seeding.torch_seed_scope(main_seed, "init", "tiny_llama"):
        llm = LlamaForCausalLM(config)
    with tempfile.TemporaryDirectory(prefix="ilora_tiny_llama_") as tmp:
        llm.save_pretrained(tmp)
        del llm
        return load_base_llm(Path(tmp), torch.float32)


def resize_new_tokens(llm: LlamaForCausalLM, vocab_size: int, main_seed: int) -> None:
    """§13.6.2 step 3: ``resize_token_embeddings(vocab_size)`` once, inside the ``("init", "new_tokens")`` scope.

    [코드] U model/model_interface.py:319, L model/model_interface.py:207 resize without a seed; here the CPU
    generator is seeded and restored around it, so the new rows (embedding and lm_head rows 32000-32004) are a
    function of the seed. 4.28.0 draws, all on the CPU generator: N(0, 1) for a new fp32 Embedding (discarded),
    N(0, 0.02) in the base dtype (``_init_weights``), the old rows copied back, then the same for a new Linear
    lm_head (T4.28 modeling_utils.py:1467-1484, :1539-1570). The LLM must be on CPU.
    """
    for name, tensor in itertools.chain(llm.named_parameters(), llm.named_buffers()):
        if tensor.device.type != "cpu":
            raise ModelError(f"resize_new_tokens needs the LLM on CPU (13.6.2 step 3); {name} is on {tensor.device}")
    old_rows = llm.get_input_embeddings().weight.shape[0]
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= old_rows:
        raise ModelError(f"vocab_size must be an int > the current {old_rows} rows, got {vocab_size!r}")
    with seeding.torch_seed_scope(main_seed, "init", "new_tokens"):
        llm.resize_token_embeddings(vocab_size)
    rows = (llm.get_input_embeddings().weight.shape[0], llm.get_output_embeddings().weight.shape[0],
            llm.config.vocab_size)
    if rows != (vocab_size,) * 3:
        raise ModelError(f"after resize (embedding rows, lm_head rows, config.vocab_size) = {rows}, "
                         f"expected {vocab_size}")


def parameter_counts(llm: nn.Module, projector: nn.Module, omega_source: Router, *, base_vocab: int,
                     sasrec: Optional[nn.Module] = None) -> dict[str, int]:
    """Parameter numbers of §13.8 #13 (TrainableParams [SPEC §12]); ``ILoRAModel.parameter_report`` uses it.

    ``base`` = every LLaMA parameter that is not a LoRA tensor (after the resize), ``new_token_rows`` =
    the rows added by the resize in the embedding and in the untied lm_head (2 x 5 x hidden),
    ``base_before_resize = base - new_token_rows``, ``lora`` = the LoRA A and B tensors of every
    MoE-LoRA layer, ``projector``, ``gate`` (``Router.trainable_parameters``: 256 for dynamic, 0 otherwise),
    ``trainable = lora + projector + gate``. Works on meta tensors (shapes only).
    """
    lora = lora_named_parameters(llm)
    lora_ids = {id(p) for _, p in lora}
    base = sum(p.numel() for p in llm.parameters() if id(p) not in lora_ids)
    hidden = llm.config.hidden_size
    new_rows = (llm.get_input_embeddings().weight.shape[0] - base_vocab) * hidden
    head = llm.get_output_embeddings()
    if head is not None and not llm.config.tie_word_embeddings:
        new_rows += (head.weight.shape[0] - base_vocab) * hidden
    n_lora = sum(p.numel() for _, p in lora)
    n_projector = sum(p.numel() for p in projector.parameters())
    n_gate = sum(p.numel() for p in omega_source.trainable_parameters())
    n_sasrec = sum(p.numel() for p in sasrec.parameters()) if sasrec is not None else 0
    return {
        "base_before_resize": base - new_rows, "new_token_rows": new_rows, "base": base,
        "lora": n_lora, "projector": n_projector, "gate": n_gate, "trainable": n_lora + n_projector + n_gate,
        "sasrec": n_sasrec, "frozen": base + n_sasrec, "total": base + n_sasrec + n_lora + n_projector + n_gate,
        "lora_tensors": len(lora), "wrapped_modules": sum(1 for m in llm.modules() if isinstance(m, MoELoRALinear)),
    }


# ----------------------------------------------------------------------------
# The model (§13.6.1)
# ----------------------------------------------------------------------------
class ILoRAModel(nn.Module):
    """Llama-2 + MoE-LoRA + SASRec + projector + omega source (§3.1 ``ILoRAModel``). Build it with ``build_model``.

    Submodule names fix the parameter and checkpoint names: ``llm.model.layers.0.self_attn.q_proj.lora_A.weight``,
    ``projector.mlp_proj.0.weight``, ``omega_source.gate.proj.weight`` (§13.6.7). ``gate_ctx`` is the one
    ``GateContext`` shared with every MoE-LoRA layer; ``ids`` the M5 ``TokenIds``; ``n_wrapped`` the number of
    wrapped Linear layers (224, the tiny model 14). ``device`` and ``gen_config`` are read-only.
    """

    llm: LlamaForCausalLM
    sasrec: SASRecEncoder
    projector: nn.Module
    omega_source: Router
    gate_ctx: GateContext
    ids: TokenIds
    n_wrapped: int

    def __init__(self, *, llm: LlamaForCausalLM, sasrec: SASRecEncoder, projector: nn.Module, omega_source: Router,
                 gate_ctx: GateContext, ids: TokenIds, n_wrapped: int, gen_config: GenerationConfig,
                 pad_item: int, device: Union[torch.device, str]) -> None:
        super().__init__()
        self.llm = llm
        self.sasrec = sasrec
        self.projector = projector
        self.omega_source = omega_source
        self.gate_ctx = gate_ctx                  # plain shared object, not a submodule (M8 §12.6.4)
        self.ids = ids
        self.n_wrapped = n_wrapped
        self.num_experts = gate_ctx.num_experts   # K: lora.num_experts, the only definition (§12.6.9)
        self.pad_item = pad_item                  # ItemCatalog.pad_id = 1682 ([코드] U data/movielens_data.py:18)
        self._gen_config = gen_config
        self._device = torch.device(device)
        self._group_names = self._collect_group_names()

    # ------------------------------------------------------------------ read-only attributes
    @property
    def device(self) -> torch.device:
        """The model device (``model.device``); batches are moved here by ``train_forward``/``generate``."""
        return self._device

    @property
    def gen_config(self) -> GenerationConfig:
        """A copy of the greedy ``GenerationConfig`` used by ``generate`` (M11 checks and records it)."""
        return copy.deepcopy(self._gen_config)

    def extra_repr(self) -> str:
        return (f"device={self._device}, n_wrapped={self.n_wrapped}, num_experts={self.num_experts}, "
                f"routing_mode={self.omega_source.routing_mode}")

    # ------------------------------------------------------------------ parameter groups (§13.6.7)
    def _collect_group_names(self) -> dict[str, tuple[str, ...]]:
        names_by_id = {id(p): name for name, p in self.named_parameters()}
        groups: dict[str, tuple[str, ...]] = {
            "projector": tuple(sorted(f"projector.{n}" for n, _ in self.projector.named_parameters())),
        }
        gate = self.omega_source.trainable_parameters()
        if gate:
            groups["gate"] = tuple(sorted(names_by_id[id(p)] for p in gate))
        groups["lora"] = tuple(sorted(f"llm.{n}" for n, _ in lora_named_parameters(self.llm)))
        return groups

    def param_groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """``projector``, ``gate`` (dynamic only) and ``lora``: ``(name, parameter)`` sorted by name (D19).

        The name lists are fixed at build time; M10 makes one optimizer group per key in this order.
        """
        params = dict(self.named_parameters())
        return {group: [(name, params[name]) for name in names] for group, names in self._group_names.items()}

    # ------------------------------------------------------------------ modes (§13.6.7, D12)
    def train(self, mode: bool = True) -> "ILoRAModel":
        """``nn.Module.train`` then SASRec back to eval (D12; M3 overrides ``train`` as well)."""
        super().train(mode)
        self.sasrec.eval()
        return self

    def sasrec_is_eval(self) -> bool:
        """SASRec and every submodule of it have ``training == False`` (M10 checks it after ``train()``)."""
        return not any(module.training for module in self.sasrec.modules())

    # ------------------------------------------------------------------ batch checks (§13.6.3, §13.6.5, §13.6.6)
    def _check_batch(self, batch: Union[TrainBatch, EvalBatch], kind: str) -> int:
        """§13.6.3 step 1 plus the padding/label rules of ``train`` (§13.6.5) or ``eval`` (§13.6.6). Returns B.

        Runs on the tensors where they are (CPU batches from M5) with one synchronisation; a failure names the
        first bad row, its ``sample_id`` and every broken rule of that row.
        """
        is_train = kind == "train"
        cls = TrainBatch if is_train else EvalBatch
        if not isinstance(batch, cls):
            raise ModelError(f"{kind} needs a {cls.__name__}, got {type(batch).__name__}")
        if not isinstance(batch.hybrid, bool):
            raise ModelError(f"{kind} batch: hybrid must be bool, got {batch.hybrid!r}")
        names = ["input_ids", "attention_mask", "hist_ids", "len_seq", "cand_ids"] + (["labels"] if is_train else [])
        tensors = {name: getattr(batch, name) for name in names}
        for name, t in tensors.items():
            if not isinstance(t, Tensor) or t.dtype != torch.int64:
                raise ModelError(f"{kind} batch: {name} must be an int64 tensor, got "
                                 f"{getattr(t, 'dtype', type(t).__name__)}")
        input_ids, mask = batch.input_ids, batch.attention_mask
        if input_ids.dim() != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ModelError(f"{kind} batch: input_ids must be [B, L] with B, L >= 1, got {tuple(input_ids.shape)}")
        b, length = input_ids.shape
        want = {"attention_mask": (b, length), "hist_ids": (b, HIST_LEN), "len_seq": (b,), "cand_ids": (b, NUM_CANDS)}
        if is_train:
            want["labels"] = (b, length)
        for name, shape in want.items():
            if tuple(tensors[name].shape) != shape:
                raise ModelError(f"{kind} batch: {name} must be {list(shape)}, got {list(tensors[name].shape)}")
        devices = sorted({str(t.device) for t in tensors.values()})
        if len(devices) != 1:
            raise ModelError(f"{kind} batch tensors are on several devices: {devices}")
        sample_ids = list(batch.sample_ids)
        if len(sample_ids) != b or not all(isinstance(s, str) for s in sample_ids):
            raise ModelError(f"{kind} batch: sample_ids must be {b} str, got {sample_ids!r}")

        checks = self._row_checks(batch, is_train)
        ok = torch.stack([value for value, _ in checks.values()]).all(dim=0)
        if bool(ok.all()):                        # the one synchronisation of the normal path
            return b
        row = int((~ok).nonzero()[0, 0])
        reasons = [f"{name} ({detail(row)})" for name, (value, detail) in checks.items() if not bool(value[row])]
        raise ModelError(f"{kind} batch row {row} (sample_id {sample_ids[row]!r}, "
                         f"{'hybrid' if batch.hybrid else 'text-only'}): " + "; ".join(reasons))

    def _row_checks(self, batch: Union[TrainBatch, EvalBatch], is_train: bool) -> dict:
        """``{rule: (ok [B] bool, detail(row) -> str)}`` for ``_check_batch``."""
        ids = self.ids
        input_ids, mask = batch.input_ids, batch.attention_mask
        len_seq, hist, cand = batch.len_seq, batch.hist_ids, batch.cand_ids
        length = input_ids.shape[1]
        dev = input_ids.device
        real = mask == 1
        n_real = real.sum(dim=1)
        positions = torch.arange(length, device=dev)
        if is_train:   # §13.6.5 step 2: right padding, real tokens at 0..n-1 (4.28.0 default arange position ids)
            layout = positions[None, :] < n_real[:, None]
            first = torch.zeros_like(n_real)
            side = "right padding (attention_mask 1...1 0...0)"
        else:          # §13.6.6 step 2, D22: left padding, last column real
            layout = positions[None, :] >= (length - n_real)[:, None]
            first = (length - n_real).clamp(max=length - 1)
            side = "left padding (attention_mask 0...0 1...1)"
        first_token = input_ids.gather(1, first[:, None]).squeeze(1)
        is_h, is_c = input_ids == ids.history_emb, input_ids == ids.cans_emb
        is_ph, is_item = input_ids == ids.ph, input_ids == ids.item_emb
        n_h, n_c, n_ph, n_item = (x.sum(dim=1) for x in (is_h, is_c, is_ph, is_item))
        n_cands = cand.shape[1]
        zeros = torch.zeros_like(len_seq)
        if batch.hybrid:   # [LLaRA 간접 §4.2 p.5-6]: title + behaviour token for every history item and candidate
            want_h, want_c, want_ph = len_seq, zeros + n_cands, zeros
        else:              # text-only: "title [PH]" for both lists
            want_h, want_c, want_ph = zeros, zeros, len_seq + n_cands
        special_at_pad = ((is_h | is_c | is_ph | is_item) & ~real).sum(dim=1)
        hist_real = torch.arange(hist.shape[1], device=dev)[None, :] < len_seq[:, None]
        hist_ok = torch.where(hist_real, (hist >= 0) & (hist < self.pad_item), hist == self.pad_item).all(dim=1)
        in_vocab = (input_ids >= 0) & (input_ids < ids.vocab_size)

        def first_false(values: Tensor) -> int:
            return int((~values).nonzero()[0, 0])

        checks = {
            "attention_mask is 0/1 with at least one 1": (
                ((mask == 0) | (mask == 1)).all(dim=1) & (n_real >= 1),
                lambda r: f"{int(n_real[r])} ones, values {sorted(set(mask[r].tolist()))}"),
            side: ((real == layout).all(dim=1),
                   lambda r: f"{int(n_real[r])} real tokens, first mask values {mask[r, :4].tolist()}, "
                             f"last {mask[r, -4:].tolist()}"),
            "first real token is BOS": (first_token == ids.bos,
                                        lambda r: f"id {int(first_token[r])}, BOS is {ids.bos}"),
            "token ids inside the vocabulary": (in_vocab.all(dim=1),
                                                lambda r: f"id {int(input_ids[r, first_false(in_vocab[r])])}"),
            "placeholder counts": (
                (n_h == want_h) & (n_c == want_c) & (n_ph == want_ph),
                lambda r: f"[HistoryEmb] {int(n_h[r])} (want {int(want_h[r])}), [CansEmb] {int(n_c[r])} "
                          f"(want {int(want_c[r])}), [PH] {int(n_ph[r])} (want {int(want_ph[r])})"),
            "no [ItemEmb] in the input": (n_item == 0, lambda r: f"{int(n_item[r])} [ItemEmb] tokens"),
            "placeholders only where attention_mask = 1": (
                special_at_pad == 0, lambda r: f"{int(special_at_pad[r])} placeholders at padded positions"),
            f"len_seq in [{MIN_LEN_SEQ}, {hist.shape[1]}]": (
                (len_seq >= MIN_LEN_SEQ) & (len_seq <= hist.shape[1]), lambda r: f"len_seq {int(len_seq[r])}"),
            f"hist_ids: first len_seq ids in 0-{self.pad_item - 1}, the rest {self.pad_item}": (
                hist_ok, lambda r: f"hist_ids {hist[r].tolist()}, len_seq {int(len_seq[r])}"),
            f"cand_ids in 0-{self.pad_item - 1}": (
                ((cand >= 0) & (cand < self.pad_item)).all(dim=1), lambda r: f"cand_ids {cand[r].tolist()}"),
        }
        if is_train:
            labels = batch.labels
            has_label = labels != IGNORE_INDEX
            label_ok = ~has_label | (real & (labels == input_ids))

            def label_detail(r: int) -> str:
                p = first_false(label_ok[r])
                return (f"position {p}: label {int(labels[r, p])}, input id {int(input_ids[r, p])}, "
                        f"attention_mask {int(mask[r, p])}")

            checks[f"labels[:, 0] == {IGNORE_INDEX}"] = (labels[:, 0] == IGNORE_INDEX,
                                                         lambda r: f"labels[{r}, 0] = {int(labels[r, 0])}")
            checks["labels equal input_ids where set, inside the attention mask"] = (label_ok.all(dim=1), label_detail)
            checks["at least one label token"] = (
                has_label.sum(dim=1) >= 1, lambda r: "no answer token has a label (the HF loss would be NaN, 13.6.10)")
        return checks

    # ------------------------------------------------------------------ z, omega and inputs_embeds
    def _z_omega(self, batch: Union[TrainBatch, EvalBatch]) -> tuple[Tensor, Tensor]:
        """§13.6.4: ``z = sasrec(hist_ids, len_seq)`` and ``omega = omega_source(z, sample_ids)``, autocast off.

        z = SR-EMB(i<n) [논문 p.5 식 10], omega = Softmax(W_g z) [논문 p.6 식 11] (or one-hot / ones). This is the
        only place gate values are produced [SPEC §4]; omega keeps its graph for the dynamic gate.
        """
        b = batch.hist_ids.shape[0]
        with torch.autocast(device_type=self._device.type, enabled=False):          # D23
            z = self.sasrec(batch.hist_ids, batch.len_seq)
            omega = self.omega_source(z, batch.sample_ids)
        if z.dtype != torch.float32 or z.requires_grad or tuple(z.shape) != (b, self.sasrec.hidden_size):
            raise ModelError(f"z must be [{b}, {self.sasrec.hidden_size}] float32 without grad (M3), got "
                             f"{list(z.shape)} {z.dtype} requires_grad={z.requires_grad}")
        if tuple(omega.shape) != (b, self.num_experts):
            raise ModelError(f"omega must be [{b}, {self.num_experts}] (lora.num_experts), got {list(omega.shape)}")
        return z, omega

    def _embed_inputs(self, batch: Union[TrainBatch, EvalBatch]) -> Tensor:
        """§13.6.3 steps 2-5 for a batch on the model device that passed ``_check_batch``: ``inputs_embeds``.

        Text-only: the token embeddings as they are (the projector is not called, 식 7). Hybrid: the k-th
        ``[HistoryEmb]`` of row b gets the behaviour token of its k-th history item and the k-th ``[CansEmb]``
        its k-th candidate (the rule of [코드] U model/model_interface.py:453-461 without the zip truncation),
        Proj(SR-EMB(i)) [논문 p.4 식 5-6], written with one out-of-place ``index_put``.
        """
        emb = self.llm.get_input_embeddings()(batch.input_ids)                    # [B, L, H], base dtype, no grad
        if not batch.hybrid:
            return emb
        ids = self.ids
        hist_pos = torch.arange(batch.hist_ids.shape[1], device=batch.hist_ids.device)
        ids_h = batch.hist_ids[hist_pos[None, :] < batch.len_seq[:, None]]        # row-major, oldest first
        ids_c = batch.cand_ids.reshape(-1)                                         # row-major, M2 order
        pos_h = (batch.input_ids == ids.history_emb).nonzero()                    # row-major (b, t)
        pos_c = (batch.input_ids == ids.cans_emb).nonzero()
        if pos_h.shape[0] != ids_h.shape[0] or pos_c.shape[0] != ids_c.shape[0]:
            raise ModelError(f"placeholder counts do not match the items: [HistoryEmb] {pos_h.shape[0]} for "
                             f"{ids_h.shape[0]} history items, [CansEmb] {pos_c.shape[0]} for {ids_c.shape[0]} "
                             "candidates (run _check_batch first)")
        rep = self.sasrec.embed_items(torch.cat([ids_h, ids_c]))                  # [n, 64] fp32, no_grad, fp32 region
        tok = self.projector(rep)                                                  # [n, H] under the caller's autocast
        finite = torch.isfinite(tok).all(dim=1)
        if not bool(finite.all()):                                                 # M6 §10.6.3: checked here once
            bad = int((~finite).nonzero()[0, 0])
            pos = torch.cat([pos_h, pos_c])[bad].tolist()
            raise ModelError(f"behaviour token {bad} (batch row {pos[0]}, position {pos[1]}, sample_id "
                             f"{batch.sample_ids[pos[0]]!r}) is not finite ({tok.dtype})")
        pos = torch.cat([pos_h, pos_c])
        return emb.index_put((pos[:, 0], pos[:, 1]), tok.to(emb.dtype))

    # ------------------------------------------------------------------ training (§13.6.5)
    def train_forward(self, batch: TrainBatch) -> TrainForward:
        """Loss of one right-padded ``TrainBatch``: mean CE of the answer tokens [논문 p.4 식 7, p.5 식 8].

        Runs in the caller's autocast (M10, M0 precision tools). The LLaMA body gets ``inputs_embeds`` and
        ``attention_mask`` with ``use_cache=False`` ([코드] U model/model_interface.py:49-57); no position_ids,
        since right padding keeps real tokens at 0..n-1. ``[결정]`` the loss uses only the label positions:
        ``lm_head`` on those hidden states, fp32 cross entropy, mean over the label tokens (the HF loss of T4.28
        models/llama/modeling_llama.py:703-713 without the [B, L, V] logits and without its silent NaN).
        """
        self._check_batch(batch, "train")
        n_label = int((batch.labels[:, 1:] != IGNORE_INDEX).sum())
        batch = batch.to(self._device)
        z, omega = self._z_omega(batch)
        emb = self._embed_inputs(batch)
        with self.gate_ctx.use(omega) as use:
            hidden = self.llm.model(inputs_embeds=emb, attention_mask=batch.attention_mask, use_cache=False,
                                    return_dict=True).last_hidden_state                # after the last RMSNorm
        if use.calls != self.n_wrapped:
            raise ModelError(f"MoE-LoRA layers read omega {use.calls} times, expected {self.n_wrapped} (one forward)")
        labels = batch.labels[:, 1:]
        sel = labels != IGNORE_INDEX
        logits = self.llm.lm_head(hidden[:, :-1][sel])                                  # [N, V], autocast dtype
        loss = F.cross_entropy(logits.float(), labels[sel], reduction="mean")
        return TrainForward(loss=loss, n_label_tokens=n_label, omega=omega.detach(), z=z)

    # ------------------------------------------------------------------ generation (§13.6.6)
    @torch.no_grad()
    def generate(self, batch: EvalBatch) -> GenerationOutput:
        """Greedy new tokens of a left-padded ``EvalBatch`` with the stock 4.28.0 ``generate`` (D13, D22).

        Only ``inputs_embeds``, ``attention_mask`` and the ``GenerationConfig`` are passed: 4.28.0 starts from a
        ``[BOS]`` placeholder [B, 1], builds position_ids from the mask (cumsum - 1, T4.28
        models/llama/modeling_llama.py:733-739) and uses the embeddings at the first step only. The same omega
        holds for every step inside one ``GateContext.use`` (식 12). Eval mode only (no dropout, §1.3).
        """
        if self.training:
            raise ModelError("generate runs in eval mode only (call model.eval(); 13.6.6 step 1)")
        if self._gen_config.use_cache is not True:
            raise ModelError("gen_config.use_cache must be True (13.6.6)")
        b = self._check_batch(batch, "eval")
        batch = batch.to(self._device)
        z, omega = self._z_omega(batch)
        emb = self._embed_inputs(batch)
        with self.gate_ctx.use(omega) as use:
            seq = self.llm.generate(inputs_embeds=emb, attention_mask=batch.attention_mask,
                                    generation_config=self._gen_config)
        max_new = self._gen_config.max_new_tokens
        if seq.dim() != 2 or seq.shape[0] != b:
            raise ModelError(f"generate returned shape {list(seq.shape)} for a batch of {b}")
        steps = seq.shape[1] - 1
        if not 1 <= steps <= max_new:
            raise ModelError(f"generate returned {steps} new tokens, expected 1..{max_new}")
        if not bool((seq[:, 0] == self.ids.bos).all()):
            raise ModelError(f"first returned column is not the BOS placeholder {self.ids.bos}: {seq[:, 0].tolist()}")
        if use.calls != self.n_wrapped * steps:
            raise ModelError(f"MoE-LoRA layers read omega {use.calls} times for {steps} steps, expected "
                             f"{self.n_wrapped} x {steps}")
        return GenerationOutput(new_token_ids=seq[:, 1:].to("cpu", copy=True).contiguous(),
                                omega=omega.detach().to("cpu", copy=True), z=z.to("cpu", copy=True))

    # ------------------------------------------------------------------ checkpoint state (§13.6.8)
    def trainable_state(self) -> dict[str, Tensor]:
        """Every ``param_groups`` name -> CPU contiguous float32 copy (no ``requires_grad`` filter, M10 §14.6.6)."""
        return {name: p.detach().to("cpu", copy=True).contiguous()
                for items in self.param_groups().values() for name, p in items}

    def load_trainable_state(self, state: Mapping[str, Tensor], strict: bool = True) -> None:
        """Copy ``state`` into the trainable parameters (§13.6.8).

        ``strict``: the key set must equal the ``param_groups`` names (missing or extra keys are errors; the
        ``generated_frozen.*`` rows are compared by M10 ``load_model_state`` and are not accepted here). Shapes
        and dtypes must match. Nothing is copied unless every check passes.
        """
        if not isinstance(state, Mapping):
            raise ModelError(f"state must be a mapping name -> tensor, got {type(state).__name__}")
        params = {name: p for items in self.param_groups().values() for name, p in items}
        missing = sorted(set(params) - set(state))
        extra = sorted(set(state) - set(params))
        problems = []
        if strict and missing:
            problems.append(f"missing keys {missing[:10]} ({len(missing)})")
        if strict and extra:
            hint = (" (generated_frozen.* rows are compared by M10 load_model_state, not loaded)"
                    if any(k.startswith(GENERATED_FROZEN_PREFIX) for k in extra) else "")
            problems.append(f"unexpected keys {extra[:10]} ({len(extra)}){hint}")
        names = [name for name in params if name in state]
        for name in names:
            value, param = state[name], params[name]
            if not isinstance(value, Tensor):
                problems.append(f"{name}: not a tensor ({type(value).__name__})")
            elif tuple(value.shape) != tuple(param.shape) or value.dtype != param.dtype:
                problems.append(f"{name}: {list(value.shape)} {value.dtype} != {list(param.shape)} {param.dtype}")
        if problems:
            raise ModelError("load_trainable_state: " + "; ".join(problems))
        with torch.no_grad():
            for name in names:
                params[name].copy_(state[name])

    def generated_frozen_state(self) -> dict[str, Tensor]:
        """The five new token rows of the input embedding and lm_head (base dtype, CPU) and their ids (§13.6.8).

        Not trained, but made by our seeded resize, so the checkpoint keeps them and M10 compares them bit by bit
        with a newly built model (§1.3).
        """
        token_ids = torch.tensor(self.ids.new_token_ids, dtype=torch.int64)
        embed = self.llm.get_input_embeddings().weight.detach()
        head = self.llm.get_output_embeddings().weight.detach()
        return {
            "generated_frozen.token_ids": token_ids,
            "generated_frozen.embed_rows": embed[token_ids.to(embed.device)].to("cpu", copy=True).contiguous(),
            "generated_frozen.lm_head_rows": head[token_ids.to(head.device)].to("cpu", copy=True).contiguous(),
        }

    def parameter_report(self) -> dict[str, int]:
        """``parameter_counts`` of this model (§13.8 #13, TrainableParams [SPEC §12])."""
        return parameter_counts(self.llm, self.projector, self.omega_source, base_vocab=self.ids.base_vocab,
                                sasrec=self.sasrec)

    # ------------------------------------------------------------------ build checks (§13.6.2 step 8)
    def _post_build_check(self, base_dtype: torch.dtype, mcfg: "ModelConfig") -> None:
        """Trainable parameter list, dtypes, devices, gradient checkpointing off, SASRec eval, generation config."""
        problems = []
        groups = self.param_groups()
        dynamic = self.omega_source.routing_mode == "dynamic"
        want_keys = [g for g in PARAM_GROUPS if dynamic or g != "gate"]
        if list(groups) != want_keys:
            problems.append(f"param_groups keys {list(groups)} != {want_keys}")
        sizes = {g: len(items) for g, items in groups.items()}
        want_sizes = {"projector": 4, "lora": 2 * self.n_wrapped, **({"gate": 1} if dynamic else {})}
        if sizes != want_sizes:
            problems.append(f"group tensor counts {sizes} != {want_sizes}")
        grouped = [name for items in groups.values() for name, _ in items]
        if len(set(grouped)) != len(grouped):
            problems.append("a parameter is in more than one group")
        params = dict(self.named_parameters())
        requires_grad = sorted(name for name, p in params.items() if p.requires_grad)
        if requires_grad != sorted(grouped):
            outside = sorted(set(requires_grad) - set(grouped))
            frozen = sorted(set(grouped) - set(requires_grad))
            problems.append(f"requires_grad parameters outside the groups {outside[:10]}, frozen group "
                            f"parameters {frozen[:10]} (13.6.7)")
        not_fp32 = [name for name in grouped if params[name].dtype != torch.float32]
        if not_fp32:
            problems.append(f"trainable parameters not float32 (D9): {not_fp32[:5]}")
        lora_ids = {id(p) for _, p in lora_named_parameters(self.llm)}
        base_bad = sorted(f"{name}: {p.dtype}" for name, p in self.llm.named_parameters()
                          if id(p) not in lora_ids and p.dtype != base_dtype)
        if base_bad:
            problems.append(f"base parameters not {base_dtype}: {base_bad[:5]}")
        sasrec_bad = sorted(name for name, p in self.sasrec.named_parameters() if p.dtype != torch.float32)
        if sasrec_bad:
            problems.append(f"SASRec parameters not float32 (D23): {sasrec_bad[:5]}")
        off_device = sorted(name for name, t in itertools.chain(self.named_parameters(), self.named_buffers())
                            if t.device != self._device)
        if off_device:
            problems.append(f"tensors not on {self._device}: {off_device[:5]}")
        if self.llm.is_gradient_checkpointing:
            problems.append("gradient checkpointing is on (the recomputation would run without omega, 13.6.5)")
        if not self.sasrec_is_eval():
            problems.append("SASRec is not in eval mode (D12)")
        if problems:
            raise ModelError("build check failed (13.6.2 step 8): " + "; ".join(problems))
        check_generation_config(self._gen_config, mcfg, self.ids)


# ----------------------------------------------------------------------------
# build_model (§13.6.2, fixed order)
# ----------------------------------------------------------------------------
def build_model(cfg: "Config", bundle: TokenizerBundle, catalog: "ItemCatalog", *, llm_path: Optional[Path],
                hard_gate: Optional[HardGate] = None) -> ILoRAModel:
    """Build the model in the order of §13.6.2 and return it in eval mode on ``model.device``.

    1. tokenizer ids re-checked (``check_token_ids``); 2. LLaMA: ``llama2_7b`` from ``llm_path`` (``LLM_PATH``,
    fp16) or ``tiny_random`` (``llm_path`` None, CPU fp32), load info and config checked; 3. the five new token
    rows in the ``("init", "new_tokens")`` scope; 4. every LLaMA parameter frozen (embed_tokens, lm_head and norm
    included, 식 3 "W remains frozen"); 5. MoE-LoRA on layers x targets (M8, scope ``("init", "lora")``);
    6. SASRec (M3), projector (M6, ``("init", "projector")``) and ``Router`` (M7, ``("init", "gate")`` for
    dynamic; ``hard_gate`` is the M15 ``ClusterHardGate`` for cluster_hard and None otherwise); 7. ``.to(device)``;
    8. post checks. Every initial value comes from its own seed scope, so the order of 5-6 does not change values.
    """
    seed = cfg.run.seed
    mcfg = cfg.model
    ids = check_token_ids(bundle)                                                       # 1.
    base_dtype = torch_dtype_of(mcfg.torch_dtype)
    device = torch.device(mcfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():                         # 7. (checked before loading)
        raise ModelError(f"model.device {mcfg.device} needs CUDA, which is not available")
    started = time.perf_counter()
    if mcfg.backbone == "llama2_7b":                                                    # 2.
        if llm_path is None:
            raise ModelError("model.backbone llama2_7b needs llm_path (LLM_PATH, 4.2)")
        llm = load_base_llm(Path(llm_path), base_dtype)
        shape = LLAMA2_7B_SHAPE
    elif mcfg.backbone == "tiny_random":
        if llm_path is not None:
            raise ModelError(f"model.backbone tiny_random takes llm_path None, got {llm_path}")
        if (mcfg.device, base_dtype) != ("cpu", torch.float32):
            raise ModelError(f"tiny_random runs on cpu in float32 only (13.6.9), got {mcfg.device} {base_dtype}")
        llm = build_tiny_llama(ids.base_vocab, seed)
        shape = TINY_LLAMA_SHAPE
    else:
        raise ModelError(f"model.backbone must be one of {BACKBONES}, got {mcfg.backbone!r}")
    check_llama_config(llm.config, shape=shape, base_vocab=ids.base_vocab)
    log.info("LLaMA %s loaded on CPU as %s in %.1f s (%s)", mcfg.backbone, base_dtype, time.perf_counter() - started,
             llm_path if llm_path is not None else "random tiny_llama")
    resize_new_tokens(llm, len(bundle.tokenizer), seed)                                 # 3.
    llm.requires_grad_(False)                                                           # 4. after the resize
    gate_ctx = GateContext(cfg.lora.num_experts)                                        # 5.
    wrapped = attach_moe_lora(llm, cfg.lora, gate_ctx, seed)
    expected = llm.config.num_hidden_layers * len(cfg.lora.target_modules)
    if len(wrapped) != expected:
        raise ModelError(f"wrapped {len(wrapped)} modules, expected layers x targets = {expected}")
    sasrec = SASRecEncoder.from_config(cfg, catalog, history_len=HIST_LEN)              # 6.
    projector = build_projector(sasrec.hidden_size, llm.config.hidden_size, seed)
    router = Router(cfg.router, num_experts=cfg.lora.num_experts, z_dim=sasrec.hidden_size, main_seed=seed,
                    hard_gate=hard_gate)
    model = ILoRAModel(llm=llm, sasrec=sasrec, projector=projector, omega_source=router, gate_ctx=gate_ctx,
                       ids=ids, n_wrapped=len(wrapped), gen_config=generation_config_for(mcfg, ids),
                       pad_item=catalog.pad_id, device=device)
    model.to(device)                                                                    # 7. dtypes kept
    model.eval()
    model._post_build_check(base_dtype, mcfg)                                           # 8.
    report = model.parameter_report()
    log.info("model built on %s in %.1f s: routing %s, K %d, %d MoE-LoRA modules, trainable %d (lora %d, "
             "projector %d, gate %d)", device, time.perf_counter() - started, router.routing_mode,
             cfg.lora.num_experts, model.n_wrapped, report["trainable"], report["lora"], report["projector"],
             report["gate"])
    return model


__all__ = [
    "ModelError", "BACKBONES", "TORCH_DTYPES", "PARAM_GROUPS", "OMEGA_SOURCE_PREFIX", "GENERATED_FROZEN_PREFIX",
    "GENERATED_FROZEN_KEYS", "LLAMA2_7B_SHAPE", "TINY_LLAMA_SHAPE", "TINY_MAX_POSITIONS", "TrainForward",
    "GenerationOutput", "SeqEncoder", "OmegaSource", "torch_dtype_of", "check_token_ids", "generation_config_for",
    "check_generation_config", "load_base_llm", "check_llama_config", "tiny_llama_config", "build_tiny_llama",
    "resize_new_tokens", "parameter_counts", "ILoRAModel", "build_model",
]
