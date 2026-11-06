"""M3 SASRec 인코더 (DESIGN.md §7; §3.1 ``z``·아이템 표현; D12, D23).

The pretrained SASRec of the released code (``rec_model/movielens.pt``, git blob
``41c057dc``, the same file in ``U`` and ``L``) stays frozen and gives

- ``z = encoder(hist_ids, len_seq)``: the history representation of 식 10,
  ``[B, 64] float32`` (gate input M7, records M12, clusters M15), and
- ``encoder.embed_items(ids)``: item representations of 식 5, ``[..., 64] float32``
  (projector input M6).

The weights come from the plain state file that ``rebuild/tools/convert_sasrec.py``
writes once (§7.6.1). It is read with ``torch.load(weights_only=True)`` and
checked with a content digest (§7.6.2), so no code runs at load time.
``SASRecEncoder`` re-implements only what ``cacul_h`` and ``cacu_x`` compute
(``[코드] U recommender/A_SASRec_final_bce_llm.py:202-222``, attention and FFN
of ``SASRecModules_ori.py`` blob ``3ac3c975``) with the same operations in the
same order, so that on CPU it equals the original bit for bit at the same batch
split and thread count (§7.6.3, §7.6.5). It has no dropout module and is always
in eval mode (D12), its parameters never require gradients, and it computes in
fp32 outside autocast with CUDA TF32 off (D23, ``strict_fp32``). Every check
raises at once with the offending values (§7.6.6).
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Iterator, Mapping, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from . import run
from .data import HISTORY_LEN, PAD_ID, pad_histories

if TYPE_CHECKING:
    from .data import ItemCatalog
    from .sections import Config

# ----------------------------------------------------------------------------
# File formats and constants (§7.6.1-§7.6.3)
# ----------------------------------------------------------------------------
SASREC_STATE_FORMAT = "ilora-rebuild-sasrec-v1"          # §7.6.2 state file "format"
SASREC_REFERENCE_FORMAT = "ilora-rebuild-sasrec-ref-v1"   # §7.6.1 step 12 reference file "format"
EXPORT_CHUNK = 4096      # 기준 계산과 같은 묶음 크기 (tools/convert_sasrec.py REF_CHUNK)
EXPORT_THREADS = 1       # CPU export의 torch 스레드 수 (tools/convert_sasrec.py REF_THREADS)
CUDA_REFERENCE_ATOL = 1e-4   # §7.8 #12: CUDA z against the CPU reference; CPU must be bitwise (§7.8 #5)

STATE_FILE_KEYS = ("format", "arch", "state_dict", "dropped_keys", "source", "digest")          # §7.6.2
ARCH_KEYS = ("hidden_size", "item_num", "state_size", "num_heads", "layernorm_eps")             # §7.6.2
REFERENCE_FILE_KEYS = ("format", "sample_ids", "hist_ids", "len_seq", "z", "item_ids", "item_emb",
                       "chunk", "num_threads", "state_digest")                                 # §7.6.1 step 12

# Mask fill of the original attention, kept as the same Python int expression so that the
# multiplication is the same operation (float32 value -4294967296.0, -inf in fp16, §7.5).
_MASK_FILL = -2 ** 32 + 1    # [코드] U SASRecModules_ori.py:64, :71
_AUTOCAST_DEVICE_TYPES = ("cpu", "cuda")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_HEX40_RE = re.compile(r"[0-9a-f]{40}")
_SAMPLE_ID_RE = re.compile(r"(train|val|test):(0|[1-9][0-9]*)")

DeviceLike = Union[torch.device, str]


class SASRecError(RuntimeError):
    """State file, reference file or encoder construction does not match the design (§7.6.6)."""


def _is_int(value: object) -> bool:
    """Python int, bool excluded (state file values are written as plain Python numbers, §7.6.1 step 12)."""
    return type(value) is int


def _check_hex(name: str, value: object, pattern: "re.Pattern[str]") -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise SASRecError(f"{name} must be a lowercase hex digest ({pattern.pattern}), got {value!r}")
    return value


def _as_device(device: DeviceLike) -> torch.device:
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type not in _AUTOCAST_DEVICE_TYPES:
        raise ValueError(f"SASRec runs on cpu or cuda devices, got {dev}")
    return dev


def _load_weights_only(path: Path) -> Any:
    """``torch.load(path, map_location="cpu", weights_only=True)``; read errors are re-raised with the path.

    torch 2.0.0's own loader touches ``TypedStorage`` and warns that it is deprecated; only that
    message is silenced here.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated", category=UserWarning)
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with the path, never replaced by a default
        raise SASRecError(f"{path}: cannot be read with torch.load(weights_only=True): "
                          f"{type(exc).__name__}: {exc}") from exc


# ----------------------------------------------------------------------------
# Architecture, state layout and content digest (§7.6.2)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SASRecArch:
    """Sizes read from the converted state file (never config keys, §7.6.7 [SPEC §4]).

    paper_spec weights: hidden 64, item_num 1682 (padding row 1682), state_size 10, 1 head,
    LayerNorm eps 1e-5 (pickle inspection §7.4).
    """
    hidden_size: int
    item_num: int
    state_size: int
    num_heads: int
    layernorm_eps: float

    def __post_init__(self) -> None:
        for name in ("hidden_size", "item_num", "state_size", "num_heads"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise SASRecError(f"SASRecArch.{name} must be a Python int >= 1, got {value!r} "
                                  f"({type(value).__name__})")
        eps = self.layernorm_eps
        if type(eps) is not float or not math.isfinite(eps) or eps <= 0.0:
            raise SASRecError(f"SASRecArch.layernorm_eps must be a positive Python float, got {eps!r}")
        if self.hidden_size % self.num_heads:
            raise SASRecError(f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads} "
                              "([코드] U SASRecModules_ori.py:31)")

    def as_dict(self) -> dict[str, Any]:
        """Plain dict in ``ARCH_KEYS`` order (the state file ``arch`` entry)."""
        return asdict(self)


def expected_state_shapes(arch: SASRecArch) -> dict[str, tuple[int, ...]]:
    """Name -> shape of the 20 state tensors (§7.6.2): the original keys without ``s_fc``.

    Names are those of ``U recommender/A_SASRec_final_bce_llm.py:155-169`` and
    ``SASRecModules_ori.py:10-12, :33-35``; ``Conv1d(k=1)`` weights keep their ``[h, h, 1]`` shape.
    """
    h = arch.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "item_embeddings.weight": (arch.item_num + 1, h),      # row item_num = padding
        "positional_embeddings.weight": (arch.state_size, h),
    }
    for norm in ("ln_1", "ln_2", "ln_3"):
        shapes[f"{norm}.weight"] = (h,)
        shapes[f"{norm}.bias"] = (h,)
    for proj in ("q", "k", "v"):
        shapes[f"mh_attn.linear_{proj}.weight"] = (h, h)
        shapes[f"mh_attn.linear_{proj}.bias"] = (h,)
    for conv in ("w_1", "w_2"):
        shapes[f"feed_forward.{conv}.weight"] = (h, h, 1)
        shapes[f"feed_forward.{conv}.bias"] = (h,)
    shapes["feed_forward.layer_norm.weight"] = (h,)
    shapes["feed_forward.layer_norm.bias"] = (h,)
    return shapes


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Content digest of a state dict (§7.6.2).

    Keys in sorted order; per key ``f"{key}|{dtype}|{shape joined by ','}\\n"`` (dtype without
    ``torch.``, e.g. ``float32``) and then the raw tensor bytes (CPU, contiguous, little-endian)
    go into one SHA-256. It hashes content, not file bytes, so file names and the save format do
    not matter. The same function serves the tool, ``load_sasrec_state`` and the epoch-end
    recomputation of M10 (``state_digest(model.sasrec.state_dict())``, §14.6.4).
    """
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str):
            raise TypeError(f"state keys must be str, got {key!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state[{key!r}] must be a torch.Tensor, got {type(tensor).__name__}")
        t = tensor.detach().to("cpu").contiguous()
        dtype = str(t.dtype).replace("torch.", "", 1)
        shape = ",".join(str(s) for s in t.shape)
        h.update(f"{key}|{dtype}|{shape}\n".encode("utf-8"))
        array = t.numpy()
        h.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
    return h.hexdigest()


def load_sasrec_state(path: Path, expected_digest: str) -> tuple[SASRecArch, dict[str, torch.Tensor], str]:
    """Read the converted state file (§7.6.2) and return ``(arch, state, digest)``.

    ``torch.load(..., weights_only=True)`` on CPU, then strict checks (§7.6.6): exact file keys,
    ``format``, ``arch`` of Python ints and a float, the 20 tensor names with their shapes,
    float32, finite values, and the recomputed digest equal to both the file's ``digest`` and
    ``expected_digest`` (``sasrec.state_digest``). Any difference raises ``SASRecError``.
    """
    path = Path(path)
    _check_hex("sasrec.state_digest (expected digest)", expected_digest, _HEX64_RE)
    if not path.is_file():
        raise SASRecError(f"converted SASRec state file not found: {path} "
                          "(run rebuild/tools/convert_sasrec.py once, DESIGN 7.6.1)")
    obj = _load_weights_only(path)
    if not isinstance(obj, dict):
        raise SASRecError(f"{path}: expected a dict, got {type(obj).__name__}")
    if set(obj) != set(STATE_FILE_KEYS):
        raise SASRecError(f"{path}: keys {sorted(obj)} != {sorted(STATE_FILE_KEYS)}")
    if obj["format"] != SASREC_STATE_FORMAT:
        raise SASRecError(f"{path}: format {obj['format']!r} != {SASREC_STATE_FORMAT!r}")

    raw_arch = obj["arch"]
    if not isinstance(raw_arch, dict) or set(raw_arch) != set(ARCH_KEYS):
        raise SASRecError(f"{path}: arch must be a dict with keys {list(ARCH_KEYS)}, got {raw_arch!r}")
    arch = SASRecArch(**{key: raw_arch[key] for key in ARCH_KEYS})

    raw_state = obj["state_dict"]
    if not isinstance(raw_state, dict):
        raise SASRecError(f"{path}: state_dict must be a dict, got {type(raw_state).__name__}")
    expected = expected_state_shapes(arch)
    if set(raw_state) != set(expected):
        raise SASRecError(f"{path}: state keys differ from the design (7.6.2): missing "
                          f"{sorted(set(expected) - set(raw_state))}, unexpected {sorted(set(raw_state) - set(expected))}")
    state: dict[str, torch.Tensor] = {}
    for key, shape in expected.items():
        tensor = raw_state[key]
        if not isinstance(tensor, torch.Tensor):
            raise SASRecError(f"{path}: state_dict[{key!r}] is {type(tensor).__name__}, not a tensor")
        if tensor.dtype != torch.float32:
            raise SASRecError(f"{path}: state_dict[{key!r}] is {tensor.dtype}, expected torch.float32")
        if tuple(tensor.shape) != shape:
            raise SASRecError(f"{path}: state_dict[{key!r}] has shape {tuple(tensor.shape)}, expected {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise SASRecError(f"{path}: state_dict[{key!r}] has NaN/Inf values")
        state[key] = tensor.contiguous()

    dropped = obj["dropped_keys"]
    if not isinstance(dropped, list) or not all(isinstance(k, str) for k in dropped):
        raise SASRecError(f"{path}: dropped_keys must be a list of str, got {dropped!r}")
    source = obj["source"]
    if not isinstance(source, dict) or set(source) != {"path", "git_blob", "sha256"}:
        raise SASRecError(f"{path}: source must be a dict with path, git_blob, sha256, got {source!r}")
    _check_hex(f"{path}: source.git_blob", source["git_blob"], _HEX40_RE)
    _check_hex(f"{path}: source.sha256", source["sha256"], _HEX64_RE)

    digest = state_digest(state)
    if obj["digest"] != digest:
        raise SASRecError(f"{path}: recomputed state digest {digest} != digest in the file {obj['digest']!r}")
    if digest != expected_digest:
        raise SASRecError(f"{path}: state digest {digest} != sasrec.state_digest {expected_digest}")
    return arch, state, digest


# ----------------------------------------------------------------------------
# fp32 region (D23, §7.6.5)
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _strict_fp32_region(dev: torch.device) -> Iterator[None]:
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if dev.type == "cuda":
            # torch 2.0.0 defaults: cudnn.allow_tf32 True (FFN Conv1d would run in TF32 on A100),
            # cuda.matmul.allow_tf32 False. cudnn.flags() is not used: it overwrites every cuDNN flag.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=dev.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def strict_fp32(device: torch.device) -> ContextManager[None]:
    """Autocast off for ``device.type`` and, on CUDA, cuBLAS and cuDNN TF32 off; the caller's flags
    come back afterwards (§7.6.3, §7.6.5, D23). M7 computes the gate in the same region (§11.6.3).
    """
    return _strict_fp32_region(_as_device(device))


# ----------------------------------------------------------------------------
# Original computation (§7.4), dropout positions kept as comments (D12)
# ----------------------------------------------------------------------------
class _MultiHeadAttention(nn.Module):
    """``SASRecModules_ori.MultiHeadAttention`` (blob 3ac3c975, :26-94) without its Dropout and Softmax modules.

    Same operations in the same order (§7.6.3): split/concat over heads, scores divided by
    sqrt(hidden_size) (not the head size, :59), key mask = key sum != 0 (:62-65), then the causal
    mask (:68-72), fill value -2^32+1, softmax, query mask = query sum != 0 (:78-80), weighted sum,
    residual with the queries (:92). Submodule names equal the original state keys.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        factory = {"device": "cpu", "dtype": torch.float32}
        self.linear_q = nn.Linear(hidden_size, hidden_size, **factory)   # [코드] :33-35 (num_units = hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size, **factory)
        self.linear_v = nn.Linear(hidden_size, hidden_size, **factory)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        Q = self.linear_q(queries)                                                   # :48
        K = self.linear_k(keys)                                                      # :49
        V = self.linear_v(keys)                                                      # :50
        split_size = self.hidden_size // self.num_heads                              # :53
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)                     # :54
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)                     # :55
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)                     # :56
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # :59
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)               # :62
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)                    # :63
        key_paddings = torch.ones_like(matmul_output) * _MASK_FILL                                  # :64
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # :65
        diag_vals = torch.ones_like(matmul_output[0, :, :])                                         # :68
        tril = torch.tril(diag_vals)                                                                # :69
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)                     # :70
        causality_paddings = torch.ones_like(causality_mask) * _MASK_FILL                           # :71
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # :72
        matmul_output_sm = F.softmax(matmul_output_m2, dim=-1)        # :75 (nn.Softmax(dim=-1), :37)
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)          # :78
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])                          # :79
        matmul_output_qm = matmul_output_sm * query_mask                                           # :80
        # (원본 :83 attention dropout 자리: D12로 항상 eval이라 항등, 모듈을 두지 않음)
        output_ws = torch.bmm(matmul_output_qm, V_)                                                # :86
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # :89
        return output + queries                                                                    # :92


class _PositionwiseFeedForward(nn.Module):
    """``SASRecModules_ori.PositionwiseFeedForward`` (blob 3ac3c975, :7-22) without its Dropout module.

    ``layer_norm(w_2(relu(w_1(x))) + x)`` with ``Conv1d(h, h, 1)`` kept as convolutions (not Linear),
    so that the CPU kernels and therefore the bits are those of the original (§7.6.3).
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        factory = {"device": "cpu", "dtype": torch.float32}
        self.w_1 = nn.Conv1d(hidden_size, hidden_size, 1, **factory)              # :10
        self.w_2 = nn.Conv1d(hidden_size, hidden_size, 1, **factory)              # :11
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps, **factory)          # :12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                                                             # :16
        output = x.transpose(1, 2)                                               # :17
        output = self.w_2(F.relu(self.w_1(output)))                              # :18
        output = output.transpose(1, 2)                                          # :19
        # (원본 :20 dropout 자리: D12로 항상 eval이라 항등, 모듈을 두지 않음)
        return self.layer_norm(output + residual)                                # :21


# ----------------------------------------------------------------------------
# Encoder (§7.6.3, §7.6.4)
# ----------------------------------------------------------------------------
class SASRecEncoder(nn.Module):
    """Frozen SASRec: ``z = encoder(hist_ids, len_seq)`` and ``encoder.embed_items(ids)`` (§3.1, §7.6.3).

    Other modules use only ``encoder(hist_ids, len_seq)``, ``embed_items``, ``hidden_size`` and
    ``state_digest`` (M9 ``SeqEncoder``); ``encode_all`` is for the reference comparison (check-env,
    M13). Built on CPU; M9 moves it with the model. ``from_config`` is the normal constructor;
    ``SASRecEncoder(arch, digest)`` alone has placeholder weights until ``load_state_dict``.
    """

    hidden_size: int
    item_num: int
    state_size: int
    state_digest: str

    def __init__(self, arch: SASRecArch, digest: str) -> None:
        super().__init__()
        if not isinstance(arch, SASRecArch):
            raise TypeError(f"arch must be a SASRecArch, got {type(arch).__name__}")
        _check_hex("digest", digest, _HEX64_RE)
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.item_num = arch.item_num            # = padding id ([코드] U data/movielens_data.py:18)
        self.state_size = arch.state_size
        self.num_heads = arch.num_heads
        self.state_digest = digest
        factory = {"device": "cpu", "dtype": torch.float32}
        # The default initialisation draws are overwritten by the state; fork_rng keeps the global CPU
        # generator untouched, so building the encoder does not shift any later random stream.
        with torch.random.fork_rng(devices=[]):
            # [코드] U recommender/A_SASRec_final_bce_llm.py:155-169 (names = original state keys)
            self.item_embeddings = nn.Embedding(arch.item_num + 1, arch.hidden_size, **factory)
            self.positional_embeddings = nn.Embedding(arch.state_size, arch.hidden_size, **factory)
            self.ln_1 = nn.LayerNorm(arch.hidden_size, eps=arch.layernorm_eps, **factory)
            self.ln_2 = nn.LayerNorm(arch.hidden_size, eps=arch.layernorm_eps, **factory)
            self.ln_3 = nn.LayerNorm(arch.hidden_size, eps=arch.layernorm_eps, **factory)
            self.mh_attn = _MultiHeadAttention(arch.hidden_size, arch.num_heads)
            self.feed_forward = _PositionwiseFeedForward(arch.hidden_size, arch.layernorm_eps)
        self.requires_grad_(False)               # frozen: never in an optimizer (§7.6.5)
        super().train(False)                     # D12

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, item_num={self.item_num}, state_size={self.state_size}, "
                f"num_heads={self.num_heads}, state_digest={self.state_digest[:12]}...")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: "Config", catalog: "ItemCatalog", history_len: int) -> "SASRecEncoder":
        """Load ``sasrec.state_path`` (checked against ``sasrec.state_digest``), build on CPU, cross-check
        with M1 and freeze (§7.6.3, §7.6.6). Device moves are M9's (§13.6.2 step 7).

        ``catalog.pad_id`` and ``catalog.num_items`` must equal the state's ``item_num`` and
        ``history_len`` (M1 ``HISTORY_LEN``, M5) its ``state_size`` [SPEC §4].
        """
        path = run.resolve_path(cfg.sasrec.state_path)
        arch, state, digest = load_sasrec_state(path, cfg.sasrec.state_digest)
        encoder = cls(arch, digest)
        encoder.load_state_dict(state, strict=True)
        problems = []
        pad_id = getattr(catalog, "pad_id", None)
        num_items = getattr(catalog, "num_items", None)
        if pad_id != arch.item_num:
            problems.append(f"catalog.pad_id {pad_id!r} != SASRec item_num {arch.item_num} (padding row)")
        if num_items != arch.item_num:
            problems.append(f"catalog.num_items {num_items!r} != SASRec item_num {arch.item_num}")
        if not _is_int(history_len) or history_len != arch.state_size:
            problems.append(f"history length {history_len!r} != SASRec state_size {arch.state_size}")
        if problems:
            raise SASRecError(f"SASRec state {path} does not fit the data (7.6.6): " + "; ".join(problems))
        encoder.requires_grad_(False)               # 고정 (load_state_dict copies values only)
        encoder.eval()
        loaded = state_digest(encoder.state_dict())
        if loaded != digest:
            raise SASRecError(f"encoder state digest {loaded} != loaded state digest {digest}")
        return encoder

    def train(self, mode: bool = True) -> "SASRecEncoder":
        """Always eval, whatever ``mode`` is (D12): a parent's ``.train()`` cannot turn it to training,
        unlike the released code where Lightning switched SASRec dropout on (§7.5)."""
        return super().train(False)

    # ------------------------------------------------------------------ checks
    def _param_device(self) -> torch.device:
        return self.item_embeddings.weight.device

    def _check_state(self) -> torch.device:
        """Parameters float32, frozen and on one device; encoder and submodules in eval (RuntimeError)."""
        devices = set()
        for name, param in self.named_parameters():
            devices.add(param.device)
            if param.dtype != torch.float32:
                raise RuntimeError(f"SASRec parameter {name} is {param.dtype}; the encoder computes in float32 only "
                                   "(D23), it is never cast with .half()/.to(dtype)")
            if param.requires_grad:
                raise RuntimeError(f"SASRec parameter {name} requires grad; the encoder is frozen (7.6.5)")
        if len(devices) != 1:
            raise RuntimeError(f"SASRec parameters are on several devices: {sorted(str(d) for d in devices)}")
        training = [name or "<encoder>" for name, module in self.named_modules() if module.training]
        if training:
            raise RuntimeError(f"SASRec modules in training mode: {training}; the encoder is always eval (D12)")
        return devices.pop()

    def _check_inputs(self, hist_ids: torch.Tensor, len_seq: torch.Tensor, device: torch.device) -> None:
        """``hist_ids [B, state_size]`` int64 right-padded with ``item_num``, ``len_seq [B]`` int64 (ValueError)."""
        if not isinstance(hist_ids, torch.Tensor) or not isinstance(len_seq, torch.Tensor):
            raise ValueError(f"hist_ids and len_seq must be tensors, got {type(hist_ids).__name__} and "
                             f"{type(len_seq).__name__}")
        if hist_ids.dtype != torch.int64:
            raise ValueError(f"hist_ids must be int64, got {hist_ids.dtype}")
        if len_seq.dtype != torch.int64:
            raise ValueError(f"len_seq must be int64, got {len_seq.dtype}")
        if hist_ids.dim() != 2 or hist_ids.shape[0] < 1 or hist_ids.shape[1] != self.state_size:
            raise ValueError(f"hist_ids must be [B, {self.state_size}] with B >= 1, got {list(hist_ids.shape)}")
        if tuple(len_seq.shape) != (hist_ids.shape[0],):
            raise ValueError(f"len_seq must be [{hist_ids.shape[0]}], got {list(len_seq.shape)}")
        if hist_ids.device != device or len_seq.device != device:
            raise ValueError(f"hist_ids ({hist_ids.device}) and len_seq ({len_seq.device}) must be on the encoder "
                             f"device {device}")
        positions = torch.arange(self.state_size, device=device)
        real = positions[None, :] < len_seq[:, None]
        ids_ok = torch.where(real, (hist_ids >= 0) & (hist_ids < self.item_num), hist_ids == self.item_num)
        rows_ok = ids_ok.all(dim=1) & (len_seq >= 1) & (len_seq <= self.state_size)
        if bool(rows_ok.all()):          # one synchronisation for all value checks
            return
        row = int((~rows_ok).nonzero()[0, 0])
        ids = [int(x) for x in hist_ids[row].tolist()]
        n = int(len_seq[row])
        pad = self.item_num
        if not 1 <= n <= self.state_size:
            reason = f"len_seq {n} is outside [1, {self.state_size}]"
        elif pad in ids[:n]:
            reason = f"padding id {pad} inside the first len_seq = {n} positions (left or middle padding)"
        elif any(not 0 <= x < pad for x in ids[:n]):
            reason = f"item id outside 0-{pad - 1} in the first {n} positions"
        else:
            reason = f"positions >= len_seq = {n} must all be the padding id {pad} (right padding only)"
        raise ValueError(f"hist_ids row {row} = {ids} with len_seq {n}: {reason}")

    # ------------------------------------------------------------------ computation
    def forward(self, hist_ids: torch.Tensor, len_seq: torch.Tensor) -> torch.Tensor:
        """``z [B, hidden_size] float32``: ``cacul_h`` of ``U recommender/A_SASRec_final_bce_llm.py:202-217``.

        Hidden state after ``ln_3`` at position ``len_seq - 1`` (the last real item, 식 10: the
        target is never an input). no_grad, fp32, autocast off, CUDA TF32 off; z is a plain tensor
        (not inference mode), so it can feed the gate's autograd graph as a constant.
        """
        device = self._param_device()
        self._check_inputs(hist_ids, len_seq, device)                                  # 1. ValueError
        self._check_state()                                                            # 2. RuntimeError
        with torch.no_grad(), strict_fp32(device):                                     # 3.
            positions = torch.arange(self.state_size, device=device)
            e = self.item_embeddings(hist_ids) + self.positional_embeddings(positions)   # [코드] :205-206
            # (원본 :207 emb_dropout 자리: D12로 항상 eval이라 항등, 모듈을 두지 않음)
            mask = torch.ne(hist_ids, self.item_num).float().unsqueeze(-1)              # :208
            seq = e * mask                                                               # :209
            a = self.mh_attn(self.ln_1(seq), seq)                                        # :210-211 query = ln_1 출력
            f = self.feed_forward(self.ln_2(a))                                          # :212
            f = f * mask                                                                 # :213
            f = self.ln_3(f)                                                             # :214
            # extract_axis_1 (:13-18, :215) as one gather; [B, 1, h] -> [B, h] (3.1 z)
            index = (len_seq - 1)[:, None, None].expand(-1, 1, self.hidden_size)
            z = f.gather(1, index).squeeze(1)
        if tuple(z.shape) != (hist_ids.shape[0], self.hidden_size) or z.dtype != torch.float32:   # 4.
            raise RuntimeError(f"z must be [{hist_ids.shape[0]}, {self.hidden_size}] float32, got "
                               f"{list(z.shape)} {z.dtype}")
        if not bool(torch.isfinite(z).all()):
            row = int((~torch.isfinite(z).all(dim=1)).nonzero()[0, 0])
            raise RuntimeError(f"z has NaN/Inf at row {row} (hist_ids {hist_ids[row].tolist()})")
        return z

    def embed_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Item representations ``[..., hidden_size] float32``: ``cacu_x`` (``U :219-222``).

        ``item_ids`` int64 on the encoder device with every value in ``0..item_num - 1``; the
        padding id ``item_num`` (1682) is not an item and is rejected like any other value outside
        the range (ValueError, 3.1 아이템 표현).
        """
        device = self._param_device()
        if not isinstance(item_ids, torch.Tensor):
            raise ValueError(f"item_ids must be a tensor, got {type(item_ids).__name__}")
        if item_ids.dtype != torch.int64:
            raise ValueError(f"item_ids must be int64, got {item_ids.dtype}")
        if item_ids.device != device:
            raise ValueError(f"item_ids ({item_ids.device}) must be on the encoder device {device}")
        bad = (item_ids < 0) | (item_ids >= self.item_num)
        if item_ids.numel() and bool(bad.any()):
            values = sorted({int(x) for x in item_ids[bad].tolist()})
            raise ValueError(f"item ids must be 0-{self.item_num - 1} (the padding id {self.item_num} is not an "
                             f"item), got {values[:10]}")
        self._check_state()
        with torch.no_grad(), strict_fp32(device):
            out = self.item_embeddings(item_ids)
        if out.dtype != torch.float32 or tuple(out.shape) != (*item_ids.shape, self.hidden_size):
            raise RuntimeError(f"item representations must be [..., {self.hidden_size}] float32, got "
                               f"{list(out.shape)} {out.dtype}")
        return out

    def encode_all(self, histories: Sequence[Sequence[int]], *, chunk: int = EXPORT_CHUNK,
                   num_threads: int = EXPORT_THREADS) -> torch.Tensor:
        """z of every history, ``[N, hidden_size]`` float32 on CPU, in input order (one split per call).

        M1 ``pad_histories(histories, PAD_ID, HISTORY_LEN)`` (5.6.13), then ``forward`` on
        consecutive slices of ``chunk`` rows. On CPU ``torch.set_num_threads(num_threads)`` holds
        during the call and the previous value comes back. With the reference file's ``chunk``
        and ``num_threads`` and one split per call, the CPU result equals the original bit for bit
        (§7.6.4); M12/M15 tables are made by ``records.compute_representations`` instead (§7.6.8).
        """
        for name, value in (("chunk", chunk), ("num_threads", num_threads)):
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        hist_ids, len_seq = pad_histories(histories, PAD_ID, HISTORY_LEN)
        device = self._param_device()
        previous = torch.get_num_threads() if device.type == "cpu" else None
        parts = []
        try:
            if previous is not None:
                torch.set_num_threads(num_threads)
            for start in range(0, hist_ids.shape[0], chunk):
                z = self(hist_ids[start:start + chunk].to(device), len_seq[start:start + chunk].to(device))
                parts.append(z.to("cpu"))
        finally:
            if previous is not None:
                torch.set_num_threads(previous)
        return torch.cat(parts)


# ----------------------------------------------------------------------------
# Reference comparison (check-env, M13; §7.6.1, §7.8 #5, #12)
# ----------------------------------------------------------------------------
def _load_reference(path: Path, encoder: SASRecEncoder) -> dict[str, Any]:
    """Read and check the reference file of ``tools/convert_sasrec.py`` (weights_only)."""
    if not path.is_file():
        raise SASRecError(f"SASRec reference file not found: {path} (rebuild/tools/convert_sasrec.py, DESIGN 7.6.1)")
    ref = _load_weights_only(path)
    if not isinstance(ref, dict) or set(ref) != set(REFERENCE_FILE_KEYS):
        keys = sorted(ref) if isinstance(ref, dict) else type(ref).__name__
        raise SASRecError(f"{path}: keys {keys} != {sorted(REFERENCE_FILE_KEYS)}")
    if ref["format"] != SASREC_REFERENCE_FORMAT:
        raise SASRecError(f"{path}: format {ref['format']!r} != {SASREC_REFERENCE_FORMAT!r}")
    if ref["state_digest"] != encoder.state_digest:
        raise SASRecError(f"{path}: reference made for SASRec state {ref['state_digest']!r}, encoder has "
                          f"{encoder.state_digest}")
    for name in ("chunk", "num_threads"):
        if not _is_int(ref[name]) or ref[name] < 1:
            raise SASRecError(f"{path}: {name} must be an int >= 1, got {ref[name]!r}")
    ids = ref["sample_ids"]
    if not isinstance(ids, list) or not ids or not all(isinstance(s, str) and _SAMPLE_ID_RE.fullmatch(s) for s in ids):
        raise SASRecError(f"{path}: sample_ids must be a non-empty list of 'split:row' strings")
    if len(set(ids)) != len(ids):
        raise SASRecError(f"{path}: sample_ids repeat")
    item_ids = ref["item_ids"]
    if not isinstance(item_ids, torch.Tensor) or item_ids.dim() != 1 or item_ids.shape[0] < 1:
        raise SASRecError(f"{path}: item_ids must be a non-empty 1-d tensor, got {item_ids!r:.80}")
    n, m, h, t = len(ids), item_ids.shape[0], encoder.hidden_size, encoder.state_size
    expected = {"hist_ids": ((n, t), torch.int64), "len_seq": ((n,), torch.int64), "z": ((n, h), torch.float32),
                "item_ids": ((m,), torch.int64), "item_emb": ((m, h), torch.float32)}
    for name, (shape, dtype) in expected.items():
        value = ref[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            got = (list(value.shape), value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__
            raise SASRecError(f"{path}: {name} must be {list(shape)} {dtype}, got {got}")
    return ref


def _split_blocks(sample_ids: Sequence[str], path: Path) -> dict[str, tuple[int, int]]:
    """``split -> (start, stop)``: each split is one contiguous block of the reference rows."""
    blocks: dict[str, tuple[int, int]] = {}
    current, start = None, 0
    for i, sid in enumerate([*sample_ids, None]):
        split = None if sid is None else sid.split(":", 1)[0]
        if split != current:
            if current is not None:
                if current in blocks:
                    raise SASRecError(f"{path}: rows of split {current!r} are not one contiguous block")
                blocks[current] = (start, i)
            current, start = split, i
    return blocks


def verify_against_reference(encoder: SASRecEncoder, reference_path: Path, device: torch.device) -> dict:
    """Compare the encoder on ``device`` with the original-code reference outputs (§7.6.1, §7.6.4).

    Per split, ``encode_all`` on the reference histories with the file's ``chunk`` and
    ``num_threads``; ``embed_items`` on the reference item ids. CPU: z and item rows must be
    bitwise equal (difference 0, §7.8 #5, smoke S0); CUDA: max |z - z_ref| <= 1e-4 and item rows
    bitwise (§7.8 #12). The reference rows must also be exactly M1 ``pad_histories`` of their
    histories. Returns the measured values (``passed`` True); a failed comparison raises
    ``SASRecError`` with the values in the message and on ``.report`` (§7.6.6).
    """
    dev = _as_device(device)
    if not isinstance(encoder, SASRecEncoder):
        raise TypeError(f"encoder must be a SASRecEncoder, got {type(encoder).__name__}")
    on = encoder._param_device()
    if on.type != dev.type or (dev.index is not None and on.index != dev.index):
        raise ValueError(f"the encoder is on {on}, the comparison was asked for {dev} (move it with .to first)")
    path = Path(reference_path)
    ref = _load_reference(path, encoder)
    tolerance = 0.0 if dev.type == "cpu" else CUDA_REFERENCE_ATOL
    report: dict[str, Any] = {
        "passed": False, "device": str(on), "reference_path": str(path), "state_digest": encoder.state_digest,
        "chunk": ref["chunk"], "num_threads": ref["num_threads"], "tolerance": tolerance,
        "n": len(ref["sample_ids"]), "splits": {},
    }
    ok = True
    worst = 0.0
    for split, (start, stop) in _split_blocks(ref["sample_ids"], path).items():
        hist, lens = ref["hist_ids"][start:stop], ref["len_seq"][start:stop]
        histories = [row[:n] for row, n in zip(hist.tolist(), lens.tolist())]
        padded, padded_len = pad_histories(histories, PAD_ID, HISTORY_LEN)
        padding_matches = torch.equal(padded, hist) and torch.equal(padded_len, lens)
        z = encoder.encode_all(histories, chunk=ref["chunk"], num_threads=ref["num_threads"])
        z_ref = ref["z"][start:stop]
        diff = float((z - z_ref).abs().max())
        bitwise = torch.equal(z, z_ref)
        report["splits"][split] = {"n": stop - start, "max_abs_diff": diff, "bitwise": bitwise,
                                   "padding_matches": padding_matches}
        worst = max(worst, diff)
        ok &= padding_matches and (bitwise if dev.type == "cpu" else diff <= tolerance)
    item_emb = encoder.embed_items(ref["item_ids"].to(on)).to("cpu")
    item_bitwise = torch.equal(item_emb, ref["item_emb"])
    report["item_emb"] = {"n": int(ref["item_ids"].shape[0]),
                          "max_abs_diff": float((item_emb - ref["item_emb"]).abs().max()), "bitwise": item_bitwise}
    report["max_abs_diff"] = worst
    report["bitwise"] = all(s["bitwise"] for s in report["splits"].values())
    ok &= item_bitwise
    report["passed"] = bool(ok)
    if not ok:
        error = SASRecError(f"SASRec differs from the reference outputs of the original code on {on} "
                            f"(CPU: bitwise, CUDA: <= {CUDA_REFERENCE_ATOL}; embed_items bitwise): {report}")
        error.report = report        # type: ignore[attr-defined]
        raise error
    return report


__all__ = [
    "SASREC_STATE_FORMAT", "SASREC_REFERENCE_FORMAT", "EXPORT_CHUNK", "EXPORT_THREADS", "CUDA_REFERENCE_ATOL",
    "STATE_FILE_KEYS", "ARCH_KEYS", "REFERENCE_FILE_KEYS", "SASRecError", "SASRecArch",
    "expected_state_shapes", "state_digest", "load_sasrec_state", "strict_fp32", "SASRecEncoder",
    "verify_against_reference",
]
