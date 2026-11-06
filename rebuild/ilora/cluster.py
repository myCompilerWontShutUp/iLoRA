"""M15 SASRec 군집과 cluster-hard 배정 (DESIGN.md §19; D25, D12, D23, D2, D8; [SPEC §0.8, §7-§11]).

Two halves:

- the ``cluster`` command (``run_cluster``, §19.6.2-§19.6.6): z of every train, validation and test sample from
  the frozen SASRec through M12 ``records.compute_representations(gate_fn=None)`` (CPU, fp32, the export
  settings of the ``records`` section), one KMeans fitted on the train z only (``build_fit_set``,
  ``fit_kmeans``: scikit-learn 1.2.2 exactly, every 1.2.2 default written out, OpenMP limited to
  ``cluster.fit_num_threads``, ConvergenceWarning and FutureWarning raised as errors), ``predict`` for every
  split (validation and test are never fitted [SPEC §0.8]), a float64 numpy cross-check of the nearest centre
  and the files of §19.2 under ``<run>/cluster/``: the z tables, the joblib model, ``centroids.npz``, the
  assignment CSVs, ``cluster_summary.json``, two figures and ``manifest.json`` with the sha256 of every file
  and input;
- cluster-hard routing (§19.6.7): ``load_cluster_artifacts`` re-checks a ``cluster/`` folder against its
  manifest and the current run (SASRec state sha256, data sha256, unit, number of clusters, fit split, sample
  ids), ``parse_expert_mapping`` reads ``cluster.expert_mapping``, ``ClusterHardGate`` (the M7 ``HardGate``)
  turns sample ids into one-hot omega by a dict lookup (no KMeans inside training or evaluation [SPEC §10]) and
  ``verify_live_encoder`` compares z computed on the training device with the stored train z.

The only random source is KMeans ``random_state`` = ``cluster.random_state`` (1234), used as given and not
through ``derive_seed`` (§4.6.4 exception, [SPEC §7]). Cluster numbers are scikit-learn's, never renumbered;
they carry no meaning [SPEC §9]. Nothing is overwritten (M0). Errors are ``ClusterError`` (a ``ValueError``)
with the cause in the message; an unknown sample id in ``ClusterHardGate`` raises ``KeyError`` (§19.6.7).
scikit-learn, threadpoolctl, joblib and matplotlib are imported inside the functions that need them, so the
training path (``load_cluster_artifacts``, ``ClusterHardGate``, ``verify_live_encoder``) never loads KMeans.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import re
import time
import types
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import run
from .data import (DATA_DIR, EXPECTED_SAMPLES, HISTORY_LEN, ITEM_FILE, MIN_LEN_SEQ, PAD_ID, SPLIT_FILES, SPLITS,
                   Sample, pad_histories)
from .records import (SPEC_SPLIT_NAMES, RecordsError, RepresentationTable, compute_representations, load_npz,
                      save_npz, validate_gate)
from .router import ROW_SUM_ATOL

if TYPE_CHECKING:   # type names only (§19.6.1): M3 encoder, M0 config sections, the scikit-learn estimator
    from sklearn.cluster import KMeans

    from .sasrec import SASRecEncoder
    from .sections import ClusterConfig, Config

logger = logging.getLogger("ilora.cluster")

# ----------------------------------------------------------------------------
# Files and fixed values (§19.2, §19.6.5, §19.6.6, §19.6.10)
# ----------------------------------------------------------------------------
CLUSTER_DIR: Final = "cluster"                     # <cluster run>/cluster/ = cluster.artifact_dir of E4 (§19.6.7)
ARTIFACT_FORMAT: Final = "ilora-rebuild-cluster-v1"   # manifest.json "format" [결정]
REQUIRED_SKLEARN: Final = run.REQUIRED_LIBRARIES["sklearn"]   # 1.2.2 (§19.6.4; U requirements.txt:112)
UNITS: Final = ("sequence", "user")                # cluster.unit: D25, user = variant_user_level_cluster (§19.6.8)
FIT_SPLIT: Final = "train"                         # the only split KMeans is fitted on [SPEC §0.8, §7]
MANIFEST_FILE: Final = "manifest.json"
SUMMARY_FILE: Final = "cluster_summary.json"       # [SPEC §7]
CENTROIDS_FILE: Final = "centroids.npz"            # pickle-free copy of the centres (§19.2)
USER_ASSIGNMENTS_FILE: Final = "train_user_assignments.csv"   # unit = user only (§19.6.8)
FIGURES_DIR: Final = "figures"
PCA_FIGURE: Final = "sasrec_pca_clusters"          # [SPEC §8]
HEATMAP_FIGURE: Final = "cluster_dynamic_gate_heatmap"   # [SPEC §8]
ASSIGNMENT_FIELDS: Final = ("sample_id", "split", "row", "user_id", "cluster_id", "sq_dist_1", "sq_dist_2")  # [SPEC §7]
USER_ASSIGNMENT_FIELDS: Final = ("user_id", "fit_sample_id", "cluster_id")
CENTROID_KEYS: Final = ("cluster_centers", "inertia", "n_iter", "fit_ids_sha256")
FLOAT_FORMAT: Final = "%.9g"                       # real numbers in the CSV files (§19.6.5)
NOT_RUN_DYNAMIC: Final = "NOT_RUN: dynamic_export_dir is null"   # §19.6.6
DYNAMIC_ITEMS: Final = ("cluster_dynamic_gate_mean", "cluster_dynamic_gate_std",
                        "contingency_cluster_vs_dynamic_argmax", "cluster_dynamic_argmax_distribution",
                        "nmi_cluster_vs_dynamic_argmax")   # §19.6.6, one value per split when the E2 export is given
# The data files whose sha256 binds the artifacts to the samples (M0 meta.json "inputs", §19.6.5).
DATA_FILES: Final = tuple(f"{DATA_DIR}/{name}" for name in (*SPLIT_FILES.values(), ITEM_FILE))
NUMPY_CHUNK: Final = 8192                          # rows per float64 block of the numpy nearest-centre check
_MAPPING_RE: Final = re.compile(r"[0-9]+:[0-9]+(?:,[0-9]+:[0-9]+)*")   # §19.6.7: re.fullmatch, no ^...$, no \d
_PAIR_RE: Final = re.compile(r"[0-9]+:[0-9]+")
_ZIP_DATE: Final = (1980, 1, 1, 0, 0, 0)            # fixed zip member time: the same arrays give the same bytes
_MAX_LISTED_IDS: Final = 1000                       # sample ids listed per mismatch kind in the summary


class ClusterError(ValueError):
    """A cluster artifact, input or setting breaks the M15 rules (§19.6.9); the message names the cause."""


def kmeans_file_name(n_clusters: int, random_state: int) -> str:
    """``kmeans_k{C}_seed{random_state}.joblib`` [SPEC §7] (paper_spec: ``kmeans_k4_seed1234.joblib``)."""
    return f"kmeans_k{n_clusters}_seed{random_state}.joblib"


def representation_file(split: str) -> str:
    """``representations_{train,validation,test}.npz``: SPEC split spelling in the name (16.2)."""
    return f"representations_{SPEC_SPLIT_NAMES[split]}.npz"


def assignment_file(split: str) -> str:
    """``{train,validation,test}_assignments.csv`` (§19.2)."""
    return f"{SPEC_SPLIT_NAMES[split]}_assignments.csv"


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _positive_number(value: object) -> bool:
    """A finite real number > 0 (bool excluded): tolerances and KMeans ``tol``."""
    return (isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)) and float(value) > 0)


def ids_sha256(sample_ids: Sequence[str]) -> str:
    """sha256 of the ids, each followed by ``"\\n"`` (UTF-8): fingerprint of a fit list (§19.6.3, manifest)."""
    return hashlib.sha256("".join(f"{sid}\n" for sid in sample_ids).encode("utf-8")).hexdigest()


def check_sklearn_version() -> str:
    """scikit-learn must be exactly 1.2.2 (§19.6.4): 1.3 removed algorithm 'auto'/'full' and 1.4 changed the
    n_init default to 'auto' [SPEC §7]. Read at call time, so a different install fails before any KMeans use."""
    import sklearn

    got = getattr(sklearn, "__version__", None)
    if got != REQUIRED_SKLEARN:
        raise ClusterError(f"scikit-learn {got} != {REQUIRED_SKLEARN} (19.6.4): KMeans results and defaults "
                           "change between versions, so clustering does not run")
    return got


def library_versions() -> dict[str, Optional[str]]:
    """Versions recorded with the artifacts (§19.6.5): scikit-learn, threadpoolctl, joblib, numpy, scipy, torch."""
    import joblib
    import scipy
    import sklearn
    import threadpoolctl

    return {"sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__,
            "joblib": joblib.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__}


def _plain(obj: object) -> object:
    """Plain JSON values (numpy scalars and arrays, paths, tuples)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return str(obj)


def _new_file(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise ClusterError(f"{path} exists; cluster artifacts are never overwritten (M0)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_bytes(path: Path, data: bytes) -> str:
    """Temporary name + ``os.replace`` (never over an existing file); returns the sha256."""
    path = _new_file(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, obj: object) -> str:
    """UTF-8 JSON, fixed key order, no NaN, LF; token-like strings refused (M0 ``find_secrets``)."""
    data = _plain(obj)
    hits = run.find_secrets(data)
    if hits:
        raise run.SecretLeakError(f"refusing to write token-like values at {hits} to {path}")
    return _write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _csv_bytes(lines: Sequence[Sequence[object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(lines)
    return buffer.getvalue().encode("utf-8")


def _read_csv(path: Path, fields: Sequence[str]) -> list[list[str]]:
    path = Path(path)
    if not path.is_file():
        raise ClusterError(f"{path} not found")
    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = list(csv.reader(f))
    if not lines or tuple(lines[0]) != tuple(fields):
        raise ClusterError(f"{path}: header {lines[0] if lines else None} != {list(fields)}")
    body = lines[1:]
    for i, line in enumerate(body, start=2):
        if len(line) != len(fields):
            raise ClusterError(f"{path} line {i}: {len(line)} cells, expected {len(fields)}")
    if not body:
        raise ClusterError(f"{path}: no rows")
    return body


def _write_npz(path: Path, members: Sequence[tuple[str, np.ndarray]]) -> str:
    """An ``np.load(allow_pickle=False)`` file with fixed member times; returns the sha256."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, array in members:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3                     # same bytes on Windows and Linux
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
    return _write_bytes(path, buffer.getvalue())

# ----------------------------------------------------------------------------
# Loaded artifacts, expert mapping and the hard gate (§19.6.1, §19.6.7)
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ClusterArtifacts:
    """A checked ``cluster/`` folder: what cluster-hard routing reads (§19.6.1).

    ``assignments`` maps every sample id of the three splits of the manifest's data subset to its cluster id;
    it does not depend on the expert mapping (§19.2). ``centers`` float32 [C, d] come from ``centroids.npz``,
    ``manifest`` is the parsed ``manifest.json``. Built by ``load_cluster_artifacts``; ``assignments`` and
    ``manifest`` are read-only views and ``centers`` is a read-only array (``eq=False``: arrays have no single
    truth value).
    """
    dir: Path
    unit: Literal["sequence", "user"]
    n_clusters: int
    centers: np.ndarray                    # float32 [C, d]
    assignments: Mapping[str, int]         # sample_id -> cluster_id (all three splits)
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ClusterError(f"ClusterArtifacts.unit must be one of {UNITS}, got {self.unit!r}")
        if not _is_int(self.n_clusters) or self.n_clusters < 2:
            raise ClusterError(f"ClusterArtifacts.n_clusters must be an int >= 2, got {self.n_clusters!r}")
        c = int(self.n_clusters)
        centers = self.centers
        if (not isinstance(centers, np.ndarray) or centers.dtype != np.float32 or centers.ndim != 2
                or centers.shape[0] != c or centers.shape[1] < 1):
            got = f"{centers.dtype} {list(centers.shape)}" if isinstance(centers, np.ndarray) else type(centers).__name__
            raise ClusterError(f"ClusterArtifacts.centers must be float32 [{c}, d], got {got}")
        if not isinstance(self.assignments, Mapping) or not self.assignments:
            raise ClusterError("ClusterArtifacts.assignments must be a non-empty mapping sample_id -> cluster_id")
        table: dict[str, int] = {}
        for sid, cluster in self.assignments.items():
            if not isinstance(sid, str) or not _is_int(cluster) or not 0 <= int(cluster) < c:
                raise ClusterError(f"ClusterArtifacts.assignments[{sid!r}] = {cluster!r} is not a cluster id 0..{c - 1}")
            table[sid] = int(cluster)
        if not isinstance(self.manifest, Mapping):
            raise ClusterError(f"ClusterArtifacts.manifest must be a mapping, got {type(self.manifest).__name__}")
        frozen_centers = np.array(centers, dtype=np.float32, copy=True)
        frozen_centers.setflags(write=False)
        object.__setattr__(self, "dir", Path(self.dir))
        object.__setattr__(self, "n_clusters", c)
        object.__setattr__(self, "centers", frozen_centers)
        object.__setattr__(self, "assignments", types.MappingProxyType(table))
        object.__setattr__(self, "manifest", types.MappingProxyType(dict(self.manifest)))


def parse_expert_mapping(text: str, *, n_clusters: int, num_experts: int) -> tuple[int, ...]:
    """``cluster.expert_mapping`` -> the expert of cluster c at index c (§19.6.7, [SPEC §9]).

    The text must match ``re.fullmatch(r"[0-9]+:[0-9]+(?:,[0-9]+:[0-9]+)*")``: no ``^...$`` (``$`` also matches
    before a final newline) and no ``\\d`` (it accepts non-ASCII digits). Each cluster 0..C-1 appears exactly
    once and every expert is in 0..K-1; two clusters may share an expert (``ClusterHardGate`` logs the unused
    experts). Every rejection names the offending token: empty entries, duplicates, missing clusters, values
    out of range, spaces, signs, a final newline, non-ASCII digits. The main experiment is ``0:0,1:1,2:2,3:3``.
    """
    if not isinstance(text, str):
        raise ClusterError(f"cluster.expert_mapping must be a str, got {type(text).__name__}: {text!r}")
    for name, value in (("n_clusters", n_clusters), ("num_experts", num_experts)):
        if not _is_int(value) or value < 1:
            raise ClusterError(f"{name} must be an int >= 1, got {value!r}")
    tokens = text.split(",")
    if _MAPPING_RE.fullmatch(text) is None:
        bad = next((t for t in tokens if _PAIR_RE.fullmatch(t) is None), text)
        what = "an empty entry ''" if bad == "" else f"the token {bad!r}"
        raise ClusterError(f"cluster.expert_mapping {text!r}: {what} is not '<cluster>:<expert>' written with the "
                           "ASCII digits 0-9 (format 'c:e,c:e,...' without spaces, signs or a final newline)")
    experts: dict[int, int] = {}
    first: dict[int, str] = {}
    for token in tokens:
        cluster, expert = (int(part) for part in token.split(":"))
        if cluster >= n_clusters:
            raise ClusterError(f"cluster.expert_mapping {text!r}: the token {token!r} names cluster {cluster}, "
                               f"outside 0..{n_clusters - 1}")
        if expert >= num_experts:
            raise ClusterError(f"cluster.expert_mapping {text!r}: the token {token!r} names expert {expert}, "
                               f"outside 0..{num_experts - 1}")
        if cluster in experts:
            raise ClusterError(f"cluster.expert_mapping {text!r}: the token {token!r} repeats cluster {cluster} "
                               f"(already given by {first[cluster]!r}); each cluster appears exactly once")
        experts[cluster] = expert
        first[cluster] = token
    missing = [c for c in range(n_clusters) if c not in experts]
    if missing:
        raise ClusterError(f"cluster.expert_mapping {text!r}: clusters {missing} have no expert (each of "
                           f"0..{n_clusters - 1} appears exactly once)")
    return tuple(experts[c] for c in range(n_clusters))


def _same_device(actual: torch.device, requested: torch.device) -> bool:
    return actual.type == requested.type and (requested.index is None or actual.index == requested.index)


class ClusterHardGate:
    """The M7 ``HardGate`` of cluster_hard (§19.6.7, §11.6.1): ``gate(sample_ids, device) -> omega [B, K]``.

    ``sample_id`` -> cluster (a dict lookup in the stored assignments; no KMeans in the loop [SPEC §10]) ->
    expert ``mapping[cluster]`` -> ``torch.zeros(B, K, float32, device).scatter_(1, idx, 1.0)``, made directly on
    the requested device and never cast to fp16 (§19.6.9). Every call checks shape, dtype, device, finite 0/1
    values and row sums of exactly 1.0 (M12 ``validate_gate`` plus the one-hot rule, [SPEC §11]). An unknown id
    raises ``KeyError`` naming the id and the artifact folder (no fallback). ``lookups`` counts the looked-up
    samples (M10 logs the training increments, M12 16.6.2); ``unused_experts`` lists experts no cluster maps to.
    """

    def __init__(self, artifacts: ClusterArtifacts, mapping: tuple[int, ...], num_experts: int) -> None:
        if not isinstance(artifacts, ClusterArtifacts):
            raise ClusterError(f"ClusterHardGate needs ClusterArtifacts, got {type(artifacts).__name__}")
        if not _is_int(num_experts) or num_experts < 1:
            raise ClusterError(f"num_experts must be an int >= 1, got {num_experts!r}")
        mapping = tuple(mapping)
        if len(mapping) != artifacts.n_clusters or not all(_is_int(e) and 0 <= e < num_experts for e in mapping):
            raise ClusterError(f"mapping {mapping} must give an expert 0..{num_experts - 1} for each of the "
                               f"{artifacts.n_clusters} clusters (parse_expert_mapping)")
        self.artifacts = artifacts
        self.mapping: tuple[int, ...] = tuple(int(e) for e in mapping)
        self.num_experts = int(num_experts)
        self._expert = {sid: self.mapping[cluster] for sid, cluster in artifacts.assignments.items()}
        self.unused_experts: tuple[int, ...] = tuple(k for k in range(self.num_experts) if k not in self.mapping)
        self.lookups = 0
        if self.unused_experts:
            logger.warning("cluster.expert_mapping %s leaves experts %s without a cluster: they are never active in "
                           "cluster_hard (19.6.7)", list(self.mapping), list(self.unused_experts))

    def __call__(self, sample_ids: Sequence[str], device: torch.device) -> Tensor:
        ids = list(sample_ids)
        if not ids:
            raise ClusterError("ClusterHardGate needs at least one sample id (B >= 1)")
        experts = []
        for sid in ids:
            expert = self._expert.get(sid) if isinstance(sid, str) else None
            if expert is None:
                raise KeyError(f"sample_id {sid!r} has no cluster assignment in {self.artifacts.dir} "
                               "(no fallback, 19.6.7)")
            experts.append(expert)
        dev = torch.device(device)
        idx = torch.tensor(experts, dtype=torch.int64, device=dev)
        omega = torch.zeros(len(ids), self.num_experts, dtype=torch.float32, device=dev).scatter_(1, idx[:, None], 1.0)
        self._check(omega, ids, dev)
        self.lookups += len(ids)
        return omega

    def _check(self, omega: Tensor, ids: Sequence[str], dev: torch.device) -> None:
        b, k = len(ids), self.num_experts
        if omega.dtype != torch.float32 or tuple(omega.shape) != (b, k) or not _same_device(omega.device, dev):
            raise ClusterError(f"one-hot omega must be float32 [{b}, {k}] on {dev}, got {omega.dtype} "
                               f"{list(omega.shape)} on {omega.device}")
        try:
            validate_gate(omega, batch=b, num_experts=k, rowsum_atol=ROW_SUM_ATOL, sample_ids=ids)
        except RecordsError as exc:
            raise ClusterError(f"cluster_hard omega breaks the gate contract: {exc}") from exc
        one_hot = ((omega == 0) | (omega == 1)).all(dim=1) & (omega.sum(dim=1) == 1)
        if not bool(one_hot.all()):
            row = int((~one_hot).nonzero()[0, 0])
            raise ClusterError(f"cluster_hard omega row {row} ({ids[row]}) is not one-hot: {omega[row].tolist()}")

# ----------------------------------------------------------------------------
# Fit set, KMeans and the numerical checks (§19.6.2-§19.6.5, §19.6.8, §19.6.9)
# ----------------------------------------------------------------------------
def _check_z(z: object, name: str, *, dim: Optional[int] = None) -> np.ndarray:
    """z for fit/predict (§19.6.9): numpy float32 [N, d] (N, d >= 1), C-contiguous, finite."""
    if not isinstance(z, np.ndarray):
        raise ClusterError(f"{name} must be a numpy array, got {type(z).__name__}")
    if z.dtype != np.float32:
        raise ClusterError(f"{name} must be float32 (19.6.9), got {z.dtype}; float64 is refused here because "
                           "scikit-learn 1.2.2 fails on it with 'Buffer dtype mismatch' for a float32 model")
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ClusterError(f"{name} must be 2-dim [N, d] with N, d >= 1, got shape {list(z.shape)}")
    if dim is not None and z.shape[1] != dim:
        raise ClusterError(f"{name} has {z.shape[1]} columns, expected d = {dim}")
    if not z.flags.c_contiguous:
        raise ClusterError(f"{name} must be C-contiguous (19.6.9); a strided or Fortran-order view would be "
                           "copied silently by scikit-learn")
    finite = np.isfinite(z).all(axis=1)
    if not bool(finite.all()):
        raise ClusterError(f"{name} has NaN/Inf in rows {np.flatnonzero(~finite)[:5].tolist()}")
    return z


def build_fit_set(train: RepresentationTable, samples: Sequence[Sample], *, unit: str
                  ) -> tuple[np.ndarray, tuple[str, ...]]:
    """The KMeans input and its sample ids (§19.6.3, §19.6.8, D25; leak checks [SPEC §0.8, §7]).

    ``samples`` is the M1 train list (``LoadedData.train``) and ``train`` its z table. ``sequence``: every train
    sample in M1 row order (66,126 rows for the full data; 25 of them repeat an earlier (history, len_seq) and
    are kept). ``user``: per pseudo user (D14) the train window with the largest row, in user_id order. Checks:
    the table is the train split and lists exactly the ids of ``samples`` in order; every sample is an M1 train
    ``Sample`` in strictly ascending row order (any validation or test sample fails); the result is the raw z
    as float32 [n, d] (d = the z width, never 2-D PCA coordinates), C-contiguous and finite, with ``train:`` ids.
    """
    if unit not in UNITS:
        raise ClusterError(f"cluster.unit must be one of {UNITS}, got {unit!r}")
    if not isinstance(train, RepresentationTable) or train.split != FIT_SPLIT:
        got = train.split if isinstance(train, RepresentationTable) else type(train).__name__
        raise ClusterError(f"the fit table must be the train RepresentationTable, got {got!r} (fit split train only, "
                           "[SPEC §0.8])")
    samples = list(samples)
    foreign = [s for s in samples if not isinstance(s, Sample) or s.split != FIT_SPLIT]
    if foreign:
        first = foreign[0].sample_id if isinstance(foreign[0], Sample) else repr(foreign[0])
        raise ClusterError(f"the fit list holds {len(foreign)} samples that are not M1 train samples (first {first}); "
                           "KMeans is fitted on train samples only [SPEC §0.8]")
    rows = [s.row for s in samples]
    if not rows or any(b <= a for a, b in zip(rows, rows[1:])):
        raise ClusterError("the fit list must be the non-empty M1 train list in strictly ascending row order")
    ids = tuple(s.sample_id for s in samples)
    if ids != train.sample_ids:
        raise ClusterError("the train z table does not list the M1 train samples in M1 order (19.6.3)")
    z = _check_z(train.z, "train z")
    if unit == "sequence":
        index = np.arange(len(samples), dtype=np.int64)
    else:
        last: dict[int, int] = {}
        for i, s in enumerate(samples):          # ascending rows: the last window of a user has its largest row
            last[s.user_id] = i
        index = np.array([last[u] for u in sorted(last)], dtype=np.int64)
    z_fit = z if unit == "sequence" else np.ascontiguousarray(z[index])
    fit_ids = tuple(ids[int(i)] for i in index)
    if any(not sid.startswith(f"{FIT_SPLIT}:") for sid in fit_ids):          # leak check 1 (19.6.3)
        raise ClusterError("fit ids outside the train split")
    if unit == "sequence" and fit_ids != ids:
        raise ClusterError("the sequence fit list differs from the M1 train list")
    _check_z(z_fit, "z_fit", dim=train.dim)                                 # leak check 2: raw [n, d] float32
    return z_fit, fit_ids


def fit_kmeans(z_fit: np.ndarray, ccfg: "ClusterConfig") -> "KMeans":
    """One KMeans fit on the train z (§19.6.4, [SPEC §7]).

    ``KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, tol=1e-4, algorithm="lloyd",
    random_state=1234, copy_x=True)`` with the values of the ``cluster`` section: every scikit-learn 1.2.2
    default written out, scikit-learn exactly 1.2.2. ``fit`` is called once, inside
    ``threadpool_limits(limits=cluster.fit_num_threads, user_api="openmp")``: Lloyd adds per-thread partial sums
    in lock order, so the result depends on the thread count (one thread repeated bit for bit). ConvergenceWarning
    (fewer distinct clusters than n_clusters at the end, e.g. duplicate points, sklearn 1.2.2
    ``cluster/_kmeans.py:1497-1505``) and FutureWarning are raised as errors. Empty clusters during the Lloyd
    iterations are moved silently by scikit-learn, so the empty-cluster check of the assignment is separate.
    """
    check_sklearn_version()
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    z = _check_z(z_fit, "z_fit")
    for name in ("n_clusters", "n_init", "max_iter", "fit_num_threads"):
        value = getattr(ccfg, name)
        if not _is_int(value) or value < (2 if name == "n_clusters" else 1):
            raise ClusterError(f"cluster.{name} = {value!r} is not a valid int")
    if ccfg.init != "k-means++" or ccfg.algorithm != "lloyd" or not _positive_number(ccfg.tol):
        raise ClusterError(f"cluster.init/algorithm/tol = {ccfg.init!r}/{ccfg.algorithm!r}/{ccfg.tol!r}; "
                           "19.6.10 allows k-means++, lloyd and a float > 0")
    if z.shape[0] < ccfg.n_clusters:
        raise ClusterError(f"{z.shape[0]} fit rows < n_clusters {ccfg.n_clusters}")
    km = KMeans(n_clusters=ccfg.n_clusters, init=ccfg.init, n_init=ccfg.n_init, max_iter=ccfg.max_iter,
                tol=ccfg.tol, algorithm=ccfg.algorithm, random_state=ccfg.random_state, copy_x=True)
    try:
        with threadpool_limits(limits=ccfg.fit_num_threads, user_api="openmp"), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", FutureWarning)
            km.fit(z)
    except (ConvergenceWarning, FutureWarning) as exc:
        raise ClusterError(f"KMeans fit stopped on a warning promoted to an error (19.6.4): "
                           f"{type(exc).__name__}: {exc}") from exc
    threads = getattr(km, "_n_threads", None)
    if not _is_int(threads) or threads > ccfg.fit_num_threads:
        raise ClusterError(f"KMeans ran with {threads} OpenMP threads, cluster.fit_num_threads = "
                           f"{ccfg.fit_num_threads}: the thread limit did not apply (19.6.4)")
    centers = km.cluster_centers_
    if centers.dtype != np.float32 or centers.shape != (ccfg.n_clusters, z.shape[1]):
        raise ClusterError(f"KMeans centres are {centers.dtype} {list(centers.shape)}, expected float32 "
                           f"[{ccfg.n_clusters}, {z.shape[1]}]")
    return km


def predict(km: "KMeans", z: np.ndarray) -> np.ndarray:
    """Cluster ids of ``z`` with a fitted KMeans, int64 [N] (§19.6.5).

    ``z`` must be float32 [N, d], C-contiguous and finite: float64 makes scikit-learn 1.2.2 fail with "Buffer
    dtype mismatch" and a non-contiguous array would be copied silently, so both are refused here first with
    this module's message. Validation and test z only ever pass through here, never through ``fit``.
    """
    check_sklearn_version()
    centers = getattr(km, "cluster_centers_", None)
    if not isinstance(centers, np.ndarray) or centers.ndim != 2:
        raise ClusterError("predict needs a fitted KMeans (cluster_centers_ missing)")
    x = _check_z(z, "predict input z", dim=int(centers.shape[1]))
    labels = np.asarray(km.predict(x)).astype(np.int64)
    if labels.shape != (x.shape[0],) or bool((labels < 0).any()) or bool((labels >= centers.shape[0]).any()):
        raise ClusterError(f"KMeans.predict returned labels outside 0..{centers.shape[0] - 1}")
    return labels


def nearest_two(z: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float64 cross-check of ``predict`` (§19.6.5): squared distances to every centre as sums of squared
    differences in float64 (no norm expansion) -> (nearest cluster int64 [N], first index on ties; nearest and
    second nearest squared distance, float64 [N] each)."""
    x = np.asarray(z)
    c = np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1] or c.shape[0] < 2:
        raise ClusterError(f"nearest_two needs z [N, d] and >= 2 centres [C, d], got {list(x.shape)} and {list(c.shape)}")
    n = x.shape[0]
    labels = np.empty(n, dtype=np.int64)
    first = np.empty(n, dtype=np.float64)
    second = np.empty(n, dtype=np.float64)
    for start in range(0, n, NUMPY_CHUNK):
        block = x[start:start + NUMPY_CHUNK].astype(np.float64)
        diff = block[:, None, :] - c[None, :, :]
        dist = (diff * diff).sum(axis=2)
        order = np.argsort(dist, axis=1, kind="stable")
        take = np.arange(block.shape[0])
        labels[start:start + block.shape[0]] = order[:, 0]
        first[start:start + block.shape[0]] = dist[take, order[:, 0]]
        second[start:start + block.shape[0]] = dist[take, order[:, 1]]
    return labels, first, second


def z_scale(z: np.ndarray) -> dict:
    """Size of the representation before clustering (§19.6.2, [SPEC §7]): per-dimension mean and standard
    deviation (ddof 0) with their minimum and maximum (and max/min for the standard deviation) and the row L2
    norm statistics, in float64. Recorded only: the main experiment clusters the raw z (no scaler, no PCA)."""
    x = np.asarray(z, dtype=np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    norms = np.linalg.norm(x, axis=1)
    smallest = float(std.min())
    return {"n": int(x.shape[0]), "d": int(x.shape[1]), "ddof": 0, "clustered_input": "raw z (no StandardScaler, no PCA)",
            "per_dim_mean": {"min": float(mean.min()), "max": float(mean.max()), "values": mean.tolist()},
            "per_dim_std": {"min": smallest, "max": float(std.max()),
                            "max_over_min": float(std.max()) / smallest if smallest > 0 else None,
                            "values": std.tolist()},
            "row_l2_norm": {"mean": float(norms.mean()), "std": float(norms.std()), "min": float(norms.min()),
                            "max": float(norms.max())}}


def silhouette_train(z_fit: np.ndarray, labels: np.ndarray) -> dict:
    """Exact silhouette of the whole fit set (§19.6.6): ``silhouette_score(metric="euclidean",
    sample_size=None)``, so no sampling and no random_state (about a minute for 66,126 x 64 on one CPU). The
    last digits may depend on BLAS threads; the value is recorded, never compared between runs."""
    from sklearn.metrics import silhouette_score

    start = time.perf_counter()
    value = float(silhouette_score(z_fit, labels, metric="euclidean", sample_size=None))
    return {"value": value, "n": int(len(labels)), "metric": "euclidean", "sample_size": None,
            "seconds": round(time.perf_counter() - start, 3)}

# ----------------------------------------------------------------------------
# Fit and assignment of every split (§19.6.3-§19.6.5, §19.6.8)
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SplitAssignment:
    """Cluster assignment of one split (§19.6.5) in M1 order.

    ``cluster_ids`` is what the CSV stores; ``predicted`` is ``predict`` of each sample's own z (the same array
    except train with ``unit = user``, where a sample takes its user's cluster). ``sq_dist_1``/``sq_dist_2`` are
    the float64 nearest and second nearest squared distances; ``numpy_mismatch_ids`` lists the samples whose
    float64 nearest centre differs from ``predicted``.
    """
    split: str
    sample_ids: tuple[str, ...]
    rows: np.ndarray
    user_ids: np.ndarray
    cluster_ids: np.ndarray
    predicted: np.ndarray
    sq_dist_1: np.ndarray
    sq_dist_2: np.ndarray
    numpy_mismatch_ids: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class ClusterFit:
    """``fit_and_assign`` result: the KMeans, its fit set, the fit-set clusters (``predict``; colours of the PCA
    figure, labels of the silhouette), the number of fit rows where ``labels_`` differs from ``predict`` and the
    assignment of each split."""
    unit: str
    km: Any
    fit_ids: tuple[str, ...]
    z_fit: np.ndarray
    fit_clusters: np.ndarray
    labels_mismatches: int
    splits: Mapping[str, SplitAssignment]
    fit_seconds: float

    @property
    def n_clusters(self) -> int:
        return int(self.km.n_clusters)


def _check_tables(tables: Mapping[str, RepresentationTable], samples: Mapping[str, Sequence[Sample]]) -> int:
    dims = set()
    for split in SPLITS:
        table, chosen = tables.get(split), samples.get(split)
        if not isinstance(table, RepresentationTable) or table.split != split or chosen is None:
            raise ClusterError(f"{split}: a RepresentationTable of that split and its M1 samples are required")
        if table.sample_ids != tuple(s.sample_id for s in chosen):
            raise ClusterError(f"{split}: the z table does not list the M1 samples in M1 order")
        dims.add(table.dim)
    if len(dims) != 1:
        raise ClusterError(f"the z tables have different widths {sorted(dims)}")
    return dims.pop()


def fit_and_assign(tables: Mapping[str, RepresentationTable], samples: Mapping[str, Sequence[Sample]],
                   ccfg: "ClusterConfig") -> ClusterFit:
    """Fit on train once and assign every split (§19.6.3-§19.6.5, §19.6.8).

    ``tables`` and ``samples`` are keyed by the contract splits (train, val, test) in M1 order. KMeans sees only
    ``build_fit_set`` (one ``fit``); each split's z goes through ``predict`` only, ``labels_`` is not used as the
    assignment (its disagreements with ``predict`` on the fit set are counted). Every split is also checked
    with ``nearest_two``. With ``unit = user`` a train sample takes the cluster of its user's fit window. A train
    cluster without samples is a failure: that expert would never be trained.
    """
    _check_tables(tables, samples)
    start = time.perf_counter()
    z_fit, fit_ids = build_fit_set(tables["train"], samples["train"], unit=ccfg.unit)
    km = fit_kmeans(z_fit, ccfg)
    fit_seconds = round(time.perf_counter() - start, 3)
    fit_clusters = predict(km, z_fit)
    labels_mismatches = int((np.asarray(km.labels_).astype(np.int64) != fit_clusters).sum())
    user_cluster: dict[int, int] = {}
    if ccfg.unit == "user":
        user_of = {s.sample_id: s.user_id for s in samples["train"]}
        user_cluster = {user_of[sid]: int(c) for sid, c in zip(fit_ids, fit_clusters.tolist())}
    splits: dict[str, SplitAssignment] = {}
    for split in SPLITS:
        table = tables[split]
        predicted = predict(km, table.z)
        nearest, first, second = nearest_two(table.z, km.cluster_centers_)
        if split == FIT_SPLIT and ccfg.unit == "user":
            cluster_ids = np.array([user_cluster[int(u)] for u in table.user_ids.tolist()], dtype=np.int64)
        else:
            cluster_ids = predicted
        mismatch = tuple(table.sample_ids[i] for i in np.flatnonzero(nearest != predicted).tolist())
        splits[split] = SplitAssignment(split=split, sample_ids=table.sample_ids, rows=table.rows.copy(),
                                        user_ids=table.user_ids.copy(), cluster_ids=cluster_ids, predicted=predicted,
                                        sq_dist_1=first, sq_dist_2=second, numpy_mismatch_ids=mismatch)
    sizes = np.bincount(splits[FIT_SPLIT].cluster_ids, minlength=ccfg.n_clusters)
    empty = np.flatnonzero(sizes == 0).tolist()
    if empty:
        raise ClusterError(f"train clusters {empty} have no sample (sizes {sizes.tolist()}): their experts would "
                           "never be trained (19.6.5)")
    return ClusterFit(unit=ccfg.unit, km=km, fit_ids=fit_ids, z_fit=z_fit, fit_clusters=fit_clusters,
                      labels_mismatches=labels_mismatches, splits=types.MappingProxyType(splits),
                      fit_seconds=fit_seconds)


# ----------------------------------------------------------------------------
# Model, centres and assignment files (§19.2, §19.6.5)
# ----------------------------------------------------------------------------
def write_assignments(path: Path, assignment: SplitAssignment) -> str:
    """``{split}_assignments.csv``: ``sample_id, split, row, user_id, cluster_id, sq_dist_1, sq_dist_2``
    [SPEC §7]; ``split`` holds the contract value (val), distances ``'%.9g'``. Returns the sha256."""
    lines: list[Sequence[object]] = [ASSIGNMENT_FIELDS]
    for i, sid in enumerate(assignment.sample_ids):
        lines.append([sid, assignment.split, int(assignment.rows[i]), int(assignment.user_ids[i]),
                      int(assignment.cluster_ids[i]), FLOAT_FORMAT % float(assignment.sq_dist_1[i]),
                      FLOAT_FORMAT % float(assignment.sq_dist_2[i])])
    return _write_bytes(path, _csv_bytes(lines))


def read_assignments(path: Path, *, split: str, n_clusters: int) -> tuple[tuple[str, ...], np.ndarray, np.ndarray,
                                                                          np.ndarray]:
    """(sample_ids, rows, user_ids, cluster_ids) of an assignment CSV, checked: header, the split value of every
    row, ``sample_id = split:row``, strictly ascending rows, cluster ids 0..C-1, finite distances."""
    body = _read_csv(path, ASSIGNMENT_FIELDS)
    ids, rows, users, clusters = [], [], [], []
    for i, line in enumerate(body, start=2):
        sid, value_split, row, user, cluster, first, second = line
        try:
            row_i, user_i, cluster_i = int(row), int(user), int(cluster)
            distances = (float(first), float(second))
        except ValueError as exc:
            raise ClusterError(f"{path} line {i}: {exc}") from exc
        if value_split != split or sid != f"{split}:{row_i}" or not 0 <= cluster_i < n_clusters:
            raise ClusterError(f"{path} line {i}: {line[:5]} is not a {split} row with a cluster id 0..{n_clusters - 1}")
        if not all(math.isfinite(d) for d in distances) or (rows and row_i <= rows[-1]):
            raise ClusterError(f"{path} line {i}: rows must ascend strictly and distances be finite")
        ids.append(sid)
        rows.append(row_i)
        users.append(user_i)
        clusters.append(cluster_i)
    return (tuple(ids), np.array(rows, dtype=np.int64), np.array(users, dtype=np.int64),
            np.array(clusters, dtype=np.int64))


def write_user_assignments(path: Path, fit: ClusterFit, user_ids: Sequence[int]) -> str:
    """``train_user_assignments.csv`` (unit = user, §19.6.8): ``user_id, fit_sample_id, cluster_id`` per user."""
    lines: list[Sequence[object]] = [USER_ASSIGNMENT_FIELDS]
    for user, sid, cluster in zip(user_ids, fit.fit_ids, fit.fit_clusters.tolist()):
        lines.append([int(user), sid, int(cluster)])
    return _write_bytes(path, _csv_bytes(lines))


def read_user_assignments(path: Path, *, n_clusters: int) -> dict[int, tuple[str, int]]:
    """user_id -> (fit sample id, cluster id) of ``train_user_assignments.csv``."""
    out: dict[int, tuple[str, int]] = {}
    for i, (user, sid, cluster) in enumerate(_read_csv(path, USER_ASSIGNMENT_FIELDS), start=2):
        try:
            user_i, cluster_i = int(user), int(cluster)
        except ValueError as exc:
            raise ClusterError(f"{path} line {i}: {exc}") from exc
        if user_i in out or not sid.startswith(f"{FIT_SPLIT}:") or not 0 <= cluster_i < n_clusters:
            raise ClusterError(f"{path} line {i}: repeated user, non-train fit id or cluster outside 0..{n_clusters - 1}")
        out[user_i] = (sid, cluster_i)
    return out


def save_kmeans(path: Path, km: "KMeans") -> str:
    """The fitted KMeans as joblib (§19.2, [SPEC §7]); temporary name + ``os.replace``. Returns the sha256."""
    import joblib

    path = _new_file(path)
    tmp = Path(str(path) + ".tmp")
    joblib.dump(km, tmp)
    os.replace(tmp, path)
    return run.content_sha256(path)


def load_kmeans(path: Path) -> "KMeans":
    """Read the joblib model back (scikit-learn 1.2.2 only: unpickling across versions is not trusted)."""
    check_sklearn_version()
    import joblib
    from sklearn.cluster import KMeans

    path = Path(path)
    if not path.is_file():
        raise ClusterError(f"{path} not found")
    km = joblib.load(path)
    if not isinstance(km, KMeans) or not hasattr(km, "cluster_centers_"):
        raise ClusterError(f"{path} does not hold a fitted KMeans")
    return km


def write_centroids(path: Path, fit: ClusterFit) -> str:
    """``centroids.npz`` (§19.2): ``cluster_centers`` float32 [C, d], ``inertia``, ``n_iter``, ``fit_ids_sha256``."""
    km = fit.km
    return _write_npz(path, [("cluster_centers", np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)),
                             ("inertia", np.array(float(km.inertia_), dtype=np.float64)),
                             ("n_iter", np.array(int(km.n_iter_), dtype=np.int64)),
                             ("fit_ids_sha256", np.array(ids_sha256(fit.fit_ids), dtype=np.str_))])


def read_centroids(path: Path) -> dict[str, object]:
    """``centroids.npz`` back (``allow_pickle=False``), members checked."""
    path = Path(path)
    if not path.is_file():
        raise ClusterError(f"{path} not found")
    with np.load(path, allow_pickle=False) as npz:
        if sorted(npz.files) != sorted(CENTROID_KEYS):
            raise ClusterError(f"{path}: members {sorted(npz.files)} != {sorted(CENTROID_KEYS)}")
        centers = npz["cluster_centers"]
        out = {"cluster_centers": centers, "inertia": float(npz["inertia"]), "n_iter": int(npz["n_iter"]),
               "fit_ids_sha256": str(npz["fit_ids_sha256"][()])}
    if centers.dtype != np.float32 or centers.ndim != 2:
        raise ClusterError(f"{path}: cluster_centers must be float32 [C, d], got {centers.dtype} {list(centers.shape)}")
    return out

def verify_saved_model(out_dir: Path, fit: ClusterFit, tables: Mapping[str, RepresentationTable]) -> dict:
    """§19.6.5: the saved joblib model is read back and predicts every split again; the result must equal the
    CSV files on disk (``unit = user``: the user table for the fit windows and, through it, the train CSV), and
    the centres must equal ``centroids.npz`` and the fitted model bit for bit."""
    out = Path(out_dir)
    km = load_kmeans(out / kmeans_file_name(fit.n_clusters, fit.km.random_state))
    centers = np.ascontiguousarray(fit.km.cluster_centers_)
    stored = read_centroids(out / CENTROIDS_FILE)
    if (km.cluster_centers_.tobytes() != centers.tobytes()
            or np.ascontiguousarray(stored["cluster_centers"]).tobytes() != centers.tobytes()):
        raise ClusterError("the reloaded joblib model or centroids.npz differs from the fitted centres")
    checked: dict[str, int] = {}
    users = None
    if fit.unit == "user":
        users = read_user_assignments(out / USER_ASSIGNMENTS_FILE, n_clusters=fit.n_clusters)
        again = predict(km, fit.z_fit)
        if [cluster for _, cluster in users.values()] != again.tolist() or \
                [sid for sid, _ in users.values()] != list(fit.fit_ids):
            raise ClusterError("train_user_assignments.csv differs from the reloaded model's predict on the fit windows")
    for split in SPLITS:
        table = tables[split]
        ids, _, user_ids, clusters = read_assignments(out / assignment_file(split), split=split,
                                                      n_clusters=fit.n_clusters)
        if ids != table.sample_ids:
            raise ClusterError(f"{assignment_file(split)} lists other samples than the {split} z table")
        if users is not None and split == FIT_SPLIT:
            expected = np.array([users[int(u)][1] for u in user_ids.tolist()], dtype=np.int64)
        else:
            expected = predict(km, table.z)
        if not np.array_equal(clusters, expected):
            raise ClusterError(f"{assignment_file(split)}: {int((clusters != expected).sum())} rows differ from the "
                               "reloaded joblib model (19.6.5)")
        checked[SPEC_SPLIT_NAMES[split]] = len(ids)
    return {"passed": True, "kmeans_file": kmeans_file_name(fit.n_clusters, fit.km.random_state), "rows": checked}


# ----------------------------------------------------------------------------
# Comparison with the dynamic gate of an E2 export (§19.6.2, §19.6.6)
# ----------------------------------------------------------------------------
def load_dynamic_export(export_dir: Path, tables: Mapping[str, RepresentationTable], *, atol: float
                        ) -> tuple[dict[str, RepresentationTable], dict]:
    """The E2 export folder ``records/export/epoch{a}`` named by ``cluster.dynamic_export_dir`` (§19.6.2, §19.6.6).

    For each split ``representations_{train,validation,test}.npz`` (M12 ``load_npz``) must hold omega, say
    ``routing_mode = dynamic`` in its meta (and in the folder's ``manifest.json`` when there is one), list exactly
    the sample ids of the cluster z table, and agree on z within ``atol`` (max |dz| matched by sample_id): the
    KMeans input is then the dynamic gate input [SPEC §10]. Returns the export tables in the cluster order and
    the provenance block of the summary (path, checkpoint epoch, NPZ sha256, routing_mode, max |dz|).
    """
    path = Path(export_dir)
    if not path.is_dir():
        raise ClusterError(f"cluster.dynamic_export_dir {path} is not a folder")
    if not _positive_number(atol):
        raise ClusterError(f"z_crosscheck_atol must be a finite number > 0, got {atol!r}")
    epochs: set[object] = set()
    manifest_sha = None
    if (path / MANIFEST_FILE).is_file():
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("routing_mode") != "dynamic":
            raise ClusterError(f"{path / MANIFEST_FILE}: routing_mode {manifest.get('routing_mode')!r} != 'dynamic' "
                               "(the comparison needs the E2 dynamic export, 19.6.6)")
        epochs.add(manifest.get("epoch"))
        manifest_sha = run.content_sha256(path / MANIFEST_FILE)
    info: dict[str, Any] = {"path": str(path), "routing_mode": "dynamic", "atol": atol, "manifest_sha256": manifest_sha,
                            "npz_sha256": {}, "rows": {}, "z_max_abs_diff": {}}
    aligned: dict[str, RepresentationTable] = {}
    for split in SPLITS:
        file = path / representation_file(split)
        try:
            table = load_npz(file)
        except RecordsError as exc:
            raise ClusterError(f"dynamic export: {exc}") from exc
        mode = table.meta.get("routing_mode")
        if table.split != split or table.omega is None or mode != "dynamic":
            raise ClusterError(f"{file}: split {table.split!r}, omega {'present' if table.omega is not None else 'missing'}, "
                               f"routing_mode {mode!r}; the comparison needs the {split} table of a dynamic export")
        ours = tables[split]
        index = {sid: i for i, sid in enumerate(table.sample_ids)}
        missing = [sid for sid in ours.sample_ids if sid not in index]
        if missing or len(index) != len(ours.sample_ids):
            raise ClusterError(f"{file}: the sample set differs from the cluster {split} samples ({len(missing)} "
                               f"missing, first {missing[:3]}; {len(index)} vs {len(ours.sample_ids)} ids)")
        order = np.array([index[sid] for sid in ours.sample_ids], dtype=np.int64)
        z_other = table.z[order]
        if z_other.shape != ours.z.shape:
            raise ClusterError(f"{file}: z shape {list(z_other.shape)} != cluster z {list(ours.z.shape)}")
        diff = float(np.abs(z_other.astype(np.float64) - ours.z.astype(np.float64)).max())
        if not diff <= atol:
            raise ClusterError(f"{file}: max |dz| {diff:.6g} against the cluster z > cluster.z_crosscheck_atol {atol} "
                               "(the KMeans input would not be the dynamic gate input, [SPEC §10])")
        checkpoint = table.meta.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            epochs.add(checkpoint.get("epoch"))
        name = SPEC_SPLIT_NAMES[split]
        info["npz_sha256"][name] = run.content_sha256(file)
        info["rows"][name] = len(table)
        info["z_max_abs_diff"][name] = diff
        aligned[split] = RepresentationTable(split=split, sample_ids=ours.sample_ids, rows=ours.rows,
                                             user_ids=ours.user_ids, z=np.ascontiguousarray(z_other),
                                             omega=np.ascontiguousarray(table.omega[order]),
                                             argmax=np.ascontiguousarray(table.argmax[order]), meta=table.meta)
    epochs.discard(None)
    if len(epochs) > 1:
        raise ClusterError(f"{path}: the NPZ files and manifest name different checkpoint epochs {sorted(map(str, epochs))}")
    info["epoch"] = epochs.pop() if epochs else None
    return aligned, info


def dynamic_comparison(assignment: SplitAssignment, export: RepresentationTable, n_clusters: int) -> dict:
    """Cluster vs dynamic gate of one split (§19.6.6): omega mean and standard deviation (ddof 0) per cluster
    [C][K] (null for a cluster without samples), the cluster x dynamic-argmax contingency [C][K] (argmax = first
    index, the M12 rule), its row-normalised distribution and one NMI (``average_method="arithmetic"``), a summary
    that does not depend on the meaningless cluster numbers."""
    from sklearn.metrics import normalized_mutual_info_score

    if export.sample_ids != assignment.sample_ids or export.omega is None or export.argmax is None:
        raise ClusterError(f"{assignment.split}: the export table must hold omega for the same samples in order")
    labels = assignment.cluster_ids
    omega = export.omega.astype(np.float64)
    argmax = export.argmax.astype(np.int64)
    k = int(omega.shape[1])
    means: list[Optional[list[float]]] = []
    stds: list[Optional[list[float]]] = []
    contingency: list[list[int]] = []
    distribution: list[Optional[list[float]]] = []
    for c in range(n_clusters):
        mask = labels == c
        counts = np.bincount(argmax[mask], minlength=k)
        contingency.append(counts.tolist())
        if mask.any():
            means.append(omega[mask].mean(axis=0).tolist())
            stds.append(omega[mask].std(axis=0).tolist())
            distribution.append((counts / counts.sum()).tolist())
        else:
            means.append(None)
            stds.append(None)
            distribution.append(None)
    nmi = float(normalized_mutual_info_score(labels, argmax, average_method="arithmetic"))
    return {"n": int(len(labels)), "num_experts": k, "cluster_dynamic_gate_mean": means, "cluster_dynamic_gate_std": stds,
            "contingency_cluster_vs_dynamic_argmax": contingency, "cluster_dynamic_argmax_distribution": distribution,
            "nmi_cluster_vs_dynamic_argmax": nmi}


def draw_figures(out_dir: Path, fit: ClusterFit, train_gate_means: Optional[np.ndarray],
                 formats: Sequence[str]) -> dict[str, object]:
    """``figures/sasrec_pca_clusters`` (PCA of the fit set for the picture only, colour = cluster, every fit point)
    and, with a dynamic export, ``figures/cluster_dynamic_gate_heatmap`` (train mean dynamic gate per cluster,
    range [0, 1]) with the M12 functions [SPEC §8]. Returns the written file names per figure (or NOT_RUN)."""
    from .analysis import pca_2d, plot_cluster_gate_heatmap, plot_pca_clusters

    figures = Path(out_dir) / FIGURES_DIR
    fit_set = "train samples" if fit.unit == "sequence" else "last train window per user"
    plot_pca_clusters(figures / PCA_FIGURE, pca_2d(fit.z_fit), fit.fit_clusters, n_clusters=fit.n_clusters,
                      title=f"SASRec z, PCA of the fit set ({fit_set})", formats=formats)
    written: dict[str, object] = {PCA_FIGURE: [f"{FIGURES_DIR}/{PCA_FIGURE}.{fmt}" for fmt in formats]}
    if train_gate_means is None:
        written[HEATMAP_FIGURE] = NOT_RUN_DYNAMIC
    else:
        plot_cluster_gate_heatmap(figures / HEATMAP_FIGURE, train_gate_means,
                                  title="Mean dynamic gate per cluster (train)", formats=formats)
        written[HEATMAP_FIGURE] = [f"{FIGURES_DIR}/{HEATMAP_FIGURE}.{fmt}" for fmt in formats]
    return written

# ----------------------------------------------------------------------------
# Summary, manifest and the cluster command (§19.6.5, §19.6.6, §19.6.1)
# ----------------------------------------------------------------------------
def _section_dict(section: object) -> dict:
    return {f.name: _plain(getattr(section, f.name)) for f in dataclasses.fields(section)}


def _numpy_check(assignment: SplitAssignment) -> dict:
    """float64 cross-check of one split for the summary (§19.6.5): number of samples whose numpy nearest centre
    differs from ``predict``, their ids and (sq_dist_1, sq_dist_2), and the smallest gap sq_dist_2 - sq_dist_1."""
    listed = list(assignment.numpy_mismatch_ids[:_MAX_LISTED_IDS])
    position = {sid: i for i, sid in enumerate(assignment.sample_ids)} if listed else {}
    return {"count": len(assignment.numpy_mismatch_ids), "listed": len(listed), "sample_ids": listed,
            "sq_dist": [[float(assignment.sq_dist_1[position[sid]]), float(assignment.sq_dist_2[position[sid]])]
                        for sid in listed],
            "min_gap_sq_dist": float((assignment.sq_dist_2 - assignment.sq_dist_1).min())}


def len_seq_by_cluster(samples: Sequence[Sample], cluster_ids: np.ndarray, n_clusters: int) -> dict:
    """``train_len_seq_by_cluster`` (§19.6.6): counts of len_seq 3..10 x cluster (short histories may gather in a
    cluster that the 10-item evaluation samples rarely reach)."""
    lengths = list(range(MIN_LEN_SEQ, HISTORY_LEN + 1))
    counts = np.zeros((len(lengths), n_clusters), dtype=np.int64)
    for s, cluster in zip(samples, cluster_ids.tolist()):
        counts[len(s.history) - MIN_LEN_SEQ, cluster] += 1
    return {"len_seq": lengths, "cluster_id": list(range(n_clusters)), "counts": counts.tolist()}


def cluster_summary(cfg: "Config", fit: ClusterFit, samples: Mapping[str, Sequence[Sample]], *, scale: Mapping,
                    silhouette: Mapping, dynamic: Optional[Mapping], comparison: Optional[Mapping[str, Mapping]],
                    figures: Mapping, reload: Mapping, timings: Mapping) -> dict:
    """``cluster_summary.json`` content (§19.6.6, [SPEC §7]); the dynamic items are ``NOT_RUN`` without an export."""
    km, c = fit.km, fit.n_clusters
    summary: dict[str, Any] = {
        "unit": fit.unit, "n_clusters": c, "settings": _section_dict(cfg.cluster),
        "kmeans_params": _plain(km.get_params()), "versions": library_versions(),
        "fit": {"split": FIT_SPLIT, "rows": len(fit.fit_ids), "fit_ids_sha256": ids_sha256(fit.fit_ids),
                "inertia": float(km.inertia_), "n_iter": int(km.n_iter_), "openmp_threads": int(km._n_threads),
                "labels_vs_predict_mismatches": fit.labels_mismatches,
                "numpy_check_mismatches": sum(len(a.numpy_mismatch_ids) for a in fit.splits.values()),
                "seconds": fit.fit_seconds},
        "numpy_check": {SPEC_SPLIT_NAMES[s]: _numpy_check(a) for s, a in fit.splits.items()},
    }
    for split, assignment in fit.splits.items():
        sizes = np.bincount(assignment.cluster_ids, minlength=c)
        name = SPEC_SPLIT_NAMES[split]
        summary[f"{name}_cluster_size"] = sizes.tolist()
        summary[f"{name}_cluster_ratio"] = (sizes / sizes.sum()).tolist()
    summary["silhouette_score_train"] = silhouette.get("value")
    summary["silhouette"] = dict(silhouette)
    summary["train_len_seq_by_cluster"] = len_seq_by_cluster(samples[FIT_SPLIT], fit.splits[FIT_SPLIT].cluster_ids, c)
    summary["z_scale"] = dict(scale)
    summary["dynamic_export"] = dict(dynamic) if dynamic is not None else NOT_RUN_DYNAMIC
    for item in DYNAMIC_ITEMS:
        summary[item] = (NOT_RUN_DYNAMIC if comparison is None else
                         {SPEC_SPLIT_NAMES[s]: comparison[s][item] for s in SPLITS})
    summary["figures"] = dict(figures)
    summary["reload_check"] = dict(reload)
    summary["timings_s"] = dict(timings)
    summary["created_utc"] = run.utc_now().isoformat()
    return summary


def write_manifest(out_dir: Path, cfg: "Config", fit: ClusterFit, *, sasrec_info: Mapping, data_sha256: Mapping,
                   dynamic: Optional[Mapping]) -> str:
    """``manifest.json`` (§19.6.5): sha256 of every file of the folder, the SASRec state, the data files (the M0
    meta.json values), the data subset (with smoke_users and train_rows for smoke), fit split, rows and id
    sha256, the ``cluster`` section, library versions and ``threadpool_info()``."""
    from threadpoolctl import threadpool_info

    out = Path(out_dir)
    files = {p.relative_to(out).as_posix(): run.content_sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != MANIFEST_FILE}
    smoke = cfg.data.subset == "smoke"
    manifest = {
        "format": ARTIFACT_FORMAT, "unit": fit.unit, "n_clusters": fit.n_clusters, "fit_split": FIT_SPLIT,
        "fit_rows": len(fit.fit_ids), "fit_ids_sha256": ids_sha256(fit.fit_ids),
        "rows": {SPEC_SPLIT_NAMES[s]: len(a.sample_ids) for s, a in fit.splits.items()},
        "kmeans_file": kmeans_file_name(fit.n_clusters, cfg.cluster.random_state),
        "data": {"subset": cfg.data.subset, "smoke_users": list(cfg.data.smoke_users) if smoke else None,
                 "train_rows": list(cfg.data.train_rows) if smoke else None, "seed": cfg.run.seed,
                 "title_mode": cfg.data.title_mode, "sha256": dict(data_sha256)},
        "sasrec": dict(sasrec_info), "cluster_config": _section_dict(cfg.cluster),
        "z": {"source": "records.compute_representations(gate_fn=None)", "device": "cpu",
              "batch_size": cfg.records.export_batch_size, "num_threads": cfg.records.export_num_threads},
        "dynamic_export": dict(dynamic) if dynamic is not None else None,
        "versions": library_versions(), "threadpool_info": _plain(threadpool_info()),
        "created_utc": run.utc_now().isoformat(), "files": files,
    }
    return _write_json(out / MANIFEST_FILE, manifest)


def write_cluster_artifacts(cfg: "Config", out_dir: Path, tables: Mapping[str, RepresentationTable],
                            samples: Mapping[str, Sequence[Sample]], *, z_dim: int, sasrec_info: Mapping,
                            data_sha256: Mapping[str, Optional[str]], repo_root: Path,
                            timings: Optional[Mapping[str, float]] = None) -> dict:
    """Everything after z (§19.6.2-§19.6.6) into ``out_dir`` (``<run>/cluster/``); returns the summary.

    1. the z tables as ``representations_{train,validation,test}.npz`` (16.2 format, z only); 2. ``z_scale`` of
    the train z and, with ``cluster.dynamic_export_dir``, the E2 export check; 3. ``fit_and_assign``; 4. joblib
    model, ``centroids.npz`` and assignment CSVs, then the reload check; 5. silhouette of the fit set; 6. the
    dynamic comparison; 7. figures; 8. ``cluster_summary.json``; 9. ``manifest.json``.
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ClusterError(f"{out} is not empty; cluster artifacts are never overwritten (M0)")
    out.mkdir(parents=True, exist_ok=True)
    ccfg = cfg.cluster
    if _check_tables(tables, samples) != z_dim:
        raise ClusterError(f"z has {tables[FIT_SPLIT].dim} columns, the SASRec hidden size is {z_dim} (raw z, 19.6.3)")
    times = dict(timings or {})
    for split in SPLITS:
        save_npz(out / representation_file(split), tables[split])
    scale = z_scale(tables[FIT_SPLIT].z)
    export_tables, dynamic = None, None
    if ccfg.dynamic_export_dir is not None:
        export_tables, dynamic = load_dynamic_export(run.resolve_path(ccfg.dynamic_export_dir, repo_root), tables,
                                                     atol=ccfg.z_crosscheck_atol)
    fit = fit_and_assign(tables, samples, ccfg)
    times["fit"] = fit.fit_seconds
    logger.info("cluster: KMeans on %d %s rows (%s unit): inertia %.6g, n_iter %d, %.2f s; train sizes %s",
                len(fit.fit_ids), FIT_SPLIT, fit.unit, fit.km.inertia_, fit.km.n_iter_, fit.fit_seconds,
                np.bincount(fit.splits[FIT_SPLIT].cluster_ids, minlength=fit.n_clusters).tolist())
    save_kmeans(out / kmeans_file_name(fit.n_clusters, ccfg.random_state), fit.km)
    write_centroids(out / CENTROIDS_FILE, fit)
    if fit.unit == "user":
        user_of = {s.sample_id: s.user_id for s in samples[FIT_SPLIT]}
        write_user_assignments(out / USER_ASSIGNMENTS_FILE, fit, [user_of[sid] for sid in fit.fit_ids])
    for split in SPLITS:
        write_assignments(out / assignment_file(split), fit.splits[split])
    reload = verify_saved_model(out, fit, tables)
    silhouette = silhouette_train(fit.z_fit, fit.fit_clusters)
    times["silhouette"] = silhouette.get("seconds")
    logger.info("cluster: silhouette_score_train %s over %d fit rows (%s s)", silhouette.get("value"),
                len(fit.fit_ids), silhouette.get("seconds"))
    comparison, train_means = None, None
    if export_tables is not None:
        comparison = {s: dynamic_comparison(fit.splits[s], export_tables[s], fit.n_clusters) for s in SPLITS}
        train_means = np.array(comparison[FIT_SPLIT]["cluster_dynamic_gate_mean"], dtype=np.float64)
    start = time.perf_counter()
    figures = draw_figures(out, fit, train_means, cfg.records.figure_formats)
    times["figures"] = round(time.perf_counter() - start, 3)
    summary = cluster_summary(cfg, fit, samples, scale=scale, silhouette=silhouette, dynamic=dynamic,
                              comparison=comparison, figures=figures, reload=reload, timings=times)
    _write_json(out / SUMMARY_FILE, summary)
    write_manifest(out, cfg, fit, sasrec_info=sasrec_info, data_sha256=data_sha256, dynamic=dynamic)
    return summary


def run_cluster(cfg: "Config", run_dir: Path, *, repo_root: Path) -> None:
    """``main.py cluster`` (§19.6.1, §19.2): the artifacts in ``<run_dir>/cluster/`` from the converted SASRec.

    CPU only (§19.6.9). M1 samples (``load_dataset``), the frozen M3 encoder (``SASRecEncoder.from_config``: eval,
    fp32), z of each split with M12 ``compute_representations(gate_fn=None)`` and the ``records`` export settings
    (``export_num_threads`` 1, ``export_batch_size`` 1024; the same function and values as the M12 export, so z
    needs no E2 run, §19.6.2), then ``write_cluster_artifacts``. The data sha256 recorded are the M0
    ``run.input_hashes`` values of the four data files (the ``meta.json`` inputs).
    """
    check_sklearn_version()
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    out = run_dir / CLUSTER_DIR
    if out.exists():
        raise ClusterError(f"{out} exists; cluster artifacts are never overwritten (M0)")
    from .data import load_dataset
    from .sasrec import SASRecEncoder

    start = time.perf_counter()
    data = load_dataset(cfg, repo_root)
    encoder = SASRecEncoder.from_config(cfg, data.catalog, HISTORY_LEN)
    state_path = run.resolve_path(cfg.sasrec.state_path, repo_root)
    sasrec_info = {"state_path": str(state_path), "state_sha256": run.content_sha256(state_path),
                   "state_digest": encoder.state_digest}
    rc = cfg.records
    tables: dict[str, RepresentationTable] = {}
    for split in SPLITS:
        table = compute_representations(data.samples(split), encoder, None, num_experts=cfg.lora.num_experts,
                                        batch_size=rc.export_batch_size, num_threads=rc.export_num_threads,
                                        rowsum_atol=rc.gate_rowsum_atol)
        tables[split] = dataclasses.replace(table, meta={**table.meta, "source": "cluster", "sasrec": sasrec_info})
    z_seconds = round(time.perf_counter() - start, 3)
    logger.info("cluster: z of %s samples (d %d) in %.2f s", {s: len(t) for s, t in tables.items()},
                encoder.hidden_size, z_seconds)
    inputs = run.input_hashes(repo_root)
    write_cluster_artifacts(cfg, out, tables, {split: data.samples(split) for split in SPLITS},
                            z_dim=int(encoder.hidden_size), sasrec_info=sasrec_info,
                            data_sha256={rel: inputs.get(rel) for rel in DATA_FILES}, repo_root=repo_root,
                            timings={"data_and_z": z_seconds})
    logger.info("cluster: artifacts written to %s in %.1f s", out, time.perf_counter() - start)

# ----------------------------------------------------------------------------
# Loading for cluster_hard runs (§19.6.7 steps 1-2)
# ----------------------------------------------------------------------------
def _required_files(unit: str, n_clusters: int, random_state: object) -> list[str]:
    names = [kmeans_file_name(n_clusters, random_state), CENTROIDS_FILE, SUMMARY_FILE,
             *(assignment_file(s) for s in SPLITS), representation_file(FIT_SPLIT)]
    return names + ([USER_ASSIGNMENTS_FILE] if unit == "user" else [])


def _split_counts(ids: Sequence[str]) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    for sid in ids:
        split = sid.split(":", 1)[0] if isinstance(sid, str) else None
        if split not in counts:
            raise ClusterError(f"expected id {sid!r} is not a 'split:row' sample id")
        counts[split] += 1
    return counts


def _check_sample_ids(manifest: Mapping, keys: set, train_users: Mapping[str, int], expected: Sequence[str],
                      where: Path) -> None:
    """Sample ids against the manifest's data subset (§19.6.7 step 1, M1 5.6.7-5.6.8)."""
    wanted = set(expected)
    if len(wanted) != len(expected):
        raise ClusterError("expected_ids has duplicate sample ids")
    counts = _split_counts(expected)
    current_full = counts == dict(EXPECTED_SAMPLES)       # the full M1 data (66,126 / 94 / 95)
    data = manifest.get("data") if isinstance(manifest.get("data"), Mapping) else {}
    subset = data.get("subset")
    missing = sorted(wanted - keys)
    extra = sorted(keys - wanted)
    if subset == "full":
        if missing or (current_full and extra):
            raise ClusterError(f"{where}: sample ids of this run are not covered by the full-data artifacts "
                               f"({len(missing)} missing, first {missing[:3]}; {len(extra) if current_full else 0} extra)")
        return
    if subset != "smoke":
        raise ClusterError(f"{where}: manifest data.subset {subset!r} is neither full nor smoke")
    if current_full:
        raise ClusterError(f"{where}: artifacts of a smoke subset cannot be used by a full-data run (19.6.7)")
    if missing or extra:
        raise ClusterError(f"{where}: smoke artifacts must cover exactly this run's samples ({len(missing)} missing, "
                           f"first {missing[:3]}; {len(extra)} extra, first {extra[:3]}); smoke_users/train_rows differ?")
    smoke_users = list(data.get("smoke_users") or [])
    users = {train_users[sid] for sid in expected if sid.startswith(f"{FIT_SPLIT}:")}
    if smoke_users != [len(users), counts["val"], counts["test"]]:
        raise ClusterError(f"{where}: manifest smoke_users {smoke_users} != this run's [{len(users)}, {counts['val']}, "
                           f"{counts['test']}]")
    train_rows = list(data.get("train_rows") or [])
    if train_rows and train_rows != sorted(int(sid.split(":")[1]) for sid in expected if sid.startswith("train:")):
        raise ClusterError(f"{where}: manifest train_rows differ from this run's train samples")


def load_cluster_artifacts(artifact_dir: Path, *, expected_ids: Sequence[str], sasrec_sha256: str,
                           data_sha256: Mapping[str, str], ccfg: "ClusterConfig") -> ClusterArtifacts:
    """Read and check a ``cluster/`` folder before cluster-hard training, evaluation or export (§19.6.7 step 1).

    1. ``manifest.json``: format, the required files listed (joblib model, centres, the three assignment CSVs,
       the train z table, the summary; the user table for ``unit = user``) and every listed file present with
       its sha256 (one changed byte fails); 2. the converted SASRec state sha256 and the data file sha256 (M0
       ``meta.json`` inputs) equal this run's; 3. ``unit``, ``n_clusters`` and ``fit_split = train`` equal this
       config [SPEC §11]; 4. the assignment CSVs (header, split, ``split:row`` ids, ascending rows, cluster ids
       0..C-1, no id twice) and ``centroids.npz``; 5. ``expected_ids`` (all ids of the current three splits):
       artifacts of the full data serve a full run (equal sets) and a smoke run (``expected_ids`` ⊆ assignments,
       M1 smoke keeps rows and ids, 5.6.7); artifacts of a smoke subset serve only a smoke run with exactly the
       same samples (equal sets, and the manifest's smoke_users and train_rows agree with them), never a full
       run. The joblib model is not unpickled: no KMeans runs during training or evaluation [SPEC §10].
    """
    folder = Path(artifact_dir)
    path = folder / MANIFEST_FILE
    if not path.is_file():
        raise ClusterError(f"{folder}: no {MANIFEST_FILE}; not a cluster artifact folder (cluster.artifact_dir)")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ClusterError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != ARTIFACT_FORMAT or \
            not isinstance(manifest.get("files"), Mapping):
        got = manifest.get("format") if isinstance(manifest, dict) else type(manifest).__name__
        raise ClusterError(f"{path}: format {got!r} != {ARTIFACT_FORMAT!r} or no file list")
    files: Mapping[str, str] = manifest["files"]
    unit, n_clusters = manifest.get("unit"), manifest.get("n_clusters")
    random_state = (manifest.get("cluster_config") or {}).get("random_state")
    absent = [name for name in _required_files(unit, n_clusters, random_state) if name not in files]
    if absent:
        raise ClusterError(f"{path}: required artifacts {absent} are not in the manifest")
    for name, expected in files.items():
        file = folder / name
        if not file.is_file():
            raise ClusterError(f"{folder}: {name} listed in the manifest is missing")
        got = run.content_sha256(file)
        if got != expected:
            raise ClusterError(f"{folder}: {name} sha256 {got} != manifest {expected} (artifact changed)")
    stored_sasrec = (manifest.get("sasrec") or {}).get("state_sha256")
    if stored_sasrec != sasrec_sha256:
        raise ClusterError(f"{folder}: the clusters were made with SASRec state sha256 {stored_sasrec}, this run uses "
                           f"{sasrec_sha256} [SPEC §11]")
    for rel, sha in ((manifest.get("data") or {}).get("sha256") or {}).items():
        if data_sha256.get(rel) != sha:
            raise ClusterError(f"{folder}: data file {rel} sha256 {data_sha256.get(rel)} != artifacts {sha}")
    if not (manifest.get("data") or {}).get("sha256"):
        raise ClusterError(f"{path}: no data sha256 recorded")
    if (unit, n_clusters, manifest.get("fit_split")) != (ccfg.unit, ccfg.n_clusters, FIT_SPLIT):
        raise ClusterError(f"{folder}: artifacts unit/n_clusters/fit_split = {unit}/{n_clusters}/"
                           f"{manifest.get('fit_split')}, this config cluster.unit/n_clusters = {ccfg.unit}/"
                           f"{ccfg.n_clusters} with fit split {FIT_SPLIT}")
    assignments: dict[str, int] = {}
    train_users: dict[str, int] = {}
    for split in SPLITS:
        ids, _, users, clusters = read_assignments(folder / assignment_file(split), split=split, n_clusters=n_clusters)
        if len(ids) != (manifest.get("rows") or {}).get(SPEC_SPLIT_NAMES[split]):
            raise ClusterError(f"{folder}: {assignment_file(split)} has {len(ids)} rows, manifest rows differ")
        assignments.update(zip(ids, clusters.tolist()))
        if split == FIT_SPLIT:
            train_users = dict(zip(ids, users.tolist()))
    centers = read_centroids(folder / CENTROIDS_FILE)["cluster_centers"]
    if centers.shape[0] != n_clusters:
        raise ClusterError(f"{folder}: centroids.npz has {centers.shape[0]} centres, n_clusters {n_clusters}")
    _check_sample_ids(manifest, set(assignments), train_users, list(expected_ids), folder)
    artifacts = ClusterArtifacts(dir=folder, unit=unit, n_clusters=n_clusters, centers=centers,
                                 assignments=assignments, manifest=manifest)
    logger.info("cluster artifacts %s: unit %s, %d clusters, %d assignments, data subset %s", folder, unit, n_clusters,
                len(assignments), manifest["data"].get("subset"))
    return artifacts


def verify_live_encoder(artifacts: ClusterArtifacts, encoder: "SASRecEncoder", samples: Sequence[Sample], *,
                        device: torch.device, n: int, atol: float) -> dict:
    """Live z on the training device against the stored train z (§19.6.7 step 2, after ``build_model``).

    The first ``n`` samples of ``samples`` (the M1 train list of this run, row order) go through ``encoder`` on
    ``device`` (eval, fp32, TF32 off inside M3) and are compared with the same sample ids of
    ``artifacts.dir / "representations_train.npz"`` (re-checked against the manifest sha256, M12 ``load_npz``):
    the first ``n`` rows of that file when the run uses the artifacts' own samples. max |dz| must be <= ``atol``:
    the SASRec used in training is the one used for the clusters. ``n`` < 1 or > the number of train samples is a
    ``ValueError`` (§19.6.9). No KMeans inference. Returns the measurement (logged; M10 records it).
    """
    start = time.perf_counter()
    if not isinstance(artifacts, ClusterArtifacts):
        raise ClusterError(f"verify_live_encoder needs ClusterArtifacts, got {type(artifacts).__name__}")
    chosen = list(samples)
    if not _is_int(n) or not 1 <= n <= len(chosen):
        raise ClusterError(f"live check n = {n!r} must be an int in 1..{len(chosen)} (the train samples of this run)")
    if not _positive_number(atol):
        raise ClusterError(f"live check atol must be a finite number > 0, got {atol!r}")
    first = chosen[:int(n)]
    rows = [s.row for s in first]
    if any(not isinstance(s, Sample) or s.split != FIT_SPLIT for s in first) or any(b <= a for a, b in zip(rows, rows[1:])):
        raise ClusterError("verify_live_encoder takes the M1 train samples in row order")
    name = representation_file(FIT_SPLIT)
    path = artifacts.dir / name
    expected_sha = (artifacts.manifest.get("files") or {}).get(name)
    got_sha = run.content_sha256(path) if path.is_file() else None
    if got_sha is None or got_sha != expected_sha:
        raise ClusterError(f"{path}: sha256 {got_sha} != manifest {expected_sha} (stored z changed or missing)")
    table = load_npz(path)
    index = {sid: i for i, sid in enumerate(table.sample_ids)}
    ids = [s.sample_id for s in first]
    unknown = [sid for sid in ids if sid not in index]
    if unknown:
        raise ClusterError(f"{path}: no stored z for {len(unknown)} of the first {n} train samples (first {unknown[:3]})")
    stored = table.z[np.array([index[sid] for sid in ids], dtype=np.int64)]
    dev = torch.device(device)
    param = next(iter(encoder.parameters()), None)
    if param is None or not _same_device(param.device, dev):
        raise ClusterError(f"the encoder is on {None if param is None else param.device}, not on {dev}")
    if getattr(encoder, "state_digest", None) != (artifacts.manifest.get("sasrec") or {}).get("state_digest"):
        raise ClusterError("the encoder state digest differs from the SASRec digest of the cluster artifacts")
    hist_ids, len_seq = pad_histories([s.history for s in first], PAD_ID, HISTORY_LEN)
    with torch.no_grad():
        live = encoder(hist_ids.to(dev), len_seq.to(dev))
    live_cpu = live.detach().to("cpu", torch.float32).numpy()
    if live_cpu.shape != stored.shape:
        raise ClusterError(f"live z {list(live_cpu.shape)} != stored z {list(stored.shape)}")
    diff = float(np.abs(live_cpu.astype(np.float64) - stored.astype(np.float64)).max())
    result = {"passed": bool(diff <= atol), "n": int(n), "max_abs_diff": diff, "atol": atol, "device": str(dev),
              "train_npz": str(path), "train_npz_sha256": got_sha, "npz_prefix": ids == list(table.sample_ids[:n]),
              "first_sample_id": ids[0], "last_sample_id": ids[-1], "sample_ids_sha256": ids_sha256(ids),
              "state_digest": encoder.state_digest, "seconds": round(time.perf_counter() - start, 3)}
    logger.info("cluster live check %s: n=%d max_abs_diff=%.9g atol=%g device=%s",
                "passed" if result["passed"] else "FAILED", n, diff, atol, dev)
    if not result["passed"]:
        raise ClusterError(f"live z check failed: max |dz| {diff:.6g} > cluster.live_check_atol {atol} on {dev} over the "
                           f"first {n} train samples (the training SASRec differs from the clustering one, 19.6.7)")
    return result


__all__ = [
    "ClusterError", "CLUSTER_DIR", "ARTIFACT_FORMAT", "REQUIRED_SKLEARN", "UNITS", "FIT_SPLIT", "MANIFEST_FILE",
    "SUMMARY_FILE", "CENTROIDS_FILE", "USER_ASSIGNMENTS_FILE", "FIGURES_DIR", "PCA_FIGURE", "HEATMAP_FIGURE",
    "ASSIGNMENT_FIELDS", "USER_ASSIGNMENT_FIELDS", "CENTROID_KEYS", "FLOAT_FORMAT", "NOT_RUN_DYNAMIC", "DYNAMIC_ITEMS",
    "DATA_FILES", "kmeans_file_name", "representation_file", "assignment_file", "ids_sha256", "check_sklearn_version",
    "library_versions", "ClusterArtifacts", "parse_expert_mapping", "ClusterHardGate", "build_fit_set", "fit_kmeans",
    "predict", "nearest_two", "z_scale", "silhouette_train", "SplitAssignment", "ClusterFit", "fit_and_assign",
    "write_assignments", "read_assignments", "write_user_assignments", "read_user_assignments", "save_kmeans",
    "load_kmeans", "write_centroids", "read_centroids", "verify_saved_model", "load_dynamic_export",
    "dynamic_comparison", "draw_figures", "len_seq_by_cluster", "cluster_summary", "write_manifest",
    "write_cluster_artifacts", "run_cluster", "load_cluster_artifacts", "verify_live_encoder",
]
