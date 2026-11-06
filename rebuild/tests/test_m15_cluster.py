"""M15 SASRec 군집과 cluster-hard 배정 (DESIGN.md §19.8 #1-#11; test names of §17.6.9).

#12-#14 (VESSL_REQUIRED) are run-folder tests in ``tests/vessl/test_run_outputs.py``. Synthetic z rows come from
``seeding.np_rng(0, "test", <name>)`` (§17.6.2) as Gaussian blob mixtures, so that KMeans converges in a few
iterations; synthetic samples are M1 ``Sample`` objects. The real path runs on the converted SASRec asset: the
``smoke_cluster_dir`` fixture (M15 ``run_cluster`` on ``smoke_cpu``, smoke train 156) and, marked slow, the
``cluster`` command on the full data (#11). Tests that need full-size artifacts without the minute of the exact
silhouette replace ``cluster.silhouette_train`` with a stub (the silhouette itself runs in the smoke fixture and
in #11). Folders from fixtures are copied to ``tmp_path`` before anything is changed.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from ilora import cluster, records, run, seeding
from ilora.cluster import ClusterArtifacts, ClusterError, ClusterHardGate
from ilora.data import HISTORY_LEN, SPLITS, Sample
from ilora.records import RepresentationTable

K = 4                                   # lora.num_experts (paper_spec)
D = 64                                  # SASRec hidden size
SPEC = records.SPEC_SPLIT_NAMES         # file-name spelling train / validation / test
FAKE_SASREC = {"state_path": "synthetic", "state_sha256": "0" * 64, "state_digest": "0" * 64}
# matplotlib 3.7.1 calls pyparsing names that newer pyparsing releases mark deprecated (library internals)
pytestmark = pytest.mark.filterwarnings("ignore:'(parseString|resetCache|enablePackrat)' deprecated")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def blob_z(name: str, n: int, d: int = D, *, centers: int = 6, spread: float = 1.0, noise: float = 1.5) -> np.ndarray:
    """[n, d] float32 C-contiguous rows of a Gaussian blob mixture drawn from ``np_rng(0, "test", name)``."""
    rng = seeding.np_rng(0, "test", name)
    means = rng.normal(0.0, spread, size=(centers, d))
    labels = rng.integers(0, centers, size=n)
    return np.ascontiguousarray((means[labels] + rng.normal(0.0, noise, size=(n, d))).astype(np.float32))


def synthetic_samples(split: str, n: int, *, per_user: int = 1, user_offset: int = 0) -> tuple[Sample, ...]:
    """M1-shaped samples: rows 0, 2, 4, ..., history lengths 3..10 in turn, ``per_user`` windows per pseudo user."""
    return tuple(Sample(split=split, row=2 * i, user_id=user_offset + i // per_user,
                        history=tuple(range(3 + i % 8)), target=100 + i % 1500) for i in range(n))


def table_of(samples, z: np.ndarray, meta=None) -> RepresentationTable:
    return RepresentationTable(split=samples[0].split, sample_ids=tuple(s.sample_id for s in samples),
                               rows=np.array([s.row for s in samples], dtype=np.int64),
                               user_ids=np.array([s.user_id for s in samples], dtype=np.int64),
                               z=np.ascontiguousarray(z, dtype=np.float32), omega=None, argmax=None, meta=meta or {})


def tables_for(samples: dict, name: str, *, d: int = D, **blobs) -> dict:
    """z tables of ``samples`` (all splits) from one blob mixture."""
    sizes = [len(samples[split]) for split in SPLITS]
    z_all = blob_z(name, sum(sizes), d, **blobs)
    out, start = {}, 0
    for split, n in zip(SPLITS, sizes):
        out[split] = table_of(samples[split], z_all[start:start + n])
        start += n
    return out


def synthetic_data(name: str, sizes=(240, 40, 40), *, d: int = 16, per_user: int = 20):
    """Small data with four well separated blobs: (tables, samples)."""
    samples = {"train": synthetic_samples("train", sizes[0], per_user=per_user),
               "val": synthetic_samples("val", sizes[1], user_offset=1000),
               "test": synthetic_samples("test", sizes[2], user_offset=2000)}
    return tables_for(samples, name, d=d, centers=4, spread=6.0, noise=0.5), samples


def data_samples(data) -> dict:
    return {split: data.samples(split) for split in SPLITS}


def all_ids(data) -> list[str]:
    return [s.sample_id for split in SPLITS for s in data.samples(split)]


def data_sha(repo_root: Path) -> dict:
    return {rel: sha for rel, sha in run.input_hashes(repo_root).items() if rel in cluster.DATA_FILES}


def stub_silhouette(z_fit, labels) -> dict:
    return {"value": 0.0, "n": int(len(labels)), "metric": "euclidean", "sample_size": None, "seconds": 0.0,
            "stub": "test stub (the exact value is computed in smoke_cluster_dir and M15-11)"}


def copy_artifacts(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def flip_byte(path: Path, offset: int = -2) -> None:
    data = bytearray(path.read_bytes())
    data[offset % len(data)] ^= 0x01
    path.write_bytes(bytes(data))


def load_kwargs(cfg, data, state_sha256: str, repo_root: Path) -> dict:
    return {"expected_ids": all_ids(data), "sasrec_sha256": state_sha256, "data_sha256": run.input_hashes(repo_root),
            "ccfg": cfg.cluster}


# ----------------------------------------------------------------------------------------------- #1
MAPPING_CASES = [
    ("identity", "0:0,1:1,2:2,3:3", (0, 1, 2, 3)),
    ("reversed", "0:3,1:2,2:1,3:0", (3, 2, 1, 0)),
    ("missing_cluster", "0:0,1:1,2:2", "clusters [3] have no expert"),
    ("duplicate_cluster", "0:0,1:1,2:2,3:3,2:0", "'2:0' repeats cluster 2"),
    ("expert_4", "0:0,1:1,2:2,3:4", "'3:4' names expert 4"),
    ("negative", "0:0,1:1,2:2,3:-1", "'3:-1'"),
    ("empty_entry", "0:0,,1:1,2:2,3:3", "an empty entry"),
    ("space", "0:0, 1:1,2:2,3:3", "' 1:1'"),
    ("dash", "0-0,1:1,2:2,3:3", "'0-0'"),
    ("trailing_newline", "0:0,1:1,2:2,3:3\n", "'3:3\\n'"),
    ("non_ascii_digit", "٠:0,1:1,2:2,3:3", "'٠:0'"),
]


@pytest.mark.criterion("M15-1")
@pytest.mark.parametrize("text,expected", [case[1:] for case in MAPPING_CASES], ids=[case[0] for case in MAPPING_CASES])
def test_parse_mapping(text, expected):
    """Two accepted mappings; nine rejections, each naming the offending token (§19.6.7)."""
    if isinstance(expected, tuple):
        assert cluster.parse_expert_mapping(text, n_clusters=4, num_experts=K) == expected
        return
    with pytest.raises(ClusterError) as err:
        cluster.parse_expert_mapping(text, n_clusters=4, num_experts=K)
    assert expected in str(err.value)


# ----------------------------------------------------------------------------------------------- #2
@pytest.mark.criterion("M15-2")
def test_cluster_hard_gate(tmp_path):
    """One-hot omega: shape, float32, 0/1, row sums exactly 1, expert = mapping[cluster], requested device, lookups;
    an unknown id raises KeyError with the id and the artifact folder."""
    ids = [f"train:{r}" for r in range(8)] + [f"val:{r}" for r in range(4)] + [f"test:{r}" for r in range(4)]
    clusters = {sid: i % 4 for i, sid in enumerate(ids)}
    artifacts = ClusterArtifacts(dir=tmp_path, unit="sequence", n_clusters=4, centers=np.zeros((4, 8), np.float32),
                                 assignments=clusters, manifest={"format": cluster.ARTIFACT_FORMAT})
    with pytest.raises(TypeError):
        artifacts.assignments["train:0"] = 3                     # read-only view
    assert not artifacts.centers.flags.writeable
    mapping = cluster.parse_expert_mapping("0:3,1:2,2:1,3:0", n_clusters=4, num_experts=K)
    gate = ClusterHardGate(artifacts, mapping, K)
    assert gate.lookups == 0 and gate.unused_experts == ()
    batch = ids[3:13]
    for device in ("cpu", torch.device("cpu")):
        before = gate.lookups
        omega = gate(batch, device)
        assert omega.shape == (len(batch), K) and omega.dtype == torch.float32
        assert omega.device == torch.device(device)
        assert bool(((omega == 0) | (omega == 1)).all())
        assert torch.equal(omega.sum(dim=1), torch.ones(len(batch)))
        assert omega.argmax(dim=1).tolist() == [mapping[clusters[sid]] for sid in batch]
        assert gate.lookups == before + len(batch)
    before = gate.lookups
    with pytest.raises(KeyError) as err:
        gate(["train:0", "val:999"], "cpu")
    assert "'val:999'" in err.value.args[0] and str(tmp_path) in err.value.args[0]
    assert gate.lookups == before                               # nothing counted for a failed lookup
    with pytest.raises(ClusterError):
        gate([], "cpu")
    for bad in ((0, 1, 2), (0, 1, 2, 4)):
        with pytest.raises(ClusterError):
            ClusterHardGate(artifacts, bad, K)
    # two clusters on one expert is allowed; the unused expert is logged (19.6.7)
    seen: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = seen.append
    log = logging.getLogger("ilora.cluster")
    log.addHandler(handler)
    try:
        shared = ClusterHardGate(artifacts, cluster.parse_expert_mapping("0:0,1:0,2:1,3:2", n_clusters=4,
                                                                         num_experts=K), K)
    finally:
        log.removeHandler(handler)
    assert shared.unused_experts == (3,)
    assert any("[3]" in record.getMessage() for record in seen)
    assert shared(["train:1"], "cpu").argmax(dim=1).tolist() == [0]

# ----------------------------------------------------------------------------------------------- #3
@pytest.mark.criterion("M15-3")
def test_fit_uses_train_only(full_data, profile_cfg, monkeypatch):
    """build_fit_set(sequence) = the M1 train list in order; a validation sample in it fails; KMeans.fit runs once on
    [train rows, d]; validation and test pass through predict only (19.6.3)."""
    from sklearn.cluster import KMeans

    samples = data_samples(full_data)
    tables = tables_for(samples, "fit_train_only")
    train_ids = tuple(s.sample_id for s in full_data.train)
    z_fit, fit_ids = cluster.build_fit_set(tables["train"], samples["train"], unit="sequence")
    assert fit_ids == train_ids and len(fit_ids) == 66_126
    assert z_fit.dtype == np.float32 and z_fit.shape == (66_126, D) and z_fit.flags.c_contiguous
    assert np.array_equal(z_fit, tables["train"].z)
    mixed = list(full_data.train)
    mixed[100] = full_data.val[0]
    with pytest.raises(ClusterError, match="not M1 train samples"):
        cluster.build_fit_set(tables["train"], mixed, unit="sequence")
    with pytest.raises(ClusterError, match="train RepresentationTable"):
        cluster.build_fit_set(tables["val"], samples["val"], unit="sequence")
    with pytest.raises(ClusterError, match="ascending row order"):
        cluster.build_fit_set(tables["train"], list(reversed(full_data.train)), unit="sequence")
    fits, predicts = [], []
    original_fit, original_predict = KMeans.fit, KMeans.predict

    def spy_fit(self, X, y=None, sample_weight=None):
        fits.append(X)
        return original_fit(self, X, y, sample_weight)

    def spy_predict(self, X, sample_weight=None):
        predicts.append(X)
        return original_predict(self, X, sample_weight)

    monkeypatch.setattr(KMeans, "fit", spy_fit)
    monkeypatch.setattr(KMeans, "predict", spy_predict)
    fit = cluster.fit_and_assign(tables, samples, profile_cfg("smoke_cpu").cluster)
    assert len(fits) == 1
    assert fits[0].shape == (len(full_data.train), D) and np.array_equal(fits[0], tables["train"].z)
    for split in ("val", "test"):
        assert not any(np.shares_memory(x, tables[split].z) for x in fits)
        assert any(x is tables[split].z for x in predicts)
        assert np.array_equal(fit.splits[split].cluster_ids, fit.splits[split].predicted)
    assert fit.fit_ids == train_ids and fit.km.labels_.shape == (66_126,)


# ----------------------------------------------------------------------------------------------- #4
@pytest.mark.criterion("M15-4")
def test_fit_deterministic_one_thread(tmp_path, profile_cfg, monkeypatch, repo_root):
    """Synthetic 66,126 x 64 z: two fits with fit_num_threads 1 are bit-identical; the saved joblib model reloaded
    predicts the CSV assignments; centroids.npz holds cluster_centers_ (19.6.4, 19.6.5)."""
    cfg = profile_cfg("smoke_cpu")
    assert cfg.cluster.fit_num_threads == 1
    z = blob_z("z", 66_126)
    km1 = cluster.fit_kmeans(z, cfg.cluster)
    km2 = cluster.fit_kmeans(z, cfg.cluster)
    assert km1._n_threads == 1
    assert km1.cluster_centers_.tobytes() == km2.cluster_centers_.tobytes()
    assert km1.labels_.tobytes() == km2.labels_.tobytes()
    assert (km1.inertia_, km1.n_iter_) == (km2.inertia_, km2.n_iter_)
    samples = {"train": synthetic_samples("train", 66_126, per_user=90),
               "val": synthetic_samples("val", 94, user_offset=754), "test": synthetic_samples("test", 95, user_offset=848)}
    extra = blob_z("z_eval", 189)
    tables = {"train": table_of(samples["train"], z), "val": table_of(samples["val"], extra[:94]),
              "test": table_of(samples["test"], extra[94:])}
    monkeypatch.setattr(cluster, "silhouette_train", stub_silhouette)
    out = tmp_path / "cluster"
    cluster.write_cluster_artifacts(cfg, out, tables, samples, z_dim=D, sasrec_info=FAKE_SASREC,
                                    data_sha256=data_sha(repo_root), repo_root=repo_root)
    km = cluster.load_kmeans(out / "kmeans_k4_seed1234.joblib")
    for split in SPLITS:
        ids, _, _, clusters = cluster.read_assignments(out / f"{SPEC[split]}_assignments.csv", split=split, n_clusters=4)
        assert ids == tables[split].sample_ids
        assert np.array_equal(clusters, km.predict(tables[split].z).astype(np.int64)), split
    with np.load(out / "centroids.npz", allow_pickle=False) as npz:
        assert npz["cluster_centers"].dtype == np.float32
        assert npz["cluster_centers"].tobytes() == km.cluster_centers_.tobytes() == km1.cluster_centers_.tobytes()
        assert float(npz["inertia"]) == km1.inertia_ and int(npz["n_iter"]) == km1.n_iter_
        assert str(npz["fit_ids_sha256"][()]) == cluster.ids_sha256(tables["train"].sample_ids)


# ----------------------------------------------------------------------------------------------- #5
@pytest.mark.criterion("M15-5")
def test_predict_input_checks(profile_cfg):
    """float64, strided, Fortran-order, wrong width, NaN and tensor inputs are refused with this module's message."""
    ccfg = profile_cfg("smoke_cpu").cluster
    z = blob_z("predict_checks", 400, 8, centers=4, spread=6.0, noise=0.5)
    km = cluster.fit_kmeans(z, ccfg)
    labels = cluster.predict(km, z)
    assert labels.dtype == np.int64 and labels.shape == (400,)
    with pytest.raises(ValueError, match="Buffer dtype mismatch"):    # the library failure the check replaces (19.6.5)
        km.predict(z.astype(np.float64))
    nan = z.copy()
    nan[7, 3] = np.nan
    bad = {"float64": (z.astype(np.float64), "float32"), "strided": (z[::2], "C-contiguous"),
           "fortran": (np.asfortranarray(z), "C-contiguous"), "width": (np.ascontiguousarray(z[:, :7]), "columns"),
           "nan": (nan, "NaN"), "tensor": (torch.from_numpy(z), "numpy array")}
    for name, (value, message) in bad.items():
        with pytest.raises(ClusterError, match=message):
            cluster.predict(km, value)
    with pytest.raises(ClusterError, match="float32"):
        cluster.fit_kmeans(z.astype(np.float64), ccfg)


@pytest.mark.criterion("M15-5")
def test_sklearn_version_pinned(monkeypatch, profile_cfg, repo_root, tmp_path):
    """A scikit-learn other than 1.2.2 (1.2.1 simulated) stops fit, predict and the cluster command (19.6.4)."""
    import sklearn

    cfg = profile_cfg("smoke_cpu")
    z = blob_z("version_pin", 200, 8, centers=4, spread=6.0, noise=0.5)
    km = cluster.fit_kmeans(z, cfg.cluster)
    assert cluster.check_sklearn_version() == "1.2.2"
    monkeypatch.setattr(sklearn, "__version__", "1.2.1")
    for call in (cluster.check_sklearn_version, lambda: cluster.fit_kmeans(z, cfg.cluster),
                 lambda: cluster.predict(km, z)):
        with pytest.raises(ClusterError, match="scikit-learn 1.2.1 != 1.2.2"):
            call()
    with pytest.raises(ClusterError, match="1.2.1"):
        cluster.run_cluster(cfg, tmp_path, repo_root=repo_root)
    assert not (tmp_path / "cluster").exists()
    assert any("sklearn 1.2.1" in problem for problem in run.library_version_problems())    # M0 check-env pin


# ----------------------------------------------------------------------------------------------- #6
@pytest.mark.criterion("M15-6")
def test_convergence_warning_is_error(profile_cfg):
    """Three distinct points and n_clusters 4: the promoted ConvergenceWarning stops the fit (19.6.4).

    The points have small integer coordinates and each appears 8 times, so every mean and distance is exact in
    float32 and copies of one point always get one label. With non-exact values two coincident centres can split
    the copies of a point by rounding (measured: 8-dim normal points x 10 ended with 4 labels and no warning).
    """
    from sklearn.exceptions import ConvergenceWarning

    ccfg = profile_cfg("smoke_cpu").cluster
    points = np.array([[0, 0, 0, 0], [3, 0, 0, 0], [0, 3, 0, 0]], dtype=np.float32)
    z = np.ascontiguousarray(np.repeat(points, 8, axis=0))
    with pytest.raises(ClusterError, match="ConvergenceWarning") as err:
        cluster.fit_kmeans(z, ccfg)
    assert isinstance(err.value.__cause__, ConvergenceWarning)
    assert "distinct clusters (3)" in str(err.value)
    four = np.ascontiguousarray(np.repeat(np.vstack([points, points[:1] + 5.0]), 10, axis=0))
    assert sorted(set(cluster.fit_kmeans(four, ccfg).labels_.tolist())) == [0, 1, 2, 3]      # control


# ----------------------------------------------------------------------------------------------- #7
@pytest.mark.criterion("M15-7")
def test_user_level_fit_set(full_data, profile_cfg, repo_root, tmp_path):
    """unit = user on the M1 structure: 754 fit rows = the largest train row of each user block, every train sample
    takes its user's cluster, validation and test are predicted (19.6.8); the artifacts add
    train_user_assignments.csv and are loaded only with cluster.unit = user."""
    ccfg = profile_cfg("smoke_cpu", "cluster.unit=user").cluster
    samples = data_samples(full_data)
    tables = tables_for(samples, "user_level")
    z_fit, fit_ids = cluster.build_fit_set(tables["train"], samples["train"], unit="user")
    last: dict[int, int] = {}
    for s in full_data.train:
        last[s.user_id] = max(last.get(s.user_id, -1), s.row)
    assert len(fit_ids) == len(last) == 754
    assert fit_ids == tuple(f"train:{last[u]}" for u in sorted(last))
    index = {sid: i for i, sid in enumerate(tables["train"].sample_ids)}
    assert np.array_equal(z_fit, tables["train"].z[[index[sid] for sid in fit_ids]])
    fit = cluster.fit_and_assign(tables, samples, ccfg)
    assert fit.unit == "user" and fit.fit_ids == fit_ids and fit.km.labels_.shape == (754,)
    user_cluster = dict(zip(sorted(last), fit.fit_clusters.tolist()))
    assert fit.splits["train"].cluster_ids.tolist() == [user_cluster[s.user_id] for s in full_data.train]
    for split in ("val", "test"):
        assert np.array_equal(fit.splits[split].cluster_ids, cluster.predict(fit.km, tables[split].z))
    cfg = profile_cfg("smoke_cpu", "data.subset=full", "cluster.unit=user")
    out = tmp_path / "cluster"
    cluster.write_cluster_artifacts(cfg, out, tables, samples, z_dim=D, sasrec_info=FAKE_SASREC,
                                    data_sha256=data_sha(repo_root), repo_root=repo_root)
    users = cluster.read_user_assignments(out / "train_user_assignments.csv", n_clusters=4)
    assert list(users) == sorted(last) and [sid for sid, _ in users.values()] == list(fit_ids)
    assert [c for _, c in users.values()] == fit.fit_clusters.tolist()
    train_csv = cluster.read_assignments(out / "train_assignments.csv", split="train", n_clusters=4)[3]
    assert train_csv.tolist() == [users[s.user_id][1] for s in full_data.train]
    kwargs = {"expected_ids": all_ids(full_data), "sasrec_sha256": FAKE_SASREC["state_sha256"],
              "data_sha256": run.input_hashes(repo_root)}
    artifacts = cluster.load_cluster_artifacts(out, ccfg=cfg.cluster, **kwargs)
    first = full_data.train[0]
    assert artifacts.unit == "user" and artifacts.assignments[first.sample_id] == users[first.user_id][1]
    with pytest.raises(ClusterError, match="unit"):
        cluster.load_cluster_artifacts(out, ccfg=profile_cfg("smoke_cpu").cluster, **kwargs)

# ----------------------------------------------------------------------------------------------- #8
def write_export(folder: Path, tables: dict, name: str, *, mode: str = "dynamic", epoch: int = 2,
                 z_change=None) -> dict:
    """An E2 export imitation: ``representations_{train,validation,test}.npz`` with omega (16.2 format)."""
    rng = seeding.np_rng(0, "test", name)
    written = {}
    for split in SPLITS:
        table = tables[split]
        logits = rng.normal(0.0, 1.5, size=(len(table), K))
        omega = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)).astype(np.float32)
        z = table.z.copy()
        if z_change is not None and z_change[0] == split:
            z[z_change[1], z_change[2]] += np.float32(z_change[3])
        written[split] = dataclasses.replace(table, z=z, omega=omega, argmax=np.argmax(omega, axis=1).astype(np.int64),
                                             meta={"source": "export", "routing_mode": mode, "checkpoint": {"epoch": epoch}})
        records.save_npz(folder / f"representations_{SPEC[split]}.npz", written[split])
    return written


@pytest.mark.criterion("M15-8")
def test_summary_synthetic(tmp_path, profile_cfg, repo_root):
    """Sizes sum to the split counts, ratios to 1, contingency to n, cluster gate means have row sums 1 (1e-6),
    NMI in [0, 1]; without dynamic_export_dir the dynamic items are NOT_RUN and no heatmap is drawn (19.6.6)."""
    tables, samples = synthetic_data("summary")
    sizes = {SPEC[split]: len(tables[split]) for split in SPLITS}
    out = tmp_path / "plain" / "cluster"
    cluster.write_cluster_artifacts(profile_cfg("smoke_cpu"), out, tables, samples, z_dim=16, sasrec_info=FAKE_SASREC,
                                    data_sha256=data_sha(repo_root), repo_root=repo_root)
    plain = json.loads((out / "cluster_summary.json").read_text(encoding="utf-8"))
    for name, n in sizes.items():
        assert sum(plain[f"{name}_cluster_size"]) == n and len(plain[f"{name}_cluster_size"]) == 4
        assert abs(sum(plain[f"{name}_cluster_ratio"]) - 1.0) <= 1e-12
    table = plain["train_len_seq_by_cluster"]
    assert table["len_seq"] == list(range(3, 11))
    assert [sum(col) for col in zip(*table["counts"])] == plain["train_cluster_size"]
    assert plain["fit"]["split"] == "train" and plain["fit"]["rows"] == sizes["train"]
    assert -1.0 <= plain["silhouette_score_train"] <= 1.0 and plain["silhouette"]["sample_size"] is None
    assert plain["z_scale"]["d"] == 16 and plain["z_scale"]["per_dim_std"]["min"] > 0
    for name in sizes:                                  # numpy float64 cross-check of predict (19.6.5)
        block = plain["numpy_check"][name]
        assert block["count"] == len(block["sample_ids"]) == len(block["sq_dist"]) and block["min_gap_sq_dist"] >= 0.0
    centers = cluster.read_centroids(out / "centroids.npz")["cluster_centers"].astype(np.float64)
    with open(out / "validation_assignments.csv", "r", encoding="utf-8", newline="") as f:
        lines = list(csv.DictReader(f))
    distances = np.sort(((tables["val"].z.astype(np.float64)[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
    assert [line["sq_dist_1"] for line in lines] == ["%.9g" % v for v in distances[:, 0]]
    assert [line["sq_dist_2"] for line in lines] == ["%.9g" % v for v in distances[:, 1]]
    assert plain["dynamic_export"] == cluster.NOT_RUN_DYNAMIC
    assert all(plain[item] == cluster.NOT_RUN_DYNAMIC for item in cluster.DYNAMIC_ITEMS)
    assert plain["figures"]["cluster_dynamic_gate_heatmap"] == cluster.NOT_RUN_DYNAMIC
    assert (out / "figures" / "sasrec_pca_clusters.png").is_file()
    assert not list((out / "figures").glob("cluster_dynamic_gate_heatmap.*"))
    export = tmp_path / "records" / "export" / "epoch2"
    written = write_export(export, tables, "summary_export")
    cfg = profile_cfg("smoke_cpu", f"cluster.dynamic_export_dir={json.dumps(export.as_posix())}")
    out2 = tmp_path / "dynamic" / "cluster"
    cluster.write_cluster_artifacts(cfg, out2, tables, samples, z_dim=16, sasrec_info=FAKE_SASREC,
                                    data_sha256=data_sha(repo_root), repo_root=repo_root)
    summary = json.loads((out2 / "cluster_summary.json").read_text(encoding="utf-8"))
    assert summary["dynamic_export"]["routing_mode"] == "dynamic" and summary["dynamic_export"]["epoch"] == 2
    assert summary["dynamic_export"]["z_max_abs_diff"] == {name: 0.0 for name in sizes}
    for split in SPLITS:
        name = SPEC[split]
        size = summary[f"{name}_cluster_size"]
        contingency = summary["contingency_cluster_vs_dynamic_argmax"][name]
        assert sum(map(sum, contingency)) == sizes[name] and [sum(row) for row in contingency] == size
        for c, (mean, std, dist) in enumerate(zip(summary["cluster_dynamic_gate_mean"][name],
                                                  summary["cluster_dynamic_gate_std"][name],
                                                  summary["cluster_dynamic_argmax_distribution"][name])):
            if size[c] == 0:
                assert mean is None and std is None and dist is None
                continue
            assert abs(sum(mean) - 1.0) <= 1e-6 and min(std) >= 0.0 and abs(sum(dist) - 1.0) <= 1e-12
        assert 0.0 <= summary["nmi_cluster_vs_dynamic_argmax"][name] <= 1.0
    labels = np.array(cluster.read_assignments(out2 / "train_assignments.csv", split="train", n_clusters=4)[3])
    omega = written["train"].omega.astype(np.float64)
    expected = [omega[labels == c].mean(axis=0).tolist() for c in range(4)]
    assert np.allclose(summary["cluster_dynamic_gate_mean"]["train"], expected, rtol=0, atol=1e-12)
    assert (out2 / "figures" / "cluster_dynamic_gate_heatmap.png").is_file()
    assert json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))["dynamic_export"]["epoch"] == 2


# ----------------------------------------------------------------------------------------------- #9
LOAD_REJECTS = ["csv_byte", "sasrec_sha256", "ids_missing", "ids_extra", "n_clusters", "smoke_for_full"]


@pytest.mark.criterion("M15-9")
@pytest.mark.parametrize("case", LOAD_REJECTS)
def test_load_artifacts_rejects(case, smoke_cluster_dir, cpu_cfg, smoke_data, full_data, sasrec_state_path,
                                repo_root, profile_cfg, tmp_path):
    """A changed CSV byte, another SASRec state, a missing or extra sample id, another n_clusters and smoke artifacts
    for a full run are each refused (19.6.7 step 1)."""
    folder = copy_artifacts(smoke_cluster_dir, tmp_path / "cluster")
    kwargs = load_kwargs(cpu_cfg, smoke_data, run.content_sha256(sasrec_state_path), repo_root)
    assert len(cluster.load_cluster_artifacts(folder, **kwargs).assignments) == len(kwargs["expected_ids"])   # control
    if case == "csv_byte":
        flip_byte(folder / "validation_assignments.csv")
        match = "validation_assignments.csv sha256"
    elif case == "sasrec_sha256":
        kwargs["sasrec_sha256"] = "1" * 64
        match = "SASRec state sha256"
    elif case == "ids_missing":
        smoke = set(kwargs["expected_ids"])
        kwargs["expected_ids"] = kwargs["expected_ids"] + [next(s.sample_id for s in full_data.train
                                                                if s.sample_id not in smoke)]
        match = "1 missing"
    elif case == "ids_extra":
        kwargs["expected_ids"] = kwargs["expected_ids"][:-1]
        match = "1 extra"
    elif case == "n_clusters":
        kwargs["ccfg"] = profile_cfg("smoke_cpu", "cluster.n_clusters=5").cluster
        match = "n_clusters"
    else:
        kwargs["expected_ids"] = all_ids(full_data)
        match = "smoke subset cannot be used by a full-data run"
    with pytest.raises(ClusterError, match=match):
        cluster.load_cluster_artifacts(folder, **kwargs)


@pytest.mark.criterion("M15-9")
def test_full_artifacts_for_smoke(full_data, smoke_data, profile_cfg, repo_root, tmp_path, monkeypatch):
    """Artifacts of the full data serve a smoke run (smoke ids are a subset of the assignments) and a full run."""
    monkeypatch.setattr(cluster, "silhouette_train", stub_silhouette)
    cfg = profile_cfg("smoke_cpu", "data.subset=full")
    samples = data_samples(full_data)
    out = tmp_path / "cluster"
    cluster.write_cluster_artifacts(cfg, out, tables_for(samples, "full_artifacts"), samples, z_dim=D,
                                    sasrec_info=FAKE_SASREC, data_sha256=data_sha(repo_root), repo_root=repo_root)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data"]["subset"] == "full" and manifest["data"]["smoke_users"] is None
    assert manifest["fit_rows"] == 66_126 and manifest["rows"] == {"train": 66_126, "validation": 94, "test": 95}
    kwargs = {"sasrec_sha256": FAKE_SASREC["state_sha256"], "data_sha256": run.input_hashes(repo_root),
              "ccfg": cfg.cluster}
    smoke_ids = all_ids(smoke_data)
    artifacts = cluster.load_cluster_artifacts(out, expected_ids=smoke_ids, **kwargs)
    assert set(smoke_ids) < set(artifacts.assignments) and len(artifacts.assignments) == 66_315
    gate = ClusterHardGate(artifacts, cluster.parse_expert_mapping("0:0,1:1,2:2,3:3", n_clusters=4, num_experts=K), K)
    assert gate(smoke_ids, "cpu").argmax(dim=1).tolist() == [artifacts.assignments[sid] for sid in smoke_ids]
    assert len(cluster.load_cluster_artifacts(out, expected_ids=all_ids(full_data), **kwargs).assignments) == 66_315
    with pytest.raises(ClusterError, match="1 missing"):
        cluster.load_cluster_artifacts(out, expected_ids=smoke_ids + ["train:999999"], **kwargs)


def rewrite_train_z(folder: Path, row: int, delta: float) -> None:
    """Change one stored train z row and record the new sha256 in the manifest (a consistent tampered copy)."""
    path = folder / "representations_train.npz"
    table = records.load_npz(path)
    z = table.z.copy()
    z[row] += np.float32(delta)
    path.unlink()
    records.save_npz(path, dataclasses.replace(table, z=z))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["representations_train.npz"] = run.content_sha256(path)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@pytest.mark.criterion("M15-9")
def test_live_check_reads_train_npz(smoke_cluster_dir, cpu_cfg, smoke_data, sasrec_state_path, repo_root, tmp_path):
    """verify_live_encoder compares live z with the first n rows of representations_train.npz; one changed byte of
    that file is refused through the manifest sha256; n outside 1..train count is a ValueError (19.6.9)."""
    from ilora.sasrec import SASRecEncoder

    folder = copy_artifacts(smoke_cluster_dir, tmp_path / "cluster")
    kwargs = load_kwargs(cpu_cfg, smoke_data, run.content_sha256(sasrec_state_path), repo_root)
    artifacts = cluster.load_cluster_artifacts(folder, **kwargs)
    encoder = SASRecEncoder.from_config(cpu_cfg, smoke_data.catalog, HISTORY_LEN)
    n, cpu = cpu_cfg.cluster.live_check_n, torch.device("cpu")
    result = cluster.verify_live_encoder(artifacts, encoder, smoke_data.train, device=cpu, n=n, atol=1e-4)
    stored = records.load_npz(folder / "representations_train.npz")
    assert result["passed"] and result["n"] == n == 64 and result["npz_prefix"] and result["max_abs_diff"] <= 1e-4
    assert result["sample_ids_sha256"] == cluster.ids_sha256(stored.sample_ids[:n])
    for bad_n in (0, len(smoke_data.train) + 1):
        with pytest.raises(ValueError):
            cluster.verify_live_encoder(artifacts, encoder, smoke_data.train, device=cpu, n=bad_n, atol=1e-4)
    for row, passes in ((0, False), (n, True)):        # only the first n stored rows take part
        work = copy_artifacts(folder, tmp_path / f"row{row}")
        rewrite_train_z(work, row, 1e-3)
        changed = cluster.load_cluster_artifacts(work, **kwargs)
        if passes:
            assert cluster.verify_live_encoder(changed, encoder, smoke_data.train, device=cpu, n=n, atol=1e-4)["passed"]
        else:
            with pytest.raises(ClusterError, match="live z check failed"):
                cluster.verify_live_encoder(changed, encoder, smoke_data.train, device=cpu, n=n, atol=1e-4)
    flip_byte(folder / "representations_train.npz")
    with pytest.raises(ClusterError, match="sha256"):
        cluster.verify_live_encoder(artifacts, encoder, smoke_data.train, device=cpu, n=n, atol=1e-4)

# ----------------------------------------------------------------------------------------------- #10
@pytest.mark.criterion("M15-10")
@pytest.mark.parametrize("case", ["same", "z_1e-4", "cluster_hard_export"])
def test_dynamic_export_crosscheck(case, tmp_path):
    """E2 export imitation: equal z passes, a 1e-4 change of one z value fails, a cluster_hard export fails (19.6.2)."""
    tables, _ = synthetic_data("crosscheck", sizes=(60, 12, 12))
    export = tmp_path / "records" / "export" / "epoch1"
    written = write_export(export, tables, "crosscheck_export", epoch=1,
                           mode="cluster_hard" if case == "cluster_hard_export" else "dynamic",
                           z_change=("val", 3, 5, 1e-4) if case == "z_1e-4" else None)
    if case != "same":
        match = r"max \|dz\|" if case == "z_1e-4" else "routing_mode 'cluster_hard'"
        with pytest.raises(ClusterError, match=match):
            cluster.load_dynamic_export(export, tables, atol=1e-5)
        return
    aligned, info = cluster.load_dynamic_export(export, tables, atol=1e-5)
    assert info["routing_mode"] == "dynamic" and info["epoch"] == 1 and info["atol"] == 1e-5
    assert info["rows"] == {"train": 60, "validation": 12, "test": 12}
    assert info["z_max_abs_diff"] == {"train": 0.0, "validation": 0.0, "test": 0.0}
    for split in SPLITS:
        assert info["npz_sha256"][SPEC[split]] == run.content_sha256(export / f"representations_{SPEC[split]}.npz")
        assert aligned[split].sample_ids == tables[split].sample_ids
        assert np.array_equal(aligned[split].omega, written[split].omega)
    fewer = dict(tables)
    fewer["test"] = table_of(synthetic_samples("test", 11, user_offset=2000), tables["test"].z[:11])
    with pytest.raises(ClusterError, match="sample set differs"):
        cluster.load_dynamic_export(export, fewer, atol=1e-5)


# ----------------------------------------------------------------------------------------------- #11
@pytest.mark.slow
@pytest.mark.criterion("M15-11")
def test_cluster_command_real_sasrec(sasrec_state_path, full_data, profile_cfg, repo_root, tmp_path, record_property):
    """The cluster command on the full data with the converted SASRec, twice: fit 66,126 train rows, z, centres and
    the three assignment files equal bit for bit (meta_json, manifest, silhouette and figures are not compared).
    The silhouette value and the times are recorded only (TBD, [SPEC §0.11])."""
    from ilora.commands import run_command

    sets = ["data.subset=full"]
    cfg = profile_cfg("smoke_cpu", *sets)       # the paper_spec cluster section on CPU (tiny backbone, no GPU check)
    outs = []
    for name in ("first", "second"):
        run_dir = run_command("cluster", cfg, sets, profile="cluster_full", root=tmp_path / name, repo_root=repo_root,
                              argv=["pytest", "cluster", *sets])
        assert run.read_meta(run_dir)["status"] == "completed"
        outs.append(run_dir / "cluster")
    first, second = outs
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fit_split"] == "train" and manifest["fit_rows"] == 66_126 and manifest["unit"] == "sequence"
    assert manifest["fit_ids_sha256"] == cluster.ids_sha256([s.sample_id for s in full_data.train])
    assert manifest["rows"] == {"train": 66_126, "validation": 94, "test": 95}
    assert manifest["data"]["subset"] == "full" and manifest["sasrec"]["state_sha256"] == run.content_sha256(sasrec_state_path)
    for split in SPLITS:
        a = records.load_npz(first / f"representations_{SPEC[split]}.npz")
        b = records.load_npz(second / f"representations_{SPEC[split]}.npz")
        assert a.sample_ids == b.sample_ids == tuple(s.sample_id for s in full_data.samples(split))
        assert a.omega is None and a.z.dtype == np.float32 and a.z.shape[1] == D
        assert a.z.tobytes() == b.z.tobytes(), split
        name = f"{SPEC[split]}_assignments.csv"
        assert run.content_sha256(first / name) == run.content_sha256(second / name), name
    ca, cb = cluster.read_centroids(first / "centroids.npz"), cluster.read_centroids(second / "centroids.npz")
    assert ca["cluster_centers"].tobytes() == cb["cluster_centers"].tobytes()
    assert (ca["inertia"], ca["n_iter"], ca["fit_ids_sha256"]) == (cb["inertia"], cb["n_iter"], cb["fit_ids_sha256"])
    summary = json.loads((first / "cluster_summary.json").read_text(encoding="utf-8"))
    assert summary["fit"]["rows"] == 66_126 and summary["fit"]["openmp_threads"] == 1
    assert sum(summary["train_cluster_size"]) == 66_126 and min(summary["train_cluster_size"]) > 0
    assert summary["silhouette"]["n"] == 66_126 and summary["silhouette"]["sample_size"] is None
    assert -1.0 <= summary["silhouette_score_train"] <= 1.0
    artifacts = cluster.load_cluster_artifacts(first, expected_ids=all_ids(full_data),
                                               sasrec_sha256=run.content_sha256(sasrec_state_path),
                                               data_sha256=run.input_hashes(repo_root), ccfg=cfg.cluster)
    assert len(artifacts.assignments) == 66_315
    for key in ("silhouette_score_train", "train_cluster_size", "validation_cluster_size", "test_cluster_size"):
        record_property(key, summary[key])
    record_property("silhouette_seconds", summary["silhouette"]["seconds"])
    record_property("timings_s", summary["timings_s"])
    record_property("fit", summary["fit"])


# ----------------------------------------------------------------------------------------------- smoke artifacts
@pytest.mark.criterion("M15-3", "M15-8")
def test_smoke_cluster_outputs(smoke_cluster_dir, smoke_data):
    """The smoke_cpu cluster run (smoke S4 without an export): 156 train-only fit rows, every file of 19.2 with its
    manifest sha256, z-only tables, the reloaded model predicts the CSVs, the dynamic items NOT_RUN."""
    folder = smoke_cluster_dir
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == cluster.ARTIFACT_FORMAT and manifest["fit_split"] == "train"
    assert manifest["fit_rows"] == 156 and manifest["rows"] == {"train": 156, "validation": 10, "test": 10}
    assert manifest["fit_ids_sha256"] == cluster.ids_sha256([s.sample_id for s in smoke_data.train])
    assert manifest["data"]["subset"] == "smoke" and manifest["data"]["smoke_users"] == [2, 10, 10]
    assert manifest["data"]["train_rows"] == [] and set(manifest["data"]["sha256"]) == set(cluster.DATA_FILES)
    expected = {"kmeans_k4_seed1234.joblib", "centroids.npz", "cluster_summary.json", "figures/sasrec_pca_clusters.png",
                *(f"representations_{SPEC[s]}.npz" for s in SPLITS), *(f"{SPEC[s]}_assignments.csv" for s in SPLITS)}
    assert set(manifest["files"]) == expected
    on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert on_disk == expected | {"manifest.json"}
    assert all(run.content_sha256(folder / name) == sha for name, sha in manifest["files"].items())
    km = cluster.load_kmeans(folder / "kmeans_k4_seed1234.joblib")
    for split in SPLITS:
        table = records.load_npz(folder / f"representations_{SPEC[split]}.npz")
        assert table.omega is None and table.dim == D and table.sample_ids == tuple(s.sample_id for s in
                                                                                   smoke_data.samples(split))
        clusters = cluster.read_assignments(folder / f"{SPEC[split]}_assignments.csv", split=split, n_clusters=4)[3]
        assert np.array_equal(clusters, cluster.predict(km, table.z)), split
    summary = json.loads((folder / "cluster_summary.json").read_text(encoding="utf-8"))
    assert min(summary["train_cluster_size"]) > 0 and sum(summary["train_cluster_size"]) == 156
    assert summary["reload_check"]["passed"] and summary["fit"]["openmp_threads"] == 1
    assert all(summary[item] == cluster.NOT_RUN_DYNAMIC for item in cluster.DYNAMIC_ITEMS)
