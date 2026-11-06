"""M12 gate 분석·그림 (DESIGN.md §16.6.6, §16.6.7; [SPEC §5, §6, §6-1, §8]).

Gate diversity of one split (``gate_summary``): mean utilization, argmax distribution with ties,
normalized entropy, per-expert standard deviation, Jensen-Shannon divergence to the mean gate in bits,
top1 - top2 and, for train, the share of between-user variance per expert. The numbers come with the
fixed reference points of the collapse diagnostic (``reference_points``: SPEC Case A and Case B, the
ten rows of paper Fig. 3c) and nothing is decided by a threshold [SPEC §5]: ``write_interpretation``
writes a fixed template whose last part says that a person writes the interpretation.

Figures follow [SPEC §6-1]: matplotlib's Agg canvas (no pyplot, no display), DejaVu Sans, normal style
and weight, white background, no grid, constrained layout, 300 dpi, the default colour cycle and
``viridis``, English ASCII text only, no annotations inside the axes, ``Expert 1..K`` for
``gate_0..gate_{K-1}``. ``pca_2d`` is for pictures only and never feeds clustering [SPEC §8].
Statistics use float64; omega arrives as float32 (§16.6.8). No random numbers are used: the heatmap
rows are picked with ``linspace`` and Fig. 3c's counterpart takes the first ten test rows (§16.6.8).
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Iterator, Mapping, Optional, Sequence

import numpy as np
from scipy.special import entr, rel_entr

if TYPE_CHECKING:   # M12 records type (§16.6.1); records imports this module inside run_export
    from matplotlib.figure import Figure

    from .records import RepresentationTable

FIGURE_DPI: Final = 300                 # [SPEC §6-1] 기본 DPI 300
HEATMAP_MAX_ROWS: Final = 500           # [결정] §16.6.7 (statistics always use every row)
ENTROPY_BINS: Final = 20                # [결정] §16.6.7: [0, 1] in 20 bins
TEN_SEQUENCES: Final = 10               # [논문 p.7 그림 3] "for ten sequences"
FIGURE_FORMATS: Final = ("png", "pdf")  # records.figure_formats choices [SPEC §6]
STATISTICS: Final = ("mean", "std", "median", "min", "max")      # [SPEC §5] 3, 5 (std with ddof 0, [결정])
CASE_B_GATE: Final = (0.90, 0.03, 0.03, 0.04)                    # [SPEC §5] Case B example (K = 4)
# [논문 p.7 그림 3c (암시)]: the 40 percentages read from the figure (rows = sequences, columns = experts 1-4);
# rows are normalized to sum 1 before use. Selection rule, data set and layer are unknown (§16.3).
PAPER_FIG3C_PERCENT: Final = (
    (17.7, 32.2, 26.7, 23.4), (19.0, 19.0, 15.2, 46.7), (25.7, 28.7, 9.7, 35.9), (42.5, 19.9, 18.8, 18.8),
    (22.1, 29.7, 26.5, 21.6), (34.5, 17.2, 22.8, 25.5), (24.0, 34.0, 16.2, 25.8), (33.7, 13.8, 34.2, 18.3),
    (9.7, 31.1, 31.5, 27.7), (22.3, 15.1, 35.6, 27.0))
SPLIT_LABELS: Final = {"train": "train", "val": "validation", "test": "test"}   # SPEC spelling in figures
SUMMARY_JSON: Final = "gate_summary.json"
SUMMARY_CSV: Final = "gate_summary.csv"
INTERPRETATION_TXT: Final = "gate_interpretation.txt"
TEN_SEQUENCES_CSV: Final = "gate_ten_sequences.csv"
FIGURES_DIR: Final = "figures"
SUMMARY_CSV_FIELDS: Final = ("metric", "expert", "statistic", "value")
DEFINITIONS: Final = {
    "mean_utilization": "mean over samples of omega_k (SPEC 5.1)",
    "argmax": "index of the first maximum of each row; count and ratio per expert; argmax_tie_rows = rows whose "
              "maximum occurs more than once (SPEC 5.2)",
    "normalized_entropy": "H(p) / ln K with 0 log 0 = 0 (scipy.special.entr) per sample (SPEC 5.3)",
    "expert_std": "standard deviation of omega_k over samples, ddof 0 (SPEC 5.4)",
    "js_to_mean_bits": "Jensen-Shannon divergence between each sample's omega and the mean gate m, base 2: "
                       "(0.5 sum rel_entr(p, M) + 0.5 sum rel_entr(m, M)) / ln 2 with M = (p + m) / 2 (SPEC 5.5)",
    "top1_minus_top2": "largest minus second largest omega of each sample (SPEC 5.6, auxiliary dispersion statistic)",
    "user_eta_squared": "train only: sum_u n_u (mean_uk - mean_k)^2 / sum_i (omega_ik - mean_k)^2 over pseudo users "
                        "(D14); null when the total variance is 0",
    "statistics": "mean, std (ddof 0), median, min, max over samples",
}


class AnalysisError(ValueError):
    """Input that the gate analysis cannot use (K < 2, wrong shape, non-finite or negative omega, ...)."""


# ----------------------------------------------------------------------------
# Statistics (§16.6.6) [SPEC §5]
# ----------------------------------------------------------------------------
def _gate_matrix(omega: object, name: str = "omega") -> np.ndarray:
    p = np.asarray(omega, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1:
        raise AnalysisError(f"{name} must be [N, K] with N >= 1, got shape {list(p.shape)}")
    if p.shape[1] < 2:
        raise AnalysisError("gate analysis requires K >= 2")
    if not np.isfinite(p).all() or (p < 0).any():
        raise AnalysisError(f"{name} must be finite and >= 0")
    return p


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    """``H(p) / ln K`` per row with 0 log 0 = 0 (``scipy.special.entr``), float64 [N] (SPEC 5.3)."""
    q = _gate_matrix(p, "p")
    return entr(q).sum(axis=-1) / math.log(q.shape[1])


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits (0-1) per row of ``p`` against ``q`` (broadcast), float64 (SPEC 5.5).

    ``(0.5 sum rel_entr(p, M) + 0.5 sum rel_entr(q, M)) / ln 2`` with ``M = (p + q) / 2``; ``rel_entr`` uses the
    natural logarithm, hence the division by ln 2 (``design/tmp/reference_points.py``).
    """
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    if a.shape[-1] != b.shape[-1] or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise AnalysisError(f"p and q must be finite with the same K, got {list(a.shape)} and {list(b.shape)}")
    if (a < 0).any() or (b < 0).any():
        raise AnalysisError("p and q must be >= 0")
    m = 0.5 * (a + b)
    return (0.5 * rel_entr(a, m).sum(axis=-1) + 0.5 * rel_entr(b, m).sum(axis=-1)) / math.log(2.0)


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max())}


def _top1_minus_top2(p: np.ndarray) -> np.ndarray:
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def _user_view(p: np.ndarray, user_ids: np.ndarray) -> dict:
    """eta^2_k = between-user sum of squares / total sum of squares per expert (pseudo users, D14) [결정]."""
    users = np.asarray(user_ids)
    if users.ndim != 1 or users.shape[0] != p.shape[0]:
        raise AnalysisError(f"user_ids must have one value per row ({p.shape[0]}), got shape {list(users.shape)}")
    uniq, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    mean = p.mean(axis=0)
    eta: list[Optional[float]] = []
    reasons: list[Optional[str]] = []
    for k in range(p.shape[1]):
        total = float(((p[:, k] - mean[k]) ** 2).sum())
        user_mean = np.bincount(inverse, weights=p[:, k], minlength=uniq.size) / counts
        between = float((counts * (user_mean - mean[k]) ** 2).sum())
        if total == 0.0:
            eta.append(None)
            reasons.append("total variance of this expert is 0")
        else:
            eta.append(between / total)
            reasons.append(None)
    return {"n_users": int(uniq.size),
            "windows_per_user": {"min": int(counts.min()), "median": float(np.median(counts)),
                                 "max": int(counts.max())},
            "eta_squared": eta, "eta_squared_null_reason": reasons}


def gate_summary(omega: np.ndarray, *, user_ids: Optional[np.ndarray]) -> dict:
    """SPEC §5 statistics of ``omega [N, K]`` (float32 in, float64 inside); ``user_ids`` only for train.

    ``K < 2`` raises "gate analysis requires K >= 2" (E1 has an export but no analysis, §16.8 #11).
    Keys: ``n``, ``num_experts``, ``metrics`` (per-expert lists under ``per_expert``, otherwise the five
    ``STATISTICS``), ``user_view`` (train), ``reference_points`` and ``definitions``.
    """
    p = _gate_matrix(omega)
    n, k = p.shape
    mean = p.mean(axis=0)
    argmax = p.argmax(axis=1)                               # first index of the row maximum
    counts = np.bincount(argmax, minlength=k)
    ties = int(((p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())
    metrics: dict[str, dict] = {
        "mean_utilization": {"per_expert": mean.tolist()},
        "argmax_count": {"per_expert": [int(c) for c in counts]},
        "argmax_ratio": {"per_expert": (counts / n).tolist()},
        "argmax_tie_rows": {"value": ties},
        "normalized_entropy": _stats(normalized_entropy(p)),
        "expert_std": {"per_expert": p.std(axis=0).tolist()},
        "js_to_mean_bits": _stats(js_divergence_bits(p, mean[None, :])),
        "top1_minus_top2": _stats(_top1_minus_top2(p)),
    }
    out: dict[str, Any] = {"n": int(n), "num_experts": int(k), "metrics": metrics}
    if user_ids is not None:
        view = _user_view(p, np.asarray(user_ids))
        metrics["user_eta_squared"] = {"per_expert": view["eta_squared"]}
        out["user_view"] = view
    out["reference_points"] = reference_points(k)
    out["definitions"] = dict(DEFINITIONS)
    return out


def _point(gate: Sequence[float], note: str) -> dict:
    """Statistics of the case where every sample has the same ``gate`` (JS and std are 0 by construction)."""
    p = np.asarray([gate, gate], dtype=np.float64)
    tie = bool((p[0] == p[0].max()).sum() > 1)
    return {"gate": [float(g) for g in gate], "normalized_entropy": float(normalized_entropy(p)[0]),
            "js_to_mean_bits": float(js_divergence_bits(p, p.mean(axis=0)[None, :])[0]),
            "expert_std": float(p.std(axis=0).max()), "top1_minus_top2": float(_top1_minus_top2(p)[0]),
            "argmax_top_expert_ratio": 1.0, "argmax_tie_row_ratio": 1.0 if tie else 0.0, "note": note}


def reference_points(num_experts: int) -> dict:
    """Reference points of the collapse diagnostic, no judgement value [SPEC §5]: Case A (every gate uniform),
    Case B (every gate [0.90, 0.03, 0.03, 0.04]; for K != 4 one expert 0.9 and the rest equal [결정]),
    paper Fig. 3c (K = 4 only) and Case C (no single point)."""
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 2:
        raise AnalysisError("gate analysis requires K >= 2")
    k = num_experts
    case_b = CASE_B_GATE if k == 4 else (0.9, *([0.1 / (k - 1)] * (k - 1)))
    out: dict[str, Any] = {
        "case_a_uniform": _point([1.0 / k] * k, "every sample uniform: utilization even but no instance-wise "
                                                "difference (SPEC Case A)"),
        "case_b_one_expert": _point(case_b, "every sample puts most weight on one expert (SPEC Case B)"),
        "case_c": {"note": "no single reference point: samples differ, so JS divergence to the mean gate and the "
                           "per-expert standard deviation are above 0 (SPEC Case C)"},
    }
    if k == 4:
        fig = np.asarray(PAPER_FIG3C_PERCENT, dtype=np.float64)
        p = fig / fig.sum(axis=1, keepdims=True)
        mean = p.mean(axis=0)
        out["paper_fig3c"] = {
            "normalized_entropy": _stats(normalized_entropy(p)), "top1_minus_top2": _stats(_top1_minus_top2(p)),
            "js_to_mean_bits": _stats(js_divergence_bits(p, mean[None, :])), "mean_gate": mean.tolist(),
            "expert_std": p.std(axis=0).tolist(), "argmax_expert": [int(a) for a in p.argmax(axis=1)],
            "note": "10 rows read from iLoRA Fig. 3c (p.7), normalized to sum 1; data set, layer and the "
                    "selection of the 10 sequences are not stated in the paper"}
    else:
        out["paper_fig3c"] = {"note": f"not applicable: the figure has 4 experts, K = {k}"}
    return out

# ----------------------------------------------------------------------------
# Summary files and interpretation template (§16.6.6 출력) [SPEC §5]
# ----------------------------------------------------------------------------
def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _plain(obj: object) -> object:
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _block_rows(metric: str, value: object) -> list[tuple[str, str, str, object]]:
    if _is_number(value):
        return [(metric, "", "value", value)]
    if isinstance(value, Mapping):
        if "per_expert" in value:
            return [(metric, str(k), "value", v) for k, v in enumerate(value["per_expert"])]
        return [(metric, "", str(key), v) for key, v in value.items() if _is_number(v)]
    if isinstance(value, (list, tuple)) and value and all(v is None or _is_number(v) for v in value):
        return [(metric, str(k), "value", v) for k, v in enumerate(value)]
    return []


def summary_rows(summary: Mapping[str, object]) -> list[tuple[str, str, str, object]]:
    """Long rows ``(metric, expert, statistic, value)`` of every number in a gate summary (``gate_summary.csv``).

    ``expert`` is the 0-based index for per-expert values (files count from 0, figures from 1, §16.6.7) and
    empty otherwise; a null per-expert value (eta^2 without variance) stays None (empty cell).
    """
    rows = [("n", "", "value", summary["n"]), ("num_experts", "", "value", summary["num_experts"])]
    for name, block in summary["metrics"].items():
        rows += _block_rows(name, block)
    view = summary.get("user_view")
    if isinstance(view, Mapping):
        rows += _block_rows("user_view.n_users", view["n_users"])
        rows += _block_rows("user_view.windows_per_user", view["windows_per_user"])
    for case, block in summary["reference_points"].items():
        for key, value in block.items():
            rows += _block_rows(f"reference.{case}.{key}", value)
    return rows


def _write_text(path: Path, text: str) -> None:
    """UTF-8 with LF, temporary name + ``os.replace``; existing files are never overwritten."""
    path = Path(path)
    if path.exists():
        raise AnalysisError(f"{path} exists; analysis files are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
    os.replace(tmp, path)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(float(value))


def write_gate_summary(out_dir: Path, summary: Mapping[str, object]) -> None:
    """``gate_summary.json`` (the summary) and ``gate_summary.csv`` (``summary_rows``: the same numbers)."""
    out = Path(out_dir)
    plain = _plain(summary)
    _write_text(out / SUMMARY_JSON, json.dumps(plain, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SUMMARY_CSV_FIELDS)
    for metric, expert, statistic, value in summary_rows(plain):
        writer.writerow([metric, expert, statistic, _cell(value)])
    _write_text(out / SUMMARY_CSV, buffer.getvalue())


def _fmt(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    x = float(value)
    return f"{x:.4f}" if x == 0.0 or 1e-3 <= abs(x) < 1e4 else f"{x:.3e}"


def _stats_text(block: Mapping[str, object]) -> str:
    return ", ".join(f"{s} {_fmt(block[s])}" for s in STATISTICS)


def _experts_text(values: Sequence[object]) -> str:
    return ", ".join(f"Expert {k + 1} {_fmt(v)}" for k, v in enumerate(values))


def write_interpretation(out_dir: Path, summary: Mapping[str, object]) -> None:
    """``gate_interpretation.txt`` (UTF-8, fixed template, [SPEC §5]): 1 numbers, 2 reference points, 3 fixed
    sentences on how to read the metrics, 4 "판정하지 않음, 해석은 사람이 작성". No sentence depends on a value."""
    m = summary["metrics"]
    refs = summary["reference_points"]
    split = str(summary.get("split", "?"))
    lines = [
        "gate 해석 틀 (DESIGN.md 16.6.6, SPEC 5). 숫자는 gate_summary.json과 같은 값이다.",
        f"split {SPLIT_LABELS.get(split, split)}, n {summary['n']}, K {summary['num_experts']}, "
        f"checkpoint {summary.get('checkpoint')}, routing_mode {summary.get('routing_mode')}",
        "",
        "1. 수치",
        f"- expert 평균 utilization mean(gate_k): {_experts_text(m['mean_utilization']['per_expert'])}",
        f"- argmax expert 표본 수 (행 최댓값의 첫 index): {_experts_text(m['argmax_count']['per_expert'])}",
        f"- argmax expert 비율: {_experts_text(m['argmax_ratio']['per_expert'])}",
        f"- 최댓값이 둘 이상인 행 수 (argmax_tie_rows): {m['argmax_tie_rows']['value']}",
        f"- 정규화 entropy H(p)/ln K: {_stats_text(m['normalized_entropy'])}",
        f"- expert별 표준편차 (ddof 0): {_experts_text(m['expert_std']['per_expert'])}",
        f"- 평균 gate와의 Jensen-Shannon divergence (bit): {_stats_text(m['js_to_mean_bits'])}",
        f"- top1 - top2: {_stats_text(m['top1_minus_top2'])}",
    ]
    view = summary.get("user_view")
    if isinstance(view, Mapping):
        w = view["windows_per_user"]
        lines.append(f"- 사용자 간 분산 비율 eta^2 (pseudo user {view['n_users']}명, 사용자당 window {w['min']}-{w['max']}개, "
                     f"중앙값 {_fmt(w['median'])}): {_experts_text(view['eta_squared'])}")
    a, b, fig = refs["case_a_uniform"], refs["case_b_one_expert"], refs.get("paper_fig3c", {})
    lines += [
        "",
        "2. 참고점 (판정 기준이 아니다)",
        f"- Case A, 모든 표본이 균등 gate {[_fmt(g) for g in a['gate']]}: 정규화 entropy {_fmt(a['normalized_entropy'])}, "
        f"JS {_fmt(a['js_to_mean_bits'])} bit, expert별 표준편차 {_fmt(a['expert_std'])}, "
        f"top1 - top2 {_fmt(a['top1_minus_top2'])}",
        f"- Case B, 모든 표본이 {[_fmt(g) for g in b['gate']]}: 정규화 entropy {_fmt(b['normalized_entropy'])}, "
        f"JS {_fmt(b['js_to_mean_bits'])} bit, expert별 표준편차 {_fmt(b['expert_std'])}, argmax가 한 expert에 100%, "
        f"top1 - top2 {_fmt(b['top1_minus_top2'])}",
    ]
    if "normalized_entropy" in fig:
        h, t, j = fig["normalized_entropy"], fig["top1_minus_top2"], fig["js_to_mean_bits"]
        lines.append(f"- 논문 그림 3c의 10행 (그림에서 읽은 값을 행 합 1로 정규화, 데이터셋·층·선택 기준 미상): 정규화 entropy "
                     f"평균 {_fmt(h['mean'])} (범위 {_fmt(h['min'])}-{_fmt(h['max'])}), top1 - top2 평균 {_fmt(t['mean'])} "
                     f"(범위 {_fmt(t['min'])}-{_fmt(t['max'])}), 평균 gate와의 JS 평균 {_fmt(j['mean'])} bit")
    else:
        lines.append(f"- 논문 그림 3c: {fig.get('note', 'K = 4에서만 비교')}")
    lines += [
        "- Case C는 단일 참고점이 없다. 표본마다 gate가 다르면 JS divergence와 expert별 표준편차가 0보다 크다.",
        "",
        "3. 지표 읽는 법",
        "- 평균 utilization은 표본 전체의 expert별 평균 비중이다. 표본마다 gate가 다른지는 이 값만으로 알 수 없다 "
        "(Case A도 평균은 균등하다).",
        "- 정규화 entropy는 표본 하나의 gate가 얼마나 고르게 퍼져 있는지를 0-1로 나타낸다. 1은 균등, 0은 한 expert에 전부다.",
        "- 평균 gate와의 JS divergence와 expert별 표준편차는 표본 사이에서 gate가 얼마나 달라지는지를 나타낸다. "
        "모든 표본의 gate가 같으면 둘 다 0이다.",
        "- argmax 분포는 최댓값이 같은 행을 첫 expert로 센다. 그래서 동률 행 수를 함께 적는다.",
        "- top1 - top2는 argmax 요약이 얼마나 뚜렷한지를 나타낸다. 값이 작으면 argmax가 작은 차이로 정해진다.",
        "- eta^2 (train만)는 expert별 gate 분산 가운데 사용자 사이 차이가 차지하는 비율이다. 나머지는 같은 사용자의 "
        "window 사이 차이다.",
        "",
        "4. 판정하지 않음. 해석은 사람이 작성한다.",
    ]
    _write_text(Path(out_dir) / INTERPRETATION_TXT, "\n".join(lines) + "\n")

# ----------------------------------------------------------------------------
# Figures (§16.6.7) [SPEC §6, §6-1, §8]
# ----------------------------------------------------------------------------
_RC: Final = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.style": "normal", "font.weight": "normal",
    "axes.titleweight": "normal", "axes.labelweight": "normal", "figure.titleweight": "normal",
    "axes.grid": False, "axes.unicode_minus": False,          # ASCII hyphen-minus in tick labels
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.edgecolor": "white",
    "savefig.dpi": FIGURE_DPI, "image.cmap": "viridis", "figure.constrained_layout.use": True,
    "legend.frameon": False,
}


@contextlib.contextmanager
def figure_style() -> Iterator[None]:
    """rc settings of every M12/M15 figure (§16.6.7): DejaVu Sans, normal style and weight, white, no grid, 300 dpi."""
    import matplotlib

    with matplotlib.rc_context(_RC):
        yield


def _figure(size: tuple[float, float]) -> "Figure":
    """A Figure on the Agg canvas (no pyplot state, no display), constrained layout."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _check_text(*texts: str) -> None:
    for text in texts:
        if not isinstance(text, str) or not text.isascii():
            raise AnalysisError(f"figure text must be English ASCII (16.6.7), got {text!r}")


def _save(fig: "Figure", stem: Path, formats: Sequence[str]) -> None:
    """``<stem>.<format>`` for every format at 300 dpi; existing files are never overwritten."""
    formats = tuple(formats)
    if not formats or any(f not in FIGURE_FORMATS for f in formats) or len(set(formats)) != len(formats):
        raise AnalysisError(f"figure formats must be a non-empty subset of {FIGURE_FORMATS}, got {formats}")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = stem.parent / f"{stem.name}.{fmt}"
        if path.exists():
            raise AnalysisError(f"{path} exists; figures are never overwritten")
        fig.savefig(path, format=fmt, dpi=FIGURE_DPI, facecolor="white")


def _expert_labels(k: int) -> list[str]:
    return [f"Expert {i + 1}" for i in range(k)]          # [SPEC §6-1]: Expert 1..K = gate_0..gate_{K-1}


def _vector(values: object, name: str, dtype: type = np.float64) -> np.ndarray:
    v = np.asarray(values, dtype=dtype)
    if v.ndim != 1 or v.size < 1 or not np.isfinite(v).all():
        raise AnalysisError(f"{name} must be a finite 1-dim array, got shape {list(v.shape)}")
    return v


def heatmap_order(omega: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Heatmap row order (§16.6.7): argmax ascending, then the maximum weight descending, then row number."""
    p = np.asarray(omega, dtype=np.float64)
    return np.lexsort((np.asarray(rows), -p.max(axis=1), p.argmax(axis=1)))


def plot_gate_heatmap(stem: Path, omega: np.ndarray, rows: np.ndarray, *, title: str,
                      formats: Sequence[str] = ("png",)) -> None:
    """Rows = samples (``heatmap_order``), columns = Expert 1..K, colour range fixed to [0, 1], no y ticks.

    Above 500 rows the picture shows 500 rows picked by ``linspace`` from the sorted order (statistics elsewhere
    use every row) [SPEC §6-1].
    """
    p = _gate_matrix(omega)
    r = np.asarray(rows)
    if r.shape != (p.shape[0],):
        raise AnalysisError(f"rows must have {p.shape[0]} values, got shape {list(r.shape)}")
    _check_text(title)
    shown = p[heatmap_order(p, r)]
    n = shown.shape[0]
    if n > HEATMAP_MAX_ROWS:
        pick = np.linspace(0, n - 1, HEATMAP_MAX_ROWS).astype(np.int64)     # strictly increasing for n > 500
        shown = shown[pick]
    with figure_style():
        fig = _figure((4.0, 4.8))
        ax = fig.add_subplot()
        image = ax.imshow(shown, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
        ax.set_xticks(np.arange(p.shape[1]), labels=_expert_labels(p.shape[1]))
        ax.set_yticks([])
        ax.set_ylabel("Samples" if n <= HEATMAP_MAX_ROWS else f"Samples ({HEATMAP_MAX_ROWS} of {n}, evenly spaced)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax).set_label("Gate weight")
        _save(fig, stem, formats)


def _bars(stem: Path, values: np.ndarray, *, title: str, ylabel: str, formats: Sequence[str],
          reference: Optional[float] = None) -> None:
    _check_text(title, ylabel)
    k = values.size
    with figure_style():
        fig = _figure((4.0, 3.0))
        ax = fig.add_subplot()
        x = np.arange(k)
        ax.bar(x, values, width=0.6, color="C0")
        top = float(values.max())
        if reference is not None:
            ax.axhline(reference, color="0.3", linestyle="--", linewidth=1.0, label="1/K")
            ax.legend(loc="best")
            top = max(top, reference)
        ax.set_xticks(x, labels=_expert_labels(k))
        ax.set_ylim(0.0, top * 1.15 if top > 0 else 1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save(fig, stem, formats)


def plot_mean_utilization(stem: Path, mean: np.ndarray, *, title: str, formats: Sequence[str] = ("png",)) -> None:
    """K rectangular bars of mean(gate_k) and a dashed 1/K line (legend "1/K")."""
    m = _vector(mean, "mean")
    if m.size < 2:
        raise AnalysisError("gate analysis requires K >= 2")
    _bars(stem, m, title=title, ylabel="Mean gate weight", formats=formats, reference=1.0 / m.size)


def plot_argmax_counts(stem: Path, counts: np.ndarray, *, title: str, formats: Sequence[str] = ("png",)) -> None:
    """K rectangular bars: samples whose argmax is each expert."""
    c = _vector(counts, "counts")
    if c.size < 2 or (c < 0).any():
        raise AnalysisError("argmax counts need K >= 2 non-negative values")
    _bars(stem, c, title=title, ylabel="Samples", formats=formats)


def plot_entropy_hist(stem: Path, h: np.ndarray, *, title: str, formats: Sequence[str] = ("png",)) -> None:
    """Histogram of the normalized entropy over [0, 1] in 20 bins (values clipped to [0, 1] against rounding)."""
    values = _vector(h, "h")
    _check_text(title)
    with figure_style():
        fig = _figure((4.0, 3.0))
        ax = fig.add_subplot()
        ax.hist(np.clip(values, 0.0, 1.0), bins=np.linspace(0.0, 1.0, ENTROPY_BINS + 1), color="C0",
                edgecolor="white", linewidth=0.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized entropy")
        ax.set_ylabel("Samples")
        ax.set_title(title)
        _save(fig, stem, formats)


def plot_ten_sequences(stem: Path, omega: np.ndarray, *, title: str, labels: Sequence[str],
                       formats: Sequence[str] = ("png",)) -> None:
    """Fig. 3c counterpart: one horizontal stacked bar per sequence, top to bottom in the given (row) order,
    x = gate weight 0-1, legend Expert 1..K, no numbers in the bars; ``labels`` are the y tick labels."""
    p = _gate_matrix(omega)
    labels = list(labels)
    if len(labels) != p.shape[0]:
        raise AnalysisError(f"{p.shape[0]} rows need {p.shape[0]} labels, got {len(labels)}")
    _check_text(title, *labels)
    with figure_style():
        fig = _figure((5.0, 3.6))
        ax = fig.add_subplot()
        y = np.arange(p.shape[0])
        left = np.zeros(p.shape[0])
        for k in range(p.shape[1]):
            ax.barh(y, p[:, k], left=left, height=0.7, color=f"C{k % 10}", label=f"Expert {k + 1}")
            left = left + p[:, k]
        ax.set_yticks(y, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Gate weight")
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        _save(fig, stem, formats)

def pca_2d(z_fit: np.ndarray) -> np.ndarray:
    """Two PCA coordinates for pictures only, never a clustering input [SPEC §7, §8]:
    ``PCA(n_components=2, svd_solver="full")`` (LAPACK, no random numbers) on float64, [N, 2]."""
    from sklearn.decomposition import PCA

    z = np.asarray(z_fit)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise AnalysisError(f"pca_2d needs finite [N, d] with N, d >= 2, got shape {list(z.shape)}")
    return PCA(n_components=2, svd_solver="full").fit_transform(z.astype(np.float64))


def plot_pca_clusters(stem: Path, coords: np.ndarray, labels: np.ndarray, *, n_clusters: int, title: str,
                      formats: Sequence[str] = ("png",)) -> None:
    """``sasrec_pca_clusters`` (M15, [SPEC §8]): small points drawn in row order in one call (no cluster is drawn
    on top as a block), colour = cluster, legend ``Cluster 0..C-1``, axes PC 1 / PC 2, nothing else."""
    from matplotlib.colors import to_rgba_array
    from matplotlib.lines import Line2D

    c = np.asarray(coords, dtype=np.float64)
    lab = np.asarray(labels)
    if c.ndim != 2 or c.shape[1] != 2 or not np.isfinite(c).all():
        raise AnalysisError(f"coords must be finite [N, 2], got shape {list(c.shape)}")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 1:
        raise AnalysisError(f"n_clusters must be an int >= 1, got {n_clusters!r}")
    if lab.shape != (c.shape[0],) or lab.dtype.kind not in "iu" or (lab < 0).any() or (lab >= n_clusters).any():
        raise AnalysisError(f"labels must be {c.shape[0]} integers in 0..{n_clusters - 1}")
    _check_text(title)
    with figure_style():
        palette = to_rgba_array([f"C{i % 10}" for i in range(n_clusters)])
        fig = _figure((4.6, 4.0))
        ax = fig.add_subplot()
        ax.scatter(c[:, 0], c[:, 1], c=palette[lab], s=2, linewidths=0, rasterized=True)
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=4, color=palette[i], label=f"Cluster {i}")
                   for i in range(n_clusters)]
        ax.legend(handles=handles, loc="best")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(title)
        _save(fig, stem, formats)


def plot_cluster_gate_heatmap(stem: Path, means: np.ndarray, *, title: str, formats: Sequence[str] = ("png",)) -> None:
    """``cluster_dynamic_gate_heatmap`` (M15, [SPEC §8]): rows ``Cluster c``, columns ``Expert k+1``, range [0, 1]."""
    p = np.asarray(means, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1 or not np.isfinite(p).all():
        raise AnalysisError(f"means must be finite [C, K], got shape {list(p.shape)}")
    _check_text(title)
    with figure_style():
        fig = _figure((4.0, 3.2))
        ax = fig.add_subplot()
        image = ax.imshow(p, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
        ax.set_xticks(np.arange(p.shape[1]), labels=_expert_labels(p.shape[1]))
        ax.set_yticks(np.arange(p.shape[0]), labels=[f"Cluster {c}" for c in range(p.shape[0])])
        ax.set_title(title)
        fig.colorbar(image, ax=ax).set_label("Mean gate weight")
        _save(fig, stem, formats)


# ----------------------------------------------------------------------------
# One split end to end (§16.6.5 step 8)
# ----------------------------------------------------------------------------
def run_gate_analysis(table: "RepresentationTable", out_dir: Path, *, ckpt_tag: str,
                      formats: Sequence[str] = ("png",)) -> None:
    """Gate analysis of one export table into ``out_dir`` (``records/analysis/epoch{e}/{split}/``).

    Writes ``gate_summary.json``/``.csv``, ``gate_interpretation.txt`` and ``figures/`` (``gate_heatmap``,
    ``expert_mean_utilization``, ``argmax_expert_distribution``, ``entropy_distribution``; for test also
    ``gate_ten_sequences`` with its values in ``gate_ten_sequences.csv``: the first 10 rows in row order, a
    rule that never looks at the gates) in every ``formats`` (``records.figure_formats``). The ``user_view``
    is computed for train only. K < 2 raises "gate analysis requires K >= 2".
    """
    omega = getattr(table, "omega", None)
    if omega is None:
        raise AnalysisError("the table has no omega (z-only tables are not analysed)")
    if omega.ndim != 2 or omega.shape[1] < 2:
        raise AnalysisError("gate analysis requires K >= 2")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty; analysis folders are never overwritten")
    split = table.split
    label = SPLIT_LABELS.get(split, split)
    meta = dict(table.meta)
    summary = {"split": split, "checkpoint": ckpt_tag, "routing_mode": meta.get("routing_mode"),
               "source": {key: meta.get(key) for key in ("source", "checkpoint", "sasrec", "source_run")
                          if key in meta}}
    summary.update(gate_summary(omega, user_ids=table.user_ids if split == "train" else None))
    write_gate_summary(out, summary)
    write_interpretation(out, summary)
    p = np.asarray(omega, dtype=np.float64)
    figures = out / FIGURES_DIR
    plot_gate_heatmap(figures / "gate_heatmap", omega, table.rows, title=f"Gate weights per sample ({label})",
                      formats=formats)
    plot_mean_utilization(figures / "expert_mean_utilization", p.mean(axis=0),
                          title=f"Mean gate weight per expert ({label})", formats=formats)
    plot_argmax_counts(figures / "argmax_expert_distribution", np.bincount(p.argmax(axis=1), minlength=p.shape[1]),
                       title=f"Argmax expert ({label})", formats=formats)
    plot_entropy_hist(figures / "entropy_distribution", normalized_entropy(p),
                      title=f"Normalized gate entropy ({label})", formats=formats)
    if split == "test":
        n = min(TEN_SEQUENCES, len(table.sample_ids))
        ids = list(table.sample_ids[:n])
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["sample_id", "row", *(f"gate_{k}" for k in range(p.shape[1]))])
        for i in range(n):
            writer.writerow([ids[i], int(table.rows[i]), *("%.9g" % float(v) for v in omega[i].tolist())])
        _write_text(out / TEN_SEQUENCES_CSV, buffer.getvalue())
        plot_ten_sequences(figures / "gate_ten_sequences", omega[:n], labels=ids,
                           title=f"Gate weights of the first {n} test sequences", formats=formats)


__all__ = [
    "AnalysisError", "FIGURE_DPI", "HEATMAP_MAX_ROWS", "ENTROPY_BINS", "TEN_SEQUENCES", "STATISTICS", "CASE_B_GATE",
    "PAPER_FIG3C_PERCENT", "SUMMARY_JSON", "SUMMARY_CSV", "INTERPRETATION_TXT", "TEN_SEQUENCES_CSV", "FIGURES_DIR",
    "normalized_entropy", "js_divergence_bits", "gate_summary", "reference_points", "summary_rows",
    "write_gate_summary", "write_interpretation", "figure_style", "heatmap_order", "plot_gate_heatmap",
    "plot_mean_utilization", "plot_argmax_counts", "plot_entropy_hist", "plot_ten_sequences", "pca_2d",
    "plot_pca_clusters", "plot_cluster_gate_heatmap", "run_gate_analysis",
]
