"""Figures drawn only from the raw files of run folders (matplotlib + seaborn, no torch, no model).
A figure whose data is missing is skipped with a log line, never filled with placeholder data."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from .io import find, read_frame, read_json  # noqa: E402
from .schema import FP16_MAX, KNOWN_MASSIVE_DIMS, MODULE_TYPES, short_name  # noqa: E402

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='notebook')
except Exception:  # seaborn only sets the style
    sns = None

UPSTREAM, NEW, LIMIT = '#E67E22', '#1F5FBF', '#D62728'   # original code orange, modified code blue, limits red
PRECISION_COLORS = {'16': '#8E44AD', 'bf16': '#0F9D8A'}  # fp16 purple, bf16 teal
SEVERITY_STYLE = {'warn': ('^', '#555555'), 'high': ('D', '#111111'), 'critical': ('X', LIMIT)}
DPI = 300


def log(msg):
    print(f'[plot] {msg}', flush=True)


class Run:
    def __init__(self, run_dir):
        self.dir = run_dir
        self.name = os.path.basename(os.path.abspath(run_dir))
        self.meta = read_json(os.path.join(run_dir, 'meta.json'))
        if not self.meta:  # older diagnose_nan.py folders: summary.json only
            summary = read_json(os.path.join(run_dir, 'summary.json'))
            self.meta = {'precision': summary.get('config', {}).get('precision'),
                         'max_lr': summary.get('config', {}).get('lr'), 'config': summary.get('config', {})}
        self.precision = str(self.meta.get('precision') or '')
        self.micro = self._numeric(read_frame(find(run_dir, 'micro_batches')))
        self.steps = self._numeric(read_frame(find(run_dir, 'optimizer_steps')))
        self.stats = self._numeric(read_frame(find(run_dir, 'activation_stats')))
        self.anomalies = self._numeric(read_frame(os.path.join(run_dir, 'anomalies.csv')))
        self.gpu = self._numeric(read_frame(os.path.join(run_dir, 'gpu.csv')))
        if self.micro is not None and 'm_t' not in self.micro and 'max_abs_lowp_out' in self.micro:
            self.micro['m_t'] = self.micro['max_abs_lowp_out']
            self.micro['m_t_module'] = self.micro.get('max_abs_lowp_module')

    @staticmethod
    def _numeric(frame):
        if frame is None:
            return None
        for c in frame.columns:
            if frame[c].dtype == object:
                conv = pd.to_numeric(frame[c], errors='coerce')
                if conv.notna().sum() == frame[c].replace('', np.nan).notna().sum():
                    frame[c] = conv
        return frame

    @property
    def color(self):
        return PRECISION_COLORS.get(self.precision, NEW)

    @property
    def label(self):
        p = {'16': 'fp16', 'bf16': 'bf16'}.get(self.precision, self.precision or '?')
        lr = self.meta.get('max_lr')
        cond = self.meta.get('condition')
        return f"{self.name} ({p}{', lr ' + format(float(lr), 'g') if lr not in (None, '') else ''}" \
               f"{', ' + cond if cond else ''})"


LINESTYLES = ('-', '--', ':', '-.')


def _log_axis(ax, axis='y'):
    """Log axis readable over a narrow range: ticks at 1, 2, 5 x 10^k, printed as plain numbers."""
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}'))
    a.set_minor_formatter(NullFormatter())


def _styles(runs):
    """(color, linestyle) per run: color from the precision, line style separates runs of one precision."""
    seen, out = {}, []
    for r in runs:
        n = seen.get(r.color, 0)
        seen[r.color] = n + 1
        out.append((r.color, LINESTYLES[n % len(LINESTYLES)]))
    return out


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log(f'wrote {path}')
    return path


def _limit_lines(ax, abs_threshold=None):
    ax.axhline(FP16_MAX, color=LIMIT, ls='--', lw=1.2, label=f'fp16 max {FP16_MAX:,.0f}')
    if abs_threshold:
        ax.axhline(abs_threshold, color='#888888', ls=':', lw=1, label=f'anomaly threshold {abs_threshold:,.0f}')


def _anomaly_markers(ax, run, x='micro_batch'):
    if run.anomalies is None:
        return
    for sev, (marker, color) in SEVERITY_STYLE.items():
        ev = run.anomalies[run.anomalies['severity'] == sev]
        if len(ev):
            ax.scatter(ev[x], ev['value'], marker=marker, color=color, s=28, zorder=5, label=f'{sev} event ({len(ev)})')


def fig_m_t(runs, out_dir):
    have = [r for r in runs if r.micro is not None and 'm_t' in r.micro]
    if not have:
        return log('m_t: no micro_batches data, skipped')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for r, (color, ls) in zip(have, _styles(have)):
        ax.plot(r.micro['micro_batch'], r.micro['m_t'], color=color, ls=ls, lw=0.7, alpha=0.85, label=r.label)
    _anomaly_markers(ax, have[0])
    _limit_lines(ax, have[0].meta.get('settings', {}).get('abs_threshold'))
    ax.set_yscale('log')
    _log_axis(ax)
    ax.set(xlabel='micro-batch', ylabel='m_t = max |activation| (fp16/bf16 outputs)',
           title='Largest activation per micro-batch')
    ax.legend(fontsize=7, loc='lower right')
    return _save(fig, out_dir, 'm_t.png')


def fig_modules(run, out_dir, top=15):
    if run.stats is None:
        return log('modules: no activation_stats, skipped')
    s = run.stats.assign(short=run.stats['module'].astype(str).map(short_name))
    order = s.groupby('short')['max_abs'].max().sort_values(ascending=False).index[:top]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [3, 2]})
    data = [s.loc[s['short'] == m, 'max_abs'].dropna().values for m in order]
    axes[0].boxplot(data, vert=False, labels=list(order), showfliers=True, flierprops={'markersize': 2})
    axes[0].invert_yaxis()
    axes[0].set_xscale('log')
    _log_axis(axes[0], 'x')
    axes[0].axvline(FP16_MAX, color=LIMIT, ls='--', lw=1.2, label=f'fp16 max {FP16_MAX:,.0f}')
    every = run.meta.get('settings', {}).get('stats_every', 1)
    axes[0].set(xlabel='max |activation| per recorded micro-batch',
                title=f'Top {len(order)} modules by max |activation| (activation_stats, every {every} micro-batch)')
    axes[0].tick_params(axis='y', labelsize=7)
    axes[0].legend(fontsize=7)
    if run.micro is not None and 'm_t_module' in run.micro:
        vc = run.micro['m_t_module'].dropna().astype(str).map(short_name).value_counts().head(top)
        axes[1].barh(vc.index[::-1], vc.values[::-1], color=run.color)
        axes[1].set(xlabel='micro-batches', title='Module that produced m_t')
        axes[1].tick_params(axis='y', labelsize=7)
    else:
        axes[1].set_visible(False)
    return _save(fig, out_dir, 'module_max_abs.png')


def fig_layer_heatmap(run, out_dir):
    if run.stats is None:
        return log('layer heatmap: no activation_stats, skipped')
    s = run.stats[(run.stats['layer'] >= 0) & run.stats['module_type'].isin(MODULE_TYPES)]
    if not len(s):
        return log('layer heatmap: no per-layer projection rows, skipped')
    pivot = s.groupby(['layer', 'module_type'])['max_abs'].max().unstack().reindex(columns=list(MODULE_TYPES))
    fig, ax = plt.subplots(figsize=(7, 9))
    vals = pivot.values.astype(float)
    positive = vals[np.isfinite(vals) & (vals > 0)]
    im = ax.imshow(vals, aspect='auto', cmap='viridis',
                   norm=LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set(xlabel='module type', ylabel='decoder layer', title='Max |activation| by layer and module')
    fig.colorbar(im, ax=ax, label='max |activation| (log scale)')
    return _save(fig, out_dir, 'layer_module_heatmap.png')


def fig_quantile_trend(run, out_dir, module=None):
    if run.stats is None:
        return log('quantile trend: no activation_stats, skipped')
    s = run.stats.assign(short=run.stats['module'].astype(str).map(short_name))
    if module is None:
        module = s.groupby('short')['max_abs'].max().idxmax()
    m = s[s['short'] == module].sort_values('micro_batch')
    if not len(m):
        return log(f'quantile trend: module {module} not in activation_stats, skipped')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, style, lab in (('median_abs', '-', 'median |a|'), ('p99_abs', '-', 'p99 |a|'), ('max_abs', '-', 'max |a|')):
        ax.plot(m['micro_batch'], m[col], ls=style, lw=0.8, label=lab,
                color={'median_abs': '#7F8C8D', 'p99_abs': run.color, 'max_abs': '#2C3E50'}[col])
    _limit_lines(ax)
    ax.set_yscale('log')
    _log_axis(ax)
    ax.set(xlabel='micro-batch', ylabel='|activation|',
           title=f'Typical vs largest values: {module}')
    ax.legend(fontsize=8)
    safe = module.replace('.', '_').replace('#', '_')
    return _save(fig, out_dir, f'quantile_trend_{safe}.png')


def fig_loss_grad(runs, out_dir):
    have = [r for r in runs if r.steps is not None and len(r.steps)]
    if not have:
        return log('loss/grad norm: no optimizer_steps data, skipped')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for r, (color, ls) in zip(have, _styles(have)):
        axes[0].plot(r.steps['optimizer_step'], r.steps['mean_loss'], color=color, ls=ls, lw=1, label=r.label)
        axes[1].plot(r.steps['optimizer_step'], r.steps['grad_norm'], color=color, ls=ls, lw=1, label=r.label)
    warm = have[0].meta.get('config', {}).get('warmup_steps') if isinstance(have[0].meta.get('config'), dict) else None
    for ax in axes:
        if warm:
            ax.axvline(float(warm), color='#999999', ls=':', lw=1, label=f'warmup end (step {warm})')
    axes[0].set(ylabel='loss (mean over the step)', title='Training loss and gradient norm')
    axes[1].set(xlabel='optimizer step', ylabel='gradient L2 norm (unscaled)')
    axes[1].set_yscale('log')
    _log_axis(axes[1])
    axes[0].legend(fontsize=7)
    return _save(fig, out_dir, 'loss_grad_norm.png')


def fig_loss_scale(runs, out_dir):
    have = [r for r in runs if r.steps is not None and 'scale_after' in r.steps and r.steps['scale_after'].notna().any()]
    if not have:
        return log('loss scale: no GradScaler data (bf16 or missing), skipped')
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for r, (color, ls) in zip(have, _styles(have)):
        st = r.steps
        ax.step(st['optimizer_step'], st['scale_after'], where='post', color=color, ls=ls, lw=1.2, label=r.label)
        sk = st[st['skipped_by_grad_scaler'].astype(str).str.lower().isin(['true', '1'])]
        ax.scatter(sk['optimizer_step'], sk['scale_after'], marker='x', color=LIMIT, s=40, zorder=5,
                   label=f'skipped step ({len(sk)})')
    ax.set_yscale('log', base=2)
    ax.set(xlabel='optimizer step', ylabel='loss scale S', title='GradScaler loss scale (fp16)')
    ax.legend(fontsize=7)
    return _save(fig, out_dir, 'loss_scale.png')


def fig_anomaly_positions(run, out_dir, top=20):
    if run.anomalies is None:
        return log('anomaly positions: no anomalies, skipped')
    ev = run.anomalies
    serious = ev[ev['severity'].isin(['warn', 'high', 'critical'])]
    ev = serious if len(serious) else ev
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, title in ((axes[0], 'token_pos', 'token position'), (axes[1], 'feature_dim', 'feature dimension')):
        vc = ev[col].dropna()
        vc = vc[vc.astype(str) != ''].astype(int).astype(str).value_counts().head(top)
        if not len(vc):
            ax.set_visible(False)
            continue
        colors = [UPSTREAM if col == 'feature_dim' and int(k) in KNOWN_MASSIVE_DIMS else run.color for k in vc.index]
        ax.bar(vc.index, vc.values, color=colors)
        ax.set(xlabel=title, ylabel='events', title=f'Anomaly events by {title}')
        ax.tick_params(axis='x', rotation=60, labelsize=7)
    axes[1].set_title(f'Anomaly events by feature dimension (orange = {KNOWN_MASSIVE_DIMS})')
    fig.suptitle(f"{'severity warn and above' if len(serious) else 'all events'}, {len(ev)} events", fontsize=9)
    return _save(fig, out_dir, 'anomaly_positions.png')


def fig_gpu(run, out_dir):
    if run.gpu is None or not len(run.gpu):
        return log('GPU: no gpu.csv, skipped')
    g = run.gpu.copy()
    t = pd.to_datetime(g['timestamp'], errors='coerce')
    minutes = (t - t.iloc[0]).dt.total_seconds() / 60 if t.notna().all() else pd.Series(range(len(g)))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(minutes, g['memory_used_mib'] / 1024, color=run.color)
    axes[0].axhline(g['memory_total_mib'].iloc[0] / 1024, color=LIMIT, ls='--', lw=1.2, label='total memory')
    axes[0].set(ylabel='memory used (GiB)', title='GPU resources')
    axes[0].legend(fontsize=7)
    axes[1].plot(minutes, g['utilization_gpu_pct'], color=run.color)
    axes[1].set(ylabel='utilization (%)')
    axes[2].plot(minutes, g['power_draw_w'], color=run.color)
    axes[2].set(ylabel='power (W)', xlabel='time since first sample (min)')
    return _save(fig, out_dir, 'gpu.png')


def fig_base_vs_trained(run, bases, out_dir, focus=('layers.30.mlp.down_proj',), top=10):
    bases = [b for b in bases if b.micro is not None]
    if not bases or run.micro is None:
        return log('base vs trained: no base run given, skipped')
    n = min(len(b.micro) for b in bases)
    trained_first = run.micro[run.micro['micro_batch'] <= n]
    shades = [UPSTREAM, '#F5B041', '#FAD7A0']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    series = [(b.label, b.micro['m_t'].dropna().values, shades[i % len(shades)]) for i, b in enumerate(bases)]
    series += [(f'trained, first {n}', trained_first['m_t'].dropna().values, NEW),
               ('trained, whole run', run.micro['m_t'].dropna().values, '#123B7A')]
    bp = axes[0].boxplot([s[1] for s in series], vert=False, labels=[s[0] for s in series], patch_artist=True)
    for patch, s in zip(bp['boxes'], series):
        patch.set_facecolor(s[2])
    axes[0].set_xscale('log')
    _log_axis(axes[0], 'x')
    axes[0].axvline(FP16_MAX, color=LIMIT, ls='--', lw=1.2)
    axes[0].set(xlabel='m_t per micro-batch', title='m_t: before training vs trained run')
    axes[0].tick_params(axis='y', labelsize=7)
    if run.stats is not None and all(b.stats is not None for b in bases):
        def per_module(frame, first=None):
            f = frame if first is None else frame[frame['micro_batch'] <= first]
            return f.assign(short=f['module'].astype(str).map(short_name)).groupby('short')['max_abs'].max()
        t_all = per_module(run.stats)
        mods = list(dict.fromkeys(list(focus) + list(t_all.sort_values(ascending=False).index[:top])))
        mods = [m for m in mods if m in t_all.index]
        width = 0.8 / (len(bases) + 1)
        y = np.arange(len(mods))
        for i, b in enumerate(bases):
            v = per_module(b.stats).reindex(mods)
            axes[1].barh(y + i * width, v.values, height=width, color=shades[i % len(shades)], label=b.label)
        axes[1].barh(y + len(bases) * width, per_module(run.stats, n).reindex(mods).values, height=width, color=NEW,
                     label=f'trained, first {n}')
        axes[1].set_yticks(y + width * len(bases) / 2, mods, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xscale('log')
        _log_axis(axes[1], 'x')
        axes[1].axvline(FP16_MAX, color=LIMIT, ls='--', lw=1.2)
        axes[1].set(xlabel='max |activation| over the micro-batches', title='Per module')
        axes[1].legend(fontsize=7)
    else:
        axes[1].set_visible(False)
        log('base vs trained: per-module panel needs activation_stats in every run, skipped')
    return _save(fig, out_dir, 'base_vs_trained.png')


def fig_live(run, out_dir):
    """2x2 overview for scripts/plot_live.py."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    ax_loss, ax_grad, ax_scale, ax_m = axes.flat
    if run.micro is not None:
        ax_loss.plot(run.micro['micro_batch'], run.micro['loss'], color=run.color, lw=0.5, alpha=0.7)
        ax_m.plot(run.micro['micro_batch'], run.micro['m_t'], color=run.color, lw=0.6)
        _anomaly_markers(ax_m, run)
    _limit_lines(ax_m, run.meta.get('settings', {}).get('abs_threshold'))
    ax_m.set_yscale('log')
    _log_axis(ax_m)
    if run.steps is not None and len(run.steps):
        ax_grad.plot(run.steps['optimizer_step'], run.steps['grad_norm'], color=run.color)
        ax_grad.set_yscale('log')
        _log_axis(ax_grad)
        if run.steps['scale_after'].notna().any():
            ax_scale.step(run.steps['optimizer_step'], run.steps['scale_after'], where='post', color=run.color)
            sk = run.steps[run.steps['skipped_by_grad_scaler'].astype(str).str.lower().isin(['true', '1'])]
            ax_scale.scatter(sk['optimizer_step'], sk['scale_after'], marker='x', color=LIMIT, zorder=5)
            ax_scale.set_yscale('log', base=2)
    ax_loss.set(xlabel='micro-batch', ylabel='loss', title='Loss')
    ax_grad.set(xlabel='optimizer step', ylabel='gradient L2 norm', title='Gradient norm')
    ax_scale.set(xlabel='optimizer step', ylabel='loss scale S', title='GradScaler loss scale (fp16)')
    ax_m.set(xlabel='micro-batch', ylabel='m_t', title='Largest activation (log)')
    ax_m.legend(fontsize=6, loc='lower right')
    fig.suptitle(run.label, fontsize=9)
    return _save(fig, out_dir, 'live.png')
