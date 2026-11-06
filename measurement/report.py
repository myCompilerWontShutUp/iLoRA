"""Markdown reports built only from the raw files of run folders (no torch, no model)."""
import os

from .io import find, read_frame, read_json
from .schema import KNOWN_MASSIVE_DIMS, SEVERITIES, short_name

RANK = {s: i for i, s in enumerate(SEVERITIES)}


def _table(frame, columns, headers=None):
    headers = headers or columns
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    for _, r in frame.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            cells.append(f'{v:.6g}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def _counts(series, top):
    vc = series.value_counts().head(top)
    return '\n'.join(['| value | events |', '| --- | --- |'] + [f'| {k} | {v} |' for k, v in vc.items()])


def _truthy(series):
    return series.astype(str).str.lower().isin(['true', '1'])


def write_anomaly_report(run_dir):
    import pandas as pd
    meta = read_json(os.path.join(run_dir, 'meta.json'))
    events = read_frame(os.path.join(run_dir, 'anomalies.csv'))
    micro = read_frame(find(run_dir, 'micro_batches'))
    s = meta.get('settings', {})
    lines = [f"# Anomaly report: {os.path.basename(os.path.abspath(run_dir))}", '',
             f"precision {meta.get('precision')}, max lr {meta.get('max_lr')}, "
             f"micro-batches {0 if micro is None else len(micro)}. Rules: abs >= {s.get('abs_threshold')}, "
             f"high >= {s.get('high_fraction')} x {s.get('fp16_max')}, critical > {s.get('fp16_max')} or non-finite, "
             f"ratio >= {s.get('ratio_threshold')} x median of the last {s.get('ratio_window')} micro-batches, "
             f"record (new per-module maximum) after {s.get('record_warmup')} micro-batches.", '']
    if events is None:
        lines.append('No anomaly events.')
    else:
        events = events.copy()
        events['rank'] = events['severity'].map(RANK).fillna(-1)
        events['module_short'] = events['module'].astype(str).map(short_name)
        counts = events['severity'].value_counts()
        lines.append('Events by severity: ' + ', '.join(f'{k} {int(counts.get(k, 0))}' for k in reversed(SEVERITIES)))
        lines += ['', '## Top 20 events', '']
        top = events.sort_values(['rank', 'value'], ascending=[False, False]).head(20)
        lines.append(_table(top, ['event_id', 'severity', 'rule', 'micro_batch', 'optimizer_step', 'module_short',
                                  'value', 'token_pos', 'token_str', 'feature_dim', 'ratio_to_median'],
                            ['id', 'severity', 'rule', 'micro-batch', 'step', 'module', 'value', 'token pos',
                             'token', 'feature dim', 'ratio']))
        serious = events[events['rank'] >= RANK['warn']]
        basis = serious if len(serious) else events
        lines += ['', f"## Frequency ({'severity warn and above' if len(serious) else 'all events'}, "
                      f"{len(basis)} events)", '']
        for title, col, top_n in (('module', 'module_short', 15), ('layer', 'layer', 40),
                                  ('feature dimension', 'feature_dim', 15), ('token position', 'token_pos', 15)):
            lines += [f'### By {title}', '', _counts(basis[col].dropna().astype(str), top_n), '']
        lines += ['## Massive-activation check (events)', '',
                  f"- at the first token / BOS: {int(_truthy(basis['is_first_token']).sum())} of {len(basis)}",
                  f"- in feature dims {KNOWN_MASSIVE_DIMS}: {int(_truthy(basis['is_known_massive_dim']).sum())} of {len(basis)}", '']
    if micro is not None and 'm_t' in micro:
        m = pd.to_numeric(micro['m_t'], errors='coerce')
        n = len(m)
        q = max(1, n // 4)
        lines += ['## m_t location (all micro-batches)', '']
        if 'm_t_is_first_token' in micro:
            lines.append(f"- m_t at the first token / BOS: {int(_truthy(micro['m_t_is_first_token']).sum())} of {n}")
            lines.append(f"- m_t in feature dims {KNOWN_MASSIVE_DIMS}: "
                         f"{int(_truthy(micro['m_t_is_known_massive_dim']).sum())} of {n}")
        mod = micro['m_t_module'].dropna().astype(str).map(short_name).value_counts().head(5)
        lines += ['- most frequent m_t modules: ' + ', '.join(f'{k} ({v})' for k, v in mod.items()), '',
                  '## Trend of m_t', '',
                  '| part | micro-batches | median m_t | max m_t |', '| --- | --- | --- | --- |',
                  f'| first 25% | 1-{q} | {m.iloc[:q].median():.6g} | {m.iloc[:q].max():.6g} |',
                  f'| last 25% | {n - q + 1}-{n} | {m.iloc[-q:].median():.6g} | {m.iloc[-q:].max():.6g} |', '']
    with open(os.path.join(run_dir, 'anomaly_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _module_table(run_dir, first_n=None):
    """Per-module max / median of per-micro-batch max |value| (activation_stats), or None."""
    import pandas as pd
    stats = read_frame(find(run_dir, 'activation_stats'))
    if stats is None:
        return None
    if first_n is not None:
        stats = stats[pd.to_numeric(stats['micro_batch']) <= first_n]
    stats = stats.assign(short=stats['module'].astype(str).map(short_name),
                         max_abs=pd.to_numeric(stats['max_abs'], errors='coerce'),
                         median_abs=pd.to_numeric(stats['median_abs'], errors='coerce'))
    g = stats.groupby('short')
    return pd.DataFrame({'max': g['max_abs'].max(), 'median_of_max': g['max_abs'].median(),
                         'median_of_median_abs': g['median_abs'].median()})


def _m_t(run_dir, first_n=None):
    import pandas as pd
    micro = read_frame(find(run_dir, 'micro_batches'))
    if micro is None:
        return None
    col = 'm_t' if 'm_t' in micro else 'max_abs_lowp_out'
    if first_n is not None:
        micro = micro[pd.to_numeric(micro['micro_batch']) <= first_n]
    return micro.assign(m=pd.to_numeric(micro[col], errors='coerce'),
                        module=micro.get('m_t_module', micro.get('max_abs_lowp_module')).astype(str).map(short_name))


def _label(run_dir):
    meta = read_json(os.path.join(run_dir, 'meta.json'))
    bits = [meta.get('kind') or 'run', meta.get('condition') or '', f"precision {meta.get('precision')}"
            if meta.get('precision') else '']
    return f"{os.path.basename(os.path.abspath(run_dir))} ({', '.join(b for b in bits if b)})"


def compare_base_vs_trained(base_dirs, trained_dir, out_path, focus=('layers.30.mlp.down_proj',), top=10):
    import pandas as pd
    base_dirs = list(base_dirs)
    lines = ['# Base activation vs trained run', '', f'- trained: `{_label(trained_dir)}`']
    lines += [f'- base: `{_label(b)}`' for b in base_dirs]
    base_m = [_m_t(b) for b in base_dirs]
    n = min(len(m) for m in base_m if m is not None) if any(m is not None for m in base_m) else None
    trained_first = _m_t(trained_dir, n)
    trained_all = _m_t(trained_dir)
    lines += ['', f'Base runs measured {n} micro-batches; the trained run is compared over its first {n} '
                  'micro-batches and over the whole run.', '']
    if trained_first is not None:
        for b, bm in zip(base_dirs, base_m):
            if bm is None:
                continue
            k = min(len(bm), len(trained_first))
            same = (list(bm['sample_ids'].astype(str)[:k]) == list(trained_first['sample_ids'].astype(str)[:k])
                    if 'sample_ids' in bm and 'sample_ids' in trained_first else None)
            lines.append(f"- batch order of `{os.path.basename(b)}` vs trained run: "
                         f"{'same' if same else ('different' if same is not None else 'unknown')} for {k} micro-batches")
    lines += ['', '## m_t (largest |value| over all fp16/bf16 module outputs, per micro-batch)', '',
              '| run | micro-batches | median | p99 | max | most frequent module |', '| --- | --- | --- | --- | --- | --- |']
    rows = [(f'`{os.path.basename(b)}`', m) for b, m in zip(base_dirs, base_m)]
    rows += [(f'trained, first {n}', trained_first), ('trained, whole run', trained_all)]
    for name, m in rows:
        if m is None or not len(m):
            lines.append(f'| {name} | 0 | | | | |')
            continue
        lines.append(f"| {name} | {len(m)} | {m['m'].median():.6g} | {m['m'].quantile(0.99):.6g} | "
                     f"{m['m'].max():.6g} | {m['module'].value_counts().index[0]} |")
    base_tables = [(_label(b), _module_table(b)) for b in base_dirs]
    t_first, t_all = _module_table(trained_dir, n), _module_table(trained_dir)
    lines += ['', '## Per module (activation_stats)', '']
    if t_all is None:
        lines.append('The trained run has no activation_stats (older diagnose_nan.py format). Only m_t is compared; '
                     'per-module values exist only for micro-batches where the module produced m_t:')
        if trained_all is not None:
            for f in focus:
                sel = trained_all[trained_all['module'] == f]['m']
                lines.append(f"- `{f}`: m_t in {len(sel)} micro-batches, max {sel.max() if len(sel) else float('nan'):.6g}")
    else:
        mods = list(dict.fromkeys(list(focus) + list(t_all['max'].sort_values(ascending=False).index[:top])))
        header = ['module'] + [f'base {i + 1} max' for i in range(len(base_tables))] + \
                 [f'trained first {n} max', 'trained max', 'trained median of max', 'trained / base max']
        lines += ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
        for mod in mods:
            cells = [mod]
            base_max = None
            for _, tb in base_tables:
                v = tb['max'].get(mod) if tb is not None else None
                cells.append(f'{v:.6g}' if v is not None and pd.notna(v) else '')
                base_max = v if base_max is None else base_max
            tf = t_first['max'].get(mod) if t_first is not None else None
            ta, tm = t_all['max'].get(mod), t_all['median_of_max'].get(mod)
            cells += [f'{tf:.6g}' if tf is not None and pd.notna(tf) else '', f'{ta:.6g}' if pd.notna(ta) else '',
                      f'{tm:.6g}' if pd.notna(tm) else '',
                      f'{ta / base_max:.3g}' if base_max and pd.notna(ta) else '']
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += ['', 'Base runs list modules of their own condition: with LoRA disabled the lora_A / lora_B '
                      'submodules do not run and have no rows.']
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return out_path
