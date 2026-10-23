"""Writes a run folder: micro-batch and optimizer-step rows, activation statistics and anomalies.
Independent of Lightning (used by measurement.callback and scripts/measure_base_activation.py)."""
import math
import os

import numpy as np

from . import schema
from .activation import ActivationMonitor, FULL, parse_module_name
from .anomaly import AnomalyTracker, TokenInfo
from .io import CsvWriter
from .report import write_anomaly_report


def batch_fields(batch, pad_id):
    """Fields derived from the batch alone (same target masking as MInterface.forward)."""
    tokens = batch['tokens']
    ids = tokens.input_ids
    out = {'placeholder_prompt': bool(batch.get('flag', False)), 'seq_len': int(ids.shape[1]),
           'mean_tokens': round(float(tokens.attention_mask.sum(dim=1).float().mean()), 1),
           'target_tokens': -1, 'min_target_tokens_per_sample': -1,
           'sample_ids': ' '.join(str(i) for i in batch['sample_idx'].tolist()) if 'sample_idx' in batch else ''}
    try:
        targets = ids.masked_fill(ids == pad_id, -100)
        targets = targets.masked_fill((tokens.token_type_ids == 0)[:, 1:], -100)
        per_sample = (targets != -100).sum(dim=1)
        out.update(target_tokens=int(per_sample.sum()), min_target_tokens_per_sample=int(per_sample.min()))
    except (RuntimeError, AttributeError):
        pass
    return out


class Recorder:
    FLUSH_EVERY = 16

    def __init__(self, run_dir, settings, precision, extra_micro_fields=(), activation_monitor=True):
        self.run_dir = run_dir
        self.s = settings
        self.precision = precision
        self.micro_fields = schema.names(schema.MICRO_BATCHES) + [f for f in extra_micro_fields
                                                                 if f not in schema.names(schema.MICRO_BATCHES)]
        self.micro = CsvWriter(os.path.join(run_dir, 'micro_batches.csv.gz'), self.micro_fields)
        self.steps = CsvWriter(os.path.join(run_dir, 'optimizer_steps.csv.gz'), schema.names(schema.OPTIMIZER_STEPS))
        self.monitor = ActivationMonitor(settings) if activation_monitor else None
        self.stats = self.anomalies = self.tracker = None
        if self.monitor is not None:
            self.stats = CsvWriter(os.path.join(run_dir, 'activation_stats.csv.gz'), schema.names(schema.ACTIVATION_STATS))
            self.anomalies = CsvWriter(os.path.join(run_dir, 'anomalies.csv'), schema.names(schema.ANOMALIES))
            self.tracker = AnomalyTracker(settings, run_dir, precision, self.anomalies)
        self.tokenizer = None
        self.n_micro = 0
        self.summary = {'micro_batches': 0, 'optimizer_steps': 0, 'nonfinite_losses': 0, 'stats_rows': 0,
                        'max_m_t': None, 'max_m_t_module': '', 'max_m_t_micro_batch': None}
        self._closed = False

    def attach(self, root, tokenizer=None, skip=()):
        self.tokenizer = tokenizer
        if self.monitor is not None:
            self.monitor.attach(root, skip=skip)
        return self.monitor.hooked_modules if self.monitor is not None else 0

    def begin_micro(self):
        self.n_micro += 1
        if self.monitor is not None:
            every = self.s.stats_every  # 0 or less: never (m_t and anomalies still every micro-batch)
            self.monitor.begin(collect_full=every > 0 and (self.n_micro - 1) % every == 0)
        return self.n_micro

    def end_micro(self, base_row, batch, ctx):
        """base_row: the caller's fields (epoch, steps, lr, loss, ...). Adds m_t, nonfinite_count,
        activation stats and anomalies, writes the micro-batch row and returns (row, events)."""
        row = dict.fromkeys(self.micro_fields, '')
        row.update(base_row)
        row['micro_batch'] = self.n_micro
        events = []
        if self.monitor is not None:
            mb = self.monitor.end()
            tokens = TokenInfo.from_batch(batch, self.tokenizer)
            row.update(self._m_t_fields(mb, tokens))
            row['stats_recorded'] = bool(mb.full)
            self._write_stats(mb, ctx, tokens)
            events = self.tracker.process(dict(ctx, micro_batch=self.n_micro), mb, tokens)
        else:
            row['stats_recorded'] = False
        loss = row.get('loss')
        if isinstance(loss, float) and not math.isfinite(loss):
            self.summary['nonfinite_losses'] += 1
        self.summary['micro_batches'] = self.n_micro
        self.micro.writerow({k: row.get(k, '') for k in self.micro_fields})
        if self.n_micro % self.FLUSH_EVERY == 0:
            self.flush()
        return row, events

    def _m_t_fields(self, mb, tokens):
        out = {'m_t': 0.0, 'm_t_module': '', 'max_abs_lowp_out': 0.0, 'max_abs_lowp_module': '', 'nonfinite_count': 0}
        if not len(mb):
            return out
        light = mb.light
        out['nonfinite_count'] = int(light[:, 4].sum())
        finite = np.isfinite(light[:, 0]) & (light[:, 4] == 0)
        if not finite.any():
            return out
        i = int(np.argmax(np.where(finite, light[:, 0], -np.inf)))
        rec = mb.records[i]
        value, signed, row, feature = (float(v) for v in light[i, :4])
        layer, mtype = parse_module_name(rec['name'])
        where = tokens.describe(rec['name'], rec['shape'], row, feature)
        out.update({'m_t': value, 'm_t_module': rec['name'], 'max_abs_lowp_out': value,
                    'max_abs_lowp_module': rec['name'], 'm_t_layer': layer, 'm_t_module_type': mtype,
                    'm_t_sample_idx': where['sample_idx'], 'm_t_sample_id': where['sample_id'],
                    'm_t_token_pos': where['token_pos'], 'm_t_token_id': where['token_id'],
                    'm_t_token_str': where['token_str'], 'm_t_feature_dim': where['feature_dim'],
                    'm_t_sign': int(np.sign(signed)), 'm_t_is_first_token': where['is_first_token'],
                    'm_t_is_known_massive_dim': where['is_known_massive_dim']})
        if self.summary['max_m_t'] is None or value > self.summary['max_m_t']:
            self.summary.update(max_m_t=value, max_m_t_module=rec['name'], max_m_t_micro_batch=self.n_micro)
        return out

    def _write_stats(self, mb, ctx, tokens):
        for i, full in mb.full.items():
            rec = mb.records[i]
            layer, mtype = parse_module_name(rec['name'])
            value, _, row, feature, nonfinite = mb.light[i]
            token = int(row) % rec['shape'][1] if tokens.aligned(rec['name'], rec['shape']) else -1
            stats = dict(zip(FULL, (float(v) for v in full)))
            self.stats.writerow({
                'micro_batch': self.n_micro, 'optimizer_step': ctx.get('optimizer_step'), 'module': rec['name'],
                'layer': layer, 'module_type': mtype, 'dtype': rec['dtype'], 'numel': rec['numel'],
                'max_abs': float(value), 'max_token_pos': token, 'max_feature_dim': int(feature),
                'mean_abs': stats['mean_abs'], 'std': stats['std'], 'median_abs': stats['median_abs'],
                'p90_abs': stats['p90_abs'], 'p99_abs': stats['p99_abs'], 'p999_abs': stats['p999_abs'],
                'nonfinite_count': int(nonfinite), 'frac_over_1e3': stats['frac_over_1e3'],
                'frac_over_1e4': stats['frac_over_1e4']})
            self.summary['stats_rows'] += 1

    def write_step(self, step_row):
        self.steps.writerow({k: step_row.get(k, '') for k in schema.names(schema.OPTIMIZER_STEPS)})
        self.summary['optimizer_steps'] += 1
        self.flush()

    def flush(self):
        for w in (self.micro, self.steps, self.stats, self.anomalies):
            if w is not None:
                w.flush()

    def close(self):
        """Close the files and write README.txt and anomaly_report.md. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self.monitor is not None:
            self.monitor.detach()
        for w in (self.micro, self.steps, self.stats, self.anomalies):
            if w is not None:
                w.close()
        with open(os.path.join(self.run_dir, 'README.txt'), 'w', encoding='utf-8') as f:
            f.write(schema.readme_text())
        if self.anomalies is not None:
            write_anomaly_report(self.run_dir)
