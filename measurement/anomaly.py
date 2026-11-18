"""Anomaly tracker: decides, per hooked module and micro-batch, whether its max |value| is an event,
writes anomalies.csv and per-event dumps (anomaly_dumps/<event_id>.npz + .txt)."""
import math
import os
import time
from collections import Counter, deque

import numpy as np

from .activation import fetch_dump_tensors, parse_module_name, row_coords, short_name
from .schema import ANOMALIES, KNOWN_MASSIVE_DIMS, SEVERITIES, names

SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


class TokenInfo:
    """Input tokens of the micro-batch, used to translate a (sample, token) position."""

    def __init__(self, input_ids=None, sample_ids=None, tokenizer=None):
        self.input_ids = input_ids              # numpy [N, L] or None
        self.sample_ids = sample_ids or []
        self.tokenizer = tokenizer
        self.bos = getattr(tokenizer, 'bos_token_id', None) if tokenizer is not None else None

    @classmethod
    def from_batch(cls, batch, tokenizer):
        ids = None
        try:
            ids = batch['tokens'].input_ids.detach().cpu().numpy()
        except Exception:
            pass
        sample_ids = []
        if isinstance(batch, dict) and 'sample_idx' in batch:
            try:
                sample_ids = [int(v) for v in batch['sample_idx'].detach().cpu().tolist()]
            except Exception:
                sample_ids = []
        return cls(ids, sample_ids, tokenizer)

    def aligned(self, name, shape):
        """True if output position [sample, token] corresponds to input_ids[sample, token]."""
        return (self.input_ids is not None and name.startswith('llama_model') and len(shape) == 3
                and shape[0] == self.input_ids.shape[0] and shape[1] == self.input_ids.shape[1])

    def describe(self, name, shape, row, feature):
        """Location fields of the element at (row, feature) of an output viewed as [-1, features]."""
        sample, token = row_coords(shape, row)
        out = {'sample_idx': sample, 'sample_id': '', 'token_pos': '', 'token_id': '', 'token_str': '',
               'feature_dim': int(feature), 'is_first_token': '',
               'is_known_massive_dim': bool(shape[-1] == 4096 and int(feature) in KNOWN_MASSIVE_DIMS)}
        if 0 <= sample < len(self.sample_ids):
            out['sample_id'] = self.sample_ids[sample]
        if self.aligned(name, shape) and token >= 0:
            tid = int(self.input_ids[sample, token])
            out.update(token_pos=token, token_id=tid, token_str=self.token_str(tid),
                       is_first_token=bool(token == 0 or (self.bos is not None and tid == self.bos)))
        return out

    def token_str(self, tid):
        if self.tokenizer is None:
            return ''
        try:
            return str(self.tokenizer.convert_ids_to_tokens(int(tid)))
        except Exception:
            return ''

    def prompt(self, sample):
        if self.input_ids is None or self.tokenizer is None or not 0 <= sample < len(self.input_ids):
            return ''
        try:
            return self.tokenizer.decode([int(t) for t in self.input_ids[sample]], skip_special_tokens=False)
        except Exception:
            return ''


class AnomalyTracker:
    def __init__(self, settings, run_dir, precision, writer):
        self.s = settings
        self.run_dir = run_dir
        self.precision = precision
        self.writer = writer
        self.window = {}
        self.best = {}
        self.seen = Counter()
        self.next_id = 1
        self.dumps = 0
        self.severity_counts = Counter()
        self.print_rank = SEVERITY_RANK[settings.print_level]

    def process(self, ctx, mb, tokens):
        """ctx: micro_batch, optimizer_step, lr, amp_scale, loss. Returns the list of event rows."""
        if not len(mb):
            return []
        by_layer = {}
        for i, rec in enumerate(mb.records):
            by_layer.setdefault(parse_module_name(rec['name'])[0], []).append(i)
        events = []
        s = self.s
        for i, rec in enumerate(mb.records):
            name = rec['name']
            value, signed, row, feature, nonfinite = (float(v) for v in mb.light[i])
            hist = self.window.setdefault(name, deque(maxlen=s.ratio_window))
            median = float(np.median(hist)) if len(hist) >= s.ratio_min_history else None
            finite = math.isfinite(value) and nonfinite == 0
            rules, rank = [], -1
            if not finite:
                rules, rank = ['nonfinite'], SEVERITY_RANK['critical']
            else:
                if value > s.fp16_max:
                    rules.append('over_fp16_max')
                    rank = max(rank, SEVERITY_RANK['critical'])
                elif value >= s.high_fraction * s.fp16_max:
                    rules.append('high')
                    rank = max(rank, SEVERITY_RANK['high'])
                if value >= s.abs_threshold:
                    rules.append('abs')
                    rank = max(rank, SEVERITY_RANK['warn'])
                if median is not None and median > 0 and value >= s.ratio_threshold * median:
                    rules.append('ratio')
                    rank = max(rank, SEVERITY_RANK['warn'])
                best = self.best.get(name)
                if self.seen[name] >= s.record_warmup and best is not None and value > best:
                    rules.append('record')
                    rank = max(rank, SEVERITY_RANK['info'])
            self.seen[name] += 1
            if finite:
                hist.append(value)
                if name not in self.best or value > self.best[name]:
                    self.best[name] = value
            if rules:
                events.append(self._event(ctx, mb, i, by_layer, tokens, value, signed, row, feature,
                                          SEVERITIES[rank], rules, median))
        if events:
            self.writer.flush()
        return events

    def _event(self, ctx, mb, i, by_layer, tokens, value, signed, row, feature, severity, rules, median):
        rec = mb.records[i]
        name, shape = rec['name'], rec['shape']
        layer, mtype = parse_module_name(name)
        where = tokens.describe(name, shape, row, feature)
        event_id = self.next_id
        self.next_id += 1
        ratio = value / median if (median and math.isfinite(value)) else ''
        event = {
            'event_id': event_id, 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'micro_batch': ctx.get('micro_batch'), 'optimizer_step': ctx.get('optimizer_step'),
            'lr': ctx.get('lr'), 'amp_scale': ctx.get('amp_scale'), 'loss': ctx.get('loss'),
            'precision': self.precision, 'module': name, 'layer': layer, 'module_type': mtype, 'value': value,
            'sign': int(np.sign(signed)) if math.isfinite(signed) else '', 'severity': severity,
            'rule': ','.join(rules), 'module_window_median': median if median is not None else '',
            'ratio_to_median': ratio, 'dump_file': '',
        }
        event.update(where)
        if self.dumps < self.s.dump_limit or SEVERITY_RANK[severity] >= SEVERITY_RANK['high']:
            event['dump_file'] = self._dump(event, rec, mb, by_layer.get(layer, []), tokens)
            self.dumps += 1
        self.writer.writerow({k: event.get(k, '') for k in names(ANOMALIES)})
        self.severity_counts[severity] += 1
        if SEVERITY_RANK[severity] >= self.print_rank:
            tok = f"{where['token_pos']}({where['token_str']})" if where['token_pos'] != '' else '-'
            ratio_txt = f' ratio={ratio:.1f}x' if ratio != '' else ''
            print(f"[ANOMALY {severity}] step {ctx.get('optimizer_step')} mb {ctx.get('micro_batch')} "
                  f"{short_name(name)} value={value:.6g} token_pos={tok} dim={where['feature_dim']}"
                  f"{ratio_txt} rule={event['rule']}", flush=True)
        return event

    def _dump(self, event, rec, mb, same_layer, tokens):
        rel = os.path.join('anomaly_dumps', f"{event['event_id']}.npz")
        t = fetch_dump_tensors(rec)
        coords = []
        aligned = tokens.aligned(rec['name'], rec['shape'])
        for r, f in zip(t['top_rows'], t['top_feats']):
            sample, token = row_coords(rec['shape'], r)
            coords.append((sample, token if aligned else -1, int(f)))
        sample = event['sample_idx']
        ids = (tokens.input_ids[sample].astype(np.int64) if aligned and tokens.input_ids is not None
               else np.zeros(0, dtype=np.int64))
        np.savez(os.path.join(self.run_dir, rel),
                 token_vector=t['token_vec'].astype(np.float32), top_values=t['top_signed'].astype(np.float32),
                 top_abs=t['top_abs'].astype(np.float32), top_coords=np.asarray(coords, dtype=np.int64).reshape(-1, 3),
                 input_ids=ids, output_shape=np.asarray(rec['shape'], dtype=np.int64),
                 same_layer_modules=np.asarray([mb.records[j]['name'] for j in same_layer]),
                 same_layer_max_abs=np.asarray([mb.light[j][0] for j in same_layer], dtype=np.float64))
        with open(os.path.join(self.run_dir, rel[:-4] + '.txt'), 'w', encoding='utf-8') as f:
            for k in names(ANOMALIES):
                if k != 'dump_file':
                    f.write(f'{k}: {event.get(k, "")}\n')
            f.write(f"output_shape: {list(rec['shape'])}  dtype: {rec['dtype']}\n\n")
            prompt = tokens.prompt(sample) if aligned else ''
            f.write('prompt (decoded input_ids of the sample):\n' + (prompt or '(not a LLaMA token-aligned output)') + '\n')
        return rel
