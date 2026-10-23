"""Column names and meanings of every file in a run folder (README.txt is generated from here),
and module-name helpers. No torch import."""
import re

FP16_MAX = 65504.0
KNOWN_MASSIVE_DIMS = (1415, 2533)  # LLaMA2-7B, Sun et al. 2024 (Massive Activations in LLMs)
MODULE_TYPES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
SEVERITIES = ('info', 'warn', 'high', 'critical')
_LAYER = re.compile(r'(?:^|\.)layers\.(\d+)(?:\.|$)')
_PREFIXES = ('llama_model.base_model.model.model.', 'llama_model.base_model.model.', 'llama_model.')


def parse_module_name(name):
    """(decoder layer or -1, q_proj ... down_proj or 'other'); '#n' call suffixes are ignored."""
    base = name.split('#', 1)[0]
    m = _LAYER.search(base)
    last = base.rsplit('.', 1)[-1]
    return (int(m.group(1)) if m else -1), (last if last in MODULE_TYPES else 'other')


def short_name(name):
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

MICRO_BATCHES = [
    ('epoch', 'training epoch (0-based)'),
    ('micro_batch', 'micro-batch index in this run (1-based)'),
    ('in_optimizer_step', 'optimizer step this micro-batch belongs to (1-based)'),
    ('steps_completed', 'optimizer steps completed after this micro-batch'),
    ('lr', 'learning rate of parameter group 0 (projector / LoRA); the router group uses 0.3x'),
    ('loss', 'language-model loss of this micro-batch (not divided by accumulation)'),
    ('loss_finite', 'loss is finite'),
    ('placeholder_prompt', 'collator used [PH] placeholders instead of SASRec item embeddings'),
    ('seq_len', 'padded token length L of the batch'),
    ('mean_tokens', 'mean non-padding token count per sample'),
    ('target_tokens', 'number of target (answer) tokens in the batch'),
    ('min_target_tokens_per_sample', 'smallest per-sample target token count'),
    ('gate_finite', 'router gate values are all finite'),
    ('gate_min', 'smallest router gate value'),
    ('gate_max', 'largest router gate value'),
    ('max_abs_lowp_out', 'same as m_t (kept from the earlier diagnose_nan.py format)'),
    ('max_abs_lowp_module', 'same as m_t_module (kept from the earlier format)'),
    ('amp_scale', 'GradScaler loss scale after this micro-batch (empty for bf16)'),
    ('seconds', 'wall time of this micro-batch, s'),
    ('sample_ids', 'dataset sample ids of the batch, space separated'),
    ('m_t', 'largest finite |value| over the outputs of all fp16/bf16-output modules'),
    ('m_t_module', 'module that produced m_t (#2, #3 ... = 2nd, 3rd call of that module)'),
    ('m_t_layer', 'decoder layer of m_t_module (-1 if not in a layer)'),
    ('m_t_module_type', 'q_proj ... down_proj, or other'),
    ('m_t_sample_idx', 'position of the sample inside the batch'),
    ('m_t_sample_id', 'dataset sample id of that sample'),
    ('m_t_token_pos', 'token position (only for LLaMA modules shaped [batch, L, features])'),
    ('m_t_token_id', 'input token id at that position'),
    ('m_t_token_str', 'that token, decoded by the tokenizer'),
    ('m_t_feature_dim', 'index along the last (feature) dimension'),
    ('m_t_sign', 'sign of the value (+1 / -1)'),
    ('m_t_is_first_token', 'token position 0 or the BOS token'),
    ('m_t_is_known_massive_dim', 'feature dim is 1415 or 2533 of a 4096-wide output'),
    ('nonfinite_count', 'inf/NaN elements over all fp16/bf16 module outputs of this micro-batch'),
    ('stats_recorded', 'activation_stats.csv.gz has rows for this micro-batch'),
]

OPTIMIZER_STEPS = [
    ('epoch', 'training epoch (0-based)'),
    ('optimizer_step', 'optimizer step (1-based)'),
    ('lr', 'learning rate of parameter group 0'),
    ('mean_loss', 'mean finite micro-batch loss of the step'),
    ('finite_losses', 'micro-batches of the step with a finite loss'),
    ('micro_batches', 'micro-batches in the step'),
    ('grad_norm', 'L2 norm of all trainable-parameter gradients after unscaling'),
    ('nonfinite_grad_params', 'parameter tensors with a non-finite gradient'),
    ('first_nonfinite_grad_param', 'first such parameter'),
    ('skipped_by_grad_scaler', 'GradScaler skipped the step (loss scale decreased)'),
    ('scale_before', 'loss scale before the step'),
    ('scale_after', 'loss scale after the step'),
    ('params_finite_before_step', 'all trainable parameters finite before the step'),
    ('params_finite_after_step', 'all trainable parameters finite after the step'),
    ('nonfinite_params', 'parameter tensors with non-finite values after the step'),
    ('first_nonfinite_param', 'first such parameter'),
    ('max_abs_param', 'largest |value| over trainable parameters after the step'),
]

ACTIVATION_STATS = [
    ('micro_batch', 'micro-batch index'),
    ('optimizer_step', 'optimizer step the micro-batch belongs to'),
    ('module', 'module name (#2 ... for repeated calls)'),
    ('layer', 'decoder layer (-1 if none)'),
    ('module_type', 'q_proj ... down_proj, or other'),
    ('dtype', 'dtype of the output tensor'),
    ('numel', 'number of elements'),
    ('max_abs', 'largest |value| (inf/NaN if the output has non-finite values)'),
    ('max_token_pos', 'token position of max_abs (-1 if not a [batch, L, features] LLaMA output)'),
    ('max_feature_dim', 'feature index of max_abs'),
    ('mean_abs', 'mean |value| over all elements, float32 accumulation'),
    ('std', 'standard deviation of the values over all elements, float32'),
    ('median_abs', '50th percentile of |value| (random sample, see meta.json)'),
    ('p90_abs', '90th percentile of |value| (sample)'),
    ('p99_abs', '99th percentile of |value| (sample)'),
    ('p999_abs', '99.9th percentile of |value| (sample)'),
    ('nonfinite_count', 'inf/NaN elements'),
    ('frac_over_1e3', 'fraction of elements with |value| > 1e3'),
    ('frac_over_1e4', 'fraction of elements with |value| > 1e4'),
]

ANOMALIES = [
    ('event_id', 'event number; anomaly_dumps/<event_id>.npz/.txt when dumped'),
    ('time', 'UTC time'),
    ('micro_batch', 'micro-batch index'),
    ('optimizer_step', 'optimizer step the micro-batch belongs to'),
    ('lr', 'learning rate'),
    ('amp_scale', 'GradScaler loss scale'),
    ('loss', 'micro-batch loss'),
    ('precision', '16 (fp16 AMP) or bf16'),
    ('module', 'module name'),
    ('layer', 'decoder layer'),
    ('module_type', 'q_proj ... down_proj, or other'),
    ('value', 'max |value| of the module output in this micro-batch'),
    ('sign', 'sign of that value'),
    ('severity', 'info < warn < high < critical'),
    ('rule', 'rules that fired: nonfinite, over_fp16_max, high, abs, ratio, record'),
    ('sample_idx', 'position in the batch'),
    ('sample_id', 'dataset sample id'),
    ('token_pos', 'token position'),
    ('token_id', 'input token id'),
    ('token_str', 'decoded token'),
    ('feature_dim', 'feature index'),
    ('is_first_token', 'token position 0 or BOS'),
    ('is_known_massive_dim', 'feature dim 1415 or 2533 of a 4096-wide output'),
    ('module_window_median', 'median of this module\'s max |value| over the previous window'),
    ('ratio_to_median', 'value / module_window_median'),
    ('dump_file', 'dump path relative to the run folder (empty if not dumped)'),
]

GPU = [
    ('timestamp', 'nvidia-smi timestamp'),
    ('memory_used_mib', 'GPU memory used, MiB (includes the PyTorch cache)'),
    ('memory_total_mib', 'GPU memory total, MiB'),
    ('utilization_gpu_pct', 'GPU utilization, %'),
    ('power_draw_w', 'power draw, W'),
]

DUMP_CONTENTS = [
    ('token_vector', 'float32 [features]: the module output at (sample, token) of the event'),
    ('top_values', 'float32 [32]: the 32 largest |value| of the same output tensor (signed)'),
    ('top_coords', 'int64 [32, 3]: (sample, token, feature) of top_values; token -1 if n/a'),
    ('input_ids', 'int64 [L]: input ids of the sample (empty if not a LLaMA token output)'),
    ('same_layer_modules / same_layer_max_abs', 'max |value| of the other modules of that layer in the micro-batch'),
    ('<event_id>.txt', 'event fields and the decoded prompt of the sample'),
]


def names(spec):
    return [n for n, _ in spec]


def readme_text(extra_sections=()):
    lines = ['Run folder written by the measurement package (experiments/IMPLEMENTATION_NOTES.md 11j).',
             '', 'Files', '-----',
             'meta.json                 git commit / status, command line, config, environment, settings, paths',
             'micro_batches.csv.gz      one row per micro-batch',
             'optimizer_steps.csv.gz    one row per optimizer step',
             'activation_stats.csv.gz   one row per (module, recorded micro-batch)',
             'anomalies.csv             one row per anomaly event',
             'anomaly_dumps/            per-event npz + txt',
             'anomaly_report.md         summary of anomalies.csv (written at the end)',
             'gpu.csv                   GPU resources, when scripts/gpu_monitor.py ran',
             'figures/                  written by analysis/plot_measurement.py and scripts/plot_live.py',
             '',
             'Gzip files are flushed during the run. After a hard kill the last flush is readable with',
             'measurement.io.read_frame(); plain gzip tools may report an unexpected end of file.', '']
    for title, spec in (('micro_batches.csv.gz', MICRO_BATCHES), ('optimizer_steps.csv.gz', OPTIMIZER_STEPS),
                        ('activation_stats.csv.gz', ACTIVATION_STATS), ('anomalies.csv', ANOMALIES),
                        ('gpu.csv', GPU), ('anomaly_dumps/<event_id>.npz', DUMP_CONTENTS)):
        lines += [title, '-' * len(title)]
        lines += [f'{n:<34} {d}' for n, d in spec]
        lines.append('')
    for title, body in extra_sections:
        lines += [title, '-' * len(title), body, '']
    lines += ['Commands', '--------',
              'python scripts/gpu_monitor.py <run_dir> --interval 60          # separate process, stops when the run ends',
              'python scripts/plot_live.py <run_dir> --watch 300               # live figure + warnings',
              'python analysis/plot_measurement.py <run_dir> [--compare <run_dir2> ...] [--base <base_run_dir>]',
              'python scripts/inspect_anomaly.py <run_dir> <event_id>',
              'python scripts/finalize_run.py <run_dir>                        # README + report after a hard kill',
              '']
    return '\n'.join(lines)
