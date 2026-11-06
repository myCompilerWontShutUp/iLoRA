"""Figures of a measurement run folder, from its raw files only (no GPU, no model), PNG 300 dpi into
<run_dir>/figures/. Figures whose data is missing are skipped with a log line.

    python analysis/plot_measurement.py <run_dir> [--compare <run_dir2> ...] [--base <base_run> ...]
                                        [--module layers.30.mlp.down_proj]

Figures: m_t.png, module_max_abs.png, layer_module_heatmap.png, quantile_trend_<module>.png,
loss_grad_norm.png, loss_scale.png, anomaly_positions.png, gpu.png, base_vs_trained.png.
Colors: fp16 purple, bf16 teal, limits red dashed, base (before training) orange, trained blue.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement import plots  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--compare', nargs='*', default=[], help='Other runs overlaid on m_t / loss / grad norm / scale.')
    p.add_argument('--base', nargs='*', default=[], help='measure_base_activation.py run folders.')
    p.add_argument('--module', default=None, help='Module for the quantile trend (default: largest max).')
    p.add_argument('--out_dir', default=None, help='Default: <run_dir>/figures')
    a = p.parse_args(argv)
    run = plots.Run(a.run_dir)
    others = [plots.Run(d) for d in a.compare]
    bases = [plots.Run(d) for d in a.base]
    out = a.out_dir or os.path.join(a.run_dir, 'figures')
    written = [plots.fig_m_t([run] + others, out), plots.fig_modules(run, out), plots.fig_layer_heatmap(run, out),
               plots.fig_quantile_trend(run, out, a.module), plots.fig_loss_grad([run] + others, out),
               plots.fig_loss_scale([run] + others, out), plots.fig_anomaly_positions(run, out),
               plots.fig_gpu(run, out), plots.fig_base_vs_trained(run, bases, out)]
    print(f'[plot] {sum(w is not None for w in written)} figure(s) in {out}')
    return written


if __name__ == '__main__':
    main()
