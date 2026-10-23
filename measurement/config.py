"""Measurement settings and their command-line flags (shared by main.py, diagnose_nan.py and
measure_base_activation.py)."""
import argparse
from dataclasses import asdict, dataclass

from .schema import FP16_MAX


@dataclass
class MeasureSettings:
    stats_every: int = 1             # activation_stats.csv.gz every N micro-batches, 0 = never (m_t and anomalies: always)
    quantile_sample: int = 100_000   # elements sampled per output for the quantiles
    sample_seed: int = 1234          # seed of the measurement's own generator (training RNG is never used)
    topk: int = 32                   # largest |values| kept per output for dumps
    module_regex: str = ''           # only hook modules whose name matches (empty = all)
    abs_threshold: float = 16384.0   # rule "abs": max |value| >= this (25% of the fp16 maximum)
    high_fraction: float = 0.5       # rule "high": >= high_fraction * fp16 maximum -> severity high
    fp16_max: float = FP16_MAX       # rule "over_fp16_max": > this -> severity critical
    ratio_threshold: float = 4.0     # rule "ratio": >= ratio_threshold * module's window median
    ratio_window: int = 200          # micro-batches in that window
    ratio_min_history: int = 20      # window entries needed before "ratio" can fire
    record_warmup: int = 100         # rule "record" (new per-module maximum) only after this many micro-batches
    dump_limit: int = 200            # dumps for severity info/warn; high and critical are always dumped
    print_level: str = 'warn'        # print events of this severity and above

    def as_dict(self):
        return asdict(self)


def add_measurement_args(parser):
    g = parser.add_argument_group('measurement')
    d = MeasureSettings()
    g.add_argument('--stats_every', type=int, default=d.stats_every)
    g.add_argument('--quantile_sample', type=int, default=d.quantile_sample)
    g.add_argument('--sample_seed', type=int, default=d.sample_seed)
    g.add_argument('--topk', type=int, default=d.topk)
    g.add_argument('--module_regex', default=d.module_regex)
    g.add_argument('--abs_threshold', type=float, default=d.abs_threshold)
    g.add_argument('--high_fraction', type=float, default=d.high_fraction)
    g.add_argument('--ratio_threshold', type=float, default=d.ratio_threshold)
    g.add_argument('--ratio_window', type=int, default=d.ratio_window)
    g.add_argument('--ratio_min_history', type=int, default=d.ratio_min_history)
    g.add_argument('--record_warmup', type=int, default=d.record_warmup)
    g.add_argument('--dump_limit', type=int, default=d.dump_limit)
    g.add_argument('--print_level', default=d.print_level, choices=['info', 'warn', 'high', 'critical'])
    g.add_argument('--no_activation_monitor', action='store_true',
                   help='No activation hooks: no m_t, activation_stats or anomalies (for speed baselines).')
    g.add_argument('--measure_root', default='runs', help='Parent folder of run folders.')
    g.add_argument('--measure_name', default=None, help='Run folder name prefix.')
    return parser


def settings_from_args(args):
    return MeasureSettings(**{k: getattr(args, k) for k in MeasureSettings().as_dict()})


def measurement_parser():
    """Stand-alone parser used by main.py so that measurement flags never enter its args/hparams."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # never claims a main.py flag by prefix
    p.add_argument('--measure', action='store_true',
                   help='Write a run folder with step/micro-batch logs, activation stats and anomalies.')
    add_measurement_args(p)
    return p
