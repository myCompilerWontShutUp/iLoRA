"""Write compare_base_vs_trained.md from run folders only (CPU, no model).

    python scripts/compare_base_vs_trained.py --base runs/base_base_p16_<time> runs/base_lora_p16_<time> \
        --trained runs/nan_diag_p16_lr0.0001_<time> [--out <file.md>] [--focus layers.30.mlp.down_proj]
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement.report import compare_base_vs_trained  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', nargs='+', required=True)
    p.add_argument('--trained', required=True)
    p.add_argument('--out', default=None, help='Default: <first base folder>/compare_base_vs_trained.md')
    p.add_argument('--focus', nargs='*', default=['layers.30.mlp.down_proj'])
    a = p.parse_args()
    out = a.out or os.path.join(a.base[0], 'compare_base_vs_trained.md')
    if os.path.exists(out):
        raise SystemExit(f'{out} exists; pass --out to write elsewhere')
    print(compare_base_vs_trained(a.base, a.trained, out, focus=tuple(a.focus)))


if __name__ == '__main__':
    main()
