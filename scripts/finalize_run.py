"""Write README.txt and anomaly_report.md for a run folder whose process was killed before it could
(CPU only, reads the flushed raw files). Never touches the raw data files.

    python scripts/finalize_run.py <run_dir>
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from measurement.io import read_json, write_json  # noqa: E402
from measurement.report import write_anomaly_report  # noqa: E402
from measurement.schema import readme_text  # noqa: E402


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = sys.argv[1]
    readme = os.path.join(run_dir, 'README.txt')
    if not os.path.exists(readme):
        with open(readme, 'w', encoding='utf-8') as f:
            f.write(readme_text())
    if os.path.exists(os.path.join(run_dir, 'anomalies.csv')):
        write_anomaly_report(run_dir)
    meta_path = os.path.join(run_dir, 'meta.json')
    meta = read_json(meta_path)
    if meta and not meta.get('end_time_utc'):
        meta.update(status='killed (finalized afterwards by scripts/finalize_run.py)',
                    finalized_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(meta_path, meta)
    print(f'finalized {run_dir}')


if __name__ == '__main__':
    main()
