"""File helpers that survive a run being killed: periodically flushed (gzip) CSV writers and readers
that accept a truncated gzip stream. No torch import."""
import csv
import gzip
import io
import json
import os
import zlib


class CsvWriter:
    """csv.DictWriter over a plain or gzip text file. `flush()` makes everything written so far
    readable after a hard kill (gzip: a zlib sync flush plus an OS-level write)."""

    def __init__(self, path, fieldnames):
        if os.path.exists(path):
            raise FileExistsError(f'{path} exists; run folders are never overwritten')
        self.path = path
        if path.endswith('.gz'):
            self._f = gzip.open(path, 'wt', newline='', encoding='utf-8', compresslevel=6)
        else:
            self._f = open(path, 'w', newline='', encoding='utf-8')
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames, extrasaction='raise')
        self._w.writeheader()
        self.rows = 0
        self.flush()

    def writerow(self, row):
        self._w.writerow(row)
        self.rows += 1

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

    def flush(self):
        if not self._f.closed:
            self._f.flush()

    def close(self):
        if not self._f.closed:
            self._f.close()


def read_text(path):
    """Whole file as text. A gzip file cut off by a kill is read up to its last complete flush, and
    a trailing partial line is dropped."""
    if not path.endswith('.gz'):
        with open(path, encoding='utf-8', newline='') as f:
            return f.read()
    chunks = []
    with open(path, 'rb') as raw:
        gz = gzip.GzipFile(fileobj=raw)
        try:
            while True:
                chunk = gz.read(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
        except (EOFError, zlib.error, OSError):
            pass
    text = b''.join(chunks).decode('utf-8', errors='replace')
    if text and not text.endswith('\n'):
        text = text[:text.rfind('\n') + 1]
    return text


def find(run_dir, stem):
    """`<stem>.csv.gz` or `<stem>.csv` inside run_dir (older diagnose_nan.py folders use .csv)."""
    for name in (f'{stem}.csv.gz', f'{stem}.csv'):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return None


def read_rows(path):
    if path is None or not os.path.exists(path):
        return []
    return list(csv.DictReader(io.StringIO(read_text(path))))


def read_frame(path):
    """pandas DataFrame of a (possibly truncated) CSV, or None if the file is missing or empty.
    Floats are parsed with float_precision='round_trip': pandas' default fast parser changes the
    last bit of about a third of the values, round_trip returns exactly what was written."""
    import pandas as pd
    if path is None or not os.path.exists(path):
        return None
    text = read_text(path)
    if not text.strip():
        return None
    frame = pd.read_csv(io.StringIO(text), low_memory=False, float_precision='round_trip')
    return frame if len(frame) else None


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=repr, allow_nan=True)
    os.replace(tmp, path)


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)
