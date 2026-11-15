import csv, os, time
from contextlib import contextmanager

"""`**kw` in `add` and `timer` can be `step id`"""

_DEFAULT = {"task_id": -1, "perturbation_id": "none"}
_ctx = dict(_DEFAULT)
_rows = {"experiment": [], "episode": [], "step": []}

_folder = None
_files, _writers, _keys = {}, {}, {}    # -- per level: open file, csv writer, header columns

def set_ctx(**kw):
    _ctx.update(kw)

def reset_ctx():
    _ctx.clear()
    _ctx.update(_DEFAULT)

def set_folder(folder):
    global _folder
    _folder = folder
    os.makedirs(folder, exist_ok=True)
    for level, rows in _rows.items():
        if rows:
            _rewrite(level)

def _rewrite(level):
    if level in _files:
        _files[level].close()
    rows = _rows[level]
    _keys[level] = list(dict.fromkeys(k for r in rows for k in r))
    f = open(f"{_folder}/{level}.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=_keys[level], restval="")   # -- missing columns are treated as empty cell
    w.writeheader()
    w.writerows(rows)
    f.flush()
    _files[level], _writers[level] = f, w

def add(level, stage, dt, **kw):
    row = {**_ctx, "stage": stage, "time_s": dt, **kw}
    _rows.setdefault(level, []).append(row)
    if _folder is None:
        return
    if level not in _files or not set(row) <= set(_keys[level]):
        _rewrite(level)
    else:
        _writers[level].writerow(row)
        _files[level].flush()

@contextmanager
def timer(level, stage, **kw):
    t0 = time.perf_counter()
    yield
    add(level, stage, time.perf_counter() - t0, **kw)

def save(folder=None):
    if _folder is None:
        set_folder(folder)
    for f in _files.values():
        f.close()
    _files.clear()