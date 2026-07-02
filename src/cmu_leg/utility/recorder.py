"""Per-tick numeric trace for one output stream.

One Recorder per output you want to log (state, torque, drive command, ...).
It is buffered: `append` only touches memory, so it is safe to call inside the
1 kHz loop; all disk I/O happens in `save` after the loop. Columns are taken
from the dataclass fields, so it works on any flat output type without hand-
writing a dict each time. Tuple fields (e.g. an IMU's 3-axis accel) expand into
`name_0, name_1, ...` columns. A leading `t` column timestamps every row, so the
per-stream CSVs all share the loop's timeline and can be joined afterward.

This is the numeric counterpart to Python's `logging` (text events); use that
for mode transitions / faults, this for the per-tick trace.

    state_log = Recorder("state")
    ...
    state_log.append(t, state)          # in the loop (memory only)
    ...
    state_log.save(make_run_dir("standing"))   # once, after the loop
"""

import csv
from dataclasses import fields
from pathlib import Path


class Recorder:
    def __init__(self, name: str):
        self._name = name
        self._cols: list[str] | None = None   # fixed on first append
        self._rows: list[list] = []

    def append(self, t: float, obj) -> None:
        flat = self._flatten(obj)
        if self._cols is None:
            self._cols = ["t", *flat.keys()]
        self._rows.append([t, *flat.values()])

    def column(self, name: str) -> list:
        """Return one recorded column by name, for post-loop plotting. `name` is
        `t` or any flattened field (e.g. `ankle_angle`, `thigh_acc_0`)."""
        if self._cols is None:
            return []
        i = self._cols.index(name)
        return [row[i] for row in self._rows]

    def save(self, out_dir) -> Path | None:
        if self._cols is None:
            return None   # nothing was recorded
        path = Path(out_dir) / f"{self._name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._cols)
            w.writerows(self._rows)
        print("saved", path)
        return path

    @staticmethod
    def _flatten(obj) -> dict:
        out: dict = {}
        for fld in fields(obj):
            val = getattr(obj, fld.name)
            if isinstance(val, (tuple, list)):
                for i, v in enumerate(val):
                    out[f"{fld.name}_{i}"] = v
            else:
                out[fld.name] = val
        return out
