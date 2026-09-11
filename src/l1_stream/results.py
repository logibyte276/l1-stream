"""One row per run, in a CSV, instead of terminal scrollback.

WHY THIS EXISTS. The pilot study's results lived in copied-and-pasted terminal
output. That was enough to keep the findings, but it made three things
impossible: you could not sort by a factor, you could not compute a mean across
repeats, and when a machine was wiped the analysis had to be re-read by eye out
of a document.

A factorial design makes that much worse. Three speeds x two environments x
three repeats is 18 drives, and each one replays at four voxel sizes and two
deskew settings -- 144 rows. That is a spreadsheet, not scrollback.

    log = ResultsLog("personal/results.csv")
    log.append(run_row)          # dict; new keys are absorbed, not dropped

The header is the union of every key ever written. Adding a metric later
rewrites the file with the new column and blanks for older rows, rather than
silently dropping it -- an analysis that quietly loses a column is worse than
one that fails.
"""

from __future__ import annotations

import csv
from pathlib import Path

__all__ = ["ResultsLog", "flatten", "run_row"]


def run_row(path, run, meta=None, truth=None, **extra) -> dict:
    """Build one results row from a replay.

    ``run`` is duck-typed rather than imported, which keeps this module free of
    any l1_stream dependency -- it stays a CSV writer that happens to know the
    shape of a result, not a second place where the pipeline is defined.

    ``meta`` is an optional :class:`~l1_stream.metadata.RecordingMeta`. Folding
    it in here is the point: the experimental factors (truth, speed,
    environment) end up in the same row as the metrics, so the table can be
    grouped without a second lookup. A row with no sidecar gets blank factor
    columns, which is a visible gap rather than a silent one.
    """
    net, plen = run.net_displacement, run.path_length
    row = {
        "recording": Path(path).name,
        # --- experimental factors, from the sidecar ---
        "kind": getattr(meta, "kind", None),
        "environment": getattr(meta, "environment", None),
        "speed_mps": getattr(meta, "speed_mps", None),
        "truth_m": getattr(meta, "truth_m", None),
        "truth_method": getattr(meta, "truth_method", None),
        # --- what was run ---
        **{f"cfg.{k}": v for k, v in run.config.items()},
        # --- what came out ---
        "frames": len(run.xyz),
        "points_per_frame": round(float(run.sizes.mean()), 1),
        "path_length_m": round(plen, 4),
        "net_displacement_m": round(net, 4),
        "loop_pct_of_path": round(100 * net / max(plen, 1e-9), 3),
        "jitter_mm": round(run.jitter_mm, 2),
        "rotation_deg_mean": round(float(run.rotation_deg.mean()), 3),
        "threshold_final": round(float(run.thresholds[-1]), 4),
        "z_span_m": round(float(run.xyz[:, 2].max() - run.xyz[:, 2].min()), 4),
        # --- cost. replay_host is not optional context: a timing from a
        # desktop says nothing about whether the Orin keeps up. ---
        "replay_host": getattr(run, "replay_host", None),
        "replay_seconds": round(getattr(run, "replay_seconds", 0.0), 3),
        "ms_per_frame": round(getattr(run, "ms_per_frame", 0.0), 2),
        "budget_pct": round(getattr(run, "budget_pct", 0.0), 1),
    }
    if truth is None:
        truth = getattr(meta, "truth_m", None)
    row["truth_used_m"] = truth
    if truth:
        row["scale_error_pct"] = round(100 * (net - truth) / truth, 3)
    row.update(extra)
    return row


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into ``a.b`` keys so a config or a provenance block
    can go straight into a CSV column each. Lists become their repr, because a
    list in a results table is almost always a mistake worth seeing."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


class ResultsLog:
    """Append-only CSV whose header grows to fit whatever you write."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as fh:
            return list(csv.DictReader(fh))

    def fieldnames(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as fh:
            return next(csv.reader(fh), [])

    def append(self, row: dict) -> None:
        """Add one row. Keys not yet in the header trigger a rewrite that adds
        the column; existing rows get an empty cell for it."""
        row = flatten(row)
        existing = self.fieldnames()

        if not existing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            return

        new_keys = [k for k in row if k not in existing]
        if not new_keys:
            with self.path.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=existing).writerow(row)
            return

        # Widening the table. Read, extend, rewrite -- these files are small
        # (hundreds of rows), and correctness beats an append-only fast path.
        old = self.rows()
        fields = existing + new_keys
        with self.path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in old:
                w.writerow(r)
            w.writerow(row)
