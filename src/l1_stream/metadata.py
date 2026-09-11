"""A recording that describes itself.

WHY THIS EXISTS. Two data-interpretation failures on this project had the same
cause -- metadata that lived in a filename or in someone's head instead of next
to the data:

  * ``line_drive_8m.l1raw`` was a 7 m drive. The name was wrong and nothing
    else recorded the truth, so every number derived from it was ambiguous
    between -5% and -17% error.
  * A config comment read ``4.890 vs 4.870`` with no units, no truth value and
    no recording name. Six weeks later nobody could check the subtraction, and
    the conclusion it supported turned out to be backwards.

A ``.l1raw`` is raw wire and deliberately carries no header -- that is what
lets a later parser fix apply retroactively. So the context goes in a sidecar
JSON next to it, written at capture time, when the truth is still known.

    personal/line_drive_01.l1raw
    personal/line_drive_01.l1raw.json     <- this

The sidecar also captures SOFTWARE PROVENANCE: git commit, package versions,
the config in force. ``slam-results.md`` already says to freeze the kiss-icp
version for the duration of a study, because identical APIs do not guarantee
identical numbers. Recording the version is how you find out afterwards whether
you actually did.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["RecordingMeta", "provenance", "sidecar_path"]


def sidecar_path(recording: str | Path) -> Path:
    """``x.l1raw`` -> ``x.l1raw.json``. Suffix appended, never replaced, so the
    sidecar sorts next to its recording and can never collide with another."""
    return Path(str(recording) + ".json")


def _git_commit(start: Path | None = None) -> str | None:
    """Short SHA of the repo containing this package, with ``-dirty`` if the
    tree has uncommitted changes. None when git is unavailable."""
    cwd = str(start or Path(__file__).resolve().parent)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return None


def _version(module: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(module)
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib >=3.8
        return None


def provenance() -> dict:
    """Everything needed to reproduce the software side of a recording.

    Deliberately cheap and total: every field degrades to None rather than
    raising, because a provenance failure must never cost you the recording.
    """
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "l1_stream_version": _version("l1-stream"),
        "kiss_icp_version": _version("kiss-icp"),
        "numpy_version": _version("numpy"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


@dataclass
class RecordingMeta:
    """What a recording IS -- the part the bytes cannot tell you.

    Every field is optional so that capturing something is never blocked on
    knowing everything, but the four that matter for an experiment are
    ``truth_m``, ``truth_method``, ``speed_mps`` and ``environment``. A drive
    without ``truth_method`` is a drive whose ground truth you cannot audit
    later -- which is exactly how three of five pilot line drives became
    unusable.

    ``truth_method`` carries HOW the truth was obtained, and that includes what
    it was measured from. A DISPLACEMENT is reference-free as long as the same
    part of the car sits at both marks (the sensor is rigid, so it travels
    exactly as far as the front wheels). A RANGE is measured from the sensor.
    Those two are not comparable, so say which one a value is --
    "suitcase-at-0-mark" and "wall-range-from-sensor" are different methods.
    """

    # --- what the drive was ---
    kind: str | None = None            # "line" | "loop" | "pivot" | "stationary"
    truth_m: float | None = None       # tape/laser distance, or 0.0 for a closed loop
    truth_method: str | None = None    # "suitcase-at-0-mark", "tape", "closed-loop"
    speed_mps: float | None = None
    environment: str | None = None     # "cluttered-room" | "bare-corridor" | ...
    notes: str = ""

    # --- how it was captured ---
    duration_s: float | None = None
    datagrams: int | None = None
    bytes_written: int | None = None
    port: int | None = None

    # --- software state ---
    config: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=provenance)

    def save(self, recording: str | Path) -> Path:
        path = sidecar_path(recording)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=False) + "\n")
        return path

    @classmethod
    def load(cls, recording: str | Path) -> RecordingMeta:
        """Load the sidecar for a recording. Raises FileNotFoundError if the
        recording has none -- silence would let an unlabelled file back into
        an analysis, which is the failure this module exists to prevent."""
        data = json.loads(sidecar_path(recording).read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load_or_none(cls, recording: str | Path) -> RecordingMeta | None:
        try:
            return cls.load(recording)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def summary(self) -> str:
        bits = [b for b in (
            self.kind,
            self.environment,
            f"{self.speed_mps} m/s" if self.speed_mps is not None else None,
            f"truth {self.truth_m} m ({self.truth_method})"
            if self.truth_m is not None else "NO TRUTH RECORDED",
        ) if b]
        return "  ".join(bits)
