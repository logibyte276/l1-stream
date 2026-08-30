"""Non-overlapping frame assembly for registration.

:class:`RotatedScanAccumulator` exists to make a *picture*: a rolling window
that always shows the last N scans. A registration algorithm needs something
different -- a sequence of disjoint frames, each carrying per-point times.

It is tempting to reuse the accumulator by calling ``get_points()`` and then
``reset()``. That does remove the overlap, but three problems remain:

1. **The timestamps are already gone.** ``get_points()`` returns ``(N, 3)``.
   KISS-ICP's ``register_frame(frame, timestamps)`` needs a per-point time to
   deskew, and ``drop_zero_returns`` filters inside the accumulator, so the
   mask that would realign times with points is not recoverable afterwards.
   Without times you must set ``deskew=False``, which throws away the
   intra-frame motion correction.
2. **``reset()`` also clears the pending queue.** That queue holds scans which
   arrived before the IMU sample that timestamps them -- structurally, the
   *newest* scans in the batch. Clearing it every frame discards a systematic
   slice of each frame's trailing edge, and because ``reset()`` does not touch
   the counters, ``scans_unmatched`` will not report the loss.
3. **Frame spans would follow your call cadence, not capture time.** Whatever
   happened to arrive since the last call becomes a "frame", so spans jitter
   with loop timing and GIL scheduling. KISS-ICP's constant-velocity motion
   model assumes roughly uniform intervals: alternating 0.15 s and 0.31 s
   frames make every initial guess wrong by that ratio, and the normalised
   deskew timestamps are stretched differently frame to frame.

So this class re-does the IMU matching, but cuts frames on **capture time**,
keeps per-point times alongside the points, and never drops a pending scan.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .protocol import LidarIMU, LidarScan
from .rotation import rotate_points

logger = logging.getLogger(__name__)

__all__ = ["Frame", "FrameAssembler"]

#: Used when rotate_with_imu is False, where it is never actually applied.
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


@dataclass
class Frame:
    """One registration frame: points, per-point times, and its time span."""

    points: np.ndarray       #: (N, 3) float64, C-contiguous
    timestamps: np.ndarray   #: (N,) float64 in [0, 1]; 1.0 is the LAST point
    t_start: float           #: absolute capture time of the first point
    t_end: float             #: absolute capture time of the last point
    n_scans: int

    @property
    def span(self) -> float:
        """Wall-clock duration. Watch this: KISS-ICP's constant-velocity
        motion model assumes frames are roughly uniform in time."""
        return self.t_end - self.t_start

    def __len__(self) -> int:
        return len(self.points)


class FrameAssembler:
    """Turn a scan/IMU stream into disjoint frames with per-point timestamps.

    Args:
        frame_duration: Target frame length in seconds. This is the parameter
            worth sweeping first. The L1 emits ~21,600 points/s, so 0.2 s is
            roughly 4,300 points -- thin by LiDAR-odometry standards (a
            64-beam sweep is ~130k), which is why one azimuth revolution
            (~1,960 points at 11 Hz) is usually too sparse to register.
            Longer frames buy geometry at the cost of more intra-frame motion
            for deskew to absorb.
        max_time_gap: Largest scan-to-IMU timestamp mismatch accepted.
        rotate_with_imu: Rotate each scan into the common IMU frame before
            stacking. See the note in the module docstring of ``rotation`` --
            and run this both ways, it is a real open question for this rig.
            When False the IMU is not consulted at all, so the assembler runs
            on a scan-only stream; ``imu_samples`` may be empty.
        drop_zero_returns: Strip points at exactly the origin. These are rays
            that never came back, and they pin a scan match with a fake dense
            blob at the sensor position.
        min_points: Frames thinner than this are dropped and counted rather
            than fed to registration, where they would produce a bad pose
            instead of no pose.
    """

    def __init__(
        self,
        frame_duration: float = 0.2,
        max_time_gap: float = 0.01,
        rotate_with_imu: bool = True,
        drop_zero_returns: bool = True,
        min_points: int = 300,
        pending_maxlen: int = 400,
    ):
        if frame_duration <= 0:
            raise ValueError("frame_duration must be > 0")
        if max_time_gap <= 0:
            raise ValueError("max_time_gap must be > 0")

        self.frame_duration = float(frame_duration)
        self.max_time_gap = float(max_time_gap)
        self.rotate_with_imu = bool(rotate_with_imu)
        self.drop_zero_returns = bool(drop_zero_returns)
        self.min_points = int(min_points)

        self._pending: deque[LidarScan] = deque(maxlen=pending_maxlen)
        # (scan_stamp, xyz, absolute_times) kept until a frame closes.
        self._batch: list[tuple[float, np.ndarray, np.ndarray]] = []

        self.frames_emitted = 0
        self.frames_too_sparse = 0
        self.scans_unmatched = 0
        self.scans_ingested = 0
        self.points_ingested = 0

    # --- ingestion ---

    def add(
        self, scans: Iterable[LidarScan], imu_samples: Sequence[LidarIMU]
    ) -> list[Frame]:
        """Feed a batch of scans plus the current IMU window.

        Returns every frame that closed as a result -- usually zero or one,
        but more if the caller fell behind. Scans whose IMU sample has not
        arrived yet are held, not dropped.
        """
        self._pending.extend(scans)
        if not self._pending:
            return self._emit_ready()

        if not self.rotate_with_imu:
            # The quaternion would be discarded anyway, so requiring a match
            # here would make this mode depend on a stream it does not use.
            # Worse, with no IMU at all every scan would wait in _pending
            # forever and no frame would ever close -- a silent hang.
            for scan in self._pending:
                self._ingest(scan, _IDENTITY_QUAT)
            self._pending.clear()
            return self._emit_ready()

        if imu_samples:
            raw = np.fromiter(
                (s.stamp for s in imu_samples), dtype=np.float64, count=len(imu_samples)
            )
            # UDP does not guarantee delivery order, so sort before searching.
            order = np.argsort(raw, kind="stable")
            stamps = raw[order]
            quats = np.asarray(
                [imu_samples[i].quaternion for i in order], dtype=np.float64
            )
        else:
            stamps = None
            quats = None

        keep: list[LidarScan] = []
        for scan in self._pending:
            if stamps is None:
                keep.append(scan)
                continue

            i = int(np.searchsorted(stamps, scan.stamp))
            best, best_gap = None, np.inf
            for k in (i - 1, i):
                if 0 <= k < len(stamps):
                    gap = abs(float(stamps[k]) - scan.stamp)
                    if gap < best_gap:
                        best, best_gap = k, gap

            if best is None or best_gap > self.max_time_gap:
                # Newer than every IMU sample we hold => the sample is simply
                # still in flight, so wait. Otherwise it can never match.
                if best is not None and scan.stamp > float(stamps[-1]):
                    keep.append(scan)
                else:
                    self.scans_unmatched += 1
                continue

            self._ingest(scan, quats[best])

        self._pending = deque(keep, maxlen=self._pending.maxlen)
        return self._emit_ready()

    def _ingest(self, scan: LidarScan, quaternion: np.ndarray) -> None:
        p = scan.points
        xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
        # `time` is documented as "relative time of this point from cloud
        # stamp". If the publisher leaves it at zero the frame degrades to
        # one timestamp per 120-point packet -- an error of at most 5.56 ms,
        # which is tolerable. Check np.ptp(points["time"]) on real data.
        t_abs = float(scan.stamp) + p["time"].astype(np.float64)

        if self.drop_zero_returns and len(xyz):
            # Mask BOTH arrays with the same mask, or points and times desync
            # and every deskew correction lands on the wrong point.
            keep = np.any(xyz != 0.0, axis=1)
            xyz, t_abs = xyz[keep], t_abs[keep]

        if not len(xyz):
            return
        if self.rotate_with_imu:
            xyz = rotate_points(xyz, quaternion)

        self._batch.append((float(scan.stamp), xyz, t_abs))
        self.scans_ingested += 1
        self.points_ingested += len(xyz)

    # --- frame closing ---

    def _emit_ready(self) -> list[Frame]:
        out: list[Frame] = []
        if not self._batch:
            return out

        # key= is not optional: a bare tuple comparison would fall through to
        # comparing the ndarrays and raise on ambiguous truth value.
        self._batch.sort(key=lambda entry: entry[0])

        while self._batch:
            t0 = self._batch[0][0]
            if self._batch[-1][0] - t0 < self.frame_duration:
                break  # frame still open; more scans may belong to it
            n = 0
            for stamp, _xyz, _t in self._batch:
                if stamp - t0 >= self.frame_duration:
                    break
                n += 1
            take, self._batch = self._batch[:n], self._batch[n:]
            frame = self._build(take)
            if frame is not None:
                out.append(frame)
        return out

    def flush(self) -> Frame | None:
        """Close the frame in progress. Call at end of a recording."""
        if not self._batch:
            return None
        self._batch.sort(key=lambda entry: entry[0])
        take, self._batch = self._batch, []
        return self._build(take)

    def _build(self, batch) -> Frame | None:
        xyz = np.concatenate([b[1] for b in batch], axis=0)
        t = np.concatenate([b[2] for b in batch], axis=0)

        if len(xyz) < self.min_points:
            self.frames_too_sparse += 1
            logger.warning(
                "Dropping a %d-point frame (min_points=%d); odometry will see a "
                "gap and its constant-velocity guess will be short.",
                len(xyz), self.min_points,
            )
            return None

        lo, hi = float(t.min()), float(t.max())
        span = hi - lo
        # Deskew reference is the END of the frame (measured, not assumed), so
        # 1.0 must be the last point and the pose belongs to t_end.
        norm = (t - lo) / span if span > 1e-9 else np.ones(len(t), dtype=np.float64)

        self.frames_emitted += 1
        return Frame(
            points=np.ascontiguousarray(xyz, dtype=np.float64),
            timestamps=np.ascontiguousarray(norm, dtype=np.float64),
            t_start=lo,
            t_end=hi,
            n_scans=len(batch),
        )

    def stats(self) -> dict:
        return {
            "frames_emitted": self.frames_emitted,
            "frames_too_sparse": self.frames_too_sparse,
            "scans_ingested": self.scans_ingested,
            "scans_pending": len(self._pending),
            "scans_unmatched": self.scans_unmatched,
            "points_ingested": self.points_ingested,
            "scans_in_open_frame": len(self._batch),
        }
