"""IMU-based orientation compensation for accumulated LiDAR scans.

A single L1 scan packet carries at most 120 points -- a narrow slice, not a
sweep. To see anything you have to accumulate many packets over a time window.
But the sensor is rotating (and, on a robot, moving) *during* that window, so
the packets are expressed in different sensor orientations. Stacking them raw
smears the map.

:class:`RotatedScanAccumulator` fixes the rotation part of that: each scan
packet is rotated once, by the IMU orientation nearest its own timestamp, into
a common gravity-aligned frame before being stacked.

**What this does NOT do.** It corrects orientation only, not translation. If
the robot drives forward while accumulating, the accumulated cloud still
smears along the direction of travel. Removing that requires pose estimation
(odometry / SLAM), not an IMU orientation alone. This is *inter-packet*
compensation across packets; it is complementary to, not a substitute for,
the *intra-frame* deskewing that a registration algorithm does inside a single
assembled frame using an estimated velocity.

**Unverified assumption.** This code applies the IMU quaternion directly to
point coordinates, which is only exactly right if the IMU axes and the point
cloud axes coincide. Any fixed mounting offset between the two inside the L1
would show up as a constant rotation error. I have not confirmed from Unitree
documentation whether such an extrinsic exists for this sensor -- if your
accumulated floor plane comes out consistently tilted while the robot is
level, that is the first thing to check.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from itertools import count
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .protocol import LidarIMU, LidarScan

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_quaternion",
    "quaternion_to_matrix",
    "rotate_points",
    "RotatedScanAccumulator",
]

_EPS = 1e-12


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Return ``(x, y, z, w)`` scaled to unit length, as float64.

    Why bother: the rotation formula below is only a *rotation* when the
    quaternion has length 1. If the length is off by a factor k, every point
    gets scaled by k^2 as well as rotated -- the cloud silently grows or
    shrinks. The IMU sends float32s, so its quaternion is already unit to
    about 1e-7, which is harmless; this guard exists for the cases that are
    not harmless: an all-zero quaternion during sensor warm-up, or a value you
    constructed by hand.

    An all-zero (degenerate) quaternion falls back to identity rather than
    dividing by zero.
    """
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        logger.warning("Degenerate (zero-length) quaternion; falling back to identity.")
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert ``(x, y, z, w)`` to a 3x3 rotation matrix (float64).

    Useful when you need the rotation as a matrix -- composing transforms,
    building a 4x4 pose, handing something to a library that wants SE(3).
    """
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rotate_points(
    points_xyz: np.ndarray,
    quaternion: Sequence[float],
    normalize: bool = True,
) -> np.ndarray:
    """Rotate an ``(N, 3)`` array by ``quaternion`` given as ``(x, y, z, w)``.

    Uses the standard vector form ``v' = v + 2w(q x v) + 2q x (q x v)``, which
    is the same rotation a 3x3 matrix would give but costs about 15 flops per
    point instead of 15 multiply-adds plus building the matrix -- worth it when
    you are rotating one small scan at a time, thousands of times a second.

    Set ``normalize=False`` only if you have already normalised the quaternion
    and are rotating in a hot loop.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3), got {points_xyz.shape}")

    q = normalize_quaternion(quaternion) if normalize else np.asarray(
        quaternion, dtype=np.float64
    ).reshape(4)
    q_xyz = q[:3]
    w = q[3]

    t = 2.0 * np.cross(q_xyz, points_xyz)
    return points_xyz + w * t + np.cross(q_xyz, t)


class RotatedScanAccumulator:
    """Rolling window of scans, each rotated once into a common frame.

    Each scan is rotated exactly once no matter how many display frames it
    survives, so the per-frame cost is a concatenate, not N rotations.

    Behaviour guarantees:

    * **No scan is dropped for arriving early.** Scan and IMU packets are
      separate datagrams, so a scan routinely shows up before the IMU sample
      that timestamps it. Such scans wait in a pending queue and are retried on
      the next update rather than being discarded.
    * **Matching is O(S log I).** IMU samples are sorted and binary-searched
      rather than scanned linearly for every scan.
    * **Reordered delivery is safe.** UDP does not guarantee arrival order, so
      IMU samples are sorted by timestamp before searching.
    * **The concatenated output is cached** and rebuilt only when something
      changed, so calling :meth:`get_points` repeatedly between updates is free.
    * **Unmatchable scans are counted, not silently lost.** Watch
      ``stats()["scans_unmatched"]`` -- a climbing value means the IMU stream is
      absent, or that ``max_time_gap`` is too tight for your buffer sizes.
    """

    def __init__(
        self,
        max_scans: int = 100,
        max_time_gap: float = 0.01,
        pending_maxlen: Optional[int] = None,
        drop_zero_returns: bool = True,
        rate_window: float = 1.0,
    ):
        if max_scans < 1:
            raise ValueError("max_scans must be >= 1")
        if max_time_gap <= 0:
            raise ValueError("max_time_gap must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")

        self.max_time_gap = float(max_time_gap)
        self.drop_zero_returns = drop_zero_returns

        # Entries are (stamp, seq, points), kept sorted by (stamp, seq).
        # The seq counter is a tiebreaker so tuple comparison never falls
        # through to comparing numpy arrays, which raises on ambiguous truth.
        self._rotated: Deque[Tuple[float, int, np.ndarray]] = deque(maxlen=max_scans)
        self._seq = count()
        self._pending: Deque[LidarScan] = deque(
            maxlen=pending_maxlen if pending_maxlen is not None else max_scans
        )
        self._cache: Optional[np.ndarray] = None

        self.scans_accumulated = 0
        self.scans_unmatched = 0
        self.points_accumulated = 0

        # Points-per-second: recomputed once per `rate_window` seconds rather
        # than on every add() call, so it's a stable rate rather than a spiky
        # per-call number. `_now()` is a hook so tests can drive it with a
        # fake clock instead of racing against a real one.
        self.rate_window = float(rate_window)
        self._rate_window_start_time: float = self._now()
        self._rate_window_start_points: int = 0
        self.points_per_second: float = 0.0

    @staticmethod
    def _now() -> float:
        """Monotonic clock, isolated behind a method so tests can override it
        instead of racing against a real one with sleep()."""
        return time.monotonic()

    # --- ingestion ---

    def update(self, lidar_stream) -> int:
        """Pull everything new from a :class:`~l1_stream.stream.LidarStream`.

        Returns the number of scans successfully rotated and added this call.
        """
        new_scans = lidar_stream.scans.drain()
        imu_samples = lidar_stream.recent_imu(lidar_stream.imu_capacity)
        return self.add(new_scans, imu_samples)

    def add(
        self,
        scans: Iterable[LidarScan],
        imu_samples: Sequence[LidarIMU],
    ) -> int:
        """Core ingestion, decoupled from any stream object.

        Taking plain sequences here (rather than reaching into a stream) is
        what makes offline replay and unit testing possible without hardware.
        """
        # Checked first, unconditionally, so the rate decays toward zero
        # during a real gap in data rather than freezing at its last value --
        # the early returns below would otherwise skip it entirely.
        self._update_rate()

        for scan in scans:
            # The pending queue is bounded, so it can evict. Count those rather
            # than losing them invisibly -- a climbing counter here is the
            # signal that IMU samples are not arriving at all.
            if len(self._pending) == self._pending.maxlen:
                self.scans_unmatched += 1
            self._pending.append(scan)

        if not self._pending or not imu_samples:
            return 0

        stamps = np.fromiter(
            (s.stamp for s in imu_samples), dtype=np.float64, count=len(imu_samples)
        )
        order = np.argsort(stamps, kind="stable")
        sorted_stamps = stamps[order]
        newest_imu = float(sorted_stamps[-1])

        still_pending: List[LidarScan] = []
        added = 0

        for scan in self._pending:
            # A scan newer than every IMU sample is not unmatchable -- its IMU
            # sample simply has not arrived yet. Hold it and retry next call.
            if scan.stamp > newest_imu + self.max_time_gap:
                still_pending.append(scan)
                continue

            idx = self._closest_index(sorted_stamps, scan.stamp)
            match = imu_samples[int(order[idx])]
            if abs(match.stamp - scan.stamp) > self.max_time_gap:
                # Older than anything we hold, or sitting in a real IMU gap.
                # It will never get a better match, so drop it and count it.
                self.scans_unmatched += 1
                continue

            xyz = scan.xyz(dtype=np.float64, drop_zero_returns=self.drop_zero_returns)
            if len(xyz) == 0:
                continue

            self._insert_rotated(scan.stamp, rotate_points(xyz, match.quaternion))
            self.points_accumulated += len(xyz)
            self.scans_accumulated += 1
            added += 1

        self._pending = deque(still_pending, maxlen=self._pending.maxlen)
        if added:
            self._cache = None
        return added

    def _update_rate(self, now: Optional[float] = None) -> None:
        """Recompute :attr:`points_per_second` once every ``rate_window``
        seconds. Called on every :meth:`add`, but only actually updates the
        published rate on the window boundary -- points_accumulated changes
        far more often than the rate is useful to re-read.
        """
        now = now if now is not None else self._now()
        elapsed = now - self._rate_window_start_time
        if elapsed >= self.rate_window:
            new_points = self.points_accumulated - self._rate_window_start_points
            self.points_per_second = new_points / elapsed
            self._rate_window_start_time = now
            self._rate_window_start_points = self.points_accumulated

    @staticmethod
    def _closest_index(sorted_stamps: np.ndarray, target: float) -> int:
        """Index into ``sorted_stamps`` of the value nearest ``target``."""
        pos = int(np.searchsorted(sorted_stamps, target))
        if pos == 0:
            return 0
        if pos >= len(sorted_stamps):
            return len(sorted_stamps) - 1
        before, after = sorted_stamps[pos - 1], sorted_stamps[pos]
        return pos - 1 if (target - before) <= (after - target) else pos

    def _insert_rotated(self, stamp: float, points: np.ndarray) -> None:
        """Insert a rotated scan, keeping ``_rotated`` sorted by timestamp.

        Insertion order is not capture order: UDP does not guarantee delivery
        order, so a scan can arrive after one captured later than it -- either
        reordered within a single batch, or delayed into a later call entirely.
        Two things depend on capture order being right: consumers that feed
        scans sequentially to a registration algorithm (which assume each frame
        is later than the last), and eviction -- a maxlen deque drops the
        least-recently-*inserted* item, which without this would sometimes
        discard a newer scan and keep an older one.

        The fast path is a plain O(1) append, because in-order arrival is the
        overwhelming majority. The out-of-order path re-sorts, which is O(n log
        n) on a deque of at most ``max_scans`` items -- cheaper in practice than
        maintaining a heap for a case that rarely fires.
        """
        item = (stamp, next(self._seq), points)

        if not self._rotated or (item[0], item[1]) >= (
            self._rotated[-1][0], self._rotated[-1][1]
        ):
            self._rotated.append(item)  # maxlen evicts the oldest by stamp
            return

        items = list(self._rotated)
        items.append(item)
        items.sort(key=lambda entry: (entry[0], entry[1]))
        self._rotated.clear()
        # extend() on a maxlen deque keeps the LAST maxlen items, which after
        # sorting are the newest by timestamp -- exactly the eviction we want.
        self._rotated.extend(items)

    # --- output ---

    def get_points(self) -> np.ndarray:
        """All accumulated points as one ``(N, 3)`` float64 array, oldest first.

        Cached: repeated calls between updates do not re-concatenate. float64
        because Open3D's ``Vector3dVector`` and most registration libraries
        require doubles and will otherwise copy or raise.
        """
        if self._cache is None:
            if not self._rotated:
                self._cache = np.empty((0, 3), dtype=np.float64)
            else:
                self._cache = np.concatenate(
                    [points for _stamp, _seq, points in self._rotated], axis=0
                )
        return self._cache

    def reset(self) -> None:
        """Clear all accumulated and pending data. Counters are kept."""
        self._rotated.clear()
        self._pending.clear()
        self._cache = None

    def stats(self) -> dict:
        return {
            "scans_held": len(self._rotated),
            "scans_pending": len(self._pending),
            "scans_accumulated": self.scans_accumulated,
            "scans_unmatched": self.scans_unmatched,
            "points_accumulated": self.points_accumulated,
            "points_held": len(self.get_points()),
            "points_per_second": self.points_per_second,
        }

    def __len__(self) -> int:
        return len(self._rotated)
