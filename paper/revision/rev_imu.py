"""IMU pre-rotation modes and IMU/clock facts for the revision (comment M1).

l1-stream v1.0.0 has two IMU settings: ``rotate_with_imu=True`` (full
quaternion) and ``False`` (no rotation). The reviewer asked for a third,
roll-and-pitch only ("gravity-only"). This module adds it WITHOUT changing the
package, so l1-stream v1.0.0 (the Zenodo-archived version cited in the paper)
still reproduces every original number.

How gravity-only works. Each IMU sample's quaternion is replaced, as the
recording is read, by the same orientation with its heading (yaw about the
vertical axis) removed. Tilt is kept, so packets are still levelled, but
nothing from the gyro-integrated heading reaches the frames. The normal
``l1_stream.offline.replay`` then runs unchanged.

Rotation convention (from l1-stream): quaternions are (x, y, z, w), scalar
last, and ``rotate_points(p, q)`` applies R(q) to sensor-frame points to put
them in the IMU's gravity-aligned frame. Removing yaw is therefore
R' = Rz(-yaw) R, with yaw = atan2(R[1,0], R[0,0]) (Z-Y-X decomposition).
That is only valid if the IMU frame's vertical axis is z, which
:func:`imu_report` checks from the accelerometer before gravity-only runs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math

import numpy as np

import l1_stream.offline as _offline
from l1_stream.offline import replay
from l1_stream.protocol import LidarIMU, LidarScan
from l1_stream.recording import Replayer
from l1_stream.rotation import normalize_quaternion, quaternion_to_matrix

__all__ = [
    "IMU_MODES",
    "imu_report",
    "quat_multiply",
    "replay_with_imu_mode",
    "strip_yaw",
    "yaw_deg",
]

IMU_MODES = ("off", "gravity", "full")

#: Largest tilt of the IMU frame's measured vertical from its z axis before
#: gravity-only mode refuses to run. Yaw removal about z is meaningless if
#: z is not vertical.
MAX_VERTICAL_TILT_DEG = 5.0


# --- quaternion helpers (x, y, z, w) ----------------------------------------

def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product a * b, both (x, y, z, w). Rotation b is applied first."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + bw * ax + ay * bz - az * by,
        aw * by + bw * ay + az * bx - ax * bz,
        aw * bz + bw * az + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def yaw_deg(quaternion) -> float:
    """Heading of R(q) about z, degrees (Z-Y-X decomposition)."""
    r = quaternion_to_matrix(quaternion)
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def strip_yaw(quaternion) -> np.ndarray:
    """Same orientation with the heading about z removed: Rz(-yaw) R(q).

    The result keeps roll and pitch exactly, so the gravity direction seen in
    the sensor frame is unchanged; only the gyro-integrated heading is gone.
    """
    q = normalize_quaternion(quaternion)
    half = -0.5 * math.radians(yaw_deg(q))
    q_unyaw = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
    out = quat_multiply(q_unyaw, q)
    return out / np.linalg.norm(out)


# --- gravity-only replay ------------------------------------------------------

class _GravityOnlyReplayer(Replayer):
    """A Replayer whose IMU messages carry yaw-free quaternions."""

    def iter_messages(self, pace: bool = False):
        for recv_time, message in super().iter_messages(pace=pace):
            if isinstance(message, LidarIMU):
                message = dataclasses.replace(
                    message, quaternion=tuple(float(v) for v in strip_yaw(message.quaternion))
                )
            yield recv_time, message


@contextlib.contextmanager
def _gravity_only_reader():
    """Make l1_stream.offline.replay read recordings through the yaw-free
    Replayer for the duration of the block, then restore the original."""
    original = _offline.Replayer
    _offline.Replayer = _GravityOnlyReplayer
    try:
        yield
    finally:
        _offline.Replayer = original


def replay_with_imu_mode(path: str, imu_mode: str, **cfg):
    """``l1_stream.offline.replay`` with an IMU mode instead of a bool.

    off      -> rotate_with_imu=False (packets stacked as the sensor sent them)
    gravity  -> rotate_with_imu=True with yaw removed from every quaternion
    full     -> rotate_with_imu=True (the paper's baseline)
    """
    if imu_mode not in IMU_MODES:
        raise ValueError(f"imu_mode must be one of {IMU_MODES}, got {imu_mode!r}")
    if "rotate_with_imu" in cfg:
        raise TypeError("pass imu_mode, not rotate_with_imu")
    if imu_mode == "off":
        return replay(path, rotate_with_imu=False, **cfg)
    if imu_mode == "full":
        return replay(path, rotate_with_imu=True, **cfg)
    with _gravity_only_reader():
        return replay(path, rotate_with_imu=True, **cfg)


# --- facts for the Methods section -------------------------------------------

def imu_report(path: str, still_seconds: float = 2.0) -> dict:
    """Measure the IMU and clock facts the reviewer asked about.

    * vertical axis: the accelerometer during the first ``still_seconds``
      (the car is parked), rotated by R(q) into the IMU frame. Its angle from
      the z axis says whether that frame is gravity-aligned with z vertical.
    * yaw: heading change between the first and last IMU sample, and its
      full range over the recording (6-axis IMU, so this is gyro drift plus
      any real turning).
    * clocks: scan and IMU stamps (same SDK reader), the gap from each scan
      to its nearest IMU sample, the size of the stamps (Unix time is about
      1.7e9 s), and the per-point time offsets inside packets.
    """
    imu_t, imu_q, imu_a, scan_t, point_dt_max = [], [], [], [], 0.0
    for _recv, msg in Replayer(path).iter_messages():
        if isinstance(msg, LidarIMU):
            imu_t.append(msg.stamp)
            imu_q.append(msg.quaternion)
            imu_a.append(msg.linear_acceleration)
        elif isinstance(msg, LidarScan):
            scan_t.append(msg.stamp)
            if len(msg.points):
                point_dt_max = max(point_dt_max, float(np.max(msg.points["time"])))

    imu_t = np.asarray(imu_t, dtype=np.float64)
    scan_t = np.asarray(scan_t, dtype=np.float64)
    if len(imu_t) < 10 or len(scan_t) < 10:
        raise ValueError(f"{path}: too few IMU ({len(imu_t)}) or scan ({len(scan_t)}) messages")
    order = np.argsort(imu_t)
    imu_t = imu_t[order]
    imu_q = np.asarray(imu_q, dtype=np.float64)[order]
    imu_a = np.asarray(imu_a, dtype=np.float64)[order]

    still = imu_t <= imu_t[0] + still_seconds
    up = np.mean([quaternion_to_matrix(q) @ a for q, a in zip(imu_q[still], imu_a[still], strict=True)], axis=0)
    g = float(np.linalg.norm(up))
    up_unit = up / g if g > 0 else np.array([np.nan, np.nan, np.nan])
    tilt = math.degrees(math.acos(min(1.0, abs(float(up_unit[2]))))) if g > 0 else float("nan")

    yaws = np.unwrap(np.radians([yaw_deg(q) for q in imu_q]))
    yaws = np.degrees(yaws)

    idx = np.clip(np.searchsorted(imu_t, scan_t), 1, len(imu_t) - 1)
    gaps = np.minimum(np.abs(scan_t - imu_t[idx - 1]), np.abs(imu_t[idx] - scan_t))

    span = float(imu_t[-1] - imu_t[0])
    return {
        "imu_rate_hz": round((len(imu_t) - 1) / span, 1) if span > 0 else None,
        "scan_rate_hz": round((len(scan_t) - 1) / float(scan_t[-1] - scan_t[0]), 1),
        "gravity_mps2": round(g, 3),
        "vertical_tilt_from_z_deg": round(tilt, 2),
        "vertical_sign": "+z" if up_unit[2] > 0 else "-z",
        "yaw_change_deg": round(float(yaws[-1] - yaws[0]), 2),
        "yaw_range_deg": round(float(yaws.max() - yaws.min()), 2),
        "scan_imu_gap_median_ms": round(1000 * float(np.median(gaps)), 3),
        "scan_imu_gap_max_ms": round(1000 * float(np.max(gaps)), 3),
        "stamp_first_s": round(float(min(imu_t[0], scan_t[0])), 3),
        "stamps_look_like_unix_time": bool(min(imu_t[0], scan_t[0]) > 1e9),
        "point_time_offset_max_ms": round(1000 * point_dt_max, 3),
        "duration_s": round(span, 2),
    }
