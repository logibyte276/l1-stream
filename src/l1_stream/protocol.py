"""
Wire format for the Unitree L1 UDP bridge.

The Unitree SDK's `unilidar_publisher_udp` example (which you run separately,
next to the sensor) sends datagrams shaped like:

    [msgType: uint32][dataSize: uint32][payload ...]

    msgType 101 -> IMU packet,  payload matches "=dI4f3f3f"
    msgType 102 -> Scan packet, payload matches "=dII" + up to 120 * "fffffI"

This module is deliberately pure: no sockets, no threads, no global state. That
makes every function here testable without hardware, which is why the ``pack_*``
helpers exist alongside the ``parse_*`` ones -- they let tests (and a replay
tool) build byte-identical packets in memory.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MSG_TYPE_IMU",
    "MSG_TYPE_SCAN",
    "MAX_POINTS_PER_SCAN",
    "POINT_DTYPE",
    "LidarPoint",
    "LidarScan",
    "LidarIMU",
    "LidarMessage",
    "parse_imu",
    "parse_scan",
    "parse_packet",
    "pack_imu_packet",
    "pack_scan_packet",
]

# --- protocol constants -----------------------------------------------------

MSG_TYPE_IMU = 101
MSG_TYPE_SCAN = 102

#: The C++ side packs a fixed-size array of this many points per scan packet
#: (only the first ``valid_points_num`` are meaningful). Used here as a safety
#: cap so a corrupt or oversized packet can't make us allocate wildly.
MAX_POINTS_PER_SCAN = 120

#: One point = 5 little-endian float32s (x, y, z, intensity, time) + 1 uint32
#: (ring). 24 bytes, no padding -- matches ``PointUnitree`` in the SDK header.
POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "<f4"), ("time", "<f4"), ("ring", "<u4"),
])

_HEADER_STRUCT = struct.Struct("=II")        # msgType, dataSize
_IMU_STRUCT = struct.Struct("=dI4f3f3f")     # stamp, id, quat[4], gyro[3], accel[3]
_SCAN_HEADER_STRUCT = struct.Struct("=dII")  # stamp, id, validPointsNum

HEADER_SIZE = _HEADER_STRUCT.size            # 8
IMU_PAYLOAD_SIZE = _IMU_STRUCT.size          # 52
SCAN_HEADER_SIZE = _SCAN_HEADER_STRUCT.size  # 16

#: Size of a full scan datagram as the SDK publisher sends it (it always
#: transmits all 120 point slots regardless of how many are valid). Exposed
#: because it is larger than a 1500-byte Ethernet MTU and therefore gets IP
#: fragmented -- see the README section on network sizing.
FULL_SCAN_DATAGRAM_SIZE = HEADER_SIZE + SCAN_HEADER_SIZE + MAX_POINTS_PER_SCAN * POINT_DTYPE.itemsize


# --- data containers --------------------------------------------------------

@dataclass
class LidarPoint:
    """A single LiDAR return.

    Convenience view onto one row of a :class:`LidarScan`'s ``points`` array.
    For bulk processing use :meth:`LidarScan.xyz` instead -- building one of
    these per point in a loop throws away the vectorised parsing below.
    """

    x: float
    y: float
    z: float
    intensity: float
    time: float
    ring: int


@dataclass
class LidarScan:
    """One scan packet: a narrow slice of returns, not a full 360 sweep."""

    stamp: float
    id: int
    valid_points_num: int
    points: np.ndarray  # structured array, dtype=POINT_DTYPE, len == valid_points_num

    def xyz_intensity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Nx3 float32 xyz, N-length float32 intensity)``."""
        xyz = np.stack(
            [self.points["x"], self.points["y"], self.points["z"]], axis=1
        )
        return xyz, self.points["intensity"]

    def xyz(self, dtype=np.float64, drop_zero_returns: bool = False) -> np.ndarray:
        """Return just the Nx3 coordinates, cast to ``dtype``.

        ``dtype`` defaults to float64 because most downstream consumers
        (Open3D's ``Vector3dVector``, KISS-ICP, most registration code) require
        double precision and will either copy silently or raise on float32.

        Set ``drop_zero_returns=True`` to strip points at exactly the origin.
        The sensor emits those for rays that never came back, and they are not
        real geometry -- feeding them to a registration algorithm creates a
        fake dense blob at the sensor position that pins the scan match.
        """
        xyz = np.stack(
            [self.points["x"], self.points["y"], self.points["z"]], axis=1
        ).astype(dtype, copy=False)
        if drop_zero_returns and len(xyz):
            xyz = xyz[np.any(xyz != 0.0, axis=1)]
        return np.ascontiguousarray(xyz, dtype=dtype)

    def point(self, i: int) -> LidarPoint:
        """Single point as a :class:`LidarPoint`. Fine for occasional lookups."""
        p = self.points[i]
        return LidarPoint(
            float(p["x"]), float(p["y"]), float(p["z"]),
            float(p["intensity"]), float(p["time"]), int(p["ring"]),
        )

    def __len__(self) -> int:
        return int(self.valid_points_num)


@dataclass
class LidarIMU:
    """One IMU sample.

    ``quaternion`` is ``(x, y, z, w)`` -- scalar LAST. This matches the
    ``float quaternion[4]  // quaternion: [x,y,z,w]`` field in
    ``unitree_lidar_sdk.h``. Many libraries (Open3D, ROS tf, Eigen's
    constructor) use scalar-FIRST ``(w, x, y, z)``; mixing the two silently
    produces a wrong-but-plausible rotation, so convert explicitly at every
    boundary.
    """

    stamp: float
    id: int
    quaternion: tuple[float, float, float, float]      # (x, y, z, w)
    angular_velocity: tuple[float, float, float]       # rad/s
    linear_acceleration: tuple[float, float, float]    # m/s^2

    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        """Same rotation, reordered to scalar-first ``(w, x, y, z)``."""
        x, y, z, w = self.quaternion
        return (w, x, y, z)


LidarMessage = LidarScan | LidarIMU


# --- parsing ----------------------------------------------------------------

def parse_imu(payload: bytes) -> LidarIMU:
    """Parse an IMU payload (everything after the 8-byte header).

    Raises ``struct.error`` if the payload is too short.
    """
    data = _IMU_STRUCT.unpack(payload[:IMU_PAYLOAD_SIZE])
    return LidarIMU(
        stamp=data[0],
        id=data[1],
        quaternion=data[2:6],
        angular_velocity=data[6:9],
        linear_acceleration=data[9:12],
    )


def parse_scan(payload: bytes) -> LidarScan:
    """Parse a scan payload (everything after the 8-byte header).

    Raises ``struct.error`` if the payload is shorter than the scan header.
    Truncated point data is clamped rather than raising, because a partially
    delivered packet still carries usable geometry.
    """
    stamp, scan_id, valid_points_num = _SCAN_HEADER_STRUCT.unpack(
        payload[:SCAN_HEADER_SIZE]
    )
    points_bytes = payload[SCAN_HEADER_SIZE:]

    # Clamp against BOTH the protocol max and the bytes actually received.
    # Without the second clamp a truncated packet makes np.frombuffer raise
    # ValueError (not struct.error), which is easy to miss in a caller's
    # except clause.
    available = len(points_bytes) // POINT_DTYPE.itemsize
    count = min(int(valid_points_num), MAX_POINTS_PER_SCAN, available)

    # .copy() is deliberate: np.frombuffer returns a READ-ONLY view that also
    # keeps the whole original datagram alive. Both are bad once these arrays
    # get stored in a ring buffer -- callers can't modify in place, and memory
    # balloons. ~3 KB per scan, so the copy is cheap.
    points = np.frombuffer(points_bytes, dtype=POINT_DTYPE, count=count).copy()

    return LidarScan(stamp=stamp, id=scan_id, valid_points_num=count, points=points)


def parse_packet(data: bytes) -> LidarMessage | None:
    """Parse one full UDP datagram.

    Returns ``None`` -- never raises -- if the datagram is too short, an
    unknown message type, or malformed. This function runs on a background
    thread; an escaping exception there would kill the reader silently and
    freeze every buffer with no visible error.
    """
    if len(data) < HEADER_SIZE:
        logger.warning("Received undersized packet (%d bytes), ignoring.", len(data))
        return None

    msg_type, declared_size = _HEADER_STRUCT.unpack(data[:HEADER_SIZE])
    payload = data[HEADER_SIZE:]

    if declared_size > len(payload):
        # Truncation is recoverable for scans (clamped below) but is always
        # worth surfacing, since it usually means MTU fragmentation loss.
        logger.warning(
            "Packet claims %d payload bytes but only %d arrived (msgType=%d).",
            declared_size, len(payload), msg_type,
        )

    try:
        if msg_type == MSG_TYPE_IMU:
            return parse_imu(payload)
        if msg_type == MSG_TYPE_SCAN:
            return parse_scan(payload)
        logger.warning("Unknown message type: %d", msg_type)
        return None
    except (struct.error, ValueError) as exc:
        logger.warning("Failed to parse packet (msgType=%d): %s", msg_type, exc)
        return None


# --- packing (for tests, replay, and simulation) ----------------------------

def pack_imu_packet(
    stamp: float,
    imu_id: int,
    quaternion: Sequence[float],
    angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    linear_acceleration: Sequence[float] = (0.0, 0.0, 0.0),
) -> bytes:
    """Build a byte-identical IMU datagram. ``quaternion`` is ``(x, y, z, w)``."""
    payload = _IMU_STRUCT.pack(
        float(stamp), int(imu_id),
        *[float(v) for v in quaternion],
        *[float(v) for v in angular_velocity],
        *[float(v) for v in linear_acceleration],
    )
    return _HEADER_STRUCT.pack(MSG_TYPE_IMU, len(payload)) + payload


def pack_scan_packet(
    stamp: float,
    scan_id: int,
    points: np.ndarray,
    pad_to_max: bool = True,
) -> bytes:
    """Build a scan datagram from a structured array of :data:`POINT_DTYPE`.

    ``pad_to_max=True`` reproduces what the SDK publisher does: always
    transmit all 120 point slots. Set it False to emit a compact packet.
    """
    points = np.asarray(points, dtype=POINT_DTYPE)
    if points.ndim != 1:
        raise ValueError("points must be a 1-D structured array")
    valid = len(points)
    if valid > MAX_POINTS_PER_SCAN:
        raise ValueError(
            f"{valid} points exceeds the protocol maximum of {MAX_POINTS_PER_SCAN}"
        )

    body = points.tobytes()
    if pad_to_max:
        body += b"\x00" * ((MAX_POINTS_PER_SCAN - valid) * POINT_DTYPE.itemsize)

    payload = _SCAN_HEADER_STRUCT.pack(float(stamp), int(scan_id), valid) + body
    return _HEADER_STRUCT.pack(MSG_TYPE_SCAN, len(payload)) + payload
