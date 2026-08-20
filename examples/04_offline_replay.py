"""Use the accumulator with no hardware at all.

`RotatedScanAccumulator.add()` takes plain sequences rather than reaching into
a live stream, which is what makes recorded-data replay and unit testing
possible. Anything you build on top of `add()` will work offline for free.
"""

import math

import numpy as np

from l1_stream import LidarIMU, RotatedScanAccumulator, pack_scan_packet, parse_packet
from l1_stream.protocol import POINT_DTYPE


def synthetic_scan(stamp, scan_id, n=60):
    """One narrow vertical slice of a 2 m cylindrical room."""
    pts = np.zeros(n, dtype=POINT_DTYPE)
    elev = np.linspace(-0.3, 1.5, n)
    pts["x"] = 2.0
    pts["y"] = 0.0
    pts["z"] = elev
    return parse_packet(pack_scan_packet(stamp, scan_id, pts))


def yaw_imu(stamp, imu_id, yaw_rad):
    return LidarIMU(
        stamp=stamp, id=imu_id,
        quaternion=(0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2)),
        angular_velocity=(0.0, 0.0, 0.0), linear_acceleration=(0.0, 0.0, 9.81),
    )


acc = RotatedScanAccumulator(max_scans=400, max_time_gap=0.01)
for i in range(360):
    t = 1.0 + i * 0.001
    acc.add([synthetic_scan(t, i)], [yaw_imu(t, i, math.radians(i))])

pts = acc.get_points()
radius = np.linalg.norm(pts[:, :2], axis=1)
print(f"{len(pts)} points accumulated")
print(f"radius: mean={radius.mean():.3f} m, spread={radius.std():.2e} m")
print("A correct rotation reconstructs the cylinder: spread should be ~0.")
print(acc.stats())
