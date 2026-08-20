"""Minimal: connect, wait for data, print what arrives.

Start the SDK publisher first (your serial port may not be ttyUSB0 -- see the
README section "Which serial port?"):

    ./unilidar_publisher_udp /dev/ttyUSB0

Then:

    python examples/01_read_scans.py
"""

import logging
import time

from l1_stream import LidarStream

logging.basicConfig(level=logging.INFO, format="%(message)s")

with LidarStream.for_history(seconds=2.0) as lidar:
    if not lidar.wait_until_ready(timeout=5.0):
        raise SystemExit("No data in 5s -- is the SDK UDP publisher running?")

    for _ in range(20):
        scan = lidar.latest_scan
        imu = lidar.latest_imu

        # xyz() gives float64 and can strip zero-returns, which is what almost
        # every downstream library actually wants.
        xyz = scan.xyz(drop_zero_returns=True)
        print(f"scan #{scan.id}: {len(xyz)} real points, stamp={scan.stamp:.3f}")
        print(f"imu  #{imu.id}: quat(xyzw)={tuple(round(v, 3) for v in imu.quaternion)}")
        print(f"  {lidar.stats()}")
        time.sleep(0.25)
