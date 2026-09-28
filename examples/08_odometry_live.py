"""Live odometry. Do this LAST, after the parameters are settled offline.

Note the assembler gets its OWN feed and does not share the visualiser's
accumulator. The visualiser wants an overlapping rolling window; registration
needs disjoint frames. Sharing one object gives one of them the wrong thing.
"""

import logging
import time

import numpy as np

from l1_stream import LidarStream
from l1_stream.frames import FrameAssembler
from l1_stream.odometry import KissOdometry
from l1_stream.offline import DEFAULTS

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Take the tuned configuration from l1_stream.config (via offline.DEFAULTS),
# so the live pipeline runs exactly what was validated offline.
# tests/test_config.py fails if a literal value is written here instead.
assembler = FrameAssembler(
    frame_duration=DEFAULTS["frame_duration"],
    rotate_with_imu=DEFAULTS["rotate_with_imu"],
)
odom = KissOdometry(
    voxel_size=DEFAULTS["voxel_size"],
    max_range=DEFAULTS["max_range"],
    min_range=DEFAULTS["min_range"],
    deskew=DEFAULTS["deskew"],
    initial_threshold=DEFAULTS["initial_threshold"],
)
logging.info("config %s", DEFAULTS)

with LidarStream.for_history(2.0) as lidar:
    last_report = time.monotonic()
    try:
        while True:
            scans = lidar.scans.drain()
            imu = lidar.recent_imu(lidar.imu_capacity)
            if not scans:
                time.sleep(0.005)
                continue

            for frame in assembler.add(scans, imu):
                pose = odom.register(frame)
                x, y, z = pose[:3, 3]
                print(f"\rx={x:+7.2f} y={y:+7.2f} z={z:+6.2f} m  "
                      f"path={odom.path_length():6.2f} m  "
                      f"thr={odom.threshold:.2f}  pts={len(frame):5d}",
                      end="", flush=True)

            now = time.monotonic()
            if now - last_report >= 10.0:
                last_report = now
                logging.info("\n%s", assembler.stats())
    except KeyboardInterrupt:
        print()
        np.save("trajectory.npy", odom.trajectory())
        print(f"{len(odom.poses)} poses, path {odom.path_length():.2f} m "
              f"-> trajectory.npy")
