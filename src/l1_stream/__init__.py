"""
l1_stream
==========

Python client for a Unitree L1 LiDAR whose data is being streamed over UDP.

**Prerequisite:** this package is a consumer only. Something else must already
be publishing packets -- normally the ``unilidar_publisher_udp`` example that
ships with the Unitree LiDAR SDK, run separately on the machine the sensor is
plugged into. Nothing here talks to the serial port.

Quick start::

    from l1_stream import LidarStream

    with LidarStream.for_history(seconds=2.0) as lidar:
        lidar.wait_until_ready()
        scan = lidar.latest_scan
        xyz = scan.xyz()            # (N, 3) float64

Live view (requires the ``viz`` extra)::

    from l1_stream.visualizer import LiveVisualizer
    LiveVisualizer().run()

The visualiser is not imported here: pulling it in eagerly would make
``import l1_stream`` fail on machines without Open3D, which is most headless
Jetsons. Import :mod:`l1_stream.visualizer` explicitly when you want it.
"""

from .protocol import (
    MAX_POINTS_PER_SCAN,
    MSG_TYPE_IMU,
    MSG_TYPE_SCAN,
    POINT_DTYPE,
    LidarIMU,
    LidarMessage,
    LidarPoint,
    LidarScan,
    pack_imu_packet,
    pack_scan_packet,
    parse_imu,
    parse_packet,
    parse_scan,
)
from .receiver import LidarUDPReceiver
from .ring_buffer import RingBuffer
from .rotation import (
    RotatedScanAccumulator,
    normalize_quaternion,
    quaternion_to_matrix,
    rotate_points,
)
from .stream import LidarStream
from .frames import Frame, FrameAssembler
from .recording import DatagramRecorder, Replayer, raw_datagrams
from .odometry import KissOdometry          # safe: kiss_icp imports lazily

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # protocol
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
    # transport
    "LidarUDPReceiver",
    "LidarStream",
    "RingBuffer",
    # geometry
    "rotate_points",
    "normalize_quaternion",
    "quaternion_to_matrix",
    "RotatedScanAccumulator",
]
