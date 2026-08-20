"""Command-line entry points.

Installed as ``l1-monitor`` and ``l1-visualize`` (see ``[project.scripts]`` in
``pyproject.toml``). Both are thin wrappers -- all real behaviour lives in the
library so it stays usable from your own code.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .stream import LidarStream

__all__ = ["monitor_main", "visualize_main"]


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, default=12345, help="UDP port to bind")
    parser.add_argument("--ip", default="0.0.0.0", help="local interface to bind")
    parser.add_argument(
        "--history", type=float, default=2.0,
        help="seconds of scan/IMU history to buffer",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def monitor_main(argv=None) -> int:
    """Print live throughput stats. Use this to confirm the bridge is alive."""
    parser = _common_parser("Monitor a Unitree L1 UDP stream.")
    parser.add_argument(
        "--interval", type=float, default=0.5, help="seconds between status lines"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    with LidarStream.for_history(args.history, port=args.port, ip=args.ip) as lidar:
        print(f"Listening on {args.ip}:{args.port} ...")
        if not lidar.wait_until_ready(timeout=5.0):
            print("No data within 5s -- is the SDK UDP publisher running and pointed at this port?")

        try:
            while True:
                scan = lidar.latest_scan
                imu = lidar.latest_imu
                if scan is not None:
                    print(
                        f"Scan #{scan.id}: {scan.valid_points_num} pts  "
                        f"stamp={scan.stamp:.3f}"
                    )
                if imu is not None:
                    q = imu.quaternion
                    print(
                        f"IMU  #{imu.id}: stamp={imu.stamp:.3f}  "
                        f"quat(xyzw)=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f})"
                    )
                print(f"  {lidar.stats()}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


def visualize_main(argv=None) -> int:
    """Open the live Open3D view."""
    parser = _common_parser("Live-visualise a Unitree L1 UDP stream.")
    parser.add_argument("--max-scans", type=int, default=100,
                        help="rotated scan packets kept on screen")
    parser.add_argument("--refresh-hz", type=float, default=30.0,
                        help="target render rate (ceiling)")
    parser.add_argument("--max-time-gap", type=float, default=0.01,
                        help="max scan-to-IMU timestamp mismatch, seconds")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--no-axes", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    # visualizer.py imports fine without Open3D -- the real ImportError is
    # raised lazily inside open(), so it has to be caught around run() too, or
    # the user gets a traceback instead of the install hint.
    from .visualizer import LiveVisualizer

    stream = LidarStream.for_history(args.history, port=args.port, ip=args.ip)
    try:
        LiveVisualizer(
            stream=stream,
            max_scans=args.max_scans,
            max_time_gap=args.max_time_gap,
            refresh_hz=args.refresh_hz,
            point_size=args.point_size,
            show_axes=not args.no_axes,
        ).run()
    except ImportError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(monitor_main())
