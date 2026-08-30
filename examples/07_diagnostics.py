"""Answer the open questions from a recording, before tuning anything.

    # park the car, let it sit and rotate in place, record 60 s, then:
    python examples/07_diagnostics.py stationary.l1raw

Checks, in the order they will bite you:

1. Is ``points["time"]`` populated? If not, deskew degrades to one timestamp
   per 120-point packet (max error 5.56 ms -- tolerable, but you should know).
2. What is the self-hit radius? Points from the chassis are static in the
   sensor frame, register perfectly against themselves every frame, and pin
   the solution at zero motion. ``min_range`` must clear them.
3. Does the adaptive threshold actually adapt while stationary, and how much
   does the pose drift when nothing is moving? Stationary drift is the noise
   floor for every other measurement you make.
"""

import argparse

import numpy as np

from l1_stream.protocol import LidarIMU, LidarScan
from l1_stream.recording import Replayer

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--odometry", action="store_true",
               help="Also run KISS-ICP to measure stationary drift.")
p.add_argument("--min-range", type=float, default=0.4)
args = p.parse_args()

ranges, point_times, quats, n_scans, n_imu = [], [], [], 0, 0
for _t, msg in Replayer(args.path).iter_messages():
    if isinstance(msg, LidarScan):
        n_scans += 1
        point_times.append(msg.points["time"].astype(np.float64))
        xyz = msg.xyz(drop_zero_returns=True)
        if len(xyz):
            ranges.append(np.linalg.norm(xyz, axis=1))
    elif isinstance(msg, LidarIMU):
        n_imu += 1
        quats.append(msg.quaternion)

if not ranges:
    raise SystemExit("No scans in that recording.")

ranges = np.concatenate(ranges)
point_times = np.concatenate(point_times)

print(f"scans {n_scans}   imu {n_imu}   points {len(ranges)}")

# --- 1. per-point time ------------------------------------------------------
spread = float(np.ptp(point_times))
print(f"\n[1] points['time'] spread within a packet: {spread*1000:.3f} ms")
if spread < 1e-9:
    print("    ZERO -> the publisher does not fill it. FrameAssembler will fall")
    print("    back to one time per packet; max deskew error 5.56 ms.")
else:
    print("    populated (expect ~5.56 ms for a 120-point blade)")

# --- 2. self-hit radius -----------------------------------------------------
print("\n[2] near-range histogram (look for a spike = your own chassis):")
edges = np.arange(0.0, 1.55, 0.05)
counts, _ = np.histogram(ranges, bins=edges)
peak = counts.max() or 1
for lo, c in zip(edges[:-1], counts, strict=True):
    if c:
        print(f"    {lo:4.2f}-{lo+0.05:4.2f} m  {c:8d}  {'#' * int(40*c/peak)}")
below = float((ranges < args.min_range).mean())
print(f"    {100*below:.2f}% of returns fall below min_range={args.min_range} m")
print("    Set min_range just ABOVE the static spike, not below it.")

# --- 3. IMU yaw drift -------------------------------------------------------
if quats:
    q = np.asarray(quats, dtype=np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.unwrap(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    print(f"\n[3] IMU yaw swept {np.degrees(np.ptp(yaw)):.1f}° "
          f"over the recording, net {np.degrees(yaw[-1]-yaw[0]):+.1f}°")
    print("    If the car never turned, that net value is pure drift: a 6-axis")
    print("    IMU has no heading reference, so yaw is unobservable.")

# --- 4. stationary odometry drift ------------------------------------------
if args.odometry:
    from l1_stream.frames import FrameAssembler
    from l1_stream.odometry import KissOdometry

    a = FrameAssembler(frame_duration=0.2)
    o = KissOdometry(min_range=args.min_range)
    thresholds = []
    for scans, imu in Replayer(args.path).iter_batches(period=0.05):
        for f in a.add(scans, imu):
            o.register(f)
            thresholds.append(o.threshold)
    if o.poses:
        xyz = o.trajectory()
        print(f"\n[4] stationary drift over {len(o.poses)} frames:")
        print(f"    net {np.linalg.norm(xyz[-1]-xyz[0]):.3f} m, "
              f"path {o.path_length():.3f} m  <- both should be ~0")
        t = np.array(thresholds)
        print(f"    adaptive threshold  start {t[0]:.3f}  final {t[-1]:.3f}  "
              f"(range {t.min():.3f}-{t.max():.3f} m, "
              f"{'ADAPTS' if np.ptp(t) > 1e-6 else 'NEVER MOVES'})")
