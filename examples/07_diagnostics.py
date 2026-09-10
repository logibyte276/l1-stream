"""Answer the open questions from a recording, before tuning anything.

    # park the car, record 60 s, then:
    python examples/07_diagnostics.py stationary.l1raw

Checks, in the order they will bite you:

1. Is ``points["time"]`` populated? If not, deskew degrades to one timestamp
   per packet. MEASURED on this rig: 60 points/packet at 360 packets/s, so a
   packet spans ~2.78 ms, not the 5.56 ms a 120-point blade would give.
2. Near-range structure. Fine 1 cm bins below 0.30 m, because that is where a
   chassis lives and 5 cm bins hide it.
   ** THIS CANNOT DETERMINE THE SELF-HIT RADIUS ON ITS OWN. ** Parked, your
   chassis and a nearby wall are both static and look identical. Use
   ``examples/10_selfhit.py`` on a PIVOT recording for the actual answer; this
   histogram only tells you what is out there and what each cut-off costs.
3. IMU yaw drift. A 6-axis IMU has no heading reference, so a parked net yaw
   is pure gyro drift.
4. Stationary odometry drift, the noise floor for every other measurement.
"""

import argparse

import numpy as np

from l1_stream.protocol import LidarIMU, LidarScan
from l1_stream.recording import Replayer

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--odometry", action="store_true",
               help="Also run KISS-ICP to measure stationary drift.")
p.add_argument("--min-range", type=float, default=0.25)
p.add_argument("--near", type=float, default=0.30,
               help="show 1 cm bins below this range, 5 cm bins above")
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
print(f"\n[1] points['time'] spread: {spread*1000:.3f} ms")
if spread < 1e-9:
    print("    ZERO -> the publisher does not fill it. FrameAssembler will fall")
    print("    back to one time per packet.")
else:
    print(f"    populated. At 21,600 pts/s that implies "
          f"{spread*21600:.0f} points per packet.")

# --- 2. near-range structure ------------------------------------------------
# Fine bins where the chassis lives, coarse bins beyond. Empty near-field bins
# are printed too: knowing there are NO returns below 0.10 m is a real result,
# and a loop that skips empty bins hides it.
near_edges = np.arange(0.0, args.near + 1e-9, 0.01)
far_edges = np.arange(args.near, 1.55, 0.05)
edges = np.concatenate([near_edges[:-1], far_edges])
counts, _ = np.histogram(ranges, bins=edges)
peak = counts.max() or 1

print(f"\n[2] range histogram (1 cm bins below {args.near:.2f} m):")
# Empty near-field bins are reported, because "nothing at all below 0.10 m" is
# a real finding -- but as a collapsed run, since 30 lines of "(empty)" is
# noise rather than information.
run_start = None
for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True):
    near = lo < args.near
    if c == 0 and near:
        run_start = lo if run_start is None else run_start
        continue
    if run_start is not None:
        n_bins = int(round((lo - run_start) / 0.01))
        print(f"    {run_start:4.2f}-{lo:4.2f} m  {0:8d}  "
              f"<- NO RETURNS AT ALL ({n_bins} empty bins)")
        run_start = None
    if c:
        print(f"    {lo:4.2f}-{hi:4.2f} m  {c:8d}  {'#' * int(40*c/peak)}")
if run_start is not None:
    print(f"    {run_start:4.2f}-{args.near:4.2f} m  {0:8d}  <- NO RETURNS AT ALL")

print("\n    what each cut-off costs you:")
for mr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    frac = 100 * float((ranges < mr).mean())
    star = "  <- current" if abs(mr - args.min_range) < 1e-9 else ""
    print(f"      min_range {mr:.2f} m  discards {frac:5.2f}% of returns{star}")
print("\n    Reminder: a parked recording CANNOT separate chassis from wall.")
print("    Run examples/10_selfhit.py on a PIVOT recording for that.")

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
    o = KissOdometry(min_range=args.min_range, voxel_size=0.15, deskew=False)
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
        print(f"    jitter {1000*o.path_length()/max(len(o.poses)-1, 1):.1f} mm/frame "
              f"(directly measured: true motion is zero)")
        t = np.array(thresholds)
        print(f"    adaptive threshold  start {t[0]:.3f}  final {t[-1]:.3f}  "
              f"(range {t.min():.3f}-{t.max():.3f} m, "
              f"{'ADAPTS' if np.ptp(t) > 1e-6 else 'NEVER MOVES'})")
