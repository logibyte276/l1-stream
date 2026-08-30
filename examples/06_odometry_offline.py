"""Run KISS-ICP over a recording. This is where tuning happens.

    python examples/06_odometry_offline.py drive_01.l1raw --frame-duration 0.2
    python examples/06_odometry_offline.py drive_01.l1raw --no-imu-rotation

The second form is ablation #1 (see below). Because replay is not paced, a
60 s drive re-runs in a second or two -- which is the entire reason to record
before wiring odometry into the live loop.
"""

import argparse
import logging

import numpy as np

from l1_stream.frames import FrameAssembler
from l1_stream.odometry import KissOdometry
from l1_stream.recording import Replayer

logging.basicConfig(level=logging.WARNING, format="%(message)s")

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--frame-duration", type=float, default=0.2)
p.add_argument("--voxel-size", type=float, default=0.25)
p.add_argument("--max-range", type=float, default=25.0)
p.add_argument("--min-range", type=float, default=0.4,
               help="Measure this: it must clear your chassis self-hits.")
p.add_argument("--initial-threshold", type=float, default=0.4)
p.add_argument("--no-imu-rotation", action="store_true",
               help="ABLATION 1: feed raw body-frame points and let KISS-ICP "
                    "estimate full SE(3). Then last_pose IS the sensor pose, "
                    "and nothing depends on the 6-axis IMU's drifting yaw.")
p.add_argument("--no-deskew", action="store_true")
p.add_argument("--out", default=None, help="Save the trajectory as .npy")
args = p.parse_args()

assembler = FrameAssembler(
    frame_duration=args.frame_duration,
    rotate_with_imu=not args.no_imu_rotation,
)
odom = KissOdometry(
    voxel_size=args.voxel_size,
    max_range=args.max_range,
    min_range=args.min_range,
    deskew=not args.no_deskew,
    initial_threshold=args.initial_threshold,
)

spans, sizes = [], []
for scans, imu in Replayer(args.path).iter_batches(period=0.05):
    for frame in assembler.add(scans, imu):
        odom.register(frame)
        spans.append(frame.span)
        sizes.append(len(frame))

tail = assembler.flush()
if tail is not None:
    odom.register(tail)
    spans.append(tail.span)
    sizes.append(len(tail))

if not odom.poses:
    raise SystemExit("No frames were registered. Check assembler.stats() below.\n"
                     f"{assembler.stats()}")

xyz = odom.trajectory()
spans, sizes = np.array(spans), np.array(sizes)

print(f"frames            {len(odom.poses)}")
print(f"points/frame      mean {sizes.mean():.0f}  min {sizes.min()}  max {sizes.max()}")
print(f"frame span        mean {spans.mean()*1000:.1f} ms  "
      f"std {spans.std()*1000:.1f} ms   <- want a SMALL std")
print(f"path length       {odom.path_length():.2f} m")
print(f"net displacement  {np.linalg.norm(xyz[-1] - xyz[0]):.2f} m")
print(f"loop error        {odom.loop_closure_error():.3f} m "
      f"({100*odom.loop_closure_error()/max(odom.path_length(),1e-9):.1f}% of path)")
print(f"z range           {xyz[:,2].min():+.2f} .. {xyz[:,2].max():+.2f} m "
      f"<- the L1 sees only the hemisphere ABOVE itself, so z is weakly observable")
print(f"final threshold   {odom.threshold:.3f} m")
print(f"assembler         {assembler.stats()}")

if args.out:
    np.save(args.out, xyz)
    print(f"trajectory -> {args.out}")
