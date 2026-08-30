"""Ablation 1: does IMU pre-rotation help or hurt? Both runs, one recording.

    python examples/09_ablation.py pivot_loop.l1raw

An ablation switches off exactly ONE part of the system and holds everything
else fixed, so any difference in the result is attributable to that part. This
script enforces the "everything else fixed" half: both conditions read the same
file and get identical parameters. You only have to get the recording right.

WHAT THE RECORDING NEEDS:

* **Fast pivots.** Gentle driving does not separate the two conditions -- the
  IMU-extrinsic error grows with how fast you turn, so a smooth drive returns
  a tie and teaches you nothing.
* **A closed loop.** Finish at the same physical spot, same heading. Tape-mark
  the floor. That is what gives loop-closure error a known right answer (zero).

AND: one run is not a result. Do 3-5 recordings. A 4-of-5 win is a finding; a
3-2 split means they are equivalent, and you should then prefer --no-imu-
rotation because its pose is directly usable with no composition step.
"""

import argparse

import numpy as np

from l1_stream.frames import FrameAssembler
from l1_stream.odometry import KissOdometry
from l1_stream.recording import Replayer

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--frame-duration", type=float, default=0.2)
p.add_argument("--voxel-size", type=float, default=0.25)
p.add_argument("--max-range", type=float, default=25.0)
p.add_argument("--min-range", type=float, default=0.4)
p.add_argument("--initial-threshold", type=float, default=0.4)
p.add_argument("--closed-loop", action="store_true", default=True,
               help="The drive returned to its start (default: assume yes).")
p.add_argument("--open-path", dest="closed_loop", action="store_false",
               help="The drive did NOT return to its start; skip loop error.")
p.add_argument("--save", action="store_true", help="Write both trajectories as .npy")
args = p.parse_args()


def run(rotate_with_imu: bool) -> dict:
    assembler = FrameAssembler(
        frame_duration=args.frame_duration, rotate_with_imu=rotate_with_imu
    )
    odom = KissOdometry(
        voxel_size=args.voxel_size, max_range=args.max_range,
        min_range=args.min_range, initial_threshold=args.initial_threshold,
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

    xyz = odom.trajectory()
    if len(xyz) < 2:
        return {"failed": "fewer than 2 frames registered",
                "stats": assembler.stats()}

    path_len = odom.path_length()
    loop = odom.loop_closure_error()
    return {
        "failed": None,
        "frames": len(odom.poses),
        "pts_mean": float(np.mean(sizes)),
        "span_ms": float(np.mean(spans) * 1000),
        "span_std_ms": float(np.std(spans) * 1000),
        "path_len": path_len,
        "loop_err": loop,
        "loop_pct": 100 * loop / max(path_len, 1e-9),
        "z_span": float(xyz[:, 2].max() - xyz[:, 2].min()),
        "unmatched": assembler.scans_unmatched,
        "sparse": assembler.frames_too_sparse,
        "xyz": xyz,
    }


print(f"recording  {args.path}")
print(f"parameters frame={args.frame_duration}s voxel={args.voxel_size} "
      f"range={args.min_range}-{args.max_range} thr={args.initial_threshold}")
print("           (identical for both conditions -- that is what makes it an ablation)\n")

a = run(rotate_with_imu=True)
b = run(rotate_with_imu=False)

for name, r in (("IMU rotation ON", a), ("IMU rotation OFF", b)):
    if r["failed"]:
        print(f"!! {name} FAILED: {r['failed']}\n   {r['stats']}")

if a["failed"] or b["failed"]:
    raise SystemExit("\nOne condition did not produce a trajectory; no comparison possible.")

rows = [
    ("frames registered",     "frames",      "{:.0f}",  None),
    ("points / frame",        "pts_mean",    "{:.0f}",  None),
    ("frame span (ms)",       "span_ms",     "{:.1f}",  None),
    ("frame span std (ms)",   "span_std_ms", "{:.1f}",  "lower"),
    ("path length (m)",       "path_len",    "{:.2f}",  None),
    ("loop error (m)",        "loop_err",    "{:.3f}",  "lower"),
    ("loop error (% path)",   "loop_pct",    "{:.2f}",  "lower"),
    ("z excursion (m)",       "z_span",      "{:.2f}",  "lower"),
    ("scans unmatched",       "unmatched",   "{:.0f}",  "lower"),
    ("frames too sparse",     "sparse",      "{:.0f}",  "lower"),
]

print(f"{'':24}{'IMU ON':>12}{'IMU OFF':>12}   better")
print("-" * 62)
for label, key, fmt, direction in rows:
    va, vb = a[key], b[key]
    mark = ""
    # Float noise is not a difference. Anything under this is a tie.
    if direction == "lower" and abs(va - vb) > 1e-6 * max(abs(va), abs(vb), 1.0):
        mark = "IMU ON" if va < vb else "IMU OFF"
    print(f"{label:24}{fmt.format(va):>12}{fmt.format(vb):>12}   {mark}")

print()
if not args.closed_loop:
    print("Open path: loop error is meaningless here. Compare path length against")
    print("a tape measure instead, and watch z excursion on a flat floor.")
else:
    winner = "IMU ON" if a["loop_pct"] < b["loop_pct"] else "IMU OFF"
    margin = abs(a["loop_pct"] - b["loop_pct"])
    print(f"Lower loop error on THIS recording: {winner}  (by {margin:.2f} pp)")
    if margin < 0.5:
        print("That margin is small. Treat this as a tie, not a result.")
    print("One recording is one data point. Repeat on 3-5 drives before concluding;")
    print("if it splits, prefer IMU OFF -- its last_pose is the sensor pose directly.")

if args.save:
    np.save("traj_imu_on.npy", a["xyz"])
    np.save("traj_imu_off.npy", b["xyz"])
    print("\nsaved traj_imu_on.npy, traj_imu_off.npy")
