"""The odometry report. One command, every metric, from one recording.

    python examples/06_odometry_offline.py personal/drive.l1raw
    python examples/06_odometry_offline.py personal/drive.l1raw --truth 5.0
    python examples/06_odometry_offline.py personal/drive.l1raw --voxel-size 0.10

Replay is unpaced, so a 60 s drive re-runs in seconds -- which is the whole
reason to record before wiring odometry into the live loop.

This absorbs what used to be three separate scripts (an odometry summary, a
per-frame rotation checker, and a speed/clipping profiler). They shared the
same replay loop and had drifted apart on defaults; now the loop and the
defaults both live in l1_stream.offline.

READING THE OUTPUT. Different tests answer different questions, and mixing
them up has burned this project more than once:

  * ``net`` against a tape measure  -> SCALE. A loop cannot see scale error:
    a uniform shortfall cancels exactly around a symmetric loop.
  * ``loop error`` on a closed loop -> HEADING. It says nothing about scale.
  * ``path / net``                  -> jitter plus any real weaving.
  * ``jitter``                      -> the estimator alone, no path assumption.
  * ``clipping``                    -> whether the recording caught the whole
    drive. A late start looks EXACTLY like a scale error.
"""

import argparse
import logging

import numpy as np

from l1_stream.offline import add_args, config_from_args, replay

logging.basicConfig(level=logging.WARNING, format="%(message)s")

p = argparse.ArgumentParser()
p.add_argument("path")
add_args(p)
p.add_argument("--truth", type=float, default=None,
               help="tape-measured displacement in m; prints scale error")
p.add_argument("--still", type=float, default=0.15,
               help="speed below this (m/s) counts as stopped, for the "
                    "clipping check. Must clear the jitter floor "
                    "(~12 mm/frame parked at 0.2 s = 0.06 m/s).")
p.add_argument("--profile", action="store_true", help="print a speed profile")
p.add_argument("--out", default=None, help="save the trajectory as .npy")
args = p.parse_args()

run = replay(args.path, **config_from_args(args))
if len(run.xyz) < 2:
    raise SystemExit(f"Only {len(run.xyz)} frames registered.\n{run.assembler_stats}")

c = run.config
print("config            " + "  ".join(
    f"{k}={v}" for k, v in (
        ("frame", c["frame_duration"]), ("voxel", c["voxel_size"]),
        ("range", f"{c['min_range']}-{c['max_range']}"),
        ("imu_rot", c["rotate_with_imu"]), ("deskew", c["deskew"]))))

print(f"frames            {len(run.xyz)}")
print(f"points/frame      mean {run.sizes.mean():.0f}  "
      f"min {run.sizes.min()}  max {run.sizes.max()}")
print(f"frame span        mean {run.spans.mean()*1000:.1f} ms  "
      f"std {run.spans.std()*1000:.1f} ms   <- want a SMALL std")

path_len, net = run.path_length, run.net_displacement
print(f"\npath length       {path_len:.3f} m")
print(f"net displacement  {net:.3f} m")
print(f"path / net        {path_len/max(net, 1e-9):.3f}   "
      f"<- jitter plus any real weaving")
if args.truth:
    print(f"scale error       {100*(net-args.truth)/args.truth:+.1f}%  "
          f"vs a measured {args.truth:.2f} m   <- SCALE (open drives only)")
print(f"loop error        {net:.3f} m ({100*net/max(path_len,1e-9):.2f}% of path)"
      f"   <- HEADING, and only meaningful if you returned to the start")

rot = run.rotation_deg
print(f"\nper-frame rotation  mean {rot.mean():.2f} deg  "
      f"p95 {np.percentile(rot, 95):.2f} deg")
print(f"jitter              {run.jitter_mm:.1f} mm/frame  "
      f"(2nd-difference estimate; parked floor was 12.6 mm)")
print(f"z range             {run.xyz[:,2].min():+.3f} .. {run.xyz[:,2].max():+.3f} m"
      f"   <- hemisphere-above FOV makes z weakly observable")

t = run.thresholds
print(f"adaptive threshold  start {t[0]:.3f}  final {t[-1]:.3f}  "
      f"(range {t.min():.3f}-{t.max():.3f} m)")

# --- did the recording bracket the drive? -----------------------------------
clip = run.clipping(args.still)
print(f"\nclipping check (stopped < {args.still} m/s, peak {run.speed.max():.2f} m/s)")
print(f"  frames at rest before motion: {clip['head']}   after motion: {clip['tail']}")
if not clip["clipped"]:
    print("  OK -- the recording brackets the drive at both ends.")
else:
    for where, s in clip["clipped"]:
        print(f"  ** CLIPPED AT {where}: first/last frame speed {s:.2f} m/s, not ~0.")
        print(f"     >= {1000*s*c['frame_duration']:.0f} mm of real motion is missing "
              f"({100*s*c['frame_duration']/max(net,1e-9):.1f}% of the reported net).")
    print("  Re-record with ~5 s at rest before and after. A late start looks")
    print("  exactly like a scale error and is not one.")

if args.profile:
    print("\nspeed profile (one char per frame):")
    ramp, sp = " .:!#", run.speed
    hi = max(sp.max(), 1e-9)
    bar = "".join(ramp[min(4, int(5 * s / hi))] for s in sp)
    for i in range(0, len(bar), 74):
        print(f"  {i:4d} |{bar[i:i+74]}")

print(f"\nassembler         {run.assembler_stats}")

if args.out:
    np.save(args.out, run.xyz)
    print(f"trajectory -> {args.out}")
