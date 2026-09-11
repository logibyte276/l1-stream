"""Map accuracy against the building itself. No motion capture required.

    # see what is in the map and pick bounding boxes off the printed extent
    python examples/12_map_quality.py personal/room.l1raw

    # then measure two walls and the corner between them
    python examples/12_map_quality.py personal/room.l1raw \
        --plane  -1 3.0 -0.4   4 3.6 1.5 \
        --plane  3.0 -1 -0.4   3.6 4 1.5 \
        --expect-angle 90

WHY THIS IS THE RIGHT GROUND TRUTH INDOORS. Walls are flat, corners are square
and opposite walls are parallel, all to a tolerance far better than this sensor.
That gives per-map error metrics for free, where loop closure gives you one
endpoint number that is provably blind to scale.

READING THE OUTPUT:

  * ``rms`` is the metric to watch. It responds to range noise AND to
    registration error, so it degrades when either does. For reference the
    L1's own single-point accuracy is +/-20 mm, so an rms below about 20 mm
    means the map is as flat as the sensor can see.
  * ``inlier fraction`` well below 1.0 means the box caught something that is
    not the wall. Tighten it before believing the rms.
  * ``angle`` between two walls should be 90 (a corner) or 0 (opposite walls).
    The deviation is a HEADING error and is independent of scale -- so this and
    a tape-measured distance together cover both failure modes.

The z axis is weak: the L1 sees only the hemisphere above itself, so floor
planes will be sparse or missing. Prefer walls.
"""

import argparse
from pathlib import Path

import numpy as np

from l1_stream.mapmetrics import accumulate_map, box_mask, fit_plane, plane_angle_deg
from l1_stream.metadata import RecordingMeta
from l1_stream.offline import add_args, config_from_args
from l1_stream.results import ResultsLog

p = argparse.ArgumentParser()
p.add_argument("path")
add_args(p)
p.add_argument("--map-voxel", type=float, default=0.03,
               help="Map downsample. Do NOT confuse with --voxel-size, which is "
                    "the registration voxel; this one only bounds memory.")
p.add_argument("--plane", nargs=6, type=float, action="append", metavar="X",
               help="Bounding box 'xlo ylo zlo xhi yhi zhi' selecting one wall. "
                    "Repeat for more planes.")
p.add_argument("--expect-angle", type=float, default=None,
               help="Expected angle between consecutive planes, e.g. 90 or 0.")
p.add_argument("--expect-range", type=float, default=None, metavar="M",
               help="Laser-measured sensor-to-wall distance, for a STATIONARY "
                    "recording. Reports whether the LiDAR's ranges are biased "
                    "-- which is a different question from whether the "
                    "trajectory scale is.")
p.add_argument("--level", action="store_true",
               help="Report each plane's tilt from vertical. On a level-parked "
                    "recording, fit the CEILING (the L1 sees the hemisphere "
                    "ABOVE itself, so the floor is sparse) and the residual "
                    "tilt is the IMU->LiDAR extrinsic error.")
p.add_argument("--map-deskew", action=argparse.BooleanOptionalAction, default=True,
               help="Deskew FOR THE MAP, independently of --deskew. ON by "
                    "default: intra-frame smear is speed*frame_duration "
                    "(100 mm at 0.5 m/s), which sits below the 0.15 m "
                    "registration voxel but is 3.3x the map voxel.")
p.add_argument("--log", default=None,
               help="append one row PER PLANE to this CSV.")
p.add_argument("--threshold", type=float, default=0.05,
               help="RANSAC inlier band, m. Looser than sensor accuracy on "
                    "purpose so a bad map still yields a fit you can inspect.")
args = p.parse_args()

meta = RecordingMeta.load_or_none(args.path)
print(f"recording  {args.path}")
print(f"           {meta.summary() if meta else 'NO SIDECAR -- context unknown'}")

cfg = config_from_args(args)
print("config     " + "  ".join(f"{k}={v}" for k, v in cfg.items()))

cloud = accumulate_map(args.path, map_voxel=args.map_voxel,
                       map_deskew=args.map_deskew, **cfg)
if args.map_deskew != cfg["deskew"]:
    print(f"           map deskew={args.map_deskew} while registration "
          f"deskew={cfg['deskew']} -- deliberate, see --help")
lo, hi = cloud.min(axis=0), cloud.max(axis=0)
print(f"\nmap        {len(cloud):,} points at {args.map_voxel} m voxel")
print(f"extent     x {lo[0]:6.2f} .. {hi[0]:6.2f}    "
      f"y {lo[1]:6.2f} .. {hi[1]:6.2f}    z {lo[2]:6.2f} .. {hi[2]:6.2f}")

if not args.plane:
    print("\nNo --plane boxes given. Pick walls from the extent above and "
          "re-run with --plane xlo ylo zlo xhi yhi zhi.")
    raise SystemExit(0)

fits = []
p_ = None
print(f"\n{'plane':7}{'points':>9}{'inliers':>9}{'frac':>7}"
      f"{'rms (mm)':>10}{'max (mm)':>10}{'dist (m)':>10}  normal")
print("-" * 84)
for i, box in enumerate(args.plane):
    sel = cloud[box_mask(cloud, box[:3], box[3:])]
    if len(sel) < 3:
        print(f"{i:<7}{len(sel):>9}   too few points in that box -- widen it")
        fits.append(None)
        continue
    f = fit_plane(sel, threshold=args.threshold)
    fits.append(f)
    n = f.normal
    print(f"{i:<7}{f.n_points:>9}{f.n_inliers:>9}{f.inlier_fraction:>7.2f}"
          f"{1000*f.rms:>10.1f}{1000*f.max_abs:>10.1f}"
          f"{f.distance_from_origin:>10.3f}  "
          f"[{n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}]")

good = [f for f in fits if f is not None]
if len(good) >= 2:
    print()
    for i in range(len(fits) - 1):
        a, b = fits[i], fits[i + 1]
        if a is None or b is None:
            continue
        ang = plane_angle_deg(a, b)
        line = f"angle {i}-{i+1}   {ang:6.2f} deg"
        if args.expect_angle is not None:
            err = ang - args.expect_angle
            line += f"   expected {args.expect_angle:.0f}, off by {err:+.2f} deg"
        print(line)

if args.expect_range is not None:
    print()
    for i, f in enumerate(fits):
        if f is None:
            continue
        d = f.distance_from_origin
        err = 100 * (d - args.expect_range) / args.expect_range
        print(f"range {i}   measured {d:.3f} m  vs laser {args.expect_range:.3f} m"
              f"   {err:+.2f}%")
    print("  Only meaningful on a STATIONARY recording, where the map origin is")
    print("  the sensor. ~-5% here means the RANGES are biased (scale the points);")
    print("  ~0% means the ranges are fine and the scale error is in registration")
    print("  (scale the poses). Those need opposite fixes.")

if args.level:
    print()
    up = np.array([0.0, 0.0, 1.0])
    for i, f in enumerate(fits):
        if f is None:
            continue
        # A horizontal plane's normal is PARALLEL to vertical, so the angle
        # between them IS the tilt. (For a wall the normal is perpendicular and
        # the same number reads 90.)
        tilt = plane_angle_deg(f.normal, up)
        kind = "off horizontal" if tilt < 45 else "off vertical (this is a WALL)"
        print(f"plane {i}   {min(tilt, 90-tilt):.2f} deg {kind}")
    print("  Parked level, fitting the CEILING: with --imu-rotation this residual")
    print("  IS the IMU->LiDAR extrinsic error. Run again with --no-imu-rotation")
    print("  to see the raw mount tilt; the difference separates the two.")

worst = max((f.rms for f in good), default=0.0)
print(f"\nworst plane rms {1000*worst:.1f} mm "
      f"(L1 single-point accuracy is +/-20 mm)")
if np.any([f.inlier_fraction < 0.8 for f in good]):
    print("At least one box is under 80% inliers -- it is catching more than a "
          "wall. Tighten it before comparing runs.")

if args.log:
    # One row per plane, not one per recording: a map has as many quality
    # numbers as you measured walls, and averaging them here would throw away
    # the fact that one wall can be fine while another is not.
    log = ResultsLog(args.log)
    for i, f in enumerate(fits):
        if f is None:
            continue
        log.append({
            "recording": Path(args.path).name,
            "environment": getattr(meta, "environment", None),
            "speed_mps": getattr(meta, "speed_mps", None),
            **{f"cfg.{k}": v for k, v in cfg.items()},
            "map_voxel": args.map_voxel,
            "map_deskew": args.map_deskew,
            "plane": i,
            "points": f.n_points,
            "inliers": f.n_inliers,
            "inlier_fraction": round(f.inlier_fraction, 4),
            "plane_rms_mm": round(1000 * f.rms, 2),
            "plane_dist_m": round(f.distance_from_origin, 4),
            "plane_max_mm": round(1000 * f.max_abs, 2),
            "normal_x": round(float(f.normal[0]), 4),
            "normal_y": round(float(f.normal[1]), 4),
            "normal_z": round(float(f.normal[2]), 4),
        })
    print(f"\nlogged {sum(f is not None for f in fits)} rows -> {args.log}")
