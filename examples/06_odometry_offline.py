"""The odometry report. One command, every metric, from one recording.

    python examples/06_odometry_offline.py personal/drive.l1raw
    python examples/06_odometry_offline.py personal/drive.l1raw --truth 5.0
    python examples/06_odometry_offline.py personal/drive.l1raw --voxel-size 0.10
    python examples/06_odometry_offline.py personal/drive.l1raw --scale 1.037
    python examples/06_odometry_offline.py personal/drive.l1raw --verbose

AN ABLATION IS TWO RUNS OF THIS SCRIPT. Switch off exactly one component, hold
everything else fixed, and the difference is attributable to that component:

    python examples/06_odometry_offline.py personal/loop.l1raw --deskew \
        --tag ablated=deskew --tag condition=ON  --log personal/results.csv
    python examples/06_odometry_offline.py personal/loop.l1raw --no-deskew \
        --tag ablated=deskew --tag condition=OFF --log personal/results.csv

"Everything else fixed" is guaranteed by l1_stream.config, not by discipline:
both runs take every other parameter from DEFAULTS, and every one is written to
the results row, so a difference you did not intend is visible in the table.
There used to be a separate 09_ablation.py enforcing this; it became redundant
once the defaults lived in one place. (09 is now 09_map_offline.)

ONE RUN IS ONE DATA POINT. Do 3-5 recordings before believing a result. The
min_range "finding" on this project looked convincing on one recording and
reversed on the second.

Replay is unpaced, so a 60 s drive re-runs in seconds -- which is the whole
reason to record before wiring odometry into the live loop.

This absorbs what used to be three separate scripts (an odometry summary, a
per-frame rotation checker, and a speed/clipping profiler). They shared the
same replay loop and had drifted apart on defaults; now the loop and the
defaults both live in l1_stream.offline.

READING THE OUTPUT. The headline line depends on what the sidecar says the
recording is, because different drives answer different questions:

  * a line drive with a truth  -> net vs truth = SCALE. A loop cannot see scale:
    a uniform shortfall cancels exactly around a symmetric loop.
  * a loop                     -> loop error = HEADING. It says nothing about scale.
  * parked                     -> drift, the estimator's noise floor.

Below it: path length and jitter (the estimator alone, no path assumption),
compute, and whether the recording caught the whole drive -- a late start looks
EXACTLY like a scale error. Warnings appear only when something is wrong.
--verbose adds frame, threshold and assembler details.

--scale multiplies the trajectory before anything is reported. It is for trying
scale factors by hand and is NOT read from config.SCALE_FACTOR.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from l1_stream.metadata import RecordingMeta
from l1_stream.offline import add_args, config_from_args, replay
from l1_stream.results import ResultsLog, run_row

logging.basicConfig(level=logging.WARNING, format="%(message)s")

p = argparse.ArgumentParser()
p.add_argument("path")
add_args(p)
p.add_argument("--truth", type=float, default=None,
               help="tape-measured displacement in m. Defaults to the "
                    "sidecar's truth_m; pass this to override it.")
p.add_argument("--scale", type=float, default=1.0,
               help="Multiply the whole trajectory by this before reporting (1.0 = raw "
                    "KISS-ICP output). Distances, speeds, jitter, --out and the logged "
                    "row all use the scaled trajectory; rotation is untouched. "
                    "Experimental: NOT read from config.SCALE_FACTOR.")
p.add_argument("--still", type=float, default=0.15,
               help="speed below this (m/s) counts as stopped, for the "
                    "clipping check. Must clear the jitter floor "
                    "(~12 mm/frame parked at 0.2 s = 0.06 m/s).")
p.add_argument("--verbose", action="store_true",
               help="also print frame, threshold and assembler details")
p.add_argument("--profile", action="store_true", help="print a speed profile")
p.add_argument("--out", default=None, help="save the trajectory as .npy (scaled if --scale)")
p.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
               help="Extra column(s) on the logged row. Repeatable. Use it to "
                    "label whatever you are grouping by -- an ablation "
                    "(--tag ablated=deskew --tag condition=ON), a repeat "
                    "(--tag repeat=2), a re-run after the mount moved. Two runs "
                    "differing in one flag ARE an ablation; the tags are what "
                    "let you pair them in the table afterwards.")
p.add_argument("--log", default=None,
               help="append one row to this CSV. Capture the metric when it is "
                    "computed -- a table you plan to rebuild later will not exist.")
args = p.parse_args()

meta = RecordingMeta.load_or_none(args.path)

# Truth comes from the sidecar unless you say otherwise. Overriding is allowed
# and sometimes right (a re-measured mark, a deliberate what-if) -- it just says
# so, because a printed number and a logged number must never quietly come from
# different ground truths.
truth = args.truth
truth_note = ""
if truth is None and meta is not None:
    truth = meta.truth_m
elif truth is not None and meta is not None and meta.truth_m not in (None, truth):
    truth_note = f"  (--truth overrides the sidecar's {meta.truth_m} m)"

run = replay(args.path, **config_from_args(args))
if len(run.xyz) < 2:
    raise SystemExit(f"Only {len(run.xyz)} frames registered.\n{run.assembler_stats}")

# The first pose is the origin, so scaling every position scales every distance
# travelled. Done once, here, so every metric below -- and the logged row -- agree.
raw_net = run.net_displacement
if args.scale != 1.0:
    run.xyz = run.xyz * args.scale

c = run.config
kind = getattr(meta, "kind", None)
path_len, net = run.path_length, run.net_displacement
rot = run.rotation_deg
clip = run.clipping(args.still)
warnings: list[str] = []


def onoff(v):
    return "on" if v else "off"


# --- summary -----------------------------------------------------------------

print(f"{Path(args.path).name}   {meta.summary() if meta else 'no sidecar'}")
print(f"config   frame {c['frame_duration']}  voxel {c['voxel_size']}  "
      f"range {c['min_range']}-{c['max_range']}  imu_rot {onoff(c['rotate_with_imu'])}  "
      f"deskew {onoff(c['deskew'])}"
      + (f"  scale x{args.scale}" if args.scale != 1.0 else ""))
print()

raw = f"   (raw {raw_net:.3f} m)" if args.scale != 1.0 else ""
if kind == "loop":
    print(f"  loop error   {net:.3f} m   {100 * net / max(path_len, 1e-9):.2f}% of the path"
          f"{raw}")
elif kind == "stationary":
    print(f"  drift        {1000 * net:.0f} mm   while parked "
          f"{len(run.xyz) * c['frame_duration']:.0f} s")
elif truth:
    print(f"  net          {net:.3f} m   truth {truth:.3f} m   "
          f"scale error {100 * (net - truth) / truth:+.1f}%  "
          f"({1000 * (net - truth):+.0f} mm){raw}{truth_note}")
else:
    print(f"  net          {net:.3f} m{raw}   no truth: add --truth, or record it in the sidecar")

print(f"  path         {path_len:.3f} m   jitter {run.jitter_mm:.1f} mm/frame   "
      f"rotation {rot.mean():.2f} deg/frame")
print(f"  compute      {run.ms_per_frame:.0f} ms/frame   {run.budget_pct:.0f}% of the "
      f"{1000 * c['frame_duration']:.0f} ms budget on {run.replay_host}")

if kind != "stationary":        # parked jitter trips the speed threshold; meaningless
    if clip["clipped"]:
        for where, s in clip["clipped"]:
            missing = s * c["frame_duration"]
            share = ("" if kind == "loop"
                     else f" ({100 * missing / max(net, 1e-9):.1f}% of net)")
            warnings.append(f"CLIPPED at {where}: moving {s:.2f} m/s in that frame -- "
                            f">= {1000 * missing:.0f} mm missing{share}. "
                            "Re-record with ~5 s at rest.")
    else:
        print("  recording    starts and ends at rest")

if run.budget_pct > 100:
    warnings.append("over the frame budget on this machine: it cannot keep up live")
if run.spans.std() > 0.010:
    warnings.append(f"uneven frames (span std {1000 * run.spans.std():.1f} ms): KISS-ICP's "
                    "motion model expects even frames")
st = run.assembler_stats
if st.get("frames_too_sparse") or st.get("scans_unmatched"):
    warnings.append(f"assembler dropped {st.get('frames_too_sparse', 0)} sparse frames and "
                    f"{st.get('scans_unmatched', 0)} scans with no IMU match")

if warnings:
    print()
    for w in warnings:
        print(f"  ! {w}")

# --- detail ------------------------------------------------------------------

if args.verbose:
    t = run.thresholds
    print(f"\n  frames       {len(run.xyz)}   points/frame mean {run.sizes.mean():.0f} "
          f"(min {run.sizes.min()}, max {run.sizes.max()})   "
          f"span {1000 * run.spans.mean():.1f} +/- {1000 * run.spans.std():.1f} ms")
    print(f"  path / net   {path_len / max(net, 1e-9):.3f}   "
          f"rotation p95 {np.percentile(rot, 95):.2f} deg   "
          f"z {run.xyz[:, 2].min():+.3f} .. {run.xyz[:, 2].max():+.3f} m")
    print(f"  threshold    start {t[0]:.3f}  final {t[-1]:.3f}  (range {t.min():.3f}-{t.max():.3f} m)")
    print(f"  clipping     rest frames before motion {clip['head']}, after {clip['tail']}, "
          f"peak speed {run.speed.max():.2f} m/s (still < {args.still} m/s)")
    print(f"  assembler    {st}")

if args.profile:
    print("\n  speed profile (one char per frame):")
    ramp, sp = " .:!#", run.speed
    hi = max(sp.max(), 1e-9)
    bar = "".join(ramp[min(4, int(5 * s / hi))] for s in sp)
    for i in range(0, len(bar), 74):
        print(f"  {i:4d} |{bar[i:i+74]}")

if args.out or args.log:
    print()
if args.out:
    np.save(args.out, run.xyz)
    print(f"  trajectory -> {args.out}")

if args.log:
    # Always logged, so scaled and raw rows can never be mistaken for each other.
    tags = {"scale": args.scale, "raw_net_displacement_m": round(raw_net, 4)}
    for kv in args.tag:
        if "=" not in kv:
            raise SystemExit(f"--tag needs KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        tags[k] = v
    ResultsLog(args.log).append(run_row(args.path, run, meta, truth=truth, **tags))
    note = "" if meta is not None else "   (no sidecar: speed/environment columns blank)"
    print(f"  logged -> {args.log}{note}")
