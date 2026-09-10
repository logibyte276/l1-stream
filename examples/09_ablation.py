"""Ablation: switch off ONE part of the pipeline, hold everything else fixed.

    python examples/09_ablation.py personal/loop.l1raw
    python examples/09_ablation.py personal/loop.l1raw --ablate deskew

An ablation removes exactly one component and measures what breaks. The
discipline is entirely in "exactly one" -- change two things and an improvement
tells you nothing about which one caused it. This script enforces that half:
both runs come from the same recording and share every parameter except the
one named. You only have to get the recording right.

WHAT THE RECORDING NEEDS. Fast pivots and a closed loop back to a marked start
(a wall corner pins heading as well as position). A gentle drive will not
separate the conditions, and without a real return the loop error has no
referent.

AND: one run is one data point. Do 3-5 recordings before believing a result.
The min_range "finding" on this project looked convincing on one recording and
reversed on the second.

RESULT ON THIS RIG (2026-09-04): ablating imu-rotation cost 32-126x on loop
closure across two recordings. Ablating deskew IMPROVED things slightly. Both
are recorded in claude/slam-results.md.
"""

import argparse
import logging

from l1_stream.offline import add_args, config_from_args, replay

logging.basicConfig(level=logging.WARNING, format="%(message)s")

ABLATIONS = {
    "imu-rotation": ("rotate_with_imu", True, False),
    "deskew": ("deskew", True, False),
}

p = argparse.ArgumentParser()
p.add_argument("path")
add_args(p)
p.add_argument("--ablate", choices=sorted(ABLATIONS), default="imu-rotation")
p.add_argument("--truth", type=float, default=None,
               help="tape-measured displacement, m (open drives only)")
p.add_argument("--closed-loop", action=argparse.BooleanOptionalAction, default=True,
               help="the drive returned to its start, so loop error is meaningful")
args = p.parse_args()

key, on_val, off_val = ABLATIONS[args.ablate]
base = config_from_args(args)

runs = {}
for label, value in (("ON", on_val), ("OFF", off_val)):
    runs[label] = replay(args.path, **{**base, key: value})

for label, r in runs.items():
    if len(r.xyz) < 2:
        raise SystemExit(f"{label} produced {len(r.xyz)} frames.\n{r.assembler_stats}")

fixed = {k: v for k, v in base.items() if k != key}
print(f"recording   {args.path}")
print("held fixed  " + "  ".join(f"{k}={v}" for k, v in fixed.items()))
print(f"varying     {key}\n")

ROWS = [
    ("frames",              lambda r: len(r.xyz),                       "{:.0f}",  None),
    ("points/frame",        lambda r: r.sizes.mean(),                   "{:.0f}",  None),
    ("path length (m)",     lambda r: r.path_length,                    "{:.2f}",  None),
    ("net displacement (m)", lambda r: r.net_displacement,              "{:.3f}",  None),
    ("loop error (% path)", lambda r: 100*r.net_displacement/max(r.path_length, 1e-9),
                                                                        "{:.2f}",  "lower"),
    ("rotation (deg/frame)", lambda r: r.rotation_deg.mean(),           "{:.2f}",  "lower"),
    ("jitter (mm/frame)",   lambda r: r.jitter_mm,                      "{:.1f}",  "lower"),
    ("final threshold (m)", lambda r: r.thresholds[-1],                 "{:.3f}",  "lower"),
    ("z excursion (m)",     lambda r: r.xyz[:,2].max() - r.xyz[:,2].min(), "{:.3f}", "lower"),
]
if args.truth:
    ROWS.insert(4, ("scale error (%)",
                    lambda r: abs(100*(r.net_displacement-args.truth)/args.truth),
                    "{:.2f}", "lower"))

names = list(runs)
print(f"{'':24}{names[0]:>14}{names[1]:>14}   better  ({key})")
print("-" * 68)
for label, fn, fmt, direction in ROWS:
    a, b = fn(runs[names[0]]), fn(runs[names[1]])
    mark = ""
    # Float noise is not a difference; anything under this is a tie.
    if direction == "lower" and abs(a - b) > 1e-6 * max(abs(a), abs(b), 1.0):
        mark = names[0] if a < b else names[1]
    print(f"{label:24}{fmt.format(a):>14}{fmt.format(b):>14}   {mark}")

print()
if args.closed_loop:
    key_fn = ROWS[5][1] if args.truth else ROWS[4][1]
    a, b = key_fn(runs[names[0]]), key_fn(runs[names[1]])
    winner = names[0] if a < b else names[1]
    margin = abs(a - b)
    print(f"Lower loop error on THIS recording: {winner}  (by {margin:.2f} pp)")
    if margin < 0.5:
        print("That margin is small. Treat it as a tie, not a result.")
else:
    print("Open path: loop error is meaningless. Compare net displacement")
    print("against a tape measure with --truth instead.")
print("One recording is one data point. Repeat on 3-5 drives before concluding.")

# A uniform scale error cancels exactly around a symmetric loop, so a closed
# loop cannot see it. Straight line measures scale; loop measures heading.
if args.closed_loop and args.truth:
    print("\nNOTE: --truth on a closed loop is not a scale test. A uniform")
    print("scale error cancels around a loop. Use an open drive for scale.")
