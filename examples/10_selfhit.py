"""Find the chassis self-hit radius from a PIVOT recording.

    # spin the car in place ~360 deg, then:
    python examples/10_selfhit.py pivot.l1raw

WHY A PIVOT AND NOT A STATIONARY RECORDING. A plain range histogram cannot
separate your chassis from the room, because when the sensor is parked BOTH
are static -- a wall at 0.4 m and a motor mount at 0.4 m look identical. The
range histogram in 07_diagnostics.py is therefore inconclusive by construction,
which is a flaw in that script, not in your data.

Pivoting breaks the tie. A point on the chassis sits at a FIXED range in a
FIXED sensor-frame direction no matter how the car turns. A point on a wall
does not: hold the sensor-frame direction constant, spin the car, and the
range sweeps across whatever the room happens to contain. So:

    low range variance in a fixed (ring, azimuth) cell  ->  rigidly attached
    high variance                                       ->  world geometry

Set min_range just above the outermost rigidly-attached return.
"""

import argparse

import numpy as np

from l1_stream.protocol import LidarScan
from l1_stream.recording import Replayer

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--az-bin", type=float, default=2.0, help="azimuth bin, degrees")
p.add_argument("--min-obs", type=int, default=25, help="min samples per cell")
p.add_argument("--static-std", type=float, default=0.03,
               help="range std below this = rigidly attached, metres")
p.add_argument("--max-range", type=float, default=6.0, help="analysis cutoff")
args = p.parse_args()

xs, ys, zs, rings, spans, counts, stamps = [], [], [], [], [], [], []
for _t, msg in Replayer(args.path).iter_messages():
    if not isinstance(msg, LidarScan) or not len(msg.points):
        continue
    pts = msg.points
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
    keep = np.any(xyz != 0.0, axis=1)
    if not keep.any():
        continue
    xs.append(xyz[keep, 0])
    ys.append(xyz[keep, 1])
    zs.append(xyz[keep, 2])
    rings.append(pts["ring"][keep].astype(np.int64))
    spans.append(float(np.ptp(pts["time"])))
    counts.append(int(msg.valid_points_num))
    stamps.append(float(msg.stamp))

if not xs:
    raise SystemExit("No usable scans in that recording.")

x = np.concatenate(xs)
y = np.concatenate(ys)
z = np.concatenate(zs)
ring = np.concatenate(rings)
r = np.sqrt(x*x + y*y + z*z)
counts = np.asarray(counts)
spans = np.asarray(spans)
stamps = np.asarray(stamps)

# --- packet geometry (this is what 07's check [1] should have reported) -----
duration = stamps.max() - stamps.min()
print(f"packets {len(counts)} over {duration:.1f} s  ->  {len(counts)/duration:.0f} packets/s")
print(f"points/packet  mean {counts.mean():.1f}  min {counts.min()}  max {counts.max()}")
print(f"point rate     {counts.sum()/duration:.0f} pts/s")
print(f"per-packet time span  mean {spans.mean()*1000:.3f} ms  max {np.ptp(spans)*1000:.3f} ms spread")
print(f"rings present  {sorted(np.unique(ring).tolist())}")

# --- static-structure test --------------------------------------------------
sel = r < args.max_range
r_s = r[sel]
ring_s = ring[sel]
az = np.degrees(np.arctan2(y[sel], x[sel]))
azbin = np.floor((az + 180.0) / args.az_bin).astype(np.int64)

key = ring_s * 100000 + azbin
uniq, inv = np.unique(key, return_inverse=True)
n = np.bincount(inv).astype(np.float64)
s1 = np.bincount(inv, weights=r_s)
s2 = np.bincount(inv, weights=r_s * r_s)
mean = s1 / n
std = np.sqrt(np.clip(s2 / n - mean * mean, 0.0, None))

ok = n >= args.min_obs
mean, std, n = mean[ok], std[ok], n[ok]
static = std < args.static_std

print(f"\ncells with >={args.min_obs} obs: {ok.sum()}   "
      f"rigidly attached (std < {args.static_std} m): {static.sum()}")

if not static.any():
    print("\nNo rigidly-attached returns found. Either the pivot was too small to")
    print("separate them, or nothing on the chassis is in view. Check that the car")
    print("actually turned; a stationary recording CANNOT answer this question.")
    raise SystemExit(0)

print("\nstatic-return fraction by range band:")
edges = np.arange(0.0, min(args.max_range, 3.0) + 0.1, 0.1)
for lo, hi in zip(edges[:-1], edges[1:], strict=True):
    band = (mean >= lo) & (mean < hi)
    if not band.any():
        continue
    frac = static[band].mean()
    bar = "#" * int(40 * frac)
    print(f"  {lo:4.2f}-{hi:4.2f} m  {int(band.sum()):4d} cells  "
          f"{100*frac:5.1f}% static  {bar}")

outer = float(mean[static].max())
print(f"\noutermost rigidly-attached return: {outer:.3f} m")
print(f"SUGGESTED min_range = {outer + 0.05:.2f} m   (outermost + 5 cm margin)")
print(f"that discards {100*float((r < outer + 0.05).mean()):.1f}% of all returns")
print("\nSanity-check it in the visualiser before trusting it: a band that is")
print("100% static all the way out to 1 m usually means the car is parked")
print("against a wall, not that the chassis is a metre wide.")
