"""Static wall check: is the L1's range scale right? (comment M1)

Park the car facing one wall at several laser-measured distances, record ~15 s
at each, then:

    python revision/rev_wall_check.py wall/W1.l1raw wall/W2.l1raw ... \
        --log wall_results.csv --plot wall_fit.png

The truth for each recording is the sidecar's truth_m (05_record.py --truth),
i.e. the Bosch distance from the front of the front wheels to the wall. Pass
--truth d1 d2 ... to override.

WHAT IT MEASURES. Raw L1 points straight from the recording: no IMU rotation,
no frame assembly, no KISS-ICP. For each recording it finds the vertical
planes in the cloud, picks the wall the car faces, and takes the plane's
perpendicular distance from the sensor origin D. Then it fits

    D = s * d_bosch + c

s is the range SCALE. s = 1.000 means ranges are the right size. If every
range were k % short, s would be 1 - k/100 -- and the scene, and with it the
odometry, would shrink by k %. The paper's odometry slope (-4.52 cm/m) would
need s = 0.955. c is the horizontal distance from the L1's origin back to the
front of the front wheels; check it with a ruler as a sanity test.

Using the slope rather than D - d at one position is what makes this work:
c is not known precisely, and at a single distance an offset and a scale
error look the same.

WHICH WALL. Side walls keep the same distance as the car moves along the
room and the wall behind gets FARTHER as the car gets closer to the front
wall, so the front wall is the plane whose distance rises with the Bosch
distance. It is picked by correlation (sign only), never by how close the
slope is to 1, so the choice cannot bias the answer. Override with --normal.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from l1_stream.metadata import RecordingMeta
from l1_stream.protocol import LidarIMU, LidarScan
from l1_stream.recording import Replayer
from l1_stream.results import ResultsLog

ODOM_SLOPE = -0.0452        # paper: -4.52 cm per metre
MIN_RANGE, MAX_RANGE = 0.25, 25.0


# --- point cloud -------------------------------------------------------------

def load_static_cloud(path: str, skip_s: float = 1.0, max_s: float | None = None):
    """All raw points of a parked recording (sensor frame) and the 'up' vector
    from the mean accelerometer reading. Skips the first ``skip_s`` seconds."""
    pts, acc = [], []
    t0 = None
    for _recv, msg in Replayer(path).iter_messages():
        stamp = msg.stamp
        if t0 is None:
            t0 = stamp
        if stamp - t0 < skip_s:
            continue
        if max_s is not None and stamp - t0 > skip_s + max_s:
            break
        if isinstance(msg, LidarScan):
            xyz = msg.xyz(drop_zero_returns=True)
            if len(xyz):
                pts.append(xyz)
        elif isinstance(msg, LidarIMU):
            acc.append(msg.linear_acceleration)
    if not pts:
        raise ValueError(f"{path}: no points")
    cloud = np.concatenate(pts)
    r = np.linalg.norm(cloud, axis=1)
    cloud = cloud[(r >= MIN_RANGE) & (r <= MAX_RANGE)]
    if acc:
        up = np.mean(np.asarray(acc, dtype=np.float64), axis=0)
        up = up / np.linalg.norm(up)
    else:
        up = np.array([0.0, 0.0, 1.0])
    return cloud, up


def voxel_first(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


# --- planes ------------------------------------------------------------------

def _refine(points: np.ndarray, normal: np.ndarray, offset: float, band: float):
    """Least-squares plane through every point within ``band`` of the given one."""
    near = np.abs(points @ normal + offset) <= band
    sel = points[near]
    centroid = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
    n = vt[-1]
    c = float(-n @ centroid)
    if c > 0:            # orient the normal from the sensor towards the plane
        n, c = -n, -c
    d = sel @ n + c
    return n, c, near, float(np.sqrt(np.mean(d**2)))


def vertical_planes(cloud: np.ndarray, up: np.ndarray, *, max_planes: int = 10,
                    band: float = 0.03, max_tilt_deg: float = 15.0,
                    iterations: int = 300, seed: int = 0) -> list[dict]:
    """Peel off the largest near-vertical planes one at a time (RANSAC + refit)."""
    rng = np.random.default_rng(seed)
    work = voxel_first(cloud, 0.01)
    if len(work) > 60000:
        work = work[rng.choice(len(work), 60000, replace=False)]
    full = cloud
    planes = []
    min_inliers = max(300, int(0.01 * len(work)))
    sin_tilt = math.sin(math.radians(max_tilt_deg))
    for _ in range(max_planes):
        if len(work) < min_inliers:
            break
        best, best_count = None, 0
        for _ in range(iterations):
            # Local sampling: a seed point plus two neighbours within 0.3 m.
            # Three points drawn from the whole room rarely land on the same
            # wall when that wall is a small share of the cloud.
            a = work[rng.integers(len(work))]
            near_seed = np.flatnonzero(np.sum((work - a) ** 2, axis=1) < 0.09)
            if len(near_seed) < 3:
                continue
            b, c = work[rng.choice(near_seed, 2, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n = n / norm
            if abs(n @ up) > sin_tilt:      # not vertical
                continue
            count = int(np.count_nonzero(np.abs(work @ n - n @ a) <= band))
            if count > best_count:
                best, best_count = (n, float(-n @ a)), count
        if best is None or best_count < min_inliers:
            break
        n, c, _near, _ = _refine(work, best[0], best[1], band)
        n, c, near_full, rms = _refine(full, n, c, band)
        planes.append({"normal": n, "distance": -c, "rms_mm": 1000 * rms,
                       "inliers": int(near_full.sum())})
        work = work[np.abs(work @ n + c) > band]
        full = full[~near_full]
    return planes


def pick_front_wall(per_rec: list[list[dict]], truths: np.ndarray, *,
                    normal: np.ndarray | None = None, max_angle_deg: float = 8.0):
    """Choose one plane per recording: the same wall in all of them."""
    cos_max = math.cos(math.radians(max_angle_deg))
    if normal is not None:
        normal = normal / np.linalg.norm(normal)
        chosen = []
        for planes in per_rec:
            same = [p for p in planes if p["normal"] @ normal >= cos_max]
            if not same:
                raise SystemExit("A recording has no plane within "
                                 f"{max_angle_deg} deg of --normal.")
            chosen.append(max(same, key=lambda p: p["inliers"]))
        return chosen, None

    tracks = []
    refs = []
    for p in sorted(per_rec[0], key=lambda p: -p["inliers"]):   # one ref per orientation
        if all(p["normal"] @ q["normal"] < cos_max for q in refs):
            refs.append(p)
    for ref in refs:
        track = [ref]
        for planes in per_rec[1:]:
            # Same orientation; among parallel planes (a wall and a cupboard
            # face, say) take the one with the most points.
            same = [p for p in planes if p["normal"] @ ref["normal"] >= cos_max]
            if not same:
                break
            track.append(max(same, key=lambda p: p["inliers"]))
        if len(track) == len(per_rec):
            d = np.array([p["distance"] for p in track])
            r = float(np.corrcoef(truths, d)[0, 1]) if d.std() > 0 else 0.0
            tracks.append((r, track))
    if not tracks:
        raise SystemExit("No wall was found in every recording. Check the recordings, "
                         "or pass --normal from one of the printed candidates.")
    tracks.sort(key=lambda t: t[0], reverse=True)
    return tracks[0][1], tracks


def fit_line(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS y = s x + c with standard errors and 95 % CIs (t with n-2 df)."""
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    s = float(((x - xm) * (y - ym)).sum() / sxx)
    c = float(ym - s * xm)
    resid = y - (s * x + c)
    out = {"n": n, "slope": s, "offset_m": c,
           "resid_rms_mm": 1000 * float(np.sqrt(np.mean(resid**2))), "resid": resid}
    if n > 2:
        sigma = math.sqrt(float((resid**2).sum()) / (n - 2))
        se_s = sigma / math.sqrt(sxx)
        se_c = sigma * math.sqrt(1 / n + xm**2 / sxx)
        tcrit = _t975(n - 2)
        out.update(slope_se=se_s, offset_se=se_c, tcrit=tcrit,
                   slope_ci=(s - tcrit * se_s, s + tcrit * se_s),
                   offset_ci=(c - tcrit * se_c, c + tcrit * se_c))
    return out


def _t975(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:  # small table so this runs on the Orin without scipy
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(df, 1.96)


# --- main ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Static wall range-scale check.")
    p.add_argument("recordings", nargs="+")
    p.add_argument("--truth", nargs="+", type=float, default=None,
                   help="Bosch distances (m), one per recording; default: sidecar truth_m")
    p.add_argument("--normal", nargs=3, type=float, default=None,
                   help="force the wall whose normal is nearest this (sensor frame)")
    p.add_argument("--skip", type=float, default=1.0, help="seconds to skip at the start")
    p.add_argument("--seconds", type=float, default=None, help="use at most this many seconds")
    p.add_argument("--log", default=None, help="append one row per position to this CSV")
    p.add_argument("--plot", default=None, help="save a fit + residual figure (needs matplotlib)")
    args = p.parse_args()

    if args.truth is not None and len(args.truth) != len(args.recordings):
        raise SystemExit("--truth needs one value per recording")
    truths = []
    for i, rec in enumerate(args.recordings):
        if args.truth is not None:
            truths.append(args.truth[i])
            continue
        meta = RecordingMeta.load_or_none(rec)
        if meta is None or meta.truth_m is None:
            raise SystemExit(f"{rec}: no truth_m in the sidecar; pass --truth")
        truths.append(float(meta.truth_m))
    order = np.argsort(truths)
    recs = [args.recordings[i] for i in order]
    truths = np.asarray(truths, dtype=np.float64)[order]
    if len(recs) < 3:
        raise SystemExit("Need at least 3 positions for a slope with a confidence interval.")

    per_rec, ups, npts = [], [], []
    for rec in recs:
        cloud, up = load_static_cloud(rec, skip_s=args.skip, max_s=args.seconds)
        per_rec.append(vertical_planes(cloud, up))
        ups.append(up)
        npts.append(len(cloud))

    chosen, tracks = pick_front_wall(
        per_rec, truths, normal=None if args.normal is None else np.asarray(args.normal))
    dist = np.array([c["distance"] for c in chosen])
    if tracks is not None and tracks[0][0] < 0.99:
        print(f"WARNING: the chosen wall's distance tracks the Bosch distance with r = "
              f"{tracks[0][0]:.3f} only. Check the candidates below; force one with --normal.")

    if tracks is not None and len(tracks) > 1:
        print("Walls seen at every position (r = correlation of distance with Bosch):")
        for r, tr in tracks:
            n = tr[0]["normal"]
            tag = "  <- front wall" if tr is chosen else ""
            print(f"  r {r:+.3f}   normal [{n[0]:+.2f} {n[1]:+.2f} {n[2]:+.2f}]   "
                  f"{tr[0]['distance']:.2f} .. {tr[-1]['distance']:.2f} m{tag}")
        print()

    fit = fit_line(truths, dist)
    print(f"{'recording':24s}{'Bosch m':>9}{'L1 plane m':>12}{'resid mm':>10}"
          f"{'rms mm':>8}{'inliers':>9}")
    for rec, d, D, c, res in zip(recs, truths, dist, chosen, fit["resid"], strict=True):
        print(f"{Path(rec).stem:24s}{d:9.4f}{D:12.4f}{1000*res:10.1f}"
              f"{c['rms_mm']:8.1f}{c['inliers']:9d}")

    s = fit["slope"]
    print(f"\nrange scale s = {s:.4f}", end="")
    if "slope_ci" in fit:
        lo, hi = fit["slope_ci"]
        print(f"  (95% CI {lo:.4f} to {hi:.4f}, n = {fit['n']})")
        print(f"  = range error {100*(s-1):+.2f}% (CI {100*(lo-1):+.2f} to {100*(hi-1):+.2f}%)")
        need = 1 + ODOM_SLOPE
        print(f"  ranges alone would explain the odometry bias if s = {need:.4f}: "
              f"{'inside' if lo <= need <= hi else 'outside'} the CI")
        print(f"  s = 1.0000 is {'inside' if lo <= 1 <= hi else 'outside'} the CI")
    else:
        print()
    print(f"offset c = {fit['offset_m']:.4f} m  (L1 origin to front of front wheels; "
          f"check with a ruler)")
    print(f"residual rms {fit['resid_rms_mm']:.1f} mm")
    tilt = [math.degrees(math.acos(min(1.0, abs(float(c['normal'] @ up)))))
            for c, up in zip(chosen, ups, strict=True)]
    print(f"wall normal vs vertical: {min(tilt):.1f}-{max(tilt):.1f} deg "
          f"(90 = wall is vertical in the sensor's own gravity estimate)")

    if args.log:
        log = ResultsLog(args.log)
        for rec, d, D, c, res, npt in zip(recs, truths, dist, chosen, fit["resid"], npts, strict=True):
            log.append({
                "recording": Path(rec).name, "bosch_m": d, "l1_plane_m": round(float(D), 5),
                "resid_mm": round(1000 * float(res), 2), "plane_rms_mm": round(c["rms_mm"], 2),
                "inliers": c["inliers"], "points": npt,
                "normal_x": round(float(c["normal"][0]), 4),
                "normal_y": round(float(c["normal"][1]), 4),
                "normal_z": round(float(c["normal"][2]), 4),
                "fit_slope": round(s, 5), "fit_offset_m": round(fit["offset_m"], 5),
                "fit_slope_ci_lo": round(fit.get("slope_ci", (np.nan,))[0], 5),
                "fit_slope_ci_hi": round(fit.get("slope_ci", (np.nan, np.nan))[1], 5),
            })
        print(f"logged -> {args.log}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 9})
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
        xs = np.linspace(truths.min(), truths.max(), 50)
        a1.plot(truths, dist, "o", color="#1f4e79", label="L1 wall plane")
        a1.plot(xs, s * xs + fit["offset_m"], "-", color="#1f4e79", lw=1,
                label=f"fit, s = {s:.4f}")
        a1.plot(xs, (1 + ODOM_SLOPE) * (xs - truths.min()) + dist[0], "--", color="#b03a2e",
                lw=1, label="s needed to explain odometry bias")
        a1.set_ylabel("L1 distance to wall (m)")
        a1.legend(frameon=False, fontsize=8)
        a2.axhline(0, color="0.6", lw=0.8)
        a2.plot(truths, 1000 * fit["resid"], "o", color="#1f4e79")
        a2.set_ylabel("residual (mm)")
        a2.set_xlabel("Bosch distance, front wheels to wall (m)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"figure -> {args.plot}")


if __name__ == "__main__":
    main()
