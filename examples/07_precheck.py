"""Pre-flight. Everything you should know about a rig BEFORE tuning odometry.

    # motors idle, car parked, record ~60 s:
    python examples/07_precheck.py personal/stationary.l1raw

    # then spin the car in place ~360 deg and record again:
    python examples/07_precheck.py personal/pivot.l1raw

    # the histogram, spectrum and tables behind the summary:
    python examples/07_precheck.py personal/pivot.l1raw --verbose

This absorbs what used to be three scripts -- 07_diagnostics, 10_selfhit and
11_vibration. They were split by ANALYSIS (points / geometry / IMU), which is
the wrong seam: all three answer the same question, "is this rig fit to tune
on", and all three produce constants or go/no-go verdicts that
06_odometry_offline then consumes. Nothing here runs odometry, on purpose --
a script that CALIBRATES and a script that MEASURES must stay separate, or you
cannot tell whether a moved number came from new data or a new constant.

THE CATCH THE OLD SPLIT HID: the checks need different recordings, and they
CONTRADICT each other.

    self-hit radius   needs the car to ROTATE   (see below)
    IMU yaw drift     needs the car to SIT STILL

Run one recording through the old scripts and half the output was quietly
meaningless -- 07 would happily print "net yaw +38 deg, that is pure drift"
from a pivot recording, where it is not drift at all, it is the pivot. So this
script CLASSIFIES the recording first and runs only the checks that recording
can support, then names the ones it skipped and what to record to get them.

WHY THE SELF-HIT TEST NEEDS A PIVOT. A range histogram cannot separate your
chassis from the room: parked, a wall at 0.4 m and a motor mount at 0.4 m are
both static and look identical. Pivoting breaks the tie. A point on the
chassis keeps a FIXED range in a FIXED sensor-frame direction however the car
turns; a point on a wall does not. So low range variance in a fixed
(ring, azimuth) cell means rigidly attached, high variance means world
geometry. The near-range histogram is still printed on a pivot, where it
finally becomes a useful confirming view rather than a Rorschach test.

ON VIBRATION AND MOTORS. Idle motors contribute nothing, so a parked recording
measures the LiDAR's own 11 Hz rotor and the mount's response to it -- which is
most of the story. But it is not all of it: brushed 540s under load add
commutator ripple and rotor imbalance, and a parked recording has never seen
that. A pivot has both motors turning under load, so it measures shake in a
condition closer to driving. That makes the pivot the better recording for
BOTH of this script's headline numbers, which is the real reason these two
analyses belong in one file.
"""

import argparse
from pathlib import Path

import numpy as np

from l1_stream.config import DEFAULTS
from l1_stream.protocol import LidarIMU, LidarScan
from l1_stream.recording import Replayer

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("path")
p.add_argument("--near", type=float, default=0.30,
               help="show 1 cm range bins below this, 5 cm above")
p.add_argument("--az-bin", type=float, default=2.0, help="azimuth bin, degrees")
p.add_argument("--min-obs", type=int, default=25, help="min samples per cell")
p.add_argument("--static-std", type=float, default=0.03,
               help="range std below this = rigidly attached, metres")
# Deliberately NOT called --max-range. That name already means the odometry
# range gate, and one name for two different quantities is how this project
# has burned itself before.
p.add_argument("--analysis-range", type=float, default=6.0,
               help="ignore returns beyond this for the self-hit test")
p.add_argument("--spin-hz", type=float, default=11.0,
               help="LiDAR azimuth rate, prime suspect for mount resonance")
p.add_argument("--pivot-deg", type=float, default=45.0,
               help="yaw sweep above this counts as a pivot recording")
p.add_argument("--static-deg", type=float, default=5.0,
               help="yaw sweep below this counts as a stationary recording")
p.add_argument("--verbose", action="store_true",
               help="also print the histogram, spectrum and tables behind the summary")
args = p.parse_args()

# --- one pass over the recording -------------------------------------------

xs, ys, zs, rings, pkt_spans, pkt_counts, scan_stamps = [], [], [], [], [], [], []
imu_stamps, quats, gyro, accel = [], [], [], []

for _t, msg in Replayer(args.path).iter_messages():
    if isinstance(msg, LidarScan):
        pts = msg.points
        if not len(pts):
            continue
        xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)
        # Mask points and ring with the SAME mask or the two desync.
        keep = np.any(xyz != 0.0, axis=1)
        pkt_spans.append(float(np.ptp(pts["time"])))
        pkt_counts.append(int(msg.valid_points_num))
        scan_stamps.append(float(msg.stamp))
        if not keep.any():
            continue
        xs.append(xyz[keep, 0])
        ys.append(xyz[keep, 1])
        zs.append(xyz[keep, 2])
        rings.append(pts["ring"][keep].astype(np.int64))
    elif isinstance(msg, LidarIMU):
        imu_stamps.append(float(msg.stamp))
        quats.append(msg.quaternion)
        gyro.append(msg.angular_velocity)
        accel.append(msg.linear_acceleration)

if not xs:
    raise SystemExit("No usable scans in that recording.")

x = np.concatenate(xs)
y = np.concatenate(ys)
z = np.concatenate(zs)
ring = np.concatenate(rings)
r = np.sqrt(x * x + y * y + z * z)
pkt_spans = np.asarray(pkt_spans)
pkt_counts = np.asarray(pkt_counts)
scan_stamps = np.asarray(scan_stamps)

skipped: list[tuple[str, str]] = []
warnings: list[str] = []

# --- [1] stream health ------------------------------------------------------

duration = float(scan_stamps.max() - scan_stamps.min())
if imu_stamps:
    imu_t = np.sort(np.asarray(imu_stamps, dtype=np.float64))
    dt = np.diff(imu_t)
    fs = 1.0 / float(np.median(dt))
else:
    fs = None

# --- [2] what kind of recording is this? ------------------------------------

yaw_sweep = yaw_net = None
if quats:
    q = np.asarray(quats, dtype=np.float64)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    yaw_sweep = float(np.degrees(np.ptp(yaw)))
    yaw_net = float(np.degrees(yaw[-1] - yaw[0]))

if yaw_sweep is None:
    kind = "unknown"
elif yaw_sweep >= args.pivot_deg:
    kind = "pivot"
elif yaw_sweep <= args.static_deg:
    kind = "stationary"
else:
    kind = "ambiguous"

# --- [3] per-point time (a hard precondition for deskew) --------------------

spread = float(np.max(pkt_spans)) if len(pkt_spans) else 0.0
time_ok = spread > 1e-9

# --- [4] near-range structure ----------------------------------------------

near_edges = np.arange(0.0, args.near + 1e-9, 0.01)
far_edges = np.arange(args.near, 1.55, 0.05)
edges = np.concatenate([near_edges[:-1], far_edges])
counts, _ = np.histogram(r, bins=edges)
cur_min_range = DEFAULTS["min_range"]

# --- [5] self-hit radius (pivot only) ---------------------------------------

suggested_min_range = outer = None
bands = []
selfhit_line = ""
if kind != "pivot":
    selfhit_line = "skipped: needs a pivot (spin the car in place ~360 deg)"
    skipped.append(("self-hit radius / min_range",
                    "spin the car in place ~360 deg and record that"))
else:
    sel = r < args.analysis_range
    r_s, ring_s = r[sel], ring[sel]
    az = np.degrees(np.arctan2(y[sel], x[sel]))
    azbin = np.floor((az + 180.0) / args.az_bin).astype(np.int64)

    key = ring_s * 100000 + azbin
    _uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    s1 = np.bincount(inv, weights=r_s)
    s2 = np.bincount(inv, weights=r_s * r_s)
    mean = s1 / n
    std = np.sqrt(np.clip(s2 / n - mean * mean, 0.0, None))

    ok = n >= args.min_obs
    mean, std = mean[ok], std[ok]
    static = std < args.static_std

    if not static.any():
        selfhit_line = "no rigidly-attached returns found: pivot too small, re-record a bigger one"
        skipped.append(("self-hit radius / min_range",
                        "re-record a larger pivot; this one separated nothing"))
    else:
        band_edges = np.arange(0.0, min(args.analysis_range, 3.0) + 0.1, 0.1)
        for lo, hi in zip(band_edges[:-1], band_edges[1:], strict=True):
            band = (mean >= lo) & (mean < hi)
            if band.any():
                bands.append((lo, hi, int(band.sum()), float(static[band].mean())))
        outer = float(mean[static].max())
        suggested_min_range = outer + 0.05
        match = ("matches config" if abs(suggested_min_range - cur_min_range) < 0.03
                 else f"config has {cur_min_range:.2f} -- reconcile")
        selfhit_line = (f"chassis out to {outer:.3f} m -> min_range {suggested_min_range:.2f} "
                        f"suggested, discards "
                        f"{100 * float((r < suggested_min_range).mean()):.1f}% ({match})")
        if any(lo >= 0.9 and frac >= 0.99 for lo, _hi, _n, frac in bands):
            warnings.append("100% static out past 1 m usually means parked against a wall")

# --- [6] vibration ----------------------------------------------------------

smear = None
vib = None
if fs is None or len(imu_stamps) < 64:
    skipped.append(("vibration spectrum", "record at least a few seconds of IMU"))
else:
    order = np.argsort(np.asarray(imu_stamps, dtype=np.float64))
    g = np.asarray(gyro, dtype=np.float64)[order]
    a = np.asarray(accel, dtype=np.float64)[order]

    def highpass(sig, fs, cutoff_s=0.5):
        """Subtract a centred moving average.

        On a pivot the raw gyro RMS is dominated by the pivot itself, which is
        real rotation and not shake. Removing a 0.5 s moving mean leaves the
        oscillation and kills the manoeuvre, so the same number is comparable
        between a parked and a pivoting recording.
        """
        w = max(3, int(round(cutoff_s * fs)) | 1)
        if len(sig) <= 2 * w:
            return sig - sig.mean(axis=0)
        k = np.ones(w) / w
        base = np.stack([np.convolve(sig[:, i], k, mode="same")
                         for i in range(sig.shape[1])], axis=1)
        return (sig - base)[w:-w]

    def spectrum(sig, fs):
        """Dominant non-DC frequency and its per-axis amplitude."""
        sig = sig - sig.mean(axis=0)
        win = np.hanning(len(sig))[:, None]
        mag = np.abs(np.fft.rfft(sig * win, axis=0)) * 2.0 / win.sum()
        freq = np.fft.rfftfreq(len(sig), 1.0 / fs)
        band = freq > 0.5
        total = mag[band].sum(axis=1)
        i = int(np.argmax(total))
        return freq[band][i], mag[band][i], freq[band], total

    g_hp = highpass(g, fs)
    g_rms = np.sqrt((g_hp ** 2).mean(axis=0))
    gf, gm, gfreq, gtot = spectrum(g, fs)
    theta = float(np.linalg.norm(gm) / (2 * np.pi * gf)) if gf > 0 else 0.0
    # Below ~0.01 deg/s the "dominant frequency" is just the largest noise bin,
    # and reporting it as a resonance would be reading tea leaves.
    quiet = np.degrees(float(np.linalg.norm(gm))) < 0.01
    af, am, _, _ = spectrum(a, fs)
    disp = float(np.linalg.norm(am) / (2 * np.pi * af) ** 2) if af > 0 else 0.0

    # A windowed FFT spreads one physical peak over several adjacent bins, so a
    # plain argsort returns the same peak five times. Take the max, blank a
    # 1 Hz neighbourhood, repeat -- that yields genuinely separate modes.
    # Stop at 1% of the dominant: below that they are noise floor.
    peaks = []
    work = gtot.copy()
    floor = 0.01 * float(work.max())
    for _ in range(5):
        i = int(np.argmax(work))
        if work[i] <= floor:
            break
        peaks.append((float(gfreq[i]), float(work[i])))
        work[np.abs(gfreq - gfreq[i]) < 1.0] = 0.0

    smear = DEFAULTS["max_range"] * theta
    cycles = gf * DEFAULTS["frame_duration"]
    vib = {"gf": gf, "gm": gm, "theta": theta, "quiet": quiet, "af": af, "am": am,
           "disp": disp, "g_rms": g_rms, "peaks": peaks, "cycles": cycles}
    if not quiet:
        # Only worth shouting about when the wobble is big enough to matter.
        if cycles > 0.5:
            warnings.append(f"{cycles:.1f} vibration cycles per frame: too fast for "
                            "deskew's constant-velocity model -- fix the mount")
        if abs(gf - args.spin_hz) < 1.0:
            warnings.append("vibration peak at the LiDAR spin rate: mount resonance "
                            "-- stiffen it (mass makes it worse)")
        if smear > DEFAULTS["voxel_size"]:
            warnings.append(f"wobble smear {1000 * smear:.0f} mm at "
                            f"{DEFAULTS['max_range']:.0f} m exceeds the registration voxel")

# --- [7] IMU yaw drift (stationary only) ------------------------------------

if kind != "stationary":
    skipped.append(("IMU yaw drift rate", "park the car and record ~60 s"))

# --- report -----------------------------------------------------------------

sweep = f"  (IMU yaw swept {yaw_sweep:.1f} deg)" if yaw_sweep is not None else ""
print(f"{Path(args.path).name}   {duration:.1f} s   {kind.upper()}{sweep}")
if kind == "ambiguous":
    warnings.insert(0, f"between --static-deg {args.static_deg} and --pivot-deg "
                       f"{args.pivot_deg}: re-record parked, or spin ~360 deg")
print()

imu_txt = f"IMU {fs:.1f} Hz" if fs else "IMU none"
print(f"  stream      {pkt_counts.sum() / duration:,.0f} pts/s   "
      f"{len(pkt_counts) / duration:.0f} packets/s   {imu_txt}")
if time_ok:
    print(f"  point time  {spread * 1000:.3f} ms per packet   OK, deskew possible")
else:
    print("  point time  0 -- publisher does not fill points['time']   FAILED")
    warnings.append("no per-point times: deskew falls back to one timestamp per packet")
print(f"  near range  closest return {r.min():.3f} m   min_range {cur_min_range:.2f} "
      f"discards {100 * float((r < cur_min_range).mean()):.2f}%")
print(f"  self-hit    {selfhit_line}")
if vib is None:
    print("  vibration   skipped: not enough IMU")
elif vib["quiet"]:
    print(f"  vibration   none measurable   (smear {1000 * smear:.1f} mm at "
          f"{DEFAULTS['max_range']:.0f} m)")
else:
    print(f"  vibration   {vib['gf']:.1f} Hz, +/-{np.degrees(vib['theta']):.3f} deg wobble "
          f"-> {1000 * smear:.1f} mm at {DEFAULTS['max_range']:.0f} m "
          f"({100 * smear / DEFAULTS['voxel_size']:.0f}% of a voxel)")
if kind == "stationary":
    print(f"  yaw drift   {yaw_net / duration * 60:+.2f} deg/min")
else:
    print("  yaw drift   skipped: needs a parked recording")

if warnings:
    print()
    for w in warnings:
        print(f"  ! {w}")
if skipped:
    print("\n  not checked: " + "; ".join(f"{what} ({how})" for what, how in skipped))

if not args.verbose:
    print("\n  --verbose for the histogram, spectrum and tables behind these numbers")
    raise SystemExit(0)

# --- detail (--verbose) -----------------------------------------------------

print("\n--- stream")
print(f"  packets {len(pkt_counts)}   points {len(r)} kept / {pkt_counts.sum()} reported   "
      f"per packet mean {pkt_counts.mean():.1f} (min {pkt_counts.min()}, "
      f"max {pkt_counts.max()})   rings {sorted(np.unique(ring).tolist())}")
if fs:
    print(f"  imu {len(imu_t)} samples, interval median {np.median(dt) * 1000:.2f} ms, "
          f"p99 {np.percentile(dt, 99) * 1000:.2f} ms")

print(f"\n--- range histogram (1 cm bins below {args.near:.2f} m)")
peak = counts.max() or 1
run_start = None
for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True):
    if c == 0 and lo < args.near:
        run_start = lo if run_start is None else run_start
        continue
    if run_start is not None:
        print(f"  {run_start:4.2f}-{lo:4.2f} m  {0:8d}  (no returns)")
        run_start = None
    if c:
        print(f"  {lo:4.2f}-{hi:4.2f} m  {c:8d}  {'#' * int(40 * c / peak)}")
if run_start is not None:
    print(f"  {run_start:4.2f}-{args.near:4.2f} m  {0:8d}  (no returns)")
print("\n  min_range cut-off -> returns discarded")
for mr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    star = "  <- config" if abs(mr - cur_min_range) < 1e-9 else ""
    print(f"  {mr:.2f} m  {100 * float((r < mr).mean()):5.2f}%{star}")

if bands:
    print("\n--- self-hit: static-return fraction by range band")
    for lo, hi, ncell, frac in bands:
        print(f"  {lo:4.2f}-{hi:4.2f} m  {ncell:4d} cells  {100 * frac:5.1f}% static  "
              f"{'#' * int(40 * frac)}")

if vib is not None:
    gr = np.degrees(vib["g_rms"])
    print("\n--- vibration")
    print(f"  gyro RMS (high-passed)  x {gr[0]:.2f}  y {gr[1]:.2f}  z {gr[2]:.2f} deg/s")
    print(f"  gyro dominant  {vib['gf']:.1f} Hz, {np.degrees(np.linalg.norm(vib['gm'])):.2f} "
          f"deg/s   accel dominant {vib['af']:.1f} Hz, {np.linalg.norm(vib['am']):.2f} m/s^2 "
          f"-> +/-{vib['disp'] * 1000:.2f} mm")
    print("  distinct gyro peaks: " + ", ".join(
        f"{f:.1f} Hz {np.degrees(m):.3f} deg/s"
        + (" (spin)" if abs(f - args.spin_hz) < 1.0 else "") for f, m in vib["peaks"]))
    print("  smear from wobble: " + ", ".join(
        f"{rng:g} m {1000 * rng * vib['theta']:.1f} mm"
        for rng in (1.0, 5.0, 10.0, DEFAULTS["max_range"])))
    print(f"  {vib['cycles']:.1f} oscillation cycles per {DEFAULTS['frame_duration']:.2f} s frame")

if yaw_sweep is not None:
    print(f"\n--- yaw   swept {yaw_sweep:.1f} deg, net {yaw_net:+.1f} deg over {duration:.1f} s")
