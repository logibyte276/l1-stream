"""Pre-flight. Everything you should know about a rig BEFORE tuning odometry.

    # motors idle, car parked, record ~60 s:
    python examples/07_precheck.py personal/stationary.l1raw

    # then spin the car in place ~360 deg and record again:
    python examples/07_precheck.py personal/pivot.l1raw

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

# --- [1] stream health ------------------------------------------------------

duration = float(scan_stamps.max() - scan_stamps.min())
print(f"[1] stream health   ({duration:.1f} s of recording)")
print(f"    packets   {len(pkt_counts)}  ->  {len(pkt_counts)/duration:.0f}/s")
print(f"    points    {len(r)} kept, {pkt_counts.sum()} reported  ->  "
      f"{pkt_counts.sum()/duration:.0f} pts/s")
print(f"    per packet  mean {pkt_counts.mean():.1f}  "
      f"min {pkt_counts.min()}  max {pkt_counts.max()}")
print(f"    rings     {sorted(np.unique(ring).tolist())}")
if imu_stamps:
    imu_t = np.sort(np.asarray(imu_stamps, dtype=np.float64))
    dt = np.diff(imu_t)
    fs = 1.0 / float(np.median(dt))
    print(f"    imu       {len(imu_t)} samples  ->  {fs:.1f} Hz  "
          f"(interval median {np.median(dt)*1000:.2f} ms, "
          f"p99 {np.percentile(dt, 99)*1000:.2f} ms)")
else:
    fs = None
    print("    imu       NONE in this recording")

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

print(f"\n[2] recording type: {kind.upper()}"
      + (f"   (IMU yaw swept {yaw_sweep:.1f}deg)" if yaw_sweep is not None else ""))
if kind == "ambiguous":
    print(f"    Between --static-deg ({args.static_deg}) and --pivot-deg "
          f"({args.pivot_deg}). Too much turning to trust as a drift measurement,")
    print("    too little to separate chassis from room. Re-record one or the other.")

# --- [3] per-point time (a hard precondition for deskew) --------------------

spread = float(np.max(pkt_spans)) if len(pkt_spans) else 0.0
print(f"\n[3] points['time'] spread: {spread*1000:.3f} ms per packet")
time_ok = spread > 1e-9
if not time_ok:
    print("    ZERO -> the publisher does not fill it. FrameAssembler falls back")
    print("    to one timestamp per packet, and deskew loses its resolution.")
else:
    print(f"    populated. At 21,600 pts/s that implies {spread*21600:.0f} "
          f"points per packet.")

# --- [4] near-range structure ----------------------------------------------

near_edges = np.arange(0.0, args.near + 1e-9, 0.01)
far_edges = np.arange(args.near, 1.55, 0.05)
edges = np.concatenate([near_edges[:-1], far_edges])
counts, _ = np.histogram(r, bins=edges)
peak = counts.max() or 1

print(f"\n[4] range histogram (1 cm bins below {args.near:.2f} m)")
# Empty near-field bins are reported, because "nothing at all below 0.10 m" is
# a real finding -- but collapsed into one line, since 30 lines of "(empty)"
# is noise rather than information.
run_start = None
for lo, hi, c in zip(edges[:-1], edges[1:], counts, strict=True):
    if c == 0 and lo < args.near:
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
    frac = 100 * float((r < mr).mean())
    star = "  <- current min_range" if abs(mr - DEFAULTS["min_range"]) < 1e-9 else ""
    print(f"      min_range {mr:.2f} m  discards {frac:5.2f}% of returns{star}")

# --- [5] self-hit radius (pivot only) ---------------------------------------

suggested_min_range = None
print("\n[5] self-hit radius")
if kind != "pivot":
    print("    SKIPPED -- needs a pivot recording. Parked, your chassis and a")
    print("    nearby wall are both static and indistinguishable by range alone.")
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
    print(f"    cells with >={args.min_obs} obs: {int(ok.sum())}   "
          f"rigidly attached (std < {args.static_std} m): {int(static.sum())}")

    if not static.any():
        print("    No rigidly-attached returns found. Either the pivot was too")
        print("    small to separate them, or nothing on the chassis is in view.")
        skipped.append(("self-hit radius / min_range",
                        "re-record a larger pivot; this one separated nothing"))
    else:
        print("\n    static-return fraction by range band:")
        band_edges = np.arange(0.0, min(args.analysis_range, 3.0) + 0.1, 0.1)
        for lo, hi in zip(band_edges[:-1], band_edges[1:], strict=True):
            band = (mean >= lo) & (mean < hi)
            if not band.any():
                continue
            frac = float(static[band].mean())
            print(f"      {lo:4.2f}-{hi:4.2f} m  {int(band.sum()):4d} cells  "
                  f"{100*frac:5.1f}% static  {'#' * int(40 * frac)}")
        outer = float(mean[static].max())
        suggested_min_range = outer + 0.05
        print(f"\n    outermost rigidly-attached return: {outer:.3f} m")
        print(f"    SUGGESTED min_range = {suggested_min_range:.2f} m "
              f"(outermost + 5 cm), discarding "
              f"{100*float((r < suggested_min_range).mean()):.1f}% of all returns")
        print("    Sanity-check in the visualiser first: a band that is 100% static")
        print("    out to 1 m usually means the car is parked against a wall.")

# --- [6] vibration ----------------------------------------------------------

print("\n[6] vibration")
smear = None
if fs is None or len(imu_stamps) < 64:
    print(f"    SKIPPED -- only {len(imu_stamps)} IMU samples; need a longer "
          f"recording.")
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
    print(f"    gyro RMS (high-passed)  x {np.degrees(g_rms[0]):6.2f}  "
          f"y {np.degrees(g_rms[1]):6.2f}  z {np.degrees(g_rms[2]):6.2f}  deg/s")
    if kind == "pivot":
        print("    high-passed so the pivot itself does not count as shake.")

    gf, gm, gfreq, gtot = spectrum(g, fs)
    theta = float(np.linalg.norm(gm) / (2 * np.pi * gf)) if gf > 0 else 0.0
    # Below ~0.01 deg/s the "dominant frequency" is just the largest noise bin,
    # and reporting it as a resonance would be reading tea leaves.
    quiet = np.degrees(float(np.linalg.norm(gm))) < 0.01
    if quiet:
        print("    no measurable angular vibration (dominant peak under "
              "0.01 deg/s -- noise floor)")
    else:
        print(f"    dominant  {gf:.1f} Hz, amplitude "
              f"{np.degrees(np.linalg.norm(gm)):.2f} deg/s  ->  wobble "
              f"+/- {np.degrees(theta):.3f} deg")

    af, am, _, _ = spectrum(a, fs)
    disp = float(np.linalg.norm(am) / (2 * np.pi * af) ** 2) if af > 0 else 0.0
    print(f"    accel dominant  {af:.1f} Hz, "
          f"{np.linalg.norm(am):.2f} m/s^2  ->  shift +/- {disp*1000:.2f} mm")

    print("\n    distinct peaks (gyro):")
    # A windowed FFT spreads one physical peak over several adjacent bins, so a
    # plain argsort returns the same peak five times. Take the max, blank a
    # 1 Hz neighbourhood, repeat -- that yields genuinely separate modes.
    # Stop at 1% of the dominant: below that they are noise floor, and printing
    # five "0.000 deg/s" lines makes a quiet rig look like it has five modes.
    work = gtot.copy()
    floor = 0.01 * float(work.max())
    shown = 0
    for _ in range(5):
        i = int(np.argmax(work))
        if work[i] <= floor:
            break
        tag = "  <- LiDAR azimuth spin" if abs(gfreq[i] - args.spin_hz) < 1.0 else ""
        print(f"      {gfreq[i]:6.1f} Hz   {np.degrees(work[i]):7.3f} deg/s{tag}")
        work[np.abs(gfreq - gfreq[i]) < 1.0] = 0.0
        shown += 1
    if shown <= 1:
        print("      (nothing else above 1% of the dominant peak)")

    print("\n    point smear from the wobble alone (translation excluded):")
    voxel = DEFAULTS["voxel_size"]
    for rng in (1.0, 5.0, 10.0, DEFAULTS["max_range"]):
        s = rng * theta
        print(f"      at {rng:5.1f} m  ->  {1000*s:7.1f} mm"
              + (f"   > voxel_size {voxel}" if s > voxel else ""))
    smear = DEFAULTS["max_range"] * theta

    cycles = gf * DEFAULTS["frame_duration"]
    print(f"\n    {cycles:.1f} oscillation cycles inside one "
          f"{DEFAULTS['frame_duration']:.2f} s frame")
    if cycles > 0.5:
        print("    Deskew assumes CONSTANT velocity. Above half a cycle per frame")
        print("    that model is simply wrong, and shortening the frame does not")
        print("    fix it -- the mount does.")
    if abs(gf - args.spin_hz) < 1.0:
        print("    Dominant peak sits at the azimuth spin rate: the mount is")
        print("    resonating with the LiDAR's own rotor. STIFFEN it (shorter")
        print("    standoffs, thicker walls, more infill) to push resonance above")
        print("    the spin rate. Adding MASS lowers it toward the spin rate and")
        print("    makes it worse. Soft isolation also works, by decoupling.")

# --- [7] IMU yaw drift (stationary only) ------------------------------------

print("\n[7] IMU yaw drift")
if kind != "stationary":
    print(f"    SKIPPED -- the car turned ({yaw_sweep:.1f}deg swept), so net yaw is"
          if yaw_sweep is not None else "    SKIPPED -- no IMU in this recording.")
    if yaw_sweep is not None:
        print("    mostly the manoeuvre, not drift. Needs a parked recording.")
    skipped.append(("IMU yaw drift rate", "park the car and record ~60 s"))
else:
    print(f"    swept {yaw_sweep:.1f}deg, net {yaw_net:+.1f}deg over {duration:.1f} s "
          f"-> {yaw_net/duration*60:+.2f} deg/min")
    print("    The car did not turn, so that is pure gyro drift: a 6-axis IMU has")
    print("    no heading reference, and yaw is unobservable.")

# --- [8] verdict ------------------------------------------------------------

print("\n" + "=" * 68)
print("VERDICT")
print(f"  deskew precondition   {'OK' if time_ok else 'FAILED'} "
      f"(per-point times {'populated' if time_ok else 'absent'})")
if suggested_min_range is not None:
    cur = DEFAULTS["min_range"]
    verdict = "matches config" if abs(suggested_min_range - cur) < 0.03 else \
        f"config has {cur:.2f} -- reconcile"
    print(f"  min_range             {suggested_min_range:.2f} m measured, {verdict}")
if smear is not None:
    v = DEFAULTS["voxel_size"]
    print(f"  vibration             {1000*smear:.1f} mm smear at "
          f"{DEFAULTS['max_range']:.0f} m vs {1000*v:.0f} mm voxel "
          f"({100*smear/v:.1f}% of a voxel)")
if skipped:
    print("\n  NOT CHECKED on this recording:")
    for what, how in skipped:
        print(f"    - {what}\n        {how}")
print("=" * 68)
print("\nThen tune: python examples/06_odometry_offline.py <a DRIVE recording>")
