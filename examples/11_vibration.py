"""Measure sensor shake from the IMU already in a recording.

    python examples/11_vibration.py stationary.l1raw

WHY THIS IS ANSWERABLE FOR FREE. The L1's IMU runs at 250 Hz, so by Nyquist it
resolves anything below 125 Hz cleanly -- including the 11 Hz azimuth spin,
which is the usual culprit when a LiDAR on a 3D-printed mount buzzes. The data
is already in every recording you have made; nothing new needs capturing.

WHAT SHAKE ACTUALLY COSTS YOU. Three separate things, in rough order of damage:

1. ANGULAR SHAKE IS AMPLIFIED BY RANGE. A 1 degree wobble is 17 mm of point
   displacement at 1 m and 436 mm at 25 m. This is why a millimetre-scale
   vibration can push KISS-ICP's adaptive threshold up to a metre: model
   deviation is measured on the points, and the far ones move furthest.
2. DESKEW CANNOT REMOVE IT. Deskew assumes constant velocity across a frame.
   Oscillation is the opposite of constant velocity, and at 11 Hz a 0.2 s frame
   contains ~2 full cycles. That smear stays in the frame.
3. IT IS REAL MOTION, NOT ERROR. ICP reports it faithfully, which is why it
   shows up as path length while the net displacement stays ~0.

WHAT HELPS. Rotational shake is largely removable by the IMU pre-rotation you
already have (rotate_with_imu=True) -- at 250 Hz the IMU resolves an 11 Hz
oscillation with ~22 samples per cycle. TRANSLATIONAL shake is not: this design
uses the IMU for orientation only and never integrates the accelerometer.
"""

import argparse

import numpy as np

from l1_stream.protocol import LidarIMU
from l1_stream.recording import Replayer

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--spin-hz", type=float, default=11.0,
               help="LiDAR azimuth rate, the prime suspect for mount resonance")
p.add_argument("--frame-duration", type=float, default=0.2)
p.add_argument("--voxel-size", type=float, default=0.25)
args = p.parse_args()

stamps, gyro, accel = [], [], []
for _t, msg in Replayer(args.path).iter_messages():
    if isinstance(msg, LidarIMU):
        stamps.append(msg.stamp)
        gyro.append(msg.angular_velocity)
        accel.append(msg.linear_acceleration)

if len(stamps) < 64:
    raise SystemExit(f"Only {len(stamps)} IMU samples; need a longer recording.")

stamps = np.asarray(stamps, dtype=np.float64)
order = np.argsort(stamps)
stamps, gyro, accel = stamps[order], np.asarray(gyro)[order], np.asarray(accel)[order]

dt = np.diff(stamps)
fs = 1.0 / np.median(dt)
print(f"IMU  {len(stamps)} samples over {stamps[-1]-stamps[0]:.1f} s  ->  {fs:.1f} Hz")
print(f"     sample interval  median {np.median(dt)*1000:.2f} ms  "
      f"p99 {np.percentile(dt, 99)*1000:.2f} ms")
if np.percentile(dt, 99) > 3 * np.median(dt):
    print("     WARNING: irregular sampling; the spectrum below is approximate.")


def spectrum(sig, fs):
    """Dominant non-DC frequency and its amplitude, per axis and combined."""
    sig = sig - sig.mean(axis=0)            # drop DC: gravity and any bias
    win = np.hanning(len(sig))[:, None]
    mag = np.abs(np.fft.rfft(sig * win, axis=0)) * 2.0 / win.sum()
    freq = np.fft.rfftfreq(len(sig), 1.0 / fs)
    band = freq > 0.5                        # ignore slow drift and DC
    total = mag[band].sum(axis=1)
    peak = int(np.argmax(total))
    return freq[band][peak], mag[band][peak], freq[band], total


print("\n--- angular (gyro) ---")
g_rms = np.sqrt((gyro**2).mean(axis=0))
print(f"RMS rate      x {np.degrees(g_rms[0]):6.2f}  y {np.degrees(g_rms[1]):6.2f}  "
      f"z {np.degrees(g_rms[2]):6.2f}  deg/s")
gf, gm, gfreq, gtot = spectrum(gyro, fs)
print(f"dominant      {gf:.1f} Hz   amplitude {np.degrees(np.linalg.norm(gm)):.2f} deg/s")

# For sinusoidal motion, angular displacement = rate_amplitude / (2*pi*f).
theta = np.linalg.norm(gm) / (2 * np.pi * gf) if gf > 0 else 0.0
print(f"implied wobble  +/- {np.degrees(theta):.3f} deg at {gf:.1f} Hz")

print("\n--- linear (accel, gravity removed) ---")
a_ac = accel - accel.mean(axis=0)
print(f"RMS accel     x {np.sqrt((a_ac[:,0]**2).mean()):5.2f}  "
      f"y {np.sqrt((a_ac[:,1]**2).mean()):5.2f}  "
      f"z {np.sqrt((a_ac[:,2]**2).mean()):5.2f}  m/s^2")
af, am, _, _ = spectrum(accel, fs)
disp = np.linalg.norm(am) / (2 * np.pi * af) ** 2 if af > 0 else 0.0
print(f"dominant      {af:.1f} Hz   amplitude {np.linalg.norm(am):.2f} m/s^2")
print(f"implied shift  +/- {disp*1000:.2f} mm at {af:.1f} Hz")

print("\n--- distinct vibration peaks (gyro) ---")
# A windowed FFT spreads one physical peak over several adjacent bins, so a
# plain argsort returns the same peak five times. Take the max, blank a 1 Hz
# neighbourhood around it, repeat -- that yields genuinely separate modes.
work = gtot.copy()
for _ in range(5):
    i = int(np.argmax(work))
    if work[i] <= 0:
        break
    tag = "  <-- LiDAR azimuth spin" if abs(gfreq[i] - args.spin_hz) < 1.0 else ""
    print(f"  {gfreq[i]:6.1f} Hz   {np.degrees(work[i]):7.3f} deg/s{tag}")
    work[np.abs(gfreq - gfreq[i]) < 1.0] = 0.0

print("\n--- what it costs you ---")
print("point smear from the wobble alone (translation excluded):")
for rng in (1.0, 5.0, 10.0, 25.0):
    print(f"   at {rng:5.1f} m  ->  {1000*rng*theta:7.1f} mm"
          + ("   > voxel_size" if rng * theta > args.voxel_size else ""))
cycles = gf * args.frame_duration
print(f"\noscillation cycles inside one {args.frame_duration:.2f} s frame: {cycles:.1f}")
if cycles > 0.5:
    print("  Deskew assumes CONSTANT velocity. With more than half a cycle per")
    print("  frame that model is simply wrong, and the smear is not removable")
    print("  by shortening the frame either -- you would need to fix the mount.")
print("\nIf the dominant peak sits at the azimuth spin rate, your mount is")
print("resonating with the LiDAR's own rotor. STIFFEN it (shorter standoffs,")
print("thicker walls, more infill) to push resonance well above the spin rate.")
print("Adding mass lowers the resonant frequency toward the spin rate and makes")
print("it worse. Soft isolation (sorbothane, foam) also works, by decoupling.")
