# l1-stream

[![CI](https://github.com/logibyte276/l1-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/logibyte276/l1-stream/actions/workflows/ci.yml)

A Python client for a **Unitree L1 LiDAR** streamed over UDP: packet parsing,
thread-safe buffering, IMU-based orientation compensation, an optional live
Open3D view, and — on top of that — wire-level recording/replay and **KISS-ICP
LiDAR odometry**.

Built for the awkward part of working with this sensor — a single scan packet
carries at most 120 points, arriving at ~180 Hz on a separate datagram from the
~250 Hz IMU stream, so anything useful means buffering both and matching them by
timestamp without losing packets when your consumer loop stutters.

- **Pure Python, one hard dependency** — numpy. Open3D and kiss-icp are
  optional, so the package installs in seconds on a headless Jetson.
- **No hardware needed to develop or test.** `pack_imu_packet` /
  `pack_scan_packet` build byte-identical datagrams in memory, so the whole
  pipeline is testable offline. Not one test touches a sensor.
- **Importable, not just runnable.** Nothing opens a socket, spawns a thread, or
  creates a window at import time.

## Prerequisite: something must already be publishing

This package is a **consumer only** — it never touches the serial port. Before
any of it works, the Unitree LiDAR SDK's UDP publisher must be running on the
machine the sensor is plugged into:

```bash
# from your unilidar_sdk build directory
./unilidar_publisher_udp /dev/ttyUSB0                    # -> 127.0.0.1:12345 (see below re: port names)
./unilidar_publisher_udp /dev/ttyUSB0 192.168.1.50 12345 # -> another machine
```

The SDK lives at [unitreerobotics/unilidar_sdk](https://github.com/unitreerobotics/unilidar_sdk).
Confirm packets are flowing with `l1-monitor` before writing any code — if the
counters stay at zero, the problem is upstream of this package.

### Which serial port?

`/dev/ttyUSB0` is just the usual default, not a rule. **Nothing in this package
touches the serial port** — that is entirely the publisher's business, and the
UDP split is exactly why you can develop the Python side on a laptop that has
never seen the sensor. What matters here is the **UDP port** (`--port`, default
12345), which is independent of whatever serial device the publisher opened.

To find your serial device:

```bash
ls /dev/ttyUSB* /dev/ttyACM*   # candidates (ACM shows up for some USB chips)
sudo dmesg -w                  # then unplug/replug the LiDAR and watch
lsusb                          # identify the USB-to-serial chip
```

You also need permission to open it, or you get a bare "failed to open serial
port" with no hint that it is a permissions problem:

```bash
sudo usermod -aG dialout $USER   # then log out and back in
```

### The renumbering trap

`/dev/ttyUSB0` is **not stable across reboots**. The number is assigned in the
order devices enumerate, so if you have more than one USB-serial device — say a
LiDAR *and* a microcontroller driving motors — they can silently swap. Then your
publisher opens the microcontroller, your motor code opens the LiDAR, both fail
in confusing ways, and neither error message says "wrong device."

Fix it once with a udev rule that gives each device a stable name. First read
its attributes:

```bash
udevadm info -a -n /dev/ttyUSB0 | grep -m3 -E 'idVendor|idProduct|serial'
```

Then create `/etc/udev/rules.d/99-robot-serial.rules`:

```
# Replace the IDs with what udevadm printed for YOUR devices.
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", ATTRS{serial}=="0001", SYMLINK+="lidar"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", ATTRS{serial}=="mcu"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/lidar /dev/mcu
```

Now run `./unilidar_publisher_udp /dev/lidar` and the name never moves again.

**Caveat:** cheap CH340 chips often ship with **no unique serial number**, so
`ATTRS{serial}` can't tell two of them apart. If that is your situation, match on
the physical USB port instead and always plug into the same one:

```
SUBSYSTEM=="tty", KERNELS=="1-2.3", SYMLINK+="mcu"
```

Get the `KERNELS` value from the `looking at parent device` lines of the same
`udevadm info -a` output.

## Install

```bash
git clone https://github.com/logibyte276/l1-stream.git
cd l1-stream

pip install -e .            # core: numpy only
pip install -e ".[viz]"     # + Open3D for the live viewer
pip install -e ".[dev]"     # + pytest, scipy, ruff
pip install -e ".[slam]"    # + kiss-icp, for the odometry pipeline
```

The odometry tests **skip** without the `[slam]` extra — see
[Development](#development).

### On aarch64 (Jetson): Python 3.10 and numpy<2

Developed and tested on an NVIDIA Jetson Orin Nano (JetPack 6, Ubuntu 22.04,
aarch64) with Python 3.10, numpy 1.26.4, and Open3D 0.18.0.

```bash
conda create -n l1 python=3.10
conda activate l1
export PIP_CONSTRAINT=$(pwd)/constraints.txt
pip install -e ".[viz,dev,slam]"
```

Two constraints, both driven by Open3D:

**Python 3.10.** Open3D publishes no aarch64 wheel for Python 3.13 —
`pip install open3d` there reports `from versions: none`. That sets the upper
bound on the interpreter version.

**numpy<2.** On aarch64/py310 the newest available Open3D is 0.18.0, which was
built against numpy 1.x headers and breaks under numpy 2's C ABI. Open3D 0.18
declares no numpy upper bound of its own, so pip will happily resolve numpy 2
and produce an install that imports cleanly and then fails at runtime.
`constraints.txt` pins `numpy<2`. Set `PIP_CONSTRAINT` before installing
*anything* into the environment, so a later install (scipy, matplotlib, evo —
anything that depends on numpy) cannot silently upgrade numpy out from under
Open3D.

To make the constraint permanent for a conda env:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export PIP_CONSTRAINT=$(pwd)/constraints.txt" \
  > "$CONDA_PREFIX/etc/conda/activate.d/pip_constraint.sh"
```

Neither constraint applies on x86_64, where newer Open3D and numpy 2 work fine.
The core package is unaffected either way — the ceiling belongs to the `[viz]`
extra, not to `l1-stream` itself, so a headless install stays unconstrained.

---

## Quick start

```python
from l1_stream import LidarStream

with LidarStream.for_history(seconds=2.0) as lidar:
    if not lidar.wait_until_ready(timeout=5.0):
        raise SystemExit("no data — is the SDK UDP publisher running?")

    scan = lidar.latest_scan
    xyz = scan.xyz(drop_zero_returns=True)   # (N, 3) float64, C-contiguous
    print(len(xyz), "points", lidar.stats())
```

Accumulate and de-rotate a rolling window:

```python
from l1_stream import LidarStream, RotatedScanAccumulator

acc = RotatedScanAccumulator(max_scans=150, max_time_gap=0.01)
with LidarStream.for_history(2.0) as lidar:
    while True:
        acc.update(lidar)
        points = acc.get_points()   # (N, 3) float64, one common frame
```

Record a drive, then replay it through odometry as many times as you like:

```python
from l1_stream.offline import replay

run = replay("drive_01.l1raw")           # tuned config by default
print(run.path_length, run.net_displacement, run.jitter_mm)
```

Live view:

```python
from l1_stream.visualizer import LiveVisualizer
LiveVisualizer(max_scans=150, refresh_hz=30).run()
```

Command line:

```bash
l1-monitor --port 12345              # throughput + drop counters
l1-visualize --max-scans 200         # live Open3D window
```

`examples/` is numbered in the order you actually use it. **01–04** cover the
library itself and need no odometry; `04_offline_replay.py` needs no hardware at
all. **05–09** are the recording-and-odometry workflow, described next.

---

## Odometry

```bash
# 1. record
python examples/05_record.py drive_01.l1raw --duration 60

# 2. pre-flight -- is this rig fit to tune on?
python examples/07_precheck.py pivot.l1raw        # self-hit radius, vibration
python examples/07_precheck.py stationary.l1raw   # IMU drift, stream health

# 3. tune offline against a DRIVE recording
python examples/06_odometry_offline.py drive_01.l1raw --truth 5.0

# 4. change exactly one thing and compare
python examples/09_ablation.py drive_01.l1raw --ablate deskew

# 5. only once the parameters are settled
python examples/08_odometry_live.py
```

Replay is not paced, so a 60 s drive re-runs in a second or two. That is the
whole reason to record before wiring odometry into the live loop.

`07_precheck.py` wants **two different recordings** and tells you which checks
each one supports. The self-hit test needs the car to rotate; the IMU drift test
needs it to sit still. Those contradict, so the script classifies the recording
and names what it skipped rather than printing a meaningless number.

### One config, enforced

Every tuned constant lives in **`src/l1_stream/config.py`** and nowhere else.
`FrameAssembler` and `KissOdometry` take their signature defaults from it, so
constructing either one bare gives the tuned configuration.

```python
DEFAULTS = {
    "frame_duration":    0.2,    # 0.05 and 0.5 both measured worse
    "rotate_with_imu":   True,   # ESSENTIAL: 32-126x worse on loop closure without
    "voxel_size":        0.15,   # 0.10 also real-time viable; 0.15 keeps margin
    "max_range":         25.0,   # trimming to 10 measurably hurt rotation
    "min_range":         0.25,   # chassis measured at 0.2 m radius, plus margin
    "deskew":            False,  # UNRESOLVED -- see Known limitations
    "initial_threshold": 0.4,    # adaptive settles at 0.32-0.55
}
```

`UNTUNED` in the same file holds parameters that are exposed but were never
swept, listed separately so nobody mistakes *"it is in the config"* for
*"somebody measured it"*.

This is **enforced, not merely documented**. `tests/test_config.py` fails if a
signature default drifts from `config.py`, or if any example writes a literal
number for a tuned parameter — including as an `argparse` default.

Why it needs enforcing: four scripts each grew their own copy of these constants
and drifted apart. `08_odometry_live.py` — the one that drives the actual car —
ran `voxel 0.25 / min_range 0.40 / deskew on` for weeks after tuning had moved
all three. Every copy looked plausible in isolation, and their outputs looked
comparable when they were not.

### kiss-icp gotchas, all verified against 1.3.0

- `register_frame(frame, timestamps)` returns `(frame, source)` — the processed
  clouds, **not the pose**. The pose is `odom.last_pose`.
- `mapping.voxel_size` defaults to `None`, which **crashes** `VoxelHashMap`.
- Mutating the config after `KissICP(cfg)` is **inert**. `voxelize()` is the
  lone exception; it reads the config live.
- The deskew reference is the **end** of a frame, so per-point timestamps must
  put `1.0` on the last point and the pose belongs to `t_end`.
- The "adaptive threshold" is a rotation-error readout scaled roughly linearly
  by `max_range`, not a distance you can reason about directly.
- Frames must be **disjoint**. Overlapping frames bias ICP toward zero motion,
  because the shared points already align at zero displacement.

These are version-specific, which is why `[slam]` should pin narrowly rather
than accept any 1.x.

---

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | `DEFAULTS` / `UNTUNED`. The single source of tuned constants. Imports nothing else. |
| `protocol.py` | Wire format, dataclasses, parse **and pack** functions. Pure — no I/O, no threads, no state. |
| `ring_buffer.py` | `RingBuffer`: thread-safe, bounded, drop-oldest. |
| `receiver.py` | `LidarUDPReceiver`: blocking, one packet at a time. |
| `stream.py` | `LidarStream`: background reader thread → two ring buffers. |
| `rotation.py` | Quaternion math + `RotatedScanAccumulator`. |
| `recording.py` | `DatagramRecorder` / `Replayer`: the raw wire, byte for byte. |
| `frames.py` | `FrameAssembler`: disjoint frames with per-point timestamps. |
| `odometry.py` | `KissOdometry`: frame in, pose out. |
| `offline.py` | `replay()`: the one implementation of "run a recording through the pipeline". |
| `visualizer.py` | `LiveVisualizer`. Lazily imports Open3D. |
| `cli.py` | `l1-monitor`, `l1-visualize`. |

The split exists so the parts you can test without hardware are separated from
the parts you can't. `protocol.py`, `ring_buffer.py`, `rotation.py`, `config.py`
and `frames.py` have no I/O at all and are fully covered by tests.

### Wire format

```
[msgType: uint32][dataSize: uint32][payload ...]

  101  IMU   "=dI4f3f3f"                 → 52 bytes
  102  Scan  "=dII" + 120 × "fffffI"     → 2896 bytes
```

**Quaternions are `(x, y, z, w)` — scalar LAST**, matching `float quaternion[4];
// [x,y,z,w]` in `unitree_lidar_sdk.h`. Open3D, ROS `tf`, and Eigen's constructor
all use scalar-*first*. Mixing them gives a valid-looking but wrong rotation that
is nearly impossible to spot by eye, so `LidarIMU.quaternion_wxyz()` exists to
make the conversion explicit at every boundary.

The convention is verified, not assumed. `test_matches_scipy` cross-checks
`rotate_points` against `scipy.spatial.transform.Rotation` (which also takes
scalar-last) over random quaternions. That check earns its keep because the
suite's other rotation tests compare `rotate_points` to `quaternion_to_matrix` —
both this package's own code — so a shared convention error would pass them both
while producing silently wrong geometry. scipy is the only independent reference
in the suite, which is why the test uses `pytest.importorskip` rather than a
`try`/`except ImportError`/`return`: a missing dependency must surface as
**SKIPPED** in the pytest summary, not as a green PASS for a test that never ran.

---

## Design notes

Decisions that aren't obvious from the API, and the reasoning behind them.

### Buffering

Both ring buffers are **bounded and drop-oldest**. On a moving robot a stale
reading is worse than no reading — a scan from five seconds ago doesn't describe
where anything is now. Blocking the producer instead would stall the socket
thread and make the kernel drop packets anyway, just less visibly. Watch
`total_dropped` in `stats()`: if it climbs steadily, your consumer is slower than
the sensor and you are sampling rather than capturing.

Size buffers in **seconds of history**, not packets — `LidarStream.for_history(2.0)`.
The two buffers should span comparable wall-clock time; if the IMU buffer covers
less than the scan buffer, timestamp matching quietly starts failing for the
oldest scans.

The OS receive buffer matters too. Linux defaults to ~208 KB, about 0.4 s of scan
traffic, so `LidarUDPReceiver` requests 1 MB via `SO_RCVBUF`.

### Scan/IMU matching

Scans and IMU samples arrive as separate datagrams, so a scan frequently shows up
*before* the sample that timestamps it. `RotatedScanAccumulator` holds those in a
pending queue and retries, rather than treating "no match yet" as "no match ever."
Scans too old to ever match are dropped **and counted** in
`stats()["scans_unmatched"]`.

Matching is a binary search over sorted timestamps — O(S log I) — and the sort is
not decorative: UDP does not guarantee delivery order.

### Why frames are not accumulator output

`RotatedScanAccumulator` exists to make a *picture*: a rolling window of the last
N scans. Registration needs the opposite — **disjoint** frames carrying per-point
times. Reusing the accumulator via `get_points()` + `reset()` fails three ways:
the per-point timestamps are already gone, `reset()` also clears the pending
queue (systematically discarding each frame's newest scans, uncounted), and frame
spans would then follow your call cadence rather than capture time, which breaks
the constant-velocity motion model. `FrameAssembler` re-does the IMU matching and
cuts on capture time instead.

### Rotation

`rotate_points` uses the vector form `v + 2w(q×v) + 2q×(q×v)`, which is ~15 flops
per point versus building a 3×3 matrix, worth it when you rotate one small scan at
a time thousands of times a second. It is only a *rotation* for a unit quaternion —
off-norm input scales every point by |q|² — so input is normalised, with an
identity fallback for the degenerate all-zero case a sensor can emit during warm-up.

Each scan is rotated **once** on ingestion, not once per displayed frame, and the
concatenated output is cached until something changes.

### Recording the wire, not the objects

A recording is only useful for debugging if it survives a change to the parser.
Pickling `LidarScan` objects freezes today's interpretation of the bytes, so the
first time you fix a parsing bug every old recording becomes a record of the bug.
Storing datagrams keeps the recording authoritative — `parse_packet` runs at
replay time, so a fixed parser retroactively fixes every file you already have.
It also preserves the cases most worth studying: truncated datagrams from MTU
fragmentation loss, non-zero padding slots, and unknown message types.

### Parsing

The parser is deliberately paranoid about the wire, because UDP gives you no
framing guarantees and a publisher may transmit more bytes than are meaningful:

- Point counts are clamped against **both** `validPointsNum` and the bytes that
  actually arrived, so a truncated datagram yields the points that survived rather
  than an exception.
- Only the first `validPointsNum` points are read. Publishers commonly transmit
  all 120 fixed slots regardless, and the padding is not guaranteed to be zeroed —
  a receiver that trusts the array length instead can see garbage geometry.
  `test_scan_ignores_padding_slots` pins this behaviour.
- Parsed arrays are copied out of the receive buffer. `np.frombuffer` alone returns
  a read-only view that also pins the whole datagram in memory — both bad once the
  array lands in a ring buffer.
- `parse_packet` never raises. It runs on a background thread, where an escaping
  exception would kill the reader silently and freeze every buffer with no error.

### Threading

The reader thread helps despite the GIL because it spends nearly all its time
blocked in `socket.recvfrom()`, which releases the GIL while it waits. It is doing
waiting work, not CPU work competing with your loop.

**KISS-ICP is CPU-only here.** There is no CUDA path, so the Orin's GPU sits idle
during registration; runtime is governed by `voxel_size` and frame rate.

### Network sizing

**Scan datagrams are 2904 bytes**, above the 1500-byte Ethernet MTU, so IP
fragments them into three. On loopback (MTU 65536) this never matters. Over a real
network, losing **any one fragment** discards the whole scan. If you move the
publisher to a separate machine and see scan loss, this is the first suspect —
check `l1-monitor` for a `dataSize` warning and consider a jumbo-frame MTU.

---

## Known limitations

- **Straight-line odometry reads ~5% short.** The two drives with a hard physical
  stop agree closely: 5 m reads 4.72 (−5.6%), 7 m reads 6.64 (−5.1%). It is a
  genuine multiplicative error, not a fixed offset. **Unresolved: whether the loss
  is per metre** (a true scale error, which one calibration factor would fix) **or
  per frame** (which it would not — the correction would then depend on speed).
  No scale factor is applied anywhere; the raw estimate is what you get.
- **Loop closure is blind to this.** A uniform shortfall cancels exactly around a
  symmetric loop. A straight line measures scale; a loop measures heading. They
  are not substitutes, and a good loop-closure number is not evidence of good
  scale.
- **Deskew does not replicate.** Measured better OFF in a room and on a loop,
  better ON in a bare hallway. It is defaulted OFF because that is where the first
  two measurements pointed — treat it as a coin the evidence has not landed on,
  and run both ways.
- **The IMU→LiDAR extrinsic is assumed to be identity.** The accumulator applies
  the IMU quaternion directly to point coordinates, which is only exactly right
  if the IMU axes and point cloud axes coincide inside the sensor. This has never
  been measured. Symptom if an offset does exist: the accumulated floor plane
  comes out consistently tilted while the robot is level.
- **Yaw is unobservable.** The L1's IMU is 6-axis, so it has no heading
  reference and yaw drifts (~1.9°/min measured). Roll and pitch are
  gravity-referenced and do not.
- **`drop_zero_returns` is on by default** in the accumulator, on the reasoning
  that a return at exactly (0,0,0) is the sensor origin and therefore never real
  geometry. Whether the L1 emits them at all is unverified; the filter is
  harmless either way.
- The Open3D window path is **not covered by tests** — it needs a display. Every
  non-GUI path is. On a Jetson the GUI additionally needs full OpenGL, which is
  not available over a plain SSH session.

---

## Development

```bash
pip install -e ".[dev,slam]"
pytest -q
ruff check .
```

**`ruff check .`, not `ruff check src tests`.** CI lints the whole tree,
including `examples/`. Linting the narrower pair passes locally and then fails in
CI.

**Install `[dev,slam]`, not bare `pytest`.** Two test dependencies are load-bearing
and both fail *silently* by skipping:

- **scipy** — without it `test_matches_scipy` skips, and the quaternion convention
  everything downstream depends on goes unverified.
- **kiss-icp** — without it the odometry tests skip via `pytest.importorskip`, and
  the entire SLAM pipeline goes untested.

Skips are reported in the pytest summary. A run that should be all-pass and shows
`N skipped` means that many checks did not execute. **CI currently installs only
`.[dev]`**, so the kiss-icp tests are skipping there — worth fixing in `ci.yml`
before trusting a green badge on odometry changes.

On aarch64, set `PIP_CONSTRAINT` before installing (see [Install](#install)) or
the dev install can pull numpy 2 and break the `[viz]` extra.

CI runs ruff on 3.12, pytest on 3.10/3.11/3.12, and a `python -m build` +
`twine check` packaging job. The Jetson deployment constraints (Python 3.10,
numpy<2) are an **aarch64 deployment constraint, not a package requirement** — on
x86_64 the `[viz]` extra is unconstrained.

## License

MIT — see [LICENSE](LICENSE).
