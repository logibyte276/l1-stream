# l1-stream

A Python client for a **Unitree L1 LiDAR** streamed over UDP: packet parsing,
thread-safe buffering, IMU-based orientation compensation, and an optional live
Open3D view.

Built for the awkward part of working with this sensor — a single scan packet
carries at most 120 points, arriving at ~180 Hz on a separate datagram from the
~250 Hz IMU stream, so anything useful means buffering both and matching them by
timestamp without losing packets when your consumer loop stutters.

- **Pure Python, one hard dependency** — numpy. Open3D is optional, so the
  package installs in seconds on a headless Jetson.
- **No hardware needed to develop or test.** `pack_imu_packet` /
  `pack_scan_packet` build byte-identical datagrams in memory, so the whole
  pipeline is testable offline. 76 tests, none of which touch a sensor.
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
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="mcu"
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
git clone https://github.com/YOUR_USERNAME/l1-stream.git
cd l1-stream

pip install -e .            # core: numpy only
pip install -e ".[viz]"     # + Open3D for the live viewer
pip install -e ".[dev]"     # + pytest, scipy, ruff
```

On Jetson/ARM64, Open3D often needs a prebuilt wheel from the
[Open3D releases page](https://github.com/isl-org/Open3D/releases) rather than PyPI.

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

See `examples/` for four runnable scripts, including `04_offline_replay.py`
which needs no hardware at all.

---

## Layout

| Module | Responsibility |
|---|---|
| `protocol.py` | Wire format, dataclasses, parse **and pack** functions. Pure — no I/O, no threads, no state. |
| `ring_buffer.py` | `RingBuffer`: thread-safe, bounded, drop-oldest. |
| `receiver.py` | `LidarUDPReceiver`: blocking, one packet at a time. |
| `stream.py` | `LidarStream`: background reader thread → two ring buffers. |
| `rotation.py` | Quaternion math + `RotatedScanAccumulator`. |
| `visualizer.py` | `LiveVisualizer`. Lazily imports Open3D. |
| `cli.py` | `l1-monitor`, `l1-visualize`. |

The split exists so the parts you can test without hardware are separated from
the parts you can't. `protocol.py`, `ring_buffer.py`, and `rotation.py` have no
I/O at all and are fully covered by tests.

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

### Rotation

`rotate_points` uses the vector form `v + 2w(q×v) + 2q×(q×v)`, which is ~15 flops
per point versus building a 3×3 matrix, worth it when you rotate one small scan at
a time thousands of times a second. It is only a *rotation* for a unit quaternion —
off-norm input scales every point by |q|² — so input is normalised, with an
identity fallback for the degenerate all-zero case a sensor can emit during warm-up.

Each scan is rotated **once** on ingestion, not once per displayed frame, and the
concatenated output is cached until something changes.

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

### Network sizing

**Scan datagrams are 2904 bytes**, above the 1500-byte Ethernet MTU, so IP
fragments them into three. On loopback (MTU 65536) this never matters. Over a real
network, losing **any one fragment** discards the whole scan. If you move the
publisher to a separate machine and see scan loss, this is the first suspect —
check `l1-monitor` for a `dataSize` warning and consider a jumbo-frame MTU.

---

## Known limitations

- **The IMU→LiDAR extrinsic is assumed to be identity.** The accumulator applies
  the IMU quaternion directly to point coordinates, which is only exactly right
  if the IMU axes and point cloud axes coincide inside the sensor. This has not
  been verified against Unitree documentation. Symptom if an offset does exist:
  the accumulated floor plane comes out consistently tilted while the robot is
  level.
- **Rotation only, not translation.** Drive forward while accumulating and the
  cloud still smears along the direction of travel. Removing that needs pose
  estimation, not an IMU orientation.
- **`drop_zero_returns` is on by default** in the accumulator, on the reasoning
  that a return at exactly (0,0,0) is the sensor origin and therefore never real
  geometry. Whether the L1 emits them at all is unverified; the filter is
  harmless either way.
- The Open3D window path is **not covered by tests** — it needs a display. Every
  non-GUI path is.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

CI runs the suite on Python 3.9 / 3.11 / 3.12 (`.github/workflows/ci.yml`).

## License

MIT — see [LICENSE](LICENSE).
