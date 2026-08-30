"""Record and replay the raw UDP wire, byte for byte.

Why record the *wire* and not parsed objects: a recording is only useful for
debugging if it survives a change to the parser. If you pickle ``LidarScan``
objects you have frozen today's interpretation of the bytes, and the first
time you fix a parsing bug every old recording becomes a record of the bug.
Storing datagrams keeps the recording authoritative -- ``parse_packet`` runs
at replay time, so a fixed parser retroactively fixes every old file.

It also preserves exactly the cases you most want to study: truncated
datagrams from MTU fragmentation loss, non-zero padding slots, and unknown
message types. Re-packing parsed objects would quietly launder all three.

File format (little-endian, no compression):

    magic   b"L1RAW\\x00"          8 bytes, includes a version byte
    record  [float64 recv_time][uint32 nbytes][payload ...]   repeated

``recv_time`` is the receiving host's wall clock (``time.time()``) at the
moment the datagram arrived. It exists for playback pacing and for measuring
transport latency. It is **not** the sensor timestamp -- use the ``stamp``
field inside the payload for anything that needs capture time, because the
publisher's clock and the receiver's clock are not the same clock.

Typical use::

    # capture
    with DatagramRecorder("drive_01.l1raw") as rec:
        for recv_time, data in raw_datagrams(port=12345, duration=60.0):
            rec.write(recv_time, data)

    # replay, in the same shape the live loop sees
    for scans, imu in Replayer("drive_01.l1raw").iter_batches(period=0.05):
        frames = assembler.add(scans, imu)
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from .protocol import LidarIMU, LidarScan, parse_packet

logger = logging.getLogger(__name__)

__all__ = ["MAGIC", "DatagramRecorder", "Replayer", "raw_datagrams"]

MAGIC = b"L1RAW\x00\x00\x01"          # last byte is the format version
_RECORD = struct.Struct("=dI")        # recv_time, nbytes

#: Refuse to allocate more than this for one record. A corrupt length field
#: would otherwise try to read gigabytes; the largest real datagram is under
#: 3 KB, so this is generous by three orders of magnitude.
_MAX_RECORD_BYTES = 1 << 20


def raw_datagrams(
    port: int = 12345,
    ip: str = "0.0.0.0",
    duration: float | None = None,
    timeout: float = 1.0,
    buffer_size: int = 65536,
    so_rcvbuf: int | None = 1 << 20,
) -> Iterator[tuple[float, bytes]]:
    """Yield ``(recv_time, datagram_bytes)`` straight off the socket.

    Deliberately does not use :class:`~l1_stream.receiver.LidarUDPReceiver`,
    which hands back parsed messages -- recording needs the bytes before
    anything has interpreted them.

    Stops after ``duration`` seconds, or on KeyboardInterrupt if it is None.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if so_rcvbuf:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, so_rcvbuf)
            except OSError:
                logger.warning("Could not raise SO_RCVBUF; expect drops under load.")
        sock.bind((ip, port))
        sock.settimeout(timeout)
        deadline = None if duration is None else time.monotonic() + duration
        while deadline is None or time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(buffer_size)
            except TimeoutError:   # socket.timeout is an alias of this on 3.10+
                continue
            yield time.time(), data
    finally:
        sock.close()


class DatagramRecorder:
    """Append raw datagrams to a file. Use as a context manager."""

    def __init__(self, path: str | Path, buffering: int = 1 << 20):
        self.path = Path(path)
        self._buffering = buffering
        self._fh = None
        self.records = 0
        self.bytes_written = 0

    def __enter__(self) -> DatagramRecorder:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        if self._fh is not None:
            raise RuntimeError("Recorder already open -- call close() first.")
        self._fh = self.path.open("wb", buffering=self._buffering)
        self._fh.write(MAGIC)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            logger.info(
                "Recorded %d datagrams (%.1f MB) to %s",
                self.records, self.bytes_written / 1e6, self.path,
            )

    def write(self, recv_time: float, data: bytes) -> None:
        if self._fh is None:
            raise RuntimeError("Recorder is not open.")
        self._fh.write(_RECORD.pack(float(recv_time), len(data)))
        self._fh.write(data)
        self.records += 1
        self.bytes_written += len(data)


class Replayer:
    """Read a recording back as datagrams, messages, or live-shaped batches."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.truncated = False

    # --- level 1: bytes ---

    def iter_datagrams(self) -> Iterator[tuple[float, bytes]]:
        """Yield ``(recv_time, bytes)`` in recorded order.

        A file cut short by a crash or a full disk is common and is not an
        error worth losing the rest of the drive over: the trailing partial
        record is dropped and :attr:`truncated` is set to True.
        """
        with self.path.open("rb") as fh:
            magic = fh.read(len(MAGIC))
            if magic[:5] != MAGIC[:5]:
                raise ValueError(f"{self.path} is not an L1RAW recording.")
            if magic != MAGIC:
                logger.warning("Recording format version differs from this reader's.")
            while True:
                head = fh.read(_RECORD.size)
                if len(head) < _RECORD.size:
                    self.truncated = len(head) > 0
                    return
                recv_time, nbytes = _RECORD.unpack(head)
                if nbytes > _MAX_RECORD_BYTES:
                    raise ValueError(f"Corrupt record length {nbytes} in {self.path}.")
                data = fh.read(nbytes)
                if len(data) < nbytes:
                    self.truncated = True
                    return
                yield recv_time, data

    # --- level 2: parsed messages ---

    def iter_messages(self, pace: bool = False) -> Iterator[tuple[float, object]]:
        """Yield ``(recv_time, message)``, skipping datagrams that fail to parse.

        ``pace=True`` sleeps to reproduce the original arrival timing, which is
        what you want when driving a visualiser. Leave it False for tuning
        sweeps -- a 60 s drive then replays in a second or two.
        """
        t0_file = t0_wall = None
        for recv_time, data in self.iter_datagrams():
            if pace:
                if t0_file is None:
                    t0_file, t0_wall = recv_time, time.monotonic()
                lag = (recv_time - t0_file) - (time.monotonic() - t0_wall)
                if lag > 0:
                    time.sleep(lag)
            message = parse_packet(data)
            if message is not None:
                yield recv_time, message

    # --- level 3: the shape the live loop actually sees ---

    def iter_batches(
        self, period: float = 0.05, imu_maxlen: int = 500, pace: bool = False
    ) -> Iterator[tuple[list, list]]:
        """Yield ``(scans, imu_window)`` batches, mirroring the live consumer.

        Online you call ``lidar.scans.drain()`` and ``lidar.imu.latest_n(n)``.
        This reproduces both: scans accumulate for ``period`` seconds and are
        handed over once, while the IMU window is a rolling buffer of the last
        ``imu_maxlen`` samples -- the same trailing-window semantics as the
        ring buffer. Getting this shape right is the whole point: code tuned
        on replay then behaves identically on hardware.
        """
        scans: list[LidarScan] = []
        imu: deque[LidarIMU] = deque(maxlen=imu_maxlen)
        window_start = None

        for recv_time, message in self.iter_messages(pace=pace):
            if isinstance(message, LidarIMU):
                imu.append(message)
            elif isinstance(message, LidarScan):
                scans.append(message)
            else:
                continue

            if window_start is None:
                window_start = recv_time
            elif recv_time - window_start >= period:
                yield scans, list(imu)
                scans = []
                window_start = recv_time

        if scans or imu:
            yield scans, list(imu)
