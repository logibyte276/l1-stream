"""Push-style buffered LiDAR stream."""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from .protocol import LidarIMU, LidarScan
from .receiver import LidarUDPReceiver
from .ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

__all__ = ["LidarStream"]

#: Nominal sensor rates for the L1, used only to size default buffers.
SCAN_RATE_HZ = 180.0
IMU_RATE_HZ = 250.0


class LidarStream:
    """Drains the UDP socket on a background thread into two ring buffers.

    ``.scans`` holds :class:`~l1_stream.protocol.LidarScan`, ``.imu`` holds
    :class:`~l1_stream.protocol.LidarIMU`. Your main loop reads whichever it
    wants, whenever it wants::

        with LidarStream() as lidar:
            scan = lidar.scans.latest()       # newest scan, or None
            recent = lidar.imu.latest_n(10)   # last 10 IMU samples
            everything = lidar.scans.drain()  # all buffered scans, and clear

    **Sizing the buffers.** Think in seconds of history, not packets:
    ``maxlen ~= sensor_rate_hz * seconds_you_care_about``. Use
    :meth:`for_history` to do that arithmetic for you. The two buffers should
    cover a *comparable* time span -- if the IMU buffer spans less wall-clock
    time than the scan buffer, timestamp matching between them will start
    failing for the oldest scans in a way that is very annoying to debug.

    **On threading and the GIL.** A thread genuinely helps here even though
    Python has a GIL, because the reader thread spends nearly all its time
    blocked inside ``socket.recvfrom()`` waiting on the operating system, and
    that call releases the GIL while it waits. The thread is not doing CPU work
    that competes with your main loop; it is doing waiting work.
    """

    def __init__(
        self,
        port: int = 12345,
        ip: str = "0.0.0.0",
        scan_maxlen: int = 360,
        imu_maxlen: int = 500,
        timeout: float = 1.0,
        buffer_size: int = 65536,
        so_rcvbuf: Optional[int] = None,
    ):
        kwargs = dict(port=port, ip=ip, timeout=timeout, buffer_size=buffer_size)
        if so_rcvbuf is not None:
            kwargs["so_rcvbuf"] = so_rcvbuf
        self._receiver = LidarUDPReceiver(**kwargs)  # type: ignore[arg-type]

        self.scans: RingBuffer = RingBuffer(scan_maxlen)
        self.imu: RingBuffer = RingBuffer(imu_maxlen)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._unknown_messages = 0

    @classmethod
    def for_history(cls, seconds: float = 2.0, **kwargs) -> "LidarStream":
        """Build a stream whose buffers hold roughly ``seconds`` of data.

        Both buffers end up covering the same wall-clock window, which is the
        property the timestamp matching in :mod:`l1_stream.rotation` depends on.
        """
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        kwargs.setdefault("scan_maxlen", max(1, int(SCAN_RATE_HZ * seconds)))
        kwargs.setdefault("imu_maxlen", max(1, int(IMU_RATE_HZ * seconds)))
        return cls(**kwargs)

    # --- lifecycle ---

    def start(self) -> "LidarStream":
        if self.is_running:
            logger.warning("LidarStream already running; start() ignored.")
            return self

        self._receiver.open()
        self._stop_event.clear()
        # daemon=True so a forgotten stop() cannot hang interpreter shutdown.
        self._thread = threading.Thread(
            target=self._run, name="lidar-udp-reader", daemon=True
        )
        self._thread.start()
        logger.info("LidarStream reader thread started.")
        return self

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            # The thread can sit in recvfrom() for up to `timeout` seconds, so
            # give it at least that long to notice the stop flag.
            thread.join(timeout=max(join_timeout, self._receiver.timeout + 0.5))
            if thread.is_alive():  # pragma: no cover - timing dependent
                logger.warning("Reader thread did not exit within %.1fs.", join_timeout)
            self._thread = None
        self._receiver.close()
        logger.info("LidarStream stopped.")

    def __enter__(self) -> "LidarStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- non-blocking accessors ---
    #
    # These never touch the network and never wait on the reader thread. They
    # return SNAPSHOTS, not live references: `scan = lidar.latest_scan` copies
    # the current value once and `scan` will not update on its own afterwards.
    # Read the property fresh at the top of each loop iteration.

    @property
    def latest_scan(self) -> Optional[LidarScan]:
        """Most recent scan, or ``None``. Non-blocking."""
        return self.scans.latest()

    @property
    def latest_imu(self) -> Optional[LidarIMU]:
        """Most recent IMU sample, or ``None``. Non-blocking."""
        return self.imu.latest()

    def recent_scans(self, n: int, allow_partial: bool = True) -> List[LidarScan]:
        """The ``n`` most recent scans, oldest-first. Non-blocking."""
        return self.scans.latest_n(n, allow_partial=allow_partial)

    def recent_imu(self, n: int, allow_partial: bool = True) -> List[LidarIMU]:
        """The ``n`` most recent IMU samples, oldest-first. Non-blocking."""
        return self.imu.latest_n(n, allow_partial=allow_partial)

    @property
    def scan_capacity(self) -> int:
        return self.scans.maxlen

    @property
    def imu_capacity(self) -> int:
        return self.imu.maxlen

    # --- reader thread ---

    def _run(self) -> None:
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                msg = self._receiver.receive_once()
                consecutive_errors = 0
            except OSError as exc:
                # Socket closed out from under us during shutdown is expected.
                if not self._stop_event.is_set():
                    logger.error("Socket error in reader thread: %s", exc)
                break
            except Exception as exc:  # pragma: no cover - defensive
                # Catch-all so one weird packet cannot silently kill the thread
                # and leave the buffers frozen with no visible error. But bail
                # out if it keeps happening, rather than spinning forever at
                # 100% CPU logging the same failure.
                consecutive_errors += 1
                logger.exception("Unexpected error in reader thread: %s", exc)
                if consecutive_errors >= 100:
                    logger.error("Too many consecutive reader errors; stopping thread.")
                    break
                continue

            if msg is None:
                continue  # timeout or malformed; re-check the stop flag

            if isinstance(msg, LidarScan):
                self.scans.append(msg)
            elif isinstance(msg, LidarIMU):
                self.imu.append(msg)
            else:  # pragma: no cover - parse_packet only returns these two
                self._unknown_messages += 1

    # --- convenience ---

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until at least one scan AND one IMU sample have arrived.

        Returns ``False`` on timeout. Call this at startup so your first loop
        iteration is not full of ``None``. **Check the return value** -- ignoring
        it turns "the publisher isn't running" into a silently empty window or an
        empty cloud, which is a much harder thing to debug than an early exit.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if len(self.scans) > 0 and len(self.imu) > 0:
                return True
            time.sleep(0.02)
        return False

    def stats(self) -> dict:
        """Counters for debugging throughput and drops."""
        return {
            "running": self.is_running,
            "scans_held": len(self.scans),
            "scans_total": self.scans.total_received,
            "scans_dropped": self.scans.total_dropped,
            "imu_held": len(self.imu),
            "imu_total": self.imu.total_received,
            "imu_dropped": self.imu.total_dropped,
        }
