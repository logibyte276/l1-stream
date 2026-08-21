"""Blocking, one-packet-at-a-time UDP receiver.

This is the simple half of the API. If your consumer loop can be slower than
the sensor at any point, use :class:`l1_stream.stream.LidarStream` instead --
it drains the socket on a background thread so the kernel never has to drop
packets on your behalf.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator

from .protocol import FULL_SCAN_DATAGRAM_SIZE, LidarMessage, parse_packet

logger = logging.getLogger(__name__)

__all__ = ["LidarUDPReceiver"]

#: Default OS-level receive buffer request, in bytes. ~1 MB is roughly two
#: seconds of scan traffic at 180 Hz, which is enough slack to ride out a
#: scheduler hiccup without the kernel silently discarding datagrams.
DEFAULT_SO_RCVBUF = 1 << 20


class LidarUDPReceiver:
    """Owns a UDP socket and hands back parsed messages one at a time.

    Pure plumbing: no plotting, no printing, no threads.

    Example::

        with LidarUDPReceiver(port=12345) as receiver:
            for message in receiver.stream():
                ...
    """

    def __init__(
        self,
        port: int = 12345,
        ip: str = "0.0.0.0",
        timeout: float = 1.0,
        buffer_size: int = 65536,
        so_rcvbuf: int | None = DEFAULT_SO_RCVBUF,
    ):
        if buffer_size < FULL_SCAN_DATAGRAM_SIZE:
            raise ValueError(
                f"buffer_size={buffer_size} is smaller than a full scan datagram "
                f"({FULL_SCAN_DATAGRAM_SIZE} bytes); every scan would be truncated."
            )
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.so_rcvbuf = so_rcvbuf
        self._sock: socket.socket | None = None

    # --- lifecycle ---

    def open(self) -> None:
        """Bind the socket. Raises ``RuntimeError`` if already open.

        Failing loudly here rather than rebinding matters: silently replacing
        ``self._sock`` would leak the previous file descriptor, and on Linux the
        orphaned socket keeps consuming datagrams that nothing ever reads --
        which looks exactly like packet loss.
        """
        if self._sock is not None:
            raise RuntimeError("Socket already open -- call close() first.")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.so_rcvbuf:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.so_rcvbuf)
                except OSError as exc:  # pragma: no cover - platform dependent
                    logger.warning("Could not set SO_RCVBUF=%d: %s", self.so_rcvbuf, exc)
            sock.bind((self.ip, self.port))
            sock.settimeout(self.timeout)
        except BaseException:
            sock.close()
            raise

        self._sock = sock
        actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        logger.info(
            "Listening for LiDAR UDP data on %s:%d (SO_RCVBUF=%d bytes)",
            self.ip, self.port, actual,
        )

    def close(self) -> None:
        """Close the socket. Safe to call more than once."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            logger.info("Socket closed.")

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def __enter__(self) -> LidarUDPReceiver:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- reading ---

    def receive_once(self) -> LidarMessage | None:
        """Block up to ``timeout`` seconds for one datagram.

        Returns the parsed message, or ``None`` on timeout / unknown /
        malformed packet.
        """
        sock = self._sock
        if sock is None:
            raise RuntimeError("Socket not open -- call open() or use a 'with' block.")

        try:
            data, _addr = sock.recvfrom(self.buffer_size)
        except TimeoutError:
            return None

        return parse_packet(data)

    def stream(self) -> Iterator[LidarMessage]:
        """Yield parsed messages forever, skipping timeouts."""
        while True:
            msg = self.receive_once()
            if msg is not None:
                yield msg
