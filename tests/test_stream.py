"""End-to-end tests over a real loopback UDP socket. Still no hardware needed."""

import socket
import time

import numpy as np

from l1_stream import LidarIMU, LidarScan, LidarStream, LidarUDPReceiver
from l1_stream.protocol import POINT_DTYPE, pack_imu_packet, pack_scan_packet

PORT = 45671  # unlikely to collide with anything real


def send(port, payload):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, ("127.0.0.1", port))
    finally:
        sock.close()


def some_points(n=8):
    pts = np.zeros(n, dtype=POINT_DTYPE)
    pts["x"] = np.arange(1, n + 1, dtype=np.float32)
    pts["z"] = 1.0
    return pts


def test_receiver_roundtrip_over_loopback():
    with LidarUDPReceiver(port=PORT, timeout=1.0) as rx:
        send(PORT, pack_scan_packet(1.0, 3, some_points()))
        msg = rx.receive_once()
    assert isinstance(msg, LidarScan)
    assert msg.id == 3


def test_receiver_returns_none_on_timeout():
    with LidarUDPReceiver(port=PORT + 1, timeout=0.15) as rx:
        start = time.monotonic()
        assert rx.receive_once() is None
        assert time.monotonic() - start >= 0.1


def test_double_open_raises_instead_of_leaking_the_fd():
    rx = LidarUDPReceiver(port=PORT + 2)
    rx.open()
    try:
        rx.open()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on second open()")
    finally:
        rx.close()
    rx.close()  # idempotent


def test_receive_before_open_raises():
    rx = LidarUDPReceiver(port=PORT + 3)
    try:
        rx.receive_once()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_buffer_size_smaller_than_a_scan_is_rejected():
    # Silently truncating every single scan is far worse than failing at
    # construction time, so this is checked up front.
    try:
        LidarUDPReceiver(port=PORT + 4, buffer_size=1500)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_stream_sorts_messages_into_the_right_buffers():
    with LidarStream(port=PORT + 5, scan_maxlen=10, imu_maxlen=10, timeout=0.2) as lidar:
        for i in range(4):
            send(PORT + 5, pack_scan_packet(1.0 + i, i, some_points()))
            send(PORT + 5, pack_imu_packet(1.0 + i, i, (0, 0, 0, 1)))
        assert lidar.wait_until_ready(timeout=2.0)
        time.sleep(0.2)

        assert isinstance(lidar.latest_scan, LidarScan)
        assert isinstance(lidar.latest_imu, LidarIMU)
        assert lidar.scans.total_received == 4
        assert lidar.imu.total_received == 4
        assert lidar.stats()["running"] is True


def test_stream_survives_a_garbage_packet():
    with LidarStream(port=PORT + 6, scan_maxlen=5, imu_maxlen=5, timeout=0.2) as lidar:
        send(PORT + 6, b"\xff" * 3)              # undersized
        send(PORT + 6, b"\x00" * 64)             # msgType 0, unknown
        send(PORT + 6, pack_scan_packet(1.0, 1, some_points()))
        send(PORT + 6, pack_imu_packet(1.0, 1, (0, 0, 0, 1)))
        assert lidar.wait_until_ready(timeout=2.0)
        assert lidar.is_running  # a bad packet must not kill the reader thread


def test_wait_until_ready_reports_failure_rather_than_hanging():
    with LidarStream(port=PORT + 7, timeout=0.2) as lidar:
        assert lidar.wait_until_ready(timeout=0.4) is False


def test_stop_is_idempotent_and_start_is_guarded():
    lidar = LidarStream(port=PORT + 8, timeout=0.2)
    lidar.start()
    lidar.start()  # warns, does not spawn a second thread
    assert lidar.is_running
    lidar.stop()
    assert not lidar.is_running
    lidar.stop()


def test_for_history_sizes_both_buffers_to_the_same_window():
    lidar = LidarStream.for_history(2.0, port=PORT + 9)
    assert lidar.scan_capacity == 360   # 180 Hz x 2 s
    assert lidar.imu_capacity == 500    # 250 Hz x 2 s


def test_for_history_rejects_nonpositive():
    try:
        LidarStream.for_history(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
