"""Protocol tests. No hardware needed -- packets are built in memory."""

import struct

import numpy as np

from l1_stream import protocol as P


def make_points(n, start=0.0):
    pts = np.zeros(n, dtype=P.POINT_DTYPE)
    pts["x"] = np.arange(n, dtype=np.float32) + start
    pts["y"] = np.arange(n, dtype=np.float32) * 2.0
    pts["z"] = np.arange(n, dtype=np.float32) * 3.0
    pts["intensity"] = 10.0
    pts["time"] = np.linspace(0.0, 0.005, n, dtype=np.float32)
    pts["ring"] = np.arange(n, dtype=np.uint32) % 4
    return pts


def test_point_dtype_is_24_bytes():
    # Must match `PointUnitree` in unitree_lidar_sdk.h: 5 floats + 1 uint32,
    # no padding. If this ever changes, every scan parses as garbage.
    assert P.POINT_DTYPE.itemsize == 24


def test_struct_sizes_match_c_layout():
    assert P.HEADER_SIZE == 8
    assert P.SCAN_HEADER_SIZE == 16   # double + 2 uint32, natural alignment
    assert P.IMU_PAYLOAD_SIZE == 52   # double + uint32 + 10 floats


def test_imu_roundtrip():
    packet = P.pack_imu_packet(
        stamp=123.456, imu_id=42,
        quaternion=(0.1, 0.2, 0.3, 0.9273618),
        angular_velocity=(1.0, -2.0, 3.0),
        linear_acceleration=(0.0, 0.0, 9.81),
    )
    msg = P.parse_packet(packet)
    assert isinstance(msg, P.LidarIMU)
    assert msg.id == 42
    assert abs(msg.stamp - 123.456) < 1e-9
    assert np.allclose(msg.quaternion, (0.1, 0.2, 0.3, 0.9273618), atol=1e-6)
    assert np.allclose(msg.angular_velocity, (1.0, -2.0, 3.0), atol=1e-6)
    assert np.allclose(msg.linear_acceleration, (0.0, 0.0, 9.81), atol=1e-6)


def test_imu_quaternion_is_xyzw_not_wxyz():
    # Guards the single most damaging silent bug in this codebase: the SDK
    # header says [x,y,z,w], scalar LAST. A w-first reading produces a
    # perfectly valid but wrong rotation that is very hard to spot by eye.
    packet = P.pack_imu_packet(0.0, 0, quaternion=(0.0, 0.0, 0.0, 1.0))
    msg = P.parse_packet(packet)
    assert msg.quaternion[3] == 1.0          # w is last
    assert msg.quaternion_wxyz()[0] == 1.0   # ... and first after conversion


def test_scan_roundtrip():
    pts = make_points(37)
    packet = P.pack_scan_packet(stamp=9.5, scan_id=7, points=pts)
    msg = P.parse_packet(packet)
    assert isinstance(msg, P.LidarScan)
    assert msg.id == 7
    assert msg.valid_points_num == 37
    assert len(msg) == 37
    assert np.allclose(msg.points["x"], pts["x"])
    assert np.array_equal(msg.points["ring"], pts["ring"])


def test_scan_ignores_padding_slots():
    # The SDK publisher always transmits all 120 slots. Only validPointsNum of
    # them are real; the rest must not leak into the cloud.
    pts = make_points(5)
    packet = P.pack_scan_packet(0.0, 0, pts, pad_to_max=True)
    assert len(packet) == P.FULL_SCAN_DATAGRAM_SIZE
    msg = P.parse_packet(packet)
    assert msg.valid_points_num == 5
    assert len(msg.points) == 5


def test_scan_parsed_array_is_writable_and_owns_its_memory():
    # np.frombuffer without .copy() returns a read-only view that also pins the
    # whole datagram in memory. Both bite once these land in a ring buffer.
    msg = P.parse_packet(P.pack_scan_packet(0.0, 0, make_points(10)))
    assert msg.points.flags.writeable
    msg.points["x"][0] = 99.0  # must not raise


def test_truncated_scan_is_clamped_not_crashed():
    packet = P.pack_scan_packet(0.0, 0, make_points(120))
    truncated = packet[: P.HEADER_SIZE + P.SCAN_HEADER_SIZE + 24 * 9 + 7]
    msg = P.parse_packet(truncated)
    assert isinstance(msg, P.LidarScan)
    assert msg.valid_points_num == 9  # 9 whole points survived; the partial one is dropped


def test_scan_with_lying_valid_points_num_is_clamped():
    # A corrupt or spoofed header claiming 5000 points must not be believed.
    pts = make_points(3)
    body = pts.tobytes()
    payload = struct.pack("=dII", 0.0, 0, 5000) + body
    packet = struct.pack("=II", P.MSG_TYPE_SCAN, len(payload)) + payload
    msg = P.parse_packet(packet)
    assert msg.valid_points_num == 3


def test_unknown_message_type_returns_none():
    payload = b"\x00" * 16
    assert P.parse_packet(struct.pack("=II", 999, len(payload)) + payload) is None


def test_undersized_packet_returns_none():
    assert P.parse_packet(b"\x01\x02\x03") is None
    assert P.parse_packet(b"") is None


def test_truncated_imu_returns_none_not_exception():
    packet = P.pack_imu_packet(0.0, 0, (0, 0, 0, 1))
    assert P.parse_packet(packet[:20]) is None


def test_scan_xyz_is_float64_by_default():
    # Open3D's Vector3dVector and most registration libraries require doubles.
    msg = P.parse_packet(P.pack_scan_packet(0.0, 0, make_points(4)))
    assert msg.xyz().dtype == np.float64
    assert msg.xyz().shape == (4, 3)
    assert msg.xyz().flags.c_contiguous


def test_drop_zero_returns():
    pts = make_points(6)          # point 0 is exactly (0, 0, 0)
    msg = P.parse_packet(P.pack_scan_packet(0.0, 0, pts))
    assert len(msg.xyz(drop_zero_returns=False)) == 6
    assert len(msg.xyz(drop_zero_returns=True)) == 5


def test_pack_scan_rejects_oversized_input():
    try:
        P.pack_scan_packet(0.0, 0, make_points(121))
    except ValueError:
        return
    raise AssertionError("expected ValueError for >120 points")
