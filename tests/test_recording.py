"""Tests for the raw-datagram recorder and replayer. No sockets involved."""

import struct

import numpy as np
import pytest

from l1_stream.protocol import (
    FULL_SCAN_DATAGRAM_SIZE,
    POINT_DTYPE,
    LidarIMU,
    LidarScan,
    pack_imu_packet,
    pack_scan_packet,
)
from l1_stream.recording import MAGIC, DatagramRecorder, Replayer


def make_points(n, start=0.0):
    pts = np.zeros(n, dtype=POINT_DTYPE)
    pts["x"] = np.arange(n, dtype=np.float32) + start + 1.0
    pts["y"] = 1.0
    pts["z"] = np.arange(n, dtype=np.float32) * 0.1
    pts["time"] = np.linspace(0.0, 0.005, n, dtype=np.float32)
    return pts


def write_recording(path, datagrams):
    with DatagramRecorder(path) as rec:
        for t, data in datagrams:
            rec.write(t, data)
    return rec


# --- roundtrip --------------------------------------------------------------

def test_roundtrip_is_byte_exact(tmp_path):
    path = tmp_path / "a.l1raw"
    grams = [
        (100.0, pack_imu_packet(1.0, 0, (0, 0, 0, 1))),
        (100.1, pack_scan_packet(1.0, 0, make_points(37))),
        (100.2, b"\x00" * 3),                      # junk survives too
    ]
    write_recording(path, grams)

    back = list(Replayer(path).iter_datagrams())
    assert len(back) == 3
    for (t_in, d_in), (t_out, d_out) in zip(grams, back, strict=True):
        assert t_out == pytest.approx(t_in)
        assert d_out == d_in


def test_non_zero_padding_survives_a_roundtrip(tmp_path):
    """Re-packing parsed objects would launder this; recording bytes does not.

    The publisher always transmits 120 point slots and the padding is not
    guaranteed to be zeroed. A recording has to preserve that, because it is
    exactly the condition that catches a receiver which trusts array length
    instead of validPointsNum.
    """
    path = tmp_path / "pad.l1raw"
    good = pack_scan_packet(1.0, 0, make_points(5))
    dirty = bytearray(good)
    dirty[-24:] = b"\xAB" * 24                      # garbage in the last slot
    write_recording(path, [(0.0, bytes(dirty))])

    (_t, data), = Replayer(path).iter_datagrams()
    assert data == bytes(dirty)
    assert len(data) == FULL_SCAN_DATAGRAM_SIZE
    # ... and the parser still reports only the valid points.
    msgs = list(Replayer(path).iter_messages())
    assert msgs[0][1].valid_points_num == 5


def test_recorder_counts(tmp_path):
    path = tmp_path / "c.l1raw"
    rec = write_recording(path, [(0.0, b"abc"), (1.0, b"defg")])
    assert rec.records == 2
    assert rec.bytes_written == 7


# --- damaged files ----------------------------------------------------------

def test_truncated_trailing_record_is_dropped_not_fatal(tmp_path):
    path = tmp_path / "cut.l1raw"
    write_recording(path, [(0.0, b"aaaa"), (1.0, b"bbbb"), (2.0, b"cccc")])
    whole = path.read_bytes()
    path.write_bytes(whole[:-3])                     # a crash mid-write

    rep = Replayer(path)
    got = list(rep.iter_datagrams())
    assert [d for _t, d in got] == [b"aaaa", b"bbbb"]
    assert rep.truncated is True


def test_truncated_header_is_also_survivable(tmp_path):
    path = tmp_path / "cut2.l1raw"
    write_recording(path, [(0.0, b"aaaa")])
    path.write_bytes(path.read_bytes() + b"\x01\x02")   # partial next header

    rep = Replayer(path)
    assert [d for _t, d in rep.iter_datagrams()] == [b"aaaa"]
    assert rep.truncated is True


def test_clean_file_is_not_flagged_truncated(tmp_path):
    path = tmp_path / "ok.l1raw"
    write_recording(path, [(0.0, b"aaaa")])
    rep = Replayer(path)
    list(rep.iter_datagrams())
    assert rep.truncated is False


def test_wrong_magic_raises(tmp_path):
    path = tmp_path / "bad.l1raw"
    path.write_bytes(b"NOPE1234" + b"\x00" * 32)
    with pytest.raises(ValueError, match="not an L1RAW recording"):
        list(Replayer(path).iter_datagrams())


def test_corrupt_length_field_is_refused_not_allocated(tmp_path):
    path = tmp_path / "huge.l1raw"
    path.write_bytes(MAGIC + struct.pack("=dI", 0.0, 2**31) + b"\x00" * 8)
    with pytest.raises(ValueError, match="Corrupt record length"):
        list(Replayer(path).iter_datagrams())


# --- parsed and batched views ----------------------------------------------

def test_iter_messages_skips_unparseable_datagrams(tmp_path):
    path = tmp_path / "m.l1raw"
    write_recording(path, [
        (0.0, pack_imu_packet(1.0, 0, (0, 0, 0, 1))),
        (0.1, b"\x00\x01"),                                  # too short
        (0.2, struct.pack("=II", 999, 4) + b"\x00" * 4),     # unknown type
        (0.3, pack_scan_packet(1.0, 0, make_points(4))),
    ])
    msgs = [m for _t, m in Replayer(path).iter_messages()]
    assert len(msgs) == 2
    assert isinstance(msgs[0], LidarIMU)
    assert isinstance(msgs[1], LidarScan)


def test_iter_batches_mirrors_the_live_loop_shape(tmp_path):
    path = tmp_path / "b.l1raw"
    grams = []
    for i in range(100):                       # 0.5 s at 200 Hz
        t = 10.0 + i * 0.005
        grams.append((t, pack_imu_packet(t, i, (0, 0, 0, 1))))
        grams.append((t, pack_scan_packet(t, i, make_points(20))))
    write_recording(path, grams)

    batches = list(Replayer(path).iter_batches(period=0.1))
    assert len(batches) >= 5
    for scans, imu in batches:
        assert all(isinstance(s, LidarScan) for s in scans)
        assert all(isinstance(s, LidarIMU) for s in imu)
    # Every scan is handed over exactly once, like drain().
    assert sum(len(s) for s, _ in batches) == 100
    # The IMU window is a trailing buffer, like latest_n() -- not drained.
    assert len(batches[-1][1]) > len(batches[-1][0])


def test_iter_batches_feeds_the_assembler_end_to_end(tmp_path):
    from l1_stream.frames import FrameAssembler

    path = tmp_path / "e2e.l1raw"
    grams = []
    for i in range(400):                       # 2 s at 200 Hz
        t = 10.0 + i * 0.005
        grams.append((t, pack_imu_packet(t, i, (0, 0, 0, 1))))
        grams.append((t, pack_scan_packet(t, i, make_points(20))))
    write_recording(path, grams)

    a = FrameAssembler(frame_duration=0.2, min_points=1)
    frames = []
    for scans, imu in Replayer(path).iter_batches(period=0.05):
        frames += a.add(scans, imu)

    assert len(frames) >= 8
    assert a.scans_unmatched == 0
    for f in frames:
        assert f.span == pytest.approx(0.2, abs=0.01)
        assert f.timestamps.max() == pytest.approx(1.0)
