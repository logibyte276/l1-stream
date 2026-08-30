"""Tests for FrameAssembler. No hardware -- packets are built in memory."""

import math

import numpy as np
import pytest

from l1_stream.frames import Frame, FrameAssembler
from l1_stream.protocol import POINT_DTYPE, LidarIMU, pack_scan_packet, parse_packet

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def quat_about_z(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def make_scan(stamp, scan_id, xyz, times=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    pts = np.zeros(len(xyz), dtype=POINT_DTYPE)
    pts["x"], pts["y"], pts["z"] = xyz.T
    pts["time"] = (
        np.zeros(len(xyz), dtype=np.float32) if times is None
        else np.asarray(times, dtype=np.float32)
    )
    return parse_packet(pack_scan_packet(stamp, scan_id, pts))


def make_imu(stamp, imu_id, quaternion=IDENTITY):
    return LidarIMU(stamp=stamp, id=imu_id, quaternion=quaternion,
                    angular_velocity=(0, 0, 0), linear_acceleration=(0, 0, 0))


def marked_scan(stamp, i, n=4):
    """A scan whose points all carry `i` in x, so frames can be traced back."""
    return make_scan(stamp, i, [[float(i), 1.0, float(k)] for k in range(n)])


def drive(assembler, n_scans=100, dt=0.005, t0=1.0, quat=IDENTITY):
    """Feed n_scans one at a time, exactly as the live loop does."""
    frames = []
    for i in range(n_scans):
        t = t0 + i * dt
        frames += assembler.add([marked_scan(t, i + 1)], [make_imu(t, i, quat)])
    return frames


# --- the property the whole class exists for --------------------------------

def test_frames_do_not_overlap():
    a = FrameAssembler(frame_duration=0.2, min_points=1)
    frames = drive(a, n_scans=200)          # 1.0 s of data at 200 Hz
    assert len(frames) >= 4

    seen = set()
    for f in frames:
        ids = set(np.unique(f.points[:, 0]).tolist())
        assert not (ids & seen), "a scan appeared in two frames"
        seen |= ids


def test_every_ingested_scan_lands_in_exactly_one_frame():
    a = FrameAssembler(frame_duration=0.2, min_points=1)
    frames = drive(a, n_scans=200)
    tail = a.flush()
    if tail is not None:
        frames.append(tail)
    assert sum(f.n_scans for f in frames) == a.scans_ingested


def test_frame_span_tracks_frame_duration():
    a = FrameAssembler(frame_duration=0.2, min_points=1)
    frames = drive(a, n_scans=200, dt=0.005)
    # Half-open [t0, t0+duration) window, so the span is one scan interval short.
    for f in frames:
        assert 0.19 <= f.span <= 0.201


# --- timestamps -------------------------------------------------------------

def test_timestamps_are_normalised_to_zero_one_ending_at_one():
    a = FrameAssembler(frame_duration=0.05, min_points=1)
    frames = drive(a, n_scans=60, dt=0.005)
    assert frames
    for f in frames:
        assert f.timestamps.min() == pytest.approx(0.0)
        # KISS-ICP deskews to the END of the frame: 1.0 must be the last point,
        # so the normalised time must RISE with capture order, not fall.
        assert f.timestamps.max() == pytest.approx(1.0)
        assert np.all(np.diff(f.timestamps) >= -1e-12)
        assert f.timestamps.dtype == np.float64


def test_per_point_time_is_used_not_just_the_scan_stamp():
    a = FrameAssembler(frame_duration=1.0, min_points=1)
    a.add([make_scan(5.0, 0, [[1, 1, 0], [2, 1, 0], [3, 1, 0]],
                     times=[0.0, 0.002, 0.004])], [make_imu(5.0, 0)])
    f = a.flush()
    assert f.t_start == pytest.approx(5.0)
    assert f.t_end == pytest.approx(5.004, abs=1e-6)
    assert np.allclose(f.timestamps, [0.0, 0.5, 1.0], atol=1e-5)


def test_zero_return_mask_keeps_points_and_times_aligned():
    """The mask must hit both arrays, or every deskew lands on the wrong point."""
    a = FrameAssembler(frame_duration=1.0, min_points=1)
    a.add(
        [make_scan(
            5.0, 0,
            [[1, 1, 0], [2, 1, 0], [0, 0, 0], [4, 1, 0], [5, 1, 0]],
            times=[0.0, 0.001, 0.002, 0.003, 0.004],
        )],
        [make_imu(5.0, 0)],
    )
    f = a.flush()
    assert len(f) == 4
    assert np.allclose(f.points[:, 0], [1, 2, 4, 5])
    # times of the SURVIVORS: 0, .001, .003, .004 over a .004 span
    assert np.allclose(f.timestamps, [0.0, 0.25, 0.75, 1.0], atol=1e-5)


def test_single_instant_frame_does_not_divide_by_zero():
    a = FrameAssembler(frame_duration=1.0, min_points=1)
    a.add([make_scan(5.0, 0, [[1, 1, 0], [2, 1, 0]])], [make_imu(5.0, 0)])
    f = a.flush()
    assert np.all(np.isfinite(f.timestamps))
    assert np.allclose(f.timestamps, 1.0)


# --- IMU matching -----------------------------------------------------------

def test_scan_is_held_when_its_imu_sample_has_not_arrived():
    a = FrameAssembler(frame_duration=0.2, min_points=1)
    a.add([make_scan(2.000, 0, [[1, 1, 0]])], [make_imu(1.900, 0)])
    assert a.stats()["scans_pending"] == 1
    assert a.scans_unmatched == 0

    a.add([], [make_imu(1.900, 0), make_imu(2.001, 1)])
    assert a.stats()["scans_pending"] == 0
    assert a.scans_ingested == 1


def test_scan_too_old_to_ever_match_is_counted_not_silently_lost():
    a = FrameAssembler(frame_duration=0.2, max_time_gap=0.01, min_points=1)
    a.add([make_scan(1.000, 0, [[1, 1, 0]])],
          [make_imu(2.000, 0), make_imu(2.005, 1)])
    assert a.scans_unmatched == 1
    assert a.stats()["scans_pending"] == 0


def test_nearest_imu_sample_is_applied():
    a = FrameAssembler(frame_duration=1.0, min_points=1)
    a.add([make_scan(2.000, 0, [[1, 0, 0]])], [
        make_imu(1.996, 0, IDENTITY),
        make_imu(2.001, 1, quat_about_z(math.pi / 2)),   # nearest
        make_imu(2.008, 2, IDENTITY),
    ])
    f = a.flush()
    assert np.allclose(f.points, [[0, 1, 0]], atol=1e-6)


def test_unsorted_imu_window_still_matches():
    a = FrameAssembler(frame_duration=1.0, min_points=1)
    a.add([make_scan(2.000, 0, [[1, 0, 0]])], [
        make_imu(2.008, 2, IDENTITY),
        make_imu(2.001, 1, quat_about_z(math.pi / 2)),
        make_imu(1.996, 0, IDENTITY),
    ])
    f = a.flush()
    assert np.allclose(f.points, [[0, 1, 0]], atol=1e-6)


def test_no_imu_rotation_runs_with_no_imu_stream_at_all():
    """The ablation must not inherit a dependency on the stream it ablates.

    Before this was fixed, an unrotated scan still needed an IMU match to be
    ingested, so a scan-only stream left every scan in _pending and no frame
    ever closed -- a silent hang rather than a visible failure.
    """
    a = FrameAssembler(frame_duration=0.05, rotate_with_imu=False, min_points=1)
    frames = []
    for i in range(60):
        frames += a.add([marked_scan(1.0 + i * 0.005, i + 1)], [])   # no IMU
    assert frames, "no frame closed without an IMU stream"
    assert a.stats()["scans_pending"] == 0
    assert a.scans_unmatched == 0


def test_no_imu_rotation_ignores_a_useless_imu_window():
    """IMU samples nowhere near the scans must not cause drops in this mode."""
    a = FrameAssembler(frame_duration=0.05, max_time_gap=0.001,
                       rotate_with_imu=False, min_points=1)
    frames = []
    for i in range(60):
        t = 1.0 + i * 0.005
        frames += a.add([marked_scan(t, i + 1)], [make_imu(999.0, i)])
    assert frames
    assert a.scans_unmatched == 0


def test_imu_rotation_still_requires_a_match():
    """The fix must not loosen the rotate_with_imu=True path."""
    a = FrameAssembler(frame_duration=0.05, rotate_with_imu=True, min_points=1)
    for i in range(60):
        a.add([marked_scan(1.0 + i * 0.005, i + 1)], [])
    assert a.scans_ingested == 0
    assert a.stats()["scans_pending"] > 0


def test_rotate_with_imu_false_leaves_points_in_the_body_frame():
    a = FrameAssembler(frame_duration=1.0, rotate_with_imu=False, min_points=1)
    a.add([make_scan(2.0, 0, [[1, 0, 0]])],
          [make_imu(2.0, 0, quat_about_z(math.pi / 2))])
    f = a.flush()
    assert np.allclose(f.points, [[1, 0, 0]])


# --- ordering and robustness ------------------------------------------------

def test_out_of_order_scans_are_sorted_before_frames_are_cut():
    a = FrameAssembler(frame_duration=0.05, min_points=1)
    stamps = [1.00 + i * 0.005 for i in range(20)]
    scans = [marked_scan(t, i + 1) for i, t in enumerate(stamps)]
    imu = [make_imu(t, i) for i, t in enumerate(stamps)]
    frames = a.add(list(reversed(scans)), imu)   # delivered newest-first
    assert frames
    xs = np.concatenate([np.unique(f.points[:, 0]) for f in frames])
    assert np.all(np.diff(xs) > 0), "frames were cut from unsorted scans"


def test_sparse_frame_is_dropped_and_counted():
    a = FrameAssembler(frame_duration=0.05, min_points=1_000_000)
    frames = drive(a, n_scans=60, dt=0.005)
    assert frames == []
    assert a.frames_too_sparse > 0
    assert a.frames_emitted == 0


def test_flush_returns_none_when_nothing_is_open():
    assert FrameAssembler().flush() is None


def test_points_are_float64_and_contiguous():
    a = FrameAssembler(frame_duration=1.0, min_points=1)
    a.add([make_scan(1.0, 0, [[1, 1, 1]])], [make_imu(1.0, 0)])
    f = a.flush()
    assert f.points.dtype == np.float64      # Open3D / KISS-ICP require doubles
    assert f.points.flags.c_contiguous
    assert f.points.shape[1] == 3


def test_rejects_bad_construction():
    for kwargs in ({"frame_duration": 0}, {"max_time_gap": -1}):
        with pytest.raises(ValueError):
            FrameAssembler(**kwargs)
