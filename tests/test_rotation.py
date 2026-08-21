import math

import numpy as np

from l1_stream import (
    LidarIMU,
    RotatedScanAccumulator,
    normalize_quaternion,
    pack_scan_packet,
    parse_packet,
    quaternion_to_matrix,
    rotate_points,
)
from l1_stream.protocol import POINT_DTYPE

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def quat_about_z(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def quat_about_x(angle):
    return (math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2))


# --- quaternion basics ------------------------------------------------------

def test_identity_quaternion_is_a_no_op():
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])
    assert np.allclose(rotate_points(pts, IDENTITY), pts)


def test_90_degrees_about_z_maps_x_to_y():
    pts = np.array([[1.0, 0.0, 0.0]])
    out = rotate_points(pts, quat_about_z(math.pi / 2))
    assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)


def test_90_degrees_about_x_maps_y_to_z():
    pts = np.array([[0.0, 1.0, 0.0]])
    out = rotate_points(pts, quat_about_x(math.pi / 2))
    assert np.allclose(out, [[0.0, 0.0, 1.0]], atol=1e-12)


def test_rotation_preserves_length():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3)) * 10
    q = normalize_quaternion(rng.normal(size=4))
    out = rotate_points(pts, q)
    assert np.allclose(np.linalg.norm(pts, axis=1), np.linalg.norm(out, axis=1))


def test_vector_form_matches_matrix_form():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(50, 3))
    for _ in range(20):
        q = normalize_quaternion(rng.normal(size=4))
        assert np.allclose(rotate_points(pts, q), pts @ quaternion_to_matrix(q).T)


def test_matches_scipy_if_available():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        return  # optional cross-check
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(64, 3))
    for _ in range(20):
        q = normalize_quaternion(rng.normal(size=4))
        # scipy's from_quat also takes scalar-LAST (x, y, z, w).
        assert np.allclose(rotate_points(pts, q), Rotation.from_quat(q).apply(pts))


def test_non_unit_quaternion_is_normalised_not_scaled():
    # Without normalisation the formula scales points by |q|^2 as well as
    # rotating them -- the cloud silently grows.
    pts = np.array([[1.0, 0.0, 0.0]])
    scaled = tuple(3.0 * c for c in quat_about_z(math.pi / 2))
    assert np.allclose(rotate_points(pts, scaled), [[0.0, 1.0, 0.0]], atol=1e-12)


def test_zero_quaternion_falls_back_to_identity():
    pts = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(rotate_points(pts, (0.0, 0.0, 0.0, 0.0)), pts)


def test_rotate_points_rejects_bad_shape():
    for bad in (np.zeros((4,)), np.zeros((4, 2)), np.zeros((2, 3, 3))):
        try:
            rotate_points(bad, IDENTITY)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for shape {bad.shape}")


def test_output_is_float64_even_from_float32_input():
    pts = np.ones((3, 3), dtype=np.float32)
    assert rotate_points(pts, IDENTITY).dtype == np.float64


# --- accumulator ------------------------------------------------------------

def make_scan(stamp, scan_id, xyz):
    pts = np.zeros(len(xyz), dtype=POINT_DTYPE)
    pts["x"], pts["y"], pts["z"] = np.asarray(xyz, dtype=np.float32).T
    return parse_packet(pack_scan_packet(stamp, scan_id, pts))


def make_imu(stamp, imu_id, quaternion=IDENTITY):
    return LidarIMU(stamp=stamp, id=imu_id, quaternion=quaternion,
                    angular_velocity=(0, 0, 0), linear_acceleration=(0, 0, 0))


def test_accumulator_stacks_rotated_scans():
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    scans = [make_scan(1.000, 0, [[1, 0, 0]]), make_scan(1.002, 1, [[0, 1, 0]])]
    imu = [make_imu(1.000, 0), make_imu(1.002, 1)]
    assert acc.add(scans, imu) == 2
    assert np.allclose(acc.get_points(), [[1, 0, 0], [0, 1, 0]])


def test_accumulator_applies_the_nearest_imu_sample():
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    scan = make_scan(2.000, 0, [[1, 0, 0]])
    imu = [
        make_imu(1.996, 0, IDENTITY),
        make_imu(2.001, 1, quat_about_z(math.pi / 2)),  # nearest
        make_imu(2.008, 2, IDENTITY),
    ]
    acc.add([scan], imu)
    assert np.allclose(acc.get_points(), [[0, 1, 0]], atol=1e-6)


def test_scans_are_not_lost_when_imu_has_not_arrived_yet():
    """Draining the scan buffer must not discard scans the IMU cannot yet match."""
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    scan = make_scan(5.0, 0, [[1, 0, 0]])
    assert acc.add([scan], []) == 0          # nothing to match against yet
    assert acc.stats()["scans_pending"] == 1  # ... but it is NOT thrown away
    assert acc.add([], [make_imu(5.0, 0)]) == 1
    assert len(acc.get_points()) == 1


def test_scan_newer_than_all_imu_stays_pending():
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    acc.add([make_scan(10.0, 0, [[1, 0, 0]])], [make_imu(9.0, 0)])
    assert acc.stats()["scans_pending"] == 1
    assert acc.stats()["scans_unmatched"] == 0


def test_scan_older_than_all_imu_is_dropped_and_counted():
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    acc.add([make_scan(1.0, 0, [[1, 0, 0]])], [make_imu(9.0, 0)])
    assert acc.stats()["scans_pending"] == 0
    assert acc.stats()["scans_unmatched"] == 1


def test_out_of_order_imu_delivery_still_matches_correctly():
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    scan = make_scan(3.000, 0, [[1, 0, 0]])
    imu = [  # deliberately shuffled, as UDP may reorder
        make_imu(3.008, 2, IDENTITY),
        make_imu(3.001, 1, quat_about_z(math.pi / 2)),
        make_imu(2.996, 0, IDENTITY),
    ]
    acc.add([scan], imu)
    assert np.allclose(acc.get_points(), [[0, 1, 0]], atol=1e-6)


def test_window_evicts_oldest_scans():
    acc = RotatedScanAccumulator(max_scans=3, max_time_gap=0.01)
    for i in range(6):
        t = 1.0 + i * 0.001
        acc.add([make_scan(t, i, [[float(i), 0, 0]])], [make_imu(t, i)])
    pts = acc.get_points()
    assert len(pts) == 3
    assert np.allclose(pts[:, 0], [3, 4, 5])


def test_get_points_is_empty_array_not_none_when_cold():
    acc = RotatedScanAccumulator()
    pts = acc.get_points()
    assert pts.shape == (0, 3)
    assert pts.dtype == np.float64


def test_get_points_cache_invalidated_on_new_data():
    acc = RotatedScanAccumulator(max_scans=5, max_time_gap=0.01)
    acc.add([make_scan(1.0, 0, [[1, 0, 0]])], [make_imu(1.0, 0)])
    first = acc.get_points()
    assert acc.get_points() is first  # cached between updates
    acc.add([make_scan(1.001, 1, [[2, 0, 0]])], [make_imu(1.001, 1)])
    assert len(acc.get_points()) == 2


def test_zero_returns_are_filtered():
    acc = RotatedScanAccumulator(max_scans=5, max_time_gap=0.01, drop_zero_returns=True)
    acc.add([make_scan(1.0, 0, [[0, 0, 0], [1, 2, 3]])], [make_imu(1.0, 0)])
    assert len(acc.get_points()) == 1


def test_reset_clears_everything():
    acc = RotatedScanAccumulator(max_scans=5, max_time_gap=0.01)
    acc.add([make_scan(1.0, 0, [[1, 0, 0]])], [make_imu(1.0, 0)])
    acc.reset()
    assert len(acc.get_points()) == 0
    assert len(acc) == 0


def test_pending_overflow_is_counted_not_silent():
    acc = RotatedScanAccumulator(max_scans=5, max_time_gap=0.01, pending_maxlen=2)
    # No IMU ever arrives, so nothing can be matched and the queue overflows.
    for i in range(5):
        acc.add([make_scan(1.0 + i * 0.001, i, [[1, 0, 0]])], [])
    assert acc.stats()["scans_pending"] == 2
    assert acc.stats()["scans_unmatched"] == 3  # the 3 evicted ones are accounted for


# --- timestamp ordering (out-of-order / delayed scan arrival) --------------
#
# Order is verified through get_points(): each test scan carries a distinct x
# value, so the x column reflects the order scans sit in the window.

def test_reordered_arrival_within_one_call_is_sorted_by_capture_time():
    """UDP does not guarantee ordering, so scans can arrive newest-first."""
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    late = make_scan(1.006, 1, [[2, 0, 0]])
    early = make_scan(1.000, 0, [[1, 0, 0]])
    imu = [make_imu(1.000, 0), make_imu(1.006, 1)]

    acc.add([late, early], imu)  # delivered out of capture order

    assert np.allclose(acc.get_points(), [[1, 0, 0], [2, 0, 0]])


def test_delayed_scan_arriving_in_a_later_call_is_still_ordered():
    """A datagram delayed past a newer one must not land after it in the window."""
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)

    acc.add([make_scan(1.006, 1, [[2, 0, 0]])], [make_imu(1.006, 1)])
    # The older scan's datagram shows up a call later.
    acc.add([make_scan(1.000, 0, [[1, 0, 0]])],
            [make_imu(1.000, 0), make_imu(1.006, 1)])

    assert np.allclose(acc.get_points(), [[1, 0, 0], [2, 0, 0]])


def test_eviction_drops_oldest_by_timestamp_not_by_insertion_order():
    acc = RotatedScanAccumulator(max_scans=2, max_time_gap=0.01)

    for i, t in enumerate((1.010, 1.020), start=1):
        acc.add([make_scan(t, i, [[float(i), 0, 0]])], [make_imu(t, i)])

    # A delayed scan OLDER than both. The window is full, so letting it in must
    # not evict a newer scan to make room for it.
    acc.add([make_scan(1.000, 0, [[9, 0, 0]])],
            [make_imu(1.000, 0), make_imu(1.010, 1)])

    pts = acc.get_points()
    assert len(pts) == 2
    assert np.allclose(pts[:, 0], [1, 2])  # the 9 was correctly not kept


def test_in_order_arrivals_are_unaffected():
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    for i in range(1, 6):  # start at 1: [0,0,0] is a zero-return and is filtered
        t = 1.0 + i * 0.001
        acc.add([make_scan(t, i, [[float(i), 0, 0]])], [make_imu(t, i)])
    assert np.allclose(acc.get_points()[:, 0], [1, 2, 3, 4, 5])


def test_equal_timestamps_do_not_raise_on_array_comparison():
    """The seq tiebreaker exists so tuple compare never reaches the ndarray."""
    acc = RotatedScanAccumulator(max_scans=10, max_time_gap=0.01)
    acc.add([make_scan(1.0, 0, [[1, 0, 0]])], [make_imu(1.0, 0)])
    acc.add([make_scan(1.0, 1, [[2, 0, 0]])], [make_imu(1.0, 1)])
    # Same stamp again, delivered after an older one -- forces the sort path
    # with tied timestamps.
    acc.add([make_scan(1.0, 2, [[3, 0, 0]]), make_scan(0.999, 3, [[4, 0, 0]])],
            [make_imu(0.999, 3), make_imu(1.0, 2)])
    assert len(acc.get_points()) == 4
