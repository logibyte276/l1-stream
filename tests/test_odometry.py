"""End-to-end: does the pipeline recover a motion we already know?

Skipped when kiss-icp is absent. `importorskip`, not try/except, so a skip is
visible in the pytest summary -- a silent pass here would be worse than a
failure, since this is the only test that exercises registration at all.

The simulated sensor is deliberately L1-shaped: 120 points per packet, 180
packets/s, a 25 m range gate, and per-point times spanning 5.56 ms.
"""

import numpy as np
import pytest

from l1_stream.frames import FrameAssembler
from l1_stream.odometry import KissOdometry
from l1_stream.protocol import POINT_DTYPE, LidarIMU, pack_scan_packet, parse_packet

# After the imports, not before: KissOdometry defers its `import kiss_icp` into
# __init__, so this module imports fine without it. That keeps the imports at
# the top of the file (ruff E402) while still skipping the whole module when
# kiss-icp is absent.
pytest.importorskip("kiss_icp")

IDENTITY = (0.0, 0.0, 0.0, 1.0)
SCAN_HZ = 180.0
PTS_PER_SCAN = 120


def room_cloud(seed=0, n=40_000, half=8.0, height=3.0):
    """A closed box of surface points -- the fixed geometry the sensor sees."""
    rng = np.random.default_rng(seed)
    walls = []
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            p = np.empty((n // 5, 3))
            p[:, axis] = sign * half
            p[:, 1 - axis] = rng.uniform(-half, half, len(p))
            p[:, 2] = rng.uniform(-0.5, height, len(p))
            walls.append(p)
    ceiling = np.empty((n // 5, 3))
    ceiling[:, 0] = rng.uniform(-half, half, len(ceiling))
    ceiling[:, 1] = rng.uniform(-half, half, len(ceiling))
    ceiling[:, 2] = height
    walls.append(ceiling)
    return np.concatenate(walls, axis=0)


def simulate(world, positions, stamps, seed=1):
    """Yield (scan, imu) pairs for a sensor translating through `world`."""
    rng = np.random.default_rng(seed)
    for i, (pos, t) in enumerate(zip(positions, stamps, strict=True)):
        idx = rng.choice(len(world), PTS_PER_SCAN, replace=False)
        local = world[idx] - pos
        pts = np.zeros(PTS_PER_SCAN, dtype=POINT_DTYPE)
        pts["x"], pts["y"], pts["z"] = local.T
        pts["time"] = np.linspace(0.0, 1.0 / SCAN_HZ, PTS_PER_SCAN, dtype=np.float32)
        scan = parse_packet(pack_scan_packet(t, i, pts))
        imu = LidarIMU(stamp=t, id=i, quaternion=IDENTITY,
                       angular_velocity=(0, 0, 0), linear_acceleration=(0, 0, 0))
        yield scan, imu


def run(speed=0.5, seconds=6.0, **odom_kwargs):
    world = room_cloud()
    n = int(seconds * SCAN_HZ)
    stamps = 100.0 + np.arange(n) / SCAN_HZ
    positions = np.zeros((n, 3))
    positions[:, 0] = speed * (stamps - stamps[0])

    assembler = FrameAssembler(frame_duration=0.2)
    odom = KissOdometry(voxel_size=0.25, max_range=25.0, min_range=0.1,
                        initial_threshold=0.4, **odom_kwargs)

    imu_window = []
    for scan, imu in simulate(world, positions, stamps):
        imu_window.append(imu)
        imu_window = imu_window[-500:]
        for frame in assembler.add([scan], imu_window):
            odom.register(frame)
    return odom, assembler, positions, stamps


def test_recovers_a_known_straight_line():
    odom, assembler, positions, stamps = run(speed=0.5, seconds=6.0)

    assert len(odom.poses) >= 25, "too few frames closed"
    assert assembler.frames_too_sparse == 0
    assert assembler.scans_unmatched == 0

    xyz = odom.trajectory()
    truth = np.array([
        np.interp(t, stamps, positions[:, 0]) for t in odom.stamps
    ])

    # Absolute position error against ground truth, in metres.
    err = np.abs(xyz[:, 0] - (truth - truth[0]))
    assert err.max() < 0.30, f"max along-track error {err.max():.3f} m"

    # Cross-track: the sensor never moved in y or z.
    assert np.abs(xyz[:, 1]).max() < 0.20
    assert np.abs(xyz[:, 2]).max() < 0.20

    # Distance travelled, ~2.6 m of the 3.0 m total (first frame is the origin).
    assert 2.0 < odom.path_length() < 3.2


def test_stationary_sensor_does_not_drift():
    odom, _a, _p, _s = run(speed=0.0, seconds=4.0)
    xyz = odom.trajectory()
    assert np.linalg.norm(xyz[-1] - xyz[0]) < 0.05
    assert odom.path_length() < 0.20


def test_pose_timestamps_are_frame_ends_and_increase():
    odom, _a, _p, _s = run(speed=0.5, seconds=3.0)
    t = np.array(odom.stamps)
    assert np.all(np.diff(t) > 0)
    assert np.allclose(np.diff(t), np.diff(t)[0], atol=0.02), "frame spacing jitters"


def test_voxel_size_none_is_refused_before_it_can_crash_voxelhashmap():
    with pytest.raises(ValueError, match="voxel_size"):
        KissOdometry(voxel_size=None)


def test_min_range_must_be_below_max_range():
    with pytest.raises(ValueError, match="min_range"):
        KissOdometry(min_range=30.0, max_range=25.0)


def test_deskew_path_runs_and_agrees_broadly_with_no_deskew():
    """Both settings must produce a sane trajectory; this pins that the
    timestamps we hand register_frame are accepted in the expected [0, 1]
    convention rather than silently wrecking the solution."""
    on, _a, _p, _s = run(speed=0.5, seconds=4.0, deskew=True)
    off, _a2, _p2, _s2 = run(speed=0.5, seconds=4.0, deskew=False)
    assert abs(on.path_length() - off.path_length()) < 0.5
    assert on.trajectory()[-1][0] > 1.0 and off.trajectory()[-1][0] > 1.0


def test_register_keeps_the_preprocessed_cloud_not_the_raw_frame():
    """A map must be built from the points the POSE describes.

    KISS-ICP's register_frame returns the frame after its own motion
    compensation and range cropping. Discarding it and mapping from the raw
    frame puts chassis self-hits below min_range into the map, and -- with
    deskew on -- points that were never motion-compensated.
    """
    import numpy as np

    from l1_stream.frames import Frame
    from l1_stream.odometry import KissOdometry

    pytest.importorskip("kiss_icp")

    odom = KissOdometry(min_range=1.0, max_range=20.0, deskew=False)
    assert odom.last_preprocessed is None

    rng = np.random.default_rng(0)
    far = rng.uniform(-10, 10, (3000, 3))
    far = far[np.linalg.norm(far, axis=1) > 2.0]
    close = rng.uniform(-0.3, 0.3, (200, 3))       # chassis self-hits
    pts = np.vstack([far, close])
    frame = Frame(points=np.ascontiguousarray(pts),
                  timestamps=np.linspace(0, 1, len(pts)),
                  t_start=0.0, t_end=0.2, n_scans=1)

    odom.register(frame)
    kept = odom.last_preprocessed
    assert kept is not None
    r = np.linalg.norm(kept, axis=1)
    assert r.min() >= 1.0, "points inside min_range survived into the map cloud"
    assert r.max() <= 20.0
    assert len(kept) < len(pts), "nothing was cropped -- preprocess was skipped"


def test_map_deskew_is_independent_of_registration_deskew():
    """A map wants deskew even where registration measured fine without it.

    Intra-frame smear is speed*frame_duration -- 100 mm at 0.5 m/s. That is
    BELOW the 0.15 m registration voxel (absorbed) but 3.3x the 0.03 m map
    voxel (fully visible as thickened walls).
    """
    import numpy as np

    from l1_stream.frames import Frame
    from l1_stream.odometry import KissOdometry

    pytest.importorskip("kiss_icp")
    rng = np.random.default_rng(1)
    pts = rng.uniform(-8, 8, (2500, 3))
    pts = np.ascontiguousarray(pts[np.linalg.norm(pts, axis=1) > 1.5])
    def mk():
        return Frame(points=pts.copy(),
                     timestamps=np.linspace(0, 1, len(pts)),
                     t_start=0.0, t_end=0.2, n_scans=1)

    # KissOdometry defaults to following `deskew` -- constructing it bare must
    # not silently allocate a second preprocessor nobody asked for.
    bare = KissOdometry(deskew=False, min_range=1.0)
    assert bare.map_deskew is False

    # replay() opts in with map_deskew=True when it is actually building a map
    odom = KissOdometry(deskew=False, map_deskew=True, min_range=1.0)
    assert odom.map_deskew is True
    odom.register(mk())
    odom.register(mk())
    assert odom.last_map_cloud is not None

    # explicitly matched -> the very same array, no second preprocess
    same = KissOdometry(deskew=False, map_deskew=False, min_range=1.0)
    assert same.map_deskew is False
    same.register(mk())
    assert same.last_map_cloud is same.last_preprocessed

    # and the map cloud is still range-cropped either way
    for o in (odom, same):
        r = np.linalg.norm(o.last_map_cloud, axis=1)
        assert r.min() >= 1.0
