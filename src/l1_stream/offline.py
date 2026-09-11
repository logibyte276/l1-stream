"""One place where "replay a recording through odometry" is implemented.

WHY THIS EXISTS. Six scripts had grown their own copy of the same fifteen-line
loop -- build a FrameAssembler, build a KissOdometry, drain the Replayer,
register each frame, collect metrics -- and, worse, their own copy of the
DEFAULTS. Those defaults drifted:

    06_odometry_offline   voxel 0.15  min_range 0.25  deskew off
    07_diagnostics        voxel 0.15  min_range 0.40  deskew off  (now 07_precheck)
    09_ablation           voxel 0.25  min_range 0.40  deskew on   (deleted)
    08_odometry_live      voxel 0.25  min_range 0.40  deskew on   <- the ROBOT

Every one of those looks plausible in isolation, and the numbers they print
look comparable when they are not. The live script -- the one that actually
drives the car -- was running a configuration nobody had tuned since the first
week. That is not a tidiness problem; it is the same silent-config-drift bug
that has cost this project several wrong conclusions already.

So: :data:`DEFAULTS` is the settled configuration, in one place.
:func:`add_args` puts it on any argparse parser. :func:`replay` runs the
pipeline. Scripts do analysis, not plumbing.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULTS
from .frames import FrameAssembler
from .odometry import KissOdometry
from .recording import Replayer

__all__ = ["DEFAULTS", "OdometryRun", "add_args", "config_from_args", "replay"]
# DEFAULTS is re-exported from .config so callers have one import to reach for.



def add_args(parser) -> None:
    """Attach the standard odometry flags, carrying :data:`DEFAULTS`."""
    d = DEFAULTS
    parser.add_argument("--frame-duration", type=float, default=d["frame_duration"])
    parser.add_argument("--voxel-size", type=float, default=d["voxel_size"])
    parser.add_argument("--max-range", type=float, default=d["max_range"])
    parser.add_argument("--min-range", type=float, default=d["min_range"])
    parser.add_argument("--initial-threshold", type=float,
                        default=d["initial_threshold"])
    # BooleanOptionalAction so --help shows the default and both directions
    # exist. A store_true flag hides which way the default points, which is
    # exactly how the deskew setting became ambiguous.
    import argparse
    parser.add_argument("--imu-rotation", action=argparse.BooleanOptionalAction,
                        default=d["rotate_with_imu"],
                        help="IMU pre-rotation. ON by default: 32-126x worse without.")
    parser.add_argument("--deskew", action=argparse.BooleanOptionalAction,
                        default=d["deskew"],
                        help="OFF by default here (KISS-ICP's own default is ON). "
                             "UNRESOLVED: better off in a room, better on in a "
                             "bare hallway. Worth running both ways every time.")


def config_from_args(args) -> dict:
    """Pull a config dict out of a parsed namespace built by :func:`add_args`."""
    return {
        "frame_duration": args.frame_duration,
        "voxel_size": args.voxel_size,
        "max_range": args.max_range,
        "min_range": args.min_range,
        "initial_threshold": args.initial_threshold,
        "rotate_with_imu": args.imu_rotation,
        "deskew": args.deskew,
    }


@dataclass
class OdometryRun:
    """Everything one replay produced, plus the metrics derived from it."""

    xyz: np.ndarray                 # (K, 3) positions
    stamps: np.ndarray              # (K,) pose timestamps (end of frame)
    deltas: list                    # per-frame 4x4 last_delta
    spans: np.ndarray               # (K,) frame durations
    sizes: np.ndarray               # (K,) points per frame
    thresholds: np.ndarray          # (K,) adaptive threshold after each frame
    config: dict
    assembler_stats: dict = field(default_factory=dict)
    world_points: np.ndarray | None = None   # (M, 3) map; None unless map_voxel set
    replay_seconds: float = 0.0              # wall time of the LOOP only
    replay_host: str = ""                    # which machine did the replaying
    map_deskew: bool | None = None           # was the MAP deskewed (may differ)

    # --- cost ---

    @property
    def ms_per_frame(self) -> float:
        """Compute cost per frame, in ms.

        Timed around the registration loop only -- not imports, not kiss-icp
        construction, not the analysis prints -- because those are once-per-run
        and the live pipeline does not pay them per frame.
        """
        return 1000 * self.replay_seconds / max(len(self.xyz), 1)

    @property
    def budget_pct(self) -> float:
        """``ms_per_frame`` as a share of the frame period. Over 100% means the
        pipeline cannot keep up with the sensor ON THIS MACHINE.

        Replay is unpaced, so this measures COMPUTE, not latency -- and it
        measures it on whatever machine ran the replay. A number from a desktop
        says nothing about whether the Orin can keep up. Check ``replay_host``
        before quoting it as a real-time result.
        """
        return 100 * self.ms_per_frame / (1000 * self.config["frame_duration"])

    # --- basic geometry ---

    @property
    def path_length(self) -> float:
        if len(self.xyz) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.xyz, axis=0), axis=1).sum())

    @property
    def net_displacement(self) -> float:
        if len(self.xyz) < 2:
            return 0.0
        return float(np.linalg.norm(self.xyz[-1] - self.xyz[0]))

    @property
    def step(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.xyz, axis=0), axis=1)

    @property
    def speed(self) -> np.ndarray:
        return self.step / self.config["frame_duration"]

    # --- error metrics ---

    @property
    def rotation_deg(self) -> np.ndarray:
        """Per-frame rotation magnitude, axis-independent.

        Euler's rotation theorem: any 3D rotation is a turn of some angle about
        some axis, and trace(R) = 1 + 2cos(theta). The clip is not optional --
        float error pushes the argument past 1.0 and arccos returns NaN, which
        would poison every aggregate computed from this array.
        """
        out = []
        for d in self.deltas:
            r = np.asarray(d)[:3, :3]
            out.append(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        return np.asarray(out)

    @property
    def jitter_mm(self) -> float:
        """Estimator jitter per frame, from second differences.

        A real vehicle has inertia, so each pose should sit near the midpoint
        of its neighbours; the leftover is jitter. For per-axis sigma s, |d2|
        has magnitude s*sqrt(18) while the jitter is s*sqrt(3), hence /2.45.
        The MEDIAN keeps genuinely accelerating frames from inflating it, and
        the /0.89 corrects a bias calibrated against known synthetic jitter.

        Unlike backing jitter out of the path/net ratio, this does not assume
        the true path was straight -- real steering is smooth and passes through.
        """
        if len(self.xyz) < 3:
            return 0.0
        d2 = self.xyz[2:] - 2 * self.xyz[1:-1] + self.xyz[:-2]
        return float(np.median(np.linalg.norm(d2, axis=1))) / 2.45 / 0.89 * 1000

    def clipping(self, still: float = 0.15) -> dict:
        """Did the recording bracket the motion, or start/stop mid-drive?

        Odometry can only report motion it was recording for. A recorder that
        starts late looks exactly like an algorithmic scale error against a
        tape measure -- at 0.5 m/s one missed 0.2 s frame is 100 mm.
        """
        s = self.speed
        if len(s) == 0 or not (s > still).any():
            return {"head": 0, "tail": 0, "clipped": [], "still": still}
        head = int(np.argmax(s > still))
        tail = len(s) - int(np.argmax(s[::-1] > still))
        clipped = []
        if head == 0:
            clipped.append(("START", float(s[0])))
        if len(s) - tail == 0:
            clipped.append(("END", float(s[-1])))
        return {"head": head, "tail": len(s) - tail,
                "clipped": clipped, "still": still}


def replay(
    path: str,
    *,
    period: float = 0.05,
    map_voxel: float | None = None,
    map_deskew: bool = True,
    **cfg,
) -> OdometryRun:
    """Run one recording through the pipeline and return everything measured.

    ``cfg`` overrides :data:`DEFAULTS`; anything omitted takes the settled value,
    so a caller can never silently run an untuned configuration.

    ``map_deskew`` controls deskew FOR THE MAP ONLY, independently of the
    registration ``deskew`` setting, and defaults to True. See the note in
    ``KissOdometry.__init__``: intra-frame smear is below the registration
    voxel at survey speeds but several times the map voxel, so a map wants
    deskew even where registration measured fine without it.

    ``map_voxel`` additionally accumulates a world-frame point cloud into
    :attr:`OdometryRun.world_points`, downsampled to that voxel. It lives here
    rather than in a second replay loop because this project has already paid
    once for having six copies of this loop drift apart. Off by default: a
    60 s drive is over a million points and most callers only want metrics.
    """
    config = {**DEFAULTS, **cfg}

    assembler = FrameAssembler(
        frame_duration=config["frame_duration"],
        rotate_with_imu=config["rotate_with_imu"],
    )
    odom = KissOdometry(
        # None when not mapping: no second preprocessor, no per-frame cost.
        map_deskew=map_deskew if map_voxel else None,
        voxel_size=config["voxel_size"],
        max_range=config["max_range"],
        min_range=config["min_range"],
        deskew=config["deskew"],
        initial_threshold=config["initial_threshold"],
    )

    spans, sizes, thresholds, deltas = [], [], [], []
    chunks: list = []
    t0 = time.perf_counter()

    def take(frame):
        pose = odom.register(frame)
        spans.append(frame.span)
        sizes.append(len(frame))
        thresholds.append(odom.threshold)
        deltas.append(np.array(odom.last_delta, copy=True))
        if map_voxel:
            # Map from odom.last_preprocessed, NOT frame.points. The raw frame
            # still contains chassis self-hits below min_range and, when deskew
            # is on, is not motion-compensated -- so placing it at this pose
            # would put points into the map that the pose does not describe.
            # last_pose maps that preprocessed cloud into the map, so this is
            # correct whether or not the frame was IMU-rotated first.
            pts = odom.last_map_cloud
            if pts is not None and len(pts):
                chunks.append(pts @ pose[:3, :3].T + pose[:3, 3])

    for scans, imu in Replayer(path).iter_batches(period=period):
        for frame in assembler.add(scans, imu):
            take(frame)
    tail = assembler.flush()
    if tail is not None:
        take(tail)

    elapsed = time.perf_counter() - t0

    return OdometryRun(
        xyz=odom.trajectory(),
        stamps=np.asarray(odom.stamps),
        deltas=deltas,
        spans=np.asarray(spans),
        sizes=np.asarray(sizes),
        thresholds=np.asarray(thresholds),
        config=config,
        assembler_stats=assembler.stats(),
        world_points=_build_map(chunks, map_voxel) if map_voxel else None,
        replay_seconds=elapsed,
        replay_host=socket.gethostname(),
        map_deskew=odom.map_deskew if map_voxel else None,
    )


def _build_map(chunks: list, voxel: float) -> np.ndarray:
    from .mapmetrics import voxel_downsample
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return voxel_downsample(np.concatenate(chunks, axis=0), voxel)
