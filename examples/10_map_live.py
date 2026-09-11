"""Live map: live KISS-ICP odometry, the map it builds, in an Open3D window.

    python examples/10_map_live.py
    python examples/10_map_live.py --compare                 # M flips map deskew
    python examples/10_map_live.py --save personal/live1     # keep it on exit

The live counterpart of 09_map_offline: same odometry flags, same two deskews,
and --save writes files 09_map_offline --load opens.

    --deskew / --no-deskew            what REGISTRATION uses. Changes the poses.
    --map-deskew / --no-map-deskew    what the MAP uses. Default ON.

--compare builds both maps at once. It costs no extra registration: KISS-ICP
already hands back one cloud per setting, and this keeps both.

WHAT TO WATCH IN THE STATUS LINE
    ms/frame   registration time. Over the frame budget (200 ms) and the map
               falls behind the car.
    lag        how far the last registered frame trails the newest scan. It
               should hover near one frame; if it keeps growing, the CPU cannot
               keep up -- raise --map-voxel, drop --compare, or use --no-map-deskew.

ONE THREAD, ON PURPOSE. Registration and drawing share the loop, so the window
feels choppy while a frame registers (~50-100 ms on the Orin). A render thread
would not fix it: KISS-ICP holds Python's GIL while it works.

It needs a screen on the machine running it. Open3D over ssh -X rarely works.
Either plug a monitor into the Orin, or point the publisher at the laptop and
run this there (it then needs kiss-icp on the laptop). The UDP port can only
be read by one program, so this cannot run alongside 05_record.

KEYS   M  map deskew ON / OFF (--compare)   K  trajectory on / off
       F  follow the car                    X  clear the map
       +/-  point size    R  reset view     Q  quit (saves if --save)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from l1_stream import LidarStream
from l1_stream.frames import FrameAssembler
from l1_stream.odometry import KissOdometry
from l1_stream.offline import add_args, config_from_args

LABEL = {"deskew": "map deskew ON", "raw": "map deskew OFF"}
VIRIDIS = np.array([[0.267, 0.005, 0.329], [0.230, 0.322, 0.546], [0.128, 0.567, 0.551],
                    [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]])
# Same file format as 09_map_offline, so its --load can open what this saves.
PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
PLY_HEADER = ("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_args(p)
    p.add_argument("--map-deskew", action=argparse.BooleanOptionalAction, default=True,
                   help="Deskew the MAP with KISS-ICP's velocity (poses are unaffected).")
    p.add_argument("--compare", action="store_true",
                   help="Keep a deskewed AND a raw map; press M to flip.")
    p.add_argument("--map-voxel", type=float, default=0.05,
                   help="Map resolution, m. Coarser than 09_map_offline's 0.03 so a long "
                        "drive stays drawable live.")
    p.add_argument("--max-map-points", type=int, default=1_000_000,
                   help="Stop growing a map past this (press X to clear).")
    p.add_argument("--redraw-hz", type=float, default=5.0,
                   help="How often the map on screen is refreshed.")
    p.add_argument("--z-range", type=float, nargs=2, default=(-0.3, 2.5), metavar=("LO", "HI"),
                   help="Height colour scale, m relative to where the L1 started. Fixed, so "
                        "each point is coloured once instead of on every redraw.")
    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--save", metavar="PREFIX", default=None,
                   help="On exit write PREFIX_map_*.ply and PREFIX_traj.npy.")
    return p.parse_args(argv)


# --- map ---------------------------------------------------------------------

class VoxelMap:
    """Grows a map one frame at a time, keeping the FIRST point per voxel -- the
    same rule as mapmetrics.voxel_downsample, done incrementally so a long drive
    does not re-downsample everything on every frame."""

    _OFF = 1 << 20                 # 21 bits per axis: +/-2^20 voxels, ~52 km at 5 cm

    def __init__(self, voxel: float, max_points: int, z_range=(-0.3, 2.5)):
        self.voxel = float(voxel)
        self.max_points = int(max_points)
        self.z_range = tuple(z_range)
        self._seen: set[int] = set()
        self._chunks: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self.n = 0
        self.full = False

    def add(self, pts: np.ndarray) -> None:
        if pts is None or not len(pts) or self.full:
            return
        k = np.floor(pts / self.voxel).astype(np.int64) + self._OFF
        packed = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
        packed, first = np.unique(packed, return_index=True)
        fresh = np.fromiter((int(v) not in self._seen for v in packed), dtype=bool,
                            count=len(packed))
        if not fresh.any():
            return
        self._seen.update(packed[fresh].tolist())
        new = np.asarray(pts[np.sort(first[fresh])], dtype=np.float64)
        self._chunks.append(new)
        self._colors.append(height_colors(new, *self.z_range))
        self.n += len(new)
        if self.n >= self.max_points:
            self.full = True
            print(f"\nmap reached {self.n:,} points -- no longer growing (X clears)")

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """(xyz, rgb) of the whole map."""
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
            self._colors = [np.concatenate(self._colors)]
        if not self._chunks:
            return np.empty((0, 3)), np.empty((0, 3))
        return self._chunks[0], self._colors[0]

    def clear(self) -> None:
        self._seen.clear()
        self._chunks.clear()
        self._colors.clear()
        self.n = 0
        self.full = False


def height_colors(xyz, lo, hi):
    t = np.clip((xyz[:, 2] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    stops = np.linspace(0.0, 1.0, len(VIRIDIS))
    return np.stack([np.interp(t, stops, VIRIDIS[:, c]) for c in range(3)], axis=1)


def write_ply(path, xyz, rgb):
    arr = np.empty(len(xyz), dtype=PLY_DTYPE)
    for i, k in enumerate("xyz"):
        arr[k] = xyz[:, i]
    c = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    for i, k in enumerate(("red", "green", "blue")):
        arr[k] = c[:, i]
    with open(path, "wb") as f:
        f.write(PLY_HEADER.format(n=len(arr)).encode("ascii"))
        arr.tofile(f)


# --- the viewer --------------------------------------------------------------

class LiveMapViewer:
    def __init__(self, args, stream):
        import open3d as o3d

        self.o3d = o3d
        self.args = args
        self.stream = stream
        cfg = config_from_args(args)
        self.frame_duration = cfg["frame_duration"]

        # With --compare, map deskew is set OPPOSITE to registration deskew, so
        # KissOdometry's two clouds (last_preprocessed, last_map_cloud) are one
        # deskewed and one raw -- both maps for the cost of one registration.
        self.compare = args.compare
        map_deskew = (not cfg["deskew"]) if self.compare else args.map_deskew
        self.map_deskew = map_deskew
        self.assembler = FrameAssembler(frame_duration=cfg["frame_duration"],
                                        rotate_with_imu=cfg["rotate_with_imu"])
        self.odom = KissOdometry(voxel_size=cfg["voxel_size"], max_range=cfg["max_range"],
                                 min_range=cfg["min_range"], deskew=cfg["deskew"],
                                 map_deskew=map_deskew,
                                 initial_threshold=cfg["initial_threshold"])
        print("config     " + "  ".join(f"{k}={v}" for k, v in cfg.items())
              + f"  map_deskew={map_deskew}" + ("  (compare)" if self.compare else ""))

        names = ["deskew", "raw"] if self.compare else ["deskew" if map_deskew else "raw"]
        self.maps = {n: VoxelMap(args.map_voxel, args.max_map_points, args.z_range)
                     for n in names}
        self.shown = names[0]
        self.follow = False
        self.traj_on = True
        self.reg_ms: list[float] = []
        self.last_t_end = None

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(window_name="L1 live map", width=1400, height=900):
            raise SystemExit("Open3D could not open a window (no display?). Plug a monitor "
                             "into this machine, or point the publisher at one that has a "
                             "screen and run this there.")
        # Something with real extent first, so the camera does not fit an empty box.
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd, reset_bounding_box=False)
        self.line = o3d.geometry.LineSet()
        self.vis.add_geometry(self.line, reset_bounding_box=False)
        self.marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        self.marker.paint_uniform_color([1.0, 0.0, 0.0])
        self.marker_at = np.zeros(3)
        self.vis.add_geometry(self.marker, reset_bounding_box=False)
        self.vis.get_render_option().point_size = args.point_size
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_front([-0.5, -0.5, 0.7])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.4)

        for key, fn in (("M", self.key_map), ("K", self.key_traj),
                        ("F", self.key_follow), ("X", self.key_clear)):
            self.vis.register_key_callback(ord(key), fn)

    # -- keys -------------------------------------------------------------------

    def key_map(self, _vis):
        if not self.compare:
            print("\nonly one map -- run with --compare to flip map deskew")
            return False
        self.shown = "raw" if self.shown == "deskew" else "deskew"
        print(f"\nshowing    {LABEL[self.shown]}")
        self.redraw()
        return True

    def key_traj(self, _vis):
        self.traj_on = not self.traj_on
        self.redraw()
        return True

    def key_follow(self, _vis):
        self.follow = not self.follow
        print(f"\nfollow     {'ON' if self.follow else 'OFF'}")
        return True

    def key_clear(self, _vis):
        for m in self.maps.values():
            m.clear()
        print("\nmap cleared (trajectory kept)")
        self.redraw()
        return True

    # -- odometry ---------------------------------------------------------------

    def ingest(self, scans, imu) -> int:
        n = 0
        for frame in self.assembler.add(scans, imu):
            t0 = time.perf_counter()
            self.odom.register(frame)
            self.reg_ms.append(1000 * (time.perf_counter() - t0))
            pose = self.odom.poses[-1]
            R, t = pose[:3, :3], pose[:3, 3]
            if self.compare:
                # map_deskew was set opposite to registration deskew (see __init__)
                clouds = ({"deskew": self.odom.last_map_cloud, "raw": self.odom.last_preprocessed}
                          if self.map_deskew else
                          {"deskew": self.odom.last_preprocessed, "raw": self.odom.last_map_cloud})
            else:
                clouds = {self.shown: self.odom.last_map_cloud}
            for name, pts in clouds.items():
                if pts is not None and len(pts):
                    self.maps[name].add(pts @ R.T + t)
            self.last_t_end = frame.t_end
            n += 1
        return n

    # -- drawing ----------------------------------------------------------------

    def redraw(self):
        o3d = self.o3d
        pts, rgb = self.maps[self.shown].points()
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.update_geometry(self.pcd)

        traj = self.odom.trajectory()
        if self.traj_on and len(traj) >= 2:
            idx = np.arange(len(traj) - 1)
            self.line.points = o3d.utility.Vector3dVector(traj)
            self.line.lines = o3d.utility.Vector2iVector(np.stack([idx, idx + 1], axis=1))
            self.line.paint_uniform_color([1.0, 0.0, 0.0])
        else:
            self.line.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            self.line.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
        self.vis.update_geometry(self.line)

        if len(traj):
            self.marker.translate(traj[-1] - self.marker_at)
            self.marker_at = traj[-1].copy()
            self.vis.update_geometry(self.marker)
            if self.follow:
                self.vis.get_view_control().set_lookat(traj[-1])

    def status(self, newest_stamp):
        traj = self.odom.trajectory()
        if not len(traj):
            return
        ms = np.mean(self.reg_ms[-10:]) if self.reg_ms else 0.0
        budget = 1000 * self.frame_duration
        lag = (newest_stamp - self.last_t_end) if (newest_stamp and self.last_t_end) else 0.0
        x, y, z = traj[-1]
        warn = "  ** BEHIND" if lag > 3 * self.frame_duration else ""
        print(f"\rx={x:+6.2f} y={y:+6.2f} z={z:+5.2f} m  path {self.odom.path_length():6.2f} m  "
              f"{ms:4.0f}/{budget:.0f} ms/frame  lag {lag:4.2f} s  "
              f"map {self.maps[self.shown].n:,} pts{warn}   ", end="", flush=True)

    # -- loop -------------------------------------------------------------------

    def run(self):
        redraw_every = 1.0 / self.args.redraw_hz
        next_redraw = next_status = time.monotonic()
        newest = None
        try:
            while True:
                scans = self.stream.scans.drain()
                imu = self.stream.imu.latest_n(500)
                if scans:
                    newest = scans[-1].stamp
                    self.ingest(scans, imu)
                now = time.monotonic()
                if now >= next_redraw:
                    self.redraw()
                    next_redraw = now + redraw_every
                if now >= next_status:
                    self.status(newest)
                    next_status = now + 0.5
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                if not scans:
                    time.sleep(0.005)
        except KeyboardInterrupt:
            pass
        finally:
            print()
            self.vis.destroy_window()
            self.finish()

    def finish(self):
        traj = self.odom.trajectory()
        print(f"{len(traj)} poses, path {self.odom.path_length():.2f} m, "
              f"net {np.linalg.norm(traj[-1] - traj[0]) if len(traj) >= 2 else 0.0:.2f} m")
        if not self.args.save:
            return
        Path(self.args.save).parent.mkdir(parents=True, exist_ok=True)
        for name, m in self.maps.items():
            pts, rgb = m.points()
            if not len(pts):
                continue
            f = f"{self.args.save}_map_{name}.ply"
            write_ply(f, pts, rgb)
            print(f"saved      {f}  ({len(pts):,} points)")
        np.save(f"{self.args.save}_traj.npy", traj)
        print(f"saved      {self.args.save}_traj.npy   "
              f"view: python examples/09_map_offline.py --load {self.args.save}")


def main(argv=None):
    args = parse_args(argv)
    try:
        import open3d  # noqa: F401
    except ImportError as exc:
        raise SystemExit("needs open3d:  pip install 'l1-stream[viz]'") from exc

    with LidarStream.for_history(2.0, port=args.port) as lidar:
        if not lidar.wait_until_ready(5.0):
            print(f"No scan+IMU data on port {args.port} within 5 s. Is the publisher "
                  "running and sending to this machine?")
        LiveMapViewer(args, lidar).run()


if __name__ == "__main__":
    main()
