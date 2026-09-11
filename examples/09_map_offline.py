"""Look at the map a recording builds: KISS-ICP poses, optional map deskew, Open3D.

    python examples/09_map_offline.py personal/B1_room_5m_slow.l1raw
    python examples/09_map_offline.py personal/B1_room_5m_slow.l1raw --no-map-deskew
    python examples/09_map_offline.py personal/B1_room_5m_slow.l1raw --compare

No screen on the Orin? Build the map there, look at it anywhere:

    python examples/09_map_offline.py personal/B1_room_5m_slow.l1raw --save personal/B1 --no-show
    python examples/09_map_offline.py --load personal/B1      # needs only numpy + open3d

WHERE THE POSES COME FROM. The same replay() call 06_odometry_offline makes,
with the same config flags, so a map that looks wrong means the odometry is
wrong in the same way.

TWO DIFFERENT DESKEWS, on purpose:

    --deskew / --no-deskew            what REGISTRATION uses. Changes the poses.
    --map-deskew / --no-map-deskew    what the MAP uses. Poses unchanged; each
                                      frame's points are un-smeared with the
                                      velocity KISS-ICP estimated. Default ON.

--compare replays twice, map deskew ON and OFF, and M flips between them.
Smear shows up as thick or doubled walls; it is easiest to see from above.

KEYS   M  map deskew ON / OFF (with --compare)     K  trajectory on / off
       +/-  point size    R  reset view    H  Open3D help    Q  quit
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

try:
    from l1_stream.offline import add_args, config_from_args, replay
except ImportError:            # a laptop that only opens saved maps
    add_args = config_from_args = replay = None

LABEL = {"deskew": "map deskew ON", "raw": "map deskew OFF"}

# viridis at 0, .25, .5, .75, 1 -- colour by height without needing matplotlib
VIRIDIS = np.array([[0.267, 0.005, 0.329], [0.230, 0.322, 0.546], [0.128, 0.567, 0.551],
                    [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]])

PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
PLY_HEADER = ("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("path", nargs="?", help="recording to replay (omit with --load)")
    if add_args is not None:
        add_args(p)
    p.add_argument("--map-deskew", action=argparse.BooleanOptionalAction, default=True,
                   help="Deskew the MAP with KISS-ICP's velocity (poses are unaffected).")
    p.add_argument("--compare", action="store_true",
                   help="Build the map with map deskew ON and OFF; press M to flip.")
    p.add_argument("--map-voxel", type=float, default=0.03,
                   help="Map downsample, m. 0.03 matches 12_map_quality, so what you "
                        "see is what gets measured.")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--save", metavar="PREFIX", default=None,
                   help="Write PREFIX_map_deskew.ply / PREFIX_map_raw.ply and "
                        "PREFIX_traj.npy. No Open3D needed for this.")
    p.add_argument("--load", metavar="PREFIX", default=None,
                   help="Open maps saved with --save instead of replaying.")
    p.add_argument("--no-show", action="store_true", help="Do not open a window.")
    return p, p.parse_args(argv)


# --- colour and files --------------------------------------------------------

def height_colors(xyz: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((xyz[:, 2] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    stops = np.linspace(0.0, 1.0, len(VIRIDIS))
    return np.stack([np.interp(t, stops, VIRIDIS[:, c]) for c in range(3)], axis=1)


def write_ply(path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    arr = np.empty(len(xyz), dtype=PLY_DTYPE)
    for i, k in enumerate("xyz"):
        arr[k] = xyz[:, i]
    c = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    for i, k in enumerate(("red", "green", "blue")):
        arr[k] = c[:, i]
    with open(path, "wb") as f:
        f.write(PLY_HEADER.format(n=len(arr)).encode("ascii"))
        arr.tofile(f)


def read_ply(path) -> np.ndarray:
    """Reads the PLYs write_ply makes (not arbitrary PLY files). Returns xyz."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no PLY header")
            header += line
        m = re.search(rb"element vertex (\d+)", header)
        if m is None or header != PLY_HEADER.format(n=int(m.group(1))).encode("ascii"):
            raise ValueError(f"{path}: not a PLY written by this script")
        arr = np.fromfile(f, dtype=PLY_DTYPE, count=int(m.group(1)))
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


def load_saved(prefix):
    maps = {}
    for name in ("deskew", "raw"):
        f = Path(f"{prefix}_map_{name}.ply")
        if f.exists():
            maps[name] = read_ply(f)
            print(f"loaded     {f}  {len(maps[name]):,} points  ({LABEL[name]})")
    if not maps:
        raise SystemExit(f"No {prefix}_map_deskew.ply or {prefix}_map_raw.ply found.")
    traj_file = Path(f"{prefix}_traj.npy")
    traj = np.load(traj_file) if traj_file.exists() else np.zeros((0, 3))
    return maps, traj


# --- build -------------------------------------------------------------------

def build_maps(args, parser):
    if replay is None:
        raise SystemExit("l1_stream is not importable here. Replay on the Orin with "
                         "--save, then open it here with --load.")
    if not args.path:
        parser.error("give a recording to replay, or --load PREFIX")
    if args.map_voxel <= 0:
        parser.error("--map-voxel must be > 0")

    cfg = config_from_args(args)
    print("config     " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
    maps, traj = {}, None
    for md in ([True, False] if args.compare else [args.map_deskew]):
        t0 = time.perf_counter()
        run = replay(args.path, map_voxel=args.map_voxel, map_deskew=md, **cfg)
        name = "deskew" if md else "raw"
        maps[name] = run.world_points
        print(f"replayed   {LABEL[name]:15}  {len(run.world_points):,} map points  "
              f"({time.perf_counter() - t0:.1f} s)")
        if traj is None:
            traj = run.xyz
            print(f"trajectory {len(traj)} poses   net {run.net_displacement:.3f} m   "
                  f"path {run.path_length:.3f} m")
    return maps, traj


def save(prefix, maps, traj):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    lo, hi = np.percentile(next(iter(maps.values()))[:, 2], [2, 98])
    for name, xyz in maps.items():
        f = f"{prefix}_map_{name}.ply"
        write_ply(f, xyz, height_colors(xyz, lo, hi))
        print(f"saved      {f}")
    np.save(f"{prefix}_traj.npy", traj)
    print(f"saved      {prefix}_traj.npy")


# --- show --------------------------------------------------------------------

def show(maps, traj, *, point_size=2.0, title="map"):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Showing needs open3d (pip install 'l1-stream[viz]'). Or --save "
                         "here and --load on a machine that has it.") from exc

    names = list(maps)
    lo, hi = np.percentile(maps[names[0]][:, 2], [2, 98])     # shared, so M does not recolour
    clouds = {}
    for name in names:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(maps[name], dtype=np.float64))
        pc.colors = o3d.utility.Vector3dVector(height_colors(maps[name], lo, hi))
        clouds[name] = pc

    extras = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]  # start pose
    line = None
    if len(traj) >= 2:
        idx = np.arange(len(traj) - 1)
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(traj, dtype=np.float64)),
            lines=o3d.utility.Vector2iVector(np.stack([idx, idx + 1], axis=1)))
        line.paint_uniform_color([1.0, 0.0, 0.0])
        extras.append(line)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name=f"{title} -- {LABEL[names[0]]}",
                             width=1400, height=900):
        raise SystemExit("Open3D could not open a window (no display?). Run with "
                         "--save PREFIX --no-show here and --load PREFIX where there is "
                         "a screen.")
    vis.add_geometry(clouds[names[0]])
    for g in extras:
        vis.add_geometry(g)
    vis.get_render_option().point_size = point_size

    state = {"map": 0, "traj": True}

    def flip_map(v):
        if len(names) < 2:
            print("only one map -- run with --compare to flip map deskew")
            return False
        v.remove_geometry(clouds[names[state["map"]]], reset_bounding_box=False)
        state["map"] = 1 - state["map"]
        v.add_geometry(clouds[names[state["map"]]], reset_bounding_box=False)
        print(f"showing    {LABEL[names[state['map']]]}")
        return True

    def flip_traj(v):
        if line is None:
            return False
        if state["traj"]:
            v.remove_geometry(line, reset_bounding_box=False)
        else:
            v.add_geometry(line, reset_bounding_box=False)
        state["traj"] = not state["traj"]
        return True

    vis.register_key_callback(ord("M"), flip_map)
    vis.register_key_callback(ord("K"), flip_traj)
    print(f"showing    {LABEL[names[0]]}   keys: M map deskew  K trajectory  "
          "+/- point size  R reset  Q quit")
    vis.run()
    vis.destroy_window()


def main(argv=None):
    parser, args = parse_args(argv)
    if args.load:
        maps, traj = load_saved(args.load)
        title = Path(args.load).name
    else:
        maps, traj = build_maps(args, parser)
        title = Path(args.path).name
        if args.save:
            save(args.save, maps, traj)
    if not args.no_show:
        show(maps, traj, point_size=args.point_size, title=title)


if __name__ == "__main__":
    main()
