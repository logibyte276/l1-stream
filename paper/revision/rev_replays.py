"""Every replay the revision needs, in one unattended batch. Run on the Orin.

    python revision/rev_replays.py --recordings ~/recordings --out ~/revision_out

Groups (all by default; pick with --groups):

  imu       one IMU/clock report per D5 room recording           (M1 Methods facts)
  baseline  all 35 line drives at the frozen config, trajectories saved
            (reproduction check, R1 pose traces, minor 5 end-pose components)
  ablation  D5a-e x IMU {off, gravity, full} x deskew {on, off}  (M1, 30 replays)
  framelen  D5a-e x frame length {0.10, 0.12, 0.15, 0.18} s, plus
            0.10 s with deskew off                                  (R2, 25 replays)

Everything else stays at l1_stream.config.DEFAULTS, the paper's frozen config.

Outputs in --out:
  rev_results.csv   one row per replay, same columns as your main_results.csv
                    (l1_stream.results.run_row) plus group/condition/end-pose columns
  imu_report.csv    one row per recording
  traj/<group>/<run>__<condition>.npz   positions, stamps, per-frame deltas, ...

Safe to interrupt: re-running skips every replay already in rev_results.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

from l1_stream.config import DEFAULTS
from l1_stream.metadata import RecordingMeta
from l1_stream.results import ResultsLog, run_row

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_imu import MAX_VERTICAL_TILT_DEG, imu_report, replay_with_imu_mode  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")

ROOM = [f"D{d}{r}" for d in range(1, 7) for r in "abcde"]
CORRIDOR = [f"D5{r}_corridor" for r in "abcde"]
D5 = [f"D5{r}" for r in "abcde"]
FAIL_PCT = 20.0  # the paper's failure rule: |error| > 20 %


def build_jobs(groups: list[str]) -> list[dict]:
    """Every replay as a dict: group, run name, IMU mode, config overrides."""
    jobs = []
    if "baseline" in groups:
        for name in ROOM + CORRIDOR:
            jobs.append(dict(group="baseline", run=name, imu="full", cfg={}, cond="frozen"))
    if "ablation" in groups:
        for name in D5:
            for imu in ("off", "gravity", "full"):
                for deskew in (True, False):
                    cond = f"imu={imu},deskew={'on' if deskew else 'off'}"
                    jobs.append(dict(group="ablation", run=name, imu=imu,
                                     cfg={"deskew": deskew}, cond=cond))
    if "framelen" in groups:
        for name in D5:
            for fl in (0.10, 0.12, 0.15, 0.18):
                jobs.append(dict(group="framelen", run=name, imu="full",
                                 cfg={"frame_duration": fl}, cond=f"frame={fl:.2f}"))
            # Is deskew what breaks 0.1 s frames? Same frames, deskew off.
            jobs.append(dict(group="framelen", run=name, imu="full",
                             cfg={"frame_duration": 0.10, "deskew": False},
                             cond="frame=0.10,deskew=off"))
    for j in jobs:
        j["job"] = f"{j['group']}|{j['run']}|{j['cond']}"
    return jobs


def find_recordings(root: Path, names: set[str]) -> dict[str, Path]:
    """Map run name -> .l1raw path, matching the file stem case-insensitively."""
    by_stem = {}
    for p in root.rglob("*.l1raw"):
        by_stem.setdefault(p.stem.lower(), []).append(p)
    found, dupes = {}, []
    for name in names:
        hits = by_stem.get(name.lower(), [])
        if len(hits) > 1:
            dupes.append((name, hits))
        elif hits:
            found[name] = hits[0]
    if dupes:
        lines = "\n".join(f"  {n}: {[str(h) for h in hs]}" for n, hs in dupes)
        raise SystemExit(f"More than one recording per name -- point --recordings at "
                         f"the folder with the paper's set:\n{lines}")
    return found


def truth_for(name: str, meta) -> float:
    t = getattr(meta, "truth_m", None)
    if t is not None:
        return float(t)
    m = re.match(r"D(\d)", name)
    if not m:
        raise SystemExit(f"{name}: no truth in the sidecar and none in the name")
    return float(m.group(1))


def end_pose_components(xyz: np.ndarray) -> dict:
    """Split the end pose into along-track, sideways and vertical parts.

    Along-track = the car's starting heading: a straight-line fit (SVD) to
    the horizontal positions over the first half of the drive (from 0.1 m of
    travel), which averages out frame-to-frame jitter. Sideways = horizontal
    offset of the end pose from that line. Vertical = end z minus start z
    (z is vertical when frames are IMU-rotated).
    """
    if len(xyz) < 3:
        return {}
    rel = xyz - xyz[0]
    horiz = np.linalg.norm(rel[:, :2], axis=1)
    if horiz[-1] < 0.2:
        return {}
    early = rel[(horiz >= 0.1) & (horiz <= 0.5 * horiz[-1]), :2]
    if len(early) >= 3:
        _, _, vt = np.linalg.svd(early - early.mean(axis=0), full_matrices=False)
        heading = vt[0]
    else:
        heading = rel[-1, :2] / horiz[-1]
    if heading @ rel[-1, :2] < 0:
        heading = -heading
    side = np.array([-heading[1], heading[0]])
    end = rel[-1]
    return {
        "end_along_m": round(float(end[:2] @ heading), 4),
        "end_sideways_m": round(float(end[:2] @ side), 4),
        "end_vertical_m": round(float(end[2]), 4),
        "net_horizontal_m": round(float(np.linalg.norm(end[:2])), 5),
        "net_3d_m": round(float(np.linalg.norm(end)), 5),
    }


def save_traj(path: Path, run) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=run.xyz,
        stamps=run.stamps,
        deltas=np.asarray(run.deltas, dtype=np.float64),
        spans=run.spans,
        sizes=run.sizes,
        thresholds=run.thresholds,
        rotation_deg=run.rotation_deg,
        config=json.dumps(run.config),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--recordings", required=True, type=Path,
                   help="folder holding the paper's D-series .l1raw files (searched recursively)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--groups", nargs="+", default=["imu", "baseline", "ablation", "framelen"],
                   choices=["imu", "baseline", "ablation", "framelen"])
    p.add_argument("--dry-run", action="store_true", help="list what would run, then stop")
    args = p.parse_args()

    jobs = build_jobs(args.groups)
    needed = {j["run"] for j in jobs} | (set(D5) if "imu" in args.groups else set())
    recs = find_recordings(args.recordings, needed)
    missing = sorted(needed - set(recs))
    if missing:
        print(f"Not found under {args.recordings}: {', '.join(missing)}")
        if any(m in D5 for m in missing):
            raise SystemExit("D5a-e are required. Fix --recordings and re-run.")
        jobs = [j for j in jobs if j["run"] in recs]

    args.out.mkdir(parents=True, exist_ok=True)

    # --- IMU / clock report, and the gate for gravity-only mode ---------------
    if "imu" in args.groups or any(j["imu"] == "gravity" for j in jobs):
        rep_path = args.out / "imu_report.csv"
        rep_log = ResultsLog(rep_path)
        done = {r["recording"] for r in rep_log.rows()}
        reports = {}
        for name in D5:
            if name not in recs:
                continue
            rep = imu_report(str(recs[name]))
            reports[name] = rep
            if name not in done:
                rep_log.append({"recording": name, **rep})
        worst = max(r["vertical_tilt_from_z_deg"] for r in reports.values())
        print(f"IMU check: vertical is {worst:.2f} deg from z at worst "
              f"(gravity-only needs < {MAX_VERTICAL_TILT_DEG}).  -> {rep_path}")
        if worst >= MAX_VERTICAL_TILT_DEG:
            print("  The IMU frame's z axis is not vertical, so yaw removal about z is "
                  "not meaningful. Skipping gravity-only replays.")
            jobs = [j for j in jobs if j["imu"] != "gravity"]

    log = ResultsLog(args.out / "rev_results.csv")
    done = {r.get("job") for r in log.rows()}
    todo = [j for j in jobs if j["job"] not in done]
    print(f"{len(jobs)} replays planned, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run. Config: frozen DEFAULTS except the swept setting.")
    if args.dry_run:
        for j in todo:
            print(f"  {j['job']}")
        return

    t_start = time.monotonic()
    for i, j in enumerate(todo, 1):
        path = recs[j["run"]]
        meta = RecordingMeta.load_or_none(path)
        truth = truth_for(j["run"], meta)
        run = replay_with_imu_mode(str(path), j["imu"], **j["cfg"])

        err = 100 * (run.net_displacement - truth) / truth
        row = run_row(
            path, run, meta, truth=truth,
            job=j["job"], group=j["group"], run_name=j["run"], condition=j["cond"],
            imu_mode=j["imu"], failed=bool(abs(err) > FAIL_PCT),
            **end_pose_components(run.xyz),
        )
        log.append(row)
        save_traj(args.out / "traj" / j["group"] / f"{j['run']}__{j['cond']}.npz", run)

        elapsed = time.monotonic() - t_start
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i:3d}/{len(todo)}] {j['job']:42s} net {run.net_displacement:.4f} m  "
              f"{err:+7.2f}%  {run.budget_pct:5.1f}% budget   eta {eta/60:4.1f} min",
              flush=True)

    print(f"\nDone. Results -> {args.out / 'rev_results.csv'}")
    print(f"Copy {args.out} to the PC and run rev_analysis.py.")
    unused = sorted(set(DEFAULTS) - {"frame_duration", "deskew", "rotate_with_imu"})
    print(f"(Held at DEFAULTS: {', '.join(f'{k}={DEFAULTS[k]}' for k in unused)})")


if __name__ == "__main__":
    main()
