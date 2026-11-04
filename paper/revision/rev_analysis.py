"""Every number, table and figure the revision needs. Run on the PC.

    python revision/rev_analysis.py --rev revision_out \
        --main personal/main_tests/main_results.csv \
        --sweep personal/main_tests/voxel_frame_results.csv \
        --wall wall_results.csv --exclude D3a --out revision_analysis

--exclude drops runs whose recording on disk is not the paper's (D3a: the
original file was lost; the copy on disk is a 1 m drive). They are left out of
every replay-based section (R1, minor 5); the paper's own CSVs are untouched.

Writes to --out:
  summary.md          every result below, as tables (read this one)
  table_*.csv         the same tables as CSV, for building the paper's tables
  fig_R1_traces.png   pose traces: along-track position vs time, and the stop
  fig_R2_frames.png   0.10 s failure vs the shorter tested frame lengths

Sections: M1 ablation, M1 wall check, M1 IMU/clock facts,
M3 corridor statistics, R1 where the +2.79 cm constant comes from,
R2 frame-length margin, minor 4 spread across recordings, minor 5 end pose.

Needs numpy, pandas, scipy, statsmodels, matplotlib -- the paper's stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FAIL_PCT = 20.0
STILL = 0.15          # m/s: moving/parked cutoff (same value as 06_odometry_offline --still)
HOLD_MARGIN = 1.0     # s kept clear of the detected start/stop when measuring holds
SPEED_WINDOW = 1.0    # s: speed for start/stop detection is averaged over this window
HOLD_AVG = 1.0        # s: each end of a hold is a mean over this long (sensitivity check)


# --- loading -----------------------------------------------------------------

def _name(rec: str) -> str:
    return Path(str(rec)).name.replace(".l1raw", "")


def _err_pct(df: pd.DataFrame) -> pd.Series:
    if "scale_error_pct" in df and df["scale_error_pct"].notna().all():
        return df["scale_error_pct"].astype(float)
    truth = df["truth_used_m"] if "truth_used_m" in df else df["truth_m"]
    return 100 * (df["net_displacement_m"].astype(float) - truth.astype(float)) / truth.astype(float)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "recording" not in df or "net_displacement_m" not in df:
        raise SystemExit(f"{path}: expected the l1_stream results columns "
                         f"'recording' and 'net_displacement_m'; found {list(df.columns)}")
    df["name"] = df["recording"].map(_name)
    if "run_name" in df:                      # rev_results.csv
        df["run"] = df["run_name"]
    df["err_pct"] = _err_pct(df)
    truth = df["truth_used_m"] if "truth_used_m" in df else df["truth_m"]
    df["truth"] = truth.astype(float)
    df["err_cm"] = 100 * (df["net_displacement_m"].astype(float) - df["truth"])
    df["is_corridor"] = df["name"].str.lower().str.contains("corridor") | (
        df.get("environment", pd.Series("", index=df.index)).fillna("")
        .astype(str).str.contains("corridor"))
    return df


def load_traj(rev: Path, group: str, run: str, cond: str) -> dict | None:
    f = rev / "traj" / group / f"{run}__{cond}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in z.files}


# --- small helpers -------------------------------------------------------------

def msd(x, nd=2) -> str:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "-"
    if len(x) == 1:
        return f"{x[0]:.{nd}f}"
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


class Report:
    def __init__(self, out: Path):
        self.out = out
        self.parts: list[str] = []
        out.mkdir(parents=True, exist_ok=True)

    def h(self, title: str) -> None:
        self.parts.append(f"\n## {title}\n")
        print(f"\n== {title}")

    def line(self, text: str, echo: bool = True) -> None:
        self.parts.append(text + "\n")
        if echo:
            print("  " + text)

    def table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.out / f"table_{name}.csv", index=False)
        self.parts.append(md_table(df) + f"\n\n(also `table_{name}.csv`)\n")

    def save(self) -> Path:
        p = self.out / "summary.md"
        p.write_text("# Revision analysis\n" + "".join(self.parts))
        return p


# --- sections ------------------------------------------------------------------

def ablation(rep: Report, rev: pd.DataFrame) -> None:
    rep.h("M1 ablation: IMU pre-rotation x deskew (D5a-e, 5 m room)")
    a = rev[rev["group"] == "ablation"].copy()
    if a.empty:
        rep.line("No ablation rows.")
        return
    a["failed"] = a["err_pct"].abs() > FAIL_PCT
    ref = a[a["condition"] == "imu=full,deskew=on"].set_index("run")["err_pct"]
    rows = []
    order = [f"imu={i},deskew={d}" for i in ("off", "gravity", "full") for d in ("on", "off")]
    for cond in order:
        g = a[a["condition"] == cond]
        if g.empty:
            continue
        ok = g[~g["failed"]]
        row = {"IMU pre-rotation": cond.split(",")[0].split("=")[1],
               "deskew": cond.split(",")[1].split("=")[1],
               "worked": f"{len(ok)}/{len(g)}",
               "error % (mean ± SD)": msd(ok["err_pct"]),
               "error cm (mean ± SD)": msd(ok["err_cm"], 1),
               "rotation °/frame": msd(g["rotation_deg_mean"].astype(float), 2)}
        if cond != "imu=full,deskew=on" and len(ok) >= 2:
            paired = ok.set_index("run")["err_pct"].align(ref, join="inner")
            d = paired[0] - paired[1]
            if len(d) >= 2:
                t = stats.ttest_rel(paired[0], paired[1])
                ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                row["vs baseline (pp, 95% CI)"] = (f"{d.mean():+.2f} "
                                                  f"({d.mean()-ci:+.2f} to {d.mean()+ci:+.2f})")
                row["paired p"] = f"{t.pvalue:.3f}"
        else:
            row["vs baseline (pp, 95% CI)"] = "baseline" if cond == "imu=full,deskew=on" else "-"
            row["paired p"] = "-"
        rows.append(row)
    rep.table(pd.DataFrame(rows), "M1_ablation")
    rep.line("pp = percentage points, paired by recording (n = 5). Failed = |error| > 20 %.",
             echo=False)

    ok = a[~a["failed"]]
    full_grid = ok.groupby("condition").size()
    if len(full_grid) == 6 and (full_grid == 5).all():
        try:
            import statsmodels.formula.api as smf
            from statsmodels.stats.anova import anova_lm
            ok = ok.assign(imu=ok["condition"].str.extract(r"imu=(\w+)")[0],
                           deskew=ok["condition"].str.extract(r"deskew=(\w+)")[0])
            fit = smf.ols("err_pct ~ C(imu) * C(deskew) + C(run)", data=ok).fit()
            tab = anova_lm(fit, typ=2)
            for term, label in [("C(imu)", "IMU mode"), ("C(deskew)", "deskew"),
                                ("C(imu):C(deskew)", "interaction"), ("C(run)", "recording")]:
                r = tab.loc[term]
                rep.line(f"ANOVA {label}: F({int(r['df'])}, {int(tab.loc['Residual', 'df'])}) = "
                         f"{r['F']:.2f}, p = {r['PR(>F)']:.3f}")
        except ImportError:
            rep.line("statsmodels not installed: ANOVA skipped.")
    else:
        rep.line("Some conditions had failed replays, so no two-way ANOVA "
                 "(same exclusion rule as the paper); use the paired comparisons.")


def wall(rep: Report, path: Path | None) -> None:
    rep.h("M1 wall check (raw L1 ranges)")
    if path is None or not path.exists():
        rep.line("No wall_results.csv given (run rev_wall_check.py --log first).")
        return
    w = pd.read_csv(path)
    block = (w["fit_slope"] != w["fit_slope"].shift()).cumsum()
    last = w[block == block.iloc[-1]]          # rows of the most recent fit
    s = last["fit_slope"].iloc[-1]
    lo, hi = last["fit_slope_ci_lo"].iloc[-1], last["fit_slope_ci_hi"].iloc[-1]
    rep.table(last[["recording", "bosch_m", "l1_plane_m", "resid_mm",
                    "plane_rms_mm", "inliers"]].reset_index(drop=True), "M1_wall")
    rep.line(f"range scale s = {s:.4f} (95% CI {lo:.4f} to {hi:.4f}) = "
             f"{100*(s-1):+.2f}% range error; offset c = {last['fit_offset_m'].iloc[-1]:.3f} m")
    need = 1 - 0.0452
    rep.line(f"ranges alone would explain the odometry slope at s = {need:.4f}: "
             f"{'inside' if lo <= need <= hi else 'outside'} the CI; "
             f"s = 1 is {'inside' if lo <= 1 <= hi else 'outside'} the CI")


def imu_facts(rep: Report, path: Path) -> None:
    rep.h("M1 Methods facts: IMU and clocks (D5a-e)")
    if not path.exists():
        rep.line("No imu_report.csv.")
        return
    r = pd.read_csv(path)

    def rng(col, nd=2):
        v = r[col].astype(float)
        return f"{v.min():.{nd}f}" if v.min() == v.max() else f"{v.min():.{nd}f}–{v.max():.{nd}f}"

    rep.line(f"IMU rate {rng('imu_rate_hz', 1)} Hz; scan packets {rng('scan_rate_hz', 1)} /s")
    rep.line(f"measured vertical vs IMU-frame z: {rng('vertical_tilt_from_z_deg')}° "
             f"({', '.join(sorted(set(r['vertical_sign'])))}); |g| {rng('gravity_mps2')}")
    rep.line(f"IMU yaw change over a recording: {rng('yaw_change_deg')}° "
             f"(range within a recording {rng('yaw_range_deg')}°)")
    rep.line(f"scan-to-nearest-IMU gap: median {rng('scan_imu_gap_median_ms', 2)} ms, "
             f"max {rng('scan_imu_gap_max_ms', 2)} ms")
    unix = r["stamps_look_like_unix_time"].astype(str).str.lower().eq("true").all()
    rep.line(f"stamps look like host Unix time: {'yes' if unix else 'no'} "
             f"(first stamp {r['stamp_first_s'].iloc[0]:.0f} s)")
    rep.line(f"per-point time offset within a packet: up to {rng('point_time_offset_max_ms')} ms")


def corridor(rep: Report, main: pd.DataFrame) -> None:
    rep.h("M3 corridor vs room (5 m)")
    room = main[(~main["is_corridor"]) & (main["truth"] == 5.0)]["err_cm"].to_numpy()
    corr = main[main["is_corridor"]]["err_cm"].to_numpy()
    if len(room) < 2 or len(corr) < 2:
        rep.line("Could not find 5 m room and corridor rows in main_results.csv.")
        return
    w = stats.ttest_ind(room, corr, equal_var=False)
    va, vb = room.var(ddof=1) / len(room), corr.var(ddof=1) / len(corr)
    df = (va + vb) ** 2 / (va**2 / (len(room) - 1) + vb**2 / (len(corr) - 1))
    diff, se = room.mean() - corr.mean(), np.sqrt(va + vb)
    tc = stats.t.ppf(0.975, df)
    f = corr.var(ddof=1) / room.var(ddof=1)
    d1, d2 = len(corr) - 1, len(room) - 1
    p_f = 2 * min(stats.f.sf(f, d1, d2), stats.f.cdf(f, d1, d2))
    bf = stats.levene(corr, room, center="median")
    f_lo, f_hi = f / stats.f.ppf(0.975, d1, d2), f / stats.f.ppf(0.025, d1, d2)
    rep.line(f"room {msd(room)} cm, corridor {msd(corr)} cm (n = {len(room)}, {len(corr)})")
    rep.line(f"Welch t({df:.2f}) = {w.statistic:.2f}, p = {w.pvalue:.2f}; "
             f"mean difference room − corridor {diff:.2f} cm, 95% CI {diff-tc*se:.1f} to "
             f"{diff+tc*se:.1f} cm")
    rep.line(f"F-test on variances: F({d1}, {d2}) = {f:.1f}, two-sided p = {p_f:.3f}")
    rep.line(f"Brown–Forsythe: W = {bf.statistic:.2f}, p = {bf.pvalue:.3f}")
    rep.line(f"SD ratio {np.sqrt(f):.1f}× (95% CI {np.sqrt(f_lo):.1f}× to {np.sqrt(f_hi):.1f}×)")


def drive_segment(t: np.ndarray, xyz: np.ndarray) -> tuple[int, int] | None:
    """Indices of the first and last pose of the drive itself.

    Speed is taken over a SPEED_WINDOW-long centred window, not frame to frame:
    parked frames jitter by 2-3 cm, which at 0.2 s frames is 0.10-0.15 m/s and
    would cross STILL on its own. The drive is the longest unbroken stretch
    above STILL, so a stray jitter spike in a hold is never counted as motion.
    """
    half = SPEED_WINDOW / 2
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1
    span = t[hi] - t[lo]
    dist = np.linalg.norm(xyz[hi] - xyz[lo], axis=1)
    speed = np.where(span > 0, dist / np.where(span > 0, span, 1), 0.0)
    moving = speed > STILL
    if not moving.any():
        return None
    # longest unbroken run of moving poses
    edges = np.flatnonzero(np.diff(np.r_[0, moving.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def hold_decomposition(t: np.ndarray, xyz: np.ndarray) -> dict | None:
    """Split a line drive into start hold, motion and end hold, along-track.

    Holds are measured only up to HOLD_MARGIN s before the detected start and
    from HOLD_MARGIN s after the detected stop, so slow first/last movement
    is counted as motion, never as hold drift.
    """
    if len(t) < 5:
        return None
    seg = drive_segment(t, xyz)
    if seg is None:
        return None
    head, tail = seg
    t_on, t_off = t[head], t[tail]
    pre = int(np.searchsorted(t, t_on - HOLD_MARGIN, side="right") - 1)
    post = int(np.searchsorted(t, t_off + HOLD_MARGIN, side="left"))
    pre, post = max(pre, 0), min(post, len(t) - 1)
    heading = xyz[post, :2] - xyz[pre, :2]
    heading = heading / np.linalg.norm(heading)
    along = (xyz[:, :2] - xyz[0, :2]) @ heading
    return {
        "t_on": t_on, "t_off": t_off, "along": along,
        "start_hold_cm": 100 * (along[pre] - along[0]),
        "motion_m": along[post] - along[pre],
        "end_hold_cm": 100 * (along[-1] - along[post]),
        "overshoot_cm": 100 * (along[head:].max() - along[-1]),
        "total_m": along[-1] - along[0],
        "start_hold_s": t_on - HOLD_MARGIN - t[0],
        "end_hold_s": t[-1] - (t_off + HOLD_MARGIN),
    }


def _fit(x, y) -> str:
    r = stats.linregress(x, y)
    tc = stats.t.ppf(0.975, len(x) - 2)
    return (f"slope {r.slope:.2f} ± {r.stderr:.2f} cm/m, constant {r.intercept:+.2f} ± "
            f"{r.intercept_stderr:.2f} cm (95% CI {r.intercept - tc*r.intercept_stderr:+.2f} to "
            f"{r.intercept + tc*r.intercept_stderr:+.2f})")


def constant_source(rep: Report, rev_dir: Path, rev: pd.DataFrame, figs: dict) -> None:
    rep.h("R1 where the +2.79 cm constant comes from (room runs, pose traces)")
    base = rev[(rev["group"] == "baseline") & (~rev["is_corridor"])]
    rows, traces = [], {}
    for _, r in base.iterrows():
        z = load_traj(rev_dir, "baseline", r["run"], "frozen")
        if z is None:
            continue
        d = hold_decomposition(z["stamps"], z["xyz"])
        if d is None:
            continue
        traces[r["run"]] = (z["stamps"], d)
        rows.append({"run": r["run"], "truth_m": r["truth"],
                     "start_hold_cm": d["start_hold_cm"], "end_hold_cm": d["end_hold_cm"],
                     "overshoot_cm": d["overshoot_cm"],
                     "motion_err_cm": 100 * (d["motion_m"] - r["truth"]),
                     "total_err_cm": 100 * (d["total_m"] - r["truth"]),
                     "start_hold_s": d["start_hold_s"], "end_hold_s": d["end_hold_s"]})
    if not rows:
        rep.line("No baseline trajectories found.")
        return
    df = pd.DataFrame(rows)
    rep.line(f"{len(df)} room runs with trajectories")
    by_d = df.groupby("truth_m").agg(
        n=("run", "size"),
        start_hold_cm=("start_hold_cm", lambda v: msd(v)),
        end_hold_cm=("end_hold_cm", lambda v: msd(v)),
        overshoot_cm=("overshoot_cm", lambda v: msd(v)),
        motion_err_cm=("motion_err_cm", lambda v: msd(v)),
        total_err_cm=("total_err_cm", lambda v: msd(v)),
    ).reset_index()
    rep.table(by_d, "R1_hold_decomposition")
    df.round(3).to_csv(rep.out / "table_R1_per_run.csv", index=False)
    rep.line(f"total error vs distance:        {_fit(df['truth_m'], df['total_err_cm'])}")
    rep.line(f"motion-only error vs distance:  {_fit(df['truth_m'], df['motion_err_cm'])}")
    rep.line(f"hold drift per run (start + end): {msd(df['start_hold_cm'] + df['end_hold_cm'])} cm; "
             f"overshoot at the stop: {msd(df['overshoot_cm'])} cm")
    rep.line("If the motion-only constant is still about +2.8 cm, the constant is made while "
             "moving (start/stop transients), not during the parked holds. Look at "
             "fig_R1_traces.png to see whether it appears at the start or the stop.", echo=False)
    hold_sensitivity(rep, df, traces)
    figs["traces"] = traces


def _mean_ci(v) -> str:
    v = np.asarray(v, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    tc = stats.t.ppf(0.975, len(v) - 1)
    p = stats.ttest_1samp(v, 0).pvalue
    return f"{v.mean():+.2f} (95% CI {v.mean() - tc*se:+.2f} to {v.mean() + tc*se:+.2f}, p = {p:.3f})"


def hold_sensitivity(rep: Report, df: pd.DataFrame, traces: dict) -> None:
    """Hold drift with each end of a hold taken as a 1 s mean, not one pose.

    The table above measures each hold between two single poses, so one pose's
    jitter moves cm between "hold" and "motion" (they anti-correlate). Averaging
    over HOLD_AVG s removes most of that, and a 2 s margin checks whether the
    end hold still contains the stop transient.
    """
    rows = []
    for margin in (HOLD_MARGIN, 2.0):
        out = []
        for _, r in df.iterrows():
            t, d = traces[r["run"]]
            a = d["along"]

            def level(t0, t1, a=a, t=t):
                m = (t >= t0) & (t <= t1)
                return a[m].mean() if m.sum() >= 2 else np.nan

            t_on, t_off = d["t_on"], d["t_off"]
            s0, s1 = level(t[0], t[0] + HOLD_AVG), level(t_on - margin - HOLD_AVG, t_on - margin)
            e0, e1 = level(t_off + margin, t_off + margin + HOLD_AVG), level(t[-1] - HOLD_AVG, t[-1])
            out.append({"truth": r["truth_m"], "total": r["total_err_cm"],
                        "start": 100 * (s1 - s0), "end": 100 * (e1 - e0)})
        o = pd.DataFrame(out).dropna()
        o["both"] = o["start"] + o["end"]
        motion = o["total"] - o["both"]
        fit = stats.linregress(o["truth"], motion)
        tc = stats.t.ppf(0.975, len(o) - 2)
        rows.append(f"margin {margin:.0f} s, n = {len(o)}: start {_mean_ci(o['start'])}; "
                    f"end {_mean_ci(o['end'])}; both {_mean_ci(o['both'])}; "
                    f"motion-only constant {fit.intercept:+.2f} (95% CI "
                    f"{fit.intercept - tc*fit.intercept_stderr:+.2f} to "
                    f"{fit.intercept + tc*fit.intercept_stderr:+.2f})")
    rep.line(f"Hold drift with {HOLD_AVG:.0f} s-averaged ends (cm per run):")
    for line in rows:
        rep.line("  " + line)


def frame_lengths(rep: Report, rev_dir: Path, rev: pd.DataFrame, figs: dict) -> None:
    rep.h("R2 frame-length margin (D5a-e, voxel 0.15 m)")
    fl = rev[rev["group"] == "framelen"].copy()
    b = rev[(rev["group"] == "baseline") & (rev["run"].str.fullmatch(r"D5[a-e]"))].copy()
    b["condition"] = "frame=0.20"
    fl = pd.concat([fl, b])
    if fl.empty:
        rep.line("No frame-length rows.")
        return
    fl["frame_s"] = fl["condition"].str.extract(r"frame=([\d.]+)")[0].astype(float)
    fl["deskew"] = np.where(fl["condition"].str.contains("deskew=off"), "off", "on")
    fl["failed"] = fl["err_pct"].abs() > FAIL_PCT
    rows = []
    for (f, dk), g in fl.groupby(["frame_s", "deskew"]):
        ok = g[~g["failed"]]
        rows.append({"frame s": f"{f:.2f}", "deskew": dk, "worked": f"{len(ok)}/{len(g)}",
                     "error % (worked)": msd(ok["err_pct"]),
                     "failed runs": ", ".join(g[g["failed"]]["run"]) or "-",
                     "rotation °/frame": msd(g["rotation_deg_mean"].astype(float)),
                     "points/frame": f"{g['points_per_frame'].astype(float).mean():.0f}",
                     "budget %": f"{g['budget_pct'].astype(float).mean():.1f}"})
    tab = pd.DataFrame(rows)
    rep.table(tab, "R2_frame_length")
    for _, r in tab.iterrows():
        print(f"  {r['frame s']} s, deskew {r['deskew']}: {r['worked']} worked, "
              f"error {r['error % (worked)']} %")

    spins = []
    for _, r in fl[fl["failed"]].iterrows():
        group, cond = ("baseline", "frozen") if r["group"] == "baseline" else ("framelen",
                                                                                r["condition"])
        z = load_traj(rev_dir, group, r["run"], cond)
        if z is None:
            continue
        # Motion start from the same drive's working 0.2 s replay (same clock).
        ref = load_traj(rev_dir, "baseline", r["run"], "frozen")
        dec = hold_decomposition(ref["stamps"], ref["xyz"]) if ref is not None else None
        t_on = dec["t_on"] if dec else z["stamps"][0]
        rot = z["rotation_deg"]            # rot[k] = rotation estimated for frame k
        if (rot > 3.0).any():
            k = int(np.argmax(rot > 3.0))
            spins.append(f"{r['run']} {r['condition']}: rotation first exceeds 3°/frame at "
                         f"{z['stamps'][k] - t_on:+.1f} s from motion start "
                         f"(then {np.median(rot[k:]):.1f}°/frame median)")
        else:
            spins.append(f"{r['run']} {r['condition']}: failed without spinning "
                         f"(max {rot.max():.1f}°/frame)")
    for s in spins:
        rep.line(s)
    figs["frames"] = fl


def sweep_spread(rep: Report, sweep: pd.DataFrame | None) -> None:
    rep.h("Minor 4: spread across physical recordings in the voxel x frame sweep")
    if sweep is None:
        rep.line("No sweep CSV given.")
        return
    s = sweep.copy()
    fcol = "cfg.frame_duration" if "cfg.frame_duration" in s else "frame_duration"
    s = s[(s[fcol].astype(float) >= 0.2) & (s["err_pct"].abs() <= FAIL_PCT)]
    per = s.groupby("name")["err_pct"].agg(["mean", "std", "size"]).reset_index()
    per.columns = ["recording", "mean error %", "SD across settings", "replays"]
    rep.table(per.round(3), "minor4_per_recording")
    m = per["mean error %"]
    rep.line(f"{len(s)} replays of {len(per)} recordings: pooled {msd(s['err_pct'])} %; "
             f"across recordings {m.mean():.2f} ± {m.std(ddof=1):.2f} % (n = {len(per)})")


def end_pose(rep: Report, rev: pd.DataFrame) -> None:
    rep.h("Minor 5: end-pose components (baseline replays)")
    b = rev[rev["group"] == "baseline"].copy()
    if "end_sideways_m" not in b:
        rep.line("No end-pose columns.")
        return
    b["env"] = np.where(b["is_corridor"], "corridor", "room")
    b["sideways_cm"] = 100 * b["end_sideways_m"].astype(float)
    b["vertical_cm"] = 100 * b["end_vertical_m"].astype(float)
    net3d = b["net_3d_m"] if "net_3d_m" in b else b["net_displacement_m"]
    b["3d_minus_horiz_mm"] = 1000 * (net3d.astype(float) - b["net_horizontal_m"].astype(float))
    rows = []
    for (env, d), g in b.groupby(["env", "truth"]):
        rows.append({"environment": env, "distance m": d,
                     "sideways cm": msd(g["sideways_cm"]), "vertical cm": msd(g["vertical_cm"]),
                     "3D − horizontal mm": f"{g['3d_minus_horiz_mm'].max():.2f} max"})
    rep.table(pd.DataFrame(rows), "minor5_end_pose")
    rep.line("d_odom = 3D distance between the first and last pose (l1_stream "
             f"net_displacement). 3D and horizontal differ by at most "
             f"{b['3d_minus_horiz_mm'].max():.2f} mm.")


# --- figures ---------------------------------------------------------------------

def figures(out: Path, figs: dict, rev_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    traces = figs.get("traces")
    if traces:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
        cmap = plt.get_cmap("viridis")
        for d in range(1, 7):
            # first repeat of each distance that has a trace (D3a may be excluded)
            run = next((f"D{d}{r}" for r in "abcde" if f"D{d}{r}" in traces), None)
            if run is None:
                continue
            t, dec = traces[run]
            a1.plot(t - dec["t_on"], dec["along"], color=cmap((d - 1) / 5), lw=1.2,
                    label=f"{d} m")
        a1.set_xlabel("time from start of motion (s)")
        a1.set_ylabel("estimated along-track position (m)")
        a1.legend(frameon=False, fontsize=7, title="drive", title_fontsize=7)
        a1.set_title("(a) pose traces", loc="left", fontsize=9)
        for r in "abcde":
            run = f"D5{r}"
            if run not in traces:
                continue
            t, dec = traces[run]
            a2.plot(t - dec["t_off"], 100 * (dec["along"] - dec["along"][-1]), lw=1,
                    label=run)
        a2.axhline(0, color="0.6", lw=0.6)
        a2.set_xlim(-2, None)
        a2.set_ylim(-15, 10)
        a2.set_xlabel("time from end of motion (s)")
        a2.set_ylabel("position − final position (cm)")
        a2.legend(frameon=False, fontsize=7)
        a2.set_title("(b) the stop, 5 m drives", loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_R1_traces.png", dpi=300)
        plt.close(fig)
        made.append("fig_R1_traces.png")

    fl = figs.get("frames")
    if fl is not None and not fl.empty:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
        run = "D5a"
        # Deskew-off is not reported in the paper, so it is not drawn. 0.10 s is drawn
        # wide and underneath so it stays visible where the 0.12 s line spins with it.
        for cond, color, lw in [("frame=0.10", "#b03a2e", 3.0), ("frame=0.12", "#d4a017", 1.1),
                                ("frame=0.15", "#2e86c1", 1.1), ("frame=0.18", "#1f4e79", 1.1),
                                ("frame=0.20", "0.2", 1.1)]:
            group = "baseline" if cond == "frame=0.20" else "framelen"
            z = load_traj(rev_dir, group, run, "frozen" if cond == "frame=0.20" else cond)
            if z is None:
                continue
            t = z["stamps"] - z["stamps"][0]
            label = cond.replace("frame=", "") + " s"
            a1.plot(t[1:], np.cumsum(z["rotation_deg"][1:]), color=color, lw=lw, label=label)
            a2.plot(z["xyz"][:, 0] - z["xyz"][0, 0], z["xyz"][:, 1] - z["xyz"][0, 1],
                    color=color, lw=lw)
        a1.set_xlabel("time (s)")
        a1.set_ylabel("cumulative rotation (°)")
        a1.legend(frameon=False, fontsize=7, title=f"{run} frame length", title_fontsize=7)
        a1.set_title("(a) rotation", loc="left", fontsize=9)
        a2.set_aspect("equal", adjustable="datalim")
        a2.set_xlabel("x (m)")
        a2.set_ylabel("y (m)")
        a2.set_title("(b) estimated path", loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "fig_R2_frames.png", dpi=300)
        plt.close(fig)
        made.append("fig_R2_frames.png")
    return made


def main() -> None:
    p = argparse.ArgumentParser(description="Revision analysis: tables, stats, figures.")
    p.add_argument("--rev", required=True, type=Path, help="output folder of rev_replays.py")
    p.add_argument("--main", required=True, type=Path, help="main_results.csv (35 runs)")
    p.add_argument("--sweep", type=Path, default=None, help="voxel_frame_results.csv")
    p.add_argument("--wall", type=Path, default=None, help="wall_results.csv")
    p.add_argument("--out", type=Path, default=Path("revision_analysis"))
    p.add_argument("--exclude", nargs="*", default=[], metavar="RUN",
                   help="runs to leave out of the replay-based sections, e.g. D3a")
    args = p.parse_args()

    rev = load_csv(args.rev / "rev_results.csv")
    main_df = load_csv(args.main)
    sweep = load_csv(args.sweep) if args.sweep else None
    rep = Report(args.out)
    figs: dict = {}

    if args.exclude:
        unknown = sorted(set(args.exclude) - set(rev["run"]))
        if unknown:
            raise SystemExit(f"--exclude: no such run in rev_results.csv: {', '.join(unknown)}")
        rev = rev[~rev["run"].isin(args.exclude)].copy()
        rep.h("Excluded runs")
        rep.line(f"Left out of every replay-based section: {', '.join(args.exclude)} "
                 "(recording on disk is not the paper's).")

    ablation(rep, rev)
    wall(rep, args.wall)
    imu_facts(rep, args.rev / "imu_report.csv")
    corridor(rep, main_df)
    constant_source(rep, args.rev, rev, figs)
    frame_lengths(rep, args.rev, rev, figs)
    sweep_spread(rep, sweep)
    end_pose(rep, rev)
    made = figures(args.out, figs, args.rev)

    path = rep.save()
    (args.out / "inputs.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"\nsummary -> {path}")
    for m in made:
        print(f"figure  -> {args.out / m}")


if __name__ == "__main__":
    main()
