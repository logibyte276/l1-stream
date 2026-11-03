# Revision scripts (NHSJS review, KISS-ICP on the Unitree L1)

Standalone scripts for the reviewer's requests. They use the l1-stream v1.0.0
API and change nothing inside `src/l1_stream`, so v1.0.0 (Zenodo) still
reproduces every original number.

Put this folder at `l1-stream/paper/revision/` (Ruff passes there with the
repo's own config).

| Script | Runs on | Answers |
|---|---|---|
| `rev_replays.py` | Orin | M1 ablation, R1 pose traces, R2 frame lengths, minor 5, IMU/clock facts |
| `rev_wall_check.py` | Orin or PC | M1 static wall check (range scale) |
| `rev_analysis.py` | PC | every table, statistic and figure |
| `rev_imu.py` | (module) | gravity-only IMU mode, IMU/clock report |
| `test_rev.py` | anywhere | `pytest paper/revision/test_rev.py` |

```bash
# Orin: all replays (90, resumable)
python paper/revision/rev_replays.py --recordings <folder with D1a..D6e, D5a_corridor..> \
    --out ~/revision_out

# Orin: wall check (after recording W1..W5 with 05_record.py --kind stationary --truth <m>)
python paper/revision/rev_wall_check.py wall/W*.l1raw --log wall/wall_results.csv \
    --plot wall/wall_fit.png

# PC: everything else
python paper/revision/rev_analysis.py --rev revision_out \
    --main personal/main_tests/main_results.csv \
    --sweep personal/main_tests/voxel_frame_results.csv \
    --wall wall_results.csv --out revision_analysis
```

Recording names must match the paper's (`D5a.l1raw`, `D5a_corridor.l1raw`, ...);
check with `rev_replays.py ... --dry-run`.
