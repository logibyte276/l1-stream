from l1_stream.results import ResultsLog, flatten, run_row


def test_flatten_makes_nested_config_into_columns():
    out = flatten({"a": 1, "cfg": {"voxel_size": 0.15, "deskew": False}})
    assert out == {"a": 1, "cfg.voxel_size": 0.15, "cfg.deskew": False}


def test_appends_accumulate(tmp_path):
    log = ResultsLog(tmp_path / "r.csv")
    log.append({"run": 1, "loop_err": 0.03})
    log.append({"run": 2, "loop_err": 0.07})
    rows = log.rows()
    assert [r["run"] for r in rows] == ["1", "2"]


def test_a_new_metric_widens_the_table_instead_of_being_dropped(tmp_path):
    # The failure this guards against: adding plane_rms later and silently
    # losing it because the header was fixed at first write.
    log = ResultsLog(tmp_path / "r.csv")
    log.append({"run": 1, "loop_err": 0.03})
    log.append({"run": 2, "loop_err": 0.07, "plane_rms": 0.012})

    rows = log.rows()
    assert "plane_rms" in log.fieldnames()
    assert rows[0]["plane_rms"] == ""      # older row blank, not deleted
    assert rows[1]["plane_rms"] == "0.012"
    assert len(rows) == 2


def test_creates_parent_directories(tmp_path):
    log = ResultsLog(tmp_path / "nested" / "deeper" / "r.csv")
    log.append({"a": 1})
    assert log.path.exists()


def test_column_order_is_stable_across_widening(tmp_path):
    log = ResultsLog(tmp_path / "r.csv")
    log.append({"b": 1, "a": 2})
    log.append({"b": 3, "a": 4, "c": 5})
    assert log.fieldnames() == ["b", "a", "c"]


class _FakeRun:
    """Duck-typed stand-in for OdometryRun, to keep this module's tests free of
    the pipeline it deliberately does not import."""
    import numpy as _np
    config = {"voxel_size": 0.15, "deskew": False}
    xyz = _np.array([[0.0, 0, 0], [1, 0, 0], [2.0, 0, 0]])
    sizes = _np.array([4000, 4100, 4200])
    thresholds = _np.array([0.4, 0.38, 0.37])
    deltas = [_np.eye(4)] * 3
    net_displacement = 2.0
    path_length = 2.0
    jitter_mm = 3.0
    rotation_deg = _np.array([0.1, 0.2, 0.3])


class _FakeMeta:
    kind = "line"
    environment = "bare-corridor"
    speed_mps = 0.5
    truth_m = 2.1
    truth_method = "suitcase-at-0-mark"


def test_run_row_folds_config_and_metrics_into_one_flat_row():
    row = run_row("personal/line_01.l1raw", _FakeRun(), _FakeMeta())
    assert row["recording"] == "line_01.l1raw"
    assert row["cfg.voxel_size"] == 0.15
    assert row["environment"] == "bare-corridor"
    assert row["net_displacement_m"] == 2.0


def test_run_row_computes_scale_error_only_when_truth_exists():
    with_truth = run_row("a.l1raw", _FakeRun(), _FakeMeta())
    assert with_truth["scale_error_pct"] == round(100 * (2.0 - 2.1) / 2.1, 3)
    # No sidecar: the factor columns are blank and no scale error is invented.
    without = run_row("a.l1raw", _FakeRun(), None)
    assert "scale_error_pct" not in without
    assert without["truth_m"] is None


def test_run_row_extra_kwargs_become_columns():
    row = run_row("a.l1raw", _FakeRun(), None, ablated="deskew", condition="OFF")
    assert row["ablated"] == "deskew" and row["condition"] == "OFF"


def test_explicit_truth_overrides_the_sidecar_and_is_recorded():
    # The printed number and the logged number must come from the same truth.
    row = run_row("a.l1raw", _FakeRun(), _FakeMeta(), truth=4.0)
    assert row["truth_m"] == 2.1          # what the sidecar said
    assert row["truth_used_m"] == 4.0     # what was actually used
    assert row["scale_error_pct"] == round(100 * (2.0 - 4.0) / 4.0, 3)


def test_truth_falls_back_to_the_sidecar_when_not_given():
    row = run_row("a.l1raw", _FakeRun(), _FakeMeta())
    assert row["truth_used_m"] == 2.1


def test_cost_columns_carry_the_host_that_produced_them():
    class Timed(_FakeRun):
        replay_seconds = 1.2
        replay_host = "some-desktop"
        ms_per_frame = 400.0
        budget_pct = 200.0
    row = run_row("a.l1raw", Timed(), None)
    # A timing without its machine is not a real-time claim, it is a number.
    assert row["replay_host"] == "some-desktop"
    assert row["ms_per_frame"] == 400.0
    assert row["budget_pct"] == 200.0


def test_tags_are_how_an_ablation_pair_is_identified(tmp_path):
    # Two runs one flag apart are an ablation. Without tags they are two rows
    # you have to pair by matching every other column by hand.
    log = ResultsLog(tmp_path / "r.csv")
    for cond, deskew in (("ON", True), ("OFF", False)):
        class R(_FakeRun):
            config = {"voxel_size": 0.15, "deskew": deskew}
        log.append(run_row("loop.l1raw", R(), None,
                           ablated="deskew", condition=cond))
    rows = log.rows()
    assert [r["condition"] for r in rows] == ["ON", "OFF"]
    assert {r["ablated"] for r in rows} == {"deskew"}
    assert [r["cfg.deskew"] for r in rows] == ["True", "False"]
