import json

import pytest

from l1_stream.metadata import RecordingMeta, provenance, sidecar_path


def test_sidecar_appends_rather_than_replaces_suffix():
    # x.l1raw.json, never x.json -- so it sorts beside the recording and two
    # recordings differing only by extension cannot collide.
    assert str(sidecar_path("personal/a.l1raw")).endswith("a.l1raw.json")
    assert sidecar_path("a.l1raw").name == "a.l1raw.json"


def test_roundtrip_preserves_the_fields_that_matter(tmp_path):
    rec = tmp_path / "line_01.l1raw"
    rec.write_bytes(b"not really a recording")
    m = RecordingMeta(
        kind="line", truth_m=7.0, truth_method="suitcase-at-0-mark",
        speed_mps=0.5,
        environment="bare-corridor", notes="the drive formerly misnamed 8m",
    )
    m.save(rec)
    back = RecordingMeta.load(rec)
    assert back.truth_m == 7.0
    assert back.truth_method == "suitcase-at-0-mark"
    assert back.environment == "bare-corridor"
    assert back.kind == "line"


def test_missing_sidecar_raises_but_load_or_none_does_not(tmp_path):
    rec = tmp_path / "orphan.l1raw"
    rec.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        RecordingMeta.load(rec)
    assert RecordingMeta.load_or_none(rec) is None


def test_unknown_keys_in_an_old_sidecar_do_not_break_loading(tmp_path):
    rec = tmp_path / "old.l1raw"
    rec.write_bytes(b"x")
    sidecar_path(rec).write_text(json.dumps({"truth_m": 5.0, "retired_field": 1}))
    assert RecordingMeta.load(rec).truth_m == 5.0


def test_summary_says_so_loudly_when_there_is_no_truth():
    assert "NO TRUTH RECORDED" in RecordingMeta(kind="line").summary()
    assert "NO TRUTH" not in RecordingMeta(kind="line", truth_m=5.0).summary()


def test_provenance_is_total_and_never_raises():
    p = provenance()
    assert "recorded_utc" in p and "python" in p
    # Every field may be None (no git, package not installed) but the keys
    # must always exist, or an analysis cannot tell "unknown" from "absent".
    for key in ("git_commit", "kiss_icp_version", "l1_stream_version"):
        assert key in p
