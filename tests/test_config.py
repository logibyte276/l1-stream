"""Guard against the config drifting apart again.

This project has been bitten repeatedly by the same constants living in
several places and quietly disagreeing. The worst instance: 08_odometry_live,
the script that drives the actual robot, ran voxel 0.25 / min_range 0.40 /
deskew ON for weeks after the tuning had moved all three, because it had its
own hardcoded copy. `KissOdometry.__init__` had the same stale trio.

Consolidating them into l1_stream.config only helps if something enforces it.
These tests are that something. They need no hardware and no kiss-icp: the
signatures are inspected, not called.
"""

import ast
import inspect
from pathlib import Path

import pytest

from l1_stream import config, offline
from l1_stream.frames import FrameAssembler
from l1_stream.odometry import KissOdometry


def _defaults(func):
    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_kissodometry_signature_matches_config():
    sig = _defaults(KissOdometry.__init__)
    for key, value in config.DEFAULTS.items():
        if key in sig:
            assert sig[key] == value, (
                f"KissOdometry's default for {key!r} is {sig[key]!r} but "
                f"config.DEFAULTS says {value!r}. Import it, don't retype it."
            )


def test_kissodometry_untuned_params_match_config():
    sig = _defaults(KissOdometry.__init__)
    for key, value in config.UNTUNED.items():
        assert sig[key] == value, f"{key}: {sig[key]!r} != UNTUNED {value!r}"


def test_frameassembler_signature_matches_config():
    sig = _defaults(FrameAssembler.__init__)
    for key in ("frame_duration", "rotate_with_imu"):
        assert sig[key] == config.DEFAULTS[key], (
            f"FrameAssembler's {key} default drifted from config.DEFAULTS"
        )


def test_offline_reexports_the_same_object():
    # Not merely equal -- the SAME dict, so there is exactly one source.
    assert offline.DEFAULTS is config.DEFAULTS


def test_every_config_key_is_consumed_by_replay():
    """A key nobody reads is a value that silently does nothing."""
    src = inspect.getsource(offline.replay)
    for key in config.DEFAULTS:
        assert f'"{key}"' in src, f"replay() never reads DEFAULTS[{key!r}]"


def test_defaults_and_untuned_do_not_overlap():
    assert not (set(config.DEFAULTS) & set(config.UNTUNED))


# --- the guard that actually catches the 08_odometry_live class of bug ------

TUNED_KEYS = ("voxel_size", "min_range", "max_range", "frame_duration",
              "initial_threshold")


def _example_files():
    root = Path(__file__).resolve().parents[1] / "examples"
    return sorted(root.glob("*.py")) if root.is_dir() else []


def _is_number(node) -> bool:
    # bool is a subclass of int; a flag default of True/False is not a tuned
    # number and must not be reported as one.
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool))


def _option_dest(call) -> str | None:
    """The dest an ``add_argument("--foo-bar", ...)`` call will produce."""
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value.startswith("--"):
                return arg.value[2:].replace("-", "_")
    return None


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_examples_never_hardcode_tuned_values(path):
    """No example may write a literal number for a tuned parameter.

    They must come from l1_stream.config -- directly, or via add_args /
    DEFAULTS. A bare literal here is exactly how the live script ended up
    running an untuned configuration for weeks.

    THE HOLE THIS USED TO HAVE. add_argument calls were skipped wholesale, on
    the reasoning that a flag default is where a value legitimately appears.
    That is only true when the default is DEFAULTS[...]; a LITERAL there is
    the same bug wearing argparse. 11_vibration.py sat at
    `--voxel-size default=0.25` long after the pipeline had settled on 0.15,
    so its "does this smear exceed a voxel?" verdict was measured against a
    voxel 67% larger than the one actually in use -- and this test passed it.

    So add_argument is now checked too, just differently: for a tuned option
    the default must be a reference (DEFAULTS["voxel_size"]), never a literal.
    """
    tree = ast.parse(path.read_text())
    offences = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            dest = _option_dest(node)
            if dest not in TUNED_KEYS:
                continue
            for kw in node.keywords:
                if kw.arg == "default" and _is_number(kw.value):
                    offences.append(
                        f"--{dest.replace('_', '-')} default={kw.value.value} "
                        f"on line {node.lineno} (use DEFAULTS[{dest!r}])"
                    )
            continue

        for kw in node.keywords:
            if kw.arg in TUNED_KEYS and _is_number(kw.value):
                offences.append(f"{kw.arg}={kw.value.value} on line {node.lineno}")

    assert not offences, (
        f"{path.name} hardcodes tuned config: {offences}. "
        f"Import it from l1_stream.config instead."
    )
