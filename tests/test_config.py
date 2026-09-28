"""Keep every entry point on the one configuration in l1_stream.config.

A tuned constant copied into a script or a signature default can quietly
disagree with the rest of the pipeline while every output still looks
plausible. These tests fail when that happens. They need no hardware and no
kiss-icp: the signatures are inspected, not called.
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


def test_replay_rejects_unknown_config_keys():
    # A typo must fail loudly, not silently run the default. The check runs
    # before any file or kiss-icp access, so this needs neither.
    with pytest.raises(TypeError, match="voxel"):
        offline.replay("no-such-file.l1raw", voxel=0.10)


# --- examples must not hardcode tuned values --------------------------------

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
    DEFAULTS -- so a script cannot run a different configuration from the
    rest of the pipeline without it being visible.

    add_argument calls are checked too: a flag default is a legitimate place
    for a value only when it is a reference (DEFAULTS["voxel_size"]); a
    literal there is the same problem expressed through argparse.
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
