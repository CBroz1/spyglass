"""Position V1 must warn when run against DeepLabCut 3.x.

V1 targets the DLC 2.x TensorFlow engine. Under 3.x's PyTorch default, model
layout, snapshot naming, and scorer strings all change, so V1 fails in ways
that read as unrelated bugs (see the ``best-195`` scorer report). The warning
fires on *use*, not import -- V2 shares ``position/utils/`` and must stay quiet.
"""

import pytest


@pytest.fixture
def dlc_utils():
    from spyglass.position.v1 import dlc_utils

    yield dlc_utils


@pytest.fixture(autouse=True)
def _reset_warned(dlc_utils):
    """Clear the once-only latch so each test sees a fresh state."""
    dlc_utils._DLC3_WARNED = False
    yield
    dlc_utils._DLC3_WARNED = False


def _fake_dlc(monkeypatch, version):
    """Install a stub ``deeplabcut`` module reporting *version*."""
    import sys
    import types

    mod = types.ModuleType("deeplabcut")
    mod.__version__ = version
    monkeypatch.setitem(sys.modules, "deeplabcut", mod)


def test_warns_on_dlc3(dlc_utils, monkeypatch, caplog):
    """A 3.x version produces exactly one warning naming the version."""
    _fake_dlc(monkeypatch, "3.0.0rc14")

    with caplog.at_level("WARNING", logger="spyglass"):
        dlc_utils.warn_if_dlc3()

    assert "3.0.0rc14" in caplog.text
    assert "Position V1 targets DLC 2.x" in caplog.text
    assert "Position V2" in caplog.text  # points at the supported path


@pytest.mark.parametrize("version", ["3.0.0", "3.0.0rc14", "3.1.2.dev0"])
def test_warns_on_dlc3_prereleases(dlc_utils, monkeypatch, caplog, version):
    """Release candidates count as 3.x.

    ``Version("3.0.0rc14") >= Version("3.0.0")`` is False -- a prerelease sorts
    below its own release -- so a naive comparison stays silent for exactly the
    builds users run. The reporter who hit the ``best-195`` scorer was on
    3.0.0rc14.
    """
    _fake_dlc(monkeypatch, version)

    with caplog.at_level("WARNING", logger="spyglass"):
        dlc_utils.warn_if_dlc3()

    assert "Position V1 targets DLC 2.x" in caplog.text


def test_silent_on_dlc2(dlc_utils, monkeypatch, caplog):
    """2.x is the supported pairing -- no warning."""
    _fake_dlc(monkeypatch, "2.3.11")

    with caplog.at_level("WARNING", logger="spyglass"):
        dlc_utils.warn_if_dlc3()

    assert caplog.text == ""


def test_warns_once(dlc_utils, monkeypatch, caplog):
    """Repeat calls stay quiet; populate() would otherwise warn per key."""
    _fake_dlc(monkeypatch, "3.0.0")

    with caplog.at_level("WARNING", logger="spyglass"):
        for _ in range(5):
            dlc_utils.warn_if_dlc3()

    assert caplog.text.count("Position V1 targets DLC 2.x") == 1


@pytest.mark.parametrize("version", ["not-a-version", None])
def test_unparseable_version_is_silent(dlc_utils, monkeypatch, caplog, version):
    """A version we cannot parse must not raise or cry wolf."""
    _fake_dlc(monkeypatch, version)

    with caplog.at_level("WARNING", logger="spyglass"):
        dlc_utils.warn_if_dlc3()  # must not raise

    assert caplog.text == ""


def test_missing_dlc_is_silent(dlc_utils, monkeypatch, caplog):
    """No DeepLabCut installed -- nothing to warn about."""
    import builtins

    real_import = builtins.__import__

    def _no_dlc(name, *args, **kwargs):
        if name == "deeplabcut":
            raise ImportError("No module named 'deeplabcut'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_dlc)

    with caplog.at_level("WARNING", logger="spyglass"):
        dlc_utils.warn_if_dlc3()  # must not raise

    assert caplog.text == ""


def test_v1_entry_points_call_the_warning():
    """The three V1 DLC populate paths must invoke it.

    A guard nothing calls is worse than none -- it reads as covered.
    """
    import inspect

    from spyglass.position.v1 import (
        position_dlc_model,
        position_dlc_pose_estimation,
        position_dlc_training,
    )

    for tbl, method in [
        (position_dlc_pose_estimation.DLCPoseEstimation, "make"),
        (position_dlc_model.DLCModel, "make"),
        (position_dlc_training.DLCModelTraining, "make_fetch"),
    ]:
        src = inspect.getsource(getattr(tbl, method))
        assert "warn_if_dlc3()" in src, f"{tbl.__name__}.{method}"
