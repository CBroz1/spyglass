"""Tests for the host-agnostic filesystem helpers."""

import os
from pathlib import Path

import pytest

from spyglass.utils.path_helper_fn import (
    as_path,
    is_within,
    link_or_copy,
    safe_chmod,
)

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission bits"
)


def test_as_path_coerces_str():
    assert as_path("/a/b") == Path("/a/b")


def test_as_path_passes_through_path():
    path = Path("/a/b")
    assert as_path(path) == path


def test_as_path_none():
    assert as_path(None) is None


def test_as_path_expands_user():
    assert as_path("~/x") == Path.home() / "x"


def test_is_within_child(tmp_path):
    assert is_within(tmp_path / "a" / "b", tmp_path)


def test_is_within_self(tmp_path):
    assert is_within(tmp_path, tmp_path)


def test_is_within_sibling(tmp_path):
    assert not is_within(tmp_path.parent / "elsewhere", tmp_path)


def test_is_within_rejects_name_prefix(tmp_path):
    """A sibling whose name merely starts with the parent's is not within it.

    The substring check this replaces answered True here.
    """
    parent = tmp_path / "raw"
    sneaky = tmp_path / "raw_backup" / "f.nwb"
    assert not is_within(sneaky, parent)


def test_is_within_follows_symlink(tmp_path):
    """Containment is judged by a symlink's target, not its location."""
    outside = tmp_path / "outside"
    outside.mkdir()
    inside = tmp_path / "inside"
    inside.mkdir()
    link = inside / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("symlinks unavailable on this host")
    assert not is_within(link, inside)


@POSIX_ONLY
def test_safe_chmod_applies_mode(tmp_path):
    target = tmp_path / "f"
    target.touch()
    assert safe_chmod(target, 0o600)
    assert target.stat().st_mode & 0o777 == 0o600


def test_safe_chmod_skips_off_posix(tmp_path, monkeypatch):
    target = tmp_path / "f"
    target.touch()
    before = target.stat().st_mode
    monkeypatch.setattr("spyglass.utils.path_helper_fn._POSIX", False)
    assert not safe_chmod(target, 0o600)
    assert target.stat().st_mode == before


def test_link_or_copy_links(tmp_path):
    src = tmp_path / "src"
    src.write_text("data")
    dst = tmp_path / "dst"
    linked = link_or_copy(src, dst)
    assert dst.read_text() == "data"
    assert dst.is_symlink() is linked


def test_link_or_copy_falls_back(tmp_path, monkeypatch):
    """A host that refuses symlinks still gets the bytes at dst."""
    src = tmp_path / "src"
    src.write_text("data")
    dst = tmp_path / "dst"

    def refuse(self, target, target_is_directory=False):
        raise NotImplementedError("no symlinks here")

    monkeypatch.setattr(Path, "symlink_to", refuse)

    assert not link_or_copy(src, dst)
    assert dst.read_text() == "data"
    assert not dst.is_symlink()


def test_link_or_copy_existing_dst_raises(tmp_path):
    """An occupied destination must not be silently overwritten by the copy."""
    src = tmp_path / "src"
    src.write_text("new")
    dst = tmp_path / "dst"
    dst.write_text("old")

    with pytest.raises(FileExistsError):
        link_or_copy(src, dst)

    assert dst.read_text() == "old"
