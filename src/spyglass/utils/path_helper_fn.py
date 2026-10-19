"""Filesystem helpers that behave the same on POSIX and Windows hosts.

Spyglass builds paths from configured directories and NWB file names. Doing
that with string formatting bakes in a separator and makes containment checks
substring comparisons, both of which are wrong off POSIX. These helpers are the
one place that knowledge lives.

Only filesystem paths belong here. HDF5 object names and kachery/DANDI URIs are
always ``/``-separated regardless of host and must not be routed through
``as_path``.
"""

import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Optional, Union

from spyglass.utils.logging import logger

PathLike = Union[str, os.PathLike]

# chmod's mode argument is a POSIX concept. Windows derives a single read-only
# flag from it and ignores the rest, so applying a mode there is misleading
# rather than partially effective.
_POSIX = os.name == "posix"


def as_path(value: Optional[PathLike]) -> Optional[Path]:
    """Coerce a path-like value to a ``Path``, expanding a leading ``~``.

    Parameters
    ----------
    value : PathLike or None
        The value to coerce. ``None`` passes through.

    Returns
    -------
    pathlib.Path or None
        The coerced path, or ``None`` if ``value`` was ``None``.
    """
    if value is None:
        return None
    return Path(value).expanduser()


def is_within(child: PathLike, parent: PathLike) -> bool:
    """Return True if ``child`` is ``parent`` or sits beneath it.

    Both paths are resolved first, so a symlink is judged by its target and a
    path that does not exist yet is judged by where it would be.

    Parameters
    ----------
    child : PathLike
        The path to test.
    parent : PathLike
        The directory it must fall under.

    Returns
    -------
    bool
        True if ``child`` is contained by ``parent``.
    """
    child_path = as_path(child).resolve()
    parent_path = as_path(parent).resolve()
    return child_path.is_relative_to(parent_path)


def store_filepath(relative: PathLike) -> str:
    """Render a store-relative path the way DataJoint records it.

    DataJoint writes external ``filepath`` values with forward slashes on every
    host, so a restriction built from a native path finds nothing on Windows.

    Parameters
    ----------
    relative : PathLike
        A path relative to the store's stage directory.

    Returns
    -------
    str
        The POSIX-style spelling stored in the external table.
    """
    return Path(relative).as_posix()


def from_store_filepath(relative: str, base: PathLike) -> Path:
    """Resolve a DataJoint-stored ``filepath`` against a local directory.

    The inverse of `store_filepath`: the stored value is POSIX, the result is
    native to this host.

    Parameters
    ----------
    relative : str
        A ``filepath`` value from an external table.
    base : PathLike
        The local store directory to resolve against.

    Returns
    -------
    pathlib.Path
        The absolute path on this host.
    """
    return as_path(base) / PurePosixPath(relative)


def safe_chmod(path: PathLike, mode: int) -> bool:
    """Apply a POSIX mode, skipping hosts where it has no meaning.

    Parameters
    ----------
    path : PathLike
        The file to modify.
    mode : int
        POSIX permission bits, e.g. ``0o644``.

    Returns
    -------
    bool
        True if the mode was applied.
    """
    if not _POSIX:
        logger.debug(f"Skipping chmod {mode:o} on non-POSIX host: {path}")
        return False

    as_path(path).chmod(mode)
    return True


def link_or_copy(src: PathLike, dst: PathLike) -> bool:
    """Symlink ``dst`` to ``src``, copying instead if a link is not available.

    Unprivileged Windows accounts cannot create symlinks, and some network
    filesystems reject them. Either way the caller wants the bytes reachable at
    ``dst``, so fall back to a copy.

    Parameters
    ----------
    src : PathLike
        The existing file to point at.
    dst : PathLike
        The path to create.

    Returns
    -------
    bool
        True if a symlink was created, False if the file was copied.

    Raises
    ------
    FileExistsError
        If ``dst`` already exists. Falling back to a copy here would overwrite
        it, so an occupied destination stays the caller's decision.
    """
    src_path, dst_path = as_path(src), as_path(dst)

    try:
        dst_path.symlink_to(src_path)
        return True
    except FileExistsError:
        raise
    except (OSError, NotImplementedError) as err:
        logger.debug(f"Symlink unavailable ({err}); copying to {dst_path}")

    shutil.copy(src_path, dst_path)
    return False
