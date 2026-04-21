"""Path resolution helpers shared across the position v2 pipeline."""

from pathlib import Path
from typing import Union

from spyglass.settings import dlc_project_dir


def resolve_model_path(stored_path: Union[str, Path]) -> Path:
    """Resolve a stored model_path to an absolute Path.

    Absolute paths are returned unchanged.  Relative paths are resolved
    against ``dlc_project_dir`` when configured, otherwise against cwd.

    Parameters
    ----------
    stored_path : str or Path

    Returns
    -------
    Path
    """
    p = Path(stored_path)
    if p.is_absolute():
        return p
    base = dlc_project_dir or Path.cwd()
    return Path(base) / p


def to_stored_path(path: Path) -> str:
    """Convert an absolute model path to a portable stored string.

    Returns a path relative to ``dlc_project_dir`` when possible so that
    entries remain valid across base-directory changes.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
    """
    if dlc_project_dir:
        try:
            return str(path.relative_to(dlc_project_dir))
        except ValueError:
            pass
    return str(path)
