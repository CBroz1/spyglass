import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.testing.mock.file import mock_NWBFile


def make_mvp_hash_files(base_dir: Path):
    """
    Create minimal files:
      - sample.json (valid JSON)
      - array.npy   (valid NumPy array)
      - whatever.xyz (arbitrary contents)
    under base_dir / tmp / test_hasher
    """
    target = Path(base_dir) / "tmp" / "test_hasher"
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "sample.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"foo": 1, "bar": [1, 2, 3]}, f)

    npy_path = target / "array.npy"
    np.save(npy_path, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    xyz_path = target / "whatever.xyz"
    xyz_path.write_text("this can be anything\n", encoding="utf-8")

    nwb_path = target / "dummy.nwb"
    nwbfile = mock_NWBFile()  # creates a small, valid NWBFile for testing
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    return target


@pytest.fixture
def dir_hasher(base_dir):
    from spyglass.utils.nwb_hash import DirectoryHasher

    mvp_hash_dir = make_mvp_hash_files(base_dir)
    yield DirectoryHasher(mvp_hash_dir, keep_obj_hash=True)


def test_dir_hasher(dir_hasher):
    hash = dir_hasher.hash
    assert hash is not None
    assert isinstance(hash, str)
    assert len(hash) == 32

    cache = dir_hasher.cache
    assert "sample.json" in cache
    assert "array.npy" in cache
    assert "whatever.xyz" in cache
    assert "dummy.nwb" in cache


@pytest.fixture
def nwb_hasher(mini_path):
    from spyglass.utils.nwb_hash import NwbfileHasher

    yield NwbfileHasher(mini_path, precision_lookup=5, keep_obj_hash=True)


@pytest.mark.slow
def test_nwb_hasher(nwb_hasher):

    hash = nwb_hasher.hash
    assert hash is not None
    assert isinstance(hash, str)
    assert len(hash) == 32
    assert hash.startswith("1d393"), "Unexpected NWB file hash"

    # Check that individual object hashes are present
    cache = nwb_hasher.objs
    assert "acquisition" in cache
    assert "processing" in cache

    precision = nwb_hasher.precision.get("ProcessedElectricalSeries")
    assert precision == 5

    roundable = nwb_hasher.is_roundable(5)
    not_roundable = nwb_hasher.is_roundable(None)
    assert roundable is True and not_roundable is False

    skipped_obj = SimpleNamespace(name="version")
    assert nwb_hasher.hash_dataset(skipped_obj) is None


@pytest.fixture
def nwb_legacy_hasher(mini_path):
    from spyglass.utils.nwb_hash import NwbfileHasher

    yield NwbfileHasher(
        mini_path, precision_lookup=5, keep_obj_hash=True, legacy_mode=True
    )


@pytest.mark.slow
def test_nwb_legacy_hasher(nwb_legacy_hasher):
    hash = nwb_legacy_hasher.hash
    assert hash.startswith("4e0c"), "Unexpected NWB file hash"


# NFS silly-rename handling. On NFS, unlinking a file another process holds
# open renames it to '.nfs<24 hex>' in place instead of removing it. These
# leftovers must not influence a directory's hash. See #1662 follow-up.

NFS_STUB = ".nfs00000000068e2d5c00001069"  # observed form, 24 hex chars


def test_is_real_file(tmp_path):
    """Only the NFS stub is excluded -- ordinary dotfiles stay real."""
    from spyglass.utils.nwb_hash import is_real_file

    target = tmp_path / "test_is_real_file"
    target.mkdir(parents=True, exist_ok=True)

    regular = target / "data.npy"
    regular.write_bytes(b"payload")
    stub = target / NFS_STUB
    stub.write_bytes(b"leftover")
    dotfile = target / ".zattrs"  # legitimate, e.g. zarr metadata
    dotfile.write_text("{}")
    subdir = target / "properties"
    subdir.mkdir(exist_ok=True)

    assert is_real_file(regular) is True
    assert is_real_file(stub) is False
    assert is_real_file(dotfile) is True, "Only .nfs should be excluded"
    assert is_real_file(subdir) is False, "Directories are not files"


def test_dir_is_empty(tmp_path):
    """A dir holding only NFS leftovers counts as empty."""
    from spyglass.utils.nwb_hash import dir_is_empty

    missing = tmp_path / "never_created"
    assert dir_is_empty(missing) is True, "Missing dir should read as empty"

    target = tmp_path / "waveforms"
    target.mkdir()
    assert dir_is_empty(target) is True

    stub = target / NFS_STUB
    stub.write_bytes(b"leftover")
    assert dir_is_empty(target) is True, "Leftovers are not contents"

    (target / "waveforms.npy").write_bytes(b"real")
    assert dir_is_empty(target) is False, "Real file means not empty"


def test_hash_stable_with_nfs_leftover(tmp_path):
    """A planted .nfs leftover must not change the directory hash."""
    from spyglass.utils.nwb_hash import DirectoryHasher

    # tmp_path, not base_dir: make_mvp_hash_files writes to a fixed
    # '<arg>/tmp/test_hasher' path, so sharing base_dir would leak planted
    # files into `dir_hasher` and into the other tests below.
    target = make_mvp_hash_files(tmp_path)
    before = DirectoryHasher(target).hash

    (target / NFS_STUB).write_bytes(b"whatever was left behind")
    after = DirectoryHasher(target).hash

    assert before == after, "NFS leftover changed the directory hash"


def test_hash_stable_with_nfs_leftover_nested(tmp_path):
    """The leftover filter must recurse, not just skip top-level entries."""
    from spyglass.utils.nwb_hash import DirectoryHasher

    target = make_mvp_hash_files(tmp_path)
    nested = target / "properties"  # mirrors a real recording's layout
    nested.mkdir(exist_ok=True)
    np.save(nested / "channel_id.npy", np.arange(4, dtype=np.int32))

    before = DirectoryHasher(target).hash

    (nested / NFS_STUB).write_bytes(b"nested leftover")
    after = DirectoryHasher(target).hash

    assert before == after, "Nested NFS leftover changed the directory hash"


def test_nfs_leftover_absent_from_cache(tmp_path):
    """Leftovers must not surface in the per-file diff report cache."""
    from spyglass.utils.nwb_hash import DirectoryHasher

    target = make_mvp_hash_files(tmp_path)
    (target / NFS_STUB).write_bytes(b"leftover")

    cache = DirectoryHasher(target, keep_obj_hash=True).cache

    assert "sample.json" in cache, "Real files should still be cached"
    assert not [k for k in cache if ".nfs" in k], f"Leftover cached: {cache}"
