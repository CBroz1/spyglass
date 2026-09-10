from pathlib import Path

import numpy as np
import pytest


def test_sort_group(pop_sort_group, mini_dict):
    fetched = sum(
        (pop_sort_group.SortGroupElectrode & mini_dict).fetch("electrode_id")
    )
    expected = sum(range(1, 128))  # 1 to 127 inclusive
    assert fetched == expected, "Failed to insert into v0.SortGroupElectrode"


def test_sort_interval(pop_sort_interval, mini_dict):
    tbl = pop_sort_interval & mini_dict
    assert tbl, "Failed to insert into v0.SortInterval"


def test_pop_rec_params(pop_rec_params):
    tbl, name, params = pop_rec_params
    fetched = (tbl & dict(preproc_params_name=name)).fetch1("preproc_params")
    assert fetched == params, "Failed to insert preproc_params"


def test_pop_rec(pop_rec_v0):
    tbl = pop_rec_v0
    assert tbl, "Failed to insert into v0.Recording"


# NFS silly-rename handling. Deleting a file another process holds open
# leaves a '.nfs<hex>' stub behind instead of removing it, so a directory can
# exist, and even hold the expected number of entries, without holding a
# usable recording. See #1662 follow-up.

NFS_STUB = ".nfs00000000068e2d5c0000{:04x}"  # observed form, 24 hex chars

# Written last by SpikeInterface, so their presence means the write finished.
MARKERS = ("si_folder.json", "binary.json", "provenance.json", "probe.json")


def _make_recording_dir(target, n_segments=1, markers=MARKERS, n_stubs=0):
    """Build a directory shaped like a SpikeInterface recording.

    A complete recording holds 4 marker files, 15 ``properties/*.npy``, and
    two files per segment -- 21 files for one segment, 37 for nine.

    Parameters
    ----------
    target : Path
        Directory to build. Created if missing.
    n_segments : int, optional
        Number of cached segments to write, by default 1.
    markers : tuple, optional
        Marker files to write. Pass a subset to simulate an interrupted
        write, by default all four.
    n_stubs : int, optional
        Number of ``.nfs`` leftovers to plant, by default 0.
    """
    target = Path(target)
    props = target / "properties"
    props.mkdir(parents=True, exist_ok=True)

    for name in markers:
        (target / name).write_text("{}")
    for i in range(15):
        np.save(props / f"prop_{i}.npy", np.arange(3, dtype=np.int32))
    for seg in range(n_segments):
        np.save(target / f"times_cached_seg{seg}.npy", np.arange(2))
        (target / f"traces_cached_seg{seg}.raw").write_bytes(b"\x00" * 8)
    for i in range(n_stubs):
        (target / NFS_STUB.format(i)).write_bytes(b"leftover")

    return target


def test_validate_rejects_nfs_only_dir(spike_v0, tmp_path):
    """A directory of pure leftovers must not pass the file-count check."""
    target = tmp_path / "nfs_only"
    target.mkdir()
    for i in range(21):  # exactly normal_file_count, all of it garbage
        (target / NFS_STUB.format(i)).write_bytes(b"leftover")

    with pytest.raises(RuntimeError, match="found 0 files, plus 21 NFS"):
        spike_v0.SpikeSortingRecording()._validate_recording_path(
            str(target), key={}, make_if_missing=False
        )


def test_validate_accepts_real_plus_leftovers(spike_v0, tmp_path):
    """Leftovers alongside a complete recording must not cause a failure."""
    target = _make_recording_dir(tmp_path / "real_plus", n_stubs=3)

    spike_v0.SpikeSortingRecording()._validate_recording_path(
        str(target), key={}, make_if_missing=False
    )  # must not raise


class _Rebuilt(Exception):
    """Raised by the spy below in place of an expensive real rebuild."""


@pytest.fixture
def rebuild_spy(spike_v0, monkeypatch):
    """Report whether `_make_file` decided to rebuild, without paying for it.

    Patches `_get_filtered_recording` -- the first expensive step of a
    rebuild -- to record the call and abort. Tests that expect a rebuild
    assert `_Rebuilt` is raised; tests that expect reuse assert it is not.
    """
    calls = []

    def fake_get_filtered_recording(self, key):
        calls.append(key)
        raise _Rebuilt()

    monkeypatch.setattr(
        spike_v0.SpikeSortingRecording,
        "_get_filtered_recording",
        fake_get_filtered_recording,
    )
    yield calls


@pytest.fixture
def rec_key_and_dir(spike_v0, pop_rec_v0, tmp_path):
    """A real key plus the tmp path `_make_file` would use for it.

    Uses `base_dir=tmp_path` so nothing here touches the session fixture's
    own recording directory -- these tests mutate the directory under test.
    """
    tbl = spike_v0.SpikeSortingRecording()
    key = pop_rec_v0.fetch("KEY")[0]
    rec_path = tmp_path / tbl._get_recording_name(key)
    yield tbl, key, rec_path


def test_make_file_rebuilds_debris_dir(rec_key_and_dir, rebuild_spy, tmp_path):
    """A dir holding only NFS leftovers is debris, and must be rebuilt."""
    tbl, key, rec_path = rec_key_and_dir
    (rec_path / "properties").mkdir(parents=True)  # subdirs survive rmtree
    for i in range(21):
        (rec_path / NFS_STUB.format(i)).write_bytes(b"leftover")

    with pytest.raises(_Rebuilt):
        tbl._make_file(key, base_dir=tmp_path)

    assert rebuild_spy, "Debris directory was reused instead of rebuilt"


def test_make_file_rebuilds_partial_write(
    rec_key_and_dir, rebuild_spy, tmp_path
):
    """An interrupted write -- no si_folder.json -- must be rebuilt.

    Mirrors the 7 production directories that hold only probe.json,
    provenance.json and times_cached_seg*.npy. Their file count varies, so
    only the marker check catches them.
    """
    tbl, key, rec_path = rec_key_and_dir
    rec_path.mkdir(parents=True)
    for name in ("probe.json", "provenance.json"):
        (rec_path / name).write_text("{}")
    for seg in range(13):
        np.save(rec_path / f"times_cached_seg{seg}.npy", np.arange(2))

    with pytest.raises(_Rebuilt):
        tbl._make_file(key, base_dir=tmp_path)

    assert rebuild_spy, "Partial write was reused instead of rebuilt"


def test_discard_dir_sweeps_earlier_stale(rec_key_and_dir, tmp_path):
    """Discarding must collect the litter earlier discards left behind.

    A set-aside directory can only be deleted once the handles that blocked
    it close, so leftovers are swept on the next discard rather than retried
    in place.
    """
    tbl, _, rec_path = rec_key_and_dir
    rec_path.mkdir(parents=True)
    old = rec_path.with_name(rec_path.name + ".stale")
    old.mkdir()  # empty, as it would be once handles closed

    tbl._discard_dir(rec_path)

    assert not old.exists(), "Earlier stale directory was not swept"
    assert not rec_path.exists(), "Discarded path should be freed for reuse"


def test_make_file_reuses_intact_dir(rec_key_and_dir, rebuild_spy, tmp_path):
    """A complete recording must be reused, not rebuilt.

    The regression guard: without it, 'rebuild when in doubt' would turn
    every cache hit into an expensive recompute.
    """
    tbl, key, rec_path = rec_key_and_dir
    _make_recording_dir(rec_path)

    result = tbl._make_file(key, base_dir=tmp_path)

    assert not rebuild_spy, "Intact directory was needlessly rebuilt"
    assert result["hash"], "Expected the existing directory's hash"


def test_make_file_reuses_multisegment_dir(
    rec_key_and_dir, rebuild_spy, tmp_path
):
    """Reuse must not depend on file count -- 37 files is normal, not 21."""
    tbl, key, rec_path = rec_key_and_dir
    _make_recording_dir(rec_path, n_segments=9, n_stubs=2)

    result = tbl._make_file(key, base_dir=tmp_path)

    assert not rebuild_spy, "Complete multi-segment dir was rebuilt"
    assert result["hash"], "Expected the existing directory's hash"
