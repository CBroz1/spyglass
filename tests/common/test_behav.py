import pytest
from pandas import DataFrame

from ..conftest import TEARDOWN


@pytest.mark.slow
def test_invalid_interval(pos_src):
    """Test invalid interval"""
    with pytest.raises(ValueError):
        pos_src.get_pos_interval_name("invalid_interval")


def test_invalid_epoch_num(common):
    """Test invalid epoch num"""
    with pytest.raises(ValueError):
        common.PositionSource.get_epoch_num("invalid_epoch_num")


def test_valid_epoch_num(common):
    """Test valid epoch num"""
    epoch_num = common.PositionSource.get_epoch_num("pos 1 valid times")
    assert epoch_num == 1, "PositionSource get_epoch_num failed"


@pytest.mark.slow
def test_pos_source_make(common):
    """Test custom populate"""
    common.PositionSource().make(common.Session())


def test_pos_source_make_invalid(common):
    """Test invalid populate"""
    with pytest.raises(ValueError):
        common.PositionSource().make(dict())


def test_raw_position_fetch_nwb(common, mini_pos, mini_pos_interval_dict):
    """Test RawPosition fetch nwb"""
    fetched = DataFrame(
        (common.RawPosition & mini_pos_interval_dict)
        .fetch_nwb()[0]["raw_position"]
        .data
    )
    raw = DataFrame(mini_pos["led_0_series_0"].data)
    # compare with mini_pos
    assert fetched.equals(raw), "RawPosition fetch_nwb failed"


def test_raw_position_fetch1_df(common, mini_pos, mini_pos_interval_dict):
    """Test RawPosition fetch1 dataframe"""
    fetched = (common.RawPosition & mini_pos_interval_dict).fetch1_dataframe()
    fetched.reset_index(drop=True, inplace=True)
    fetched.columns = range(fetched.shape[1])
    fetched = fetched.iloc[:, :2]

    raw = DataFrame(mini_pos["led_0_series_0"].data)
    assert fetched.equals(raw), "RawPosition fetch1_dataframe failed"


def test_raw_position_fetch_multi_df(common, mini_pos, mini_pos_interval_dict):
    """Test RawPosition fetch1 dataframe"""
    shape = common.RawPosition().fetch1_dataframe().shape
    assert shape == (542, 8), "RawPosition.PosObj fetch1_dataframe failed"


@pytest.fixture(scope="session")
def pop_state_script(common):
    """Populate state script"""
    keys = common.StateScriptFile.key_source
    common.StateScriptFile.populate()
    yield keys


def test_populate_state_script(common, pop_state_script):
    """Test populate state script

    See #849. Expect no result for this table."""
    assert len(common.StateScriptFile.key_source) == len(
        pop_state_script
    ), "StateScript populate unexpected effect"


def test_videofile_update_entries(common, video_keys):
    """Test update entries"""
    key = common.VideoFile().fetch(as_dict=True)[0]
    common.VideoFile().update_entries(key)


def test_videofile_getabspath(common):
    """Test get absolute path"""
    key = common.VideoFile().fetch(as_dict=True)[0]
    path = common.VideoFile().get_abs_path(key)
    file_part = key["nwb_file_name"].split("2")[0] + "_0" + str(key["epoch"])
    assert file_part in path, "VideoFile get_abs_path failed"


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: expect no change.")
def test_pos_interval_no_transaction(verbose_context, common, mini_restr):
    """Test no transaction"""
    before = common.PositionIntervalMap().fetch()
    with verbose_context:
        common.PositionIntervalMap().make(mini_restr)
    after = common.PositionIntervalMap().fetch()
    expected_insertions = 4
    assert len(after) - len(before) == expected_insertions, (
        f"PositionIntervalMap failed to insert the expected number of entries. "
        f"Expected {expected_insertions}, but got {len(after) - len(before)}."
    )
    assert (
        "" in after["position_interval_name"]
    ), "PositionIntervalMap null insert failed"


def test_get_pos_interval_name(pos_src, pos_interval_01):
    """Test get pos interval name"""
    names = [f"pos {x} valid times" for x in range(1)]
    assert pos_interval_01 == names, "get_pos_interval_name failed"


def test_convert_epoch(common, mini_dict, pos_interval_01):
    this_key = (
        common.IntervalList & mini_dict & {"interval_list_name": "01_s1"}
    ).fetch1()
    ret = common.common_behav.convert_epoch_interval_name_to_position_interval_name(
        this_key
    )
    assert (
        ret == pos_interval_01[0]
    ), "convert_epoch_interval_name_to_position_interval_name failed"


def test_prepare_video_entry_with_external_file(common):
    """Test _prepare_video_entry with external_file attribute."""
    from unittest.mock import MagicMock, Mock
    from pathlib import Path

    # Create mock video object with external_file
    mock_device = MagicMock()
    mock_device.name = "camera_device 1"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "generic_video_name"
    mock_video.external_file = ["file1.mp4", "file2.mp4", "file3.mp4"]

    # Mock CameraDevice table to have the camera
    key = {"test": "key"}

    # Test with file_idx=None (should use index 0)
    video_file = common.VideoFile()
    with pytest.importorskip("unittest.mock").patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True  # Camera exists

        result = video_file._prepare_video_entry(key, mock_video)

        expected_filename = Path("file1.mp4").name
        assert expected_filename in result["path"]


def test_prepare_video_entry_with_file_idx(common):
    """Test _prepare_video_entry with specific file_idx."""
    from unittest.mock import MagicMock, Mock
    from pathlib import Path

    # Create mock video object
    mock_device = MagicMock()
    mock_device.name = "camera_device 2"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "generic_video_name"
    mock_video.external_file = ["file1.mp4", "file2.mp4", "file3.mp4"]

    key = {"test": "key"}

    # Test with file_idx=1 (should use second file)
    video_file = common.VideoFile()
    with pytest.importorskip("unittest.mock").patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True

        result = video_file._prepare_video_entry(key, mock_video, file_idx=1)

        expected_filename = Path("file2.mp4").name
        assert expected_filename in result["path"]


def test_prepare_video_entry_with_empty_external_file(common):
    """Test _prepare_video_entry with empty external_file list."""
    from unittest.mock import MagicMock, Mock

    # Create mock video object with empty external_file
    mock_device = MagicMock()
    mock_device.name = "camera_device 3"
    mock_device.camera_name = "test_camera"

    mock_video = Mock()
    mock_video.device = mock_device
    mock_video.object_id = "test_object_id"
    mock_video.name = "fallback_video_name"
    mock_video.external_file = []  # Empty list

    key = {"test": "key"}

    video_file = common.VideoFile()
    with pytest.importorskip("unittest.mock").patch.object(
        common.common_behav, "CameraDevice"
    ) as mock_camera_device:
        mock_camera_device.__and__.return_value = True

        result = video_file._prepare_video_entry(key, mock_video)

        # Should use video object name as fallback
        assert "fallback_video_name" in result["path"]
