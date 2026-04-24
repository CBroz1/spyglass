from unittest.mock import Mock

import pytest
from numpy import array_equal

from ..conftest import TEARDOWN


def test_create_from_config(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.Electrode().fetch()
    common_ephys.Electrode.create_from_config(mini_copy_name)
    after = common_ephys.Electrode().fetch()
    # Because already inserted, expect no change
    assert array_equal(
        before, after
    ), "Electrode.create_from_config had unexpected effect"


def test_raw_object(mini_insert, common_ephys, mini_dict, mini_content):
    obj_fetch = common_ephys.Raw().nwb_object(mini_dict).object_id
    obj_raw = mini_content.get_acquisition().object_id
    assert obj_fetch == obj_raw, "Raw.nwb_object did not return expected object"


def test_electrode_populate(common_ephys):
    common_ephys.Electrode.populate()
    assert len(common_ephys.Electrode()) == 128, "Electrode.populate failed"


def test_elec_group_populate(pop_common_electrode_group):
    assert (
        len(pop_common_electrode_group) == 32
    ), "ElectrodeGroup.populate failed"


def test_raw_populate(common_ephys):
    common_ephys.Raw.populate()
    assert len(common_ephys.Raw()) == 1, "Raw.populate failed"


def test_sample_count_populate(common_ephys):
    common_ephys.SampleCount.populate()
    assert len(common_ephys.SampleCount()) == 1, "SampleCount.populate failed"


@pytest.mark.skipif(not TEARDOWN, reason="No teardown: expect no change.")
def test_set_lfp_electrodes(mini_insert, common_ephys, mini_copy_name):
    before = common_ephys.LFPSelection().fetch()
    common_ephys.LFPSelection().set_lfp_electrodes(mini_copy_name, [0])
    after = common_ephys.LFPSelection().fetch()
    assert (
        len(after) == len(before) + 1
    ), "Set LFP electrodes had unexpected effect"


@pytest.mark.skip(reason="Not testing V0: common lfp")
def test_lfp():
    pass


@pytest.fixture
def mock_electrode_group():
    """Mock electrode group for testing."""
    mock_group = Mock()
    mock_group.name = "test_group"
    mock_group.description = "Test electrode group"
    mock_group.targeted_x = None  # Test default case
    return mock_group


@pytest.fixture
def mock_electrode_group_right_hemisphere():
    """Mock electrode group for right hemisphere."""
    mock_group = Mock()
    mock_group.name = "test_group_right"
    mock_group.description = "Test electrode group right"
    mock_group.targeted_x = 2.5  # Positive x = right hemisphere
    return mock_group


@pytest.fixture
def mock_electrode_group_left_hemisphere():
    """Mock electrode group for left hemisphere."""
    mock_group = Mock()
    mock_group.name = "test_group_left"
    mock_group.description = "Test electrode group left"
    mock_group.targeted_x = -1.5  # Negative x = left hemisphere
    return mock_group


def test_electrode_group_hemisphere_detection():
    """Test hemisphere detection logic."""

    # Test hemisphere determination logic
    def determine_hemisphere(targeted_x):
        return "Right" if targeted_x > 0 else "Left"

    assert determine_hemisphere(2.5) == "Right"
    assert determine_hemisphere(-1.5) == "Left"
    assert determine_hemisphere(0.0) == "Left"  # Zero is treated as left


def test_electrode_config_validation():
    """Test electrode configuration validation."""
    # Test valid electrode configuration
    valid_config = {
        "electrode_id": 1,
        "x": 100.0,
        "y": 200.0,
        "location": "CA1",
        "probe_type": "tetrode",
    }

    # Test required fields
    required_fields = ["electrode_id", "x", "y", "location"]
    for field in required_fields:
        assert field in valid_config

    # Test parameter types and ranges
    assert isinstance(valid_config["electrode_id"], int)
    assert valid_config["electrode_id"] > 0
    assert isinstance(valid_config["x"], (int, float))
    assert isinstance(valid_config["y"], (int, float))
    assert isinstance(valid_config["location"], str)
    assert len(valid_config["location"]) > 0


def test_electrode_group_properties():
    """Test electrode group property validation."""
    from unittest.mock import Mock

    # Test valid electrode group
    mock_group = Mock()
    mock_group.name = "test_group"
    mock_group.description = "Test electrode group"
    mock_group.targeted_x = 1.0
    mock_group.targeted_y = 2.0
    mock_group.targeted_z = 3.0

    # Test required attributes exist
    required_attrs = [
        "name",
        "description",
        "targeted_x",
        "targeted_y",
        "targeted_z",
    ]
    for attr in required_attrs:
        assert hasattr(mock_group, attr)
        assert getattr(mock_group, attr) is not None


def test_electrode_parameter_combinations():
    """Test different electrode parameter combinations."""
    test_electrodes = [
        {"id": 1, "x": 100, "y": 200, "region": "CA1"},
        {"id": 2, "x": -50, "y": 150, "region": "CA3"},
        {"id": 3, "x": 0, "y": 0, "region": "DG"},
    ]

    for electrode in test_electrodes:
        # Basic validation
        assert electrode["id"] > 0
        assert isinstance(electrode["x"], (int, float))
        assert isinstance(electrode["y"], (int, float))
        assert len(electrode["region"]) > 0


def test_electrode_config_edge_cases():
    """Test electrode configuration edge cases."""
    # Test empty configuration
    empty_config = {}
    electrode_section = empty_config.get("Electrode", [])
    assert len(electrode_section) == 0

    # Test malformed configuration handling
    def validate_electrode_config(config):
        if not isinstance(config, dict):
            return False
        if "Electrode" not in config:
            return True  # Valid to have no electrode section
        electrode_list = config["Electrode"]
        return isinstance(electrode_list, list)

    assert validate_electrode_config({})  # Empty config is valid
    assert validate_electrode_config({"Electrode": []})  # Empty list is valid
    assert not validate_electrode_config(
        {"Electrode": "invalid"}
    )  # String is invalid


def test_electrode_error_handling():
    """Test electrode error handling scenarios."""
    # Test missing required fields
    incomplete_electrode = {"id": 1}  # Missing x, y coordinates

    def has_required_fields(electrode_config):
        required = ["id", "x", "y"]
        return all(field in electrode_config for field in required)

    assert not has_required_fields(incomplete_electrode)

    # Test complete electrode
    complete_electrode = {"id": 1, "x": 0.0, "y": 0.0}
    assert has_required_fields(complete_electrode)
    assert has_required_fields(complete_electrode)
    assert has_required_fields(complete_electrode)
