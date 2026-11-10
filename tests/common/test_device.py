import pytest
from numpy import array_equal


def test_invalid_device(common, populate_exception, mini_insert):
    device_dict = common.DataAcquisitionDevice.fetch(as_dict=True)[0]
    device_dict["data_acquisition_device_system"] = "invalid"
    with pytest.raises(populate_exception):
        common.DataAcquisitionDevice._add_device(device_dict, test_mode=True)


def test_get_device(common, mini_content):
    dev = common.DataAcquisitionDevice.get_all_device_names(
        nwbf=mini_content, config=[]
    )
    assert len(dev) == 3, "Unexpected number of devices found"


def test_spike_gadgets_system_alias(mini_insert, common):
    assert (
        common.DataAcquisitionDevice()._add_system("MCU") == "SpikeGadgets"
    ), "SpikeGadgets MCU alias not found"


def test_replace_nan_with_default(utils):
    """Test that NaN values in probe geometry fields are properly replaced with -1.0."""
    # Test with NaN values (similar to the issue case)
    test_data = {
        "probe_id": "nTrode32_probe description",
        "probe_shank": 0,
        "contact_size": float("nan"),
        "probe_electrode": 194,
        "rel_x": float("nan"),
        "rel_y": float("nan"),
        "rel_z": float("nan"),
    }

    result = utils.dj_helper_fn._replace_nan_with_default(test_data)

    # Check that NaN values were replaced with -1.0
    assert result["contact_size"] == -1.0
    assert result["rel_x"] == -1.0
    assert result["rel_y"] == -1.0
    assert result["rel_z"] == -1.0

    # Check that non-NaN values were preserved
    assert result["probe_id"] == "nTrode32_probe description"
    assert result["probe_shank"] == 0
    assert result["probe_electrode"] == 194


def test_none_acq_device_attrs(common):
    """Test that _add_amplifier handles None values gracefully.

    Issue #1446: DataAcqDevice insertion fails when optional amplifier and
    system fields are None, despite being optional in ndx-franklab-novela
    """
    acq_dev = common.DataAcquisitionDevice()
    amp = acq_dev._add_amplifier(None)
    assert amp is None, "None amplifier should return None"

    system = common.DataAcquisitionDevice._add_system(None)
    assert system is None, "None system should return None"


@pytest.fixture
def camera_factory():
    """Build `ndx_franklab_novela.CameraDevice` objects for mapping tests."""
    import ndx_franklab_novela
    from pynwb.device import DeviceModel

    def _make(model=None, manufacturer=None, model_number=None):
        kwargs = dict(
            name="camera_device 7",
            meters_per_pixel=0.002,
            lens="Tamron 8mm",
            camera_name="mapping_test_cam",
        )
        if model is not None:
            kwargs["model"] = DeviceModel(
                name=model,
                manufacturer=manufacturer or "unset",
                **({"model_number": model_number} if model_number else {}),
            )
        elif manufacturer is not None:
            kwargs["manufacturer"] = manufacturer
        return ndx_franklab_novela.CameraDevice(**kwargs)

    return _make


def test_camera_model_prefers_model_number(common, camera_factory):
    """`model` reads the designation, not the DeviceModel's object name.

    A DeviceModel name must be unique within `/general/device_models/`, so a
    writer disambiguates it by folding in the manufacturer. Reading it into
    `model` merges the two fields.
    """
    cam = camera_factory(
        model="FLIR Blackfly S",
        manufacturer="FLIR",
        model_number="BFS-U3-16S2M",
    )
    assert common.CameraDevice.get_model(cam) == "BFS-U3-16S2M"
    assert common.CameraDevice.get_manufacturer(cam) == "FLIR"


def test_camera_model_falls_back_to_name(common, camera_factory):
    """An unset `model_number` falls back to the DeviceModel name."""
    cam = camera_factory(model="Blackfly S", manufacturer="FLIR")
    assert common.CameraDevice.get_model(cam) == "Blackfly S"
    assert common.CameraDevice.get_manufacturer(cam) == "FLIR"


def test_camera_without_model_is_readable(common, camera_factory):
    """A camera declaring no model yields empty strings, and does not raise.

    `model` is optional on CameraDevice. Reading it as a nested NWB object
    raises `ValueError: NWB object model not found`, which blocks ingestion
    of the whole file.
    """
    cam = camera_factory(manufacturer="FLIR")
    assert common.CameraDevice.get_model(cam) == ""
    assert common.CameraDevice.get_manufacturer(cam) == "FLIR"

    bare = camera_factory()
    assert common.CameraDevice.get_model(bare) == ""
    assert common.CameraDevice.get_manufacturer(bare) == ""


def test_camera_model_accepts_legacy_string(common, camera_factory):
    """A pre-pynwb-3 file stores `model` as a plain string."""
    cam = camera_factory()
    cam.fields["model"] = "Blackfly S"
    assert common.CameraDevice.get_model(cam) == "Blackfly S"


def test_camera_manufacturer_recovered_from_device(common, camera_factory):
    """An unset DeviceModel manufacturer falls back to the device's own.

    A file written against an `ndx-franklab-novela` that declares `model` as
    text, read with a core schema that declares it a `DeviceModel` link,
    yields a DeviceModel whose name carries the whole designation and whose
    manufacturer is unset. Reading only the DeviceModel leaves the column
    empty while the text attribute on the device still holds the value.
    """
    cam = camera_factory(model="Allied Vision Mako G-508B", manufacturer="x")
    cam.model.fields["manufacturer"] = None
    cam.fields["manufacturer"] = "Allied Vision"

    assert common.CameraDevice.get_manufacturer(cam) == "Allied Vision"
    # The name is what the file holds; splitting it would be a guess.
    assert common.CameraDevice.get_model(cam) == "Allied Vision Mako G-508B"
