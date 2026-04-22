"""Unit tests for v2/estim.py targeting uncovered lines for coverage improvement.

Focus areas:
- OrientationParams method conditionals (lines 60-81)
- Method logic and parameter validation (lines 304-314)
- Data processing components (testable portions)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def OrientationParams(pv2_estim):
    yield pv2_estim.OrientationParams


@pytest.fixture(scope="module")
def CentroidParams(pv2_estim):
    yield pv2_estim.CentroidParams


@pytest.fixture(scope="module")
def SmoothingParams(pv2_estim):
    yield pv2_estim.SmoothingParams


@pytest.fixture(scope="module")
def PoseParameterSet(pv2_estim):
    yield pv2_estim.PoseParameterSet


@pytest.fixture(scope="module")
def PoseInferenceRunner(pv2_estim):
    yield pv2_estim.PoseInferenceRunner


@pytest.fixture(scope="module")
def NDXPoseBuilder(pv2_estim):
    yield pv2_estim.NDXPoseBuilder


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOrientationParams:
    """Test OrientationParams dataclass methods to cover lines 60-81."""

    def test_as_dict_two_pt_method(self, OrientationParams):
        """Test to_dict() method for two_pt orientation."""
        params = OrientationParams(
            method="two_pt",
            bodypart1="nose",
            bodypart2="tail",
            led1="unused",
            led2="unused",
            led3="unused",
        )

        result = params.to_dict()

        # Should exclude led fields and empty values for two_pt method
        assert "led1" not in result
        assert "led2" not in result
        assert "led3" not in result
        assert result["method"] == "two_pt"
        assert result["bodypart1"] == "nose"
        assert result["bodypart2"] == "tail"

    def test_as_dict_bisector_method(self, OrientationParams):
        """Test to_dict() method for bisector orientation."""
        params = OrientationParams(
            method="bisector",
            bodypart1="unused",
            bodypart2="unused",
            led1="red_led",
            led2="green_led",
            led3="blue_led",
        )

        result = params.to_dict()

        # Should exclude bodypart fields and empty values for bisector method
        assert "bodypart1" not in result
        assert "bodypart2" not in result
        assert result["method"] == "bisector"
        assert result["led1"] == "red_led"
        assert result["led2"] == "green_led"
        assert result["led3"] == "blue_led"

    def test_as_dict_none_method(self, OrientationParams):
        """Test to_dict() method for none orientation."""
        params = OrientationParams(
            method="none",
            bodypart1="unused",
            bodypart2="unused",
            led1="unused",
            led2="unused",
            led3="unused",
        )

        result = params.to_dict()

        # Should exclude all bodypart and led fields for none method
        assert "bodypart1" not in result
        assert "bodypart2" not in result
        assert "led1" not in result
        assert "led2" not in result
        assert "led3" not in result
        assert result["method"] == "none"

    def test_as_dict_filters_none_and_empty(self, OrientationParams):
        """Test that to_dict() filters None values and empty strings."""
        params = OrientationParams(
            method="two_pt",
            bodypart1="nose",
            bodypart2="",  # Empty string should be filtered
        )

        result = params.to_dict()

        assert "bodypart2" not in result  # Empty string filtered out
        assert result["bodypart1"] == "nose"


class TestCentroidParams:
    """Test CentroidParams dataclass methods."""

    def test_as_dict_basic(self, CentroidParams):
        """Test CentroidParams to_dict() method."""
        params = CentroidParams(
            method="centroid",
            points={"nose": [1, 2], "tail": [3, 4]},
            max_LED_separation=10.0,
        )

        result = params.to_dict()

        assert result["method"] == "centroid"
        assert result["points"] == {"nose": [1, 2], "tail": [3, 4]}
        assert result["max_LED_separation"] == 10.0

    def test_to_dict_filters_none(self, CentroidParams):
        """Test that to_dict() filters None values."""
        params = CentroidParams(
            method="centroid",
            points={"nose": [1, 2]},
            max_LED_separation=None,  # Should be filtered out
        )

        result = params.to_dict()

        assert "max_LED_separation" not in result
        assert result["method"] == "centroid"
        assert result["points"] == {"nose": [1, 2]}


class TestSmoothingParams:
    """Test SmoothingParams dataclass methods to cover lines 108-112."""

    def test_to_dict_basic(self, SmoothingParams):
        """Test to_dict() with all parameters set."""
        params = SmoothingParams(
            interpolate=True,
            interp_params={"method": "linear"},
            smooth=True,
            smoothing_params={"window_length": 5},
            likelihood_thresh=0.95,
        )

        result = params.to_dict()

        assert result["interpolate"] is True
        assert result["smooth"] is True
        assert result["likelihood_thresh"] == 0.95
        assert result["interp_params"] == {"method": "linear"}
        assert result["smoothing_params"] == {"window_length": 5}

    def test_to_dict_filters_none(self, SmoothingParams):
        """Test that to_dict() filters None values."""
        params = SmoothingParams(
            interpolate=True,
            interp_params=None,  # Should be filtered
            smooth=False,
            smoothing_params=None,  # Should be filtered
            likelihood_thresh=0.90,
        )

        result = params.to_dict()

        assert "interp_params" not in result
        assert "smoothing_params" not in result
        assert result["interpolate"] is True
        assert result["smooth"] is False
        assert result["likelihood_thresh"] == 0.90

    def test_to_dict_default_values(self, SmoothingParams):
        """Test to_dict() with default values."""
        params = SmoothingParams()

        result = params.to_dict()

        assert result["interpolate"] is True
        assert result["smooth"] is True
        assert result["likelihood_thresh"] == 0.95
        assert "interp_params" not in result
        assert "smoothing_params" not in result


class TestPoseParameterSet:
    """Test PoseParameterSet methods to cover lines 123-136."""

    def test_to_params_dict_basic(
        self,
        OrientationParams,
        CentroidParams,
        SmoothingParams,
        PoseParameterSet,
    ):
        """Test to_params_dict() method."""
        orient = OrientationParams(method="none")
        centroid = CentroidParams(method="centroid", points={"nose": [1, 2]})
        smoothing = SmoothingParams()

        param_set = PoseParameterSet(
            params_name="test_params",
            orient=orient,
            centroid=centroid,
            smoothing=smoothing,
        )

        result = param_set.to_params_dict()

        assert result["pose_params"] == "test_params"
        assert result["orient"]["method"] == "none"
        assert result["centroid"]["method"] == "centroid"
        assert result["smoothing"]["interpolate"] is True

    @patch("spyglass.position.v2.estim.validate_orientation_params")
    @patch("spyglass.position.v2.estim.validate_centroid_params")
    @patch("spyglass.position.v2.estim.validate_smoothing_params")
    def test_validate_calls_validators(
        self,
        mock_smooth,
        mock_cent,
        mock_orient,
        OrientationParams,
        CentroidParams,
        SmoothingParams,
        PoseParameterSet,
    ):
        """Test that validate() calls all validator functions."""
        orient = OrientationParams(
            method="two_pt", bodypart1="nose", bodypart2="tail"
        )
        centroid = CentroidParams(method="centroid", points={"nose": [1, 2]})
        smoothing = SmoothingParams()

        param_set = PoseParameterSet(
            params_name="test_params",
            orient=orient,
            centroid=centroid,
            smoothing=smoothing,
        )

        param_set.validate()

        # Verify all validators were called with correct dictionaries
        mock_orient.assert_called_once()
        mock_cent.assert_called_once()
        mock_smooth.assert_called_once()


class TestPoseEstimSelectionMethods:
    """Test PoseEstimSelection methods to cover _infer_output_dir."""

    def test_infer_output_dir_basic(self, PoseEstimSelection):
        """Test _infer_output_dir with basic key."""
        selection = PoseEstimSelection()
        key = {"model_id": "test_model", "vid_group_id": "test_vid"}

        with (
            patch("spyglass.position.v2.estim.pose_output_dir", None),
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            result = selection._infer_output_dir(key)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            assert "test_model" in result
            assert "test_vid" in result

    def test_infer_output_dir_with_configured_dir(self, PoseEstimSelection):
        """Test _infer_output_dir with configured pose_output_dir."""
        selection = PoseEstimSelection()
        key = {"model_id": "model123", "vid_group_id": "vid456"}

        with (
            patch("spyglass.position.v2.estim.pose_output_dir", "/custom/path"),
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            result = selection._infer_output_dir(key)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            assert "custom" in result
            assert "model123" in result

    def test_infer_output_dir_missing_keys(self, PoseEstimSelection):
        """Test _infer_output_dir with missing keys."""
        selection = PoseEstimSelection()
        key = {}  # Empty key

        with (
            patch("spyglass.position.v2.estim.pose_output_dir", None),
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            result = selection._infer_output_dir(key)

            mock_mkdir.assert_called_once()
            assert "unknown_model" in result


class TestPoseEstimParams:
    """Test PoseEstimParams methods for coverage."""

    @patch("spyglass.position.v2.estim.default_pk_name")
    def test_insert_params_generates_id(self, mock_default_pk, PoseEstimParams):
        """Test insert_params with generated params_id."""
        mock_default_pk.return_value = "pep-20261201"

        # Mock the class methods without database
        with (
            patch.object(PoseEstimParams, "insert1"),
            patch.object(PoseEstimParams, "_info_msg"),
            patch("datajoint.hash.key_hash", return_value="testhash123"),
            patch.object(
                PoseEstimParams.__class__,
                "__and__",
                return_value=MagicMock(
                    __bool__=lambda x: False  # No existing entries
                ),
            ),
        ):
            params = {"orient": {"method": "none"}}

            result = PoseEstimParams.insert_params(params)

            mock_default_pk.assert_called_once_with(
                prefix="pep", include_hash=False
            )
            assert result["pose_estim_params_id"] == "pep-20261201"

    def test_insert_params_with_explicit_id(self, PoseEstimParams):
        """Test insert_params with explicit params_id."""
        with (
            patch.object(PoseEstimParams, "insert1"),
            patch.object(PoseEstimParams, "_info_msg"),
            patch("datajoint.hash.key_hash", return_value="testhash123"),
            patch.object(
                PoseEstimParams.__class__,
                "__and__",
                return_value=MagicMock(
                    __bool__=lambda x: False  # No existing entries
                ),
            ),
        ):
            params = {"orient": {"method": "none"}}

            result = PoseEstimParams.insert_params(
                params, params_id="custom-id"
            )

            # Should use explicit ID, not call default_pk_name
            assert result["pose_estim_params_id"] == "custom-id"

    def test_insert_params_existing_hash(self, PoseEstimParams):
        """Test insert_params with existing params hash."""
        existing_key = {"pose_estim_params_id": "existing-id"}

        # Mock the entire query chain
        existing_mock = MagicMock()
        existing_mock.__bool__ = lambda self: True
        existing_mock.fetch1.return_value = existing_key

        with (
            patch("datajoint.hash.key_hash", return_value="existinghash"),
            patch.object(
                PoseEstimParams, "__call__", return_value=PoseEstimParams()
            ),
            patch.object(
                PoseEstimParams, "__and__", return_value=existing_mock
            ),
        ):
            params = {"orient": {"method": "none"}}

            result = PoseEstimParams.insert_params(params)

            # Should return existing key without insertion
            assert result == existing_key


class TestPoseInferenceRunner:
    """Test PoseInferenceRunner class methods."""

    def test_pose_inference_runner_instantiation(self, PoseInferenceRunner):
        """Test basic instantiation of PoseInferenceRunner."""
        runner = PoseInferenceRunner()
        assert runner is not None
        # Test basic mixin functionality
        assert hasattr(runner, "_info_msg")


class TestBasicMethods:
    """Test basic methods and simple logic paths for coverage."""

    def test_pose_estim_selection_simple(self, PoseEstimSelection):
        """Test simple method paths in PoseEstimSelection."""
        selection = PoseEstimSelection()
        assert selection is not None

    def test_imports_work(
        self,
        OrientationParams,
        CentroidParams,
        NDXPoseBuilder,
        PoseEstimSelection,
    ):
        """Test that all imports work correctly."""
        orient = OrientationParams(method="none")
        assert orient.method == "none"

        centroid = CentroidParams(method="centroid", points={"nose": [1, 2]})
        assert centroid.method == "centroid"
