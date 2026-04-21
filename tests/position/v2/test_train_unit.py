"""Unit tests for train.py module functions and methods.

This file tests the train.py module with proper fixture management to avoid
database connection issues during test collection.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestHelperFunctions:
    """Test utility/helper functions in train.py module."""

    def test_default_pk_name(self):
        """Test default_pk_name generation."""
        from spyglass.position.v2.train import default_pk_name

        # Test basic functionality
        name = default_pk_name("test", {"param": "value"})
        assert name.startswith("test-")
        assert len(name) <= 32

        # Test without hash
        name_no_hash = default_pk_name(
            "test", {"param": "value"}, include_hash=False
        )
        assert name_no_hash.startswith("test-")

        # Test limit parameter
        short_name = default_pk_name(
            "verylongprefix", {"many": "params"}, limit=10
        )
        assert len(short_name) <= 10

    def test_resolve_model_path(self):
        """Test resolve_model_path function."""
        from spyglass.position.v2.train import resolve_model_path

        # Test absolute path
        abs_path = "/absolute/path/to/model.pkl"
        resolved = resolve_model_path(abs_path)
        assert resolved == Path(abs_path)

        # Test relative path - should use DLC project directory if configured
        rel_path = "relative/path/model.pkl"
        resolved = resolve_model_path(rel_path)
        # Check that it's a valid Path object (actual behavior depends on DLC config)
        assert isinstance(resolved, Path)
        assert str(resolved).endswith("relative/path/model.pkl")

    def test_prompt_default(self):
        """Test prompt_default function."""
        from spyglass.position.v2.train import prompt_default

        # Mock input to test default behavior
        with patch("builtins.input", return_value=""):
            result = prompt_default("test_key", "default_value")
            assert result == "default_value"

        # Mock input to test custom value
        with patch("builtins.input", return_value="custom_value"):
            result = prompt_default("test_key", "default_value")
            assert result == "custom_value"

        # Test abort
        with patch("builtins.input", return_value="n"):
            with pytest.raises(RuntimeError, match="Aborted by user"):
                prompt_default("test_key", "default_value")


class TestModelMethods:
    """Test Model table methods using fixtures."""

    def test_make_method_basic(self, pv2_train, model, model_sel, model_params):
        """Test basic Model.make() functionality."""
        # Mock the strategy pattern components
        with patch(
            "spyglass.position.utils.tool_strategies.ToolStrategyFactory"
        ) as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.train_model.return_value = {
                "model_id": "test_model_123",
                "model_path": "/path/to/model",
                "evaluation": {"loss": 0.05},
            }
            mock_factory.create_strategy.return_value = mock_strategy

            # Create test selection entry
            sel_key = {
                "model_params_id": "dlc_default",
                "tool": "DLC",
                "vid_group_id": "test_group",
            }

            # Mock ModelSelection fetch to return our test data
            with patch(
                "spyglass.position.v2.train.ModelSelection"
            ) as mock_model_selection:
                mock_selection_instance = MagicMock()
                mock_selection_instance.fetch1.return_value = {
                    "model_params_id": "dlc_default",
                    "tool": "DLC",
                    "vid_group_id": "test_group",
                }
                mock_model_selection.return_value = mock_selection_instance

                # Mock ModelParams fetch
                with patch(
                    "spyglass.position.v2.train.ModelParams"
                ) as mock_model_params:
                    mock_params_instance = MagicMock()
                    mock_params_instance.fetch1.return_value = {
                        "tool": "DLC",
                        "params": {"shuffle": 1, "trainingsetindex": 0},
                        "skeleton_id": "test_skeleton",
                    }
                    mock_model_params.return_value = mock_params_instance

                    # Mock VidFileGroup
                    with patch(
                        "spyglass.position.v2.train.VidFileGroup"
                    ) as mock_vfg:
                        mock_vfg_instance = MagicMock()
                        mock_vfg_instance.fetch1.return_value = {
                            "vid_group_id": "test_group",
                            "video_files": ["test1.mp4", "test2.mp4"],
                        }
                        mock_vfg.return_value = mock_vfg_instance

                        # Mock the insert operation
                        with patch.object(model, "insert1") as mock_insert:
                            with patch.object(model, "_info_msg"):
                                # Test make method
                                model.make(sel_key)

                                # Verify strategy was called
                                mock_strategy.train_model.assert_called_once()
                                mock_insert.assert_called_once()
        # Create mock metadata object
        metadata = MagicMock()
        metadata.model_id = "test_metadata"
        metadata.model_path = Path("/test/model.pkl")
        metadata.project_path = Path("/test/project")
        metadata.config_path = Path("/test/config.yaml")
        metadata.params = {"shuffle": 1, "maxiters": 1000}
        metadata.config = {"Task": "TestTask", "date": "2026-04-20"}
        metadata.latest_model = {
            "iteration": 1000,
            "trainFraction": 0.8,
            "date_trained": datetime.utcnow(),
            "snapshot": "snapshot-1000",
        }
        metadata.skeleton_id = "test_skeleton"
        metadata.parent_id = "parent_model"

        # Mock NWB components
        mock_nwbfile = MagicMock()
        mock_io = MagicMock()

        with (
            patch("pynwb.NWBFile", return_value=mock_nwbfile),
            patch("pynwb.NWBHDF5IO", return_value=mock_io),
            patch("spyglass.common.AnalysisNwbfile") as mock_analysis,
            patch("spyglass.common.Nwbfile") as mock_base_nwb,
        ):

            # Mock parent NWB files available (use empty list for simpler test)
            mock_base_nwb.return_value.fetch.return_value = []
            mock_analysis.return_value.add.return_value = None

            with patch.object(model, "_info_msg"):
                result = model._register_model_metadata(metadata)

            # Verify NWB file creation was attempted
            assert isinstance(result, str)
            assert result.endswith(".nwb")
            mock_nwbfile.add_scratch.assert_called()

    def test_train_method_basic(self, pv2_train, model):
        """Test basic Model.train() functionality."""
        # Get an existing model from the database
        existing_models = model.fetch("KEY")
        if not existing_models:
            pytest.skip("No Model entries available for testing")

        model_key = existing_models[0]
        initial_count = len(model)

        # Mock strategy to avoid actual training
        with patch(
            "spyglass.position.utils.tool_strategies.ToolStrategyFactory"
        ) as mock_factory:
            mock_strategy = MagicMock()
            mock_strategy.train_model.return_value = {
                "model_id": "continued_model_123",
                "model_path": "/tmp/continued_model.pkl",
                "evaluation": {"loss": 0.03},
            }
            mock_factory.create_strategy.return_value = mock_strategy

            with patch.object(model, "_info_msg"):
                # Test train method
                result = model.train(model_key, maxiters=1000, shuffle=2)

                # Verify result is a valid key
                assert isinstance(result, dict)
                assert "model_id" in result

                # Verify a new model was created
                assert len(model) == initial_count + 1


class TestModelParams:
    """Test ModelParams table methods."""

    def test_insert1_basic(self, pv2_train, model_params):
        """Test basic ModelParams.insert1() functionality."""
        test_params = {
            "model_params_name": "unit_test_params",
            "tool": "DLC",
            "params": {"shuffle": 1, "trainingsetindex": 0, "maxiters": 1000},
            "skeleton_id": "test_skeleton",
        }

        # Check if params already exist and delete if so
        existing = model_params & {
            "model_params_name": test_params["model_params_name"]
        }
        if existing:
            existing.delete(safemode=False)

        # Test insert - this will validate against real strategy
        initial_count = len(model_params)
        result = model_params.insert1(test_params)

        # Verify insertion worked
        assert len(model_params) == initial_count + 1
        assert result["tool"] == "DLC"
        assert result["model_params_name"] == "unit_test_params"

        # Cleanup
        (model_params & {"model_params_name": "unit_test_params"}).delete()

    def test_insert1_unsupported_tool(self, pv2_train, model_params):
        """Test insert1() with unsupported tool."""
        test_params = {"tool": "UNSUPPORTED_TOOL", "params": {"param": "value"}}

        # Mock ToolStrategyFactory to raise ValueError
        with patch(
            "spyglass.position.utils.tool_strategies.ToolStrategyFactory"
        ) as mock_factory:
            mock_factory.create_strategy.side_effect = ValueError(
                "Unsupported tool"
            )

            with pytest.raises(ValueError, match="Tool not supported"):
                model_params.insert1(test_params)

    def test_get_accepted_params(self, pv2_train, model_params):
        """Test get_accepted_params method."""
        # Test with real DLC strategy
        result = model_params.get_accepted_params("DLC")

        # Verify it returns a set of parameter names
        assert isinstance(result, set)
        assert len(result) > 0
        # DLC should accept common training parameters
        expected_params = {"shuffle", "maxiters", "trainingsetindex"}
        assert expected_params.issubset(result)
