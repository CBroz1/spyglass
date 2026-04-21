"""Unit tests for DLCStrategy training methods with proper fixture management.

These tests use the existing pytest fixture system to properly manage
Spyglass database dependencies while testing the core DLCStrategy methods.
"""

from unittest.mock import Mock, patch

import pytest


def test_dlc_strategy_prepare_dataset(pv2_train, tmp_path):
    """Test _prepare_training_dataset parameter filtering."""

    # Import within test to use established database connection
    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()

    config_path = tmp_path / "config.yaml"
    params = {
        "batch_size": 8,
        "maxiters": 1000,  # Should be filtered out for create_training_dataset
        "TrainingFraction": 0.95,
    }
    config = {"project_path": str(tmp_path)}

    with (
        patch("deeplabcut.create_training_dataset") as mock_create,
        patch(
            "spyglass.position.utils.get_param_names",
            return_value=["batch_size", "TrainingFraction"],
        ),
        patch("spyglass.position.utils.test_mode_suppress"),
    ):

        strategy._prepare_training_dataset(
            config_path, params, config, model_instance
        )

        # Verify only filtered parameters were passed
        mock_create.assert_called_once_with(
            str(config_path), batch_size=8, TrainingFraction=0.95
        )
        model_instance._info_msg.assert_called()


def test_dlc_strategy_execute_training(pv2_train, tmp_path):
    """Test _execute_training integer conversion and test mode."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()

    config_path = tmp_path / "config.yaml"

    # Test integer conversion
    params = {
        "maxiters": "500",  # String should be converted to int
        "shuffle": "1",
        "trainingsetindex": "0",
    }

    with (
        patch("deeplabcut.train_network") as mock_train,
        patch(
            "spyglass.position.utils.get_param_names",
            return_value=["maxiters", "shuffle", "trainingsetindex"],
        ),
        patch("spyglass.position.utils.suppress_print_from_package"),
        patch("spyglass.position.utils.test_mode_suppress"),
    ):

        strategy._execute_training(config_path, params, model_instance)

        # Verify string parameters were converted to integers
        call_args = mock_train.call_args[1]
        assert call_args["maxiters"] == 500
        assert call_args["shuffle"] == 1
        assert call_args["trainingsetindex"] == 0


def test_dlc_strategy_execute_training_test_mode(pv2_train, tmp_path):
    """Test _execute_training test mode adjustments."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()

    config_path = tmp_path / "config.yaml"
    params = {"test_mode": True, "maxiters": 1000}

    with (
        patch("deeplabcut.train_network") as mock_train,
        patch(
            "spyglass.position.utils.get_param_names", return_value=["maxiters"]
        ),
        patch("spyglass.position.utils.suppress_print_from_package"),
        patch("spyglass.position.utils.test_mode_suppress"),
        patch.dict("sys.modules", {"deeplabcut.core.engine": Mock()}),
    ):

        strategy._execute_training(config_path, params, model_instance)

        # In test mode, maxiters should be reduced to 2
        call_args = mock_train.call_args[1]
        assert call_args["maxiters"] == 2
        assert call_args.get("epochs") == 1
        assert call_args.get("save_epochs") == 1


def test_dlc_strategy_localize_model(pv2_train, tmp_path):
    """Test _localize_trained_model snapshot selection."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()
    model_instance._warn_msg = Mock()

    # Setup directory structure
    project_path = tmp_path / "test_project"
    project_path.mkdir()
    config_path = project_path / "config.yaml"
    config_path.touch()

    config = {"project_path": str(project_path)}

    with (
        patch(
            "deeplabcut.utils.auxiliaryfunctions.read_config",
            return_value={
                "TrainingFraction": [0.95],
                "shuffle": 1,
                "modelprefix": "",
            },
        ),
        patch(
            "deeplabcut.utils.get_model_folder",
            return_value="dlc-models/iteration-0/test-model",
        ),
    ):

        # Create mock training directory with snapshot files
        model_dir = project_path / "dlc-models/iteration-0/test-model"
        train_dir = model_dir / "train"
        train_dir.mkdir(parents=True)

        snapshot1 = train_dir / "snapshot-100.index"
        snapshot2 = train_dir / "snapshot-200.index"
        snapshot1.touch()
        snapshot2.touch()

        # Mock file modification times (snapshot2 is newer)
        with patch("os.path.getmtime") as mock_getmtime:
            mock_getmtime.side_effect = lambda p: {
                str(snapshot1): 1000,
                str(snapshot2): 2000,  # More recent
            }[str(p)]

            result_config, model_id = strategy._localize_trained_model(
                config, model_instance
            )

        # Verify the most recent snapshot (200) was selected
        info_call = str(model_instance._info_msg.call_args)
        assert "snapshot: 200" in info_call
        assert result_config == project_path / "config.yaml"
        assert model_id.startswith("mdl-")


def test_dlc_strategy_localize_model_no_snapshots(pv2_train, tmp_path):
    """Test _localize_trained_model with no snapshots."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()
    model_instance._warn_msg = Mock()

    project_path = tmp_path / "test_project"
    project_path.mkdir()
    config_path = project_path / "config.yaml"
    config_path.touch()

    config = {"project_path": str(project_path)}

    with (
        patch(
            "deeplabcut.utils.auxiliaryfunctions.read_config",
            return_value={
                "TrainingFraction": [0.95],
                "shuffle": 1,
                "modelprefix": "",
            },
        ),
        patch(
            "deeplabcut.utils.get_model_folder",
            return_value="dlc-models/iteration-0/test-model",
        ),
    ):

        # Create training directory but no snapshot files
        model_dir = project_path / "dlc-models/iteration-0/test-model"
        train_dir = model_dir / "train"
        train_dir.mkdir(parents=True)

        result_config, model_id = strategy._localize_trained_model(
            config, model_instance
        )

        # Verify warning was logged and snapshot defaults to 0
        model_instance._warn_msg.assert_called_with(
            "No snapshot files found after training"
        )
        info_call = str(model_instance._info_msg.call_args)
        assert "snapshot: 0" in info_call


def test_dlc_strategy_localize_model_missing_directory(pv2_train, tmp_path):
    """Test _localize_trained_model error for missing training directory."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()

    project_path = tmp_path / "test_project"
    project_path.mkdir()
    config = {"project_path": str(project_path)}

    with (
        patch(
            "deeplabcut.utils.auxiliaryfunctions.read_config",
            return_value={
                "TrainingFraction": [0.95],
                "shuffle": 1,
                "modelprefix": "",
            },
        ),
        patch(
            "deeplabcut.utils.get_model_folder",
            return_value="dlc-models/iteration-0/test-model",
        ),
    ):

        # Don't create the training directory - should raise error
        with pytest.raises(
            FileNotFoundError, match="Training directory not found"
        ):
            strategy._localize_trained_model(config, model_instance)
