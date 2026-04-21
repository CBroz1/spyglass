"""Direct test of DLCStrategy methods without database dependencies.

This test directly exercises the training methods by importing them
after mocking the database-dependent imports.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch


def test_dlc_strategy_methods():
    """Test DLCStrategy methods by mocking all database dependencies."""

    # Mock all the database-dependent modules before importing
    mock_modules = {
        "datajoint": Mock(),
        "spyglass.common": Mock(),
        "spyglass.common.common_device": Mock(),
        "spyglass.common.common_behav": Mock(),
        "spyglass.position.position_merge": Mock(),
        "spyglass.position": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        # Mock deeplabcut functions
        with (
            patch("deeplabcut.create_training_dataset") as mock_create,
            patch("deeplabcut.train_network") as mock_train,
            patch(
                "spyglass.position.utils.get_param_names",
                return_value=["batch_size"],
            ),
        ):

            # Now import the strategy class
            sys.path.insert(0, "src")
            from spyglass.position.utils.tool_strategies import DLCStrategy

            strategy = DLCStrategy()
            model_instance = Mock()
            model_instance._info_msg = Mock()

            # Test _prepare_training_dataset
            config_path = Path("/tmp/config.yaml")
            params = {"batch_size": 8, "invalid_param": "should_be_filtered"}
            config = {"project_path": "/tmp/project"}

            strategy._prepare_training_dataset(
                config_path, params, config, model_instance
            )

            # Verify the method was called correctly
            mock_create.assert_called_once_with(str(config_path), batch_size=8)
            print("✅ _prepare_training_dataset test passed")

            # Test _execute_training
            params = {"maxiters": "100", "shuffle": "1"}

            with patch(
                "spyglass.position.utils.get_param_names",
                return_value=["maxiters", "shuffle"],
            ):
                strategy._execute_training(config_path, params, model_instance)

                # Verify integer conversion
                call_args = mock_train.call_args[1]
                assert call_args["maxiters"] == 100
                assert call_args["shuffle"] == 1
                print("✅ _execute_training integer conversion test passed")

            print("🎉 All DLCStrategy method tests passed!")


if __name__ == "__main__":
    test_dlc_strategy_methods()
