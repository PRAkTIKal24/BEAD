#!/usr/bin/env python3
"""
Unit tests for configuration creation and parameter handling in ggl.py.

These tests verify that the configuration string generation works correctly,
particularly with complex nested structures like annealing parameters.
"""

import unittest
import tempfile
import os
import sys
import importlib
from types import SimpleNamespace

# Add parent directory to path if needed for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import create_default_config directly
from bead.src.utils.ggl import create_default_config


class TestConfigCreation(unittest.TestCase):
    """Test the configuration creation functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.workspace_name = "test_workspace"
        self.project_name = "test_project"

    def test_create_default_config(self):
        """Test that create_default_config generates a valid configuration string."""
        config_string = create_default_config(self.workspace_name, self.project_name)

        # Verify basic configuration values are properly set
        self.assertIn(
            f'c.workspace_name               = "{self.workspace_name}"', config_string
        )
        self.assertIn(
            f'c.project_name                 = "{self.project_name}"', config_string
        )

        # Verify the annealing parameters section exists
        self.assertIn("c.annealing_params = {", config_string)
        self.assertIn('"reg_param": {', config_string)
        self.assertIn('"strategy": "TRIGGER_BASED"', config_string)

    def test_config_can_be_executed(self):
        """Test that the generated configuration string can be executed as Python code."""
        config_string = create_default_config(self.workspace_name, self.project_name)

        # Create a temporary file with the config
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(config_string)
            temp_file_path = temp_file.name

        try:
            # Load the module dynamically
            module_name = "temp_config_module"
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Create a mock config object and apply the configuration
            c = SimpleNamespace()
            module.set_config(c)

            # Verify the config object has the expected attributes
            self.assertEqual(c.workspace_name, self.workspace_name)
            self.assertEqual(c.project_name, self.project_name)
            self.assertEqual(c.file_type, "h5")
            self.assertEqual(c.num_jets, 3)

            # Verify annealing parameters are properly set
            self.assertTrue(hasattr(c, "annealing_params"))
            self.assertIn("reg_param", c.annealing_params)
            self.assertIn("contrastive_weight", c.annealing_params)

            # Check the structure of annealing parameters
            reg_param = c.annealing_params["reg_param"]
            self.assertEqual(reg_param["strategy"], "TRIGGER_BASED")
            self.assertEqual(reg_param["values"], [0.001, 0.005, 0.01])
            self.assertEqual(
                reg_param["trigger_source"], "early_stopper_third_patience"
            )
            self.assertEqual(reg_param["current_index"], 0)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_config_creation_filesystem(self):
        """Test the project config file creation, requires filesystem access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define test paths
            workspace_path = os.path.join(temp_dir, self.workspace_name)
            project_path = os.path.join(workspace_path, self.project_name)
            config_dir = os.path.join(project_path, "config")
            config_file_path = os.path.join(
                config_dir, f"{self.project_name}_config.py"
            )

            # Create directories
            os.makedirs(config_dir)

            # Write the config file
            with open(config_file_path, "w") as f:
                f.write(create_default_config(self.workspace_name, self.project_name))

            # Verify file exists
            self.assertTrue(os.path.exists(config_file_path))

            # Load and execute the config
            spec = importlib.util.spec_from_file_location(
                "test_config_module", config_file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Create a mock config object and apply the configuration
            c = SimpleNamespace()
            module.set_config(c)

            # Verify annealing parameters
            self.assertTrue(hasattr(c, "annealing_params"))
            self.assertIn("reg_param", c.annealing_params)
            self.assertIn("contrastive_weight", c.annealing_params)


if __name__ == "__main__":
    unittest.main()
