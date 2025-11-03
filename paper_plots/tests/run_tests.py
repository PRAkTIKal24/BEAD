#!/usr/bin/env python3
"""
Test runner script for the BEAD package.
This script can be used to run tests from the command line,
or it can be integrated into a CI/CD pipeline.

Example usage:
    python -m tests.run_tests              # Run all tests
    python -m tests.run_tests unit         # Run only unit tests
    python -m tests.run_tests unit.annealing # Run only annealing tests
"""

import unittest
import sys
import os


def run_tests(test_path=None):
    """
    Run tests specified by test_path.

    Args:
        test_path: Dot-separated path to the test module or package to run.
                  If None, all tests will be run.

    Returns:
        True if all tests pass, False otherwise.
    """
    # Make sure the current directory is in the path to import the modules
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    if test_path is None:
        # Run all tests
        test_suite = unittest.defaultTestLoader.discover("tests")
    else:
        # Handle different test path formats
        if test_path == "unit.annealing":
            # Direct file path for common case
            file_path = os.path.join(current_dir, "tests", "unit", "test_annealing.py")
            test_suite = unittest.defaultTestLoader.discover(
                os.path.dirname(file_path), pattern=os.path.basename(file_path)
            )
        elif test_path == "unit":
            # All unit tests
            test_suite = unittest.defaultTestLoader.discover(
                os.path.join(current_dir, "tests", "unit")
            )
        else:
            # Convert path format: e.g., "unit.annealing" -> "tests.unit.test_annealing"
            if "." in test_path:
                components = test_path.split(".")
                if len(components) >= 2:
                    module_path = f"tests.{components[0]}.test_{components[1]}"
                else:
                    module_path = f"tests.{test_path}"
            else:
                module_path = f"tests.{test_path}"

            try:
                test_suite = unittest.defaultTestLoader.loadTestsFromName(module_path)
            except (ImportError, AttributeError) as e:
                print(f"Error loading tests from {module_path}: {e}")
                return False

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return True if all tests passed, False otherwise
    return result.wasSuccessful()


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = None

    # Run the tests
    success = run_tests(test_path)

    # Set exit code based on test results (useful for CI/CD pipelines)
    sys.exit(0 if success else 1)
