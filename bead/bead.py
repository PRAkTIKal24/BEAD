# Copyright 2025 BEAD Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import gzip
from math import ceil

import numpy as np

from src.utils import ggl


__all__ = (
    "create_new_project",
    "convert_csv",
    "prepare_inputs",
    "run_training",
    "run_inference",
    "run_plots",
    "run_full_chain",
    "run_diagnostics",
)


def main():
    """Calls different functions depending on argument parsed in command line.

        - if --mode=new_project: call `ggl.create_new_project` and create a new project sub directory with default config file
        - if --mode=convert_csv: call `ggl.convert_csv` to convert csv to either hdf5 (default), npy or both formats
        - if --mode=prepare_inputs: call `ggl.prepare_inputs` to read either hdt5 (default) or npy and generate torch tensors ready to train on
        - if --mode=train: call `ggl.run_training` and train the network on given data and based on the config file and check if profilers are enabled
        - if --mode=detect: call 'ggl.run_inference'
        - if --mode=plot: call `ggl.run_plots` to generate all result plots described in the paper
        - if --mode=full_chain: call `ggl.run_full_chain` to run all the steps starting from processing the csv to generating result plots
        - if --mode=diagnostics: call `ggl.run_diagnostics` to run profilers to generate runtime metrics


    Raises:
        NameError: Raises error if the chosen mode does not exist.
    """
    (
        config,
        mode,
        options,
        workspace_name,
        project_name,
        verbose,
    ) = ggl.get_arguments()

    # Define paths dict for the different paths used frequently in the pipeline
    paths = {
        "workspace_path": os.path.join("workspaces", workspace_name),
        "project_path": os.path.join("workspaces", workspace_name, project_name),
        "data_path": os.path.join("workspaces", workspace_name, "data"),
        "output_path": os.path.join(
            "workspaces", workspace_name, project_name, "output"
        ),
    }

    # Check what the options flag is set to and override the default if necessary
    if options == "h5" or options == "npy":
        config.file_type = options

    # Call the appropriate ggl function based on the mode
    if mode == "new_project":
        ggl.create_new_project(workspace_name, project_name, verbose)
    elif mode == "convert_csv":
        ggl.convert_csv(paths, config, verbose)
    elif mode == "prepare_inputs":
        ggl.prepare_inputs(paths, config, verbose)
    elif mode == "train":
        ggl.run_training(paths, config, verbose)
    elif mode == "detect":
        ggl.run_inference(paths, config, verbose)
    elif mode == "plot":
        ggl.run_plots(paths["output_path"], config, verbose)
    elif mode == "diagnostics":
        ggl.run_diagnostics(paths, config, verbose)
    elif mode == "chain":
        ggl.run_full_chain(workspace_name, project_name, paths, config, options, verbose)
    else:
        raise NameError(
            "BEAD mode "
            + mode
            + " not recognised. Use < bead --help > to see available modes."
        )
