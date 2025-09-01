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

"""
BEAD (Behavior and Anomaly Detection) main module.

This module serves as the entry point for the BEAD framework, providing command-line
interface functionality for anomaly detection workflows. It orchestrates the complete
pipeline from data preparation to model training, inference, and result visualization.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # May not be strictly needed if using torchrun  # noqa: F401

from .src.utils import ggl


def main():
    """Process command-line arguments to execute BEAD functionality.

    Parses command-line arguments and executes the appropriate functionality based on the
    specified mode. The available modes are:

    Args:
        Arguments are parsed from command line, not passed directly to this function.

    Modes:
        new_project: Create a new project with default configuration.
        convert_csv: Convert CSV data to HDF5 or NPY format.
        prepare_inputs: Process data files into PyTorch tensors for training.
        train: Train a model using the prepared data.
        detect: Run inference on data to detect anomalies.
        plot: Generate visualization plots for results.
        diagnostics: Run performance profiling and diagnostics.
        chain: Execute the full pipeline from data conversion to visualization.

    Raises:
        NameError: If the specified mode is not recognized.
    """
    (
        config,
        mode,
        options,
        workspace_name,
        project_name,
        verbose,
    ) = ggl.get_arguments()

    # Initialize DDP if configured and multiple GPUs are available
    local_rank = 0
    world_size = 1
    is_ddp_active = False

    if (
        config
        and hasattr(config, "use_ddp")
        and config.use_ddp
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        try:
            # LOCAL_RANK and WORLD_SIZE are set by torchrun or torch.distributed.launch
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if (
                world_size > 1
            ):  # Proceed with DDP only if world_size indicates multiple processes
                print(
                    f"Initializing DDP: RANK {os.environ.get('RANK')}, LOCAL_RANK {local_rank}, WORLD_SIZE {world_size}"
                )
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", init_method="env://")
                is_ddp_active = True
                if local_rank == 0 and verbose:
                    print(
                        f"DDP initialized. World size: {world_size}. Running on {torch.cuda.device_count()} GPUs."
                    )
            else:
                if verbose:
                    print(
                        "DDP use_ddp is True, but world_size is 1. Running in non-DDP mode."
                    )
                # Fallback to non-DDP if world_size is 1 (e.g. launched without torchrun on single GPU)
                config.use_ddp = False

        except KeyError:
            print(
                "DDP environment variables (LOCAL_RANK, WORLD_SIZE) not set. Running in non-DDP mode."
            )
            config.use_ddp = False  # Fallback if env vars are missing
        except Exception as e:
            print(f"Error initializing DDP: {e}. Running in non-DDP mode.")
            config.use_ddp = False  # Fallback on any other DDP init error

    # Pass DDP status and ranks to helper functions if needed, or store in config
    if config:  # Ensure config is not None (e.g. for new_project mode)
        config.is_ddp_active = is_ddp_active
        config.local_rank = local_rank
        config.world_size = world_size

    # Set CUDNN benchmark for potential speedup if input sizes are consistent
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Define paths dict for the different paths used frequently in the pipeline
    paths = {
        "workspace_path": os.path.join("bead/workspaces", workspace_name),
        "project_path": os.path.join("bead/workspaces", workspace_name, project_name),
        "data_path": os.path.join("bead/workspaces", workspace_name, "data"),
        "output_path": os.path.join(
            "bead/workspaces", workspace_name, project_name, "output"
        ),
    }
    # Check what the options flag is set to and override the default if necessary
    if options == "h5" or options == "npy":
        config.file_type = options
    elif options == "overlay_roc":
        config.skip_to_roc = True

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
        ggl.run_plots(paths, config, verbose)
    elif mode == "diagnostics":
        ggl.run_diagnostics(paths, config, verbose)
    elif mode == "chain":
        ggl.run_full_chain(
            workspace_name, project_name, paths, config, options, verbose
        )
    else:
        raise NameError(
            "BEAD mode "
            + mode
            + " not recognised. Use < bead --help > to see available modes."
        )

    # Cleanup DDP
    if is_ddp_active:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
