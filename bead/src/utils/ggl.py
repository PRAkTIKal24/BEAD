"""
Control center for BEAD command-line interface.

This module serves as the main entry point for the BEAD CLI. It handles command-line arguments, project creation, and orchestrates the execution of various modes like data conversion, training, inference, and visualization.
Think of this file as your google assistant
This file is a collection of simple helper functions and is the control center that accesses all other src files

Functions:
    get_arguments: Parse command-line arguments.
    create_default_config: Create default configuration file.
    create_new_project: Create directory structure for new project.
    convert_csv: Convert CSV files to HDF5 or NumPy format.
    prepare_inputs: Process input data and create tensors.
    run_training: Execute model training pipeline.
    run_inference: Execute model inference pipeline.
    run_plots: Generate plots from results.
    run_diagnostics: Run model diagnostics.
    run_full_chain: Execute a sequence of operations.

Classes:
    Config: Dataclass for storing configuration settings.
"""

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass

import art as ar
import numpy as np
import torch

# from art import *
from tqdm.rich import tqdm

from ..trainers import inference, training
from . import conversion, data_processing, diagnostics, helper, plotting


def get_arguments():
    """
    Determines commandline arguments specified by BEAD user. Use `--help` to see what
    options are available.

    Returns: .py, string, folder: `.py` file containing the config options, string determining what mode to run,
    projects directory where outputs go.
    """
    parser = argparse.ArgumentParser(
        prog="bead",
        description=(
            ar.text2art(" BEAD ", font="varsity")  # noqa: F405
            + "       /-----\\   /-----\\   /-----\\   /-----\\\n      /       \\ /       \\ /"
            "       \\ /       \\\n-----|         /         /         /         |-----\n      \\"
            "       / \\       / \\       / \\       /\n       \\-----/   \\-----/   \\-----/   \\"
            "-----/\n\n"
        ),
        epilog="Happy Hunting!",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        help="new_project \t creates new workspace and project directories\n\t\t"
        " as explained by the '--project' flag and sets default configs\n\n"
        "convert_csv \t converts input csv into numpy or hdf5 format as chosen in the configs\n\n"
        "prepare_inputs \t runs 'convert_csv' mode if numpy/hdf5 files dont already exist.\n\t\t Then reads the produced files,"
        "converts to tensors\n\t\t and applies required data processing methods as required\n\n"
        "train \t\t runs the training mode using hyperparameters specified in the configs.\n\t\t "
        "Trains the model on the processed data and saves the model\n\n"
        "detect \t\t runs the inference mode using the trained model. Detects anomalies in the data and saves the results\n\n"
        "plot \t\t runs the plotting mode using the results from the detect or train mode.\n\t\t"
        " Generates plots as per the paper and saves them\n\n"
        "chain \t\t runs all modes (except new_project) in the sequence prescribed by the <-o> or <--options> flag.\n\t\t"
        " For example, when using <-m chain>, when you set <-o convertcsv_prepareinputs_train_detect>\n\t\t"
        " it will run the convert_csv, prepare_inputs, train and detect modes in sequence.\n\n"
        "diagnostics \t runs the diagnostics mode. Generates runtime metrics using profilers\n\n",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        required=True,
        nargs=2,
        metavar=("WORKSPACE", "PROJECT"),
        help="Specifies workspace and project.\n"
        "e.g. < --project SVJ firstTry >"
        ", specifies workspace 'SVJ' and project 'firstTry'\n\n"
        "When combined with new_project mode:\n"
        "  1. If workspace and project exist, take no action.\n"
        "  2. If workspace exists but project does not, create project in workspace.\n"
        "  3. If workspace does not exist, create workspace directory and project.\n\n",
    )
    parser.add_argument(
        "-o",
        "--options",
        type=str,
        required=False,
        help="Additional options for convert_csv mode [h5 (default), npy], overlaying roc plots or for chain mode (see help for chain mode)\n\n",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    workspace_name = args.project[0]
    project_name = args.project[1]
    project_path = os.path.join("bead/workspaces", workspace_name, project_name)
    config_path = (
        f"bead.workspaces.{workspace_name}.{project_name}.config.{project_name}_config"
    )

    if args.mode == "new_project":
        config = None
    else:
        # Check if proejct path exists
        if not os.path.exists(project_path):
            print(
                f"Project path {project_path} does not exist. Please run --mode=new_project first."
            )
            sys.exit()
        else:
            config = Config
            importlib.import_module(config_path).set_config(config)

    return (
        config,
        args.mode,
        args.options,
        workspace_name,
        project_name,
        args.verbose,
    )


@dataclass
class Config:
    """
    Defines a configuration dataclass
    """

    workspace_name: str
    project_name: str
    file_type: str
    parallel_workers: int
    chunk_size: int
    num_jets: int
    num_constits: int
    latent_space_size: int
    normalizations: str
    invert_normalizations: bool
    train_size: float
    model_name: str
    input_level: str
    input_features: str
    model_init: str
    loss_function: str
    optimizer: str
    epochs: int
    lr: float
    batch_size: int
    early_stopping: bool
    early_stoppin_patience: int
    lr_scheduler: bool
    lr_scheduler_patience: int
    latent_space_plot_style: str
    subsample_plot: bool

    use_ddp: bool  # To toggle torch Distributed Data Parallel
    use_amp: bool  # To toggle torch Automatic Mixed Precision
    min_delta: int
    reg_param: float
    intermittent_model_saving: bool
    intermittent_saving_patience: int
    activation_extraction: bool
    deterministic_algorithm: bool
    separate_model_saving: bool
    subsample_size: int
    contrastive_temperature: float
    contrastive_weight: float
    overlay_roc: bool
    overlay_roc_projects: list
    overlay_roc_save_location: str
    overlay_roc_filename: str

    # NT-Xent loss parameters
    ntxent_sigma: float = 0.1  # Standard deviation for naive gaussian smearing strategy


def create_default_config(workspace_name: str, project_name: str) -> str:
    """
    Creates a default config file for a project.
    Args:
        workspace_name (str): Name of the workspace.
        project_name (str): Name of the project.
    Returns:
        str: Default config file.
    """

    return f"""
# === Configuration options ===

def set_config(c):
    c.workspace_name               = "{workspace_name}"
    c.project_name                 = "{project_name}"
    c.file_type                    = "h5"
    c.parallel_workers             = 16
    c.chunk_size                   = 10000
    c.num_jets                     = 3
    c.num_constits                 = 15
    c.latent_space_size            = 15
    c.normalizations               = "pj_custom"
    c.invert_normalizations        = False
    c.train_size                   = 0.95
    c.model_name                   = "Planar_ConvVAE"
    c.input_level                  = "constituent"
    c.input_features               = "4momentum_btag"
    c.model_init                   = "xavier"
    c.loss_function                = "VAEFlowLoss"
    c.optimizer                    = "adamw"
    c.epochs                       = 2
    c.lr                           = 0.001
    c.batch_size                   = 2
    c.early_stopping               = True
    c.lr_scheduler                 = True
    c.latent_space_plot_style      = "umap"
    c.subsample_plot               = False
    c.skip_to_roc                  = False


# === NT-Xent loss configuration ===
    c.ntxent_sigma                 = 0.1
    c.lr_scheduler_patience        = 30
    c.reg_param                    = 0.001
    c.intermittent_model_saving    = True
    c.intermittent_saving_patience = 100
    c.subsample_size               = 300000
    c.contrastive_temperature      = 0.07
    c.contrastive_weight           = 0.005
    c.overlay_roc                  = False
    c.overlay_roc_projects         = ["workspace_name/project_name"]
    c.overlay_roc_save_location    = "overlay_roc"
    c.overlay_roc_filename         = "combined_roc.pdf"

# === Parameter annealing configuration ===
    c.annealing_params = {{
        "reg_param": {{
            "strategy": "TRIGGER_BASED",
            "values": [0.001, 0.005, 0.01],
            "trigger_source": "early_stopper_third_patience",
            "current_index": 0
        }},
        "contrastive_weight": {{
            "strategy": "TRIGGER_BASED",
            "values": [0, 0.005, 0.01, 0.02, 0.03],
            "trigger_source": "early_stopper_half_patience",
            "current_index": 0
        }}
    }}
# === Additional configuration options ===

    c.use_ddp                      = False
    c.use_amp                      = False
    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

"""


def create_new_project(
    workspace_name: str,
    project_name: str,
    verbose: bool = False,
    base_path: str = "bead/workspaces",
) -> None:
    """
    Creates a new project directory output subdirectories and config files within a workspace.

    Args:
        workspace_name (str): Creates a workspace (dir) for storing data and projects with this name.
        project_name (str): Creates a project (dir) for storing configs and outputs with this name.
        verbose (bool, optional): Whether to print out the progress. Defaults to False.
    """

    # Create full project path
    workspace_path = os.path.join(base_path, workspace_name)
    project_path = os.path.join(base_path, workspace_name, project_name)
    if os.path.exists(project_path):
        print(f"The workspace and project ({project_path}) already exists.")
        return
    os.makedirs(project_path)

    # Create required directories
    required_directories = [
        os.path.join(workspace_path, "data", "csv"),
        os.path.join(workspace_path, "data", "h5", "tensors", "processed"),
        os.path.join(workspace_path, "data", "npy", "tensors", "processed"),
        os.path.join(project_path, "config"),
        os.path.join(project_path, "output", "results"),
        os.path.join(project_path, "output", "plots", "latent_space"),
        os.path.join(project_path, "output", "plots", "loss"),
        os.path.join(project_path, "output", "plots", "overlay_roc"),
        os.path.join(project_path, "output", "models"),
    ]

    if verbose:
        print(f"Creating project {project_name} in workspace {workspace_name}...")
    for directory in tqdm(required_directories, desc="Creating directories: "):
        if verbose:
            print(f"Creating directory {directory}...")
        os.makedirs(directory, exist_ok=True)

    # Populate default config
    with open(
        os.path.join(project_path, "config", f"{project_name}_config.py"), "w"
    ) as f:
        f.write(create_default_config(workspace_name, project_name))


def convert_csv(paths, config, verbose: bool = False):
    """
    Convert the input ''.csv' into the file_type selected in the config file ('.h5' by default)

        Separate event-level, jet-level and constituent-level data into separate datasets/files.

    Args:
        data_path (path): Path to the input csv files
        output_path (path): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information

    Outputs:
        A `ProjectName_OutputPrefix.h5` file which includes:
        - Event-level dataset
        - Jet-level dataset
        - Constituent-level dataset

        or

        A `ProjectName_OutputPrefix_{data-level}.npy` files which contain the same information as above, split into 3 separate files.
    """
    start = time.time()
    print("Converting csv to " + config.file_type + "...")

    # Required paths
    input_path = os.path.join(paths["data_path"], "csv")
    output_path = os.path.join(paths["data_path"], config.file_type)

    if not os.path.exists(input_path):
        print(
            f"Directory {input_path} does not exist. Check if you have downloaded the input csv files correctly and moved them to this location"
        )

    else:
        csv_files_not_found = True
        # List all files in the folder
        for file_name in tqdm(os.listdir(input_path), desc="Converting files: "):
            # Check if the file is a CSV file
            if file_name.endswith(".csv"):
                # Construct the full file path
                csv_file_path = os.path.join(input_path, file_name)
                # Get the base name of the file (without path) and remove the .csv extension
                output_prefix = os.path.splitext(file_name)[0]
                # Call the conversion function
                conversion.convert_csv_to_hdf5_npy_parallel(
                    csv_file=csv_file_path,
                    output_prefix=output_prefix,
                    out_path=output_path,
                    file_type=config.file_type,
                    chunk_size=config.chunk_size,
                    n_workers=config.parallel_workers,
                    verbose=verbose,
                )
                # Set the flag to True since at least one CSV file was found
                csv_files_not_found = False

        # Check if no CSV files were found
        if csv_files_not_found:
            print(f"Error: No CSV files found in the directory '{input_path}'.")
            sys.exit()

    end = time.time()

    print("Finished converting csv to " + config.file_type)
    if verbose:
        print("Conversion took:", f"{(end - start) / 60:.3} minutes")


def prepare_inputs(paths, config, verbose: bool = False):
    """
    Read the input data and generate torch tensors ready to train on.

    Select number of leading jets per event and number of leading constituents per jet to be used for training.

    Args:
        paths: Dictionary of common paths used in the pipeline
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information

    Outputs:
        Tensor files which include:
        - Event-level dataset - [evt_id, evt_weight, met, met_phi, num_jets]
        - Jet-level dataset - [evt_id, jet_id, num_constituents, jet_btag, jet_pt, jet_eta, jet_phi]
        - Constituent-level dataset - [evt_id, jet_id, constituent_id, jet_btag, constituent_pt, constituent_eta, constituent_phi]
    """
    print("Preparing input tensors...")
    start = time.time()
    input_path = os.path.join(paths["data_path"], config.file_type)
    output_path = os.path.join(paths["data_path"], config.file_type, "tensors")

    if not os.path.exists(input_path):
        print(
            f"Directory {input_path} does not exist. Make sure to run --mode = create_new_project first."
        )
    else:
        files_not_found = True
        # List all files in the folder
        for file_name in tqdm(os.listdir(input_path), desc="Preparing tensors: "):
            # Check if the file is a HDF5 file
            if file_name.endswith(config.file_type):
                # Get the base name of the file (without path) and remove the .h5 extension
                output_prefix = os.path.splitext(file_name)[0]
                # Construct the full file path
                input_file_path = os.path.join(input_path, file_name)
                # Call the selection function
                data_processing.process_and_save_tensors(
                    in_path=input_file_path,
                    out_path=output_path,
                    output_prefix=output_prefix,
                    config=config,
                    verbose=verbose,
                )
                # Set the flag to False since at least one HDF5 file was found
                files_not_found = False

        # Check if no HDF5 files were found
        if files_not_found:
            print(
                f"Error: No {config.file_type} files found in the directory '{input_path}'. Run --mode=convert_csv first."
            )
            sys.exit()

    # Load data from files whose names contain the keyword
    in_path = output_path
    out_path = in_path + "/processed"
    keywords = ["bkg_train", "bkg_test"]

    for keyword in keywords:
        try:
            events_tensor, jets_tensor, constituents_tensor = (
                helper.load_augment_tensors(in_path, keyword)
            )
            if verbose:
                print(f"Data augmented successfully for {keyword} files")
                print("Events tensor shape:", events_tensor.shape)
                print("Jets tensor shape:", jets_tensor.shape)
                print("Constituents tensor shape:", constituents_tensor.shape)
            if keyword == "bkg_train":
                torch.save(events_tensor, out_path + "/bkg_train_events.pt")
                torch.save(jets_tensor, out_path + "/bkg_train_jets.pt")
                torch.save(constituents_tensor, out_path + "/bkg_train_constituents.pt")
            elif keyword == "bkg_test":
                torch.save(events_tensor, out_path + "/bkg_test_genLabeled_events.pt")
                torch.save(jets_tensor, out_path + "/bkg_test_genLabeled_jets.pt")
                torch.save(
                    constituents_tensor,
                    out_path + "/bkg_test_genLabeled_constituents.pt",
                )

        except ValueError as e:
            print(e)
            sys.exit(1)

    keyword = "sig_test"
    try:
        events_tensor, jets_tensor, constituents_tensor = helper.load_tensors(
            in_path, keyword
        )
        if verbose:
            print("Data augmented successfully")
            print("Events tensor shape:", events_tensor.shape)
            print("Jets tensor shape:", jets_tensor.shape)
            print("Constituents tensor shape:", constituents_tensor.shape)
        torch.save(events_tensor, out_path + "/sig_test_events.pt")
        torch.save(jets_tensor, out_path + "/sig_test_jets.pt")
        torch.save(constituents_tensor, out_path + "/sig_test_constituents.pt")
    except ValueError as e:
        print(e)

    end = time.time()

    print("Finished preparing and saving input tensors")
    if verbose:
        print("Data preparation took:", f"{(end - start) / 60:.3} minutes")


def run_training(paths, config, verbose: bool = False):
    """
    Main function calling the training functions, ran when --mode=train is selected.
    The three functions called are: 'data_processing.preproc_inputs' and `training.train`.

    Args:
        paths (dictionary): Dictionary of common paths used in the pipeline
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    start = time.time()

    print("Training...")

    keyword = "bkg_train"

    # Preprocess the data for training
    data = data_processing.preproc_inputs(paths, config, keyword, verbose)
    *data_train, _, _, _ = data
    _, _, _, *data_val = data

    # Split Generator labels from train data
    data_train, gen_labels_train = helper.data_label_split(data_train)

    # Save train generator labels
    labels_path = os.path.join(
        paths["data_path"], config.file_type, "tensors", "processed"
    )
    gen_label_events, gen_label_jets, gen_label_constituents = gen_labels_train
    np.save(
        os.path.join(labels_path, "train_gen_label_event.npy"),
        gen_label_events.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(labels_path, "train_gen_label_jet.npy"),
        gen_label_jets.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(labels_path, "train_gen_label_constituent.npy"),
        gen_label_constituents.detach().cpu().numpy(),
    )

    # Split Generator labels from val data
    data_val, gen_labels_val = helper.data_label_split(data_val)

    data = data_train + data_val
    gen_labels = gen_labels_train + gen_labels_val

    # Output path
    output_path = os.path.join(paths["project_path"], "output")
    if verbose:
        print(f"Output path: {output_path}")

    trained_model = training.train(data, gen_labels, output_path, config, verbose)

    print("Training complete")

    end = time.time()

    if verbose:
        # Print model save path
        print(f"Model saved to {os.path.join(output_path, 'models', 'model.pt')}")
        print("\nThe model has the following structure:")
        print(trained_model.type)
        # print time taken in hours
        print(f"The full training pipeline took: {(end - start) / 3600:.3} hours")


def run_inference(paths, config, verbose: bool = False):
    """
    Main function calling the training functions, ran when --mode=train is selected.
    The three functions called are: `process`, `ggl.mode_init` and `training.train`.

    Args:
        paths (dictionary): Dictionary of common paths used in the pipeline
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """

    start = time.time()

    print("Inference...")

    # Preprocess the data for training
    data_bkg = data_processing.preproc_inputs(
        paths, config, keyword="bkg_test", verbose=verbose
    )
    data_sig = data_processing.preproc_inputs(
        paths, config, keyword="sig_test", verbose=verbose
    )

    # Split Generator labels from bkg_test data
    data_bkg, gen_labels = helper.data_label_split(data_bkg)

    # Save generator labels
    labels_path = os.path.join(
        paths["data_path"], config.file_type, "tensors", "processed"
    )
    gen_label_events, gen_label_jets, gen_label_constituents = gen_labels
    np.save(
        os.path.join(labels_path, "test_gen_label_event.npy"),
        gen_label_events.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(labels_path, "test_gen_label_jet.npy"),
        gen_label_jets.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(labels_path, "test_gen_label_constituent.npy"),
        gen_label_constituents.detach().cpu().numpy(),
    )
    if verbose:
        print("Generator labels saved")

    # Create bkg-sig labels
    data_bkg = helper.add_sig_bkg_label(data_bkg, label="bkg")
    data_sig = helper.add_sig_bkg_label(data_sig, label="sig")

    # Output path
    output_path = os.path.join(paths["project_path"], "output")
    model_path = os.path.join(output_path, "models", "model.pt")
    if verbose:
        print(f"Output path: {output_path}")
        print(f"Model path: {model_path}")

    done = False
    done = inference.infer(data_bkg, data_sig, model_path, output_path, config, verbose)

    end = time.time()

    if done:
        print("Inference complete")
        if verbose:
            # Print output save path
            print(f"Outputs saved to {os.path.join(output_path, 'results')}")

            # print time taken in hours
            print(f"The full inference pipeline took: {(end - start) / 3600:.3} hours")
    else:
        print("Inference failed")


def run_plots(paths, config, verbose: bool = False):
    """
    Main function calling the plotting functions, ran when --mode=plot is selected.
    The main functions this calls are: `plotting.plot_losses`, `plotting.plot_latent_variables`,
    plotting.plot_mu_logvar and `plotting.plot_roc_curve`.

    Args:
        paths (dictionary): Dictionary of common paths used in the pipeline
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    input_path = os.path.join(paths["output_path"], "results")
    output_path = os.path.join(paths["output_path"], "plots", "loss")

    if not config.skip_to_roc:
        try:
            plotting.plot_losses(input_path, output_path, config, verbose)
        except FileNotFoundError as e:
            print(e)
        try:
            plotting.plot_latent_variables(config, paths, verbose)
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Error plotting latent variables: {e}")
        try:
            plotting.plot_mu_logvar(config, paths, verbose)
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"Error plotting mu/logvar: {e}")
    try:
        plotting.plot_roc_curve(config, paths, verbose)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

    print("Plotting complete")


def run_diagnostics(project_path, verbose: bool):
    """
    Calls diagnostics.diagnose()

    Args:
        input_path (str): path to the np.array contataining the activations values
        output_path (str): path to store the diagnostics pdf
    """

    output_path = os.path.join(project_path, "plotting")
    if verbose:
        print("Performing diagnostics")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = os.path.join(project_path, "training", "activations.npy")
    diagnostics.nap_diagnose(input_path, output_path, verbose)
    if verbose:
        print("Diagnostics complete")


def run_full_chain(
    workspace_name: str,
    project_name: str,
    paths: dict,
    config: dict,
    options: str,
    verbose: bool = False,
) -> None:
    """
    Execute a sequence of operations based on the provided options string.

    Args:
        workspace_name: Name of the workspace for new projects
        project_name: Name of the project for new projects
        paths: Dictionary of file paths and directories
        config: Configuration dictionary for operations
        options: Underscore-separated string specifying the workflow sequence
        verbose: Whether to show verbose output

    Example:
        run_full_chain("my_workspace", "my_project", paths, config,
                      "newproject_convertcsv_prepareinputs_train_detect", verbose=True)
    """
    # Map option components to mode names and execution order
    OPTION_TO_MODE = {
        "newproject": "new_project",
        "convertcsv": "convert_csv",
        "prepareinputs": "prepare_inputs",
        "train": "train",
        "detect": "detect",
        "plot": "plot",
        "diagnostics": "diagnostics",
    }

    # Map modes to their corresponding functions and arguments
    MODE_OPERATIONS = {
        "new_project": {
            "func": create_new_project,
            "args": (workspace_name, project_name, verbose),
        },
        "convert_csv": {"func": convert_csv, "args": (paths, config, verbose)},
        "prepare_inputs": {"func": prepare_inputs, "args": (paths, config, verbose)},
        "train": {"func": run_training, "args": (paths, config, verbose)},
        "detect": {"func": run_inference, "args": (paths, config, verbose)},
        "plot": {"func": run_plots, "args": (paths, config, verbose)},
        "diagnostics": {"func": run_diagnostics, "args": (paths, config, verbose)},
    }

    # Split and validate options
    workflow = options.split("_")
    valid_options = set(OPTION_TO_MODE.keys())

    for step in workflow:
        if step not in valid_options:
            raise ValueError(
                f"Invalid option '{step}' in options string. "
                f"Valid options: {list(OPTION_TO_MODE.keys())}"
            )

    # Execute workflow steps in sequence
    for step in workflow:
        mode_name = OPTION_TO_MODE[step]
        operation = MODE_OPERATIONS[mode_name]

        if verbose:
            print(f"\n{'=' * 40}")
            print(f"Executing step: {mode_name.replace('_', ' ').title()}")
            print(f"{'=' * 40}")

        # Execute the operation with its arguments
        operation["func"](*operation["args"])

        if verbose:
            print(f"Completed step: {mode_name.replace('_', ' ').title()}\n")
