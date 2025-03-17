# Think of this file as your google assistant
# This file is a collection of simple helper functions and is the control center that accesses all other src files

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from math import ceil
import gzip
import time

from tqdm.rich import tqdm
from art import *

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import conversion, data_processing, helper, plotting, diagnostics
from ..trainers import training, inference
from src.utils import ggl  # Ensure ggl is imported


def get_arguments():
    """Determines commandline arguments specified by BEAD user. Use `--help` to see what
    options are available.

    Returns: .py, string, folder: `.py` file containing the config options, string determining what mode to run,
    projects directory where outputs go.
    """
    parser = argparse.ArgumentParser(
        prog="bead",
        description=(
            text2art(" BEAD ", font="varsity")
            + "       /-----\\   /-----\\   /-----\\   /-----\\\n      /       \\ /       \\ /"
            "       \\ /       \\\n-----|         /         /         /         |-----\n      \\"
            "       / \\       / \\       / \\       /\n       \\-----/   \\-----/   \\-----/   \\"
            "-----/\n\n"
        ),
        #     "\n\n\nBEAD is a deep learning based anomaly detection algorithm for new Physics searches at the LHC.\n\n"
        #     "BEAD has five main running modes:\n\n"
        #     "\t1. Data handling: Deals with handling file types, conversions between them\n "
        #     "and pre-processing the data to feed as inputs to the DL models.\n\n"
        #     "\t2. Training: Train your model to learn implicit representations of your background\n "
        #     "data that may come from multiple sources/generators to get a single, encriched latent representation of it.\n\n"
        #     "\t3. Inference: Using a model trained on an enriched background, feed in samples you want\n "
        #     "to detect anomalies in using the '--detect or -d' mode.\n\n"
        #     "\t4. Plotting: After running Inference, or Training you can generate plots similar to\n "
        #     "what is shown in the paper. These include performance plots as well as different visualizations of the learned data.\n\n"
        #     "\t5. Diagnostics: Enabling this mode allows running profilers that measure a host of metrics connected\n "
        #     "to the usage of the compute node you run on to help you optimize the code if needed(using CPU-GPU metrics).\n\n\n"
        # ),
        # epilog="Enjoy!",
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
        help="Additional options for convert_csv mode [h5 (default), npy] or for chain mode (see help for chain mode)\n\n",
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
    project_path = os.path.join("workspaces", workspace_name, project_name)
    config_path = (
        f"workspaces.{workspace_name}.{project_name}.config.{project_name}_config"
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
    """Defines a configuration dataclass"""

    file_type: str
    parallel_workers: int
    num_jets: int
    num_constits: int
    latent_space_size: int
    normalizations: str
    invert_normalizations: bool
    train_size: float
    epochs: int
    early_stopping: bool
    early_stoppin_patience: int
    lr_scheduler: bool
    lr_scheduler_patience: int

    min_delta: int
    model_name: str
    input_level: str
    input_features: str
    model_init: str
    loss_function: str
    reg_param: float
    lr: float
    batch_size: int
    test_size: float
    intermittent_model_saving: bool
    separate_model_saving: bool
    intermittent_saving_patience: int
    activation_extraction: bool
    deterministic_algorithm: bool


def create_default_config(workspace_name: str, project_name: str) -> str:
    """Creates a default config file for a project.
    Args:
        workspace_name (str): Name of the workspace.
        project_name (str): Name of the project.
    Returns:
        str: Default config file.
    """

    return f"""
# === Configuration options ===

def set_config(c):
    c.file_type                    = "h5"
    c.parallel_workers             = 4
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




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.reg_param                    = 0.001
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

"""


def create_new_project(
    workspace_name: str,
    project_name: str,
    verbose: bool = False,
    base_path: str = "workspaces",
) -> None:
    """Creates a new project directory output subdirectories and config files within a workspace.

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
        os.path.join(project_path, "output", "plots", "training"),
        os.path.join(project_path, "output", "plots", "eval"),
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
    """Convert the input ''.csv' into the file_type selected in the config file ('.h5' by default)

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
    """Read the input data and generate torch tensors ready to train on.

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
                f"Error: No {config.file_type} files found in the directory '{input_path}'."
            )
            sys.exit()

    # Load data from files whose names contain the keyword
    in_path = output_path
    out_path = in_path + "/processed"
    keywords = ["bkg_train", "bkg_test"]

    for keyword in keywords:
        try:
            events_tensor, jets_tensor, constituents_tensor = helper.load_augment_tensors(
                in_path, keyword
            )
            if verbose:
                print("Data augmented successfully")
                print("Events tensor shape:", events_tensor.shape)
                print("Jets tensor shape:", jets_tensor.shape)
                print("Constituents tensor shape:", constituents_tensor.shape)
            if keyword == "bkg_train":
                torch.save(events_tensor, out_path + f"/bkg_train_events.pt")
                torch.save(jets_tensor, out_path + f"/bkg_train_jets.pt")
                torch.save(constituents_tensor, out_path + f"/bkg_train_constituents.pt")
            elif keyword == "bkg_test":
                torch.save(events_tensor, out_path + f"/bkg_test_genLabeled_events.pt")
                torch.save(jets_tensor, out_path + f"/bkg_test_genLabeled_jets.pt")
                torch.save(constituents_tensor, out_path + f"/bkg_test_genLabeled_constituents.pt")

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
        torch.save(events_tensor, out_path + f"/sig_test_events.pt")
        torch.save(jets_tensor, out_path + f"/sig_test_jets.pt")
        torch.save(constituents_tensor, out_path + f"/sig_test_constituents.pt")
    except ValueError as e:
        print(e)
        sys.exit(1)

    end = time.time()

    print("Finished preparing and saving input tensors")
    if verbose:
        print("Data preparation took:", f"{(end - start) / 60:.3} minutes")


def run_training(paths, config, verbose: bool = False):
    """Main function calling the training functions, ran when --mode=train is selected.
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

    # Output path
    output_path = os.path.join(paths["project_path"], "output")
    if verbose:
        print(f"Output path: {output_path}")

    trained_model = training.train(*data, output_path, config, verbose)

    print("Training complete")

    if config.separate_model_saving:
        helper.encoder_saver(
            trained_model, os.path.join(output_path, "models", "encoder.pt")
        )
        helper.decoder_saver(
            trained_model, os.path.join(output_path, "models", "decoder.pt")
        )

        if verbose:
            print(
                f"Encoder saved to {os.path.join(output_path, 'models', 'encoder.pt')}"
            )
            print(
                f"Decoder saved to {os.path.join(output_path, 'models', 'decoder.pt')}"
            )

    else:
        helper.save_model(
            trained_model, os.path.join(output_path, "models", "model.pt")
        )
    end = time.time()
    if verbose:
        # Print model save path
        print(f"Model saved to {os.path.join(output_path, 'models', 'model.pt')}")
        print("\nThe model has the following structure:")
        print(trained_model.type)
        # print time taken in hours
        print(f"The full training pipeline took: {(end - start) / 3600:.3} hours")


def run_inference(paths, config, verbose: bool = False):
    """Main function calling the training functions, ran when --mode=train is selected.
        The three functions called are: `process`, `ggl.mode_init` and `training.train`.

    Args:
        paths (dictionary): Dictionary of common paths used in the pipeline
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    
    start = time.time()

    print("Inference...")

    # Preprocess the data for training
    data_bkg = data_processing.preproc_inputs(paths, config, keyword="bkg_test", verbose=verbose)
    data_sig = data_processing.preproc_inputs(paths, config, keyword="sig_test", verbose=verbose)

    # Split Generator labels from bkg_test data
    data_bkg, gen_labels = helper.data_label_split(data_bkg+data_bkg)
    *data_bkg, _, _, _ = data_bkg
    *gen_labels, _, _, _ = gen_labels

    # Create bkg-sig labels
    data_bkg = helper.add_sig_bkg_label(data_bkg, label="bkg")
    data_sig = helper.add_sig_bkg_label(data_sig, label="sig")

    data = data_bkg + data_sig
    
    # Output path
    output_path = os.path.join(paths["project_path"], "output")
    model_path = os.path.join(output_path, "models", "model.pt")
    if verbose:
        print(f"Output path: {output_path}")
        print(f"Model path: {model_path}")

    done = False
    done = inference.infer(*data, model_path, output_path, config, verbose)
    
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


def run_plots(output_path, config, verbose: bool):
    """Main function calling the two plotting functions, ran when --mode=plot is selected.
       The two main functions this calls are: `ggl.plotter` and `ggl.loss_plotter`

    Args:
        prepare_inputt_path (string): Selects base path for determining output path
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints out more information
    """
    if verbose:
        print("Plotting...")
        print(f"Saving plots to {output_path}")
    ggl.loss_plotter(
        os.path.join(output_path,"results","loss_train.npy"), output_path, config
    )
    ggl.plotter(output_path, config)


def plotter(output_path, config):
    """Calls `plotting.plot()`

    Args:
        output_path (string): Path to the output directory
        config (dataClass): Base class selecting user inputs

    """

    plotting.plot(output_path, config)
    print("=== Done ===")
    print("Your plots are available in:", os.path.join(output_path, "plotting"))


def loss_plotter(path_to_loss_data, output_path, config):
    """Calls `plotting.loss_plot()`

    Args:
        path_to_loss_data (string): Path to the values for the loss plot
        output_path (string): Path to output the data
        config (dataClass): Base class selecting user inputs

    Returns:
        .pdf file: Plot containing the loss curves
    """
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def run_diagnostics(project_path, verbose: bool):
    """Calls diagnostics.diagnose()

    Args:
        input_path (str): path to the np.array contataining the activations values
        output_path (str): path to store the diagnostics pdf
    """

    output_path = os.path.join(project_path, "plotting")
    if verbose:
        print("Performing diagnostics")
        print(f"Saving plots to {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = os.path.join(project_path, "training", "activations.npy")
    diagnostics.nap_diagnose(input_path, output_path)


def run_full_chain(workspace_name: str, 
                 project_name: str, 
                 paths: dict, 
                 config: dict, 
                 options: str, 
                 verbose: bool = False) -> None:
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
        "diagnostics": "diagnostics"
    }

    # Map modes to their corresponding functions and arguments
    MODE_OPERATIONS = {
        "new_project": {
            "func": create_new_project,
            "args": (workspace_name, project_name, verbose)
        },
        "convert_csv": {
            "func": convert_csv,
            "args": (paths, config, verbose)
        },
        "prepare_inputs": {
            "func": prepare_inputs,
            "args": (paths, config, verbose)
        },
        "train": {
            "func": run_training,
            "args": (paths, config, verbose)
        },
        "detect": {
            "func": run_inference,
            "args": (paths, config, verbose)
        },
        "plot": {
            "func": run_plots,
            "args": (paths, config, verbose)
        },
        "diagnostics": {
            "func": run_diagnostics,
            "args": (paths, config, verbose)
        }
    }

    # Split and validate options
    workflow = options.split("_")
    valid_options = set(OPTION_TO_MODE.keys())
    
    for step in workflow:
        if step not in valid_options:
            raise ValueError(f"Invalid option '{step}' in options string. "
                             f"Valid options: {list(OPTION_TO_MODE.keys())}")

    # Execute workflow steps in sequence
    for step in workflow:
        mode_name = OPTION_TO_MODE[step]
        operation = MODE_OPERATIONS[mode_name]
        
        if verbose:
            print(f"\n{'='*40}")
            print(f"Executing step: {mode_name.replace('_', ' ').title()}")
            print(f"{'='*40}")
            
        # Execute the operation with its arguments
        operation["func"](*operation["args"])
        
        if verbose:
            print(f"Completed step: {mode_name.replace('_', ' ').title()}\n")

