# Copyright 2022 Baler Contributors

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
import random
import time
import sys
import numpy as np
from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, ConcatDataset

from ..utils import helper, loss, diagnostics


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def seed_worker(worker_id):
    """PyTorch implementation to fix the seeds
    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def infer(
    data_bkg,
    data_sig,
    model_path,
    output_path,
    config,
    verbose: bool = False,
):
    """Does the entire training loop by calling the `fit()` and `validate()`. Appart from this, this is the main function where the data is converted
       to the correct type for it to be trained, via `torch.Tensor()`. Furthermore, the batching is also done here, based on `config.batch_size`,
       and it is the `torch.utils.data.DataLoader` doing the splitting.
       Applying either `EarlyStopping` or `LR Scheduler` is also done here, all based on their respective `config` arguments.
       For reproducibility, the seeds can also be fixed in this function.

    Args:
        model (modelObject): The model you wish to train
        data_bkg (Tuple): Tuple containing the background data
        data_sig (Tuple): Tuple containing the signal data
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs

    Returns:
        modelObject: fully trained model ready to perform compression and decompression
    """

    # Get the device and move tensors to the device
    device = helper.get_device()

    (
        events_bkg,
        jets_bkg,
        constituents_bkg,
        events_sig,
        jets_sig,
        constituents_sig,
    ) = [x.to(device) for x in data_bkg + data_sig]

    data_bkg = (
        events_bkg,
        jets_bkg,
        constituents_bkg,
    )

    data_sig = (
        events_sig,
        jets_sig,
        constituents_sig,
    )

    # Split data and labels
    if verbose:
        print("Splitting data and labels")
    data_bkg, labels_bkg = helper.data_label_split(data_bkg)
    data_sig, labels_sig = helper.data_label_split(data_sig)

    (
        events_bkg,
        jets_bkg,
        constituents_bkg,
        events_sig,
        jets_sig,
        constituents_sig,
    ) = (
        data_bkg + data_sig
    )

    (
        events_bkg_label,
        jets_bkg_label,
        constituents_bkg_label,
        events_sig_label,
        jets_sig_label,
        constituents_sig_label,
    ) = (
        labels_bkg + labels_sig
    )

    # Save labels
    np.save(
        os.path.join(output_path, "results", "test_event_label.npy"),
        np.concatenate([events_bkg_label, events_sig_label]),
    )
    np.save(
        os.path.join(output_path, "results", "test_jet_label.npy"),
        np.concatenate([jets_bkg_label, jets_sig_label]),
    )
    np.save(
        os.path.join(output_path, "results", "test_constituent_label.npy"),
        np.concatenate([constituents_bkg_label, constituents_sig_label]),
    )

    # Reshape tensors to pass to conv layers
    if "ConvVAE" in config.model_name or "ConvAE" in config.model_name:
        (
            events_bkg,
            jets_bkg,
            constituents_bkg,
            events_sig,
            jets_sig,
            constituents_sig,
        ) = [
            x.unsqueeze(1).float()
            for x in [
                events_bkg,
                jets_bkg,
                constituents_bkg,
                events_sig,
                jets_sig,
                constituents_sig,
            ]
        ]

    data = (
        events_bkg,
        jets_bkg,
        constituents_bkg,
        events_sig,
        jets_sig,
        constituents_sig,
    )

    labels = (
        events_bkg_label,
        jets_bkg_label,
        constituents_bkg_label,
        events_sig_label,
        jets_sig_label,
        constituents_sig_label,
    )

    # Create datasets
    ds = helper.create_datasets(*data, *labels)

    # Concatenate events, jets and constituents respectively with their labels (here val is labels)
    ds_events = ConcatDataset([ds["events_train"], ds["events_val"]])
    ds_jets = ConcatDataset([ds["jets_train"], ds["jets_val"]])
    ds_constituents = ConcatDataset([ds["constituents_train"], ds["constituents_val"]])
    ds = {
        "events": ds_events,
        "jets": ds_jets,
        "constituents": ds_constituents,
    }

    if verbose:
        # Print input shapes
        print("Events - bkg shape:         ", events_bkg.shape)
        print("Jets - bkg shape:           ", jets_bkg.shape)
        print("Constituents - bkg shape:   ", constituents_bkg.shape)
        print("Events - sig shape:         ", events_sig.shape)
        print("Jets - sig shape:           ", jets_sig.shape)
        print("Constituents - sig shape:   ", constituents_sig.shape)

        # Print label shapes
        print("Events - bkg labels shape:         ", events_bkg_label.shape)
        print("Jets - bkg labels shape:           ", jets_bkg_label.shape)
        print("Constituents - bkg labels shape:   ", constituents_bkg_label.shape)
        print("Events - sig labels shape:         ", events_sig_label.shape)
        print("Jets - sig labels shape:           ", jets_sig_label.shape)
        print("Constituents - sig labels shape:   ", constituents_sig_label.shape)

    # Calculate the input shapes to load the model
    in_shape = helper.calculate_in_shape(data, config, test_mode=True)

    # Load the model and set to eval mode for inference
    model = helper.load_model(model_path=model_path, in_shape=in_shape, config=config)
    model.eval()

    if verbose:
        print(f"Model loaded from {model_path}")
        print(f"Model architecture:\n{model}")
        print(f"Device used for inference: {device}")
        print(f"Inputs and model moved to device")
        # Pushing input data into the torch-DataLoader object and combines into one DataLoader object (a basic wrapper
        # around several DataLoader objects).
        print("Loading data into DataLoader and using batch size of ", 1)

    if config.deterministic_algorithm:
        if config.verbose:
            print("Deterministic algorithm is set to True")
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(0)

        test_dl_list = [
            DataLoader(
                ds,
                batch_size=1,  # since we want the loss for every event, which then becomes the anomaly metric
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=True,
                num_workers=config.parallel_workers,
            )
            for ds in [ds["events"], ds["jets"], ds["constituents"]]
        ]

    else:
        test_dl_list = [
            DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                num_workers=config.parallel_workers,
            )
            for ds in [ds["events"], ds["jets"], ds["constituents"]]
        ]

    # Unpacking the DataLoader lists
    test_dl_events, test_dl_jets, test_dl_constituents = test_dl_list

    if config.model_name == "pj_ensemble":
        if verbose:
            print("Model is an ensemble model")
    else:
        if config.input_level == "event":
            test_dl = test_dl_events
        elif config.input_level == "jet":
            test_dl = test_dl_jets
        elif config.input_level == "constituent":
            test_dl = test_dl_constituents
        if verbose:
            print(f"Input data is of {config.input_level} level")

    # Select Loss Function
    try:
        loss_object = helper.get_loss(config.loss_function)
        loss_fn = loss_object(config=config)
        if verbose:
            print(f"Loss Function: {config.loss_function}")
    except ValueError as e:
        print(e)

    # Output Lists
    test_loss_data = []
    reconstructed_data = []
    mu_data = []
    logvar_data = []
    z0_data = []
    zk_data = []
    log_det_jacobian_data = []

    start = time.time()

    # Registering hooks for activation extraction
    if config.activation_extraction:
        hooks = model.store_hooks()

    if verbose:
        print(f"Beginning Inference")

    # Inference
    parameters = model.parameters()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):

            inputs, labels = batch

            out = helper.call_forward(model, inputs)
            recon, mu, logvar, ldj, z0, zk = out

            # Compute the loss
            losses = loss_fn.calculate(
                recon=recon,
                target=inputs,
                mu=mu,
                logvar=logvar,
                parameters=parameters,
                log_det_jacobian=0,
            )

            test_loss_data.append(losses)
            reconstructed_data.append(recon.detach().cpu().numpy())
            mu_data.append(mu.detach().cpu().numpy())
            logvar_data.append(logvar.detach().cpu().numpy())
            log_det_jacobian_data.append(ldj.detach().cpu().numpy())
            z0_data.append(z0.detach().cpu().numpy())
            zk_data.append(zk.detach().cpu().numpy())

    end = time.time()

    # Saving activations values
    if config.activation_extraction:
        activations = diagnostics.dict_to_square_matrix(model.get_activations())
        model.detach_hooks(hooks)
        np.save(os.path.join(project_path, "activations.npy"), activations)

    if verbose:
        print(f"Inference took {(end - start) / 60:.3} minutes")

    # Convert all the data to numpy arrays
    (
        reconstructed_data,
        mu_data,
        logvar_data,
        z0_data,
        zk_data,
        log_det_jacobian_data,
    ) = [
        np.array(x)
        for x in [
            reconstructed_data,
            mu_data,
            logvar_data,
            z0_data,
            zk_data,
            log_det_jacobian_data,
        ]
    ]

    # Reshape the data
    (reconstructed_data, mu_data, logvar_data, z0_data, zk_data) = [
        x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        for x in [reconstructed_data, mu_data, logvar_data, z0_data, zk_data]
    ]

    # Save all the data
    save_dir = os.path.join(output_path, "results")
    np.save(
        os.path.join(save_dir, "test_reconstructed_data.npy"),
        reconstructed_data,
    )
    np.save(
        os.path.join(save_dir, "test_mu_data.npy"),
        mu_data,
    )
    np.save(
        os.path.join(save_dir, "test_logvar_data.npy"),
        logvar_data,
    )
    np.save(
        os.path.join(save_dir, "test_z0_data.npy"),
        z0_data,
    )
    np.save(
        os.path.join(save_dir, "test_zk_data.npy"),
        zk_data,
    )
    np.save(
        os.path.join(save_dir, "test_log_det_jacobian_data.npy"),
        log_det_jacobian_data,
    )

    helper.save_loss_components(
        loss_data=test_loss_data,
        component_names=loss_fn.component_names,
        suffix="test",
        save_dir=save_dir,
    )

    return True
