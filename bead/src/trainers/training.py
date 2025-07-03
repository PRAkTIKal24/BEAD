"""
Training functionality for anomaly detection models.
This module provides functionality for training neural network models for anomaly detection.
It includes functions for model fitting, validation, and the main training loop that handles data loading, model initialization, optimization, and early stopping.

Functions:
    fit: Performs one epoch of training on the training set.
    validate: Evaluates the model on the validation set.
    seed_worker: Sets seeds for workers to ensure reproducibility.
    train: Main function that handles the entire training process.
"""

import os
import random
import time
import warnings

import numpy as np
import torch
import torch.amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from ..utils import helper
from ..utils.annealing import AnnealingManager

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def fit(
    config,
    ddp_model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    scaler,
    is_ddp_active,
    local_rank,
    epoch_num,
    verbose: bool = False,
):
    """
    This function trains the model on the train set. It computes the losses and does the backwards propagation, and updates the optimizer as well.

    Args:
        config (dataClass): Base class selecting user inputs
        ddp_model (modelObject): The model you wish to train - explicit handling for DDP
        dataloader (torch.DataLoader): Defines the batched data which the model is trained on
        loss_fn (lossObject): Defines the loss function used to train the model
        optimizer (torch.optim): Chooses optimizer for gradient descent.
        device (torch.device): Chooses which device to use with torch
        scaler (torch.cuda.amp.GradScaler): Scaler for mixed precision training
        is_ddp_active (bool): Flag indicating if DDP is active
        local_rank (int): Local rank of the process in DDP

    Returns:
        list, model object: Training losses, Epoch_loss and trained model
    """
    # If model is DDP, actual model is model.module
    model_for_loss_params = (
        ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    )
    ddp_model.train()

    running_loss = 0.0
    num_batches_processed_this_rank = 0

    # DDP sanity check
    actual_num_batches_for_rank = len(dataloader)

    # Initialize progress bar for rank 0 or non-DDP, otherwise use dataloader directly
    if not is_ddp_active or local_rank == 0:
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch_num + 1} Training Batch",
            total=actual_num_batches_for_rank,
            disable=not verbose,
        )
    else:
        pbar = dataloader

    for _idx, batch in enumerate(pbar):
        inputs, gen_labels = batch
        inputs = inputs.to(device, non_blocking=True)
        gen_labels = gen_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16
            if config.use_amp and device.type == "cuda"
            else torch.float32,
            enabled=(config.use_amp and device.type == "cuda"),
        ):
<<<<<<< HEAD
            # Check if NT-Xent is needed based on the loss function name
            # This will match NTXentLoss, NTXentVAELoss, NTXentVAEFlowLoss, NTXentDVAELoss, etc.
            is_ntxent = "ntxent" in config.loss_function.lower()
            
            if is_ntxent:
=======
            # If NT-Xent is enabled, use dual forward pass with augmented views
            if hasattr(config, "use_ntxent") and config.use_ntxent:
>>>>>>> efdf8fd52de033cb65b953e9e261bd91779c16dd
                recon, mu, logvar, ldj, z0, zk, zk_j = helper.get_ntxent_outputs(ddp_model, inputs, config)
            else:
                # Standard single forward pass
                out = helper.call_forward(ddp_model, inputs)
                recon, mu, logvar, ldj, z0, zk = helper.unpack_model_outputs(out)
                zk_j = None  # No second view for standard training

            # Prepare common arguments for loss calculation
            loss_args = {
                "recon": recon,
                "target": inputs,
                "mu": mu,
                "logvar": logvar,
                "zk": zk,
<<<<<<< HEAD
=======
                "z0": z0,  # Include z0 for flow-based loss functions
>>>>>>> efdf8fd52de033cb65b953e9e261bd91779c16dd
                "parameters": model_for_loss_params.parameters(),
                "log_det_jacobian": ldj if hasattr(ldj, "item") else torch.tensor(0.0, device=device),
                "generator_labels": gen_labels,
            }
            
            # Only include zk_j for NT-Xent loss functions
<<<<<<< HEAD
            if is_ntxent:
=======
            if (hasattr(config, "use_ntxent") and config.use_ntxent) and \
               (hasattr(loss_fn, "calculate_ntxent") or "NTXent" in loss_fn.__class__.__name__):
>>>>>>> efdf8fd52de033cb65b953e9e261bd91779c16dd
                loss_args["zk_j"] = zk_j
                
            losses = loss_fn.calculate(**loss_args)
        loss, *_ = losses

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches_processed_this_rank += 1

    # DDP sanity check
    if num_batches_processed_this_rank == 0:
        epoch_loss_train = 0.0
        if verbose and (not is_ddp_active or local_rank == 0):
            print(
                f"[Rank {local_rank}, Epoch {epoch_num + 1}] WARNING: FIT DataLoader was empty or yielded no batches for this rank."
            )
    else:
        epoch_loss_train = running_loss / num_batches_processed_this_rank

    epoch_loss_tensor = torch.tensor(epoch_loss_train, device=device)

    # Synchronize and consolidate across all ranks
    if is_ddp_active:
        dist.barrier()
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
        dist.barrier()

    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"# Epoch {epoch_num + 1} Training Loss: {epoch_loss_tensor.item():.6f}")

    return losses, epoch_loss_tensor


def validate(
    config,
    ddp_model,
    dataloader,
    loss_fn,
    device,
    is_ddp_active,
    local_rank,
    epoch_num,
    verbose: bool = False,
):
    """
    Function used to validate the training. Not necessary for doing compression, but gives a good indication of wether the model selected is a good fit or not.

    Args:
        config (dataClass): Base class selecting user inputs
        model (modelObject): Defines the model one wants to validate. The model used here is passed directly from `fit()`.
        dataloader (torch.DataLoader): Defines the batched data which the model is validated on
        loss_fn (lossObject): Defines the loss function used to train the model
        device (torch.device): Chooses which device to use with torch
        is_ddp_active (bool): Flag indicating if DDP is active
        local_rank (int): Local rank of the process in DDP

    Returns:
        float: Validation loss
    """
    # Eexplicitly handle DDP wrapped model
    model_for_loss_params = (
        ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    )
    ddp_model.eval()

    # DDP sanity check
    running_loss = 0.0
    num_batches_processed_this_rank = 0
    actual_num_batches_for_rank = len(dataloader)

    if not is_ddp_active or local_rank == 0:
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch_num + 1} Validation Batch",
            total=actual_num_batches_for_rank,
            disable=not verbose,
        )
    else:
        pbar = dataloader

    with torch.no_grad():
        for _idx, batch in enumerate(pbar):
            inputs, gen_labels = batch
            inputs = inputs.to(device, non_blocking=True)
            gen_labels = gen_labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16
                if config.use_amp and device.type == "cuda"
                else torch.float32,
                enabled=(config.use_amp and device.type == "cuda"),
            ):
<<<<<<< HEAD
                # Check if NT-Xent is needed based on the loss function name
                # This will match NTXentLoss, NTXentVAELoss, NTXentVAEFlowLoss, NTXentDVAELoss, etc.
                is_ntxent = "ntxent" in config.loss_function.lower()
                
                if is_ntxent:
=======
                # If NT-Xent is enabled, use dual forward pass with augmented views
                if hasattr(config, "use_ntxent") and config.use_ntxent:
>>>>>>> efdf8fd52de033cb65b953e9e261bd91779c16dd
                    recon, mu, logvar, ldj, z0, zk, zk_j = helper.get_ntxent_outputs(ddp_model, inputs, config)
                else:
                    # Standard single forward pass
                    out = helper.call_forward(ddp_model, inputs)
                    recon, mu, logvar, ldj, z0, zk = helper.unpack_model_outputs(out)
                    zk_j = None  # No second view for standard validation

                # Prepare common arguments for loss calculation
                loss_args = {
                    "recon": recon,
                    "target": inputs,
                    "mu": mu,
                    "logvar": logvar,
                    "zk": zk,
<<<<<<< HEAD
=======
                    "z0": z0,  # Include z0 for flow-based loss functions
>>>>>>> efdf8fd52de033cb65b953e9e261bd91779c16dd
                    "parameters": model_for_loss_params.parameters(),
                    "log_det_jacobian": ldj if hasattr(ldj, "item") else torch.tensor(0.0, device=device),
                    "generator_labels": gen_labels,
                }
                
                # Only include zk_j for NT-Xent loss functions
<<<<<<< HEAD
                if is_ntxent:
=======
                if (hasattr(config, "use_ntxent") and config.use_ntxent) and \
                   (hasattr(loss_fn, "calculate_ntxent") or "NTXent" in loss_fn.__class__.__name__):
>>>>>>> efdf8fd52de033cb65b953e9e261bd91779c16dd
                    loss_args["zk_j"] = zk_j
                    
                losses = loss_fn.calculate(**loss_args)
            loss, *_ = losses
            running_loss += loss.item()
            num_batches_processed_this_rank += 1

    # DDP sanity check
    if num_batches_processed_this_rank == 0:
        epoch_loss_val = 0.0
        if verbose and (not is_ddp_active or local_rank == 0):
            print(
                f"[Rank {local_rank}, Epoch {epoch_num + 1}] WARNING: VALIDATE DataLoader was empty or yielded no batches for this rank."
            )
    else:
        epoch_loss_val = running_loss / num_batches_processed_this_rank

    epoch_loss_tensor = torch.tensor(epoch_loss_val, device=device)

    # Synchronize and consolidate across all ranks
    if is_ddp_active:
        dist.barrier()
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
        dist.barrier()

    if verbose and (not is_ddp_active or local_rank == 0):
        print(
            f"# Epoch {epoch_num + 1} Validation Loss: {epoch_loss_tensor.item():.6f}"
        )

    return losses, epoch_loss_tensor


def seed_worker(worker_id):
    """PyTorch implementation to fix the seeds

    Args:
        worker_id ():
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    data,
    labels,
    output_path,
    config,
    verbose: bool = False,
):
    """
    Processes the entire training loop by calling the `fit()` and `validate()`. Appart from this, this is the main function where the data is converted
    to the correct type for it to be trained, via `torch.Tensor()`. Furthermore, the batching is also done here, based on `config.batch_size`,
    and it is the `torch.utils.data.DataLoader` doing the splitting.
    Torch AMP and DDP are also implemented here, if the user has selected them in the config file. Applying either `EarlyStopping` or `LR Scheduler` is also done here, all based on their respective `config` arguments.
    For reproducibility, the seeds can also be fixed in this function using the deterministic_algorithm `config` flag.

    Args:
        model (modelObject): The model you wish to train
        data (Tuple): Tuple containing the training and validation data
        labels (Tuple): Tuple containing the training and validation labels
        project_path (string): Path to the project directory
        config (dataClass): Base class selecting user inputs
        verbose (bool): If True, prints additional information during training

    Returns:
        modelObject: fully trained model ready to perform inference
    """
    # Get DDP parameters from config
    is_ddp_active = config.is_ddp_active
    local_rank = config.local_rank
    world_size = config.world_size
    device = helper.get_device(config)
    
    # Store device in config for access by loss functions (needed for NT-Xent)
    config.device = device

    if "ConvVAE" in config.model_name or "ConvAE" in config.model_name:
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                "Expected data to be a list or tuple for ConvVAE/ConvAE preprocessing."
            )
        data = [x.unsqueeze(1).float() if x is not None else None for x in data]

    (
        events_train,
        jets_train,
        constituents_train,
        events_val,
        jets_val,
        constituents_val,
    ) = data
    (
        events_train_label,
        jets_train_label,
        constituents_train_label,
        events_val_label,
        jets_val_label,
        constituents_val_label,
    ) = labels

    datasets = helper.create_datasets(*data, *labels)

    if verbose and (not is_ddp_active or local_rank == 0):
        print(
            f"Events - Training set shape: {events_train.shape if events_train is not None else 'N/A'}"
        )
        print(
            f"Events - Validation set shape: {events_val.shape if events_val is not None else 'N/A'}"
        )
        print(
            f"Jets - Training set shape: {jets_train.shape if jets_train is not None else 'N/A'}"
        )
        print(
            f"Jets - Validation set shape: {jets_val.shape if jets_val is not None else 'N/A'}"
        )
        print(
            f"Constituents - Training set shape: {constituents_train.shape if constituents_train is not None else 'N/A'}"
        )
        print(
            f"Constituents - Validation set shape: {constituents_val.shape if constituents_val is not None else 'N/A'}"
        )

    # Calculate the input shapes to initialize the model
    input_shape = helper.calculate_in_shape(data, config)
    model = helper.model_init(input_shape, config)
    model = model.to(device)

    if is_ddp_active:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Keep true for ldj handling
        )
        if verbose and local_rank == 0:
            print(f"DDP initialized. Model wrapped. Running on {world_size} GPUs.")

    train_dataset_selected = datasets[f"{config.input_level}s_train"]
    validation_dataset_selected = (
        datasets[f"{config.input_level}s_val"] if config.train_size < 1.0 else None
    )

    # Intialize samplers in case DDP is active
    train_sampler, validation_sampler = None, None
    shuffle_train = not is_ddp_active

    if is_ddp_active:
        train_sampler = DistributedSampler(
            train_dataset_selected,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )
        if (
            validation_dataset_selected is not None
            and len(validation_dataset_selected) > 0
        ):
            validation_sampler = DistributedSampler(
                validation_dataset_selected,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False,
                drop_last=True,
            )
        else:
            validation_sampler = None

    generator_seed = torch.Generator()
    if config.deterministic_algorithm:
        if verbose and (not is_ddp_active or local_rank == 0):
            print("Deterministic algorithm is set to True")
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        generator_seed.manual_seed(0)

    # Set common DataLoader arguments
    common_loader_args = {
        "batch_size": config.batch_size,
        "drop_last": True,
        "num_workers": config.parallel_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker if config.deterministic_algorithm else None,
        "generator": generator_seed if config.deterministic_algorithm else None,
    }

    # Create DataLoaders for training and validation datasets
    train_dataloader = DataLoader(
        train_dataset_selected,
        sampler=train_sampler,
        shuffle=shuffle_train,
        **common_loader_args,
    )

    validation_dataloader = None
    if validation_dataset_selected is not None and len(validation_dataset_selected) > 0:
        validation_dataloader = DataLoader(
            validation_dataset_selected,
            sampler=validation_sampler,
            shuffle=False,
            **common_loader_args,
        )

    # Initialize loss function, optimizer
    loss_object = helper.get_loss(config.loss_function)
    loss_fn = loss_object(config=config)

    optimizer = helper.get_optimizer(config.optimizer, model.parameters(), lr=config.lr)
    amp_scaler = torch.amp.GradScaler(
        enabled=(config.use_amp and device.type == "cuda")
    )

    # Initialize early stopping and learning rate scheduler if specified
    early_stopper = (
        helper.EarlyStopping(
            patience=config.early_stopping_patience, min_delta=config.min_delta
        )
        if config.early_stopping
        else None
    )
    lr_scheduler = (
        helper.LRScheduler(optimizer=optimizer, patience=config.lr_scheduler_patience)
        if config.lr_scheduler
        else None
    )

    # Initialize hyperparameter annealing manager if configured
    annealing_manager = (
        AnnealingManager(config) if hasattr(config, "annealing_params") else None
    )

    # Containers for loss data
    train_loss_components_per_epoch = []
    validation_loss_components_per_epoch = []
    train_avg_epoch_losses = []
    validation_avg_epoch_losses = []
    start_time = time.time()

    if verbose and (not is_ddp_active or local_rank == 0):
        print(f"Beginning training for {config.epochs} epochs")

    for epoch in range(config.epochs):
        if is_ddp_active and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_ddp_active and validation_sampler is not None:
            validation_sampler.set_epoch(epoch)

        batch_train_losses_components, current_train_epoch_loss_avg = fit(
            config,
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
            amp_scaler,
            is_ddp_active,
            local_rank,
            epoch,
            verbose,
        )

        current_validation_epoch_loss_for_schedulers = current_train_epoch_loss_avg
        batch_validation_losses_components_for_log = batch_train_losses_components

        if (
            config.train_size < 1.0
            and validation_dataloader is not None
            and len(validation_dataloader) > 0
        ):
            (
                batch_validation_losses_components_for_log,
                current_validation_epoch_loss_for_schedulers,
            ) = validate(
                config,
                model,
                validation_dataloader,
                loss_fn,
                device,
                is_ddp_active,
                local_rank,
                epoch,
                verbose,
            )

        if lr_scheduler:
            lr_scheduler(current_validation_epoch_loss_for_schedulers.item())

        # Using only rank 0 to log and broadcast decisions when using DDP
        if not is_ddp_active or local_rank == 0:
            train_avg_epoch_losses.append(current_train_epoch_loss_avg.item())
            train_loss_components_per_epoch.append(batch_train_losses_components)

            validation_avg_epoch_losses.append(
                current_validation_epoch_loss_for_schedulers.item()
            )
            validation_loss_components_per_epoch.append(
                batch_validation_losses_components_for_log
            )

            if config.intermittent_model_saving and (
                epoch > 0 and (epoch + 1) % config.intermittent_saving_patience == 0
            ):
                path = os.path.join(
                    output_path, "models", f"model_epoch_{epoch + 1}.pt"
                )
                helper.save_model(model, path, config)

            if early_stopper:
                early_stopper(current_validation_epoch_loss_for_schedulers.item())
                if early_stopper.early_stop and verbose:
                    print(
                        f"Rank {local_rank}: Early stopping condition met at epoch {epoch + 1}. Will signal other ranks."
                    )

        # Apply hyperparameter annealing if configured
        # Note: This is placed after early_stopper updates so the most recent counter is used
        annealing_metrics = {}
        if lr_scheduler and hasattr(lr_scheduler, "triggered"):
            annealing_metrics["lr_scheduler_triggered"] = lr_scheduler.triggered
        if early_stopper and hasattr(early_stopper, "counter"):
            annealing_metrics["early_stopper_counter"] = early_stopper.counter
            # Check if early stopper counter has reached half of patience
            # and add as a trigger signal
            if early_stopper.counter >= early_stopper.patience / 2:
                annealing_metrics["early_stopper_half_patience"] = True
            else:
                annealing_metrics["early_stopper_half_patience"] = False

            # Check if early stopper counter has reached one-third of patience
            # and add as a trigger signal
            if early_stopper.counter >= early_stopper.patience / 3:
                annealing_metrics["early_stopper_third_patience"] = True
            else:
                annealing_metrics["early_stopper_third_patience"] = False

        if annealing_manager:
            annealed_params = annealing_manager.step(
                epoch=epoch, metrics=annealing_metrics
            )
            if annealed_params and (not is_ddp_active or local_rank == 0) and verbose:
                print(f"Annealed parameters for epoch {epoch + 1}: {annealed_params}")

        # Synchronize early stopping signal and stop across all ranks based on decision from rank 0
        if is_ddp_active:
            if local_rank == 0:
                should_stop_epoch_flag = (
                    1.0 if early_stopper and early_stopper.early_stop else 0.0
                )
                stop_signal_tensor = torch.tensor(
                    [should_stop_epoch_flag], dtype=torch.float32, device=device
                )
            else:
                stop_signal_tensor = torch.empty(1, dtype=torch.float32, device=device)

            dist.broadcast(stop_signal_tensor, src=0)

            if stop_signal_tensor.item() == 1.0:
                if verbose:
                    print(
                        f"Rank {local_rank}: Early stopping signal received at epoch {epoch + 1}. Breaking training loop."
                    )
                break
        else:
            if early_stopper and early_stopper.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}.")
                break

        if is_ddp_active and verbose:
            print(
                f"[Rank {local_rank}, Epoch {epoch + 1}] TRAIN MAIN: Reached end of epoch logic."
            )

    end_time = time.time()
    if verbose and (not is_ddp_active or local_rank == 0):
        print(
            f"Training loop finished. Total training time: {(end_time - start_time) / 60:.3} minutes"
        )

    # Save only on rank 0 if DDP is active
    if not is_ddp_active or local_rank == 0:
        helper.save_model(
            model, os.path.join(output_path, "models", "model.pt"), config
        )
        if verbose:
            print(
                f"Final model saved to: {os.path.join(output_path, 'models', 'model.pt')}"
            )

        # Prepare final data collection pass
        actual_model_for_evaluation = model.module if isinstance(model, DDP) else model
        actual_model_for_evaluation.eval()

        mu_data_list, logvar_data_list, z0_data_list, zk_data_list, ldj_data_list = (
            [],
            [],
            [],
            [],
            [],
        )

        if train_dataset_selected is not None and len(train_dataset_selected) > 0:
            final_pass_dataloader_args = common_loader_args.copy()
            final_pass_dataloader_args.pop("sampler", None)
            final_pass_dataloader_args["shuffle"] = False

            final_pass_dataloader = DataLoader(
                train_dataset_selected, **final_pass_dataloader_args
            )

            with torch.no_grad():
                for _idx, batch in enumerate(
                    tqdm(
                        final_pass_dataloader,
                        desc="Final data collection pass (Rank 0)",
                        disable=not verbose,
                    )
                ):
                    inputs, _ = batch
                    inputs = inputs.to(device, non_blocking=True)
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=(config.use_amp and device.type == "cuda"),
                    ):
                        out = helper.call_forward(actual_model_for_evaluation, inputs)
                        _, mu, logvar, ldj, z0, zk = helper.unpack_model_outputs(out)
                    mu_data_list.append(mu.detach().cpu().numpy())
                    logvar_data_list.append(logvar.detach().cpu().numpy())
                    if hasattr(ldj, "detach"):
                        ldj_data_list.append(ldj.detach().cpu().numpy())
                    elif isinstance(ldj, (float, int, np.number)):
                        ldj_data_list.append(np.array(ldj))
                    else:
                        ldj_data_list.append(np.array(0.0))
                    z0_data_list.append(z0.detach().cpu().numpy())
                    zk_data_list.append(zk.detach().cpu().numpy())
        else:
            if verbose:
                print(
                    "Skipping final data collection pass as training dataset is empty or invalid."
                )

        results_save_dir = os.path.join(output_path, "results")
        os.makedirs(results_save_dir, exist_ok=True)
        np.save(
            os.path.join(results_save_dir, "train_epoch_loss_data.npy"),
            np.array(train_avg_epoch_losses),
        )
        if validation_avg_epoch_losses:
            np.save(
                os.path.join(results_save_dir, "val_epoch_loss_data.npy"),
                np.array(validation_avg_epoch_losses),
            )

        if mu_data_list:
            np.save(
                os.path.join(results_save_dir, "train_mu_data.npy"),
                np.concatenate(mu_data_list, axis=0),
            )
        if logvar_data_list:
            np.save(
                os.path.join(results_save_dir, "train_logvar_data.npy"),
                np.concatenate(logvar_data_list, axis=0),
            )
        if z0_data_list:
            np.save(
                os.path.join(results_save_dir, "train_z0_data.npy"),
                np.concatenate(z0_data_list, axis=0),
            )
        if zk_data_list:
            np.save(
                os.path.join(results_save_dir, "train_zk_data.npy"),
                np.concatenate(zk_data_list, axis=0),
            )

        # Handling ldj with extra love
        if ldj_data_list:
            try:
                if (
                    ldj_data_list
                    and isinstance(ldj_data_list[0], np.ndarray)
                    and len(ldj_data_list[0].shape) > 0
                ):
                    final_ldj_data = np.concatenate(ldj_data_list, axis=0)
                elif ldj_data_list:
                    final_ldj_data = np.array(ldj_data_list)
                else:
                    final_ldj_data = np.array([])
                np.save(
                    os.path.join(results_save_dir, "train_log_det_jacobian_data.npy"),
                    final_ldj_data,
                )
            except Exception as e:
                if verbose:
                    print(f"Could not save train_log_det_jacobian_data: {e}")

        helper.save_loss_components(
            loss_data=train_loss_components_per_epoch,
            component_names=loss_fn.component_names,
            suffix="train",
            save_dir=results_save_dir,
        )
        if validation_loss_components_per_epoch:
            helper.save_loss_components(
                loss_data=validation_loss_components_per_epoch,
                component_names=loss_fn.component_names,
                suffix="val",
                save_dir=results_save_dir,
            )
        if verbose:
            print("Loss data and latent variables saved to path: ", results_save_dir)

    return model
