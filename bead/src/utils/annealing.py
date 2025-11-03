"""
Module for hyperparameter annealing during training.

This module provides functionality to anneal (gradually change) hyperparameters
during the training process based on different strategies.

Supported annealing strategies:
1. CONSTANT_PACE: Anneals parameters at a constant rate over a specified number of steps
2. TRIGGER_BASED: Changes parameters based on a trigger signal (e.g., from early_stopper or lr_scheduler)
3. SCHEDULED: Changes parameters according to a predefined schedule of {epoch: value} pairs

Example configuration in a config class:
```python
def set_config(c):
    # ... other config settings ...

    # Annealing configuration
    c.annealing_params = {
        # Example 1: Anneal contrastive_temperature with constant pace
        "contrastive_temperature": {
            "strategy": "CONSTANT_PACE",
            "start_value": 0.1,
            "end_value": 0.01,
            "total_steps": 10  # Gradually reduce over first 10 epochs
        },

        # Example 2: Anneal regularization parameter based on early stopper trigger
        "reg_param": {
            "strategy": "TRIGGER_BASED",
            "values": [0.001, 0.005, 0.01, 0.05],  # Values to cycle through
            "trigger_source": "early_stopper_counter",  # Will increase when counter increases
            "current_index": 0
        },

        # Example 3: Anneal batch size based on a schedule
        "batch_size": {
            "strategy": "SCHEDULED",
            "schedule": {
                5: 128,   # Change to 128 at epoch 5
                10: 192,  # Change to 192 at epoch 10
                15: 256   # Change to 256 at epoch 15
            }
        }
    }
```

In a distributed data parallel (DDP) environment, the annealing decisions
are made by rank 0 and then broadcasted to all other processes.
"""

import torch
import torch.distributed as dist
import numpy as np
from enum import Enum, auto


class AnnealingStrategy(Enum):
    """Enumeration of supported annealing strategies."""

    CONSTANT_PACE = auto()  # Anneal at a constant pace
    TRIGGER_BASED = (
        auto()
    )  # Anneal based on a trigger (e.g., early stopping, LR scheduler)
    SCHEDULED = auto()  # Anneal based on predefined schedule


class AnnealingManager:
    """
    Class to manage hyperparameter annealing during training.

    This class handles the annealing of hyperparameters based on different strategies
    and ensures proper synchronization across processes when using DDP.

    Attributes:
        config: Configuration object containing annealing settings.
        param_maps: Dictionary mapping parameter names to their objects and attributes.
        is_ddp_active: Whether DDP is active.
        world_size: Number of DDP processes.
        rank: Current process rank.
    """

    def __init__(self, config):
        """
        Initialize the annealing manager.

        Args:
            config: Configuration object with annealing settings.
        """
        self.config = config
        self.param_maps = {}

        # DDP-related attributes
        self.is_ddp_active = (
            config.is_ddp_active if hasattr(config, "is_ddp_active") else False
        )
        self.world_size = config.world_size if hasattr(config, "world_size") else 1
        self.rank = config.rank if hasattr(config, "rank") else 0

        # Initialize annealing parameters from config
        self._initialize_annealing_params()

    def _initialize_annealing_params(self):
        """Initialize annealing parameters from the config."""
        if not hasattr(self.config, "annealing_params"):
            return

        # For each parameter to be annealed
        for param_name, param_config in self.config.annealing_params.items():
            # Get target object and attribute
            obj = param_config.get("object", self.config)
            attr_name = param_config.get("attr_name", param_name)

            # Get annealing strategy
            strategy_name = param_config.get("strategy", "CONSTANT_PACE")
            strategy = AnnealingStrategy[strategy_name]

            # Get current value (handling nested attributes)
            try:
                current_value = self._get_attr_value(obj, attr_name)
            except (AttributeError, IndexError):
                # Skip if the attribute doesn't exist yet (might be created later)
                # For example, optimizer parameters that don't exist at initialization
                current_value = None

            # Get annealing settings based on strategy
            settings = {}
            if strategy == AnnealingStrategy.CONSTANT_PACE:
                settings["start_value"] = param_config.get("start_value", current_value)
                settings["end_value"] = param_config.get("end_value")
                settings["total_steps"] = param_config.get(
                    "total_steps", self.config.epochs
                )
                settings["current_step"] = 0
                if (
                    settings["start_value"] is not None
                    and settings["end_value"] is not None
                ):
                    settings["step_size"] = (
                        settings["end_value"] - settings["start_value"]
                    ) / settings["total_steps"]

            elif strategy == AnnealingStrategy.TRIGGER_BASED:
                settings["values"] = param_config.get("values", [])
                settings["current_index"] = param_config.get("current_index", 0)
                settings["trigger_source"] = param_config.get("trigger_source")
                settings["triggered"] = False

            elif strategy == AnnealingStrategy.SCHEDULED:
                settings["schedule"] = param_config.get(
                    "schedule", {}
                )  # Dict of {epoch: value}

            # Store mapping information
            self.param_maps[param_name] = {
                "object": obj,
                "attr_name": attr_name,
                "strategy": strategy,
                "settings": settings,
            }

    def step(self, epoch=None, metrics=None):
        """
        Perform one annealing step.

        Args:
            epoch: Current training epoch.
            metrics: Dictionary of metrics that might trigger annealing.

        Returns:
            dict: Dictionary of annealed parameters and their new values.
        """
        # Only rank 0 makes annealing decisions in DDP mode
        if self.is_ddp_active and self.rank != 0:
            # Non-root ranks will receive annealed values via broadcast
            return self._receive_annealed_params()

        annealed_params = {}

        for param_name, param_map in self.param_maps.items():
            obj = param_map["object"]
            attr_name = param_map["attr_name"]
            strategy = param_map["strategy"]
            settings = param_map["settings"]

            try:
                # Get current value (handling nested attributes)
                current_value = self._get_attr_value(obj, attr_name)
                new_value = current_value

                # Apply annealing based on strategy
                if strategy == AnnealingStrategy.CONSTANT_PACE:
                    if (
                        "step_size" in settings
                        and settings["current_step"] < settings["total_steps"]
                    ):
                        new_value = (
                            settings["start_value"]
                            + settings["step_size"] * settings["current_step"]
                        )
                        settings["current_step"] += 1
                    elif "end_value" in settings:
                        new_value = settings["end_value"]

                elif strategy == AnnealingStrategy.TRIGGER_BASED:
                    if metrics and settings["trigger_source"] in metrics:
                        # Check if the trigger condition is met
                        triggered = metrics[settings["trigger_source"]]
                        if triggered and not settings["triggered"]:
                            settings["triggered"] = True
                            if (
                                settings["current_index"] < len(settings["values"]) - 1
                                and len(settings["values"]) > 0
                            ):
                                settings["current_index"] += 1
                                new_value = settings["values"][
                                    settings["current_index"]
                                ]
                        elif not triggered:
                            settings["triggered"] = False

                elif strategy == AnnealingStrategy.SCHEDULED:
                    if epoch is not None and epoch in settings["schedule"]:
                        new_value = settings["schedule"][epoch]

                # Update parameter if changed
                if new_value != current_value:
                    self._set_attr_value(obj, attr_name, new_value)
                    annealed_params[param_name] = new_value
            except (AttributeError, IndexError, KeyError) as e:
                # Skip parameters that don't exist yet or can't be accessed
                # This can happen with dynamically created attributes or nested paths
                # that aren't fully initialized yet
                if self.rank == 0:
                    print(
                        f"Warning: Could not anneal parameter '{param_name}': {str(e)}"
                    )

        # Broadcast annealed parameters in DDP mode
        if self.is_ddp_active:
            self._broadcast_annealed_params(annealed_params)

        return annealed_params

    def _broadcast_annealed_params(self, annealed_params):
        """
        Broadcast annealed parameters from rank 0 to all processes.

        Args:
            annealed_params: Dictionary of annealed parameters and their new values.
        """
        if not self.is_ddp_active:
            return

        # Convert parameter dictionary to a list for broadcasting
        param_list = []
        param_types = []  # Track the parameter types for proper conversion
        for param_name, param_map in self.param_maps.items():
            if param_name in annealed_params:
                value = annealed_params[param_name]
            else:
                value = getattr(param_map["object"], param_map["attr_name"])

            # Store the original type
            param_types.append(type(value))

            # Convert value to float for broadcasting
            # Non-numeric types are handled specially
            if isinstance(value, bool):
                param_list.append(float(value))
            elif isinstance(value, (int, float)):
                param_list.append(float(value))
            else:
                # For complex types, we'll skip broadcasting
                # and handle them separately if needed in the future
                param_list.append(float("nan"))

        # Create tensor for broadcasting
        if len(param_list) > 0:
            tensor = (
                torch.tensor(param_list, dtype=torch.float32).cuda()
                if torch.cuda.is_available()
                else torch.tensor(param_list, dtype=torch.float32)
            )
            dist.broadcast(tensor, src=0)

            # Store type information for receiver to properly reconstruct values
            if self.rank == 0:
                # Send type information to other processes
                # For now, we support int, float, bool
                # A more comprehensive solution would be needed for complex types
                type_info = (
                    torch.zeros(len(param_types), dtype=torch.int32).cuda()
                    if torch.cuda.is_available()
                    else torch.zeros(len(param_types), dtype=torch.int32)
                )
                for i, t in enumerate(param_types):
                    if t is bool:
                        type_info[i] = 1
                    elif t is int:
                        type_info[i] = 2
                    # float is 0
                dist.broadcast(type_info, src=0)

    def _receive_annealed_params(self):
        """
        Receive annealed parameters from rank 0 (for non-root processes).

        Returns:
            dict: Dictionary of annealed parameters and their new values.
        """
        if not self.is_ddp_active or self.rank == 0:
            return {}

        # Create tensor for receiving
        param_list = [0.0] * len(self.param_maps)
        if len(param_list) > 0:
            # Receive values
            tensor = (
                torch.tensor(param_list, dtype=torch.float32).cuda()
                if torch.cuda.is_available()
                else torch.tensor(param_list, dtype=torch.float32)
            )
            dist.broadcast(tensor, src=0)

            # Receive type information
            type_info = (
                torch.zeros(len(param_list), dtype=torch.int32).cuda()
                if torch.cuda.is_available()
                else torch.zeros(len(param_list), dtype=torch.int32)
            )
            dist.broadcast(type_info, src=0)

            # Update local parameters with received values
            annealed_params = {}
            for idx, (param_name, param_map) in enumerate(self.param_maps.items()):
                value = tensor[idx].item()
                obj = param_map["object"]
                attr_name = param_map["attr_name"]
                current_value = getattr(obj, attr_name)

                # Convert value based on type information
                if type_info[idx] == 1:  # bool
                    new_value = bool(value)
                elif type_info[idx] == 2:  # int
                    new_value = int(value)
                else:  # float or other
                    new_value = value

                # Skip if it's NaN (indicates complex type that couldn't be broadcast)
                if np.isnan(value):
                    continue

                if new_value != current_value:
                    setattr(obj, attr_name, new_value)
                    annealed_params[param_name] = new_value

            return annealed_params

        return {}

    def _get_attr_value(self, obj, attr_path):
        """
        Get a value from an object using a dot-notation attribute path.

        Args:
            obj: The object to get the attribute from.
            attr_path: The attribute path in dot notation (e.g., "param_groups.0.weight_decay").

        Returns:
            The value of the attribute.
        """
        # If no dots, it's a direct attribute access
        if "." not in attr_path:
            return getattr(obj, attr_path)

        # Handle dot notation for nested attributes
        parts = attr_path.split(".")
        value = obj

        for part in parts:
            # Handle list/dict access with numeric index
            if part.isdigit():
                try:
                    value = value[int(part)]
                except (IndexError, TypeError) as e:
                    # Handle the case where the target is not subscriptable
                    # or the index is out of range
                    raise AttributeError(
                        f"Cannot access index {part} in {value}"
                    ) from e
            else:
                try:
                    # Try attribute access first
                    value = getattr(value, part)
                except AttributeError:
                    # Fallback to dictionary access
                    try:
                        value = value[part]
                    except (KeyError, TypeError) as e:
                        raise AttributeError(
                            f"No attribute or key '{part}' in {type(value).__name__}"
                        ) from e

        return value

    def _set_attr_value(self, obj, attr_path, value):
        """
        Set a value on an object using a dot-notation attribute path.

        Args:
            obj: The object to set the attribute on.
            attr_path: The attribute path in dot notation (e.g., "param_groups.0.weight_decay").
            value: The value to set.
        """
        # If no dots, it's a direct attribute access
        if "." not in attr_path:
            setattr(obj, attr_path, value)
            return

        # Handle dot notation for nested attributes
        parts = attr_path.split(".")
        target = obj

        # Navigate to the appropriate object
        for _i, part in enumerate(parts[:-1]):
            # Handle list/dict access with numeric index
            if part.isdigit():
                try:
                    target = target[int(part)]
                except (IndexError, TypeError) as e:
                    raise AttributeError(
                        f"Cannot access index {part} in {target}"
                    ) from e
            else:
                try:
                    # Try attribute access first
                    if hasattr(target, part):
                        target = getattr(target, part)
                    else:
                        # Fallback to dictionary access
                        target = target[part]
                except (AttributeError, KeyError, TypeError) as e:
                    raise AttributeError(
                        f"No attribute or key '{part}' in {type(target).__name__}"
                    ) from e

        # Set the attribute or item
        last_part = parts[-1]
        if last_part.isdigit():
            try:
                target[int(last_part)] = value
            except (IndexError, TypeError) as e:
                raise AttributeError(
                    f"Cannot set value at index {last_part} in {target}"
                ) from e
        else:
            try:
                # Try attribute access first
                if hasattr(target, last_part):
                    setattr(target, last_part, value)
                else:
                    # Fallback to dictionary access
                    target[last_part] = value
            except (AttributeError, TypeError) as e:
                raise AttributeError(
                    f"Cannot set attribute or key '{last_part}' in {type(target).__name__}"
                ) from e
