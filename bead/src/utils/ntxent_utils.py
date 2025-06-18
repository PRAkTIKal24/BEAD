import torch

def generate_augmented_views(x: torch.Tensor, sigma: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates two views of a given data batch for contrastive learning.
    One view is the original input, and the other has Gaussian noise added.

    Args:
        x (torch.Tensor): The input data batch.
        sigma (float): The standard deviation of the Gaussian noise to add.
                       Defaults to 0.1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                                           - view_i: The original input batch.
                                           - view_j: The input batch with added Gaussian noise.
    """
    view_i = x  # Original view

    # Generate Gaussian noise with the same shape as x and specified sigma
    noise = torch.randn_like(x) * sigma
    view_j = x + noise

    return view_i, view_j
