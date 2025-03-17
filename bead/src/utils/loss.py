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

import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch.nn import functional
from tqdm import tqdm
from torch.nn import functional as F
from torch import distributions as dist


class BaseLoss:
    """
    Base class for all loss functions.
    Each subclass must implement the calculate() method.
    """

    def __init__(self, config):
        self.config = config

    def calculate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the calculate() method.")


# ---------------------------
# Standard AE reco loss
# ---------------------------
class ReconstructionLoss(BaseLoss):
    """
    Reconstruction loss for AE/VAE models.
    Supports both MSE and L1 losses based on configuration.

    Config parameters:
      - loss_type: 'mse' (default) or 'l1'
      - reduction: reduction method (default 'mean' or 'sum')
    """

    def __init__(self, config):
        super(ReconstructionLoss, self).__init__(config)
        self.reg_param = config.reg_param
        self.component_names = ["reco"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        self.loss_type = "mse"
        self.reduction = "mean"

        if self.loss_type == "mse":
            loss = F.mse_loss(recon, target, reduction=self.reduction)
        elif self.loss_type == "l1":
            loss = F.l1_loss(recon, target, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {self.loss_type}")
        return (loss,)


# ---------------------------
# KL Divergence Loss
# ---------------------------
class KLDivergenceLoss(BaseLoss):
    """
    KL Divergence loss for VAE latent space regularization.

    Uses the formula:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(self, config):
        super(KLDivergenceLoss, self).__init__(config)
        self.component_names = ["kl"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        batch_size = mu.size(0)
        return (kl_loss / batch_size,)


# ---------------------------
# Earth Mover's Distance / Wasserstein Loss
# ---------------------------
class WassersteinLoss(BaseLoss):
    """
    Computes an approximation of the Earth Mover's Distance (Wasserstein Loss)
    between two 1D probability distributions.

    Assumes inputs are tensors of shape (batch_size, n) representing histograms or distributions.

    Config parameters:
      - dim: dimension along which to compute the cumulative sum (default: 1)
    """

    def __init__(self, config):
        super(WassersteinLoss, self).__init__(config)
        self.dim = 1
        self.component_names = ["emd"]

    def calculate(self, p, q):
        # Normalize if not already probability distributions
        p = p / (p.sum(dim=self.dim, keepdim=True) + 1e-8)
        q = q / (q.sum(dim=self.dim, keepdim=True) + 1e-8)
        p_cdf = torch.cumsum(p, dim=self.dim)
        q_cdf = torch.cumsum(q, dim=self.dim)
        loss = torch.mean(torch.abs(p_cdf - q_cdf))
        return (loss,)


# ---------------------------
# Regularization Losses
# ---------------------------
class L1Regularization(BaseLoss):
    """
    Computes L1 regularization over model parameters.

    Config parameters:
      - weight: scaling factor for the L1 regularization (default: 1e-4)
    """

    def __init__(self, config):
        super(L1Regularization, self).__init__(config)
        self.weight = self.config.reg_param
        self.component_names = ["l1"]

    def calculate(self, parameters):
        l1_loss = 0.0
        for param in parameters:
            l1_loss += torch.sum(torch.abs(param))
        return (self.weight * l1_loss,)


class L2Regularization(BaseLoss):
    """
    Computes L2 regularization over model parameters.

    Config parameters:
      - weight: scaling factor for the L2 regularization (default: 1e-4)
    """

    def __init__(self, config):
        super(L2Regularization, self).__init__(config)
        self.weight = self.config.reg_param
        self.component_names = ["l2"]

    def calculate(self, parameters):
        l2_loss = 0.0
        for param in parameters:
            l2_loss += torch.sum(param**2)
        return self.weight * l2_loss


# ---------------------------
# Energy Based Loss
# ---------------------------
class BinaryCrossEntropyLoss(BaseLoss):
    """
    Binary Cross Entropy Loss for binary classification tasks.

    Config parameters:
      - use_logits: Boolean indicating if the predictions are raw logits (default: True).
      - reduction: Reduction method for the loss ('mean', 'sum', etc., default: 'mean').

    Note: Not supported for full_chain mode yet
    """

    def __init__(self, config):
        super(BinaryCrossEntropyLoss, self).__init__(config)
        self.use_logits = True
        self.reduction = "mean"
        self.component_names = ["bce"]

    def calculate(
        self, predictions, targets, mu, logvar, parameters, log_det_jacobian=0
    ):
        """
        Calculate the binary cross entropy loss.

        Args:
            predictions (Tensor): Predicted outputs (logits or probabilities).
            targets (Tensor): Ground truth binary labels.

        Returns:
            Tensor: The computed binary cross entropy loss.
        """
        # Ensure targets are float tensors.
        targets = targets.float()
        if self.use_logits:
            loss = F.binary_cross_entropy_with_logits(
                predictions, targets, reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy(
                predictions, targets, reduction=self.reduction
            )
        return (loss,)


# ---------------------------
# ELBO Loss
# ---------------------------
class VAELoss(BaseLoss):
    """
    Total loss for VAE training.
    Combines reconstruction loss and KL divergence loss.

    Config parameters:
      - reconstruction: dict for ReconstructionLoss config.
      - kl: dict for KLDivergenceLoss config.
      - kl_weight: scaling factor for KL loss (default: 1.0)
    """

    def __init__(self, config):
        super(VAELoss, self).__init__(config)
        self.recon_loss_fn = ReconstructionLoss(config)
        self.loss_type = "mse"
        self.reduction = "mean"
        self.kl_loss_fn = KLDivergenceLoss(config)
        self.kl_weight = torch.tensor(self.config.reg_param, requires_grad=True)
        self.component_names = ["loss", "reco", "kl"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        recon_loss = self.recon_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        kl_loss = self.kl_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        loss = recon_loss[0] + self.kl_weight * kl_loss[0]
        return loss, recon_loss[0], kl_loss[0]


# ---------------------------
# Advanced VAE+Flow Loss
# ---------------------------
class VAEFlowLoss(BaseLoss):
    """
    Loss for VAE models augmented with a normalizing flow.
    Includes the log_det_jacobian term from the flow transformation.

    Config parameters:
      - reconstruction: dict for ReconstructionLoss config.
      - kl: dict for KLDivergenceLoss config.
      - kl_weight: weight for the KL divergence term.
      - flow_weight: weight for the log_det_jacobian term.
    """

    def __init__(self, config):
        super(VAEFlowLoss, self).__init__(config)
        self.recon_loss_fn = ReconstructionLoss(config)
        self.loss_type = "mse"
        self.reduction = "mean"
        self.kl_loss_fn = KLDivergenceLoss(config)
        self.kl_weight = torch.tensor(self.config.reg_param, requires_grad=True)
        self.flow_weight = torch.tensor(self.config.reg_param, requires_grad=True)
        self.component_names = ["loss", "reco", "kl"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        recon_loss = self.recon_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        kl_loss = self.kl_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        # Subtract the log-det term (maximizing likelihood).
        total_loss = (
            recon_loss[0]
            + self.kl_weight * kl_loss[0]
            - self.flow_weight * log_det_jacobian
        )
        return total_loss, recon_loss[0], kl_loss[0]


# ---------------------------
# Contrastive Loss
# ---------------------------
class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss to cluster latent vectors by event generator.

    Config parameters:
      - margin: minimum distance desired between dissimilar pairs (default: 1.0)
    """

    def __init__(self, config):
        super(ContrastiveLoss, self).__init__(config)
        self.margin = 1.0
        self.component_names = ["contrastive"]

    def calculate(self, latent, generator_flags):
        batch_size = latent.size(0)
        distances = torch.cdist(latent, latent, p=2)
        generator_flags = generator_flags.view(-1, 1)
        same_generator = (generator_flags == generator_flags.t()).float()
        pos_loss = same_generator * distances.pow(2)
        neg_loss = (1 - same_generator) * F.relu(self.margin - distances).pow(2)
        num_pairs = batch_size * (batch_size - 1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pairs
        return (loss,)


# ---------------------------
# Additional Composite Losses for VAE
# ---------------------------
class VAELossEMD(VAELoss):
    """
    VAE loss augmented with an Earth Mover's Distance (EMD) term.

    Config parameters:
      - emd_weight: weight for the EMD term.
      - emd: dict for WassersteinLoss config.
    """

    def __init__(self, config):
        super(VAELossEMD, self).__init__(config)
        self.emd_weight = self.config.reg_param
        self.emd_loss_fn = WassersteinLoss(config)
        self.component_names = ["loss", "vae_loss", "reco", "kl", "emd"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        """
        In addition to the standard VAE inputs, this loss requires:
          - emd_p: first distribution tensor (e.g. a predicted histogram)
          - emd_q: second distribution tensor (e.g. a target histogram)
        """
        base_loss = super(VAELossEMD, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        vae_loss, recon_loss, kl_loss = base_loss
        # calculate EMD against eta distributions
        emd_p = recon[:, :, -4].flatten()
        emd_q = target[:, :, -4].flatten()

        emd_loss = self.emd_loss_fn.calculate(emd_p, emd_q)
        loss = vae_loss + self.emd_weight * emd_loss
        return loss, vae_loss, recon_loss, kl_loss, emd_loss


class VAELossL1(VAELoss):
    """
    VAE loss augmented with an L1 regularization term.

    Config parameters:
      - l1_weight: weight for the L1 regularization term.
    """

    def __init__(self, config):
        super(VAELossL1, self).__init__(config)
        self.l1_weight = self.config.reg_param
        self.l1_reg_fn = L1Regularization(config)
        self.component_names = ["loss", "vae_loss", "reco", "kl", "l1"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        """
        'parameters' should be a list of model parameters to regularize.
        """
        base_loss = super(VAELossL1, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        loss, recon_loss, kl_loss = base_loss
        l1_loss = self.l1_reg_fn.calculate(parameters)
        loss = vae_loss + self.l1_weight * l1_loss
        return loss, vae_loss, recon_loss, kl_loss, l1_loss


class VAELossL2(VAELoss):
    """
    VAE loss augmented with an L2 regularization term.

    Config parameters:
      - l2_weight: weight for the L2 regularization term.
    """

    def __init__(self, config):
        super(VAELossL2, self).__init__(config)
        self.l2_weight = self.config.reg_param
        self.l2_reg_fn = L2Regularization(config)
        self.component_names = ["loss", "vae_loss", "reco", "kl", "l2"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        """
        'parameters' should be a list of model parameters to regularize.
        """
        base_loss = super(VAELossL2, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        loss, recon_loss, kl_loss = base_loss
        l2_loss = self.l2_reg_fn.calculate(parameters)
        loss = vae_loss + self.l2_weight * l2_loss
        return loss, vae_loss, recon_loss, kl_loss, l2_loss


# ---------------------------
# Additional Composite Losses for VAE with Flow
# ---------------------------
class VAEFlowLossEMD(VAEFlowLoss):
    """
    VAE loss augmented with an Earth Mover's Distance (EMD) term.

    Config parameters:
      - emd_weight: weight for the EMD term.
      - emd: dict for WassersteinLoss config.
    """

    def __init__(self, config):
        super(VAEFlowLossEMD, self).__init__(config)
        self.emd_weight = self.config.reg_param
        self.emd_loss_fn = WassersteinLoss(config)
        self.component_names = ["loss", "vae_flow_loss", "reco", "kl", "emd"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        """
        In addition to the standard VAE inputs, this loss requires:
          - emd_p: first distribution tensor (e.g. a predicted histogram)
          - emd_q: second distribution tensor (e.g. a target histogram)
        """
        base_loss = super(VAEFlowLossEMD, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        vae_loss, recon_loss, kl_loss = base_loss
        # calculate EMD against eta distributions
        emd_p = recon[:, :, -4].flatten()
        emd_q = target[:, :, -4].flatten()

        emd_loss = self.emd_loss_fn.calculate(emd_p, emd_q)
        loss = vae_loss + self.emd_weight * emd_loss
        return loss, vae_loss, recon_loss, kl_loss, emd_loss


class VAEFlowLossL1(VAEFlowLoss):
    """
    VAE loss augmented with an L1 regularization term.

    Config parameters:
      - l1_weight: weight for the L1 regularization term.
    """

    def __init__(self, config):
        super(VAEFlowLossL1, self).__init__(config)
        self.l1_weight = self.config.reg_param
        self.l1_reg_fn = L1Regularization(config)
        self.component_names = ["loss", "vae_flow_loss", "reco", "kl", "l1"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        """
        'parameters' should be a list of model parameters to regularize.
        """
        base_loss = super(VAEFlowLossL1, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        vae_loss, recon_loss, kl_loss = base_loss
        l1_loss = self.l1_reg_fn.calculate(parameters)
        loss = vae_loss + self.l1_weight * l1_loss
        return loss, vae_loss, recon_loss, kl_loss, l1_loss


class VAEFlowLossL2(VAEFlowLoss):
    """
    VAE loss augmented with an L2 regularization term.

    Config parameters:
      - l2_weight: weight for the L2 regularization term.
    """

    def __init__(self, config):
        super(VAEFlowLossL2, self).__init__(config)
        self.l2_weight = self.config.reg_param
        self.l2_reg_fn = L2Regularization(config)
        self.component_names = ["loss", "vae_flow_loss", "reco", "kl", "l2"]

    def calculate(self, recon, target, mu, logvar, parameters, log_det_jacobian=0):
        """
        'parameters' should be a list of model parameters to regularize.
        """
        base_loss = super(VAEFlowLossL2, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        vae_loss, recon_loss, kl_loss = base_loss
        l2_loss = self.l2_reg_fn.calculate(parameters)
        loss = vae_loss + self.l2_weight * l2_loss
        return loss, vae_loss, recon_loss, kl_loss, l2_loss
