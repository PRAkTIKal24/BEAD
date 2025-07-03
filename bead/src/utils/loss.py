"""
Loss functions for training autoencoder and VAE models.

This module provides various loss functions for training autoencoders and variational autoencoders,
including basic reconstruction losses, KL divergence, regularization terms, and combined losses
for specialized models like those with normalizing flows.

Classes:
    BaseLoss: Base class for all loss functions.
    ReconstructionLoss: Standard reconstruction loss (MSE or L1).
    KLDivergenceLoss: Kullback-Leibler divergence for VAE training.
    WassersteinLoss: Earth Mover's Distance approximation.
    L1Regularization: L1 weight regularization.
    L2Regularization: L2 weight regularization.
    BinaryCrossEntropyLoss: Binary cross-entropy loss.
    VAELoss: Combined loss for VAE (reconstruction + KL).
    VAEFlowLoss: Loss for VAE with normalizing flows.
    ContrastiveLoss: Contrastive loss for clustering latent vectors.
    VAELossEMD: VAE loss with Earth Mover's Distance term.
    VAELossL1: VAE loss with L1 regularization.
    VAELossL2: VAE loss with L2 regularization.
    VAEFlowLossEMD: VAE flow loss with EMD term.
    VAEFlowLossL1: VAE flow loss with L1 regularization.
    VAEFlowLossL2: VAE flow loss with L2 regularization.
    DVAELoss: Combined loss for DirichletConvVAE (reconstruction + KL(dirichlet)) inherits from VAELoss.
    DVAEFlowLoss: Combined loss for DirichletConvVAE (reconstruction + KL(dirichlet)) inherits from VAEFlowLoss.
"""

import torch
import torch.distributed as dist
from torch.nn import functional as F


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

    def calculate(self, recon, target, mu, logvar, parameters=None, log_det_jacobian=0, zk=None, **kwargs):
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

    Supports:
    - Gaussian prior
    - Dirichlet prior via Laplace approximation
    """

    def __init__(self, config, prior: str = "gaussian"):
        super(KLDivergenceLoss, self).__init__(config)
        self.component_names = ["kl"]
        self.prior = prior.lower()

        if self.prior == "dirichlet":
            self.alpha_prior_value = 5.0

    def compute_alpha_laplace(self, mu, logvar):
        """
        Compute Dirichlet concentration parameters α from
        Gaussian parameters (μ, logvar) via Laplace bridge approximation.
        """
        K = mu.size(1)
        var = torch.exp(logvar)
        exp_mu = torch.exp(mu)
        exp_minus_mu= torch.exp(-mu)
        sum_exp_minus = exp_minus_mu.sum(dim=1, keepdim=True)

        # Laplace Bridge approximation
        term = 1.0 - 2.0 / K + (exp_mu * sum_exp_minus) / (K ** 2)
        alpha = term / var
        return alpha

    def calculate(self, recon, target, mu, logvar, parameters=None, log_det_jacobian=0, zk=None, z0=None, **kwargs):
        batch_size = mu.size(0)

        if self.prior == "dirichlet":
            D_z = self.compute_alpha_laplace(mu, logvar)

            q_z = torch.distributions.Dirichlet(D_z)
            prior = torch.distributions.Dirichlet(torch.full_like(D_z, self.alpha_prior_value))

            kl_loss = torch.distributions.kl_divergence(q_z, prior)

            return (kl_loss/batch_size,)

        else:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return (kl_loss/batch_size,)


# ---------------------------
# SupCon Loss
# ---------------------------
class SupervisedContrastiveLoss(BaseLoss):
    """
    Supervised Contrastive Learning loss function.
    Based on: https://arxiv.org/abs/2004.11362
    """

    def __init__(self, config):
        super(SupervisedContrastiveLoss, self).__init__(config)
        self.temperature = (
            config.contrastive_temperature
            if hasattr(config, "contrastive_temperature")
            else 0.07
        )
        self.component_names = ["supcon"]
        # DDP related attributes
        self.is_ddp_active = (
            config.is_ddp_active if hasattr(config, "is_ddp_active") else False
        )
        self.world_size = config.world_size if hasattr(config, "world_size") else 1

    def calculate(self, features, labels, **kwargs):
        """
        Args:
            features (torch.Tensor): Latent vectors (e.g., zk), shape [batch_size, feature_dim].Assumed to be L2-normalized.
            labels (torch.Tensor): Ground truth labels (generator_ids), shape [batch_size].
        Returns:
            torch.Tensor: Supervised contrastive loss.
        """
        device = features.device

        if self.is_ddp_active and self.world_size > 1:
            # Gather features and labels from all GPUs
            gathered_features_list = [
                torch.zeros_like(features) for _ in range(self.world_size)
            ]
            gathered_labels_list = [
                torch.zeros_like(labels) for _ in range(self.world_size)
            ]

            dist.all_gather(gathered_features_list, features)
            dist.all_gather(gathered_labels_list, labels)

            features = torch.cat(gathered_features_list, dim=0)
            labels = torch.cat(gathered_labels_list, dim=0)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        # Mask to identify positive pairs
        mask = torch.eq(labels, labels.T).float().to(device)

        # Similarity definition
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = (
            anchor_dot_contrast - logits_max.detach()
        )  # Detach to avoid gradients through max

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask  # Positive pairs, excluding self

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Compute mean of log-likelihood over positive pairs
        num_pos_per_anchor = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (num_pos_per_anchor + 1e-9)

        # NLL
        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()  # Average over the batch

        return (loss,)


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

    def calculate(self, p, q, **kwargs):
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

    def calculate(self, parameters, **kwargs):
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

    def calculate(self, parameters, **kwargs):
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

    def calculate(self, predictions, targets, **kwargs):
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
        self.kl_weight = torch.tensor(self.config.reg_param)
        self.component_names = ["loss", "reco", "kl"]

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
        recon_loss = self.recon_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=log_det_jacobian, zk=zk, generator_labels=generator_labels
        )
        kl_loss = self.kl_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=log_det_jacobian, zk=zk, generator_labels=generator_labels
        )
        loss = recon_loss[0] + self.kl_weight * kl_loss[0]
        return loss, recon_loss[0], kl_loss[0]


# ---------------------------
# VAE+Flow Loss
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
        self.kl_weight = torch.tensor(self.config.reg_param)
        self.flow_weight = torch.tensor(self.config.reg_param)
        self.component_names = ["loss", "reco", "kl"]

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        z0=None,
        log_det_jacobian=0,
        generator_labels=None,
        **kwargs,
    ):
        recon_loss = self.recon_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=log_det_jacobian, zk=zk, z0=z0, generator_labels=generator_labels
        )[0]
        kl_loss = self.kl_loss_fn.calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=log_det_jacobian, zk=zk, z0=z0, generator_labels=generator_labels
        )[0]
        # Ensure log_det_jacobian is a tensor
        if not isinstance(log_det_jacobian, torch.Tensor):
            log_det_jacobian_tensor = torch.tensor(
                log_det_jacobian, device=target.device, dtype=target.dtype
            )
        else:
            log_det_jacobian_tensor = log_det_jacobian

        # Calculate mean log determinant of the Jacobian
        mean_log_det_jacobian = log_det_jacobian_tensor.mean()

        # Ensure weights are on the same device; not necessary
        kl_weight_device = self.kl_weight.to(recon_loss.device)
        flow_weight_device = self.flow_weight.to(recon_loss.device)

        total_loss = (
            recon_loss
            + kl_weight_device * kl_loss
            - flow_weight_device * mean_log_det_jacobian
        )

        return total_loss, recon_loss, kl_loss


# ---------------------------
# VAE+SupCon Loss
# ---------------------------
class VAESupConLoss(BaseLoss):
    """
    Combined loss for VAE with Supervised Contrastive Learning.

    Config parameters:
        - vae: dict for VAELoss config.
        - supcon: dict for SupervisedContrastiveLoss config.
        - contrastive_weight: weight for the contrastive loss term.
    """

    def __init__(self, config):
        super(VAESupConLoss, self).__init__(config)
        self.vae_loss_fn = VAELoss(config)
        self.supcon_loss_fn = SupervisedContrastiveLoss(config)
        self.reg_param = torch.tensor(config.reg_param)
        self.contrastive_weight = torch.tensor(config.contrastive_weight)
        self.component_names = [
            "loss",
            "vae_loss",
            "reco_loss",
            "kl_loss",
            "supcon_loss",
        ]

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
        # Calculate VAE loss components
        vae_loss, reco_loss, kl_loss = self.vae_loss_fn.calculate(
            recon, target, mu, logvar, zk, parameters, log_det_jacobian, generator_labels
        )

        # Calculate Supervised Contrastive loss only if generator_labels are provided; if not, fallback to ELBO loss
        if generator_labels is None:
            return vae_loss, vae_loss, reco_loss, kl_loss, torch.tensor(0.0)
        else:
            # L2 normalize zk for SupCon loss
            zk_normalized = F.normalize(zk, p=2, dim=1)
            # Calculate Supervised Contrastive loss
            supcon_loss = self.supcon_loss_fn.calculate(
                zk_normalized, generator_labels
            )[0]
            # Ensure weights are on the same device
            contrastive_weight_device = self.contrastive_weight.to(vae_loss.device)

            # Combine losses
            loss = vae_loss + contrastive_weight_device * supcon_loss

            return loss, vae_loss, reco_loss, kl_loss, supcon_loss


# ---------------------------
# VAE+Flow+SupCon Loss
# ---------------------------
class VAEFlowSupConLoss(BaseLoss):
    """
    Combined loss for VAE with Normalizing Flows and Supervised Contrastive Learning.

    Config parameters:
        - vaeflow: dict for VAEFlowLoss config.
        - supcon: dict for SupervisedContrastiveLoss config.
        - contrastive_weight: weight for the contrastive loss term.
    """

    def __init__(self, config):
        super(VAEFlowSupConLoss, self).__init__(config)
        self.vaeflow_loss_fn = VAEFlowLoss(config)
        self.supcon_loss_fn = SupervisedContrastiveLoss(config)
        self.contrastive_weight = torch.tensor(config.contrastive_weight)
        self.component_names = [
            "loss",
            "vaeflow_loss",
            "reco_loss",
            "kl_loss",
            "supcon_loss",
        ]

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
        # Calculate VAEFlow loss components
        vaeflow_loss, reco_loss, kl_loss = self.vaeflow_loss_fn.calculate(
            recon, target, mu, logvar, zk, parameters, log_det_jacobian, generator_labels
        )

        # Calculate Supervised Contrastive loss only if generator_labels are provided; if not, fallback to ELBO loss
        if generator_labels is None:
            return vaeflow_loss, vaeflow_loss, reco_loss, kl_loss, torch.tensor(0.0)
        else:
            # L2 normalize zk for SupCon loss
            zk_normalized = F.normalize(zk, p=2, dim=1)
            # Calculate Supervised Contrastive loss
            supcon_loss = self.supcon_loss_fn.calculate(
                zk_normalized, generator_labels
            )[0]
            # Ensure weights are on the same device
            contrastive_weight_device = self.contrastive_weight.to(vaeflow_loss.device)

            # Combine losses
            loss = vaeflow_loss + contrastive_weight_device * supcon_loss

            return loss, vaeflow_loss, reco_loss, kl_loss, supcon_loss


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

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
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

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
        """
        'parameters' should be a list of model parameters to regularize.
        """
        base_loss = super(VAELossL1, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        vae_loss, recon_loss, kl_loss = base_loss
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

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
        """
        'parameters' should be a list of model parameters to regularize.
        """
        base_loss = super(VAELossL2, self).calculate(
            recon, target, mu, logvar, parameters, log_det_jacobian=0
        )
        vae_loss, recon_loss, kl_loss = base_loss
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

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
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

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
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

    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        log_det_jacobian=0,
        generator_labels=None,
    ):
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


class DVAELoss(VAELoss):
    """
    DVAELoss: Combines reconstruction loss and Dirichlet KL divergence loss.
    Inherits from VAELoss and overrides the KL loss function to use Dirichlet prior.
    """

    def __init__(self, config):
        super(DVAELoss, self).__init__(config)
        self.kl_loss_fn = KLDivergenceLoss(config, prior="dirichlet")


class DVAEFlowLoss(VAEFlowLoss):
    """
    DVAEFlowLoss: Combines reconstruction loss and Dirichlet KL divergence loss.
    Inherits from VAEFlowLoss and overrides the KL loss function to use Dirichlet prior.
    """
    def __init__(self, config):
        super(DVAEFlowLoss, self).__init__(config)
        self.kl_loss_fn = KLDivergenceLoss(config, prior="dirichlet")


class NTXentLoss(BaseLoss):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    as used in SimCLR and similar contrastive learning frameworks.
    
    This is an unsupervised contrastive loss that works with augmented views of the same data points.
    It pulls together representations of different augmentations of the same sample (positive pairs),
    while pushing away representations of different samples (negative pairs).
    
    References:
    - SimCLR: https://arxiv.org/abs/2002.05709
    """
    
    def __init__(self, config):
        super(NTXentLoss, self).__init__(config)
        self.temperature = config.ntxent_temperature if hasattr(config, "ntxent_temperature") else 0.07
        self.component_names = ["ntxent"]
        # DDP related attributes
        self.is_ddp_active = config.is_ddp_active if hasattr(config, "is_ddp_active") else False
        self.world_size = config.world_size if hasattr(config, "world_size") else 1
    
    def calculate(self, z1, z2):
        """
        Calculate NT-Xent loss for each batch.
        
        Args:
            z1 (torch.Tensor): First set of features/representations, shape [batch_size, feature_dim].
            z2 (torch.Tensor): Second set of features/representations, shape [batch_size, feature_dim].
                These are typically augmented versions of the same inputs.
        
        Returns:
            torch.Tensor: NT-Xent loss value.
        """
        # Get batch size and ensure both representations have the same shape
        batch_size = z1.shape[0]
        assert z1.shape == z2.shape, "Shape mismatch between representation sets"
        
        # Normalize the representations (L2 norm)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate along the batch dimension to create 2N samples
        representations = torch.cat([z1, z2], dim=0)  # [2*batch_size, feature_dim]
        
        # Create similarity matrix (2N × 2N)
        similarity_matrix = torch.matmul(representations, representations.T)  # [2*batch_size, 2*batch_size]
        
        # Create positive pair mask
        # Each sample in z1 has its corresponding augmented version in z2 at the same index
        # So for i in [0, batch_size-1], (i, i+batch_size) and (i+batch_size, i) are positive pairs
        positive_mask = torch.zeros_like(similarity_matrix)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = 1  # First augmentation to second augmentation
            positive_mask[i + batch_size, i] = 1  # Second augmentation to first augmentation
            
        # Create a mask to exclude self-comparisons (diagonal)
        diag_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        
        # Apply temperature scaling to similarity scores
        similarity_matrix = similarity_matrix / self.temperature
        
        # For each row, compute the softmax denominator (sum over all non-self examples)
        exp_sim = torch.exp(similarity_matrix)
        exp_sim_masked = exp_sim * diag_mask  # Zero out self-comparisons
        denominator = exp_sim_masked.sum(dim=1, keepdim=True)
        
        # Compute the numerator (exp of similarity for positive pairs)
        numerator = torch.exp(torch.sum(similarity_matrix * positive_mask, dim=1) / torch.sum(positive_mask, dim=1))
        
        # Compute the final loss for each sample
        loss_per_sample = -torch.log(numerator / (denominator + 1e-8))
        
        # Average loss over the batch
        loss = loss_per_sample.mean()
        
        return (loss,)


class NTXentCombinedLoss(BaseLoss):
    """
    Combined loss that adds NT-Xent contrastive loss to any base loss function.
    
    This loss function wraps another loss function (e.g., VAELoss, VAEFlowLoss)
    and adds the NT-Xent contrastive loss term between two augmented views of the
    latent representations.
    
    Config parameters:
      - base_loss_function: The name of the base loss function to use (e.g., "VAELoss")
      - ntxent_weight: Weight for the NT-Xent loss term.
    """
    
    def __init__(self, config):
        super(NTXentCombinedLoss, self).__init__(config)
        
        # Intelligently parse the loss function name to determine the base loss
        if config.loss_function == "NTXentLoss":
            # For standalone NTXentLoss, use a minimal base loss (ReconstructionLoss)
            base_loss_name = "ReconstructionLoss"
        elif config.loss_function == "NTXentVAELoss":
            base_loss_name = "VAELoss"
        elif config.loss_function == "NTXentVAEFlowLoss":
            base_loss_name = "VAEFlowLoss"
        elif config.loss_function == "NTXentDVAELoss":
            base_loss_name = "DVAELoss"
        else:
            # For custom combinations or backward compatibility
            base_loss_name = getattr(config, "base_loss_function", "VAELoss")
        
        # Get the class by name using globals()
        base_loss_class = globals()[base_loss_name]
        
        # Initialize the base loss function and NT-Xent loss
        self.base_loss_fn = base_loss_class(config)
        self.ntxent_loss_fn = NTXentLoss(config)
        self.ntxent_weight = torch.tensor(config.ntxent_weight if hasattr(config, "ntxent_weight") else 1.0)
        
        # Prepare component names
        self.component_names = ["loss"] + [f"base_{name}" for name in self.base_loss_fn.component_names] + ["ntxent_loss"]
    
    def calculate(
        self,
        recon,
        target,
        mu,
        logvar,
        zk,
        parameters,
        z_aug1=None,
        z_aug2=None,
        log_det_jacobian=0,
        generator_labels=None,
        **kwargs
    ):
        """
        Calculate the combined loss.
        
        Args:
            recon: Reconstructed input
            target: Original input
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            zk: Latent representation (after flows if applicable)
            parameters: Model parameters (for regularization if applicable)
            z_aug1: First augmented latent representation
            z_aug2: Second augmented latent representation
            log_det_jacobian: Log determinant of Jacobian (for flow models)
            generator_labels: Labels for the data (if applicable)
            **kwargs: Additional arguments for the base loss
            
        Returns:
            tuple: Combined loss and component losses
        """
        # Calculate the base loss
        base_losses = self.base_loss_fn.calculate(
            recon=recon,
            target=target,
            mu=mu,
            logvar=logvar,
            zk=zk,
            parameters=parameters,
            log_det_jacobian=log_det_jacobian,
            generator_labels=generator_labels,
            **kwargs
        )
        
        # Only calculate NT-Xent loss if augmented views are provided
        if z_aug1 is not None and z_aug2 is not None:
            ntxent_loss = self.ntxent_loss_fn.calculate(z_aug1, z_aug2)[0]
            weighted_ntxent_loss = ntxent_loss * self.ntxent_weight
        else:
            ntxent_loss = torch.tensor(0.0, device=recon.device)
            weighted_ntxent_loss = ntxent_loss
        
        # Combine losses
        total_loss = base_losses[0] + weighted_ntxent_loss
        
        # Return combined loss and all components
        # Structure must match component_names = ["loss"] + [f"base_{name}" for name in self.base_loss_fn.component_names] + ["ntxent_loss"]
        return (total_loss,) + base_losses + (ntxent_loss,)
