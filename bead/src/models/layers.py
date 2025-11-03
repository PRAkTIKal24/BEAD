"""
Custom layer implementations for neural network architectures.

This module provides specialized neural network layers used in various models across
the BEAD framework. These include masked layers for autoregressive models, CNN flow layers,
graph convolutional layers, and utility functions for spline-based flows.

Classes:
    Identity: Simple identity layer that returns its input unchanged.
    MaskedLinear: Linear layer with masking for autoregressive architectures.
    MaskedConv2d: 2D convolutional layer with masking capabilities.
    CNN_Flow_Layer: Base layer for CNN-based normalizing flows.
    Dilation_Block: Block of dilated convolutions for CNN flows.
    GraphConvolution: Graph convolutional network layer.
    FCNN: Simple fully connected neural network.
    Log1pScaler: Scaler that applies log(1+x) transformation.
    L2Normalizer: Scaler that applies L2 normalization.
    SinCosTransformer: Transforms angles to sin/cos features.
    ChainedScaler: Chains multiple scalers together.

Functions:
    searchsorted: Utility for finding indices where elements should be inserted.
    unconstrained_RQS: Rational quadratic spline transformation with unconstrained inputs.
    RQS: Rational quadratic spline transformation.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, diagonal_zeros=False, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0

        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1 :, i * k : (i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i : i + 1, i * k : (i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k :, i : i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k : (i + 1) * k :, i : i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask * self.weight)

        if self.bias is not None:
            return output.add(self.bias.expand_as(output))
        else:
            return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ", diagonal_zeros="
            + str(self.diagonal_zeros)
            + ", bias="
            + str(bias)
            + ")"
        )


class MaskedConv2d(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        size_kernel=(3, 3),
        diagonal_zeros=False,
        bias=True,
    ):
        super(MaskedConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.size_kernel = size_kernel
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(
            torch.FloatTensor(out_features, in_features, *self.size_kernel)
        )
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features

        assert n_out % n_in == 0 or n_in % n_out == 0, "%d - %d" % (n_in, n_out)

        # Build autoregressive mask
        l = (self.size_kernel[0] - 1) // 2  # noqa: E741
        m = (self.size_kernel[1] - 1) // 2
        mask = np.ones(
            (n_out, n_in, self.size_kernel[0], self.size_kernel[1]), dtype=np.float32
        )
        mask[:, :, :l, :] = 0
        mask[:, :, l, :m] = 0

        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i * k : (i + 1) * k, i + 1 :, l, m] = 0
                if self.diagonal_zeros:
                    mask[i * k : (i + 1) * k, i : i + 1, l, m] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[i : i + 1, (i + 1) * k :, l, m] = 0
                if self.diagonal_zeros:
                    mask[i : i + 1, i * k : (i + 1) * k :, l, m] = 0

        return mask

    def forward(self, x):
        output = F.conv2d(x, self.mask * self.weight, bias=self.bias, padding=(1, 1))
        return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ", diagonal_zeros="
            + str(self.diagonal_zeros)
            + ", bias="
            + str(bias)
            + ", size_kernel="
            + str(self.size_kernel)
            + ")"
        )


class CNN_Flow_Layer(nn.Module):
    def __init__(
        self, dim, kernel_size, dilation, test_mode=0, rescale=True, skip=True
    ):
        super(CNN_Flow_Layer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.test_mode = test_mode
        self.rescale = rescale
        self.skip = skip
        self.usecuda = True

        if (
            self.rescale
        ):  # last layer of flow needs to account for the scale of target variable
            self.lmbd = nn.Parameter(torch.FloatTensor(self.dim).normal_())  # .cuda())

        self.conv1d = nn.Conv1d(1, 1, kernel_size, dilation=dilation)

    def forward(self, x):
        # pad zero to the right
        padded_x = F.pad(x, (0, (self.kernel_size - 1) * self.dilation))

        conv1d = self.conv1d(padded_x.unsqueeze(1)).squeeze()  # (bs, 1, width)

        w = self.conv1d.weight.squeeze()

        # make sure u[i]w[0] >= -1 to ensure invertibility for h(x)=tanh(x) and with skip

        neg_slope = 1e-2
        activation = F.leaky_relu(conv1d, negative_slope=neg_slope)
        activation_gradient = (activation >= 0).float() + (
            activation < 0
        ).float() * neg_slope

        # for 0<=h'(x)<=1, ensure u*w[0]>-1
        scale = (
            (w[0] == 0).float() * self.lmbd
            + (w[0] > 0).float() * (-1.0 / w[0] + F.softplus(self.lmbd))
            + (w[0] < 0).float() * (-1.0 / w[0] - F.softplus(self.lmbd))
        )

        if self.rescale:
            if self.test_mode:
                activation = activation.unsqueeze(dim=0)
                activation_gradient = activation_gradient.unsqueeze(dim=0)
            output = activation.mm(torch.diag(scale))
            activation_gradient = activation_gradient.mm(torch.diag(scale))
        else:
            output = activation

        if self.skip:
            output = output + x
            logdet = torch.log(torch.abs(activation_gradient * w[0] + 1)).sum(1)

        else:
            logdet = torch.log(torch.abs(activation_gradient * w[0])).sum(1)

        return output, logdet


class Dilation_Block(nn.Module):
    def __init__(self, dim, kernel_size, test_mode=0):
        super(Dilation_Block, self).__init__()

        self.block = nn.ModuleList()
        i = 0
        while 2**i <= dim:
            conv1d = CNN_Flow_Layer(
                dim, kernel_size, dilation=2**i, test_mode=test_mode
            )
            self.block.append(conv1d)
            i += 1

    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.block)):
            output, logdet = self.block[i](output)
            logdetSum += logdet

        return output, logdetSum


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_RQS(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, logabsdet


def RQS(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # if torch.min(inputs) < left or torch.max(inputs) > right:
    # raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


class EFPEmbedding(nn.Module):
    """
    Energy-Flow Polynomial (EFP) Embedding Layer.

    Transforms high-dimensional EFP features into compact embedding tokens suitable
    for transformer architectures and other downstream models. Provides learnable
    feature selection through gated sparsification and dimensionality reduction.

    Scientific Motivation:
        EFP features are high-dimensional (140-531 features per jet) and potentially
        redundant. This embedding layer enables:
        - Compression: Reduces dimensionality while preserving information
        - Selection: Learnable gating mechanism for feature importance
        - Sparsification: Threshold-based pruning to prevent overfitting
        - Standardization: Consistent output format for downstream architectures

    Architecture:
        Input: (batch_size, n_jets, n_efp_features)
        ↓ Linear projection (dimensionality reduction)
        ↓ Gated sparsification (learnable feature selection)
        ↓ Layer normalization (training stability)
        ↓ Dropout (regularization)
        Output: (batch_size, n_jets, embedding_dim)

    Args:
        n_efp_features (int): Number of input EFP features (140 or 531)
        embedding_dim (int): Output embedding dimension (default: 64)
        gate_type (str): Gate activation function ('sigmoid', 'relu6', 'tanh')
        gate_threshold (float): Sparsification threshold (default: 0.05)
        dropout_rate (float): Dropout rate for regularization (default: 0.1)
        use_layer_norm (bool): Enable layer normalization (default: True)
        monitor_sparsity (bool): Track gate activation statistics (default: True)

    Example:
        >>> embedding = EFPEmbedding(n_efp_features=140, embedding_dim=64)
        >>> efp_features = torch.randn(32, 3, 140)  # batch_size=32, n_jets=3
        >>> embeddings = embedding(efp_features)
        >>> print(embeddings.shape)  # torch.Size([32, 3, 64])
    """

    def __init__(
        self,
        n_efp_features: int,
        embedding_dim: int = 64,
        gate_type: str = "sigmoid",
        gate_threshold: float = 0.05,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        monitor_sparsity: bool = True,
    ):
        super(EFPEmbedding, self).__init__()

        # Validate inputs
        if n_efp_features <= 0:
            raise ValueError(f"n_efp_features must be positive, got {n_efp_features}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if gate_type not in ["sigmoid", "relu6", "tanh"]:
            raise ValueError(
                f"gate_type must be one of ['sigmoid', 'relu6', 'tanh'], got {gate_type}"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError(f"gate_threshold must be in [0, 1], got {gate_threshold}")

        # Store configuration
        self.n_efp_features = n_efp_features
        self.embedding_dim = embedding_dim
        self.gate_type = gate_type
        self.gate_threshold = gate_threshold
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.monitor_sparsity = monitor_sparsity

        # Core embedding layers
        self.projection = nn.Linear(n_efp_features, embedding_dim)
        self.gate = nn.Linear(n_efp_features, embedding_dim)

        # Optional normalization and regularization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        else:
            self.layer_norm = None

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        # Gate activation function
        if gate_type == "sigmoid":
            self.gate_activation = torch.sigmoid
        elif gate_type == "relu6":
            self.gate_activation = lambda x: F.relu6(x) / 6.0  # Normalize to [0, 1]
        elif gate_type == "tanh":
            self.gate_activation = (
                lambda x: (torch.tanh(x) + 1.0) / 2.0
            )  # Map to [0, 1]

        # Sparsity monitoring (if enabled)
        if self.monitor_sparsity:
            self.register_buffer("gate_activation_count", torch.zeros(1))
            self.register_buffer("total_gate_count", torch.zeros(1))

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initialize layer parameters using Xavier/Glorot initialization.

        This ensures stable gradients and proper scaling for the embedding
        transformation, especially important for high-dimensional EFP inputs.
        """
        # Xavier initialization for projection and gate layers
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.gate.weight)

        # Initialize biases to zero
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(
        self, efp_features: torch.Tensor, jet_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the EFP embedding layer.

        Args:
            efp_features (torch.Tensor): Input EFP features of shape (B, J, N_efp)
            jet_mask (torch.Tensor, optional): Jet validity mask of shape (B, J)
                                             True for valid jets, False for padded jets

        Returns:
            torch.Tensor: Embedded EFP tokens of shape (B, J, embedding_dim)

        Raises:
            ValueError: If input tensor has incorrect shape or contains invalid values
        """
        # Validate input shape
        if efp_features.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, n_jets, n_efp_features), got {efp_features.dim()}D"
            )

        batch_size, n_jets, n_features = efp_features.shape
        if n_features != self.n_efp_features:
            raise ValueError(
                f"Expected {self.n_efp_features} EFP features, got {n_features}"
            )

        # Check for invalid values (NaN, Inf)
        if torch.isnan(efp_features).any() or torch.isinf(efp_features).any():
            raise ValueError("Input contains NaN or Inf values")

        # Linear projection: (B, J, N_efp) -> (B, J, embedding_dim)
        projected = self.projection(efp_features)

        # Gated sparsification: learnable feature selection
        gate_logits = self.gate(efp_features)
        gate_weights = self.gate_activation(gate_logits)

        # Apply sparsification threshold
        if self.gate_threshold > 0.0:
            gate_weights = gate_weights * (gate_weights >= self.gate_threshold).float()

        # Apply gating to projection
        gated_projection = projected * gate_weights

        # Layer normalization (optional)
        if self.layer_norm is not None:
            gated_projection = self.layer_norm(gated_projection)

        # Dropout regularization (optional)
        if self.dropout is not None and self.training:
            gated_projection = self.dropout(gated_projection)

        # Apply jet masking if provided
        if jet_mask is not None:
            if jet_mask.shape != (batch_size, n_jets):
                raise ValueError(
                    f"jet_mask shape {jet_mask.shape} doesn't match input shape ({batch_size}, {n_jets})"
                )

            # Zero out embeddings for padded jets
            mask_expanded = jet_mask.unsqueeze(-1).float()  # (B, J, 1)
            gated_projection = gated_projection * mask_expanded

        # Update sparsity statistics (if monitoring enabled)
        if self.monitor_sparsity and self.training:
            self._update_sparsity_stats(gate_weights, jet_mask)

        return gated_projection

    def _update_sparsity_stats(
        self, gate_weights: torch.Tensor, jet_mask: torch.Tensor = None
    ):
        """
        Update running statistics for gate activation sparsity.

        Args:
            gate_weights (torch.Tensor): Gate activation weights (B, J, embedding_dim)
            jet_mask (torch.Tensor, optional): Jet validity mask (B, J)
        """
        with torch.no_grad():
            # Count active gates (above threshold)
            active_gates = (gate_weights >= self.gate_threshold).float()

            if jet_mask is not None:
                # Only count gates for valid jets
                mask_expanded = jet_mask.unsqueeze(-1).float()
                active_gates = active_gates * mask_expanded
                total_gates = jet_mask.sum() * self.embedding_dim
            else:
                total_gates = gate_weights.numel()

            # Update running counts
            self.gate_activation_count += active_gates.sum()
            self.total_gate_count += total_gates

    def get_sparsity_stats(self) -> dict:
        """
        Get current sparsity statistics.

        Returns:
            dict: Dictionary containing sparsity metrics:
                - 'sparsity_ratio': Fraction of gates below threshold
                - 'activation_ratio': Fraction of gates above threshold
                - 'total_gates_seen': Total number of gates processed
        """
        if not self.monitor_sparsity:
            return {
                "sparsity_ratio": None,
                "activation_ratio": None,
                "total_gates_seen": 0,
            }

        if self.total_gate_count == 0:
            return {
                "sparsity_ratio": 0.0,
                "activation_ratio": 0.0,
                "total_gates_seen": 0,
            }

        activation_ratio = (self.gate_activation_count / self.total_gate_count).item()
        sparsity_ratio = 1.0 - activation_ratio

        return {
            "sparsity_ratio": sparsity_ratio,
            "activation_ratio": activation_ratio,
            "total_gates_seen": self.total_gate_count.item(),
        }

    def reset_sparsity_stats(self):
        """
        Reset sparsity monitoring statistics.
        """
        if self.monitor_sparsity:
            self.gate_activation_count.zero_()
            self.total_gate_count.zero_()

    def extra_repr(self) -> str:
        """
        Extra representation string for the module.

        Returns:
            str: String representation of the layer configuration
        """
        return (
            f"n_efp_features={self.n_efp_features}, "
            f"embedding_dim={self.embedding_dim}, "
            f"gate_type={self.gate_type}, "
            f"gate_threshold={self.gate_threshold}, "
            f"dropout_rate={self.dropout_rate}, "
            f"use_layer_norm={self.use_layer_norm}, "
            f"monitor_sparsity={self.monitor_sparsity}"
        )
