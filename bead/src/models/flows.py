"""
Collection of flow strategies for variational inference and density estimation.

This module implements various normalizing flow architectures that can be used to transform
simple probability distributions into more complex ones. These flows are particularly useful
for improving the expressiveness of variational autoencoders by allowing more flexible posterior
distributions.

Classes:
    Planar: Planar flow transformation.
    Sylvester: Standard Sylvester normalizing flow.
    TriangularSylvester: Sylvester flow with triangular structure.
    IAF: Inverse Autoregressive Flow.
    CNN_Flow: Convolutional neural network based normalizing flow.
    NSF_AR: Neural Spline Flow with autoregressive structure.
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from .layers import (
    FCNN,
    Dilation_Block,
    MaskedConv2d,
    MaskedLinear,
    unconstrained_RQS,
)


class Planar(nn.Module):
    def __init__(self):
        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """Derivative of tanh"""

        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b): # zk: (B, D), u: (B, D), w: (B, D), b: (B, 1)
        # Calculate w^T u (dot product for each batch item) for invertibility constraint
        # w: (B, D), u: (B, D) => uw: (B, 1)
        uw = torch.sum(w * u, dim=1, keepdim=True)

        # Reparameterize u: u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
        # m_uw: (B, 1)
        m_uw = -1.0 + self.softplus(uw)
        # w_norm_sq: (B, 1)
        w_norm_sq = torch.sum(w**2, dim=1, keepdim=True)
        # u_hat: (B, D)
        # (m_uw - uw): (B,1) broadcasts with w:(B,D) to (B,D)
        # w_norm_sq: (B,1) broadcasts for division
        u_hat = u + (m_uw - uw) * w / (w_norm_sq + 1e-6) # Added epsilon for stability

        # Compute flow: z' = z + u_hat * h(w^T z + b)
        # linear_term (w^T z + b):
        # zk: (B, D), w: (B, D) => torch.sum(zk * w, dim=1, keepdim=True) is (B, 1)
        # b: (B, 1)
        linear_term = torch.sum(zk * w, dim=1, keepdim=True) + b

        # z_new: (B, D)
        # u_hat: (B, D)
        # self.h(linear_term): self.h((B,1)) is (B,1), broadcasts with u_hat to (B,D)
        z_new = zk + u_hat * self.h(linear_term)

        # Compute log determinant of Jacobian: log |1 + u_hat^T * psi_vec|
        # where psi_vec = h'(w^T z + b) * w
        # self.der_h(linear_term): (B, 1)
        # w: (B, D)
        # psi_term: (B, D) via broadcasting
        psi_term = self.der_h(linear_term) * w

        # log_det_jacobian:
        # psi_term: (B, D), u_hat: (B, D) => torch.sum(psi_term * u_hat, dim=1, keepdim=True) is (B, 1)
        log_det_jacobian = torch.log(torch.abs(1 + torch.sum(psi_term * u_hat, dim=1, keepdim=True)) + 1e-6) # Added epsilon for log stability
        log_det_jacobian = log_det_jacobian.squeeze(-1) # Result shape (B)

        return z_new, log_det_jacobian


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):
        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(
            torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1
        ).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer("triu_mask", Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer("diag_idx", diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):
        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer("diag_idx", diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):
    def __init__(
        self,
        z_size,
        num_flows=2,
        num_hidden=0,
        h_size=50,
        forget_bias=1.0,
        conv2d=False,
    ):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer("flip_idx", flip_idx)

        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):
        logdets = 0.0
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]

            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = torch.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets


class CNN_Flow(nn.Module):
    def __init__(self, dim, cnn_layers, kernel_size, test_mode=0, use_revert=True):
        super(CNN_Flow, self).__init__()

        # prepare reversion matrix
        self.usecuda = torch.cuda.is_available()
        self.use_revert = use_revert
        self.R = Variable(
            torch.from_numpy(np.flip(np.eye(dim), axis=1).copy()).float(),
            requires_grad=False,
        )
        if self.usecuda:
            self.R = self.R.cuda()

        self.layers = nn.ModuleList()
        for i in range(cnn_layers):
            block = Dilation_Block(dim, kernel_size, test_mode)
            self.layers.append(block)

    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.layers)):
            output, logdet = self.layers[i](output)
            # revert the dimension of the output after each block
            if self.use_revert:
                z = output.mm(self.R)
            logdetSum += logdet

        return z, logdetSum


class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.
    [Durkan et al. 2019]
    """

    def __init__(self, dim=15, K=64, B=3, hidden_dim=8, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, -1 / 2, 1 / 2)

    def forward(self, x):
        z = torch.zeros_like(x)
        logdets = 0  # torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim=1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B
            )
            logdets += ld
        return z, logdets
