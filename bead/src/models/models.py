"""
Neural network model architectures for anomaly detection.

This module provides various autoencoder and variational autoencoder architectures
with different latent space configurations, flow transformations, and architectural
choices. These models can be used for anomaly detection in particle physics data.

Classes:
    AE: Basic autoencoder architecture.
    AE_Dropout_BN: Autoencoder with dropout and batch normalization.
    ConvAE: Convolutional autoencoder.
    ConvVAE: Convolutional variational autoencoder.
    Dirichlet_ConvVAE: Convolutional Dirichlet variational autoencoder.
    Planar_ConvVAE: ConvVAE with planar normalizing flows.
    OrthogonalSylvester_ConvVAE: ConvVAE with orthogonal Sylvester flows.
    HouseholderSylvester_ConvVAE: ConvVAE with Householder Sylvester flows.
    TriangularSylvester_ConvVAE: ConvVAE with triangular Sylvester flows.
    IAF_ConvVAE: ConvVAE with inverse autoregressive flows.
    ConvFlow_ConvVAE: ConvVAE with convolutional normalizing flows.
    NSFAR_ConvVAE: ConvVAE with neural spline flows.
    TransformerAE: Autoencoder with transformer components.
    FlexibleTransformer: Flexible transformer model that can integrate with any VAE model.
    VAEWithTransformer: VAE model with integrated transformer.
"""

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from . import flows
from . import transformer_utils


class AE(nn.Module):
    # This class is a modified version of the original class by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super(AE, self).__init__(*args, **kwargs)

        self.activations = {}
        self.n_features = in_shape[-1] * in_shape[-2]
        self.z_dim = z_dim
        self.in_shape = in_shape

        # encoder
        self.en1 = nn.Linear(self.n_features, 200)
        self.dr1 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(200)

        self.en2 = nn.Linear(200, 100)
        self.dr2 = nn.Dropout(p=0.4)
        self.bn2 = nn.BatchNorm1d(100)

        self.en3 = nn.Linear(100, 50)
        self.dr3 = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(50)

        self.en4 = nn.Linear(50, self.z_dim)
        self.dr4 = nn.Dropout(p=0.2)
        self.bn4 = nn.BatchNorm1d(self.z_dim)
        self.bn5 = nn.BatchNorm1d(self.n_features)

        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        # decoder
        self.de1 = nn.Linear(self.z_dim, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, self.n_features)

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        return self.de4(h6)

    def forward(self, x):
        x = self.flatten(x)
        z = self.encode(x)
        out = self.decode(z)
        out = out.view(self.in_shape)
        return out, z

    # Implementation of activation extraction using the forward_hook method

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()


class AE_Dropout_BN(AE):
    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # encoder
        self.enc_nn = nn.Sequential(
            self.bn5,
            self.en1,
            self.dr1,
            self.bn1,
            self.leaky_relu,
            self.en2,
            self.dr2,
            self.bn2,
            self.leaky_relu,
            self.en3,
            self.dr3,
            self.bn3,
            self.leaky_relu,
            self.en4,
            self.dr4,
            self.bn4,
        )

        # decoder
        self.dec_nn = nn.Sequential(
            self.bn4,
            self.de1,
            self.leaky_relu,
            self.bn3,
            self.de2,
            self.leaky_relu,
            self.bn2,
            self.de3,
            self.leaky_relu,
            self.bn1,
            self.de4,
            self.bn5,
        )

    def enc_bn(self, x):
        return self.enc_nn(x)

    def dec_bn(self, z):
        return self.dec_nn(z)

    def forward(self, x):
        x = self.flatten(x)
        z = self.enc_bn(x)
        out = self.dec_bn(z)
        out = out.view(self.in_shape)
        return out, z


class ConvAE(nn.Module):
    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_z_mid_dim = 100
        self.conv_op_shape = None
        self.z_dim = z_dim
        self.in_shape = in_shape

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 4), stride=(1,), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=(3, 1), stride=(1,), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 1), stride=(1,), padding=1),
            nn.BatchNorm2d(8),
        )

        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Get size after flattening
        self.q_z_output_dim = self._get_qzconv_output(self.in_shape)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.BatchNorm1d(self.q_z_mid_dim),
            nn.LeakyReLU(),
        )

        self.q_z_latent = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.BatchNorm1d(self.q_z_mid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.BatchNorm1d(self.q_z_output_dim),
        )

        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 16, kernel_size=(3, 1), stride=(1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=(3, 1), stride=(1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 4), stride=(1), padding=1),
        )

    def _get_qzconv_output(self, shape):
        input = Variable(torch.rand(shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.size(1)
        return int(n_size)

    def _forward_features(self, x):
        qz = self.q_z_conv(x)
        return self.flatten(qz)

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        # Latent
        out = self.q_z_latent(out)
        return out

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z


class ConvVAE(ConvAE):
    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Latent distribution parameters
        self.q_z_mean = nn.Linear(self.q_z_mid_dim, self.z_dim)
        self.q_z_logvar = nn.Linear(self.q_z_mid_dim, self.z_dim)

        # log-det-jacobian = 0 without flows
        self.ldj = 0
        self.z_size = z_dim

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        # Latent
        mean = self.q_z_mean(out)
        logvar = self.q_z_logvar(out)
        return out, mean, logvar

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def reparameterize(self, mean, logvar):
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x):
        out, mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar, z


class Dirichlet_ConvVAE(ConvAE):
    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Latent distribution parameters
        self.q_z_mean = nn.Linear(self.q_z_mid_dim, self.z_dim)
        self.q_z_logvar = nn.Linear(self.q_z_mid_dim, self.z_dim)

        # log-det-jacobian = 0 without flows
        self.ldj = 0
        self.z_size = z_dim

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        # Latent
        mean = self.q_z_mean(out)
        logvar = self.q_z_logvar(out)
        return out, mean, logvar

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def reparameterize(self, mean, logvar):
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x):
        out, mean, logvar = self.encode(x)
        G_z = self.reparameterize(mean, logvar)
        # Apply softmax to map z to simplex (Dirichlet approximation)
        D_z = torch.nn.functional.softmax(G_z, dim=-1)
        out = self.decode(D_z)
        return out, mean, logvar, G_z, G_z, D_z


class Planar_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with planar flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0

        # Flow parameters
        flow = flows.Planar
        self.num_flows = 6  # args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.q_z_mid_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module("flow_" + str(k), flow_k)

    def forward(self, x):
        self.log_det_j = 0

        out, z_mu, z_var = self.encode(x)

        batch_size = x.size(0)
        # return amortized u an w for all flows
        u = self.amor_u(out).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(out).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(out).view(batch_size, self.num_flows, 1, 1)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))  # planar.'flow_'+k
            z_k, log_det_jacobian = flow_k(
                z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
            )
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_decoded = self.decode(z[-1])

        return x_decoded, z_mu, z_var, self.log_det_j, z[0], z[-1]


class OrthogonalSylvester_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with orthogonal flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = 4  # args.num_flows
        self.num_ortho_vecs = 5  # args.num_ortho_vecs

        assert (self.num_ortho_vecs <= self.z_size) and (self.num_ortho_vecs > 0)

        # Orthogonalization parameters
        if self.num_ortho_vecs == self.z_size:
            self.cond = 1.0e-5
        else:
            self.cond = 1.0e-6

        self.steps = 100
        identity = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer("_eye", Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(
            torch.ones(self.num_ortho_vecs, self.num_ortho_vecs), diagonal=1
        )
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.num_ortho_vecs).long()

        self.register_buffer("triu_mask", Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer("diag_idx", diag_idx)

        # Amortized flow parameters
        # Diagonal elements of R1 * R2 have to satisfy -1 < R1 * R2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(
            self.q_z_mid_dim,
            self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs,
        )

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation,
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.num_flows * self.num_ortho_vecs),
            self.diag_activation,
        )

        self.amor_q = nn.Linear(
            self.q_z_mid_dim, self.num_flows * self.z_size * self.num_ortho_vecs
        )
        self.amor_b = nn.Linear(self.q_z_mid_dim, self.num_flows * self.num_ortho_vecs)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.num_ortho_vecs)
            self.add_module("flow_" + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        # Reshape to shape (num_flows * batch_size, z_size * num_ortho_vecs)
        q = q.view(-1, self.z_size * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.resize(dim0, self.z_size, self.num_ortho_vecs)

        max_norm = 0

        # Iterative orthogonalization
        for _s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)

            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).data
            if max_norm <= self.cond:
                break

        if max_norm > self.cond:
            print("\nWARNING: orthogonalization not complete")
            print("\t Final max norm =", max_norm)

            # print()

        # Reshaping: first dimension is batch_size
        amat = amat.view(-1, self.num_flows, self.z_size, self.num_ortho_vecs)
        amat = amat.transpose(0, 1)

        return amat

    def forward(self, x):
        self.log_det_j = 0

        out, z_mu, z_var = self.encode(x)

        batch_size = x.size(0)
        # Amortized r1, r2, q, b for all flows

        full_d = self.amor_d(out)
        diag1 = self.amor_diag1(out)
        diag2 = self.amor_diag2(out)

        full_d = full_d.resize(
            batch_size, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows
        )
        diag1 = diag1.resize(batch_size, self.num_ortho_vecs, self.num_flows)
        diag2 = diag2.resize(batch_size, self.num_ortho_vecs, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(out)
        b = self.amor_b(out)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.num_ortho_vecs, self.num_flows)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))
            z_k, log_det_jacobian = flow_k(
                z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k]
            )

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_decoded = self.decode(z[-1])

        return x_decoded, z_mu, z_var, self.log_det_j, z[0], z[-1]


class HouseholderSylvester_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with householder sylvester flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0

        # Flow parameters
        flow = flows.Sylvester
        self.num_flows = 4  # args.num_flows
        self.num_householder = 8  # args.num_householder
        assert self.num_householder > 0

        identity = torch.eye(self.z_size, self.z_size)
        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer("_eye", Variable(identity))
        self._eye.requires_grad = False

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer("triu_mask", Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer("diag_idx", diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(
            self.q_z_mid_dim, self.num_flows * self.z_size * self.z_size
        )

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size),
            self.diag_activation,
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size),
            self.diag_activation,
        )

        self.amor_q = nn.Linear(
            self.q_z_mid_dim, self.num_flows * self.z_size * self.num_householder
        )

        self.amor_b = nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module("flow_" + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        # Reshape to shape (num_flows * batch_size * num_householder, z_size)
        q = q.view(-1, self.z_size)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        v = torch.div(q, norm)

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))

        amat = self._eye - 2 * vvT

        # Reshaping: first dimension is batch_size * num_flows
        amat = amat.view(-1, self.num_householder, self.z_size, self.z_size)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.num_flows, self.z_size, self.z_size)
        amat = amat.transpose(0, 1)

        return amat

    def forward(self, x):
        self.log_det_j = 0
        batch_size = x.size(0)

        out, z_mu, z_var = self.encode(x)

        batch_size = x.size(0)
        # Amortized r1, r2, q, b for all flows
        full_d = self.amor_d(out)
        diag1 = self.amor_diag1(out)
        diag2 = self.amor_diag2(out)

        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        q = self.amor_q(out)
        b = self.amor_b(out)

        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))
            q_k = q_ortho[k]

            z_k, log_det_jacobian = flow_k(
                z[k], r1[:, :, :, k], r2[:, :, :, k], q_k, b[:, :, :, k], sum_ldj=True
            )

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_decoded = self.decode(z[-1])

        return x_decoded, z_mu, z_var, self.log_det_j, z[0], z[-1]


class TriangularSylvester_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with triangular sylvester flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0

        # Flow parameters
        flow = flows.TriangularSylvester
        self.num_flows = 4  # args.num_flows

        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer("flip_idx", flip_idx)

        # Masks needed for triangular r1 and r2.
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()

        self.register_buffer("triu_mask", Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer("diag_idx", diag_idx)

        # Amortized flow parameters
        # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
        self.diag_activation = nn.Tanh()

        self.amor_d = nn.Linear(
            self.q_z_mid_dim, self.num_flows * self.z_size * self.z_size
        )

        self.amor_diag1 = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size),
            self.diag_activation,
        )
        self.amor_diag2 = nn.Sequential(
            nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size),
            self.diag_activation,
        )

        self.amor_b = nn.Linear(self.q_z_mid_dim, self.num_flows * self.z_size)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)

            self.add_module("flow_" + str(k), flow_k)

    def forward(self, x):
        self.log_det_j = 0

        out, z_mu, z_var = self.encode(x)

        batch_size = x.size(0)
        # Amortized r1, r2, b for all flows
        full_d = self.amor_d(out)
        diag1 = self.amor_diag1(out)
        diag2 = self.amor_diag2(out)

        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask

        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2

        b = self.amor_b(out)
        # Resize flow parameters to divide over K flows
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)

        # Sample z_0
        z = [self.reparameterize(z_mu, z_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))
            if k % 2 == 1:
                # Alternate with reorderering z for triangular flow
                permute_z = self.flip_idx
            else:
                permute_z = None

            z_k, log_det_jacobian = flow_k(
                z[k],
                r1[:, :, :, k],
                r2[:, :, :, k],
                b[:, :, :, k],
                permute_z,
                sum_ldj=True,
            )

            z.append(z_k)
            self.log_det_j += log_det_jacobian

        x_decoded = self.decode(z[-1])

        return x_decoded, z_mu, z_var, self.log_det_j, z[0], z[-1]


class IAF_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0
        self.h_size = 60  # args.made_h_size

        self.h_context = nn.Linear(self.q_z_mid_dim, self.h_size)

        # Flow parameters
        self.num_flows = 4
        self.flow = flows.IAF(
            z_size=self.z_size,
            num_flows=self.num_flows,
            num_hidden=1,
            h_size=self.h_size,
            conv2d=False,
        )

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        # Latent
        mean = self.q_z_mean(out)
        logvar = self.q_z_logvar(out)

        # context from previous layer
        h_context = self.h_context(out)

        return mean, logvar, h_context

    def forward(self, x):
        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)
        # iaf flows
        z_k, self.log_det_j = self.flow(z_0, h_context)
        # decode
        x_decoded = self.decode(z_k)

        return x_decoded, z_mu, z_var, self.log_det_j, z_0, z_k


class ConvFlow_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with convolutional flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0
        self.num_flows = 4  # args.num_flows # 6 for chan1
        self.kernel_size = 7  # args.convFlow_kernel_size
        self.test_mode = False

        flow_k = flows.CNN_Flow

        # Normalizing flow layers
        self.flow = flow_k(
            dim=self.z_size,
            cnn_layers=self.num_flows,
            kernel_size=self.kernel_size,
            test_mode=self.test_mode,
        )

    def forward(self, x):
        # mean and variance of z
        out, z_mu, z_var = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)
        # Normalizing flows
        z_k, logdet = self.flow(z_0)
        # decode
        x_decoded = self.decode(z_k)

        return x_decoded, z_mu, z_var, self.log_det_j, z_0, z_k


class NSFAR_ConvVAE(ConvVAE):
    """
    Variational auto-encoder with auto-regressive neural spline flows in the decoder.
    """

    def __init__(self, in_shape, z_dim, *args, **kwargs):
        super().__init__(in_shape, z_dim, *args, **kwargs)
        self.log_det_j = 0
        self.dim = self.z_size
        self.num_flows = 4  # args.num_flows

        flow = flows.NSF_AR

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow(dim=self.dim)

            self.add_module("flow_" + str(k), flow_k)

    def forward(self, x):
        # mean and variance of z
        out, z_mu, z_var = self.encode(x)
        # sample z
        z = [self.reparameterize(z_mu, z_var)]
        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))

            z_k, log_det_jacobian = flow_k(z[k])

            z.append(z_k)
            self.log_det_j += log_det_jacobian
        # decode
        x_decoded = self.decode(z[-1])

        return x_decoded, z_mu, z_var, self.log_det_j, z[0], z[-1]


class TransformerAE(nn.Module):
    """Autoencoder mixed with the Transformer Encoder layer

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_dim,
        h_dim=256,
        n_heads=1,
        latent_size=50,
        activation=torch.nn.functional.gelu,
    ):
        super(TransformerAE, self).__init__()

        self.transformer_encoder_layer_1 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=in_dim,
            activation=activation,
            dim_feedforward=h_dim,
            nhead=n_heads,
        )

        self.transformer_encoder_layer_2 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=256,
            activation=activation,
            dim_feedforward=256,
            nhead=n_heads,
        )
        self.transformer_encoder_layer_3 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=128,
            activation=activation,
            dim_feedforward=128,
            nhead=n_heads,
        )

        self.encoder_layer_1 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(in_dim, 256),
            torch.nn.LeakyReLU(),
        )

        self.encoder_layer_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
        )

        self.encoder_layer_3 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(128, latent_size),
            torch.nn.LeakyReLU(),
        )

        self.decoder_layer_3 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(latent_size, 128),
            torch.nn.LeakyReLU(),
        )
        self.decoder_layer_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(), torch.nn.Linear(128, 256), torch.nn.LeakyReLU()
        )
        self.decoder_layer_1 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(256, in_dim),
            torch.nn.LeakyReLU(),
        )

        self.transformer_decoder_layer_3 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=128,
            activation=activation,
            dim_feedforward=128,
            nhead=n_heads,
        )

        self.transformer_decoder_layer_2 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=256,
            activation=activation,
            dim_feedforward=256,
            nhead=n_heads,
        )

        self.transformer_decoder_layer_1 = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            dim_feedforward=h_dim,
            activation=activation,
            nhead=n_heads,
        )

    def encoder(self, x: torch.Tensor):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.transformer_encoder_layer_1(x)
        z = self.encoder_layer_1(z)
        z = self.transformer_encoder_layer_2(z)
        z = self.encoder_layer_2(z)
        z = self.transformer_encoder_layer_3(z)
        z = self.encoder_layer_3(z)

        return z

    def decoder(self, z: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.decoder_layer_3(z)
        x = self.transformer_decoder_layer_3(x)
        x = self.decoder_layer_2(x)
        x = self.transformer_decoder_layer_2(x)
        x = self.decoder_layer_1(x)
        x = self.transformer_decoder_layer_1(x)
        return x

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z, z, z, z, z


class FlexibleTransformer(nn.Module):
    """Flexible transformer model that can integrate with any VAE model.

    This model is designed to be used with any VAE model, processing both
    VAE latent vectors and raw EFP features. It uses the transformer utilities
    from transformer_utils.py to build a flexible and modular architecture.

    The model can be used in different modes:
    1. VAE-only mode: Only process VAE latent vectors
    2. EFP-only mode: Only process EFP features
    3. Combined mode: Process both VAE latent vectors and EFP features

    Args:
        latent_dim: Dimension of the VAE latent space.
        n_efp_features: Number of raw EFP features (e.g., 140 or 531).
        efp_embedding_dim: Dimension of the EFP embeddings after compression.
        output_dim: Dimension of the output space.
        d_model: Dimension of the transformer model.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Dimension of the feed-forward network.
        dropout: Dropout probability.
        activation: Activation function.
        norm_first: Whether to apply normalization before or after attention and feed-forward.
        use_class_attention: Whether to use class attention pooling for the output.
        max_jets: Maximum number of jets per event.
        output_activation: Activation function for the output layer.
        efp_config: Optional configuration dict for EFPEmbedding layer.
    """

    def __init__(
        self,
        latent_dim: int,
        n_efp_features: int,
        efp_embedding_dim: int,
        output_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: callable = F.gelu,
        norm_first: bool = True,
        use_class_attention: bool = True,
        max_jets: int = 3,
        output_activation: callable = None,
        efp_config: dict = None,
    ):
        super().__init__()

        # Create the transformer
        self.transformer = transformer_utils.HyperTransformer(
            latent_dim=latent_dim,
            n_efp_features=n_efp_features,
            efp_embedding_dim=efp_embedding_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            use_class_attention=use_class_attention,
            max_jets=max_jets,
            efp_config=efp_config,
        )

        # Output projection
        if use_class_attention:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            # If not using class attention, we need to pool the sequence
            self.output_projection = nn.Sequential(
                nn.Linear(d_model * (1 + max_jets), d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.GELU() if activation == F.gelu else nn.ReLU(),
                nn.Linear(d_model, output_dim),
            )

        self.use_class_attention = use_class_attention
        self.output_activation = output_activation

    def forward(
        self,
        latent_z: torch.Tensor = None,
        efp_features: torch.Tensor = None,
        jet_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            latent_z: Optional VAE latent vector of shape (batch_size, latent_dim)
            efp_features: Optional raw EFP features of shape (batch_size, n_jets, n_efp_features)
            jet_mask: Optional mask tensor of shape (batch_size, n_jets)

        Returns:
            output: Output tensor of shape (batch_size, output_dim)
        """
        # Check inputs
        if latent_z is None and efp_features is None:
            raise ValueError(
                "At least one of latent_z or efp_features must be provided"
            )

        # Process inputs through transformer
        if latent_z is not None and efp_features is not None:
            # Combined mode
            transformer_output = self.transformer(
                latent_z=latent_z,
                efp_features=efp_features,
                jet_mask=jet_mask,
            )
        elif latent_z is not None:
            # VAE-only mode
            transformer_output = self.transformer.encode_latent(latent_z=latent_z)
        else:
            # EFP-only mode
            transformer_output = self.transformer.encode_efp(
                efp_features=efp_features,
                jet_mask=jet_mask,
            )

        # Project to output space
        if self.use_class_attention:
            # If using class attention, transformer_output is already pooled
            output = self.output_projection(transformer_output)
        else:
            # If not using class attention, we need to flatten and project
            batch_size = transformer_output.size(0)
            output = self.output_projection(transformer_output.reshape(batch_size, -1))

        # Apply output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


class VAEWithTransformer(nn.Module):
    """VAE model with integrated transformer.

    This model combines a VAE with a transformer, allowing for flexible
    integration of VAE latent vectors and raw EFP features. The VAE can be
    any VAE model, and the transformer is built using the transformer utilities
    from transformer_utils.py.

    Args:
        vae_model: VAE model to use.
        n_efp_features: Number of raw EFP features (e.g., 140 or 531).
        efp_embedding_dim: Dimension of the EFP embeddings after compression.
        output_dim: Dimension of the output space.
        transformer_config: Configuration for the transformer.
        efp_config: Optional configuration dict for EFPEmbedding layer.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        n_efp_features: int,
        efp_embedding_dim: int,
        output_dim: int,
        transformer_config: dict = None,
        efp_config: dict = None,
    ):
        super().__init__()

        # Store the VAE model
        self.vae = vae_model

        # Get the latent dimension from the VAE
        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = self.vae.latent_dim
        elif hasattr(self.vae, "z_dim"):
            latent_dim = self.vae.z_dim
        elif hasattr(self.vae, "z_size"):
            latent_dim = self.vae.z_size
        else:
            raise ValueError(
                "VAE model must have 'latent_dim', 'z_dim', or 'z_size' attribute"
            )

        # Default transformer configuration
        default_config = {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "dropout": 0.1,
            "activation": F.gelu,
            "norm_first": True,
            "use_class_attention": True,
            "max_jets": 3,
            "output_activation": None,
        }

        # Update with provided configuration
        if transformer_config is not None:
            default_config.update(transformer_config)

        # Create the transformer
        self.transformer = FlexibleTransformer(
            latent_dim=latent_dim,
            n_efp_features=n_efp_features,
            efp_embedding_dim=efp_embedding_dim,
            output_dim=output_dim,
            efp_config=efp_config,
            **default_config,
        )

    def encode(self, x):
        """Encode input through the VAE encoder.

        Args:
            x: Input tensor

        Returns:
            z: Latent vector
            mu: Mean of the latent distribution (if VAE)
            logvar: Log variance of the latent distribution (if VAE)
        """
        if hasattr(self.vae, "encode"):
            return self.vae.encode(x)
        else:
            raise ValueError("VAE model must have an 'encode' method")

    def decode(self, z):
        """Decode latent vector through the VAE decoder.

        Args:
            z: Latent vector

        Returns:
            x_recon: Reconstructed input
        """
        if hasattr(self.vae, "decode"):
            return self.vae.decode(z)
        else:
            raise ValueError("VAE model must have a 'decode' method")

    def forward(
        self,
        x: torch.Tensor,
        efp_features: torch.Tensor = None,
        jet_mask: torch.Tensor = None,
        return_all: bool = False,
    ):
        """Forward pass.

        Args:
            x: Input tensor
            efp_features: Optional EFP features
            jet_mask: Optional mask tensor for jets
            return_all: Whether to return all outputs (reconstruction, transformer output, mu, logvar)

        Returns:
            If return_all=False:
                transformer_output: Output of the transformer
            If return_all=True:
                reconstruction: Reconstructed input
                transformer_output: Output of the transformer
                mu: Mean of the latent distribution (if VAE)
                logvar: Log variance of the latent distribution (if VAE)
        """
        # VAE forward pass
        vae_output = self.encode(x)

        # Handle different VAE output formats
        if isinstance(vae_output, tuple):
            if len(vae_output) == 3:
                z, mu, logvar = vae_output
            elif len(vae_output) == 2:
                z, mu = vae_output
                logvar = None
            else:
                z = vae_output[0]
                mu = logvar = None
        else:
            z = vae_output
            mu = logvar = None

        reconstruction = self.decode(z)

        # Transformer forward pass (transformer handles EFP embedding internally)
        transformer_output = self.transformer(
            latent_z=z,
            efp_features=efp_features,
            jet_mask=jet_mask,
        )

        if return_all:
            return reconstruction, transformer_output, mu, logvar
        else:
            return transformer_output

    def transform(
        self,
        x: torch.Tensor,
        efp_features: torch.Tensor = None,
        jet_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Transform input through the VAE encoder and transformer.

        This is a convenience method for inference.

        Args:
            x: Input tensor
            efp_features: Optional EFP features
            jet_mask: Optional mask tensor for jets

        Returns:
            transformer_output: Output of the transformer
        """
        # VAE encoding
        with torch.no_grad():
            vae_output = self.encode(x)
            if isinstance(vae_output, tuple):
                z = vae_output[0]
            else:
                z = vae_output

        # Transformer forward pass (transformer handles EFP embedding internally)
        return self.transformer(
            latent_z=z,
            efp_features=efp_features,
            jet_mask=jet_mask,
        )


# Utility functions for transformer configuration and creation
def get_activation_function(activation_str: str):
    """Map activation function string to callable."""
    activation_map = {
        "gelu": F.gelu,
        "relu": F.relu,
        "swish": F.silu,  # swish is an alias for silu
        "silu": F.silu,
        "leaky_relu": F.leaky_relu,
        "elu": F.elu,
        "tanh": torch.tanh,
    }

    if activation_str not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_str}")

    return activation_map[activation_str]


def get_output_activation_function(activation_str: str):
    """Map output activation function string to callable."""
    if activation_str is None or activation_str == "none":
        return None

    activation_map = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "softmax": lambda x: F.softmax(x, dim=-1),
        "log_softmax": lambda x: F.log_softmax(x, dim=-1),
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
    }

    if activation_str not in activation_map:
        raise ValueError(f"Unsupported output activation function: {activation_str}")

    return activation_map[activation_str]


def create_transformer_config_from_config(config) -> dict:
    """Create transformer configuration dictionary from BEAD Config object.

    This utility function extracts transformer-related parameters from the main
    BEAD configuration and creates a dictionary suitable for initializing
    FlexibleTransformer or VAEWithTransformer models.

    Args:
        config: BEAD Config object containing transformer parameters

    Returns:
        dict: Configuration dictionary for transformer initialization

    Example:
        >>> from bead.src.utils.ggl import Config
        >>> config = Config(...)  # Your config with transformer settings
        >>> transformer_config = create_transformer_config_from_config(config)
        >>> flexible_transformer = FlexibleTransformer(
        ...     latent_dim=20,
        ...     efp_embedding_dim=config.efp_embedding_dim,
        ...     output_dim=config.transformer_output_dim,
        ...     **transformer_config
        ... )
    """
    # Map string activation names to functions
    activation_map = {
        "gelu": F.gelu,
        "relu": F.relu,
        "swish": lambda x: x * torch.sigmoid(x),
        "silu": F.silu,  # SiLU is the same as Swish
    }

    # Map string output activation names to functions
    output_activation_map = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "softmax": lambda x: F.softmax(x, dim=-1),
        "log_softmax": lambda x: F.log_softmax(x, dim=-1),
        None: None,
        "none": None,
    }

    # Extract transformer configuration
    transformer_config = {
        "d_model": config.transformer_d_model,
        "n_heads": config.transformer_n_heads,
        "n_layers": config.transformer_n_layers,
        "d_ff": config.transformer_d_ff,
        "dropout": config.transformer_dropout,
        "activation": activation_map.get(config.transformer_activation.lower(), F.gelu),
        "norm_first": config.transformer_norm_first,
        "use_class_attention": config.transformer_use_class_attention,
        "max_jets": config.transformer_max_jets,
        "output_activation": output_activation_map.get(
            config.transformer_output_activation, None
        ),
    }

    return transformer_config


def create_flexible_transformer_from_config(
    config, latent_dim: int
) -> FlexibleTransformer:
    """Create FlexibleTransformer model from BEAD Config object.

    Args:
        config: BEAD Config object containing transformer parameters
        latent_dim: Dimension of the VAE latent space

    Returns:
        FlexibleTransformer: Configured transformer model

    Example:
        >>> from bead.src.utils.ggl import Config
        >>> config = Config(...)  # Your config with transformer settings
        >>> transformer = create_flexible_transformer_from_config(config, latent_dim=20)
    """
    transformer_config = create_transformer_config_from_config(config)

    return FlexibleTransformer(
        latent_dim=latent_dim,
        n_efp_features=config.efp_n_features,
        efp_embedding_dim=config.efp_embedding_dim,
        output_dim=config.transformer_output_dim,
        efp_config=getattr(config, "efp_config", None),
        **transformer_config,
    )


def create_vae_with_transformer_from_config(
    config, vae_model: nn.Module
) -> VAEWithTransformer:
    """Create VAEWithTransformer model from BEAD Config object.

    Args:
        config: BEAD Config object containing transformer parameters
        vae_model: Pre-initialized VAE model to integrate with transformer

    Returns:
        VAEWithTransformer: VAE model with integrated transformer

    Example:
        >>> from bead.src.utils.ggl import Config
        >>> from bead.src.models.models import ConvVAE
        >>> config = Config(...)  # Your config with transformer settings
        >>> vae = ConvVAE(in_shape=(1, 28, 28), z_dim=20)
        >>> vae_transformer = create_vae_with_transformer_from_config(config, vae)
    """
    transformer_config = create_transformer_config_from_config(config)

    return VAEWithTransformer(
        vae_model=vae_model,
        n_efp_features=config.efp_n_features,
        efp_embedding_dim=config.efp_embedding_dim,
        output_dim=config.transformer_output_dim,
        transformer_config=transformer_config,
        efp_config=getattr(config, "efp_config", None),
    )
