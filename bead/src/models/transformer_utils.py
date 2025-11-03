"""
Transformer utilities for flexible VAE integration.

This module provides reusable transformer components that can be used to build
flexible transformer architectures for VAE integration. The components are designed
to be modular and reusable, allowing for easy experimentation with different
transformer architectures.

Key components:
- MultiHeadedAttentionBlock: Multi-head attention with optional dropout and layer norm
- TransformerEncoderLayer: Transformer encoder layer with self-attention and feed-forward
- TransformerEncoder: Stack of transformer encoder layers
- ClassAttention: Attention-based pooling mechanism
- LatentProjection: Projects VAE latent vectors to transformer token space
- EFPProjection: Projects EFP embeddings to transformer token space
- HyperTransformer: Transformer that integrates VAE latent vectors and EFP embeddings
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EFPEmbedding


class MultiHeadedAttentionBlock(nn.Module):
    """Multi-head attention block with optional dropout and layer normalization.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        dropout: Dropout probability
        causal: Whether to use causal masking
        do_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
        do_layer_norm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.causal = causal
        self.do_layer_norm = do_layer_norm

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        if do_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply layer normalization if enabled
        if self.do_layer_norm:
            x_norm = self.layer_norm(x)
        else:
            x_norm = x

        # Create attention mask
        attn_mask = None
        key_padding_mask = None
        if mask is not None:
            # Use key_padding_mask instead of attn_mask for better compatibility
            # PyTorch expects key_padding_mask: (batch_size, seq_len) where True = ignore
            key_padding_mask = ~mask  # Invert mask: True positions will be ignored

        # Apply causal masking if enabled
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(x.size(1), x.size(1), device=x.device),
                diagonal=1,
            ).bool()
            if attn_mask is None:
                attn_mask = causal_mask.unsqueeze(0).expand(x.size(0), -1, -1)
            else:
                attn_mask = attn_mask | causal_mask.unsqueeze(0)

        # Apply multi-head attention
        output, attn_weights = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_attention,
        )

        if return_attention:
            return output, attn_weights
        else:
            return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward network.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to apply normalization before or after attention and feed-forward
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.gelu,
        norm_first: bool = True,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadedAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            do_layer_norm=False,  # We handle normalization separately
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.GELU() if activation == F.gelu else nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Whether to apply normalization before or after attention and feed-forward
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention block
        if self.norm_first:
            # Normformer style: norm -> attn -> residual
            attn_output = self.self_attn(self.norm1(x), mask)
            x = x + attn_output

            # Feed-forward block
            ff_output = self.ff(self.norm2(x))
            x = x + ff_output
        else:
            # Standard transformer: attn -> residual -> norm
            attn_output = self.self_attn(x, mask)
            x = self.norm1(x + attn_output)

            # Feed-forward block
            ff_output = self.ff(x)
            x = self.norm2(x + ff_output)

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to apply normalization before or after attention and feed-forward
        norm_output: Whether to apply normalization to the output
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.gelu,
        norm_first: bool = True,
        norm_output: bool = True,
    ):
        super().__init__()

        # Stack of transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(n_layers)
            ]
        )

        # Output normalization
        self.norm_output = norm_output
        if norm_output:
            self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pass through each transformer encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        # Apply output normalization if enabled
        if self.norm_output:
            x = self.norm(x)

        return x


class ClassAttention(nn.Module):
    """Attention-based pooling mechanism.

    This module converts a sequence of tokens to a single vector using attention.
    It can be used to pool the output of a transformer encoder.

    Args:
        d_model: Dimension of the model
        n_layers: Number of class attention layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Class attention layers
        self.layers = nn.ModuleList(
            [
                MultiHeadedAttentionBlock(
                    d_model=d_model,
                    n_heads=1,
                    dropout=dropout,
                    do_layer_norm=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, d_model)
        """
        batch_size = x.size(0)

        # Expand class token to batch size
        cls_token = self.class_token.expand(batch_size, -1, -1)

        # Pass through each class attention layer
        # Note: Class token doesn't need masking since it's always valid
        for layer in self.layers:
            cls_token = layer(cls_token, mask=None)

        # Apply output normalization
        cls_token = self.norm(cls_token)

        # Remove sequence dimension
        return cls_token.squeeze(1)


class TransformerVectorEncoder(nn.Module):
    """Transformer encoder followed by class attention pooling.

    This module converts a sequence of tokens to a single vector using a transformer
    encoder followed by class attention pooling.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to apply normalization before or after attention and feed-forward
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.gelu,
        norm_first: bool = True,
    ):
        super().__init__()

        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            norm_output=True,
        )

        # Class attention pooling
        self.class_attention = ClassAttention(
            d_model=d_model,
            n_layers=1,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, d_model)
        """
        # Pass through transformer encoder
        x = self.encoder(x, mask)

        # Apply class attention pooling
        return self.class_attention(x, mask)


class LatentProjection(nn.Module):
    """Projects VAE latent vectors to transformer token space.

    Args:
        latent_dim: Dimension of the VAE latent space
        d_model: Dimension of the transformer model
        dropout: Dropout probability
        use_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        # Linear projection
        self.projection = nn.Linear(latent_dim, d_model)

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            latent_z: VAE latent vector of shape (batch_size, latent_dim)

        Returns:
            Projected token of shape (batch_size, 1, d_model)
        """
        # Add sequence dimension
        x = latent_z.unsqueeze(1)

        # Linear projection
        x = self.projection(x)

        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Apply dropout
        x = self.dropout(x)

        return x


class EFPProjection(nn.Module):
    """Projects EFP embeddings to transformer token space.

    Args:
        efp_embedding_dim: Dimension of the EFP embeddings
        d_model: Dimension of the transformer model
        dropout: Dropout probability
        use_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        efp_embedding_dim: int,
        d_model: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        # Linear projection
        self.projection = nn.Linear(efp_embedding_dim, d_model)

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, efp_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            efp_embeddings: EFP embeddings of shape (batch_size, n_jets, efp_embedding_dim)

        Returns:
            Projected tokens of shape (batch_size, n_jets, d_model)
        """
        # Linear projection
        x = self.projection(efp_embeddings)

        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x = self.layer_norm(x)

        # Apply dropout
        x = self.dropout(x)

        return x


def construct_transformer_input(
    latent_tokens: torch.Tensor,
    efp_tokens: torch.Tensor,
    jet_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct transformer input by combining latent and EFP tokens.

    Args:
        latent_tokens: VAE latent tokens of shape (batch_size, 1, d_model)
        efp_tokens: EFP tokens of shape (batch_size, n_jets, d_model)
        jet_mask: Jet mask of shape (batch_size, n_jets)

    Returns:
        tokens: Combined tokens of shape (batch_size, 1 + n_jets, d_model)
        attention_mask: Attention mask of shape (batch_size, 1 + n_jets)
    """
    # Concatenate latent token (always valid) with EFP tokens
    tokens = torch.cat([latent_tokens, efp_tokens], dim=1)

    # Create attention mask: latent token always attended, EFP tokens per jet_mask
    if jet_mask is not None:
        latent_mask = torch.ones(
            jet_mask.size(0), 1, device=jet_mask.device, dtype=torch.bool
        )
        attention_mask = torch.cat([latent_mask, jet_mask], dim=1)
    else:
        # If no jet mask provided, all tokens are valid
        attention_mask = torch.ones(
            tokens.size(0), tokens.size(1), device=tokens.device, dtype=torch.bool
        )

    return tokens, attention_mask


class HyperTransformer(nn.Module):
    """HyperTransformer for multi-modal input processing.

    This transformer processes both VAE latent vectors and raw EFP features,
    combining them into a unified representation for downstream tasks.
    The EFP features are first processed through an EFPEmbedding layer for
    compression, sparsification, and regularization before transformer processing.

    Args:
        latent_dim: Dimension of the VAE latent vector
        n_efp_features: Number of raw EFP features (e.g., 140 or 531)
        efp_embedding_dim: Dimension of the EFP embeddings after compression (e.g., 64)
        d_model: Dimension of the transformer model
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Dimension of the feed-forward network
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to apply normalization before or after attention and feed-forward
        use_class_attention: Whether to use class attention pooling for the output
        max_jets: Maximum number of jets per event
        efp_config: Optional configuration dict for EFPEmbedding layer
    """

    def __init__(
        self,
        latent_dim: int,
        n_efp_features: int,
        efp_embedding_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.gelu,
        norm_first: bool = True,
        use_class_attention: bool = True,
        max_jets: int = 3,
        efp_config: Optional[dict] = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_efp_features = n_efp_features
        self.efp_embedding_dim = efp_embedding_dim
        self.d_model = d_model
        self.use_class_attention = use_class_attention
        self.max_jets = max_jets

        # EFP Embedding layer (raw EFP features -> embedded features)
        efp_embedding_config = {
            "gate_type": "sigmoid",
            "gate_threshold": 0.1,
            "dropout_rate": dropout,
            "use_layer_norm": True,
        }
        if efp_config:
            efp_embedding_config.update(efp_config)

        self.efp_embedding = EFPEmbedding(
            n_efp_features=n_efp_features,
            embedding_dim=efp_embedding_dim,
            **efp_embedding_config,
        )

        # Input projections
        self.latent_projection = LatentProjection(
            latent_dim=latent_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.efp_projection = EFPProjection(
            efp_embedding_dim=efp_embedding_dim,
            d_model=d_model,
            dropout=dropout,
        )

        # Token type embeddings (latent vs EFP)
        self.token_type_embeddings = nn.Embedding(2, d_model)

        # Positional embeddings
        self.register_buffer(
            "position_embeddings",
            self._init_sinusoidal_embeddings(max_jets + 1, d_model),
        )

        # Transformer encoder
        if use_class_attention:
            self.transformer = TransformerVectorEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
        else:
            self.transformer = TransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                norm_output=True,
            )

    def _init_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        """Initialize sinusoidal positional embeddings.

        Args:
            max_len: Maximum sequence length
            d_model: Dimension of the model

        Returns:
            Positional embeddings of shape (1, max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def encode_latent(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Encode VAE latent vector.

        Args:
            latent_z: VAE latent vector of shape (batch_size, latent_dim)

        Returns:
            Encoded latent vector of shape (batch_size, d_model) if use_class_attention=True,
            otherwise (batch_size, 1, d_model)
        """
        # Project latent vector to transformer token space
        latent_tokens = self.latent_projection(latent_z)

        # Add token type embedding (type 0 = latent)
        token_type_ids = torch.zeros(
            latent_tokens.size(0),
            latent_tokens.size(1),
            device=latent_tokens.device,
            dtype=torch.long,
        )
        latent_tokens = latent_tokens + self.token_type_embeddings(token_type_ids)

        # Add positional embedding (position 0)
        latent_tokens = latent_tokens + self.position_embeddings[:, :1, :]

        # Pass through transformer
        if self.use_class_attention:
            # If using class attention, return pooled vector
            return self.transformer(latent_tokens)
        else:
            # Otherwise, return sequence
            return self.transformer(latent_tokens)

    def encode_efp(
        self,
        efp_features: torch.Tensor,
        jet_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode EFP features.

        Args:
            efp_features: Raw EFP features of shape (batch_size, n_jets, n_efp_features)
            jet_mask: Jet mask of shape (batch_size, n_jets)

        Returns:
            Encoded EFP embeddings of shape (batch_size, d_model) if use_class_attention=True,
            otherwise (batch_size, n_jets, d_model)
        """
        # Apply EFP embedding layer first (raw features -> embedded features)
        efp_embeddings = self.efp_embedding(efp_features, jet_mask)

        # Project EFP embeddings to transformer token space
        efp_tokens = self.efp_projection(efp_embeddings)

        # Add token type embedding (type 1 = EFP)
        token_type_ids = torch.ones(
            efp_tokens.size(0),
            efp_tokens.size(1),
            device=efp_tokens.device,
            dtype=torch.long,
        )
        efp_tokens = efp_tokens + self.token_type_embeddings(token_type_ids)

        # Add positional embedding (positions 1 to n_jets)
        n_jets = efp_tokens.size(1)
        efp_tokens = efp_tokens + self.position_embeddings[:, 1 : n_jets + 1, :]

        # Pass through transformer
        if self.use_class_attention:
            # If using class attention, return pooled vector
            return self.transformer(efp_tokens, jet_mask)
        else:
            # Otherwise, return sequence
            return self.transformer(efp_tokens, jet_mask)

    def forward(
        self,
        latent_z: torch.Tensor,
        efp_features: torch.Tensor,
        jet_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            latent_z: VAE latent vector of shape (batch_size, latent_dim)
            efp_features: Raw EFP features of shape (batch_size, n_jets, n_efp_features)
            jet_mask: Jet mask of shape (batch_size, n_jets)

        Returns:
            Output tensor of shape (batch_size, d_model) if use_class_attention=True,
            otherwise (batch_size, 1 + n_jets, d_model)
        """
        # Apply EFP embedding layer first (raw features -> embedded features)
        efp_embeddings = self.efp_embedding(efp_features, jet_mask)

        # Project inputs to transformer token space
        latent_tokens = self.latent_projection(latent_z)
        efp_tokens = self.efp_projection(efp_embeddings)

        # Add token type embeddings
        latent_type_ids = torch.zeros(
            latent_tokens.size(0),
            latent_tokens.size(1),
            device=latent_tokens.device,
            dtype=torch.long,
        )
        efp_type_ids = torch.ones(
            efp_tokens.size(0),
            efp_tokens.size(1),
            device=efp_tokens.device,
            dtype=torch.long,
        )
        latent_tokens = latent_tokens + self.token_type_embeddings(latent_type_ids)
        efp_tokens = efp_tokens + self.token_type_embeddings(efp_type_ids)

        # Add positional embeddings
        n_jets = efp_tokens.size(1)
        latent_tokens = latent_tokens + self.position_embeddings[:, :1, :]
        efp_tokens = efp_tokens + self.position_embeddings[:, 1 : n_jets + 1, :]

        # Construct transformer input
        tokens, attention_mask = construct_transformer_input(
            latent_tokens=latent_tokens,
            efp_tokens=efp_tokens,
            jet_mask=jet_mask,
        )

        # Pass through transformer
        return self.transformer(tokens, attention_mask)
