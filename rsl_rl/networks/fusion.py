
from __future__ import annotations

import torch
import torch.nn as nn
import math

from rsl_rl.utils import resolve_nn_activation


class BaseFusion(nn.Module):
    """Abstract base class for multi-modal fusion layers."""

    def __init__(self, modality_dims: dict[str, int], output_dim: int):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.modality_names = list(modality_dims.keys())

    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality features into single representation."""
        raise NotImplementedError


class ConcatFusion(BaseFusion):
    """Simple concatenation-based fusion following RSL-RL patterns."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int,
        activation: str = "elu",
        normalization: bool = True,
    ):
        super().__init__(modality_dims, output_dim)

        # Calculate total input dimension
        self.total_input_dim = sum(modality_dims.values())

        # Build projection layer if needed
        if self.total_input_dim != output_dim:
            layers = []
            layers.append(nn.Linear(self.total_input_dim, output_dim))

            if normalization:
                layers.append(nn.LayerNorm(output_dim))

            layers.append(resolve_nn_activation(activation))

            self.projection = nn.Sequential(*layers)
        else:
            self.projection = nn.Identity()

    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate and optionally project modality features."""
        # Validate all expected modalities are present
        missing_modalities = set(self.modality_names) - set(modality_features.keys())
        if missing_modalities:
            raise ValueError(f"Missing modalities: {missing_modalities}")

        # Concatenate features in consistent order
        feature_list = []
        for modality_name in self.modality_names:
            feature = modality_features[modality_name]
            # Ensure 2D tensor (batch_size, feature_dim)
            if len(feature.shape) == 1:
                feature = feature.unsqueeze(0)
            elif len(feature.shape) > 2:
                feature = feature.view(feature.size(0), -1)
            feature_list.append(feature)

        # Concatenate along feature dimension
        concatenated = torch.cat(feature_list, dim=-1)

        # Project to output dimension
        return self.projection(concatenated)


class AttentionFusion(BaseFusion):
    """Cross-attention based fusion between modalities."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int,
        attention_heads: int = 4,
        attention_dim: int = None,
        activation: str = "elu",
        dropout: float = 0.1,
    ):
        super().__init__(modality_dims, output_dim)

        self.attention_heads = attention_heads
        self.attention_dim = attention_dim or output_dim

        # Ensure attention dimension is divisible by number of heads
        if self.attention_dim % attention_heads != 0:
            raise ValueError(f"attention_dim ({self.attention_dim}) must be divisible by attention_heads ({attention_heads})")

        self.head_dim = self.attention_dim // attention_heads

        # Project each modality to common attention space
        self.modality_projections = nn.ModuleDict()
        for modality_name, input_dim in modality_dims.items():
            self.modality_projections[modality_name] = nn.Linear(input_dim, self.attention_dim)

        # Multi-head attention components
        self.query_projection = nn.Linear(self.attention_dim, self.attention_dim)
        self.key_projection = nn.Linear(self.attention_dim, self.attention_dim)
        self.value_projection = nn.Linear(self.attention_dim, self.attention_dim)

        # Output projection
        self.output_projection = nn.Linear(self.attention_dim, output_dim)

        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(self.attention_dim)
        self.dropout = nn.Dropout(dropout)

        # Final activation
        self.activation = resolve_nn_activation(activation)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights following standard practice."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply cross-attention fusion between modalities."""
        batch_size = next(iter(modality_features.values())).size(0)

        # Project all modalities to attention space
        projected_features = {}
        for modality_name, features in modality_features.items():
            if modality_name in self.modality_projections:
                # Ensure 2D tensor
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                projected_features[modality_name] = self.modality_projections[modality_name](features)

        # Stack projected features for attention computation
        # Shape: (batch_size, num_modalities, attention_dim)
        feature_stack = torch.stack(list(projected_features.values()), dim=1)

        # Apply layer normalization
        feature_stack = self.layer_norm(feature_stack)

        # Compute multi-head attention
        attended_features = self._multi_head_attention(feature_stack)

        # Global pooling (mean across modalities)
        pooled_features = attended_features.mean(dim=1)  # (batch_size, attention_dim)

        # Final projection to output dimension
        output = self.output_projection(pooled_features)
        output = self.activation(output)

        return output

    def _multi_head_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention to modality features."""
        batch_size, num_modalities, _ = x.shape

        # Compute Q, K, V
        Q = self.query_projection(x)  # (batch_size, num_modalities, attention_dim)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_modalities, self.attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_modalities, self.attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_modalities, self.attention_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, attention_heads, num_modalities, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # Shape: (batch_size, attention_heads, num_modalities, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_modalities, self.attention_dim
        )

        return attended


class GatedFusion(BaseFusion):
    """Gated fusion mechanism for selective modality combination."""

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int,
        activation: str = "elu",
        normalization: bool = True,
    ):
        super().__init__(modality_dims, output_dim)

        # Project each modality to output dimension
        self.modality_projections = nn.ModuleDict()
        for modality_name, input_dim in modality_dims.items():
            layers = [nn.Linear(input_dim, output_dim)]

            if normalization:
                layers.append(nn.LayerNorm(output_dim))

            layers.append(resolve_nn_activation(activation))

            self.modality_projections[modality_name] = nn.Sequential(*layers)

        # Gating mechanism
        total_input_dim = sum(modality_dims.values())
        self.gate_network = nn.Sequential(
            nn.Linear(total_input_dim, len(modality_dims)),
            nn.Sigmoid()  # Gate weights between 0 and 1
        )

    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply gated fusion to modality features."""
        # Project each modality to output space
        projected_features = []
        raw_features = []

        for modality_name in self.modality_names:
            if modality_name in modality_features:
                raw_feature = modality_features[modality_name]
                # Ensure 2D tensor
                if len(raw_feature.shape) > 2:
                    raw_feature = raw_feature.view(raw_feature.size(0), -1)

                projected = self.modality_projections[modality_name](raw_feature)
                projected_features.append(projected)
                raw_features.append(raw_feature)

        # Stack projected features
        feature_stack = torch.stack(projected_features, dim=1)  # (batch_size, num_modalities, output_dim)

        # Compute gate weights
        concatenated_raw = torch.cat(raw_features, dim=-1)
        gate_weights = self.gate_network(concatenated_raw)  # (batch_size, num_modalities)
        gate_weights = gate_weights.unsqueeze(-1)  # (batch_size, num_modalities, 1)

        # Apply gating
        gated_features = feature_stack * gate_weights

        # Sum across modalities
        output = gated_features.sum(dim=1)  # (batch_size, output_dim)

        return output