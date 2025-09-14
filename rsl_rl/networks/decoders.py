
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class BaseDecoder(nn.Module):
    """Abstract base class for all decoders following RSL-RL patterns."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        raise NotImplementedError


class ActionDecoder(BaseDecoder):
    """Decoder for action generation from latent representations."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        output_activation: str | None = None,
        normalization: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(latent_dim, num_actions)

        # Build MLP layers following RSL-RL MLP pattern
        layers = []
        dims = [latent_dim] + hidden_dims + [num_actions]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add normalization (except for last layer)
            if normalization and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))

            # Add activation
            if i < len(dims) - 2:
                # Hidden layer activation
                layers.append(resolve_nn_activation(activation))
            elif output_activation is not None:
                # Output activation (if specified)
                layers.append(resolve_nn_activation(output_activation))

            # Add dropout (except for last layer)
            if dropout > 0.0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights following RSL-RL patterns
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following RSL-RL patterns."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to action mean."""
        return self.mlp(latent)


class ValueDecoder(BaseDecoder):
    """Decoder for value estimation from latent representations."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        normalization: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(latent_dim, 1)  # Value is always 1D output

        # Build MLP layers following RSL-RL MLP pattern
        layers = []
        dims = [latent_dim] + hidden_dims + [1]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add normalization (except for last layer)
            if normalization and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))

            # Add activation (except for last layer - value output is linear)
            if i < len(dims) - 2:
                layers.append(resolve_nn_activation(activation))

            # Add dropout (except for last layer)
            if dropout > 0.0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights following RSL-RL patterns
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following RSL-RL patterns."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to value estimate."""
        return self.mlp(latent)


class PolicyDecoder(BaseDecoder):
    """Combined decoder for both actions and values (for shared encoder architectures)."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        # Action decoder parameters
        action_hidden_dims: list[int] = [256, 128],
        action_activation: str = "elu",
        action_output_activation: str | None = None,
        # Value decoder parameters
        value_hidden_dims: list[int] = [256, 128],
        value_activation: str = "elu",
        # Shared parameters
        normalization: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(latent_dim, num_actions + 1)  # Actions + value

        # Separate action and value decoders
        self.action_decoder = ActionDecoder(
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dims=action_hidden_dims,
            activation=action_activation,
            output_activation=action_output_activation,
            normalization=normalization,
            dropout=dropout,
        )

        self.value_decoder = ValueDecoder(
            latent_dim=latent_dim,
            hidden_dims=value_hidden_dims,
            activation=value_activation,
            normalization=normalization,
            dropout=dropout,
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent representation to both actions and value."""
        actions = self.action_decoder(latent)
        value = self.value_decoder(latent)
        return actions, value

    def forward_actions(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to actions only."""
        return self.action_decoder(latent)

    def forward_value(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to value only."""
        return self.value_decoder(latent)