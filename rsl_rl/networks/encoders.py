
from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import resolve_nn_activation


class BaseEncoder(nn.Module):
    """Abstract base class for all encoders following RSL-RL patterns."""

    def __init__(self, input_shape: tuple | int, latent_dim: int):
        super().__init__()
        self.input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """Return the dimension of encoded output."""
        return self.latent_dim


class ProprioceptionEncoder(BaseEncoder):
    """MLP encoder for proprioceptive observations following RSL-RL MLP patterns."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [128, 128],
        activation: str = "elu",
        normalization: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(input_dim, latent_dim)

        # Build MLP layers following RSL-RL MLP pattern
        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add normalization (except for last layer)
            if normalization and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))

            # Add activation (except for last layer)
            if i < len(dims) - 2:
                layers.append(resolve_nn_activation(activation))

            # Add dropout (except for last layer)
            if dropout > 0.0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode proprioceptive input to latent representation."""
        # Flatten if needed (maintaining batch dimension)
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        return self.mlp(x)


class VisionEncoder(BaseEncoder):
    """CNN encoder for visual observations (RGB, depth, etc.)."""

    def __init__(
        self,
        input_shape: tuple,  # (C, H, W)
        latent_dim: int,
        channels: list[int] = [32, 64, 128],
        kernel_sizes: list[int] = [8, 4, 3],
        strides: list[int] = [4, 2, 1],
        activation: str = "relu",
        normalization: str = "batch_norm",  # "batch_norm", "layer_norm", "none"
    ):
        super().__init__(input_shape, latent_dim)

        if len(input_shape) != 3:
            raise ValueError(f"Vision encoder expects 3D input (C,H,W), got {input_shape}")

        # Validate parameter lengths
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes, and strides must have same length")

        # Build convolutional layers
        conv_layers = []
        in_channels = input_shape[0]

        for i, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            # Convolutional layer
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

            # Normalization
            if normalization == "batch_norm":
                conv_layers.append(nn.BatchNorm2d(out_channels))
            elif normalization == "layer_norm":
                # Note: LayerNorm for conv layers needs careful handling
                conv_layers.append(nn.GroupNorm(1, out_channels))  # Equivalent to LayerNorm for conv

            # Activation
            conv_layers.append(resolve_nn_activation(activation))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate conv output size
        conv_output_size = self._calculate_conv_output_size(input_shape)

        # Final linear layer to latent dimension
        self.fc = nn.Linear(conv_output_size, latent_dim)

        # Initialize weights following standard practice
        self._init_weights()

    def _calculate_conv_output_size(self, input_shape: tuple) -> int:
        """Calculate the output size after conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.numel()  # Total number of elements

    def _init_weights(self):
        """Initialize weights following RSL-RL patterns."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visual input to latent representation."""
        # Validate input shape
        if len(x.shape) != 4:  # (B, C, H, W)
            raise ValueError(f"Vision encoder expects 4D input (B,C,H,W), got {x.shape}")

        # Convolutional feature extraction
        features = self.conv_layers(x)

        # Flatten and project to latent space
        features = features.view(features.size(0), -1)  # (B, conv_output_size)
        latent = self.fc(features)

        return latent


class PrivilegedEncoder(BaseEncoder):
    """Flexible encoder for privileged information (terrain maps, object states, etc.)."""

    def __init__(
        self,
        input_shape: tuple | int,
        latent_dim: int,
        encoder_type: str = "mlp",  # "mlp", "cnn", "hybrid"
        **kwargs
    ):
        super().__init__(input_shape, latent_dim)

        self.encoder_type = encoder_type

        if encoder_type == "mlp":
            # Treat as 1D privileged information
            input_dim = input_shape if isinstance(input_shape, int) else reduce(lambda x, y: x * y, input_shape)
            self.encoder = ProprioceptionEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                **kwargs
            )
        elif encoder_type == "cnn":
            # Treat as 2D/3D privileged information (e.g., terrain heightmaps)
            if isinstance(input_shape, int):
                raise ValueError("CNN encoder requires tuple input_shape (C,H,W)")
            self.encoder = VisionEncoder(
                input_shape=input_shape,
                latent_dim=latent_dim,
                **kwargs
            )
        elif encoder_type == "hybrid":
            # For complex privileged information requiring both CNN and MLP processing
            # This could be extended for specific use cases
            raise NotImplementedError("Hybrid encoder not yet implemented")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode privileged information to latent representation."""
        return self.encoder(x)