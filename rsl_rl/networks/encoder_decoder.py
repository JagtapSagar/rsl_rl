
from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.networks.encoders import BaseEncoder, ProprioceptionEncoder, VisionEncoder, PrivilegedEncoder
from rsl_rl.networks.fusion import BaseFusion, ConcatFusion, AttentionFusion, GatedFusion
from rsl_rl.networks.decoders import ActionDecoder, ValueDecoder, PolicyDecoder


class EncoderDecoderNetwork(nn.Module):
    """Complete encoder-decoder network with multi-modal support following RSL-RL patterns."""

    def __init__(
        self,
        obs_spaces: dict,
        obs_groups: dict,
        num_actions: int,
        encoder_configs: dict,
        fusion_config: dict,
        decoder_configs: dict,
        shared_encoder: bool = False,
    ):
        super().__init__()

        self.obs_spaces = obs_spaces
        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.shared_encoder = shared_encoder

        # Build modality-specific encoders
        self.encoders = self._build_encoders(encoder_configs, obs_spaces, obs_groups)

        # Build fusion layer
        modality_dims = {name: encoder.get_output_dim() for name, encoder in self.encoders.items()}
        self.fusion = self._build_fusion(fusion_config, modality_dims)

        # Build decoders
        self.decoders = self._build_decoders(decoder_configs, self.fusion.output_dim, num_actions)

        # Initialize weights
        self._init_weights()

    def _build_encoders(self, encoder_configs: dict, obs_spaces: dict, obs_groups: dict) -> nn.ModuleDict:
        """Build modality-specific encoders based on configuration."""
        encoders = nn.ModuleDict()

        # Determine which observation groups to process
        required_obs_groups = set()
        for group_type in ["policy", "critic"]:
            if group_type in obs_groups:
                required_obs_groups.update(obs_groups[group_type])

        # Build encoders for configured modalities
        for modality_name, config in encoder_configs.items():
            # Find corresponding observation groups for this modality
            modality_obs_groups = self._get_modality_obs_groups(modality_name, required_obs_groups, obs_spaces)

            if not modality_obs_groups:
                print(f"Warning: No observation groups found for modality '{modality_name}', skipping...")
                continue

            # Calculate total input dimension for this modality
            total_input_dim = 0
            modality_input_shape = None

            for obs_group in modality_obs_groups:
                if obs_group in obs_spaces:
                    obs_shape = obs_spaces[obs_group].shape
                    if config["type"] == "cnn" and len(obs_shape) >= 3:
                        # For CNN, keep spatial dimensions (exclude batch dimension)
                        if modality_input_shape is None:
                            modality_input_shape = obs_shape[1:]  # Remove batch dimension
                        else:
                            # Multiple vision inputs - would need more complex handling
                            print(f"Warning: Multiple vision inputs for modality '{modality_name}' not fully supported")
                    else:
                        # For MLP, accumulate flattened dimensions
                        total_input_dim += reduce(lambda x, y: x * y, obs_shape, 1)

            # Create encoder based on type, using ALL config parameters
            encoder_type = config["type"]

            # Extract common parameters and type-specific parameters
            encoder_params = {k: v for k, v in config.items() if k != "type"}

            if encoder_type == "mlp":
                encoders[modality_name] = ProprioceptionEncoder(
                    input_dim=total_input_dim,
                    **encoder_params  # Pass all config parameters dynamically
                )
            elif encoder_type == "cnn":
                if modality_input_shape is None or len(modality_input_shape) < 3:
                    raise ValueError(f"CNN encoder for '{modality_name}' requires 3D input shape (C,H,W)")
                encoders[modality_name] = VisionEncoder(
                    input_shape=modality_input_shape,
                    **encoder_params  # Pass all config parameters dynamically
                )
            elif encoder_type in ["privileged", "hybrid"]:
                input_shape = modality_input_shape if modality_input_shape else total_input_dim
                encoders[modality_name] = PrivilegedEncoder(
                    input_shape=input_shape,
                    **encoder_params  # Pass all config parameters dynamically
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

        return encoders

    def _get_modality_obs_groups(self, modality_name: str, required_obs_groups: set, obs_spaces: dict) -> list[str]:
        """Determine which observation groups belong to a specific modality."""
        modality_obs_groups = []

        # Heuristic mapping based on modality name and observation group names
        for obs_group in required_obs_groups:
            if obs_group in obs_spaces:
                if modality_name == "proprioception":
                    # Proprioception typically includes joint states, actions history, etc.
                    if any(keyword in obs_group.lower() for keyword in
                           ["robot", "joint", "action", "history", "proprio", "state", "imu", "contact"]):
                        modality_obs_groups.append(obs_group)
                elif modality_name == "vision":
                    # Vision includes camera observations
                    if any(keyword in obs_group.lower() for keyword in
                           ["camera", "rgb", "depth", "image", "vision", "visual"]):
                        modality_obs_groups.append(obs_group)
                elif modality_name == "privileged":
                    # Privileged information includes terrain, ground truth dynamics, etc.
                    if any(keyword in obs_group.lower() for keyword in
                           ["terrain", "privileged", "ground_truth", "dynamics", "friction", "object_states"]):
                        modality_obs_groups.append(obs_group)

        return modality_obs_groups

    def _build_fusion(self, fusion_config: dict, modality_dims: dict[str, int]) -> BaseFusion:
        """Build fusion layer based on configuration."""
        fusion_type = fusion_config["type"]

        # Extract parameters, removing type
        fusion_params = {k: v for k, v in fusion_config.items() if k != "type"}

        if fusion_type == "concat":
            return ConcatFusion(
                modality_dims=modality_dims,
                **fusion_params  # Pass all config parameters dynamically
            )
        elif fusion_type == "attention":
            return AttentionFusion(
                modality_dims=modality_dims,
                **fusion_params  # Pass all config parameters dynamically
            )
        elif fusion_type == "gated":
            return GatedFusion(
                modality_dims=modality_dims,
                **fusion_params  # Pass all config parameters dynamically
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def _build_decoders(self, decoder_configs: dict, latent_dim: int, num_actions: int) -> nn.ModuleDict:
        """Build action and value decoders based on configuration."""
        decoders = nn.ModuleDict()

        # Action decoder
        if "action" in decoder_configs:
            action_config = decoder_configs["action"]
            decoders["action"] = ActionDecoder(
                latent_dim=latent_dim,
                num_actions=num_actions,
                **action_config  # Pass all config parameters dynamically
            )

        # Value decoder
        if "value" in decoder_configs:
            value_config = decoder_configs["value"]
            decoders["value"] = ValueDecoder(
                latent_dim=latent_dim,
                **value_config  # Pass all config parameters dynamically
            )

        # Combined policy decoder (if neither action nor value specified separately)
        if "action" not in decoders and "value" not in decoders:
            # Use combined decoder with default configurations
            decoders["policy"] = PolicyDecoder(
                latent_dim=latent_dim,
                num_actions=num_actions,
            )

        return decoders

    def _init_weights(self):
        """Initialize weights following RSL-RL patterns."""
        # Encoders and decoders have their own initialization
        # Fusion layer handles its own initialization
        pass

    def encode(self, obs: dict, obs_group_type: str = "policy") -> torch.Tensor:
        """Encode multi-modal observations to latent representation."""
        # Get relevant observation groups
        if obs_group_type not in self.obs_groups:
            raise ValueError(f"Unknown observation group type: {obs_group_type}")

        target_obs_groups = self.obs_groups[obs_group_type]

        # Encode each modality
        encoded_modalities = {}

        for modality_name, encoder in self.encoders.items():
            # Get observations for this modality
            modality_obs = self._get_modality_observations(obs, modality_name, target_obs_groups)

            if modality_obs is not None:
                encoded_modalities[modality_name] = encoder(modality_obs)

        # Fuse modalities
        if not encoded_modalities:
            raise ValueError(f"No observations found for group type: {obs_group_type}")

        return self.fusion(encoded_modalities)

    def _get_modality_observations(self, obs: dict, modality_name: str, target_obs_groups: list[str]) -> torch.Tensor | None:
        """Extract and concatenate observations for a specific modality."""
        modality_obs_groups = self._get_modality_obs_groups(modality_name, set(target_obs_groups), self.obs_spaces)

        if not modality_obs_groups:
            return None

        obs_list = []
        for obs_group in modality_obs_groups:
            if obs_group in obs:
                obs_data = obs[obs_group]
                # For vision modalities, preserve spatial structure
                if modality_name == "vision" and len(obs_data.shape) == 4:
                    # Assume single vision input for now
                    return obs_data
                else:
                    # For proprioception and other modalities, flatten
                    if len(obs_data.shape) > 2:
                        obs_data = obs_data.view(obs_data.size(0), -1)
                    obs_list.append(obs_data)

        if obs_list:
            return torch.cat(obs_list, dim=-1)
        else:
            return None

    def forward_actor(self, obs: dict) -> torch.Tensor:
        """Forward pass for actor (action generation)."""
        latent = self.encode(obs, "policy")

        if "action" in self.decoders:
            return self.decoders["action"](latent)
        elif "policy" in self.decoders:
            actions, _ = self.decoders["policy"](latent)
            return actions
        else:
            raise ValueError("No action decoder found")

    def forward_critic(self, obs: dict) -> torch.Tensor:
        """Forward pass for critic (value estimation)."""
        latent = self.encode(obs, "critic")

        if "value" in self.decoders:
            return self.decoders["value"](latent)
        elif "policy" in self.decoders:
            _, value = self.decoders["policy"](latent)
            return value
        else:
            raise ValueError("No value decoder found")

    def forward(self, obs: dict, obs_group_type: str = "policy") -> torch.Tensor:
        """General forward pass."""
        if obs_group_type == "policy":
            return self.forward_actor(obs)
        elif obs_group_type == "critic":
            return self.forward_critic(obs)
        else:
            raise ValueError(f"Unknown observation group type: {obs_group_type}")