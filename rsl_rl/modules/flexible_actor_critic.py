
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, EncoderDecoderNetwork


class FlexibleActorCritic(nn.Module):
    """Actor-Critic with configurable architectures (MLP or Encoder-Decoder) following RSL-RL patterns."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # Standard RSL-RL parameters (maintain order and defaults)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        # New parameters (added at end)
        actor_architecture="mlp",  # "mlp" or "encoder_decoder"
        critic_architecture="mlp",  # "mlp" or "encoder_decoder"
        shared_encoder=False,
        encoder_configs=None,
        fusion_config=None,
        decoder_configs=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "FlexibleActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )  # Follow RSL-RL warning pattern
        super().__init__()

        # Store configuration (RSL-RL pattern)
        self.obs_groups = obs_groups
        self.actor_architecture = actor_architecture
        self.critic_architecture = critic_architecture
        self.shared_encoder = shared_encoder
        self.num_actions = num_actions

        # Build networks based on architecture selection
        if actor_architecture == "mlp":
            self.actor = self._build_mlp_actor(obs, actor_hidden_dims, activation)
        elif actor_architecture == "encoder_decoder":
            self.actor = self._build_encoder_decoder_actor(
                obs, obs_groups, encoder_configs, fusion_config, decoder_configs
            )
        else:
            raise ValueError(f"Unknown actor architecture: {actor_architecture}")

        if critic_architecture == "mlp":
            self.critic = self._build_mlp_critic(obs, critic_hidden_dims, activation)
        elif critic_architecture == "encoder_decoder":
            if shared_encoder and actor_architecture == "encoder_decoder":
                # Share encoder with actor
                self.critic = self.actor  # Same network, different forward method
            else:
                # Separate encoder-decoder for critic
                self.critic = self._build_encoder_decoder_critic(
                    obs, obs_groups, encoder_configs, fusion_config, decoder_configs
                )
        else:
            raise ValueError(f"Unknown critic architecture: {critic_architecture}")

        # Action distribution setup (follow existing pattern exactly)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Follow RSL-RL observation normalization pattern exactly
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization

        if actor_architecture == "mlp":
            # Calculate actor observation dimensions (follows RSL-RL ActorCritic pattern)
            num_actor_obs = 0
            for obs_group in obs_groups["policy"]:
                assert len(obs[obs_group].shape) == 2, "MLP mode requires 1D observations."
                num_actor_obs += obs[obs_group].shape[-1]

            if actor_obs_normalization:
                self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
            else:
                self.actor_obs_normalizer = torch.nn.Identity()
        else:
            # For encoder-decoder, normalization is handled within encoders
            self.actor_obs_normalizer = torch.nn.Identity()

        if critic_architecture == "mlp":
            # Calculate critic observation dimensions
            num_critic_obs = 0
            for obs_group in obs_groups["critic"]:
                assert len(obs[obs_group].shape) == 2, "MLP mode requires 1D observations."
                num_critic_obs += obs[obs_group].shape[-1]

            if critic_obs_normalization:
                self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
            else:
                self.critic_obs_normalizer = torch.nn.Identity()
        else:
            # For encoder-decoder, normalization is handled within encoders
            self.critic_obs_normalizer = torch.nn.Identity()

        # Distribution setup (follow existing pattern)
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _build_mlp_actor(self, obs, hidden_dims, activation):
        """Build standard MLP actor (follows RSL-RL ActorCritic pattern)."""
        num_actor_obs = 0
        for obs_group in self.obs_groups["policy"]:
            num_actor_obs += obs[obs_group].shape[-1]

        return MLP(num_actor_obs, self.num_actions, hidden_dims, activation)

    def _build_mlp_critic(self, obs, hidden_dims, activation):
        """Build standard MLP critic (follows RSL-RL ActorCritic pattern)."""
        num_critic_obs = 0
        for obs_group in self.obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]

        return MLP(num_critic_obs, 1, hidden_dims, activation)

    def _build_encoder_decoder_actor(self, obs, obs_groups, encoder_configs, fusion_config, decoder_configs):
        """Build encoder-decoder actor network with only the encoders it needs."""
        if encoder_configs is None or fusion_config is None or decoder_configs is None:
            raise ValueError("encoder_configs, fusion_config, and decoder_configs required for encoder_decoder architecture")

        # Filter encoder configs to only include modalities available to the actor
        actor_encoder_configs = self._filter_encoder_configs_for_obs_group(
            encoder_configs, obs_groups["policy"], obs
        )

        return EncoderDecoderNetwork(
            obs_spaces=obs,
            obs_groups={"policy": obs_groups["policy"], "critic": obs_groups["policy"]},  # Actor uses policy groups for both
            num_actions=self.num_actions,
            encoder_configs=actor_encoder_configs,
            fusion_config=fusion_config,
            decoder_configs=decoder_configs,
        )

    def _build_encoder_decoder_critic(self, obs, obs_groups, encoder_configs, fusion_config, decoder_configs):
        """Build encoder-decoder critic network with only the encoders it needs."""
        if encoder_configs is None or fusion_config is None or decoder_configs is None:
            raise ValueError("encoder_configs, fusion_config, and decoder_configs required for encoder_decoder architecture")

        # Filter encoder configs to only include modalities available to the critic
        critic_encoder_configs = self._filter_encoder_configs_for_obs_group(
            encoder_configs, obs_groups["critic"], obs
        )

        # Ensure decoder configs include value decoder
        critic_decoder_configs = decoder_configs.copy()
        if "value" not in critic_decoder_configs and "policy" not in critic_decoder_configs:
            critic_decoder_configs["value"] = decoder_configs.get("critic", {"hidden_dims": [256, 128]})

        return EncoderDecoderNetwork(
            obs_spaces=obs,
            obs_groups={"policy": obs_groups["critic"], "critic": obs_groups["critic"]},  # Critic uses critic groups for both
            num_actions=1,  # Critic outputs single value
            encoder_configs=critic_encoder_configs,
            fusion_config=fusion_config,
            decoder_configs=critic_decoder_configs,
        )

    def _filter_encoder_configs_for_obs_group(self, encoder_configs: dict, obs_group_names: list, obs_spaces: dict) -> dict:
        """Filter encoder configs to only include modalities available in the given observation group."""
        filtered_configs = {}

        for modality_name, config in encoder_configs.items():
            # Check if this modality has any observations in the target group
            modality_obs_groups = self._get_modality_obs_groups(modality_name, set(obs_group_names), obs_spaces)

            if modality_obs_groups:  # If modality has observations in this group
                filtered_configs[modality_name] = config

        return filtered_configs

    def _get_modality_obs_groups(self, modality_name: str, target_obs_groups: set, obs_spaces: dict) -> list:
        """Get observation groups that belong to a specific modality (same logic as EncoderDecoderNetwork)."""
        modality_obs_groups = []

        for obs_group in target_obs_groups:
            if obs_group not in obs_spaces:
                continue

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

    # Standard RSL-RL interface methods (keep signatures identical)
    def act(self, obs, **kwargs):
        """Follow exact RSL-RL pattern."""
        if self.actor_architecture == "mlp":
            obs_processed = self.get_actor_obs(obs)
            obs_processed = self.actor_obs_normalizer(obs_processed)
            mean = self.actor(obs_processed)
        else:
            mean = self.actor.forward_actor(obs)

        self.update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, obs):
        """Follow exact RSL-RL pattern."""
        if self.actor_architecture == "mlp":
            obs_processed = self.get_actor_obs(obs)
            obs_processed = self.actor_obs_normalizer(obs_processed)
            return self.actor(obs_processed)
        else:
            return self.actor.forward_actor(obs)

    def evaluate(self, obs, **kwargs):
        """Follow exact RSL-RL pattern."""
        if self.critic_architecture == "mlp":
            obs_processed = self.get_critic_obs(obs)
            obs_processed = self.critic_obs_normalizer(obs_processed)
            return self.critic(obs_processed)
        else:
            if self.shared_encoder and self.actor_architecture == "encoder_decoder":
                return self.actor.forward_critic(obs)
            else:
                return self.critic.forward_critic(obs)

    def get_actor_obs(self, obs):
        """Follow exact RSL-RL pattern for MLP compatibility."""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        """Follow exact RSL-RL pattern for MLP compatibility."""
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def update_distribution(self, mean):
        """Update action distribution (follows RSL-RL pattern exactly)."""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    # Standard RSL-RL properties
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        """Follow exact RSL-RL pattern."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        """Follow exact RSL-RL pattern with encoder-decoder support."""
        if self.actor_obs_normalization and self.actor_architecture == "mlp":
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)

        if self.critic_obs_normalization and self.critic_architecture == "mlp":
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

        # For encoder-decoder architectures, normalization is handled within the encoders
        # No additional normalization update needed here

    def reset(self, dones=None):
        """Follow exact RSL-RL pattern."""
        pass

    def load_state_dict(self, state_dict, strict=True):
        """Follow exact RSL-RL pattern with resume indication."""
        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes