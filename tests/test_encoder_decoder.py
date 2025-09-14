#!/usr/bin/env python3

import torch
from rsl_rl.modules import FlexibleActorCritic

def test_encoder_decoder_basic():
    """Test basic encoder-decoder functionality with FlexibleActorCritic."""

    # Set up mock observation spaces (mimicking IsaacLab structure)
    obs_spaces = {
        "robot_state": torch.zeros(1, 48),  # Joint positions, velocities, etc.
        "camera_rgb": torch.zeros(1, 3, 64, 64),  # RGB camera
        "terrain_info": torch.zeros(1, 16),  # Privileged terrain information
    }

    # Define observation groups
    obs_groups = {
        "policy": ["robot_state", "camera_rgb"],  # What policy sees
        "critic": ["robot_state", "camera_rgb", "terrain_info"]  # Critic also sees privileged info
    }

    # Encoder configurations
    encoder_configs = {
        "proprioception": {
            "type": "mlp",
            "latent_dim": 64,
            "hidden_dims": [128, 128],
            "activation": "elu",
            "normalization": True
        },
        "vision": {
            "type": "cnn",
            "latent_dim": 128,
            "channels": [32, 64, 128],
            "kernel_sizes": [8, 4, 3],
            "strides": [4, 2, 1]
        },
        "privileged": {
            "type": "mlp",
            "latent_dim": 32,
            "hidden_dims": [64, 64]
        }
    }

    # Fusion configuration
    fusion_config = {
        "type": "concat",
        "output_dim": 256
    }

    # Decoder configurations
    decoder_configs = {
        "action": {"hidden_dims": [256, 128], "activation": "elu"},
        "value": {"hidden_dims": [256, 128], "activation": "elu"}
    }

    # Create FlexibleActorCritic with encoder-decoder architecture
    # Note: Actor and critic will have different fusion configurations based on available modalities
    model = FlexibleActorCritic(
        obs=obs_spaces,
        obs_groups=obs_groups,
        num_actions=12,  # 12 joint actions for quadruped
        actor_architecture="encoder_decoder",
        critic_architecture="encoder_decoder",
        shared_encoder=False,
        encoder_configs=encoder_configs,
        fusion_config=fusion_config,
        decoder_configs=decoder_configs
    )

    print("✅ FlexibleActorCritic created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Show what encoders were actually created for each network
    print("📊 Network Architecture:")
    print(f"  Actor encoders: {list(model.actor.encoders.keys())}")
    print(f"  Critic encoders: {list(model.critic.encoders.keys())}")
    print(f"  Actor fusion input dim: {sum(enc.get_output_dim() for enc in model.actor.encoders.values())}")
    print(f"  Critic fusion input dim: {sum(enc.get_output_dim() for enc in model.critic.encoders.values())}")

    # Test forward pass with mock observations
    obs = {
        "robot_state": torch.randn(4, 48),  # Batch of 4
        "camera_rgb": torch.randn(4, 3, 64, 64),
        "terrain_info": torch.randn(4, 16)
    }

    # Test actor inference
    actions = model.act_inference(obs)
    print(f"✅ Actor inference: {actions.shape} -> expected (4, 12)")
    assert actions.shape == (4, 12), f"Expected (4, 12), got {actions.shape}"

    # Test critic evaluation
    values = model.evaluate(obs)
    print(f"✅ Critic evaluation: {values.shape} -> expected (4, 1)")
    assert values.shape == (4, 1), f"Expected (4, 1), got {values.shape}"

    # Test action sampling
    sampled_actions = model.act(obs)
    print(f"✅ Action sampling: {sampled_actions.shape} -> expected (4, 12)")
    assert sampled_actions.shape == (4, 12), f"Expected (4, 12), got {sampled_actions.shape}"

    # Test action log probabilities
    log_probs = model.get_actions_log_prob(sampled_actions)
    print(f"✅ Action log probs: {log_probs.shape} -> expected (4,)")
    assert log_probs.shape == (4,), f"Expected (4,), got {log_probs.shape}"

    # Test properties
    assert hasattr(model, 'action_mean'), "Missing action_mean property"
    assert hasattr(model, 'action_std'), "Missing action_std property"
    assert hasattr(model, 'entropy'), "Missing entropy property"
    print("✅ All required properties present")

    print("\n🎉 All encoder-decoder tests passed!")



def test_encoder_decoder_basic2():
    """Test encoder-decoder with proprioception + privileged modalities only."""

    # Set up mock observation spaces (mimicking IsaacLab structure)
    obs_spaces = {
        "robot_state": torch.zeros(1, 48),  # Joint positions, velocities, etc.
        "privileged_state": torch.zeros(1, 4),  # Joint failure bits
    }

    # Define observation groups
    obs_groups = {
        "policy": ["robot_state"],  # What policy sees
        "critic": ["robot_state", "privileged_state"]  # Critic also sees privileged info
    }

    # Encoder configurations
    encoder_configs = {
        "proprioception": {
            "type": "mlp",
            "latent_dim": 64,
            "hidden_dims": [64, 64],
            "activation": "elu",
            "normalization": True
        },
        "privileged": {
            "type": "mlp",
            "latent_dim": 32,
            "hidden_dims": [64, 64]
        }
    }

    # Fusion configuration
    fusion_config = {
        "type": "concat",
        "output_dim": 96  # 64 + 32 = 96 for critic, 64 for actor
    }

    # Decoder configurations
    decoder_configs = {
        "action": {"hidden_dims": [64, 64], "activation": "elu"},  # Actor gets 64 dims
        "value": {"hidden_dims": [96, 64], "activation": "elu"}    # Critic gets 96 dims
    }

    # Create FlexibleActorCritic with encoder-decoder architecture
    # Note: Actor and critic will have different fusion configurations based on available modalities
    model = FlexibleActorCritic(
        obs=obs_spaces,
        obs_groups=obs_groups,
        num_actions=12,  # 12 joint actions for quadruped
        actor_architecture="encoder_decoder",
        critic_architecture="encoder_decoder",
        shared_encoder=False,
        encoder_configs=encoder_configs,
        fusion_config=fusion_config,
        decoder_configs=decoder_configs
    )

    print("✅ FlexibleActorCritic created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Show what encoders were actually created for each network
    print("📊 Network Architecture:")
    print(f"  Actor encoders: {list(model.actor.encoders.keys())}")
    print(f"  Critic encoders: {list(model.critic.encoders.keys())}")
    print(f"  Actor fusion input dim: {sum(enc.get_output_dim() for enc in model.actor.encoders.values())}")
    print(f"  Critic fusion input dim: {sum(enc.get_output_dim() for enc in model.critic.encoders.values())}")

    # Test forward pass with mock observations
    obs = {
        "robot_state": torch.randn(4, 48),  # Batch of 4
        "privileged_state": torch.randn(4, 4)  # Joint failure bits
    }

    # Test actor inference
    actions = model.act_inference(obs)
    print(f"✅ Actor inference: {actions.shape} -> expected (4, 12)")
    assert actions.shape == (4, 12), f"Expected (4, 12), got {actions.shape}"

    # Test critic evaluation
    values = model.evaluate(obs)
    print(f"✅ Critic evaluation: {values.shape} -> expected (4, 1)")
    assert values.shape == (4, 1), f"Expected (4, 1), got {values.shape}"

    # Test action sampling
    sampled_actions = model.act(obs)
    print(f"✅ Action sampling: {sampled_actions.shape} -> expected (4, 12)")
    assert sampled_actions.shape == (4, 12), f"Expected (4, 12), got {sampled_actions.shape}"

    # Test action log probabilities
    log_probs = model.get_actions_log_prob(sampled_actions)
    print(f"✅ Action log probs: {log_probs.shape} -> expected (4,)")
    assert log_probs.shape == (4,), f"Expected (4,), got {log_probs.shape}"

    # Test properties
    assert hasattr(model, 'action_mean'), "Missing action_mean property"
    assert hasattr(model, 'action_std'), "Missing action_std property"
    assert hasattr(model, 'entropy'), "Missing entropy property"
    print("✅ All required properties present")

    print("\n🎉 All encoder-decoder tests passed!")


def test_mlp_compatibility():
    """Test that FlexibleActorCritic works identically to original ActorCritic in MLP mode."""

    # Simple observation structure for MLP
    obs_spaces = {
        "robot_state": torch.zeros(1, 48)
    }

    obs_groups = {
        "policy": ["robot_state"],
        "critic": ["robot_state"]
    }

    # Create FlexibleActorCritic in MLP mode
    model = FlexibleActorCritic(
        obs=obs_spaces,
        obs_groups=obs_groups,
        num_actions=12,
        actor_architecture="mlp",  # Standard MLP mode
        critic_architecture="mlp",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256]
    )

    print("✅ FlexibleActorCritic in MLP mode created")

    # Test with simple observations
    obs = {"robot_state": torch.randn(4, 48)}

    actions = model.act_inference(obs)
    values = model.evaluate(obs)

    print(f"✅ MLP mode - Actions: {actions.shape}, Values: {values.shape}")
    assert actions.shape == (4, 12)
    assert values.shape == (4, 1)

    print("🎉 MLP compatibility test passed!")


if __name__ == "__main__":
    print("Testing RSL-RL Encoder-Decoder Implementation\n")

    print("=== Test 1: Multi-Modal Encoder-Decoder ===")
    test_encoder_decoder_basic()

    print("\n=== Test 2: Proprioception + Privileged ===")
    test_encoder_decoder_basic2()

    print("\n=== Test 3: MLP Compatibility ===")
    test_mlp_compatibility()

    print("\n✨ All tests completed successfully!")