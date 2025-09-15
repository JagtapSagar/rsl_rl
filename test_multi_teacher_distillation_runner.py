#!/usr/bin/env python3
"""
Test script for DistillationRunner with MultiTeacher functionality.

This script demonstrates that the updated DistillationRunner can successfully
instantiate and use MultiTeacher + MultiTeacherDistillation components.
"""

import torch
from tensordict import TensorDict
from rsl_rl.runners.distillation_runner import DistillationRunner


class MockVecEnv:
    """Mock environment for testing purposes."""

    def __init__(self):
        self.num_envs = 4
        self.num_actions = 12

    def get_observations(self):
        """Return mock observations with different observation types."""
        return TensorDict({
            'proprioception': torch.randn(self.num_envs, 48),  # Basic robot state
            'privileged': torch.randn(self.num_envs, 32)       # Privileged teacher info
        }, batch_size=[self.num_envs])


def test_distillation_runner_with_multi_teacher():
    """Test DistillationRunner instantiation and basic functionality with MultiTeacher."""

    print('=== TESTING DistillationRunner + MultiTeacher ===\n')

    # Create configuration for MultiTeacher
    train_cfg = {
        'num_steps_per_env': 24,
        'save_interval': 100,
        'obs_groups': {
            'policy': ['proprioception'],           # Student uses basic observations
            'teacher': ['proprioception', 'privileged']  # Teachers use privileged info
        },
        'policy': {
            'class_name': 'MultiTeacher',
            'student_hidden_dims': [256, 256, 256],
            'teachers': [
                {
                    'hidden_dims': [256, 256, 256],
                    'checkpoint_path': None,  # No checkpoint for test
                    'weight': 0.7
                },
                {
                    'hidden_dims': [512, 256, 128],
                    'checkpoint_path': None,  # No checkpoint for test
                    'weight': 0.3
                }
            ],
            'activation': 'elu',
            'ensemble_method': 'weighted_average'
        },
        'algorithm': {
            'class_name': 'MultiTeacherDistillation',
            'num_learning_epochs': 3,
            'learning_rate': 1e-3,
            'loss_type': 'mse',
            'alpha': 0.8,                    # Distillation vs task loss weighting
            'diversity_loss_coef': 0.05      # Teacher diversity regularization
        }
    }

    # Create mock environment
    env = MockVecEnv()

    print('Configuration:')
    print(f'  Environment: {env.num_envs} envs, {env.num_actions} actions')
    print(f'  Policy: {train_cfg["policy"]["class_name"]}')
    print(f'  Algorithm: {train_cfg["algorithm"]["class_name"]}')
    print(f'  Teachers: {len(train_cfg["policy"]["teachers"])}')
    print(f'  Student obs: {train_cfg["obs_groups"]["policy"]}')
    print(f'  Teacher obs: {train_cfg["obs_groups"]["teacher"]}')

    try:
        # THIS IS THE KEY TEST - instantiate DistillationRunner with MultiTeacher
        runner = DistillationRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=None,  # No logging for test
            device='cpu'
        )

        print('\n✅ SUCCESS: DistillationRunner instantiated with MultiTeacher!')
        print(f'   Runner type: {type(runner).__name__}')
        print(f'   Policy type: {type(runner.alg.policy).__name__}')
        print(f'   Algorithm type: {type(runner.alg).__name__}')
        print(f'   Storage type: {runner.alg.storage.training_type}')
        print(f'   Number of teachers: {len(runner.alg.policy.teachers)}')
        print(f'   Teacher weights: {runner.alg.policy.teacher_weights}')
        print(f'   Ensemble method: {runner.alg.policy.ensemble_method}')

        # Test basic forward pass
        obs = env.get_observations()
        with torch.no_grad():
            # Test student action generation
            actions = runner.alg.act(obs)
            print(f'   Student action shape: {actions.shape}')

            # Test teacher ensemble
            teacher_actions = runner.alg.policy.evaluate(obs)
            print(f'   Teacher ensemble action shape: {teacher_actions.shape}')

            # Test diversity calculation
            diversity = runner.alg.policy.get_teacher_diversity(obs)
            print(f'   Teacher diversity: {diversity.item():.4f}')

        print('\n🎉 DistillationRunner + MultiTeacher working perfectly!')
        return True

    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_backwards_compatibility():
    """Test that DistillationRunner still works with original StudentTeacher."""

    print('\n=== TESTING Backwards Compatibility ===\n')

    # Configuration for original StudentTeacher
    train_cfg = {
        'num_steps_per_env': 24,
        'save_interval': 100,
        'obs_groups': {
            'policy': ['proprioception'],
            'teacher': ['proprioception', 'privileged']
        },
        'policy': {
            'class_name': 'StudentTeacher',  # Original class
            'student_hidden_dims': [256, 256, 256],
            'teacher_hidden_dims': [256, 256, 256],
            'activation': 'elu'
        },
        'algorithm': {
            'class_name': 'Distillation',   # Original algorithm
            'num_learning_epochs': 3,
            'learning_rate': 1e-3,
            'loss_type': 'mse'
        }
    }

    env = MockVecEnv()

    try:
        runner = DistillationRunner(env, train_cfg, log_dir=None, device='cpu')
        print('✅ Backwards compatibility confirmed!')
        print(f'   Policy type: {type(runner.alg.policy).__name__}')
        print(f'   Algorithm type: {type(runner.alg).__name__}')
        return True

    except Exception as e:
        print(f'❌ Backwards compatibility failed: {e}')
        return False


if __name__ == '__main__':
    """Run all tests."""

    print('Testing MultiTeacher integration with DistillationRunner...\n')

    # Test 1: MultiTeacher functionality
    multi_teacher_success = test_distillation_runner_with_multi_teacher()

    # Test 2: Backwards compatibility
    backwards_compat_success = test_backwards_compatibility()

    # Summary
    print('\n' + '='*60)
    print('TEST SUMMARY:')
    print(f'  MultiTeacher support: {"✅ PASS" if multi_teacher_success else "❌ FAIL"}')
    print(f'  Backwards compatibility: {"✅ PASS" if backwards_compat_success else "❌ FAIL"}')

    if multi_teacher_success and backwards_compat_success:
        print('\n🎉 ALL TESTS PASSED - DistillationRunner ready for MultiTeacher!')
    else:
        print('\n⚠️  Some tests failed - check implementation')

    print('\nUsage example:')
    print('runner = DistillationRunner(env, config_with_multi_teacher)')
    print('runner.learn(num_learning_iterations=1000)')