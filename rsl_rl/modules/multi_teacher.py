
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from concurrent.futures import ThreadPoolExecutor
import threading

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class MultiTeacher(nn.Module):
    """Multi-teacher policy for distillation with multiple parallel teacher networks.

    Compatible with TensorDict observation system and current module interface.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teachers=[],  # List of teacher configs with hidden_dims, checkpoint_path, weight
        ensemble_method="weighted_average",
        activation="elu",
        init_noise_std=0.1,
        parallel_teachers=False,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "MultiTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.ensemble_method = ensemble_method
        self.parallel_teachers = parallel_teachers
        self.teacher_weights = [t['weight'] for t in teachers] if teachers else [1.0]

        # Normalize weights
        total_weight = sum(self.teacher_weights)
        self.teacher_weights = [w / total_weight for w in self.teacher_weights]

        # get the observation dimensions
        self.obs_groups = obs_groups

        # Student observations (typically subset of full observations)
        num_student_obs = 0
        for obs_group in obs_groups.get("policy", obs_groups.get("student", [])):
            assert len(obs[obs_group].shape) == 2, "The MultiTeacher module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]

        # Teacher observations (typically full privileged observations)
        num_teacher_obs = 0
        for obs_group in obs_groups.get("teacher", obs_groups.get("critic", [])):
            assert len(obs[obs_group].shape) == 2, "The MultiTeacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        # Build student network
        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)

        # Student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Build teacher networks
        self.teachers = nn.ModuleList()
        self.teacher_configs = teachers

        for i, teacher_cfg in enumerate(teachers):
            teacher_hidden_dims = teacher_cfg['hidden_dims']
            teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
            teacher.eval()
            self.teachers.append(teacher)

        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # Attention mechanism for ensemble (if needed)
        if self.ensemble_method == "attention":
            self.attention_net = MLP(num_teacher_obs, len(teachers), [64], activation)
            self.attention_net.add_module("softmax", nn.Softmax(dim=-1))

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLPs: {len(self.teachers)} networks")
        print(f"Ensemble method: {self.ensemble_method}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        # Thread pool for parallel teacher inference
        if self.parallel_teachers:
            self._thread_pool = ThreadPoolExecutor(max_workers=min(len(teachers), 8))
            self._lock = threading.Lock()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_student_obs(self, obs):
        """Extract student observations from TensorDict."""
        obs_list = []
        for obs_group in self.obs_groups.get("policy", self.obs_groups.get("student", [])):
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs):
        """Extract teacher observations from TensorDict."""
        obs_list = []
        for obs_group in self.obs_groups.get("teacher", self.obs_groups.get("critic", [])):
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def update_distribution(self, obs):
        """Update action distribution using student network."""
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        mean = self.student(student_obs)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        """Sample actions from student policy."""
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        """Get deterministic actions from student policy."""
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        return self.student(student_obs)

    def _single_teacher_inference(self, teacher_idx, teacher_observations):
        """Helper function for single teacher inference."""
        with torch.no_grad():
            return self.teachers[teacher_idx](teacher_observations)

    def evaluate(self, obs):
        """Evaluate all teachers and combine their outputs."""
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)

        with torch.no_grad():
            if self.parallel_teachers and len(self.teachers) > 1:
                # Parallel inference
                futures = []
                for i in range(len(self.teachers)):
                    future = self._thread_pool.submit(
                        self._single_teacher_inference, i, teacher_obs
                    )
                    futures.append(future)

                teacher_actions = [future.result() for future in futures]
            else:
                # Sequential inference
                teacher_actions = []
                for teacher in self.teachers:
                    teacher_actions.append(teacher(teacher_obs))

            # Combine teacher outputs based on ensemble method
            if self.ensemble_method == "weighted_average":
                combined_actions = torch.zeros_like(teacher_actions[0])
                for i, actions in enumerate(teacher_actions):
                    combined_actions += self.teacher_weights[i] * actions
                return combined_actions

            elif self.ensemble_method == "majority_vote":
                # For continuous actions, use weighted voting
                combined_actions = torch.zeros_like(teacher_actions[0])
                for i, actions in enumerate(teacher_actions):
                    combined_actions += self.teacher_weights[i] * actions
                return combined_actions

            elif self.ensemble_method == "attention":
                # Use attention weights
                attention_weights = self.attention_net(teacher_obs)  # [batch, num_teachers]
                teacher_stack = torch.stack(teacher_actions, dim=-1)  # [batch, actions, num_teachers]
                combined_actions = torch.sum(
                    teacher_stack * attention_weights.unsqueeze(1), dim=-1
                )
                return combined_actions

            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def get_teacher_diversity(self, obs):
        """Calculate diversity among teacher outputs for regularization."""
        teacher_obs = self.get_teacher_obs(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)

        with torch.no_grad():
            teacher_actions = []
            for teacher in self.teachers:
                teacher_actions.append(teacher(teacher_obs))

            # Calculate pairwise differences
            diversity_loss = 0.0
            count = 0
            for i in range(len(teacher_actions)):
                for j in range(i + 1, len(teacher_actions)):
                    diversity_loss += torch.mean((teacher_actions[i] - teacher_actions[j]) ** 2)
                    count += 1

            return diversity_loss / count if count > 0 else torch.tensor(0.0)

    def get_actions_log_prob(self, actions):
        """Get log probabilities of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        """Update observation normalization statistics."""
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)
        if self.teacher_obs_normalization:
            teacher_obs = self.get_teacher_obs(obs)
            self.teacher_obs_normalizer.update(teacher_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Returns:
            bool: Whether this training resumes a previous training.
        """

        # Check if loading from RL training or distillation training
        if any("actor" in key for key in state_dict.keys()):
            # Loading from RL training - load into first teacher by default
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            if len(self.teachers) > 0:
                self.teachers[0].load_state_dict(teacher_state_dict, strict=strict)
            self.loaded_teacher = True
            for teacher in self.teachers:
                teacher.eval()
            return False

        elif any("student" in key for key in state_dict.keys()):
            # Loading from distillation training
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            for teacher in self.teachers:
                teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def load_teacher_checkpoints(self):
        """Load individual teacher checkpoints from their specified paths."""
        for i, teacher_cfg in enumerate(self.teacher_configs):
            if teacher_cfg.get('checkpoint_path'):
                try:
                    checkpoint = torch.load(teacher_cfg['checkpoint_path'], map_location='cpu')

                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    # Extract actor parameters if present
                    teacher_state_dict = {}
                    for key, value in state_dict.items():
                        if "actor." in key:
                            teacher_state_dict[key.replace("actor.", "")] = value
                        elif not key.startswith(("critic.", "value_function.")):
                            teacher_state_dict[key] = value

                    self.teachers[i].load_state_dict(teacher_state_dict, strict=False)
                    print(f"Loaded teacher {i} from {teacher_cfg['checkpoint_path']}")

                except Exception as e:
                    print(f"Warning: Failed to load teacher {i} from {teacher_cfg['checkpoint_path']}: {e}")

        self.loaded_teacher = True
        for teacher in self.teachers:
            teacher.eval()

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def __del__(self):
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)