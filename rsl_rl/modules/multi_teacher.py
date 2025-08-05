
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from concurrent.futures import ThreadPoolExecutor
import threading

# Import from installed rsl_rl
from rsl_rl.utils import resolve_nn_activation


class MultiTeacher(nn.Module):
    """Multi-teacher policy for distillation with multiple parallel teacher networks."""
    
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teachers=[],  # List of teacher configs with hidden_dims, checkpoint_path, weight
        ensemble_method="weighted_average",
        activation="elu",
        init_noise_std=0.1,
        parallel_teachers=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "MultiTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False
        self.ensemble_method = ensemble_method
        self.parallel_teachers = parallel_teachers
        self.teacher_weights = [t['weight'] for t in teachers]
        
        # Normalize weights
        total_weight = sum(self.teacher_weights)
        self.teacher_weights = [w / total_weight for w in self.teacher_weights]

        mlp_input_dim_s = num_student_obs
        mlp_input_dim_t = num_teacher_obs

        # Build student network
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        # Build teacher networks
        self.teachers = nn.ModuleList()
        self.teacher_configs = teachers
        
        for i, teacher_cfg in enumerate(teachers):
            teacher_layers = []
            teacher_hidden_dims = teacher_cfg['hidden_dims']
            teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
            teacher_layers.append(activation)
            for layer_index in range(len(teacher_hidden_dims)):
                if layer_index == len(teacher_hidden_dims) - 1:
                    teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
                else:
                    teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                    teacher_layers.append(activation)
            teacher = nn.Sequential(*teacher_layers)
            teacher.eval()
            self.teachers.append(teacher)

        # Attention mechanism for ensemble (if needed)
        if self.ensemble_method == "attention":
            self.attention_net = nn.Sequential(
                nn.Linear(mlp_input_dim_t, 64),
                activation,
                nn.Linear(64, len(teachers)),
                nn.Softmax(dim=-1)
            )

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLPs: {len(self.teachers)} networks")
        print(f"Ensemble method: {self.ensemble_method}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        # Thread pool for parallel teacher inference
        if self.parallel_teachers:
            self._thread_pool = ThreadPoolExecutor(max_workers=min(len(teachers), 8))
            self._lock = threading.Lock()

    def reset(self, dones=None, hidden_states=None):
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

    def update_distribution(self, observations):
        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.student(observations)
        return actions_mean

    def _single_teacher_inference(self, teacher_idx, teacher_observations):
        """Helper function for single teacher inference."""
        with torch.no_grad():
            return self.teachers[teacher_idx](teacher_observations)

    def evaluate(self, teacher_observations):
        """Evaluate all teachers and combine their outputs."""
        with torch.no_grad():
            if self.parallel_teachers and len(self.teachers) > 1:
                # Parallel inference
                futures = []
                for i in range(len(self.teachers)):
                    future = self._thread_pool.submit(
                        self._single_teacher_inference, i, teacher_observations
                    )
                    futures.append(future)
                
                teacher_actions = [future.result() for future in futures]
            else:
                # Sequential inference
                teacher_actions = []
                for teacher in self.teachers:
                    teacher_actions.append(teacher(teacher_observations))

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
                attention_weights = self.attention_net(teacher_observations)  # [batch, num_teachers]
                teacher_stack = torch.stack(teacher_actions, dim=-1)  # [batch, actions, num_teachers]
                combined_actions = torch.sum(
                    teacher_stack * attention_weights.unsqueeze(1), dim=-1
                )
                return combined_actions
            
            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def get_teacher_diversity(self, teacher_observations):
        """Calculate diversity among teacher outputs for regularization."""
        with torch.no_grad():
            teacher_actions = []
            for teacher in self.teachers:
                teacher_actions.append(teacher(teacher_observations))
            
            # Calculate pairwise differences
            diversity_loss = 0.0
            count = 0
            for i in range(len(teacher_actions)):
                for j in range(i + 1, len(teacher_actions)):
                    diversity_loss += torch.mean((teacher_actions[i] - teacher_actions[j]) ** 2)
                    count += 1
            
            return diversity_loss / count if count > 0 else torch.tensor(0.0)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks."""
        
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
            # if hasattr(teacher_cfg, 'checkpoint_path') and teacher_cfg.checkpoint_path:
            if teacher_cfg['checkpoint_path']:
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