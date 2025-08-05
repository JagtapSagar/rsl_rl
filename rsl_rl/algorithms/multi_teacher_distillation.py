
import torch
import torch.nn as nn
import torch.optim as optim

# Import from installed rsl_rl
from rsl_rl.storage import RolloutStorage
from rsl_rl.modules import MultiTeacher


class MultiTeacherDistillation:
    """Multi-teacher distillation algorithm for training a student model to mimic multiple teacher models."""

    policy: MultiTeacher
    """The multi-teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        teacher_loss_weights=None,
        diversity_loss_coef=0.0,
        temperature=4.0,
        alpha=0.7,
        parallel_teachers=True,
        loss_type="mse",
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # Distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # Multi-teacher distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.teacher_loss_weights = teacher_loss_weights or [1.0] * len(policy.teachers)
        self.diversity_loss_coef = diversity_loss_coef
        self.temperature = temperature
        self.alpha = alpha  # Weighting between distillation loss and task loss
        self.parallel_teachers = parallel_teachers

        # Initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        elif loss_type == "kl_div":
            self.loss_fn = self._kl_divergence_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber, kl_div")

        self.num_updates = 0

        # Load teacher checkpoints
        if hasattr(self.policy, 'load_teacher_checkpoints'):
            self.policy.load_teacher_checkpoints()

    def _kl_divergence_loss(self, student_actions, teacher_actions):
        """KL divergence loss with temperature scaling."""
        student_soft = torch.softmax(student_actions / self.temperature, dim=-1)
        teacher_soft = torch.softmax(teacher_actions / self.temperature, dim=-1)
        return nn.functional.kl_div(
            torch.log(student_soft + 1e-8), teacher_soft, reduction='batchmean'
        ) * (self.temperature ** 2)

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        """Create rollout storage."""
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs):
        """Compute actions from student and ensemble teacher outputs."""
        # Compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # Record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        """Process environment step and record transition."""
        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        """Update the student model using multi-teacher distillation."""
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_diversity_loss = 0
        mean_total_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            for obs, privileged_observations, actions, privileged_actions, dones in self.storage.generator():
                # Student inference for gradient computation
                student_actions = self.policy.act_inference(obs)

                # Main distillation loss (student vs ensemble teacher)
                behavior_loss = self.loss_fn(student_actions, privileged_actions)

                # Diversity loss to encourage learning from diverse teachers
                diversity_loss = torch.tensor(0.0, device=self.device)
                if self.diversity_loss_coef > 0:
                    # Get individual teacher outputs for diversity calculation
                    teacher_obs = privileged_observations if hasattr(self.transition, 'privileged_observations') else obs
                    diversity_loss = self.policy.get_teacher_diversity(teacher_obs)

                # Total loss
                total_loss = self.alpha * behavior_loss + self.diversity_loss_coef * diversity_loss
                loss = loss + total_loss

                # Accumulate loss statistics
                mean_behavior_loss += behavior_loss.item()
                mean_diversity_loss += diversity_loss.item()
                mean_total_loss += total_loss.item()
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # Reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        # Calculate mean losses
        mean_behavior_loss /= cnt
        mean_diversity_loss /= cnt
        mean_total_loss /= cnt
        
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # Construct the loss dictionary
        loss_dict = {
            "behavior": mean_behavior_loss,
            "diversity": mean_diversity_loss,
            "total": mean_total_loss,
            "num_teachers": len(self.policy.teachers),
        }

        return loss_dict

    def get_teacher_individual_losses(self, obs, teacher_obs):
        """Get individual losses for each teacher (for analysis/debugging)."""
        with torch.no_grad():
            student_actions = self.policy.act_inference(obs)
            individual_losses = []
            
            for i, teacher in enumerate(self.policy.teachers):
                teacher_actions = teacher(teacher_obs)
                loss = self.loss_fn(student_actions, teacher_actions)
                individual_losses.append(loss.item())
            
            return individual_losses

    """
    Helper functions for distributed training
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them."""
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel