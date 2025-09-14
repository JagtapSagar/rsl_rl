# RSL-RL Enhancement Project Notes

## Project Overview
Making rsl-rl compatible with:
1. **PPO with Encoder-Decoder & Actor-Critic architectures**
2. **Multi-teacher Student Distillation** (teachers, student, or both may be MLPs or encoder-decoder)

**Training Context**: Teachers and students will be trained using IsaacLab environments, which provide:
- Multi-modal observations (proprioception, vision, privileged information)
- Diverse robotic tasks (legged locomotion, manipulation, etc.)
- High-fidelity physics simulation
- Massive parallelization capabilities

## Current Architecture Analysis

### Existing PPO Implementation (`rsl_rl/algorithms/ppo.py`)
- **Current Architecture**: Uses `ActorCritic` module from `rsl_rl.modules`
- **Policy Interface**: Expects policy with `act()`, `evaluate()`, `get_actions_log_prob()` methods
- **Key Features**:
  - Random Network Distillation (RND) support
  - Symmetry augmentation support
  - Multi-GPU training
  - Recurrent policy support (via `is_recurrent` flag)
  - Observation normalization

### Existing Actor-Critic Module (`rsl_rl/modules/actor_critic.py`)
- **Architecture**: Two separate MLP networks (actor + critic)
- **Input**: 1D observations only (concatenated from observation groups)
- **Actor**: MLP → action mean, learnable std parameter
- **Critic**: MLP → single value output
- **Observation Groups**: "policy" (actor) and "critic" groups
- **Normalization**: Empirical normalization for both actor and critic observations

### Existing Student-Teacher Framework (`rsl_rl/modules/student_teacher.py`)
- **Current Setup**: Single teacher → single student distillation
- **Architecture**: Both teacher and student are MLPs
- **Observation Groups**: "policy" (student) and "teacher" groups
- **Teacher**: Fixed/frozen during distillation training
- **Student**: Trainable network that learns to mimic teacher

### Network Infrastructure (`rsl_rl/networks/`)
- **MLP**: Flexible multi-layer perceptron with configurable activation
- **Normalization**: Empirical normalization for observation processing
- **Memory**: LSTM implementation for recurrent policies

## Enhancement Plan

### Phase 1: Encoder-Decoder Architecture Support

#### 1.1 Create Encoder-Decoder Networks
**New Files to Create:**
- `rsl_rl/networks/encoder_decoder.py`
  - `Encoder` class (observation → latent representation)
  - `Decoder` class (latent + context → actions/values)
  - `EncoderDecoder` class (combined architecture)

**Key Design Decisions for IsaacLab Integration:**
- **Multi-modal Observation Support**:
  - CNN encoders for camera observations (RGB, depth, segmentation)
  - MLP encoders for proprioceptive data (joint positions, velocities, forces)
  - Specialized encoders for privileged information (terrain maps, object states)
- **Modality-Specific Processing**:
  - Image preprocessing (normalization, augmentation)
  - Proprioception normalization and filtering
  - Temporal observation stacking for dynamics
- **Efficient Parallelization**:
  - GPU-optimized convolutions for visual data
  - Batched processing across thousands of parallel environments
- **Flexible Architecture**:
  - Configurable encoder types per observation modality
  - Latent space dimensionality configuration
  - Optional attention mechanisms for cross-modal fusion

#### 1.2 Enhanced Actor-Critic Module
**New File:** `rsl_rl/modules/actor_critic_encoder_decoder.py`
- Inherit common interface from base `ActorCritic`
- Support both MLP and Encoder-Decoder architectures
- Configurable architecture selection per component (actor/critic)
- Mixed architectures (e.g., Encoder-Decoder actor + MLP critic)

**Architecture Options:**
1. **Full Encoder-Decoder**: Both actor and critic use encoder-decoder
2. **Hybrid**: Actor uses encoder-decoder, critic uses MLP (or vice versa)
3. **Shared Encoder**: Common encoder, separate decoders for actor/critic

#### 1.3 PPO Integration
**Modifications to `rsl_rl/algorithms/ppo.py`:**
- No changes needed to core algorithm - uses policy interface
- Enhanced observation handling for multi-modal inputs
- Support for different normalization strategies per modality

### Phase 2: Multi-Teacher Student Distillation

#### 2.1 Enhanced Student-Teacher Framework
**New File:** `rsl_rl/modules/multi_teacher_student.py`
- Support multiple teachers (configurable number)
- Each teacher can be MLP or Encoder-Decoder
- Student can be MLP or Encoder-Decoder
- Flexible teacher-student architecture combinations

**Key Features for IsaacLab Environments:**
- **Domain-Specific Teachers**:
  - Teacher specialists for different task aspects (locomotion, manipulation, navigation)
  - Environment-specific teachers (different terrains, object types, lighting conditions)
  - Difficulty-based teachers (easy → hard curriculum progression)
- **Observation Privilege Management**:
  - Teachers trained with privileged information (ground truth terrain, object properties)
  - Student learning from limited observations (onboard sensors only)
  - Gradual privilege removal during distillation
- **Multi-Task Distillation**:
  - Teachers trained on specific sub-tasks or environments
  - Student learning generalizable policies across multiple environments
  - Task-conditional distillation weights
- **Performance-Aware Aggregation**:
  - Teacher weighting based on performance in current environment
  - Dynamic teacher selection during training
  - Ensemble averaging with performance-based weights

#### 2.2 Enhanced Distillation Algorithm
**Modifications to `rsl_rl/algorithms/distillation.py`:**
- Support multiple teacher policies
- Enhanced loss computation (multi-teacher consistency)
- Teacher selection/weighting strategies
- Progressive distillation (curriculum learning from multiple teachers)

**Loss Function Enhancements:**
- Multi-teacher consistency loss
- Architecture-aware loss weighting
- Optional adversarial distillation components

### Phase 3: Configuration & Integration

#### 3.1 Configuration Schema
**Enhanced config structure for IsaacLab environments:**
```yaml
policy:
  type: "encoder_decoder_actor_critic"  # or "actor_critic", "multi_teacher_student"
  
  # Encoder-Decoder specific
  encoder:
    # Multi-modal encoder configuration
    modalities:
      proprioception:
        type: "mlp"
        dims: [128, 128]
        normalization: true
      rgb_camera:
        type: "cnn"
        channels: [32, 64, 128]
        kernel_sizes: [8, 4, 3]
        strides: [4, 2, 1]
      depth_camera:
        type: "cnn"  
        channels: [16, 32, 64]
      privileged_info:
        type: "mlp"
        dims: [64, 64]
        enabled_for_teacher: true
        enabled_for_student: false
    
    # Cross-modal fusion
    fusion:
      type: "attention"  # or "concat", "gated"
      latent_dim: 256
  
  decoder:
    type: "mlp"
    dims: [256, 128]
  
  # Multi-teacher specific
  teachers:
    - name: "locomotion_expert"
      type: "encoder_decoder"
      path: "teachers/locomotion_teacher.pt"
      specialization: "locomotion"
      weight: 0.4
    - name: "manipulation_expert" 
      type: "mlp"
      path: "teachers/manipulation_teacher.pt"
      specialization: "manipulation"
      weight: 0.4
    - name: "navigation_expert"
      type: "encoder_decoder"
      path: "teachers/navigation_teacher.pt"
      specialization: "navigation"
      weight: 0.2
  
  student:
    type: "encoder_decoder"
    # Uses same encoder config as above
    privileged_observations: false  # Student gets limited observations
  
  # IsaacLab environment integration
  environment:
    observation_spaces:
      proprioception: ["robot_state", "actions_history"]
      rgb_camera: ["front_cam", "wrist_cam"] 
      depth_camera: ["front_depth"]
      privileged_info: ["terrain_map", "object_states"]  # Teacher only
    
    curriculum:
      enabled: true
      difficulty_levels: [0.2, 0.5, 0.8, 1.0]
      teacher_weights_per_level:
        - [1.0, 0.0, 0.0]  # Easy: locomotion only
        - [0.6, 0.4, 0.0]  # Medium: locomotion + manipulation
        - [0.4, 0.4, 0.2]  # Hard: all teachers
        - [0.33, 0.33, 0.34]  # Expert: equal weighting
```

#### 3.2 Factory Pattern Implementation
**New File:** `rsl_rl/modules/policy_factory.py`
- Centralized policy creation based on configuration
- Support for all architecture combinations
- Backward compatibility with existing configurations

### Phase 4: Testing & Validation

#### 4.1 Unit Tests
- Individual component testing (encoders, decoders, multi-teacher modules)
- Interface compatibility tests
- Configuration validation tests

#### 4.2 Integration Tests
- End-to-end PPO training with encoder-decoder policies
- Multi-teacher distillation workflows
- Mixed architecture combinations

#### 4.3 Performance Benchmarks
- Compare MLP vs Encoder-Decoder performance
- Multi-teacher vs single-teacher distillation effectiveness
- Memory and computational overhead analysis

## Implementation Priority

### High Priority (Essential)
1. Encoder-Decoder network implementations
2. Enhanced Actor-Critic with architecture flexibility
3. Multi-teacher Student-Teacher module
4. Configuration system updates

### Medium Priority (Important)
1. Advanced teacher weighting strategies
2. Attention mechanisms in decoders
3. Progressive distillation algorithms
4. Comprehensive testing suite

### Low Priority (Nice-to-have)
1. Adversarial distillation components
2. Transformer-based encoders
3. Advanced visualization tools
4. Performance optimization

## Key Design Principles

1. **Backward Compatibility**: All existing configurations should work unchanged
2. **Modular Design**: Each component should be independently testable and replaceable
3. **Interface Consistency**: Maintain existing PPO algorithm interface
4. **Configuration Driven**: Architecture selection via configuration files
5. **Performance First**: Minimize computational overhead of new features
6. **IsaacLab Optimization**: Leverage GPU parallelization and multi-modal observations
7. **Privileged Information Handling**: Clean separation between teacher and student observation spaces
8. **Curriculum Learning Support**: Progressive difficulty and teacher weighting schemes

---

# DETAILED TECHNICAL ARCHITECTURE

## 1. Encoder-Decoder Architecture Design

### 1.1 Base Encoder Interface
```python
class BaseEncoder(nn.Module):
    """Abstract base class for all encoders."""
    
    def __init__(self, input_shape: tuple, latent_dim: int):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        raise NotImplementedError
    
    def get_output_dim(self) -> int:
        """Return the dimension of encoded output."""
        return self.latent_dim
```

### 1.2 Modality-Specific Encoders

#### Vision Encoder (CNN-based)
```python
class VisionEncoder(BaseEncoder):
    """CNN encoder for image observations (RGB, depth, segmentation)."""
    
    def __init__(
        self,
        input_shape: tuple,  # (C, H, W)
        latent_dim: int,
        channels: list[int] = [32, 64, 128],
        kernel_sizes: list[int] = [8, 4, 3],
        strides: list[int] = [4, 2, 1],
        activation: str = "relu",
        normalization: str = "batch_norm",  # or "layer_norm", "none"
    ):
        # Architecture: Conv layers → Global Average Pool → Linear → latent_dim
        # Handles variable input sizes through adaptive pooling
```

#### Proprioception Encoder (MLP-based)
```python
class ProprioceptionEncoder(BaseEncoder):
    """MLP encoder for proprioceptive observations."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [128, 128],
        activation: str = "relu",
        normalization: bool = True,
        dropout: float = 0.0,
    ):
        # Architecture: Linear → Norm → Activation → Dropout → ... → latent_dim
```

#### Privileged Information Encoder
```python
class PrivilegedEncoder(BaseEncoder):
    """Encoder for privileged information (terrain maps, object states, etc.)."""
    
    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        encoder_type: str = "mlp",  # "mlp", "cnn", "hybrid"
        **kwargs
    ):
        # Flexible architecture based on privileged data type
        # Can handle both spatial (terrain maps) and vector (object states) data
```

### 1.3 Multi-Modal Fusion Layer
```python
class MultiModalFusion(nn.Module):
    """Fuses multiple encoded modalities into single representation."""
    
    def __init__(
        self,
        modality_dims: dict[str, int],  # {"vision": 128, "proprio": 64, ...}
        fusion_type: str = "attention",  # "concat", "attention", "gated"
        output_dim: int = 256,
        attention_heads: int = 4,
    ):
        
    def forward(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fusion_type == "concat":
            return self._concat_fusion(modality_features)
        elif self.fusion_type == "attention":
            return self._attention_fusion(modality_features)
        elif self.fusion_type == "gated":
            return self._gated_fusion(modality_features)
    
    def _attention_fusion(self, features: dict) -> torch.Tensor:
        # Cross-attention between modalities
        # Vision attends to proprioception and vice versa
        pass
```

### 1.4 Decoder Architecture
```python
class ActionDecoder(nn.Module):
    """Decodes latent representation to actions."""
    
    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dims: list[int] = [256, 128],
        activation: str = "relu",
        output_activation: str = None,
    ):
        # Architecture: latent_dim → hidden layers → num_actions

class ValueDecoder(nn.Module):
    """Decodes latent representation to value estimate."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int] = [256, 128],
        activation: str = "relu",
    ):
        # Architecture: latent_dim → hidden layers → 1 (value)
```

### 1.5 Complete Encoder-Decoder Network
```python
class EncoderDecoderNetwork(nn.Module):
    """Complete encoder-decoder network with multi-modal support."""
    
    def __init__(
        self,
        obs_spaces: dict,  # IsaacLab observation spaces
        num_actions: int,
        encoder_configs: dict,
        fusion_config: dict,
        decoder_configs: dict,
        shared_encoder: bool = False,  # Share encoder between actor/critic
    ):
        
        # Build modality-specific encoders
        self.encoders = nn.ModuleDict()
        for modality, config in encoder_configs.items():
            if modality == "vision":
                self.encoders[modality] = VisionEncoder(**config)
            elif modality == "proprioception":
                self.encoders[modality] = ProprioceptionEncoder(**config)
            elif modality == "privileged":
                self.encoders[modality] = PrivilegedEncoder(**config)
        
        # Build fusion layer
        modality_dims = {name: enc.get_output_dim() for name, enc in self.encoders.items()}
        self.fusion = MultiModalFusion(modality_dims, **fusion_config)
        
        # Build decoders
        self.action_decoder = ActionDecoder(self.fusion.output_dim, num_actions, **decoder_configs["action"])
        self.value_decoder = ValueDecoder(self.fusion.output_dim, **decoder_configs["value"])
    
    def encode(self, obs: dict) -> torch.Tensor:
        """Encode multi-modal observations to latent representation."""
        encoded_modalities = {}
        
        for modality, encoder in self.encoders.items():
            if modality in obs:
                encoded_modalities[modality] = encoder(obs[modality])
        
        return self.fusion(encoded_modalities)
    
    def forward_actor(self, obs: dict) -> torch.Tensor:
        latent = self.encode(obs)
        return self.action_decoder(latent)
    
    def forward_critic(self, obs: dict) -> torch.Tensor:
        latent = self.encode(obs)
        return self.value_decoder(latent)
```

## 2. Enhanced Actor-Critic Architecture

### 2.1 Flexible Actor-Critic Module
```python
class FlexibleActorCritic(nn.Module):
    """Actor-Critic with configurable architectures (MLP or Encoder-Decoder)."""
    
    is_recurrent = False
    
    def __init__(
        self,
        obs_spaces: dict,
        obs_groups: dict,
        num_actions: int,
        # Architecture selection
        actor_architecture: str = "mlp",  # "mlp", "encoder_decoder"
        critic_architecture: str = "mlp", # "mlp", "encoder_decoder" 
        shared_encoder: bool = False,
        # MLP configs (backward compatibility)
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        # Encoder-Decoder configs
        encoder_configs: dict = None,
        fusion_config: dict = None,
        decoder_configs: dict = None,
        # Common configs
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        normalization_config: dict = None,
    ):
        super().__init__()
        
        self.obs_groups = obs_groups
        self.actor_architecture = actor_architecture
        self.critic_architecture = critic_architecture
        
        # Build actor
        if actor_architecture == "mlp":
            self.actor = self._build_mlp_actor()
        elif actor_architecture == "encoder_decoder":
            self.actor = self._build_encoder_decoder_actor()
        
        # Build critic  
        if critic_architecture == "mlp":
            self.critic = self._build_mlp_critic()
        elif critic_architecture == "encoder_decoder":
            if shared_encoder and actor_architecture == "encoder_decoder":
                self.critic = self._build_shared_encoder_critic()
            else:
                self.critic = self._build_encoder_decoder_critic()
        
        # Action distribution components
        self._setup_action_distribution()
        
        # Observation normalizers
        self._setup_observation_normalizers()
    
    def _build_encoder_decoder_actor(self):
        """Build encoder-decoder actor network."""
        return EncoderDecoderNetwork(
            obs_spaces=self._get_actor_obs_spaces(),
            num_actions=self.num_actions,
            encoder_configs=self.encoder_configs,
            fusion_config=self.fusion_config,
            decoder_configs=self.decoder_configs
        )
    
    def act(self, obs: dict) -> torch.Tensor:
        """Generate actions from observations."""
        if self.actor_architecture == "mlp":
            return self._act_mlp(obs)
        else:
            return self._act_encoder_decoder(obs)
    
    def evaluate(self, obs: dict) -> torch.Tensor:
        """Evaluate state value from observations."""
        if self.critic_architecture == "mlp":
            return self._evaluate_mlp(obs)
        else:
            return self._evaluate_encoder_decoder(obs)
```

### 2.2 Observation Processing Pipeline
```python
class ObservationProcessor:
    """Handles observation preprocessing and filtering for different architectures."""
    
    def __init__(
        self,
        obs_spaces: dict,
        obs_groups: dict,
        normalization_configs: dict,
        privileged_keys: list[str] = None,
    ):
        self.obs_spaces = obs_spaces
        self.obs_groups = obs_groups
        self.privileged_keys = privileged_keys or []
        
        # Build normalizers per modality
        self.normalizers = {}
        for modality, config in normalization_configs.items():
            if config["enabled"]:
                self.normalizers[modality] = EmpiricalNormalization(
                    config["dim"], **config.get("params", {})
                )
    
    def process_observations(
        self, 
        raw_obs: dict, 
        obs_group: str,
        include_privileged: bool = True
    ) -> dict:
        """Process and filter observations for specific group."""
        
        processed_obs = {}
        target_obs_keys = self.obs_groups[obs_group]
        
        for obs_key in target_obs_keys:
            if obs_key in raw_obs:
                # Skip privileged observations for student
                if not include_privileged and obs_key in self.privileged_keys:
                    continue
                
                # Apply normalization if configured
                obs_data = raw_obs[obs_key]
                modality = self._get_modality_type(obs_key)
                
                if modality in self.normalizers:
                    obs_data = self.normalizers[modality](obs_data)
                
                processed_obs[obs_key] = obs_data
        
        return processed_obs
    
    def _get_modality_type(self, obs_key: str) -> str:
        """Determine modality type from observation key."""
        if "camera" in obs_key or "rgb" in obs_key or "depth" in obs_key:
            return "vision"
        elif "terrain" in obs_key or "object_state" in obs_key:
            return "privileged"
        else:
            return "proprioception"
```

## 3. Multi-Teacher Student Distillation Architecture

### 3.1 Teacher Management System
```python
class TeacherManager:
    """Manages multiple teacher policies with different architectures."""
    
    def __init__(
        self,
        teacher_configs: list[dict],
        device: str = "cuda"
    ):
        self.teachers = {}
        self.teacher_weights = {}
        self.teacher_specializations = {}
        
        for config in teacher_configs:
            teacher_name = config["name"]
            teacher_path = config["path"]
            
            # Load teacher policy (could be MLP or Encoder-Decoder)
            teacher_policy = self._load_teacher_policy(teacher_path, config)
            teacher_policy.eval()  # Set to evaluation mode
            
            self.teachers[teacher_name] = teacher_policy
            self.teacher_weights[teacher_name] = config.get("weight", 1.0)
            self.teacher_specializations[teacher_name] = config.get("specialization", "general")
    
    def get_teacher_actions(
        self, 
        obs: dict, 
        active_teachers: list[str] = None
    ) -> dict[str, torch.Tensor]:
        """Get actions from all or specified teachers."""
        
        if active_teachers is None:
            active_teachers = list(self.teachers.keys())
        
        teacher_actions = {}
        with torch.no_grad():
            for teacher_name in active_teachers:
                if teacher_name in self.teachers:
                    # Process observations for this teacher's architecture
                    teacher_obs = self._process_obs_for_teacher(obs, teacher_name)
                    teacher_actions[teacher_name] = self.teachers[teacher_name].act_inference(teacher_obs)
        
        return teacher_actions
    
    def aggregate_teacher_actions(
        self,
        teacher_actions: dict[str, torch.Tensor],
        aggregation_method: str = "weighted_average",
        dynamic_weights: dict[str, float] = None
    ) -> torch.Tensor:
        """Aggregate multiple teacher actions into single target."""
        
        if aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(teacher_actions, dynamic_weights)
        elif aggregation_method == "attention_weighted":
            return self._attention_weighted_aggregation(teacher_actions)
        elif aggregation_method == "performance_weighted":
            return self._performance_weighted_aggregation(teacher_actions)
```

### 3.2 Multi-Teacher Student Policy
```python
class MultiTeacherStudent(nn.Module):
    """Student policy that learns from multiple teachers."""
    
    is_recurrent = False
    
    def __init__(
        self,
        obs_spaces: dict,
        obs_groups: dict,
        num_actions: int,
        student_architecture: str = "encoder_decoder",
        teacher_manager: TeacherManager = None,
        # Student network configs
        **student_configs
    ):
        super().__init__()
        
        self.obs_groups = obs_groups
        self.teacher_manager = teacher_manager
        self.student_architecture = student_architecture
        
        # Build student network
        if student_architecture == "encoder_decoder":
            self.student = EncoderDecoderNetwork(
                obs_spaces=obs_spaces,
                num_actions=num_actions,
                **student_configs
            )
        else:  # MLP fallback
            self.student = MLP(...)
        
        # Action distribution setup
        self._setup_action_distribution()
        
        # Observation processor (excludes privileged information)
        self.obs_processor = ObservationProcessor(
            obs_spaces=obs_spaces,
            obs_groups=obs_groups,
            privileged_keys=self._get_privileged_keys(),
            normalization_configs=student_configs.get("normalization", {})
        )
    
    def act(self, obs: dict) -> torch.Tensor:
        """Student action generation (with noise for exploration)."""
        student_obs = self.obs_processor.process_observations(
            obs, "policy", include_privileged=False
        )
        
        mean = self.student.forward_actor(student_obs)
        self.update_distribution(mean)
        return self.distribution.sample()
    
    def act_inference(self, obs: dict) -> torch.Tensor:
        """Student inference (deterministic)."""
        student_obs = self.obs_processor.process_observations(
            obs, "policy", include_privileged=False
        )
        return self.student.forward_actor(student_obs)
    
    def get_teacher_targets(self, obs: dict) -> torch.Tensor:
        """Get aggregated teacher actions as distillation targets."""
        if self.teacher_manager is None:
            raise ValueError("No teacher manager configured")
        
        teacher_actions = self.teacher_manager.get_teacher_actions(obs)
        return self.teacher_manager.aggregate_teacher_actions(teacher_actions)
    
    def evaluate(self, obs: dict) -> torch.Tensor:
        """Student value estimation."""
        student_obs = self.obs_processor.process_observations(
            obs, "critic", include_privileged=False
        )
        return self.student.forward_critic(student_obs)
```

### 3.3 Enhanced Distillation Algorithm
```python
class MultiTeacherDistillation:
    """Enhanced distillation algorithm supporting multiple teachers."""
    
    def __init__(
        self,
        student_policy: MultiTeacherStudent,
        teacher_manager: TeacherManager,
        # Distillation parameters
        num_learning_epochs: int = 5,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        # Loss function configs
        behavior_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.1,
        diversity_loss_weight: float = 0.05,
        # Curriculum learning
        curriculum_config: dict = None,
        device: str = "cuda"
    ):
        self.student_policy = student_policy
        self.teacher_manager = teacher_manager
        self.curriculum_config = curriculum_config
        
        # Initialize curriculum if configured
        if curriculum_config:
            self.curriculum_scheduler = CurriculumScheduler(curriculum_config)
        
        # Loss functions
        self.behavior_loss_fn = nn.MSELoss()
        self.consistency_loss_fn = nn.MSELoss()
    
    def update(self, rollout_data):
        """Enhanced update with multi-teacher distillation."""
        total_loss = 0
        num_updates = 0
        
        # Get current curriculum stage if enabled
        if hasattr(self, 'curriculum_scheduler'):
            curriculum_stage = self.curriculum_scheduler.get_current_stage()
            active_teachers = curriculum_stage["active_teachers"]
            teacher_weights = curriculum_stage["teacher_weights"]
        else:
            active_teachers = list(self.teacher_manager.teachers.keys())
            teacher_weights = None
        
        for epoch in range(self.num_learning_epochs):
            for obs_batch, action_batch in rollout_data:
                
                # Get student predictions
                student_actions = self.student_policy.act_inference(obs_batch)
                
                # Get teacher targets (aggregated)
                teacher_actions = self.teacher_manager.get_teacher_actions(
                    obs_batch, active_teachers
                )
                aggregated_teacher_actions = self.teacher_manager.aggregate_teacher_actions(
                    teacher_actions, dynamic_weights=teacher_weights
                )
                
                # Compute losses
                behavior_loss = self.behavior_loss_fn(student_actions, aggregated_teacher_actions)
                
                # Teacher consistency loss (encourages agreement between teachers)
                consistency_loss = self._compute_teacher_consistency_loss(teacher_actions)
                
                # Total loss
                loss = (
                    self.behavior_loss_weight * behavior_loss +
                    self.consistency_loss_weight * consistency_loss
                )
                
                total_loss += loss.item()
                num_updates += 1
                
                # Gradient step
                if num_updates % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        
        return {"total_loss": total_loss / num_updates}
    
    def _compute_teacher_consistency_loss(self, teacher_actions: dict) -> torch.Tensor:
        """Compute consistency loss between teachers."""
        if len(teacher_actions) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Pairwise consistency between all teachers
        consistency_losses = []
        teacher_names = list(teacher_actions.keys())
        
        for i in range(len(teacher_names)):
            for j in range(i + 1, len(teacher_names)):
                teacher1_actions = teacher_actions[teacher_names[i]]
                teacher2_actions = teacher_actions[teacher_names[j]]
                consistency_losses.append(
                    self.consistency_loss_fn(teacher1_actions, teacher2_actions)
                )
        
        return torch.stack(consistency_losses).mean()
```

## 4. Curriculum Learning System
```python
class CurriculumScheduler:
    """Manages curriculum learning progression."""
    
    def __init__(self, curriculum_config: dict):
        self.difficulty_levels = curriculum_config["difficulty_levels"]
        self.teacher_weights_per_level = curriculum_config["teacher_weights_per_level"]
        self.teacher_names = curriculum_config["teacher_names"]
        self.progression_metric = curriculum_config.get("progression_metric", "success_rate")
        self.progression_threshold = curriculum_config.get("progression_threshold", 0.8)
        
        self.current_level = 0
        self.level_performance_history = []
    
    def get_current_stage(self) -> dict:
        """Get current curriculum stage configuration."""
        current_weights = self.teacher_weights_per_level[self.current_level]
        active_teachers = [
            name for name, weight in zip(self.teacher_names, current_weights) 
            if weight > 0
        ]
        
        return {
            "difficulty_level": self.difficulty_levels[self.current_level],
            "active_teachers": active_teachers,
            "teacher_weights": dict(zip(self.teacher_names, current_weights))
        }
    
    def update_progress(self, performance_metrics: dict):
        """Update curriculum progression based on performance."""
        current_performance = performance_metrics.get(self.progression_metric, 0.0)
        self.level_performance_history.append(current_performance)
        
        # Check if ready to progress (based on recent performance)
        if len(self.level_performance_history) >= 10:  # Minimum episodes
            recent_performance = np.mean(self.level_performance_history[-10:])
            
            if (recent_performance >= self.progression_threshold and 
                self.current_level < len(self.difficulty_levels) - 1):
                
                self.current_level += 1
                self.level_performance_history = []  # Reset for new level
                print(f"Curriculum advanced to level {self.current_level}")
```

## 5. Policy Factory System

### 5.1 Policy Factory
```python
class PolicyFactory:
    """Central factory for creating different policy architectures."""
    
    @staticmethod
    def create_policy(
        policy_config: dict,
        obs_spaces: dict,
        obs_groups: dict,
        num_actions: int,
        device: str = "cuda"
    ) -> nn.Module:
        """Create policy based on configuration."""
        
        policy_type = policy_config["type"]
        
        if policy_type == "actor_critic":
            return PolicyFactory._create_actor_critic(
                policy_config, obs_spaces, obs_groups, num_actions, device
            )
        elif policy_type == "encoder_decoder_actor_critic":
            return PolicyFactory._create_encoder_decoder_actor_critic(
                policy_config, obs_spaces, obs_groups, num_actions, device
            )
        elif policy_type == "multi_teacher_student":
            return PolicyFactory._create_multi_teacher_student(
                policy_config, obs_spaces, obs_groups, num_actions, device
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    @staticmethod
    def _create_actor_critic(config, obs_spaces, obs_groups, num_actions, device):
        """Create standard MLP Actor-Critic (backward compatibility)."""
        from rsl_rl.modules import ActorCritic
        
        return ActorCritic(
            obs=obs_spaces,
            obs_groups=obs_groups,
            num_actions=num_actions,
            **config.get("params", {})
        ).to(device)
    
    @staticmethod
    def _create_encoder_decoder_actor_critic(config, obs_spaces, obs_groups, num_actions, device):
        """Create Encoder-Decoder Actor-Critic."""
        return FlexibleActorCritic(
            obs_spaces=obs_spaces,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_architecture="encoder_decoder",
            critic_architecture=config.get("critic_architecture", "encoder_decoder"),
            shared_encoder=config.get("shared_encoder", False),
            encoder_configs=config["encoder"],
            fusion_config=config["fusion"],
            decoder_configs=config["decoder"],
            **config.get("params", {})
        ).to(device)
    
    @staticmethod  
    def _create_multi_teacher_student(config, obs_spaces, obs_groups, num_actions, device):
        """Create Multi-Teacher Student policy."""
        
        # Create teacher manager
        teacher_manager = TeacherManager(
            teacher_configs=config["teachers"],
            device=device
        )
        
        # Create student policy
        student_policy = MultiTeacherStudent(
            obs_spaces=obs_spaces,
            obs_groups=obs_groups,
            num_actions=num_actions,
            student_architecture=config["student"]["type"],
            teacher_manager=teacher_manager,
            **config["student"].get("params", {})
        ).to(device)
        
        return student_policy

class AlgorithmFactory:
    """Factory for creating training algorithms."""
    
    @staticmethod
    def create_algorithm(
        algorithm_config: dict,
        policy: nn.Module,
        device: str = "cuda"
    ):
        """Create training algorithm based on configuration."""
        
        algorithm_type = algorithm_config["type"]
        
        if algorithm_type == "ppo":
            from rsl_rl.algorithms import PPO
            return PPO(
                policy=policy,
                device=device,
                **algorithm_config.get("params", {})
            )
        elif algorithm_type == "multi_teacher_distillation":
            return MultiTeacherDistillation(
                student_policy=policy.student if hasattr(policy, 'student') else policy,
                teacher_manager=policy.teacher_manager if hasattr(policy, 'teacher_manager') else None,
                device=device,
                **algorithm_config.get("params", {})
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
```

### 5.2 Configuration Validator
```python
class ConfigValidator:
    """Validates configuration files for correctness."""
    
    @staticmethod
    def validate_policy_config(config: dict) -> tuple[bool, list[str]]:
        """Validate policy configuration."""
        errors = []
        
        # Check required fields
        if "type" not in config:
            errors.append("Missing required field: 'type'")
        
        policy_type = config.get("type")
        
        if policy_type == "encoder_decoder_actor_critic":
            errors.extend(ConfigValidator._validate_encoder_decoder_config(config))
        elif policy_type == "multi_teacher_student":
            errors.extend(ConfigValidator._validate_multi_teacher_config(config))
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_encoder_decoder_config(config: dict) -> list[str]:
        """Validate encoder-decoder specific configuration."""
        errors = []
        
        if "encoder" not in config:
            errors.append("Missing 'encoder' configuration for encoder_decoder_actor_critic")
        else:
            encoder_config = config["encoder"]
            if "modalities" not in encoder_config:
                errors.append("Missing 'modalities' in encoder configuration")
        
        if "fusion" not in config:
            errors.append("Missing 'fusion' configuration")
        
        return errors
    
    @staticmethod
    def _validate_multi_teacher_config(config: dict) -> list[str]:
        """Validate multi-teacher specific configuration."""
        errors = []
        
        if "teachers" not in config:
            errors.append("Missing 'teachers' configuration")
        elif not isinstance(config["teachers"], list) or len(config["teachers"]) == 0:
            errors.append("'teachers' must be a non-empty list")
        
        if "student" not in config:
            errors.append("Missing 'student' configuration")
        
        return errors
```

## 6. Integration with Existing PPO

### 6.1 PPO Integration Strategy
The existing PPO algorithm in `rsl_rl/algorithms/ppo.py` already uses a policy interface that our new architectures can implement. Key integration points:

**Policy Interface Compatibility:**
```python
# All our new policies must implement these methods:
class PolicyInterface:
    def act(self, obs) -> torch.Tensor:
        """Generate actions for exploration (with noise)."""
    
    def act_inference(self, obs) -> torch.Tensor:
        """Generate actions for inference (deterministic)."""
    
    def evaluate(self, obs) -> torch.Tensor:
        """Evaluate state value."""
    
    def get_actions_log_prob(self, actions) -> torch.Tensor:
        """Get log probabilities of actions."""
    
    def update_normalization(self, obs):
        """Update observation normalizers."""
    
    def reset(self, dones):
        """Reset policy state for done environments."""
    
    @property
    def is_recurrent(self) -> bool:
        """Whether policy maintains internal state."""
```

**Observation Handling:**
```python
class PPOObservationAdapter:
    """Adapts observations between different formats for PPO compatibility."""
    
    def __init__(self, policy_type: str, obs_config: dict):
        self.policy_type = policy_type
        self.obs_config = obs_config
    
    def adapt_observations(self, raw_obs):
        """Convert IsaacLab observations to policy-expected format."""
        if self.policy_type in ["encoder_decoder_actor_critic", "multi_teacher_student"]:
            # Keep as dictionary for multi-modal processing
            return self._process_multi_modal_obs(raw_obs)
        else:
            # Convert to flattened tensor for MLP policies
            return self._flatten_observations(raw_obs)
```

### 6.2 Rollout Storage Compatibility
```python
class EnhancedRolloutStorage(RolloutStorage):
    """Enhanced rollout storage supporting multi-modal observations."""
    
    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs_spaces: dict,  # Now supports dict of observation spaces
        actions_shape: tuple,
        device: str,
        obs_format: str = "dict"  # "dict" or "tensor"
    ):
        self.obs_format = obs_format
        super().__init__(training_type, num_envs, num_transitions_per_env, 
                        obs_spaces, actions_shape, device)
    
    def add_transitions(self, transition):
        """Add transitions with support for multi-modal observations."""
        if self.obs_format == "dict":
            # Store each modality separately
            for modality, obs_data in transition.observations.items():
                if modality not in self.observations:
                    self.observations[modality] = torch.zeros(
                        (self.num_transitions_per_env, self.num_envs) + obs_data.shape[1:],
                        dtype=obs_data.dtype, device=self.device
                    )
                self.observations[modality][self.step] = obs_data
        else:
            # Standard tensor storage
            self.observations[self.step] = transition.observations
        
        # Store other transition components normally
        super().add_transitions(transition)
```

## 7. Complete Class Hierarchy

### 7.1 Core Architecture Hierarchy
```
nn.Module
├── BaseEncoder (Abstract)
│   ├── VisionEncoder (CNN-based)
│   ├── ProprioceptionEncoder (MLP-based)
│   └── PrivilegedEncoder (Flexible)
│
├── MultiModalFusion
│   ├── ConcatFusion
│   ├── AttentionFusion
│   └── GatedFusion
│
├── ActionDecoder
├── ValueDecoder
├── EncoderDecoderNetwork
│
├── ActorCritic (Existing - MLP only)
├── FlexibleActorCritic (New - MLP or Encoder-Decoder)
└── MultiTeacherStudent (New - Multi-teacher distillation)
```

### 7.2 Algorithm Hierarchy
```
Algorithm Classes:
├── PPO (Existing - works with any policy implementing interface)
├── Distillation (Existing - single teacher)
└── MultiTeacherDistillation (New - multiple teachers)

Support Classes:
├── TeacherManager (Manages multiple teacher policies)
├── ObservationProcessor (Handles multi-modal observation processing)
├── CurriculumScheduler (Manages curriculum learning progression)
├── PolicyFactory (Creates policies from configuration)
├── AlgorithmFactory (Creates algorithms from configuration)
└── ConfigValidator (Validates configuration files)
```

### 7.3 Key Interfaces
```python
# Policy Interface (implemented by all policies)
class PolicyInterface(Protocol):
    is_recurrent: bool
    
    def act(self, obs) -> torch.Tensor: ...
    def act_inference(self, obs) -> torch.Tensor: ...
    def evaluate(self, obs) -> torch.Tensor: ...
    def get_actions_log_prob(self, actions) -> torch.Tensor: ...
    def update_normalization(self, obs): ...
    def reset(self, dones): ...

# Encoder Interface (implemented by all encoders)
class EncoderInterface(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def get_output_dim(self) -> int: ...

# Teacher Manager Interface
class TeacherManagerInterface(Protocol):
    def get_teacher_actions(self, obs, active_teachers=None) -> dict[str, torch.Tensor]: ...
    def aggregate_teacher_actions(self, teacher_actions, **kwargs) -> torch.Tensor: ...
```

## 8. File Structure and Organization

```
rsl_rl/
├── algorithms/
│   ├── ppo.py (existing - no changes needed)
│   ├── distillation.py (existing - minimal changes)
│   └── multi_teacher_distillation.py (new)
│
├── modules/
│   ├── actor_critic.py (existing - no changes)
│   ├── actor_critic_encoder_decoder.py (new)
│   ├── multi_teacher_student.py (new)
│   └── __init__.py (updated imports)
│
├── networks/
│   ├── mlp.py (existing)
│   ├── encoder_decoder.py (new)
│   ├── encoders.py (new - specific encoder implementations)
│   ├── decoders.py (new - specific decoder implementations)
│   ├── fusion.py (new - multi-modal fusion layers)
│   └── __init__.py (updated imports)
│
├── utils/
│   ├── policy_factory.py (new)
│   ├── observation_processor.py (new)
│   ├── config_validator.py (new)
│   └── curriculum_scheduler.py (new)
│
└── storage/
    └── enhanced_rollout_storage.py (new - multi-modal support)
```

## 9. IsaacLab Compatibility Architecture

### 9.1 Compatibility Strategy
**Important**: All enhancements stay within RSL-RL codebase - no IsaacLab modifications required.

Our architecture ensures compatibility with IsaacLab's integration patterns:

**Key Compatibility Requirements:**
- Policies must implement the standard RSL-RL interface used by IsaacLab  
- Support existing `obs_groups` mapping mechanism from IsaacLab configs
- Work with `RslRlVecEnvWrapper` without modifications
- Allow configuration through standard Python config files (not requiring IsaacLab's `@configclass`)
- Maintain backward compatibility with existing RSL-RL workflows

### 9.2 RSL-RL Native Configuration System

**Enhanced RSL-RL Policy Configurations (no IsaacLab dependencies):**
```python
# rsl_rl/config/enhanced_configs.py
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List, Any

@dataclass
class EncoderDecoderConfig:
    """Configuration for Encoder-Decoder Actor-Critic networks."""
    
    # Architecture selection
    actor_architecture: str = "encoder_decoder"  # "mlp" or "encoder_decoder"
    critic_architecture: str = "encoder_decoder" 
    shared_encoder: bool = False
    
    # Standard parameters (backward compatibility)
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"  # "scalar" or "log"
    activation: str = "elu"
    
    # MLP parameters (fallback compatibility)
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    
    # Encoder-Decoder specific parameters
    encoder_config: Optional[Dict[str, Any]] = None
    fusion_config: Optional[Dict[str, Any]] = None
    decoder_config: Optional[Dict[str, Any]] = None

@dataclass
class MultiTeacherStudentConfig:
    """Configuration for Multi-Teacher Student distillation."""
    
    # Student configuration
    student_architecture: str = "encoder_decoder"
    student_config: Dict[str, Any] = None
    
    # Teacher configuration
    teachers: List[Dict[str, Any]] = None
    
    # Distillation parameters
    aggregation_method: str = "weighted_average"
    
    # Standard parameters
    init_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    activation: str = "elu"

@dataclass
class MultiTeacherDistillationConfig:
    """Configuration for Multi-Teacher Distillation algorithm."""
    
    # Distillation parameters
    num_learning_epochs: int = 3
    gradient_length: int = 15
    learning_rate: float = 1e-3
    
    # Loss weights
    behavior_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.1
    diversity_loss_weight: float = 0.05
    
    # Curriculum learning
    curriculum_config: Optional[Dict[str, Any]] = None
    
    # Standard parameters
    max_grad_norm: Optional[float] = None
    optimizer: str = "adam"
    loss_type: str = "mse"
```

### 9.3 IsaacLab Integration Examples (User-Side)

**How users can integrate with IsaacLab (no RSL-RL changes needed):**

**Example 1: Encoder-Decoder PPO with IsaacLab:**
```python
# anymal_c_encoder_decoder_ppo_cfg.py - IsaacLab config file
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class AnymalCEncoderDecoderPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_c_encoder_decoder"
    
    # Use our enhanced actor-critic with custom class_name
    policy = RslRlPpoActorCriticCfg(
        class_name="FlexibleActorCritic",  # Points to our enhanced policy
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        # Pass encoder-decoder config through kwargs
        **{
            "actor_architecture": "encoder_decoder",
            "critic_architecture": "encoder_decoder",
            "encoder_config": {
                "modalities": {
                    "proprioception": {
                        "type": "mlp",
                        "input_dim": 48,
                        "latent_dim": 64,
                        "normalization": True
                    },
                    "vision": {
                        "type": "cnn", 
                        "input_shape": [3, 64, 64],
                        "latent_dim": 128
                    }
                }
            },
            "fusion_config": {
                "type": "attention",
                "output_dim": 256
            }
        }
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# Standard IsaacLab obs_groups remain unchanged
obs_groups = {
    "policy": ["robot_state", "front_camera"],  # Actor gets proprio + vision  
    "critic": ["robot_state", "front_camera", "privileged_info"],  # Critic gets all
}
```

**Example 2: Multi-Teacher Distillation with IsaacLab:**
```python
# anymal_c_multi_teacher_distillation_cfg.py - IsaacLab config file
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg

@configclass
class AnymalCMultiTeacherDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "anymal_c_multi_teacher_distillation"
    
    # Use our multi-teacher policy
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="MultiTeacherStudent",  # Points to our multi-teacher implementation
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        activation="elu",
        # Pass multi-teacher config through kwargs
        **{
            "student_architecture": "encoder_decoder",
            "aggregation_method": "weighted_average",
            "teachers": [
                {
                    "name": "locomotion_expert",
                    "path": "teachers/locomotion_teacher.pt",
                    "type": "encoder_decoder",
                    "weight": 0.4
                },
                {
                    "name": "terrain_expert", 
                    "path": "teachers/terrain_teacher.pt",
                    "type": "mlp",
                    "weight": 0.6
                }
            ],
            "student_config": {
                "encoder_config": {
                    "modalities": {
                        "proprioception": {"type": "mlp", "latent_dim": 64},
                        "vision": {"type": "cnn", "latent_dim": 64}
                    }
                },
                "fusion_config": {"type": "concat", "output_dim": 128}
            }
        }
    )
    
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="MultiTeacherDistillation",  # Points to our enhanced algorithm
        num_learning_epochs=3,
        gradient_length=10,
        learning_rate=1e-3,
        **{
            "behavior_loss_weight": 1.0,
            "consistency_loss_weight": 0.1,
            "curriculum_config": {
                "enabled": True,
                "teacher_weights_per_level": [[1.0, 0.0], [0.4, 0.6]]
            }
        }
    )
```

### 9.4 RSL-RL Class Registration System

**Dynamic class loading for IsaacLab compatibility:**
```python
# rsl_rl/modules/__init__.py - Enhanced with new classes
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent  
from .flexible_actor_critic import FlexibleActorCritic  # NEW
from .multi_teacher_student import MultiTeacherStudent  # NEW
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

# rsl_rl/algorithms/__init__.py - Enhanced with new algorithms  
from .ppo import PPO
from .distillation import Distillation
from .multi_teacher_distillation import MultiTeacherDistillation  # NEW

# Class registry for IsaacLab dynamic loading
POLICY_REGISTRY = {
    "ActorCritic": ActorCritic,
    "ActorCriticRecurrent": ActorCriticRecurrent,
    "FlexibleActorCritic": FlexibleActorCritic,  # NEW - supports encoder-decoder
    "MultiTeacherStudent": MultiTeacherStudent,  # NEW - multi-teacher distillation
    "StudentTeacher": StudentTeacher,
    "StudentTeacherRecurrent": StudentTeacherRecurrent,
}

ALGORITHM_REGISTRY = {
    "PPO": PPO,
    "Distillation": Distillation, 
    "MultiTeacherDistillation": MultiTeacherDistillation,  # NEW
}
```

### 9.5 Standard Training Script Usage

**Direct RSL-RL usage (no IsaacLab):**
```python
# train_direct_rsl_rl.py
from rsl_rl.algorithms import PPO
from rsl_rl.modules import FlexibleActorCritic
from rsl_rl.env import VecEnv

# Create policy with encoder-decoder
policy_config = {
    "actor_architecture": "encoder_decoder",
    "critic_architecture": "encoder_decoder",
    "encoder_config": {
        "modalities": {
            "proprioception": {"type": "mlp", "latent_dim": 64},
            "vision": {"type": "cnn", "latent_dim": 128}
        }
    },
    "fusion_config": {"type": "attention", "output_dim": 256}
}

policy = FlexibleActorCritic(
    obs=env.observation_space,
    obs_groups={"policy": ["proprio", "vision"], "critic": ["proprio", "vision", "privileged"]},
    num_actions=env.action_space.shape[0],
    **policy_config
)

# Use standard PPO (works unchanged)
algorithm = PPO(policy=policy)

# Standard training loop
algorithm.init_storage(...)
for iteration in range(max_iterations):
    # ... standard PPO training loop
```

### 9.6 Key Compatibility Points

**1. Interface Compatibility:**
- All new policies implement the standard RSL-RL policy interface
- Existing PPO algorithm works with new policies without modification
- Observation processing is backward compatible

**2. Configuration Compatibility:**
- IsaacLab can specify new policies via `class_name` parameter
- Additional configuration passed through `**kwargs` 
- Existing observation group mappings remain unchanged

**3. Environment Compatibility:**
- Works with any RSL-RL compatible environment
- No changes needed to `RslRlVecEnvWrapper`
- Standard observation and action interfaces maintained

**4. Export Compatibility:**  
- Policy export functions (`export_policy_as_jit`, `export_policy_as_onnx`) work unchanged
- Trained policies can be deployed using existing IsaacLab deployment workflows

## 10. RSL-RL Design Principles Adherence

### 10.1 Existing RSL-RL Design Patterns Analysis

**Key Design Principles Observed:**

**1. Module Structure:**
- Policies inherit from `nn.Module`
- Class attribute `is_recurrent = False/True` for recurrency indication
- Clear separation: modules (policies), algorithms (training), networks (building blocks)
- Algorithms are standalone classes, not inheriting from `nn.Module`

**2. Configuration Patterns:**
- Parameters passed directly to `__init__` with defaults
- `**kwargs` pattern with warning for unexpected arguments (see ActorCritic:32-36)
- Type hints using `| None` and modern Python syntax
- Device handling via string parameters

**3. Interface Consistency:**
- Standard methods: `act()`, `evaluate()`, `reset()`, `update_normalization()`
- Property methods for distribution access: `@property action_mean`, `@property action_std`
- Observation processing via `get_actor_obs()`, `get_critic_obs()` methods
- Clear separation between training (`act()`) and inference (`act_inference()`)

**4. Observation Handling:**
- `obs_groups` dictionary mapping algorithm needs to environment observations
- Flattening multi-group observations via `torch.cat(obs_list, dim=-1)`
- Support for both TensorDict and regular tensor observations
- Assertion checks for supported observation shapes

**5. Utility Functions:**
- Resolver pattern: `resolve_nn_activation()`, `resolve_optimizer()`
- String-to-callable conversion: `string_to_callable()`
- Helper functions follow snake_case naming

### 10.2 Architecture Alignment with RSL-RL Principles

**Updated Architecture Following RSL-RL Conventions:**

```python
# rsl_rl/modules/flexible_actor_critic.py
class FlexibleActorCritic(nn.Module):
    """Actor-Critic with configurable architectures (MLP or Encoder-Decoder)."""
    
    is_recurrent = False  # RSL-RL convention
    
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
        critic_architecture="mlp",
        shared_encoder=False,
        encoder_config=None,
        fusion_config=None,
        decoder_config=None,
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
        
        # Observation dimension calculation (RSL-RL pattern)
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if actor_architecture == "mlp":
                assert len(obs[obs_group].shape) == 2, "MLP mode requires 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        
        # Build networks using RSL-RL patterns
        if actor_architecture == "mlp":
            from rsl_rl.networks import MLP
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        else:
            self.actor = self._build_encoder_decoder_actor(obs, encoder_config, fusion_config, decoder_config)
        
        # ... similar for critic
        
        # Action distribution (follow existing pattern exactly)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # Follow RSL-RL normalization pattern exactly
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            from rsl_rl.networks import EmpiricalNormalization
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        
        # Distribution setup (follow existing pattern)
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    # Standard RSL-RL interface methods (keep signatures identical)
    def act(self, obs, **kwargs):
        """Follow exact RSL-RL pattern."""
        if self.actor_architecture == "mlp":
            obs = self.get_actor_obs(obs)
            obs = self.actor_obs_normalizer(obs)
        else:
            obs = self._process_multimodal_obs(obs, "policy")
        
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def act_inference(self, obs):
        """Follow exact RSL-RL pattern."""
        if self.actor_architecture == "mlp":
            obs = self.get_actor_obs(obs)
            obs = self.actor_obs_normalizer(obs)
            return self.actor(obs)
        else:
            obs = self._process_multimodal_obs(obs, "policy")
            return self.actor.forward_actor(obs)
    
    def evaluate(self, obs, **kwargs):
        """Follow exact RSL-RL pattern."""
        if self.critic_architecture == "mlp":
            obs = self.get_critic_obs(obs)
            obs = self.critic_obs_normalizer(obs)
            return self.critic(obs)
        else:
            obs = self._process_multimodal_obs(obs, "critic")
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
    
    # Properties follow exact RSL-RL pattern
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
        """Follow exact RSL-RL pattern."""
        if self.actor_obs_normalization and self.actor_architecture == "mlp":
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        # Enhanced for multi-modal observations
        elif self.actor_architecture == "encoder_decoder":
            self._update_multimodal_normalization(obs, "policy")
    
    def reset(self, dones=None):
        """Follow exact RSL-RL pattern."""
        pass
    
    def load_state_dict(self, state_dict, strict=True):
        """Follow exact RSL-RL pattern with resume indication."""
        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
```

### 10.3 Network Building Patterns

**Following RSL-RL utility patterns:**

```python
# rsl_rl/utils/enhanced_utils.py - Add to existing utils
def resolve_encoder_type(encoder_type: str, **config):
    """Follow RSL-RL resolver pattern."""
    if encoder_type == "mlp":
        from rsl_rl.networks.encoders import ProprioceptionEncoder
        return ProprioceptionEncoder(**config)
    elif encoder_type == "cnn":
        from rsl_rl.networks.encoders import VisionEncoder
        return VisionEncoder(**config)
    elif encoder_type == "hybrid":
        from rsl_rl.networks.encoders import HybridEncoder
        return HybridEncoder(**config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def resolve_fusion_type(fusion_type: str, **config):
    """Follow RSL-RL resolver pattern."""
    if fusion_type == "concat":
        from rsl_rl.networks.fusion import ConcatFusion
        return ConcatFusion(**config)
    elif fusion_type == "attention":
        from rsl_rl.networks.fusion import AttentionFusion
        return AttentionFusion(**config)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
```

### 10.4 Algorithm Integration Pattern

**Multi-Teacher Distillation following RSL-RL algorithm pattern:**

```python
# rsl_rl/algorithms/multi_teacher_distillation.py
class MultiTeacherDistillation:
    """Multi-Teacher Distillation algorithm."""
    
    policy: MultiTeacherStudent  # Type annotation following RSL-RL pattern
    """The multi-teacher student policy."""
    
    def __init__(
        self,
        policy,
        # Standard distillation parameters (maintain existing order)
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        # Multi-teacher specific parameters (added at end)
        behavior_loss_weight=1.0,
        consistency_loss_weight=0.1,
        curriculum_config=None,
        # Distributed training (follow RSL-RL pattern)
        multi_gpu_cfg=None,
    ):
        # Follow exact RSL-RL device and multi-GPU setup pattern
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        
        # Follow RSL-RL optimizer resolution pattern
        from rsl_rl.utils import resolve_optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)
        
        # Standard RSL-RL storage and transition pattern
        self.storage = None
        self.transition = RolloutStorage.Transition()
```

### 10.5 Error Handling and Validation

**Following RSL-RL validation patterns:**

```python
def _validate_encoder_config(self, obs, obs_groups, encoder_config):
    """Follow RSL-RL assertion pattern for validation."""
    if encoder_config is None:
        raise ValueError("encoder_config is required for encoder_decoder architecture")
    
    for obs_group in obs_groups["policy"]:
        if obs_group not in obs:
            raise ValueError(f"Observation group '{obs_group}' not found in observations")
        
        # Validate observation shapes based on modality
        obs_shape = obs[obs_group].shape
        modality = self._infer_modality_from_obs_group(obs_group)
        
        if modality == "vision":
            assert len(obs_shape) == 4, f"Vision observations must be 4D (B,C,H,W), got {obs_shape}"
        elif modality == "proprioception":
            assert len(obs_shape) == 2, f"Proprioception observations must be 1D, got {obs_shape}"
```

### 10.6 Import and Module Organization

**Following RSL-RL import patterns:**

```python
# rsl_rl/networks/__init__.py - Enhanced following existing pattern
from .mlp import MLP
from .memory import Memory
from .normalization import EmpiricalNormalization
# New additions following same pattern
from .encoders import VisionEncoder, ProprioceptionEncoder, HybridEncoder
from .fusion import ConcatFusion, AttentionFusion, GatedFusion
from .decoders import ActionDecoder, ValueDecoder

# rsl_rl/modules/__init__.py - Enhanced following existing pattern
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
# New additions following same pattern
from .flexible_actor_critic import FlexibleActorCritic
from .multi_teacher_student import MultiTeacherStudent
```

---

# PHASE 1: MULTI-TEACHER STUDENT DISTILLATION (PRIORITY)

## Implementation Plan - MLP-Based Multi-Teacher System

### Phase 1.1: Teacher Management System (Week 1)

**Core Components:**

**1. Teacher Manager (`rsl_rl/modules/teacher_manager.py`)**
```python
class TeacherManager:
    """Manages multiple MLP teacher policies for distillation."""
    
    def __init__(
        self,
        teacher_configs: list[dict],
        device: str = "cuda",
    ):
        self.teachers = {}
        self.teacher_weights = {}
        self.teacher_specializations = {}
        self.device = device
        
        for config in teacher_configs:
            self._load_teacher(config)
    
    def _load_teacher(self, config: dict):
        """Load individual teacher policy following RSL-RL patterns."""
        teacher_name = config["name"]
        teacher_path = config["path"]
        
        # Load using standard RSL-RL approach
        checkpoint = torch.load(teacher_path, map_location=self.device)
        
        # Determine teacher architecture from checkpoint
        if "actor.0.weight" in checkpoint:  # Standard ActorCritic
            from rsl_rl.modules import ActorCritic
            teacher_policy = ActorCritic(
                obs=config.get("obs_spaces"),
                obs_groups=config.get("obs_groups"),
                num_actions=config["num_actions"],
                **config.get("policy_kwargs", {})
            )
        else:
            raise ValueError(f"Unknown teacher architecture for {teacher_name}")
        
        teacher_policy.load_state_dict(checkpoint)
        teacher_policy.eval()  # Set to eval mode
        teacher_policy.to(self.device)
        
        self.teachers[teacher_name] = teacher_policy
        self.teacher_weights[teacher_name] = config.get("weight", 1.0)
        self.teacher_specializations[teacher_name] = config.get("specialization", "general")
    
    def get_teacher_actions(
        self, 
        obs: dict, 
        active_teachers: list[str] = None
    ) -> dict[str, torch.Tensor]:
        """Get deterministic actions from active teachers."""
        if active_teachers is None:
            active_teachers = list(self.teachers.keys())
        
        teacher_actions = {}
        with torch.no_grad():
            for teacher_name in active_teachers:
                if teacher_name in self.teachers:
                    teacher_policy = self.teachers[teacher_name]
                    # Use act_inference for deterministic teacher actions
                    teacher_actions[teacher_name] = teacher_policy.act_inference(obs)
        
        return teacher_actions
    
    def aggregate_teacher_actions(
        self,
        teacher_actions: dict[str, torch.Tensor],
        aggregation_method: str = "weighted_average",
        dynamic_weights: dict[str, float] = None
    ) -> torch.Tensor:
        """Aggregate multiple teacher actions following RSL-RL patterns."""
        if len(teacher_actions) == 0:
            raise ValueError("No teacher actions provided")
        
        if aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(teacher_actions, dynamic_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _weighted_average_aggregation(
        self, 
        teacher_actions: dict[str, torch.Tensor], 
        dynamic_weights: dict[str, float] = None
    ) -> torch.Tensor:
        """Weighted average aggregation of teacher actions."""
        weighted_sum = None
        total_weight = 0.0
        
        for teacher_name, actions in teacher_actions.items():
            if dynamic_weights and teacher_name in dynamic_weights:
                weight = dynamic_weights[teacher_name]
            else:
                weight = self.teacher_weights.get(teacher_name, 1.0)
            
            if weighted_sum is None:
                weighted_sum = weight * actions
            else:
                weighted_sum += weight * actions
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
```

# Curriculum Learning - DEPRIORITIZED (Nice to Have)
# Will be added later as an optional enhancement

### Phase 1.2: Multi-Teacher Student Policy (Week 1)

**Multi-Teacher Student (`rsl_rl/modules/multi_teacher_student.py`)**
```python
class MultiTeacherStudent(nn.Module):
    """Student policy learning from multiple MLP teachers."""
    
    is_recurrent = False  # RSL-RL convention
    
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        # Standard RSL-RL parameters (maintain compatibility)
        student_obs_normalization=False,
        student_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type="scalar",
        # Multi-teacher specific parameters
        teacher_configs: list[dict] = None,
        aggregation_method="weighted_average",
        **kwargs,
    ):
        if kwargs:
            print(
                "MultiTeacherStudent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # Store configuration following RSL-RL patterns
        self.obs_groups = obs_groups
        self.aggregation_method = aggregation_method
        
        # Calculate student observation dimensions (follows ActorCritic pattern)
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "MultiTeacherStudent only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        
        # Build student network (follows RSL-RL MLP pattern)
        from rsl_rl.networks import MLP
        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        
        # Student observation normalization (follows RSL-RL pattern)
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            from rsl_rl.networks import EmpiricalNormalization
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()
        
        print(f"Student MLP: {self.student}")
        
        # Initialize teacher manager
        if teacher_configs:
            self.teacher_manager = TeacherManager(teacher_configs, device="cuda")
        else:
            self.teacher_manager = None
        
        # Action distribution setup (follows RSL-RL ActorCritic pattern exactly)
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    # Standard RSL-RL interface methods
    def act(self, obs, **kwargs):
        """Student action generation with exploration noise."""
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        
        mean = self.student(obs)
        self.update_distribution(mean)
        return self.distribution.sample()
    
    def act_inference(self, obs):
        """Student deterministic action generation."""
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)
    
    def evaluate(self, obs, **kwargs):
        """Value estimation - students don't typically have critics in distillation."""
        # Note: For distillation, we typically don't train a critic
        # But maintain interface for compatibility
        return torch.zeros(obs[list(obs.keys())[0]].shape[0], 1, device=obs[list(obs.keys())[0]].device)
    
    def get_teacher_targets(self, obs):
        """Get aggregated teacher actions as distillation targets."""
        if self.teacher_manager is None:
            raise ValueError("No teachers configured for distillation")
        
        # Get teacher actions (all teachers active by default)
        teacher_actions = self.teacher_manager.get_teacher_actions(obs)
        
        # Aggregate teacher actions using static weights
        return self.teacher_manager.aggregate_teacher_actions(
            teacher_actions, 
            self.aggregation_method
        )
    
    # Standard RSL-RL helper methods
    def get_student_obs(self, obs):
        """Get student observations (follows RSL-RL get_actor_obs pattern)."""
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
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
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def update_normalization(self, obs):
        """Update observation normalization (follows RSL-RL pattern)."""
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)
    
    def reset(self, dones=None):
        """Reset policy state (follows RSL-RL pattern)."""
        pass
    
    def load_state_dict(self, state_dict, strict=True):
        """Load student state dict (follows RSL-RL pattern)."""
        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
```

### Phase 1.3: Multi-Teacher Distillation Algorithm (Week 1-2)

**Multi-Teacher Distillation (`rsl_rl/algorithms/multi_teacher_distillation.py`)**
```python
class MultiTeacherDistillation:
    """Multi-Teacher Distillation algorithm (simplified, no curriculum)."""
    
    policy: MultiTeacherStudent  # Type annotation following RSL-RL pattern
    """The multi-teacher student policy."""
    
    def __init__(
        self,
        policy,
        # Standard distillation parameters (maintain existing order)
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        # Multi-teacher specific parameters (added at end)
        behavior_loss_weight=1.0,
        consistency_loss_weight=0.1,
        # Distributed training (follow RSL-RL pattern)
        multi_gpu_cfg=None,
    ):
        # Follow exact RSL-RL device and multi-GPU setup pattern
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        
        # Policy and parameters (follow RSL-RL pattern)
        self.policy = policy
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.max_grad_norm = max_grad_norm
        
        # Loss configuration
        self.behavior_loss_weight = behavior_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        
        # Loss functions (follow RSL-RL pattern)
        if loss_type == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif loss_type == "huber":
            self.loss_fn = torch.nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Follow RSL-RL optimizer resolution pattern
        from rsl_rl.utils import resolve_optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)
        
        # Standard RSL-RL storage and transition pattern
        self.storage = None
        self.transition = RolloutStorage.Transition()
    
    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,
        actions_shape: tuple,
    ):
        """Initialize rollout storage (follow RSL-RL pattern)."""
        from rsl_rl.storage import RolloutStorage
        
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
    
    def act(self, obs, **kwargs):
        """Get student actions for rollout collection."""
        return self.policy.act(obs, **kwargs)
    
    def process_env_step(self, obs, rewards, dones, infos):
        """Process environment step (follow RSL-RL pattern)."""
        # Get teacher targets for current observations
        with torch.no_grad():
            teacher_targets = self.policy.get_teacher_targets(obs)
        
        # Store teacher targets in transition for later use
        self.transition.observations = obs
        self.transition.teacher_targets = teacher_targets
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.transition.infos = infos
        
        # Update normalization
        self.policy.update_normalization(obs)
    
    def compute_returns(self, last_obs):
        """Compute returns (simplified for distillation)."""
        # For distillation, we don't need GAE computation
        # Just prepare stored data for training
        pass
    
    def update(self):
        """Multi-teacher distillation update."""
        mean_value_loss = 0
        mean_behavior_loss = 0
        mean_consistency_loss = 0
        
        # Multi-epoch training (follow RSL-RL pattern)
        for epoch in range(self.num_learning_epochs):
            # Get mini-batches from storage
            for obs_batch, action_batch, teacher_target_batch in self.storage.mini_batch_generator():
                
                # Forward pass - student predictions
                student_actions = self.policy.act_inference(obs_batch)
                
                # Compute behavior loss (student vs aggregated teachers)
                behavior_loss = self.loss_fn(student_actions, teacher_target_batch)
                
                # Compute consistency loss (if multiple teachers)
                consistency_loss = self._compute_teacher_consistency_loss(obs_batch)
                
                # Total loss
                total_loss = (
                    self.behavior_loss_weight * behavior_loss +
                    self.consistency_loss_weight * consistency_loss
                )
                
                # Gradient step (follow RSL-RL pattern)
                self.optimizer.zero_grad()
                total_loss.backward()
                
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Accumulate losses for logging
                mean_behavior_loss += behavior_loss.item()
                mean_consistency_loss += consistency_loss.item()
        
        # Average losses over all updates
        num_updates = self.num_learning_epochs * self.storage.num_mini_batches
        mean_behavior_loss /= num_updates
        mean_consistency_loss /= num_updates
        
        # Return loss info (follow RSL-RL pattern)
        return {
            "behavior_loss": mean_behavior_loss,
            "consistency_loss": mean_consistency_loss,
            "total_loss": mean_behavior_loss + mean_consistency_loss,
        }
    
    def _compute_teacher_consistency_loss(self, obs_batch):
        """Compute consistency loss between individual teachers."""
        if self.policy.teacher_manager is None:
            return torch.tensor(0.0, device=self.device)
        
        # Get individual teacher actions
        teacher_actions = self.policy.teacher_manager.get_teacher_actions(obs_batch)
        
        if len(teacher_actions) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute pairwise consistency between teachers
        teacher_action_list = list(teacher_actions.values())
        consistency_losses = []
        
        for i in range(len(teacher_action_list)):
            for j in range(i + 1, len(teacher_action_list)):
                consistency_loss = self.loss_fn(teacher_action_list[i], teacher_action_list[j])
                consistency_losses.append(consistency_loss)
        
        return torch.stack(consistency_losses).mean() if consistency_losses else torch.tensor(0.0, device=self.device)
    
    def save(self, path):
        """Save student policy (follow RSL-RL pattern)."""
        torch.save(self.policy.state_dict(), path)
        print(f"Saved student policy to {path}")
    
    def load(self, path):
        """Load student policy (follow RSL-RL pattern)."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded student policy from {path}")
```

## Simplified Implementation Plan

### **Core Implementation (Week 1-2):**
1. **Teacher Manager** - Load and aggregate multiple teacher policies
2. **Multi-Teacher Student** - Student policy learning from teachers
3. **Multi-Teacher Distillation** - Enhanced distillation algorithm
4. **Configuration System** - Simple config handling
5. **Basic Testing** - Validation with dummy teachers

### **Benefits of Simplified Approach:**
- ✅ **Faster Implementation** - No curriculum complexity
- ✅ **Immediate Research Value** - Start multi-teacher experiments right away
- ✅ **Solid Foundation** - Easy to add curriculum later if needed
- ✅ **RSL-RL Compliant** - Follows all existing patterns

### **Future Enhancement (Optional):**
- 📋 **Curriculum Learning** - Progressive teacher weighting
- 📋 **Advanced Aggregation** - Attention-based teacher combination
- 📋 **Dynamic Teacher Selection** - Performance-based teacher activation

This gives us a **clean, focused multi-teacher system** that delivers immediate value!

---
*This document will be updated as implementation progresses*

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Design detailed encoder-decoder architecture", "status": "completed", "activeForm": "Designing detailed encoder-decoder architecture"}, {"content": "Architect multi-teacher student distillation system", "status": "completed", "activeForm": "Architecting multi-teacher student distillation system"}, {"content": "Design observation processing pipeline", "status": "completed", "activeForm": "Designing observation processing pipeline"}, {"content": "Architect policy factory and configuration system", "status": "in_progress", "activeForm": "Architecting policy factory and configuration system"}, {"content": "Design integration points with existing PPO", "status": "pending", "activeForm": "Designing integration points with existing PPO"}, {"content": "Document detailed class hierarchies and interfaces", "status": "pending", "activeForm": "Documenting detailed class hierarchies and interfaces"}]