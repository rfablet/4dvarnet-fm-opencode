from dataclasses import dataclass, field
from typing import List, Tuple
from omegaconf import MISSING


@dataclass
class DataConfig:
    system: str = "lorenz63"
    dt: float = 0.01
    T_max: float = 5.0
    obs_interval: int = 20
    R_var: float = 0.5
    B_var: float = 2.0
    num_windows: int = 2000
    window_spacing: int = 2000
    spinup_steps: int = 10000
    seed: int = 42
    sigma_true: float = 10.0
    rho_true: float = 28.0
    beta_true: float = 8 / 3
    gamma: float = 0.05
    W_L_bar: float = 0.0
    c1: float = 1.0
    c2: float = 0.1
    sigma_0: float = 0.08
    sigma_L: float = 0.20
    tau_eta: float = 5.0
    sigma_eta: float = 0.7071067811865476
    forcing_state_bias: float = 0.0
    forcing_coupling: str = "linear"
    param_bias: float = 0.0
    case: int = 1

    @property
    def num_steps(self) -> int:
        return int(self.T_max / self.dt)

    @property
    def biased_params(self) -> Tuple[float, float, float]:
        b = self.param_bias
        return (
            self.sigma_true * (1 - b),
            self.rho_true * (1 - b),
            self.beta_true * (1 + b),
        )

    @property
    def da_params(self) -> Tuple[float, float, float]:
        if self.case == 1:
            return (self.sigma_true, self.rho_true, self.beta_true)
        return self.biased_params

    @property
    def use_corrupted_forcing(self) -> bool:
        return self.case == 2


@dataclass
class ModelConfig:
    state_dim: int = 3
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    K_inner: int = 5
    N_outer: int = 10
    nu: float = 1.0
    use_obs: bool = True
    use_energy: bool = True
    dropout: float = 0.1


@dataclass
class StageConfig:
    epochs: int = 200
    lr: float = 1e-3
    gradient_clip_val: float = 10.0


@dataclass
class LossConfig:
    use_gradient: bool = True
    gradient_weight: float = 0.1


@dataclass
class TrainingConfig:
    stage1: StageConfig = field(default_factory=lambda: StageConfig(epochs=200, lr=1e-3, gradient_clip_val=10.0))
    stage2: StageConfig = field(default_factory=lambda: StageConfig(epochs=400, lr=1e-3, gradient_clip_val=1.0))
    batch_size: int = 32
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class PathsConfig:
    checkpoint_dir: str = "checkpoints"
    checkpoint_stage1: str = "checkpoints/stage1.pt"
    checkpoint_stage2: str = "checkpoints/stage2.pt"
    outputs_dir: str = "outputs"


@dataclass
class Weak4DVarConfig:
    opt_steps: int = 150
    lr: float = 0.02


@dataclass
class Strong4DVarConfig:
    max_iter: int = 40
    lr: float = 0.1


@dataclass
class EnKFConfig:
    inflation: float = 1.0


@dataclass
class BaselinesConfig:
    da_window_steps: int = 500
    N_ensemble: int = 30
    weak4dvar: Weak4DVarConfig = field(default_factory=Weak4DVarConfig)
    strong4dvar: Strong4DVarConfig = field(default_factory=Strong4DVarConfig)
    enkf: EnKFConfig = field(default_factory=EnKFConfig)


@dataclass
class CaseStudyConfig:
    param_bias: float = 0.0
    forcing_state_bias: float = 0.0
    forcing_coupling: str = "linear"


@dataclass
class CS1Config:
    param_bias: float = 0.0
    forcing_coupling: str = "linear"


@dataclass
class CS2Config:
    param_bias: float = 0.15
    forcing_state_bias: float = 0.15
    forcing_coupling: str = "quartic"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    cs1: CS1Config = field(default_factory=CS1Config)
    cs2: CS2Config = field(default_factory=CS2Config)
