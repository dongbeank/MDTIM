"""
Simple dataclass-based configuration.
Replaces Hydra + multiple YAML files with a single Python file.

Usage:
    # Default config
    cfg = Config()

    # ETTh preset
    cfg = Config.etth()

    # Energy preset
    cfg = Config.energy()

    # From command line
    cfg = Config.from_args()
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import argparse
import json


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 64
    cond_dim: int = 16
    n_blocks: int = 5
    n_heads: int = 4
    dropout: float = 0.3
    n_bins: int = 40
    length: int = 48          # sequence length (window size)
    n_channels: int = 7       # number of features
    scale_by_sigma: bool = True
    output_range: float = 1.5  # output range [-r, r], input is always [-1, 1]
    shared_channel_embedding: bool = False  # True = one embedding shared across all channels (scalability study)


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset_name: str = "etth"
    train_path: str = "./data/ETTh.csv"
    # Data split: 8:1:1 (train:val:test) - fixed ratio, no need to configure
    min_range: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    eval_batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # LR Scheduler
    lr_scheduler: str = "constant_warmup"  # constant_warmup / cosine_warmup / none
    warmup_steps: int = 2500

    # Training params
    max_steps: int = 15000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # EMA & sampling
    ema: float = 0.995
    antithetic_sampling: bool = True
    importance_sampling: bool = False
    change_of_variables: bool = False
    sampling_eps: float = 1e-3

    # Loss
    fft_weight: float = 1.0
    label_noise: bool = True
    soft_label_sigma_ratio: float = 1.0
    soft_label_window: int = 2  # 0 = one-hot (hard), >0 = soft labeling with ±window neighbors


@dataclass
class NoiseConfig:
    """Noise schedule configuration."""
    type: str = "loglinear"  # loglinear / cosine / linear / geometric
    sigma_min: float = 1e-3
    sigma_max: float = 1.0


@dataclass
class SamplingConfig:
    """Sampling/inference configuration."""
    predictor: str = "ddpm_cache"  # ddpm / ddpm_cache
    steps: int = 128
    noise_removal: bool = True


@dataclass
class Config:
    """Main configuration class."""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # Top-level settings
    seed: int = 2024
    mode: str = "train"  # train / eval / impute

    # Trainer
    precision: str = "bf16-mixed"
    devices: int = 1
    num_nodes: int = 1

    # Checkpointing
    save_dir: str = "./checkpoints"
    checkpoint_path: Optional[str] = None
    resume_from_ckpt: bool = False
    version: Optional[int] = None  # version number for loading (None = latest)

    # Imputation settings
    missing_ratio: float = 0.1
    impute_method: str = "ddpm"  # ddpm / topprob / consec7

    @classmethod
    def etth(cls) -> "Config":
        """ETTh dataset preset (7 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=256,
                cond_dim=16,
                n_blocks=5,
                n_heads=16,
                dropout=0.2,
                n_bins=40,
                length=48,
                n_channels=7,
                            ),
            data=DataConfig(
                dataset_name="etth",
                train_path="./data/ETTh.csv",
            ),
            training=TrainingConfig(
                fft_weight=1.0,
                max_steps=10000,  # Smaller dataset
            ),
        )

    @classmethod
    def energy(cls) -> "Config":
        """Energy dataset preset (28 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=256,
                cond_dim=16,
                n_blocks=5,
                n_heads=16,
                dropout=0.2,
                n_bins=40,
                length=48,
                n_channels=28,
                            ),
            data=DataConfig(
                dataset_name="energy",
                train_path="./data/energy_data.csv",
            ),
            training=TrainingConfig(
                fft_weight=1.0,
                max_steps=10000,  # Smaller dataset
            ),
        )

    @classmethod
    def fmri(cls) -> "Config":
        """fMRI dataset preset (50 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=256,
                cond_dim=16,
                n_blocks=5,
                n_heads=16,
                dropout=0.2,
                n_bins=40,
                length=48,
                n_channels=50,
                            ),
            data=DataConfig(
                dataset_name="fmri",
                train_path="./data/fMRI",
            ),
        )

    @classmethod
    def sine(cls) -> "Config":
        """Sine synthetic dataset preset (5 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=256,
                cond_dim=16,
                n_blocks=5,
                n_heads=16,
                dropout=0.2,
                n_bins=40,
                length=48,
                n_channels=5,
                            ),
            data=DataConfig(
                dataset_name="sine",
                train_path="./data/sine",
            ),
            training=TrainingConfig(
                fft_weight=1.0,
                max_steps=10000,  # Smaller dataset
            ),
        )

    @classmethod
    def mujoco(cls) -> "Config":
        """MuJoCo dataset preset (14 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=256,
                cond_dim=16,
                n_blocks=5,
                n_heads=16,
                dropout=0.2,
                n_bins=40,
                length=48,
                n_channels=14,
                            ),
            data=DataConfig(
                dataset_name="mujoco",
                train_path="./data/mujoco",
            ),
        )

    @classmethod
    def weather(cls) -> "Config":
        """Weather dataset preset (21 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=256,
                cond_dim=16,
                n_blocks=5,
                n_heads=16,
                dropout=0.2,
                n_bins=40,
                length=48,
                n_channels=21,
                            ),
            data=DataConfig(
                dataset_name="weather",
                train_path="./data/weather.csv",
            ),
            training=TrainingConfig(
                fft_weight=1.0,
                max_steps=10000,  # Smaller dataset
            ),
        )

    @classmethod
    def physionet2012(cls) -> "Config":
        """PhysioNet2012 ICU dataset preset (37 channels, 48 timesteps)."""
        return cls(
            model=ModelConfig(
                hidden_size=64,
                cond_dim=16,
                n_blocks=5,
                n_heads=4,
                dropout=0.3,
                n_bins=40,
                length=48,
                n_channels=37,
                            ),
            data=DataConfig(
                dataset_name="physionet2012",
                train_path="./data/physionet2012",
            ),
            training=TrainingConfig(
                fft_weight=0.0,  # Disable FFT loss for sparse data
            ),
        )

    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> "Config":
        """Create config from command line arguments."""
        if args is None:
            parser = cls.get_parser()
            args = parser.parse_args()

        # Start with data preset if specified
        if args.data == "etth":
            cfg = cls.etth()
        elif args.data == "energy":
            cfg = cls.energy()
        elif args.data == "fmri":
            cfg = cls.fmri()
        elif args.data == "sine":
            cfg = cls.sine()
        elif args.data == "mujoco":
            cfg = cls.mujoco()
        elif args.data == "weather":
            cfg = cls.weather()
        elif args.data == "physionet2012":
            cfg = cls.physionet2012()
        else:
            cfg = cls()

        # Override with command line args
        for key, value in vars(args).items():
            if value is None or key == "data":
                continue

            # Handle nested configs
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
            elif hasattr(cfg.data, key):
                setattr(cfg.data, key, value)
            elif hasattr(cfg.training, key):
                setattr(cfg.training, key, value)
            elif hasattr(cfg.sampling, key):
                setattr(cfg.sampling, key, value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)

        return cfg

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        """Get argument parser for command line usage."""
        parser = argparse.ArgumentParser(description="MDTM Training")

        # Data preset
        parser.add_argument("--data", type=str, default=None,
                          choices=["etth", "energy", "fmri", "sine", "mujoco", "weather", "physionet2012"],
                          help="Dataset preset")

        # Mode
        parser.add_argument("--mode", type=str, default="train",
                          choices=["train", "eval", "impute"])
        parser.add_argument("--seed", type=int, default=None)

        # Model
        parser.add_argument("--hidden_size", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--n_blocks", type=int, default=None)
        parser.add_argument("--n_bins", type=int, default=None)
        parser.add_argument("--length", type=int, default=None)
        parser.add_argument("--n_channels", type=int, default=None)
        parser.add_argument("--shared_channel_embedding", action="store_true", default=None,
                          help="Share one token embedding across all channels (scalability study)")

        # Data
        parser.add_argument("--dataset_name", type=str, default=None)
        parser.add_argument("--train_path", type=str, default=None)

        # Training
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--max_steps", type=int, default=None)
        parser.add_argument("--ema", type=float, default=None)
        parser.add_argument("--fft_weight", type=float, default=None)
        parser.add_argument("--label_noise", action="store_true", default=None)
        parser.add_argument("--soft_label_window", type=int, default=None,
                          help="Soft label window size. 0=one-hot, >0=soft labeling")

        # Trainer
        parser.add_argument("--precision", type=str, default=None)
        parser.add_argument("--devices", type=int, default=None)

        # Checkpointing
        parser.add_argument("--save_dir", type=str, default=None)
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--resume_from_ckpt", action="store_true", default=None)
        parser.add_argument("--version", type=int, default=None,
                          help="Version number to load (for eval/impute). None = latest")

        # Sampling
        parser.add_argument("--steps", type=int, default=None)

        # Imputation
        parser.add_argument("--missing_ratio", type=float, default=None)
        parser.add_argument("--impute_method", type=str, default=None,
                          choices=["ddpm", "topprob", "consec7"])

        return parser

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        nested_keys = ["model", "data", "training", "sampling", "noise"]
        return cls(
            model=ModelConfig(**data.get("model", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            noise=NoiseConfig(**data.get("noise", {})),
            **{k: v for k, v in data.items() if k not in nested_keys}
        )

    def print(self):
        """Print configuration."""
        print("=" * 50)
        print("Configuration")
        print("=" * 50)
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                print(f"\n[{key}]")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("=" * 50)


# Compatibility layer for old config access patterns
# e.g., config.model.n_bins, config.loader.batch_size
class ConfigCompat:
    """Wrapper to provide old-style attribute access."""

    def __init__(self, config: Config):
        self._config = config

        # Create loader alias for training
        self.loader = type('Loader', (), {
            'batch_size': config.training.batch_size,
            'eval_batch_size': config.training.eval_batch_size,
            'num_workers': config.training.num_workers,
            'pin_memory': config.training.pin_memory,
        })()

        # Create optim alias
        self.optim = type('Optim', (), {
            'lr': config.training.lr,
            'weight_decay': config.training.weight_decay,
            'beta1': config.training.beta1,
            'beta2': config.training.beta2,
            'eps': config.training.eps,
        })()

        # Create trainer alias
        self.trainer = type('Trainer', (), {
            'precision': config.precision,
            'devices': config.devices,
            'num_nodes': config.num_nodes,
            'max_steps': config.training.max_steps,
            'gradient_clip_val': config.training.gradient_clip_val,
            'accumulate_grad_batches': config.training.accumulate_grad_batches,
        })()

        # Create checkpointing alias
        self.checkpointing = type('Checkpointing', (), {
            'save_dir': config.save_dir,
            'resume_from_ckpt': config.resume_from_ckpt,
            'resume_ckpt_path': config.checkpoint_path,
        })()

        # Create eval alias
        self.eval = type('Eval', (), {
            'checkpoint_path': config.checkpoint_path,
            'disable_ema': False,
        })()

        # Create noise alias
        self.noise = config.noise

        # Create training alias (for direct access to training params)
        self.training = config.training

    def __getattr__(self, name):
        # Handle special methods to avoid recursion
        if name.startswith('_'):
            raise AttributeError(f"Config has no attribute '{name}'")
        # First check if it's a direct config attribute
        if hasattr(self._config, name):
            return getattr(self._config, name)
        raise AttributeError(f"Config has no attribute '{name}'")

    def __deepcopy__(self, memo):
        """Support deepcopy for Lightning's save_hyperparameters."""
        import copy
        new_config = copy.deepcopy(self._config, memo)
        return ConfigCompat(new_config)

    def __getstate__(self):
        """Support pickle/deepcopy."""
        return {'_config': self._config}

    def __setstate__(self, state):
        """Support pickle/deepcopy."""
        self.__init__(state['_config'])


def get_config(args=None) -> ConfigCompat:
    """Get config with compatibility wrapper."""
    config = Config.from_args(args)
    return ConfigCompat(config)


if __name__ == "__main__":
    # Test
    cfg = Config.etth()
    cfg.print()

    print("\n\nWith compatibility wrapper:")
    compat = ConfigCompat(cfg)
    print(f"model.n_bins: {compat.model.n_bins}")
    print(f"loader.batch_size: {compat.loader.batch_size}")
    print(f"trainer.max_steps: {compat.trainer.max_steps}")
