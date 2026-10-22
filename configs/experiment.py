
from dataclasses import dataclass
from pathlib import Path


# Locate the project root from this file's location.
# This keeps file paths consistent regardless of the terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIRECTORY = PROJECT_ROOT / "models"
LOG_DIRECTORY = PROJECT_ROOT / "logs"
RESULT_DIRECTORY = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class TrainingConfig:
    """
    Shared settings for the ZERO SPACE rocket landing experiments.

    Keeping training parameters in one place helps us reproduce
    results and compare different reward strategies fairly.
    """

    environment_id: str = "LunarLander-v3"

    total_training_steps: int = 1_000_000

    learning_rate: float = 3e-4
    rollout_steps: int = 2048
    batch_size: int = 64
    optimization_epochs: int = 10

    discount_factor: float = 0.99
    gae_lambda: float = 0.95

    clipping_range: float = 0.2
    entropy_coefficient: float = 0.0

    random_seed: int = 42

    checkpoint_interval: int = 100_000
    evaluation_episodes: int = 30
    maximum_episode_steps: int = 1000


DEFAULT_CONFIG = TrainingConfig()
