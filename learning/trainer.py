import json
from datetime import datetime, timezone
from pathlib import Path

from configs.experiment import DEFAULT_CONFIG
from learning.agent import create_rocket_agent
from learning.callbacks import create_training_callbacks
from rocket.environment import create_rocket_environment
from rocket.rewards import RewardStrategy


def train_rocket(
    experiment_name: str,
    reward_strategy: RewardStrategy,
):

    configuration = DEFAULT_CONFIG

    experiment_directory = (
        Path("models") / experiment_name
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment = create_rocket_environment(
        reward_strategy=reward_strategy
    )

    print("\n" + "=" * 60)
    print("ZERO SPACE | AUTONOMOUS ROCKET LANDING")
    print("=" * 60)

    print(f"Experiment: {experiment_name}")
    print(f"Reward strategy: {reward_strategy.value}")
    print(f"Training steps: {configuration.total_training_steps:,}")
    print(f"Random seed: {configuration.random_seed}")

    print("=" * 60 + "\n")

    metadata = {
        "experiment": experiment_name,
        "reward_strategy": reward_strategy.value,
        "training_steps": configuration.total_training_steps,
        "random_seed": configuration.random_seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = experiment_directory / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    try:

        agent = create_rocket_agent(environment)

        callbacks = create_training_callbacks(
            experiment_name=experiment_name,
            checkpoint_interval=configuration.checkpoint_interval,
        )

        agent.learn(
            total_timesteps=configuration.total_training_steps,
            callback=callbacks,
            tb_log_name=experiment_name,
        )

        model_path = experiment_directory / "final_model"

        agent.save(str(model_path))

        metadata["completed_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        metadata["model_path"] = str(model_path)

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        print("\nTraining completed successfully.")
        print(f"Model saved: {model_path}.zip")

    finally:
        environment.close()