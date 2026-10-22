import argparse

from learning.trainer import train_rocket
from rocket.rewards import RewardStrategy


AVAILABLE_EXPERIMENTS = {
    "time_penalty": RewardStrategy.TIME_PENALTY,
    "hover_reward": RewardStrategy.HOVER_REWARD,
    "progress_reward": RewardStrategy.PROGRESS_REWARD,
}


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a Lunar Lander agent with a selected "
            "reward shaping strategy."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=AVAILABLE_EXPERIMENTS.keys(),
        help="Reward experiment to train.",
    )

    arguments = parser.parse_args()

    selected_experiment = arguments.experiment

    selected_strategy = AVAILABLE_EXPERIMENTS[
        selected_experiment
    ]

    train_rocket(
        experiment_name=selected_experiment,
        reward_strategy=selected_strategy,
    )


if __name__ == "__main__":
    main()