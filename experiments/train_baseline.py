from learning.trainer import train_rocket
from rocket.rewards import RewardStrategy


def main():

    train_rocket(
        experiment_name="baseline",
        reward_strategy=RewardStrategy.BASELINE,
    )


if __name__ == "__main__":
    main()