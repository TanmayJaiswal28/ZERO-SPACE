import argparse
from pathlib import Path

from stable_baselines3 import PPO

from rocket.environment import create_rocket_environment
from rocket.rewards import RewardStrategy


def record_model_flight(
    model_name: str,
    number_of_episodes: int = 3,
):

    model_path = (
        Path("models")
        / model_name
        / "final_model.zip"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find trained model: {model_path}"
        )

    video_directory = (
        Path("results")
        / "videos"
        / model_name
    )

    environment = create_rocket_environment(
        reward_strategy=RewardStrategy.BASELINE,
        render_mode="rgb_array",
        record_video=True,
        video_directory=video_directory,
    )

    agent = PPO.load(str(model_path))

    try:

        for episode_number in range(number_of_episodes):

            observation, information = environment.reset(
                seed=2000 + episode_number
            )

            finished = False
            total_reward = 0.0

            while not finished:

                action, _ = agent.predict(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    information,
                ) = environment.step(action)

                total_reward += reward

                finished = terminated or truncated

            print(
                f"Recorded episode {episode_number + 1} | "
                f"Reward: {total_reward:.2f}"
            )

    finally:
        environment.close()

    print(f"\nVideos saved in: {video_directory}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "baseline",
            "time_penalty",
            "hover_reward",
            "progress_reward",
        ],
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
    )

    arguments = parser.parse_args()

    record_model_flight(
        model_name=arguments.model,
        number_of_episodes=arguments.episodes,
    )


if __name__ == "__main__":
    main()