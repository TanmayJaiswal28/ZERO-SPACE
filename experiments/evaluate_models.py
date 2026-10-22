from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from configs.experiment import DEFAULT_CONFIG
from rocket.environment import create_rocket_environment
from rocket.flight_analysis import create_flight_result
from rocket.rewards import RewardStrategy


MODEL_NAMES = [
    "baseline",
    "time_penalty",
    "hover_reward",
    "progress_reward",
]


def evaluate_single_model(
    model_name: str,
    number_of_episodes: int,
):

    model_path = (
        Path("models")
        / model_name
        / "final_model.zip"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    agent = PPO.load(str(model_path))

    environment = create_rocket_environment(
        reward_strategy=RewardStrategy.BASELINE
    )

    flight_results = []

    try:

        for episode_number in range(number_of_episodes):

            observation, information = environment.reset(
                seed=1000 + episode_number
            )

            finished = False

            original_reward = 0.0
            training_reward = 0.0

            flight_steps = 0

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

                original_reward += information.get(
                    "original_reward",
                    reward,
                )

                training_reward += reward

                flight_steps += 1

                finished = terminated or truncated

            result = create_flight_result(
                model_name=model_name,
                episode_number=episode_number + 1,
                original_reward=original_reward,
                training_reward=training_reward,
                flight_steps=flight_steps,
                final_observation=observation,
                terminated=terminated,
                truncated=truncated,
            )

            flight_results.append(
                result.to_dictionary()
            )

    finally:
        environment.close()

    return flight_results


def main():

    output_directory = Path("results/evaluations")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for model_name in MODEL_NAMES:

        print(f"\nEvaluating model: {model_name}")

        results = evaluate_single_model(
            model_name=model_name,
            number_of_episodes=DEFAULT_CONFIG.evaluation_episodes,
        )

        all_results.extend(results)

    results_dataframe = pd.DataFrame(all_results)

    results_dataframe.to_csv(
        output_directory / "flight_results.csv",
        index=False,
    )

    summary = (
        results_dataframe
        .groupby("model_name")
        .agg(
            mean_reward=("original_reward", "mean"),
            reward_std=("original_reward", "std"),
            average_flight_steps=("flight_steps", "mean"),
            solved_score_rate=("reached_solved_score", "mean"),
            timeout_rate=("truncated", "mean"),
        )
    )

    summary["solved_score_rate"] *= 100
    summary["timeout_rate"] *= 100

    summary.to_csv(
        output_directory / "model_comparison.csv"
    )

    print("\nZERO SPACE | MODEL COMPARISON")
    print(summary.to_string())


if __name__ == "__main__":
    main()