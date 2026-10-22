from pathlib import Path

import gymnasium as gym

from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor

from configs.experiment import DEFAULT_CONFIG
from rocket.rewards import (
    RewardStrategy,
    RocketRewardWrapper,
)


def create_rocket_environment(
    reward_strategy: RewardStrategy = RewardStrategy.BASELINE,
    render_mode: str | None = None,
    record_video: bool = False,
    video_directory: str | Path | None = None,
):
    """
    Creates a Lunar Lander environment.

    The same factory is used for training, evaluation,
    and video recording to avoid inconsistent setups.
    """

    if record_video and render_mode != "rgb_array":
        raise ValueError(
            "Video recording requires render_mode='rgb_array'."
        )

    environment = gym.make(
        DEFAULT_CONFIG.environment_id,
        render_mode=render_mode,
        max_episode_steps=DEFAULT_CONFIG.maximum_episode_steps,
    )

    environment = RocketRewardWrapper(
        environment=environment,
        strategy=reward_strategy,
        discount_factor=DEFAULT_CONFIG.discount_factor,
    )

    if record_video:

        if video_directory is None:
            raise ValueError(
                "A video directory must be provided."
            )

        video_directory = Path(video_directory)
        video_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        environment = RecordVideo(
            environment,
            video_folder=str(video_directory),
            episode_trigger=lambda episode: True,
            name_prefix=reward_strategy.value,
        )

    environment = Monitor(environment)

    return environment