from enum import Enum

import gymnasium as gym

from rocket.observations import RocketState


class RewardStrategy(str, Enum):

    BASELINE = "baseline"
    TIME_PENALTY = "time_penalty"
    HOVER_REWARD = "hover_reward"
    PROGRESS_REWARD = "progress_reward"


class RocketRewardWrapper(gym.Wrapper):
    """
    Modifies the learning reward without changing the physics.

    The original environment reward is retained separately
    so evaluation can distinguish real task performance
    from additional shaping bonuses.
    """

    def __init__(
        self,
        environment: gym.Env,
        strategy: RewardStrategy,
        discount_factor: float = 0.99,
    ):

        super().__init__(environment)

        self.strategy = strategy
        self.discount_factor = discount_factor

        self.previous_potential = 0.0
        self.episode_steps = 0

    def reset(self, **kwargs):

        observation, information = self.env.reset(**kwargs)

        rocket_state = RocketState.from_observation(
            observation
        )

        self.previous_potential = self.calculate_potential(
            rocket_state
        )

        self.episode_steps = 0

        return observation, information

    def calculate_potential(
        self,
        rocket_state: RocketState
    ) -> float:
        """
        Measures how desirable the current flight state is.

        This is a heuristic potential, not a physical energy
        calculation.
        """

        position_cost = (
            1.0 * rocket_state.distance_from_target
        )

        velocity_cost = (
            0.8 * rocket_state.speed
        )

        orientation_cost = (
            0.5 * abs(rocket_state.angle)
        )

        rotation_cost = (
            0.2 * abs(rocket_state.angular_velocity)
        )

        return -(
            position_cost
            + velocity_cost
            + orientation_cost
            + rotation_cost
        )

    def calculate_extra_reward(
        self,
        rocket_state: RocketState,
        terminated: bool,
    ) -> float:

        if self.strategy == RewardStrategy.BASELINE:
            return 0.0

        if self.strategy == RewardStrategy.TIME_PENALTY:
            return -0.01

        if self.strategy == RewardStrategy.HOVER_REWARD:

            if (
                rocket_state.is_upright
                and rocket_state.is_moving_slowly
            ):
                return 1.0

            return 0.0

        if self.strategy == RewardStrategy.PROGRESS_REWARD:

            current_potential = self.calculate_potential(
                rocket_state
            )

            # Terminal states receive zero potential.
            # This makes the shaping term consistent with
            # the discount factor used by PPO.
            next_potential = (
                0.0 if terminated else current_potential
            )

            shaping_reward = (
                self.discount_factor * next_potential
                - self.previous_potential
            )

            self.previous_potential = current_potential

            return shaping_reward - 0.01

        raise ValueError(
            f"Unknown reward strategy: {self.strategy}"
        )

    def step(self, action):

        observation, original_reward, terminated, truncated, information = (
            self.env.step(action)
        )

        self.episode_steps += 1

        rocket_state = RocketState.from_observation(
            observation
        )

        extra_reward = self.calculate_extra_reward(
            rocket_state,
            terminated or truncated,
        )

        final_reward = original_reward + extra_reward

        information = dict(information)

        information["original_reward"] = float(original_reward)
        information["shaping_reward"] = float(extra_reward)
        information["training_reward"] = float(final_reward)

        return (
            observation,
            final_reward,
            terminated,
            truncated,
            information,
        )