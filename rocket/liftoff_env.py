
import numpy as np
from rocket.orbital_env import OrbitalEnv


class LiftoffEnv(OrbitalEnv):
    """
    Curriculum environment for autonomous liftoff.

    Uses the existing Earth physics and vehicle model.
    """

    def __init__(self, target_altitude=100.0):
        super().__init__()

        self.target_altitude = target_altitude
        self.max_steps = 600

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(
            seed=seed,
            options=options
        )

        self.phase = "ON PAD"

        return observation, self._info()

    def step(self, action):
        previous_altitude = max(
            0.0,
            self._info()["altitude_m"]
        )

        (
            observation,
            original_reward,
            terminated,
            truncated,
            info
        ) = super().step(action)

        altitude = info["altitude_m"]

        altitude_gain = max(
            0.0,
            altitude - previous_altitude
        )

        # Reward actual upward progress.
        reward = altitude_gain * 0.1

        # Discourage remaining on the pad.
        if altitude < 1.0:
            reward -= 0.05

        # Mild fuel-efficiency penalty.
        reward -= float(info["throttle"]) * 0.002

        success = altitude >= self.target_altitude

        if success:
            reward += 100.0
            terminated = True
            self.phase = "TARGET REACHED"

        elif altitude < 1.0:
            self.phase = "ON PAD"

        elif info["phase"] == "IMPACT":
            self.phase = "IMPACT"

        else:
            self.phase = "ASCENT"

        # Parent environment already accumulated
        # original_reward; replace it with our reward.
        self.total_reward += reward - original_reward

        info = self._info()
        info["success"] = bool(success)

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )
