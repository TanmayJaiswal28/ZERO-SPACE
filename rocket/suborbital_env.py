
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from physics.earth3d import (
    EARTH_RADIUS,
    EARTH_MU,
    STANDARD_GRAVITY,
)

RHO0 = 1.225
SCALE_HEIGHT = 8500.0


class SuborbitalEnv(gym.Env):
    """
    ZERO SPACE EXP005

    1D vertical suborbital research environment.

    mode:
      "ascent"   -> learn to cross target altitude
      "recovery" -> learn powered vertical landing

    Action:
      [throttle] in [0, 1]

    This is not a flight-qualified vehicle model.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mode="ascent",
        target_altitude=100_000.0,
    ):
        super().__init__()

        if mode not in ("ascent", "recovery"):
            raise ValueError("Invalid training mode")

        self.mode = mode
        self.target_altitude = float(target_altitude)

        self.dt = 0.25
        self.max_steps = 4000

        # Hypothetical reusable research vehicle.
        self.dry_mass = 20_000.0
        self.initial_fuel = 180_000.0

        self.max_thrust = 5_000_000.0
        self.isp = 340.0

        self.cd = 0.35
        self.area = 10.0

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        self.reset()

    def gravity(self, altitude):
        radius = EARTH_RADIUS + max(0.0, altitude)
        return EARTH_MU / radius**2

    def density(self, altitude):
        return RHO0 * math.exp(
            -max(0.0, altitude) / SCALE_HEIGHT
        )

    def _observation(self):
        return np.array([
            self.h / 100_000.0,
            self.v / 3000.0,
            self.fuel / self.initial_fuel,
            self.mass / 200_000.0,
            self.q / 100_000.0,
            self.throttle,
            self.time / 1000.0,
        ], dtype=np.float32)

    def _info(self):
        return {
            "altitude_m": float(max(0.0, self.h)),
            "vertical_velocity_mps": float(self.v),
            "speed_mps": float(abs(self.v)),
            "acceleration_mps2": float(self.a),
            "fuel_kg": float(self.fuel),
            "mass_kg": float(self.mass),
            "throttle": float(self.throttle),
            "dynamic_pressure_pa": float(self.q),
            "heating_proxy": float(self.heat),
            "time_s": float(self.time),
            "phase": self.phase,
            "max_altitude_m": float(self.max_altitude),
            "total_reward": float(self.total_reward),
            "position_m": [
                float(EARTH_RADIUS + max(0.0, self.h)),
                0.0,
                0.0,
            ],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0.0
        self.steps = 0
        self.throttle = 0.0
        self.a = 0.0
        self.q = 0.0
        self.heat = 0.0
        self.total_reward = 0.0

        if self.mode == "ascent":
            self.h = 0.0
            self.v = 0.0
            self.fuel = self.initial_fuel
            self.phase = "ON PAD"

        else:
            # Randomized descending conditions for
            # landing curriculum training.
            self.h = float(
                self.np_random.uniform(500, 5000)
            )
            self.v = float(
                self.np_random.uniform(-300, -30)
            )
            self.fuel = float(
                self.np_random.uniform(5000, 30000)
            )
            self.phase = "RECOVERY"

        self.mass = self.dry_mass + self.fuel
        self.max_altitude = max(0.0, self.h)

        return self._observation(), self._info()

    def set_state(self, altitude, velocity, fuel):
        """
        Initialize a recovery rollout from a simulated
        flight state. Intended for environment handoff.
        """
        self.h = max(0.0, float(altitude))
        self.v = float(velocity)
        self.fuel = float(
            np.clip(fuel, 0, self.initial_fuel)
        )
        self.mass = self.dry_mass + self.fuel

        self.time = 0.0
        self.steps = 0
        self.throttle = 0.0
        self.a = 0.0
        self.q = 0.0
        self.heat = 0.0
        self.total_reward = 0.0
        self.max_altitude = self.h
        self.phase = "RECOVERY"

        return self._observation(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        self.throttle = float(
            np.clip(action[0], 0.0, 1.0)
        )

        self.steps += 1

        old_h = self.h
        old_v = self.v

        # Propellant consumption.
        requested_thrust = (
            self.max_thrust * self.throttle
        )

        required_fuel = (
            requested_thrust
            * self.dt
            / (self.isp * STANDARD_GRAVITY)
        )

        used_fuel = min(
            self.fuel,
            required_fuel
        )

        mass_before = self.dry_mass + self.fuel
        self.fuel -= used_fuel
        self.mass = self.dry_mass + self.fuel

        actual_thrust = (
            used_fuel
            * self.isp
            * STANDARD_GRAVITY
            / self.dt
        )

        average_mass = (
            mass_before + self.mass
        ) / 2

        rho = self.density(self.h)

        # Drag always opposes vertical velocity.
        drag = (
            -0.5
            * rho
            * self.cd
            * self.area
            * self.v
            * abs(self.v)
        )

        self.q = 0.5 * rho * self.v**2

        # Qualitative heating proxy, not temperature.
        self.heat = (
            math.sqrt(rho / RHO0)
            * abs(self.v)**3
        )

        self.a = (
            (actual_thrust + drag) / average_mass
            - self.gravity(self.h)
        )

        # Semi-implicit Euler integration.
        self.v += self.a * self.dt
        self.h += self.v * self.dt

        self.time += self.dt

        self.max_altitude = max(
            self.max_altitude,
            self.h
        )

        terminated = False
        success = False
        reward = 0.0

        if self.mode == "ascent":
            altitude_gain = self.h - old_h

            reward = (
                altitude_gain / 100.0
                - 0.000002 * used_fuel
                - 0.005
            )

            if self.h < 1.0:
                reward -= 0.05

            if self.q > 60_000:
                reward -= (
                    (self.q - 60_000) / 60_000
                ) * 0.1

            if self.h >= self.target_altitude:
                success = True
                terminated = True
                reward += 200
                self.phase = "SPACE REACHED"

            elif self.h <= 0 and self.max_altitude > 10:
                terminated = True
                reward -= 100
                self.phase = "IMPACT"

            elif self.h <= 0:
                self.h = 0.0
                self.v = max(0.0, self.v)
                self.phase = "ON PAD"

            elif self.v >= 0:
                self.phase = "ASCENT"

            else:
                self.phase = "DESCENT"

        else:
            # Recovery objective: slow descent and
            # approach zero altitude.
            reward = (
                (old_h - self.h) / 100.0
                - 0.001 * abs(self.v) / 10.0
                - 0.000003 * used_fuel
                - 0.002
            )

            if self.h <= 0:
                touchdown_speed = abs(self.v)

                success = (
                    self.v <= 0
                    and touchdown_speed <= 3.0
                )

                terminated = True
                self.h = 0.0

                if success:
                    reward += 300
                    self.phase = "LANDED"
                else:
                    reward -= (
                        200 + min(touchdown_speed, 500)
                    )
                    self.phase = "HARD LANDING"

            elif self.v > 0:
                self.phase = "RECOVERY BRAKING"

            else:
                self.phase = "DESCENT"

        truncated = self.steps >= self.max_steps

        self.total_reward += float(reward)

        info = self._info()
        info["success"] = bool(success)
        info["touchdown_speed_mps"] = (
            abs(self.v) if terminated and self.h <= 0
            else None
        )

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
