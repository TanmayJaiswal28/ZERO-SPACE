
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EarthAscentEnv(gym.Env):
    """
    ZERO SPACE EXP003
    Simplified radial Earth ascent.

    SI units internally.
    Educational simulation, not flight software.
    """

    metadata = {"render_modes": []}

    R_EARTH = 6_371_000.0
    MU = 3.986004418e14
    G0 = 9.80665

    def __init__(self):
        super().__init__()

        self.dt = 0.5
        self.target_altitude = 100_000.0
        self.max_steps = 2400

        # Fictional demonstration vehicle.
        # Not a validated rocket design.
        self.dry_mass = 25_000.0
        self.initial_fuel = 125_000.0
        self.max_thrust = 4_000_000.0
        self.isp = 310.0

        self.cd = 0.35
        self.area = 12.0

        # Continuous throttle: 0 to 1
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
        radius = self.R_EARTH + max(0.0, altitude)
        return self.MU / radius**2

    def density(self, altitude):
        # Simplified exponential atmosphere.
        # Not a high-altitude standard atmosphere.
        h = max(0.0, altitude)
        return 1.225 * math.exp(-h / 8500.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.altitude = 0.0
        self.velocity = 0.0
        self.fuel = self.initial_fuel
        self.time = 0.0
        self.steps = 0

        self.throttle = 0.0
        self.acceleration = 0.0
        self.dynamic_pressure = 0.0
        self.heating_proxy = 0.0
        self.max_altitude = 0.0
        self.last_reward = 0.0

        return self._observation(), self._info()

    def _observation(self):
        return np.array([
            self.altitude / self.target_altitude,
            self.velocity / 2000.0,
            self.fuel / self.initial_fuel,
            self.gravity(self.altitude) / self.G0,
            self.dynamic_pressure / 100_000.0,
            self.throttle,
            self.time / 1200.0,
        ], dtype=np.float32)

    def _info(self):
        mass = self.dry_mass + self.fuel
        g = self.gravity(self.altitude)

        return {
            "altitude_m": float(self.altitude),
            "vertical_speed_mps": float(self.velocity),
            "speed_mps": abs(float(self.velocity)),
            "acceleration_mps2": float(self.acceleration),
            "gravity_mps2": float(g),
            "mass_kg": float(mass),
            "fuel_kg": float(self.fuel),
            "throttle": float(self.throttle),
            "dynamic_pressure_pa": float(
                self.dynamic_pressure
            ),
            "heating_proxy": float(self.heating_proxy),
            "time_s": float(self.time),
            "max_altitude_m": float(self.max_altitude),
            "phase": (
                "ASCENT" if self.velocity >= 0
                else "DESCENT"
            ),
            "step": int(self.steps),
            "reward": float(self.last_reward),
        }

    def step(self, action):
        self.steps += 1
        self.time += self.dt

        self.throttle = float(
            np.clip(np.asarray(action).item(), 0, 1)
        )

        mass_before = self.dry_mass + self.fuel

        requested_thrust = (
            self.max_thrust * self.throttle
        )

        requested_burn = (
            requested_thrust
            / (self.isp * self.G0)
            * self.dt
        )

        actual_burn = min(
            self.fuel,
            requested_burn
        )

        self.fuel -= actual_burn

        thrust = (
            actual_burn
            * self.isp
            * self.G0
            / self.dt
        )

        mass = (
            mass_before
            - actual_burn / 2.0
        )

        rho = self.density(self.altitude)
        speed = abs(self.velocity)

        self.dynamic_pressure = (
            0.5 * rho * speed**2
        )

        drag = (
            0.5
            * rho
            * self.velocity
            * abs(self.velocity)
            * self.cd
            * self.area
        )

        gravity = self.gravity(self.altitude)

        self.acceleration = (
            (thrust - drag) / mass
            - gravity
        )

        previous_altitude = self.altitude

        self.velocity += (
            self.acceleration * self.dt
        )

        self.altitude += (
            self.velocity * self.dt
        )

        self.max_altitude = max(
            self.max_altitude,
            self.altitude
        )

        # Heating indicator only.
        # Not thermal protection temperature.
        self.heating_proxy = (
            math.sqrt(rho / 1.225)
            * abs(self.velocity)**3
        )

        altitude_gain = max(
            0.0,
            self.altitude - previous_altitude
        )

        # Encourage upward progress.
        reward = (
            altitude_gain / 1000.0
            - 0.01 * self.throttle
        )

        # Discourage excessive dynamic pressure.
        reward -= (
            max(
                0.0,
                self.dynamic_pressure - 50_000.0
            ) / 50_000.0
        ) * 0.1

        success = (
            self.altitude >= self.target_altitude
        )

        terminated = False

        if success:
            terminated = True
            reward += 500.0

        # Prevent underground motion.
        if self.altitude < 0:
            self.altitude = 0.0

            if self.max_altitude > 1.0:
                terminated = True
                reward -= 200.0
            else:
                self.velocity = 0.0

        # Fuel exhausted and descending.
        if (
            self.fuel <= 0
            and self.velocity < 0
            and not success
        ):
            terminated = True
            reward -= 100.0

        truncated = (
            self.steps >= self.max_steps
        )

        if truncated and not terminated:
            reward -= 100.0

        self.last_reward = float(reward)

        info = self._info()
        info["success"] = bool(success)

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
