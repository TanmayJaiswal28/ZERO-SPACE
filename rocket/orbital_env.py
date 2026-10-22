
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from physics.earth3d import (
    EARTH_RADIUS,
    EARTH_MU,
    STANDARD_GRAVITY,
    altitude,
    norm,
    unit,
    gravity,
    atmosphere_velocity,
    orbital_elements,
    integrate_rk4,
)


class OrbitalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.dt = 0.5
        self.max_steps = 3000

        # Hypothetical research vehicle.
        self.dry_mass = 20000.0
        self.initial_fuel = 180000.0
        self.max_thrust = 5_000_000.0
        self.isp = 340.0

        self.cd = 0.35
        self.area = 10.0

        self.target_altitude = 160000.0

        # Throttle + 3D direction command
        self.action_space = spaces.Box(
            low=np.array([0, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Launch from the equator, initially moving
        # with the rotating surface of Earth.
        self.position = np.array(
            [EARTH_RADIUS, 0.0, 0.0],
            dtype=float
        )

        self.velocity = atmosphere_velocity(self.position)

        self.fuel = self.initial_fuel
        self.steps = 0
        self.time = 0.0

        self.throttle = 0.0
        self.thrust_direction = np.array(
            [1.0, 0.0, 0.0]
        )

        self.acceleration = np.zeros(3)
        self.dynamic_pressure = 0.0
        self.heating_proxy = 0.0
        self.max_altitude = 0.0
        self.total_reward = 0.0
        self.phase = "ASCENT"

        return self._observation(), self._info()

    def _observation(self):
        radial = unit(self.position)
        tangential = unit(
            self.velocity
            - np.dot(self.velocity, radial) * radial
        )

        radial_speed = float(
            np.dot(self.velocity, radial)
        )

        return np.array([
            altitude(self.position) / 200000,
            radial_speed / 3000,
            norm(self.velocity) / 8000,
            self.fuel / self.initial_fuel,
            self.dynamic_pressure / 100000,
            radial[0],
            radial[1],
            radial[2],
            tangential[0],
            tangential[1],
            tangential[2],
            self.throttle,
            self.time / 1500,
        ], dtype=np.float32)

    def _info(self):
        elements = orbital_elements(
            self.position,
            self.velocity
        )

        return {
            "position_m": self.position.tolist(),
            "velocity_mps": self.velocity.tolist(),
            "altitude_m": max(
                0.0, altitude(self.position)
            ),
            "speed_mps": norm(self.velocity),
            "acceleration_mps2": norm(self.acceleration),
            "fuel_kg": self.fuel,
            "mass_kg": self.dry_mass + self.fuel,
            "throttle": self.throttle,
            "thrust_direction": self.thrust_direction.tolist(),
            "dynamic_pressure_pa": self.dynamic_pressure,
            "heating_proxy": self.heating_proxy,
            "max_altitude_m": self.max_altitude,
            "phase": self.phase,
            "time_s": self.time,
            "step": self.steps,
            "total_reward": self.total_reward,
            **elements,
        }

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=float),
            self.action_space.low,
            self.action_space.high,
        )

        self.steps += 1
        self.time += self.dt

        self.throttle = float(action[0])

        requested_direction = unit(action[1:4])

        if norm(requested_direction) < 1e-9:
            requested_direction = unit(self.position)

        self.thrust_direction = requested_direction

        mass_before = self.dry_mass + self.fuel

        requested_thrust = (
            self.max_thrust * self.throttle
        )

        fuel_needed = (
            requested_thrust * self.dt
            / (self.isp * STANDARD_GRAVITY)
        )

        fuel_used = min(
            self.fuel,
            fuel_needed
        )

        self.fuel -= fuel_used

        actual_thrust = (
            fuel_used
            * self.isp
            * STANDARD_GRAVITY
            / self.dt
        )

        thrust_vector = (
            self.thrust_direction * actual_thrust
        )

        mass_average = mass_before - fuel_used / 2

        previous_altitude = altitude(self.position)

        (
            new_position,
            new_velocity,
            self.acceleration,
            self.dynamic_pressure,
            self.heating_proxy,
        ) = integrate_rk4(
            self.position,
            self.velocity,
            mass_average,
            thrust_vector,
            self.cd,
            self.area,
            self.dt
        )

        self.position = new_position
        self.velocity = new_velocity

        h = altitude(self.position)

        self.max_altitude = max(
            self.max_altitude,
            h
        )

        elements = orbital_elements(
            self.position,
            self.velocity
        )

        # Training objective:
        # make progress toward altitude target,
        # while discouraging excessive pressure/fuel.
        progress = (
            h - previous_altitude
        ) / 1000

        reward = (
            0.1 * progress
            - 0.00002 * fuel_used
            - 0.01 * self.throttle
        )

        if self.dynamic_pressure > 60000:
            reward -= (
                (self.dynamic_pressure - 60000)
                / 60000
            ) * 0.2

        reached_altitude = (
            h >= self.target_altitude
        )

        # Diagnostic only: bound orbit above atmosphere.
        stable_orbit = (
            elements["periapsis_m"] is not None
            and elements["periapsis_m"] >= 120000
            and elements["apoapsis_m"] is not None
            and elements["apoapsis_m"] <= 500000
        )

        terminated = False
        success = False

        if stable_orbit:
            terminated = True
            success = True
            self.phase = "ORBIT"
            reward += 1000

        elif h <= 0 and self.max_altitude > 10:
            terminated = True
            self.phase = "IMPACT"
            reward -= 300

        elif h > 600000:
            terminated = True
            self.phase = "HIGH ALTITUDE"
            reward -= 100

        elif reached_altitude:
            self.phase = "SPACE ASCENT"

        else:
            self.phase = (
                "ASCENT"
                if np.dot(
                    self.velocity,
                    unit(self.position)
                ) >= 0
                else "DESCENT"
            )

        # Pad support before initial liftoff.
        if h < 0 and self.max_altitude <= 10:
            self.position = (
                unit(self.position) * EARTH_RADIUS
            )
            self.velocity = atmosphere_velocity(
                self.position
            )

        truncated = self.steps >= self.max_steps

        self.total_reward += float(reward)

        info = self._info()
        info["success"] = success
        info["reached_space"] = reached_altitude

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )
