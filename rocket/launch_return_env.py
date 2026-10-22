
"""ZERO SPACE EXP002: Educational 2D launch and return simulator.

Uses SI units internally. Not flight software.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LaunchReturnEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.dt = 0.1
        self.g = 9.80665

        self.dry_mass = 16000.0
        self.fuel_initial = 14000.0
        self.max_thrust = 620000.0
        self.isp = 290.0

        self.area = 9.0
        self.cd = 0.45

        self.target_altitude = 1200.0
        self.max_steps = 3200

        # Action 0: engine throttle, from 0 to 1.
        # Action 1: idealized lateral control, from -1 to 1.
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.x = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vz = 0.0

        self.fuel = self.fuel_initial

        self.phase = 0
        self.max_altitude = 0.0
        self.step_count = 0

        self.last_throttle = 0.0
        self.last_lateral = 0.0
        self.q = 0.0

        self.heat_flux_proxy = 0.0
        self.heat_load_proxy = 0.0
        self.last_reward = 0.0
        self.accel = 0.0

        return self._obs(), self._info()

    def _obs(self):
        return np.array([
            self.x / 300.0,
            self.z / 1500.0,
            self.vx / 100.0,
            self.vz / 200.0,
            self.fuel / self.fuel_initial,
            float(self.phase),
            self.q / 50000.0,
            self.heat_flux_proxy / 1e7,
            (self.target_altitude - self.z) / 1500.0,
            self.step_count / self.max_steps,
        ], dtype=np.float32)

    def _info(self):
        mass = self.dry_mass + self.fuel

        return {
            "x_m": self.x,
            "altitude_m": max(0.0, self.z),
            "vx_mps": self.vx,
            "vz_mps": self.vz,
            "speed_mps": math.hypot(self.vx, self.vz),
            "acceleration_mps2": self.accel,
            "fuel_kg": self.fuel,
            "mass_kg": mass,
            "throttle": self.last_throttle,
            "lateral_command": self.last_lateral,
            "phase": "ASCENT" if self.phase == 0 else "RETURN",
            "dynamic_pressure_pa": self.q,
            "heat_flux_proxy": self.heat_flux_proxy,
            "heat_load_proxy": self.heat_load_proxy,
            "max_altitude_m": self.max_altitude,
            "step": self.step_count,
            "time_s": self.step_count * self.dt,
            "reward": self.last_reward,
        }

    def step(self, action):
        throttle, lateral = np.clip(
            np.asarray(action, dtype=float),
            self.action_space.low,
            self.action_space.high,
        )

        self.step_count += 1
        self.last_throttle = float(throttle)
        self.last_lateral = float(lateral)

        mass = self.dry_mass + self.fuel

        # Engine thrust and fuel burn
        thrust = (
            self.max_thrust * float(throttle)
            if self.fuel > 0
            else 0.0
        )

        fuel_burn = min(
            self.fuel,
            thrust / (self.isp * self.g) * self.dt,
        )

        self.fuel -= fuel_burn

        thrust = min(
            thrust,
            fuel_burn * self.isp * self.g / self.dt,
        )

        # Exponential atmosphere approximation
        rho = 1.225 * math.exp(
            -max(0.0, self.z) / 8500.0
        )

        speed = math.hypot(self.vx, self.vz)

        self.q = 0.5 * rho * speed * speed

        drag = self.q * self.cd * self.area

        drag_x = -drag * self.vx / max(speed, 1e-6)
        drag_z = -drag * self.vz / max(speed, 1e-6)

        # Idealized lateral control.
        # This is not real grid-fin aerodynamics.
        lateral_force = (
            lateral * 0.05 * self.max_thrust
            if self.fuel > 0
            else 0.0
        )

        ax = (lateral_force + drag_x) / mass
        az = (thrust + drag_z) / mass - self.g

        self.accel = math.hypot(ax, az)

        # Numerical physics integration
        self.vx += ax * self.dt
        self.vz += az * self.dt

        self.x += self.vx * self.dt
        self.z += self.vz * self.dt

        # Simplified ground support during launch
        if self.phase == 0 and self.max_altitude == 0:
            if self.z < 0:
                self.z = 0.0
                self.vz = 0.0

        self.max_altitude = max(
            self.max_altitude,
            self.z,
        )

        # Qualitative heating indicator, not temperature
        self.heat_flux_proxy = (
            math.sqrt(rho / 1.225) * speed**3
        )

        self.heat_load_proxy += (
            self.heat_flux_proxy * self.dt
        )

        if (
            self.phase == 0
            and self.max_altitude >= self.target_altitude
        ):
            self.phase = 1

        # Reward function
        if self.phase == 0:
            reward = (
                0.003 * max(0.0, self.vz) * self.dt
                - 0.00001 * abs(self.x)
            )
        else:
            reward = (
                -0.0004 * abs(self.x)
                - 0.00015 * abs(self.vx)
                - 0.00006 * abs(self.vz)
                - 0.00005 * max(0.0, self.z)
            ) * self.dt

        reward -= 0.00004 * fuel_burn

        reward -= (
            0.000000005
            * max(0.0, self.q - 35000.0)
        )

        reward -= (
            0.0000000005
            * self.heat_flux_proxy
        )

        terminated = False
        truncated = self.step_count >= self.max_steps
        success = False

        # Touchdown
        if (
            self.z <= 0
            and self.step_count > 1
            and self.max_altitude > 0
        ):
            self.z = 0.0
            terminated = True

            success = (
                self.phase == 1
                and abs(self.x) < 12.0
                and abs(self.vx) < 2.0
                and abs(self.vz) < 3.0
            )

            reward += 500.0 if success else -200.0

        # Out-of-bounds failure
        if abs(self.x) > 2500.0 or self.z > 8000.0:
            terminated = True
            reward -= 200.0

        if truncated and not terminated:
            reward -= 100.0

        self.last_reward = float(reward)

        info = self._info()
        info["success"] = success
        info["terminated"] = terminated
        info["truncated"] = truncated

        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )
