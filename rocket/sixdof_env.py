
"""ZERO SPACE EXP005 — Stage 2 6DOF flight environment.

Educational simulation, not validated flight software.

Compatibility:
    Environment class: SixDOFEnv
    Observation space: 26 floats
    Action space: 12 floats
    Gymnasium reset/step interface preserved

Coordinate systems:
    World: East, North, Up (ENU)
    Rocket: +Z toward nose
    Quaternion: [w, x, y, z], body to world
"""

import math
import numpy as np
import gymnasium as gym

from gymnasium import spaces


R_E = 6_371_000.0
MU = 3.986004418e14
G0 = 9.80665


def unit_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / norm


def quat_mul(a, b):
    w, x, y, z = a
    W, X, Y, Z = b

    return np.array([
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    ])


def rotate(q, v):
    q = unit_quat(q)

    return quat_mul(
        quat_mul(q, np.r_[0.0, v]),
        np.array([q[0], -q[1], -q[2], -q[3]]),
    )[1:]


def integrate_q(q, omega_body, dt):
    angle = np.linalg.norm(omega_body) * dt

    if angle < 1e-12:
        return q.copy()

    axis = omega_body / np.linalg.norm(omega_body)

    dq = np.r_[
        math.cos(angle / 2.0),
        axis * math.sin(angle / 2.0),
    ]

    return unit_quat(quat_mul(q, dq))


class SixDOFEnv(gym.Env):
    """
    ZERO SPACE EXP005.

    Actions:
        0       Throttle
        1-2     Engine gimbal
        3-6     Aerodynamic flaps
        7-10    Airbrakes
        11      RCS yaw

    Observation:
        0-2     Position / 2000
        3-5     Velocity / 100
        6-9     Orientation quaternion
        10-12   Angular velocity
        13      Fuel fraction
        14      Dynamic pressure / 50000
        15-18   Flap positions
        19-22   Airbrake positions
        23-24   Gimbal positions
        25      Time / 120

    Existing PPO interface remains unchanged.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.dt = 0.05
        self.max_steps = 2400

        self.dry_mass = 20_000.0
        self.initial_fuel = 8_000.0

        self.max_thrust = 550_000.0
        self.isp = 310.0

        self.area = 12.0
        self.cd = 0.55
        self.length = 22.0

        self.flap_area = 1.4
        self.flap_cl = 1.0

        self.max_flap = math.radians(30)
        self.max_gimbal = math.radians(5)

        self.max_flap_rate = math.radians(35)
        self.max_brake_rate = 0.8
        self.max_gimbal_rate = math.radians(20)

        self.rcs_torque = 45_000.0

        # Catcher location in the world frame.
        self.target = np.array([
            0.0,
            0.0,
            0.0,
        ], dtype=np.float64)

        # Landing tolerances.
        self.max_landing_error = 8.0
        self.max_touchdown_vz = 3.0
        self.max_touchdown_vxy = 2.0
        self.min_upright = 0.98

        # Reward weights.
        self.w_position_progress = 0.04
        self.w_horizontal_progress = 0.12
        self.w_vertical_speed = 0.035
        self.w_horizontal_speed = 0.018
        self.w_upright = 0.35
        self.w_angular_rate = 0.25
        self.w_fuel = 0.002
        self.w_time = 0.015

        self.success_bonus = 500.0
        self.impact_penalty = -250.0
        self.timeout_penalty = -150.0

        self.max_horizontal_range = 10_000.0
        self.max_altitude_limit = 12_000.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}

        self.p = np.array([
            0.0,
            0.0,
            self.np_random.uniform(1500, 2500),
        ], dtype=np.float64)

        self.v = np.array([
            self.np_random.uniform(-5, 5),
            self.np_random.uniform(-5, 5),
            self.np_random.uniform(-90, -50),
        ], dtype=np.float64)

        tilt = self.np_random.uniform(
            -0.15,
            0.15,
            size=3,
        )

        angle = np.linalg.norm(tilt)

        if angle > 1e-9:
            self.q = np.r_[
                math.cos(angle / 2),
                tilt / angle * math.sin(angle / 2),
            ]
        else:
            self.q = np.array([
                1.0, 0.0, 0.0, 0.0
            ])

        self.q = unit_quat(self.q)

        self.omega = self.np_random.uniform(
            -0.07,
            0.07,
            size=3,
        )

        self.fuel = self.initial_fuel

        self.flaps = np.zeros(4)
        self.brakes = np.zeros(4)
        self.gimbal = np.zeros(2)

        self.throttle = 0.0
        self.rcs = 0.0

        self.qdyn = 0.0
        self.a = np.zeros(3)
        self.torque = np.zeros(3)

        self.time = 0.0
        self.steps = 0

        self.max_altitude = self.p[2]
        self.total_reward = 0.0

        self.phase = "DESCENT"
        self.failure_reason = None

        # Optional deterministic test overrides.
        if "position" in options:
            self.p = np.asarray(
                options["position"],
                dtype=np.float64,
            ).copy()

        if "velocity" in options:
            self.v = np.asarray(
                options["velocity"],
                dtype=np.float64,
            ).copy()

        if "orientation" in options:
            self.q = unit_quat(
                options["orientation"]
            )

        if "fuel" in options:
            self.fuel = float(
                np.clip(
                    options["fuel"],
                    0.0,
                    self.initial_fuel,
                )
            )

        self.max_altitude = float(self.p[2])

        self.previous_distance = self._distance()
        self.previous_horizontal_error = (
            self._horizontal_error()
        )

        return self._obs(), self._info()

    def _obs(self):
        return np.r_[
            self.p / 2000.0,
            self.v / 100.0,
            self.q,
            self.omega,
            self.fuel / self.initial_fuel,
            self.qdyn / 50000.0,
            self.flaps / self.max_flap,
            self.brakes,
            self.gimbal / self.max_gimbal,
            self.time / 120.0,
        ].astype(np.float32)

    def _distance(self):
        return float(
            np.linalg.norm(self.p - self.target)
        )

    def _horizontal_error(self):
        return float(
            np.linalg.norm(
                self.p[:2] - self.target[:2]
            )
        )

    def _upright(self):
        body_up = rotate(
            self.q,
            np.array([0.0, 0.0, 1.0]),
        )

        return float(
            np.clip(body_up[2], -1.0, 1.0)
        )

    def _info(self):
        horizontal_error = self._horizontal_error()

        return dict(
            position_m=self.p.tolist(),
            velocity_mps=self.v.tolist(),
            altitude_m=float(max(self.p[2], 0.0)),
            speed_mps=float(np.linalg.norm(self.v)),
            vertical_velocity_mps=float(self.v[2]),
            acceleration_mps2=float(
                np.linalg.norm(self.a)
            ),
            orientation_quat=self.q.tolist(),
            angular_velocity_rad_s=self.omega.tolist(),
            fuel_kg=float(self.fuel),
            mass_kg=float(
                self.dry_mass + self.fuel
            ),
            throttle=float(self.throttle),
            flap_angles_deg=np.rad2deg(
                self.flaps
            ).tolist(),
            airbrake_deployment=self.brakes.tolist(),
            gimbal_angles_deg=np.rad2deg(
                self.gimbal
            ).tolist(),
            rcs_command=float(self.rcs),
            dynamic_pressure_pa=float(self.qdyn),
            torque_nm=self.torque.tolist(),
            time_s=float(self.time),
            phase=self.phase,
            max_altitude_m=float(self.max_altitude),
            total_reward=float(self.total_reward),

            # Additional mission telemetry.
            target_position_m=self.target.tolist(),
            distance_to_target_m=self._distance(),
            horizontal_error_m=horizontal_error,
            horizontal_speed_mps=float(
                np.linalg.norm(self.v[:2])
            ),
            upright_score=self._upright(),
            failure_reason=self.failure_reason,
        )

    def _update_actuators(self, u):
        dt = self.dt

        self.throttle = float(
            (u[0] + 1.0) / 2.0
        )

        flap_target = u[3:7] * self.max_flap
        brake_target = (u[7:11] + 1.0) / 2.0
        gimbal_target = u[1:3] * self.max_gimbal

        self.flaps += np.clip(
            flap_target - self.flaps,
            -self.max_flap_rate * dt,
            self.max_flap_rate * dt,
        )

        self.brakes += np.clip(
            brake_target - self.brakes,
            -self.max_brake_rate * dt,
            self.max_brake_rate * dt,
        )

        self.gimbal += np.clip(
            gimbal_target - self.gimbal,
            -self.max_gimbal_rate * dt,
            self.max_gimbal_rate * dt,
        )

        self.rcs = float(u[11])

    def _atmosphere(self):
        altitude = max(float(self.p[2]), 0.0)

        density = 1.225 * math.exp(
            -altitude / 8500.0
        )

        return density

    def _gravity(self):
        altitude = max(float(self.p[2]), 0.0)

        gravity_z = -MU / (
            R_E + altitude
        ) ** 2

        return np.array([
            0.0,
            0.0,
            gravity_z,
        ])

    def _aerodynamic_drag(self):
        density = self._atmosphere()
        speed = float(np.linalg.norm(self.v))

        self.qdyn = (
            0.5 * density * speed ** 2
        )

        drag_area = (
            self.cd * self.area
            + 4.0 * np.mean(self.brakes) * 3.0
        )

        if speed < 1e-9:
            return np.zeros(3)

        return (
            -self.qdyn
            * drag_area
            * self.v / speed
        )

    def _engine(self):
        dt = self.dt

        requested_thrust = (
            self.max_thrust * self.throttle
        )

        requested_burn = (
            requested_thrust * dt
            / (self.isp * G0)
        )

        burn = min(
            self.fuel,
            requested_burn,
        )

        mass_before = (
            self.dry_mass + self.fuel
        )

        self.fuel -= burn

        mass_after = (
            self.dry_mass + self.fuel
        )

        mass = (
            mass_before + mass_after
        ) / 2.0

        thrust = (
            burn * self.isp * G0 / dt
        )

        gx, gy = self.gimbal

        thrust_body = thrust * np.array([
            math.sin(gx),
            math.sin(gy),
            math.cos(gx) * math.cos(gy),
        ])

        thrust_world = rotate(
            self.q,
            thrust_body,
        )

        return (
            thrust_body,
            thrust_world,
            mass,
            burn,
        )

    def _aerodynamic_torque(self):
        torque = np.zeros(3)

        locations = [
            np.array([1.2, 0.0, 7.0]),
            np.array([-1.2, 0.0, 7.0]),
            np.array([0.0, 1.2, -7.0]),
            np.array([0.0, -1.2, -7.0]),
        ]

        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]

        for i in range(4):
            force = (
                -self.qdyn
                * self.flap_area
                * self.flap_cl
                * math.sin(self.flaps[i])
                * directions[i]
            )

            torque += np.cross(
                locations[i],
                force,
            )

        torque -= (
            self.qdyn
            * np.array([3.0, 3.0, 1.0])
            * self.omega
        )

        return torque

    def _integrate_rotation(self, thrust_body, mass):
        engine_torque = np.cross(
            np.array([0.0, 0.0, -9.0]),
            thrust_body,
        )

        self.torque = (
            self._aerodynamic_torque()
            + engine_torque
            + np.array([
                0.0,
                0.0,
                self.rcs_torque * self.rcs,
            ])
        )

        inertia = np.array([
            mass * self.length ** 2 / 12.0,
            mass * self.length ** 2 / 12.0,
            mass * 2.0,
        ])

        omega_dot = (
            self.torque
            - np.cross(
                self.omega,
                inertia * self.omega,
            )
        ) / inertia

        self.omega = np.clip(
            self.omega + omega_dot * self.dt,
            -10.0,
            10.0,
        )

        self.q = integrate_q(
            self.q,
            self.omega,
            self.dt,
        )

    def _calculate_reward(self, burn):
        distance = self._distance()
        horizontal_error = self._horizontal_error()

        position_progress = (
            self.previous_distance - distance
        )

        horizontal_progress = (
            self.previous_horizontal_error
            - horizontal_error
        )

        self.previous_distance = distance
        self.previous_horizontal_error = horizontal_error

        upright = self._upright()

        vertical_speed = float(self.v[2])
        horizontal_speed = float(
            np.linalg.norm(self.v[:2])
        )

        # A descending reference speed that decreases
        # as the vehicle approaches the target altitude.
        height = max(
            self.p[2] - self.target[2],
            0.0,
        )

        target_descent_speed = -min(
            45.0,
            max(1.5, 0.08 * height),
        )

        vertical_speed_error = abs(
            vertical_speed - target_descent_speed
        )

        # Progress-based rewards prevent the agent
        # from earning indefinitely by hovering.
        reward = (
            self.w_position_progress
            * position_progress

            + self.w_horizontal_progress
            * horizontal_progress

            - self.w_vertical_speed
            * vertical_speed_error

            - self.w_horizontal_speed
            * horizontal_speed

            + self.w_upright
            * upright

            - self.w_angular_rate
            * np.linalg.norm(self.omega)

            - self.w_fuel
            * burn

            - self.w_time
        )

        # Stronger horizontal tracking near the ground.
        if height < 300.0:
            reward -= (
                0.002 * horizontal_error
            )

        # Penalize climbing during the descent mission.
        if vertical_speed > 0.0:
            reward -= (
                0.08 * vertical_speed
            )

        # Penalize excessive descent speed close to ground.
        if height < 150.0:
            safe_speed = max(
                2.0,
                0.04 * height,
            )

            if vertical_speed < -safe_speed:
                reward -= (
                    0.1
                    * (
                        abs(vertical_speed)
                        - safe_speed
                    )
                )

        return float(reward)

    def _evaluate_landing(self):
        upright = self._upright()

        vertical_speed = abs(float(self.v[2]))
        horizontal_speed = float(
            np.linalg.norm(self.v[:2])
        )

        horizontal_error = (
            self._horizontal_error()
        )

        success = bool(
            vertical_speed <= self.max_touchdown_vz
            and horizontal_speed <= self.max_touchdown_vxy
            and horizontal_error <= self.max_landing_error
            and upright >= self.min_upright
        )

        return success

    def step(self, action):
        u = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if u.shape != (12,):
            raise ValueError(
                "Expected exactly 12 actions"
            )

        if not np.all(np.isfinite(u)):
            raise ValueError(
                "Action contains NaN or infinity"
            )

        u = np.clip(u, -1.0, 1.0)

        self.steps += 1

        self._update_actuators(u)

        drag = self._aerodynamic_drag()

        (
            thrust_body,
            thrust_world,
            mass,
            burn,
        ) = self._engine()

        self.a = (
            (thrust_world + drag) / mass
            + self._gravity()
        )

        self._integrate_rotation(
            thrust_body,
            mass,
        )

        self.v = (
            self.v + self.a * self.dt
        )

        self.p = (
            self.p + self.v * self.dt
        )

        self.time += self.dt

        self.max_altitude = max(
            self.max_altitude,
            float(self.p[2]),
        )

        reward = self._calculate_reward(burn)

        terminated = False
        truncated = False
        success = False

        self.failure_reason = None

        if self.p[2] <= self.target[2]:
            success = self._evaluate_landing()

            if success:
                reward += self.success_bonus
                self.phase = "LANDED"
            else:
                reward += self.impact_penalty
                self.phase = "IMPACT"
                self.failure_reason = (
                    "Landing constraints not satisfied"
                )

            self.p[2] = self.target[2]
            terminated = True

        elif (
            self._horizontal_error()
            > self.max_horizontal_range
        ):
            reward += self.timeout_penalty
            self.phase = "OUT_OF_BOUNDS"
            self.failure_reason = (
                "Exceeded horizontal range"
            )
            terminated = True

        elif self.p[2] > self.max_altitude_limit:
            reward += self.timeout_penalty
            self.phase = "OUT_OF_BOUNDS"
            self.failure_reason = (
                "Exceeded altitude limit"
            )
            terminated = True

        elif self.v[2] >= 0.0:
            self.phase = "ASCENT"

        elif self.p[2] < 150.0:
            self.phase = "TERMINAL_DESCENT"

        else:
            self.phase = "DESCENT"

        if self.steps >= self.max_steps:
            truncated = True

            if not terminated:
                reward += self.timeout_penalty
                self.phase = "TIMEOUT"
                self.failure_reason = (
                    "Episode time limit reached"
                )

        self.total_reward += reward

        info = self._info()
        info["success"] = success

        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )
