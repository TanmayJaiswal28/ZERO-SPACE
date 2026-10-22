from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RocketState:
    """
    A readable representation of the lander's observation.

    Position and velocity values use Gymnasium's normalized
    observation coordinates, not physical SI units.
    """

    horizontal_position: float
    vertical_position: float

    horizontal_velocity: float
    vertical_velocity: float

    angle: float
    angular_velocity: float

    left_leg_contact: bool
    right_leg_contact: bool

    @classmethod
    def from_observation(cls, observation: np.ndarray):

        if len(observation) != 8:
            raise ValueError(
                f"Expected 8 observation values, received {len(observation)}"
            )

        return cls(
            horizontal_position=float(observation[0]),
            vertical_position=float(observation[1]),

            horizontal_velocity=float(observation[2]),
            vertical_velocity=float(observation[3]),

            angle=float(observation[4]),
            angular_velocity=float(observation[5]),

            left_leg_contact=bool(observation[6]),
            right_leg_contact=bool(observation[7]),
        )

    @property
    def distance_from_target(self) -> float:

        return math.hypot(
            self.horizontal_position,
            self.vertical_position
        )

    @property
    def speed(self) -> float:

        return math.hypot(
            self.horizontal_velocity,
            self.vertical_velocity
        )

    @property
    def is_upright(self) -> bool:

        return abs(self.angle) < 0.1

    @property
    def is_moving_slowly(self) -> bool:

        return (
            abs(self.horizontal_velocity) < 0.2
            and abs(self.vertical_velocity) < 0.2
        )

    @property
    def both_legs_touching(self) -> bool:

        return (
            self.left_leg_contact
            and self.right_leg_contact
        )