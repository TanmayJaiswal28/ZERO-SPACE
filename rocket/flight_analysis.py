from dataclasses import dataclass, asdict
from rocket.observations import RocketState


@dataclass
class FlightResult:

    model_name: str
    episode_number: int

    original_reward: float
    training_reward: float

    flight_steps: int

    final_horizontal_position: float
    final_vertical_velocity: float
    final_angle: float

    both_legs_touching: bool

    terminated: bool
    truncated: bool

    @property
    def reached_solved_score(self) -> bool:

        return self.original_reward >= 200

    def to_dictionary(self):

        result = asdict(self)

        result["reached_solved_score"] = (
            self.reached_solved_score
        )

        return result


def create_flight_result(
    model_name: str,
    episode_number: int,
    original_reward: float,
    training_reward: float,
    flight_steps: int,
    final_observation,
    terminated: bool,
    truncated: bool,
):

    rocket_state = RocketState.from_observation(
        final_observation
    )

    return FlightResult(
        model_name=model_name,
        episode_number=episode_number,

        original_reward=original_reward,
        training_reward=training_reward,

        flight_steps=flight_steps,

        final_horizontal_position=rocket_state.horizontal_position,
        final_vertical_velocity=rocket_state.vertical_velocity,
        final_angle=rocket_state.angle,

        both_legs_touching=rocket_state.both_legs_touching,

        terminated=terminated,
        truncated=truncated,
    )