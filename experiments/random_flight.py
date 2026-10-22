from rocket.environment import create_rocket_environment
from rocket.observations import RocketState


def run_random_flight(number_of_flights: int = 5):

    environment = create_rocket_environment(
        render_mode="human"
    )

    try:

        for flight_number in range(1, number_of_flights + 1):

            observation, information = environment.reset()

            flight_finished = False
            total_reward = 0.0
            flight_steps = 0

            print(f"\nStarting random flight {flight_number}")

            while not flight_finished:

                action = environment.action_space.sample()

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    information,
                ) = environment.step(action)

                rocket_state = RocketState.from_observation(
                    observation
                )

                total_reward += reward
                flight_steps += 1

                flight_finished = terminated or truncated

                if flight_steps % 100 == 0:

                    print(
                        f"Step {flight_steps} | "
                        f"Height: {rocket_state.vertical_position:.3f} | "
                        f"Speed: {rocket_state.speed:.3f} | "
                        f"Tilt: {rocket_state.angle:.3f}"
                    )

            print(
                f"Flight completed | "
                f"Steps: {flight_steps} | "
                f"Reward: {total_reward:.2f}"
            )

    finally:
        environment.close()


if __name__ == "__main__":
    run_random_flight()