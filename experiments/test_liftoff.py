
import numpy as np

from rocket.orbital_env import OrbitalEnv


def main():
    env = OrbitalEnv()

    observation, info = env.reset()

    print("\nZERO SPACE / LIFTOFF TEST")
    print("-" * 55)

    print(
        f"Initial altitude: {info['altitude_m']:.2f} m"
    )

    print(
        f"Initial fuel: {info['fuel_kg']:.0f} kg"
    )

    # Full throttle, radial outward direction.
    # This is a physics diagnostic, not a trained policy.
    action = np.array(
        [1.0, 1.0, 0.0, 0.0],
        dtype=np.float32
    )

    for step in range(60):
        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        if step % 5 == 0:
            print(
                f"T+{info['time_s']:5.1f}s | "
                f"Altitude: {info['altitude_m']:9.2f}m | "
                f"Speed: {info['speed_mps']:8.2f}m/s | "
                f"Throttle: {info['throttle'] * 100:5.0f}% | "
                f"Fuel: {info['fuel_kg']:9.0f}kg"
            )

        if terminated or truncated:
            print("Simulation ended:", info["phase"])
            break

    if info["max_altitude_m"] > 10:
        print("\nPASS: Physics engine supports liftoff.")
    else:
        print("\nFAIL: Rocket did not leave the pad.")

    env.close()


if __name__ == "__main__":
    main()
