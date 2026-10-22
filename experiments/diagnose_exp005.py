
from pathlib import Path
import importlib
import numpy as np

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]

MODEL = (
    ROOT / "models" / "exp005" /
    "stage2" / "attitude_final.zip"
)

ENV_MODULE = "rocket.sixdof_env"
ENV_CLASS = "SixDOFEnv"


def run():
    module = importlib.import_module(ENV_MODULE)
    Env = getattr(module, ENV_CLASS)

    env = Env()
    print("\n=== ZERO EXP005 DIAGNOSTIC ===")
    print("Environment:", type(env).__name__)
    print("Observation:", env.observation_space)
    print("Actions:", env.action_space)

    if not MODEL.exists():
        print("MODEL NOT FOUND:", MODEL)
        print("Update MODEL to your trained Stage 2 path.")
        return

    model = PPO.load(str(MODEL))

    print("Model observation:", model.observation_space)
    print("Model actions:", model.action_space)

    assert (
        model.observation_space == env.observation_space
    ), "Observation space mismatch"

    assert (
        model.action_space == env.action_space
    ), "Action space mismatch"

    obs, info = env.reset(seed=42)

    initial_obs = np.asarray(obs, dtype=float)
    print("Initial observation:", initial_obs)
    print("Initial info:", info)

    assert np.all(np.isfinite(initial_obs)), (
        "Non-finite initial observation"
    )

    last_altitude = None
    last_position = None
    last_time = None
    finished = False

    for step in range(2000):
        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        values = np.asarray(obs, dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite observation at step {step}"
            )

        if not np.isfinite(reward):
            raise ValueError(
                f"Non-finite reward at step {step}"
            )

        altitude = info.get("altitude_m")
        position = info.get("position_m")
        time_s = info.get("time_s")

        if step % 100 == 0:
            print(
                f"STEP={step:04d}",
                f"T={time_s}",
                f"ALT={altitude}",
                f"VZ={info.get('vertical_velocity_mps')}",
                f"THR={info.get('throttle')}",
                f"PHASE={info.get('phase')}",
                f"REWARD={reward:.3f}",
            )

        if (
            altitude is not None
            and last_altitude is not None
            and abs(float(altitude) - last_altitude)
                > 100000
        ):
            print(
                "WARNING: large altitude jump",
                step
            )

        if (
            time_s is not None
            and last_time is not None
            and float(time_s) <= last_time
        ):
            print(
                "WARNING: time not increasing",
                step
            )

        if altitude is not None:
            last_altitude = float(altitude)

        if position is not None:
            last_position = position

        if time_s is not None:
            last_time = float(time_s)

        obs = values.astype(np.float32)

        if terminated or truncated:
            finished = True

            print("\nEPISODE ENDED")
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Final info:", info)
            break

    if not finished:
        print("\nWARNING: No termination within 2000 steps")

    print("Last position:", last_position)
    env.close()


if __name__ == "__main__":
    run()
