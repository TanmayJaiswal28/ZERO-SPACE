
import numpy as np
from stable_baselines3 import PPO

from rocket.suborbital_env import SuborbitalEnv


def evaluate(model_path, mode, episodes=20):
    model = PPO.load(model_path, device="cpu")

    successes = 0
    max_altitudes = []
    touchdown_speeds = []

    for episode in range(episodes):
        env = SuborbitalEnv(
            mode=mode,
            target_altitude=100_000
        )

        obs, info = env.reset(seed=episode)

        for _ in range(env.max_steps):
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated or truncated:
                break

        successes += int(info.get("success", False))

        max_altitudes.append(
            info["max_altitude_m"]
        )

        if info.get("touchdown_speed_mps") is not None:
            touchdown_speeds.append(
                info["touchdown_speed_mps"]
            )

        print(
            f"{mode.upper()} "
            f"episode {episode + 1:02d} | "
            f"success={info.get('success', False)} | "
            f"phase={info['phase']} | "
            f"altitude={info['max_altitude_m']:.0f} m"
        )

        env.close()

    print("\n" + "=" * 50)
    print("MODE:", mode)
    print("SUCCESS RATE:", f"{successes / episodes * 100:.1f}%")
    print(
        "MEAN MAX ALTITUDE:",
        f"{np.mean(max_altitudes):.1f} m"
    )

    if touchdown_speeds:
        print(
            "MEAN TOUCHDOWN SPEED:",
            f"{np.mean(touchdown_speeds):.2f} m/s"
        )


if __name__ == "__main__":
    evaluate(
        "models/exp005/ascent_final.zip",
        "ascent",
    )

    evaluate(
        "models/exp005/recovery_final.zip",
        "recovery",
    )
