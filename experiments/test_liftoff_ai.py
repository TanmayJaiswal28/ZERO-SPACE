
from stable_baselines3 import PPO
from rocket.liftoff_env import LiftoffEnv


MODEL = "models/exp004/liftoff/final_liftoff.zip"

model = PPO.load(MODEL, device="cpu")

env = LiftoffEnv(target_altitude=10000.0)

obs, info = env.reset()

print("\nZERO SPACE / TRAINED LIFTOFF TEST")
print("-" * 65)

for step in range(600):
    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    if step % 20 == 0 or terminated or truncated:
        print(
            f"T+{info['time_s']:6.1f}s | "
            f"Altitude {info['altitude_m']:9.1f} m | "
            f"Throttle {info['throttle'] * 100:5.1f}% | "
            f"Reward {reward:8.3f} | "
            f"Phase {info['phase']}"
        )

    if terminated or truncated:
        break

print("-" * 65)
print("Maximum altitude:", info["max_altitude_m"])
print("Success:", info["success"])

env.close()
