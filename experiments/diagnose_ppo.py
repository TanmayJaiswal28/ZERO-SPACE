
import numpy as np
from stable_baselines3 import PPO

from rocket.orbital_env import OrbitalEnv


env = OrbitalEnv()

model = PPO.load(
    "models/exp004/final_model.zip",
    device="cpu"
)

obs, info = env.reset()

print("\nZERO SPACE — PPO DIAGNOSTIC")
print("-" * 60)

for step in range(20):
    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    print(
        f"Step {step:02d} | "
        f"Throttle: {info['throttle'] * 100:6.2f}% | "
        f"Altitude: {info['altitude_m']:9.2f} m | "
        f"Fuel: {info['fuel_kg']:9.0f} kg | "
        f"Reward: {reward:8.4f}"
    )

    if terminated or truncated:
        break

env.close()
