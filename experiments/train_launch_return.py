
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from rocket.launch_return_env import LaunchReturnEnv


OUTPUT = Path("models/exp002")
OUTPUT.mkdir(parents=True, exist_ok=True)

env = Monitor(
    LaunchReturnEnv(),
    filename=str(OUTPUT / "training.csv"),
    info_keywords=("success",),
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.995,
    tensorboard_log=str(OUTPUT / "tensorboard"),
    policy_kwargs={
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128],
        }
    },
)

checkpoint = CheckpointCallback(
    save_freq=100_000,
    save_path=str(OUTPUT),
    name_prefix="ppo",
)

model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint,
)

model.save(str(OUTPUT / "final_model"))

print("Training complete.")
print("Saved:", OUTPUT / "final_model.zip")
