
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from rocket.earth_ascent_env import EarthAscentEnv


ROOT = Path("models/exp003")
ROOT.mkdir(parents=True, exist_ok=True)

train_env = Monitor(
    EarthAscentEnv(),
    filename=str(ROOT / "train_monitor.csv"),
    info_keywords=("success",),
)

eval_env = Monitor(EarthAscentEnv())

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.995,
    verbose=1,
    device="cpu",
    tensorboard_log=str(ROOT / "tensorboard"),
    policy_kwargs={
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128],
        }
    },
)

evaluation = EvalCallback(
    eval_env,
    best_model_save_path=str(ROOT),
    log_path=str(ROOT / "evaluation"),
    eval_freq=25000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

checkpoints = CheckpointCallback(
    save_freq=100000,
    save_path=str(ROOT),
    name_prefix="checkpoint",
)

model.learn(
    total_timesteps=1_000_000,
    callback=CallbackList([
        evaluation,
        checkpoints,
    ]),
)

model.save(str(ROOT / "final_model"))

print("Training complete.")
print("Model saved to models/exp003/final_model.zip")
