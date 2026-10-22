
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from rocket.orbital_env import OrbitalEnv


OUTPUT = Path("models/exp004")
OUTPUT.mkdir(parents=True, exist_ok=True)

train_env = Monitor(
    OrbitalEnv(),
    filename=str(OUTPUT / "training.csv"),
    info_keywords=("success", "reached_space"),
)

eval_env = Monitor(
    OrbitalEnv(),
    filename=str(OUTPUT / "evaluation.csv"),
    info_keywords=("success", "reached_space"),
)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=str(OUTPUT / "tensorboard"),
    policy_kwargs={
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128],
        }
    },
)

evaluation = EvalCallback(
    eval_env,
    best_model_save_path=str(OUTPUT),
    log_path=str(OUTPUT / "eval_logs"),
    eval_freq=25000,
    n_eval_episodes=10,
    deterministic=True,
)

checkpoint = CheckpointCallback(
    save_freq=100000,
    save_path=str(OUTPUT),
    name_prefix="ppo_orbital",
)

model.learn(
    total_timesteps=1_000_000,
    callback=CallbackList([
        evaluation,
        checkpoint,
    ]),
)

model.save(str(OUTPUT / "final_model"))

print("Training finished.")
print("Saved to models/exp004/final_model.zip")
