
"""
ZERO SPACE EXP005
Stage 2 Precision Landing PPO Training

Run:
    python -m experiments.train_stage2_landing

Resume:
    python -m experiments.train_stage2_landing --resume

Quick test:
    python -m experiments.train_stage2_landing --steps 20000
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

from rocket.sixdof_env import SixDOFEnv


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "exp005" / "landing"

LOG_DIR = ROOT / "logs" / "exp005" / "landing"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class FlightTelemetryCallback(BaseCallback):
    """Print periodic training diagnostics."""

    def __init__(self, interval=10000, verbose=1):
        super().__init__(verbose)
        self.interval = interval
        self.episode_rewards = []
        self.episode_successes = []

    def _on_step(self):
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                episode = info["episode"]

                self.episode_rewards.append(
                    float(episode["r"])
                )

                # Monitor stores final environment info
                # in the same info dictionary.
                self.episode_successes.append(
                    float(info.get("success", False))
                )

        if (
            self.n_calls % self.interval == 0
            and self.verbose
        ):
            recent_rewards = self.episode_rewards[-100:]
            recent_successes = self.episode_successes[-100:]

            print("\n" + "=" * 60)
            print("ZERO SPACE EXP005 TRAINING")
            print("Timesteps:", self.num_timesteps)

            if recent_rewards:
                print(
                    "Average episode reward:",
                    round(float(np.mean(recent_rewards)), 2),
                )

                print(
                    "Observed success rate:",
                    round(
                        100 * float(np.mean(recent_successes)),
                        2,
                    ),
                    "%",
                )

            print("=" * 60 + "\n")

        return True


def make_env():
    def factory():
        env = SixDOFEnv()
        return Monitor(env)

    return factory


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("ZERO SPACE — EXP005")
    print("STAGE 2 PRECISION LANDING TRAINING")
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", args.device)
    print("Training steps:", args.steps)

    train_env = DummyVecEnv([
        make_env() for _ in range(4)
    ])

    eval_env = DummyVecEnv([
        make_env()
    ])

    normalization_path = (
        MODEL_DIR / "vecnormalize.pkl"
    )

    model_path = (
        MODEL_DIR / "landing_final.zip"
    )

    if args.resume:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model to resume: {model_path}"
            )

        if not normalization_path.exists():
            raise FileNotFoundError(
                f"No normalization data: {normalization_path}"
            )

        train_env = VecNormalize.load(
            str(normalization_path),
            train_env,
        )

        train_env.training = True
        train_env.norm_reward = True

        eval_env = VecNormalize.load(
            str(normalization_path),
            eval_env,
        )

        eval_env.training = False
        eval_env.norm_reward = False

        model = PPO.load(
            str(model_path),
            env=train_env,
            device=args.device,
        )

        reset_timesteps = False

        print("Resuming existing landing model.")

    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.99,
        )

        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=0.99,
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,

            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,

            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,

            ent_coef=0.005,
            vf_coef=0.5,

            max_grad_norm=0.5,

            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256, 128],
                    vf=[256, 256, 128],
                ),
                activation_fn=torch.nn.Tanh,
            ),

            verbose=1,
            tensorboard_log=str(LOG_DIR),
            device=args.device,
            seed=42,
        )

        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="landing_ppo",
        save_vecnormalize=True,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(
            MODEL_DIR / "best"
        ),
        log_path=str(
            LOG_DIR / "evaluation"
        ),
        eval_freq=20000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    telemetry_callback = FlightTelemetryCallback(
        interval=10000
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[
                checkpoint_callback,
                eval_callback,
                telemetry_callback,
            ],
            reset_num_timesteps=reset_timesteps,
            progress_bar=True,
        )

    finally:
        model.save(str(model_path))

        train_env.save(
            str(normalization_path)
        )

        # Save evaluation normalization statistics
        # synchronized from the training environment.
        eval_env.close()
        train_env.close()

        print("\nTraining checkpoint saved.")
        print("Model:", model_path)
        print("Normalization:", normalization_path)


if __name__ == "__main__":
    main()
