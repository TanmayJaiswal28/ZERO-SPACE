
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from rocket.suborbital_env import SuborbitalEnv


ROOT = Path("models/exp005")
ROOT.mkdir(parents=True, exist_ok=True)


def train_ascent():
    model = None

    stages = [
        (10_000, 150_000),
        (30_000, 200_000),
        (60_000, 250_000),
        (100_000, 350_000),
    ]

    for stage, (target, timesteps) in enumerate(
        stages, start=1
    ):
        print(
            f"\nASCENT STAGE {stage}: "
            f"{target / 1000:.0f} km"
        )

        env = Monitor(
            SuborbitalEnv(
                mode="ascent",
                target_altitude=target
            )
        )

        eval_env = Monitor(
            SuborbitalEnv(
                mode="ascent",
                target_altitude=target
            )
        )

        if model is None:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=128,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=0.005,
                verbose=1,
                tensorboard_log=str(
                    ROOT / "tensorboard"
                ),
            )
        else:
            model.set_env(env)

        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(
                ROOT / f"ascent_stage_{stage}_best"
            ),
            eval_freq=20_000,
            n_eval_episodes=5,
            deterministic=True,
        )

        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            callback=callback,
        )

        model.save(
            str(ROOT / f"ascent_stage_{stage}")
        )

        env.close()
        eval_env.close()

    model.save(str(ROOT / "ascent_final"))


def train_recovery():
    print("\nTRAINING RECOVERY PPO")

    env = Monitor(
        SuborbitalEnv(mode="recovery")
    )

    eval_env = Monitor(
        SuborbitalEnv(mode="recovery")
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(
            ROOT / "tensorboard"
        ),
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(
            ROOT / "recovery_best"
        ),
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=600_000,
        callback=callback,
    )

    model.save(str(ROOT / "recovery_final"))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_ascent()
    train_recovery()

    print("\nEXP005 TRAINING COMPLETE")
    print("Run evaluation before declaring success.")
