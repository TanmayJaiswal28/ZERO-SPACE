
from pathlib import Path

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from rocket.sixdof_env import SixDOFEnv


def main():
    output = Path("models/exp005/stage2")

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    env = Monitor(SixDOFEnv())
    evaluation = Monitor(SixDOFEnv())

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.995,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=str(output / "logs"),
    )

    callback = EvalCallback(
        evaluation,
        eval_freq=20000,
        n_eval_episodes=10,
        best_model_save_path=str(
            output / "best"
        ),
        verbose=1,
    )

    print()
    print("ZERO SPACE / EXP005 STAGE 2")
    print("TRAINING AUTONOMOUS ATTITUDE CONTROLLER")
    print("-" * 55)

    model.learn(
        total_timesteps=500_000,
        callback=callback,
    )

    model.save(
        str(output / "attitude_final")
    )

    env.close()
    evaluation.close()

    print()
    print("TRAINING COMPLETE")
    print(
        "Saved:",
        output / "attitude_final.zip"
    )


if __name__ == "__main__":
    main()
