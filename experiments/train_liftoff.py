
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from rocket.liftoff_env import LiftoffEnv


OUTPUT = Path("models/exp004/liftoff")
OUTPUT.mkdir(parents=True, exist_ok=True)

STAGES = [
    (100.0, 100_000),
    (1000.0, 200_000),
    (10000.0, 300_000),
]

model = None

for stage_number, (target, steps) in enumerate(
    STAGES,
    start=1
):
    print()
    print("=" * 55)
    print(f"ZERO SPACE / LIFTOFF STAGE {stage_number}")
    print(f"Target altitude: {target:,.0f} m")
    print(f"Training steps: {steps:,}")
    print("=" * 55)

    env = Monitor(
        LiftoffEnv(target_altitude=target)
    )

    eval_env = Monitor(
        LiftoffEnv(target_altitude=target)
    )

    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            gamma=0.995,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(
                OUTPUT / "tensorboard"
            ),
        )
    else:
        model.set_env(env)

    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=str(
            OUTPUT / f"stage_{stage_number}_best"
        ),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        callback=evaluator,
    )

    model.save(
        str(OUTPUT / f"stage_{stage_number}")
    )

    env.close()
    eval_env.close()

model.save(str(OUTPUT / "final_liftoff"))

print()
print("TRAINING COMPLETE")
print("Model: models/exp004/liftoff/final_liftoff.zip")
