from stable_baselines3 import PPO

from configs.experiment import DEFAULT_CONFIG


def create_rocket_agent(environment):

    configuration = DEFAULT_CONFIG

    policy_architecture = {
        "net_arch": {
            "pi": [64, 64],
            "vf": [64, 64],
        }
    }

    agent = PPO(
        policy="MlpPolicy",
        env=environment,

        learning_rate=configuration.learning_rate,
        n_steps=configuration.rollout_steps,
        batch_size=configuration.batch_size,
        n_epochs=configuration.optimization_epochs,

        gamma=configuration.discount_factor,
        gae_lambda=configuration.gae_lambda,

        clip_range=configuration.clipping_range,
        ent_coef=configuration.entropy_coefficient,

        policy_kwargs=policy_architecture,

        seed=configuration.random_seed,

        verbose=1,
        device="auto",
        tensorboard_log="logs/tensorboard",
    )

    return agent