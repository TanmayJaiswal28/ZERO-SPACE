from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)


class TrainingProgressCallback(BaseCallback):

    def __init__(
        self,
        report_interval: int = 25_000,
        verbose: int = 0,
    ):

        super().__init__(verbose)

        self.report_interval = report_interval
        self.next_report = report_interval

    def _on_step(self) -> bool:

        if self.num_timesteps >= self.next_report:

            progress = (
                self.num_timesteps
                / self.model._total_timesteps
            ) * 100

            print(
                f"\n[ZERO SPACE] "
                f"Training progress: {progress:.1f}% "
                f"| Steps: {self.num_timesteps:,}"
            )

            self.next_report += self.report_interval

        return True


def create_training_callbacks(
    experiment_name: str,
    checkpoint_interval: int,
):

    checkpoint_directory = (
        Path("models")
        / experiment_name
        / "checkpoints"
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=str(checkpoint_directory),
        name_prefix="rocket_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    progress_callback = TrainingProgressCallback()

    return [
        checkpoint_callback,
        progress_callback,
    ]