import wandb


class WandB_Logger:
    def __init__(self, project: str, config: dict) -> None:
        self.run = wandb.init(project=project, config=config)
