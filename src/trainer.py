import torch
import logging
from logger import Logger
from datamodule import DataModule
from modelmodule import ModelModule
from tqdm import tqdm, trange
from typing import Any
import wandb


class Trainer:
    """
    Use Case:
    >> model = ResNet()
    >> trainer = Trainer()
    >> trainer.fit(model, datamodule)

    """

    def __init__(self, logger: Logger, device) -> None:
        self.set_logger(logger)
        self.device = device

    def set_logger(self, logger):
        self.logger = logger

    def train(
        self,
        model_module: ModelModule,
        data_module: DataModule,
        model_save_path: str,
        num_epochs: int,
        start_epoch: int = 0,
    ):
        torch.set_grad_enabled(True)
        model_module.model.to(self.device)

        train_dataloader = data_module.train_dataloader()
        self.before_train_hooks()
        for epoch in trange(start_epoch, num_epochs):
            self.before_epoch_hooks(epoch)
            for batch in tqdm(train_dataloader):
                self.before_train_batch_hooks()
                loss = model_module.training_step(batch, self.device)
                self.after_train_batch_hooks({"loss": loss})
            artifact = wandb.Artifact("artifact.txt", type="file")
            self.after_epoch_hooks(model_module, model_save_path, artifact, epoch)
        self.after_train_hooks()

    def save_model_to_local(self, save_path):
        pass

    def before_epoch_hooks(self, epoch):
        logging.info(f"Starting epoch: {epoch}")

    def before_train_batch_hooks(self):
        pass

    def before_train_hooks(self):
        logging.info("Training has started.")

    def after_epoch_hooks(
        self, model: ModelModule, model_save_path: str, artifact: Any, epoch: int
    ):
        model.save(model_save_path + epoch)
        self.logger.log_model(model_save_path)
        self.logger.log_artifact(artifact)

    def after_train_batch_hooks(self, item: dict[str, float]):
        self.logger.log(item)

    def after_train_hooks(self):
        logging.info("Training has ended.")
