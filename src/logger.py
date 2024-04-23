from abc import ABC, abstractmethod


class Logger(ABC):

    @abstractmethod
    def log_artifact(self, artifact):
        pass

    def log(self, item: dict):
        pass

    def log_model(self, model):
        pass
