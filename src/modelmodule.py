import torch


class ModelModule:
    def __init__(self, model: torch.nn.Module, loss_function, optimizer, optimizer_lr):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr

        self.configure_optimizers()

    def forward(self, data: torch.Tensor):
        output = self.model(data)
        return output

    def training_step(self, batch, device):
        data, labels = batch

        data = data.to(device)
        labels = labels.to(device)

        output = self.forward(data)
        output = output.to(torch.float)
        labels = labels.to(torch.float)
        loss = self.loss_function(output, labels)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def load_model(self, model_load_path):
        self.model.load_state_dict(torch.load(model_load_path))

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.optimizer_lr)
