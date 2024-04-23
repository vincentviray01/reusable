from trainer import Trainer
from datamodule import DataModule
from modelmodule import ModelModule
import wandb
from torchvision import datasets, transforms
import torch

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(project="test", config={"learning_rate": lr, "epochs": epochs})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(
    "../data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    "../data", train=False, download=True, transform=transform
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output


# model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
model = Net()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam
optimizer_lr = 0.001

batch_size = 10

trainer = Trainer(wandb, device)
model_module = ModelModule(model, loss_function, optimizer, optimizer_lr)
data_module = DataModule(train_dataset, test_dataset, batch_size)


print("ALL DONE")

trainer.train(model_module, data_module, 5)
