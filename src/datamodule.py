from torch.utils.data import Dataset, DataLoader


class DataModule:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, batch_size):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self):
        pass
