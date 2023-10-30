"""
Example dummy dataset & datamodule. This is just a dataset where each sample x has a corresponding label y, where x = y.
(meaning, we just want to learn the identity function f(x) = x)
"""
import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    """Dummy dataset, x = y = from 0 to range"""
    def __init__(self, n):
        super().__init__()
        print("-- USING A DUMMY DATASET --")
        # x, y are floats from 0 to range
        self.n = n
        self.x = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        self.y = torch.arange(n, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, n, batch_size):
        super().__init__()
        self.dummy_dataset = DummyDataset(n)
        self.bs = batch_size

    def setup(self, stage):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.bs, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.bs, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dummy_dataset, batch_size=self.bs, shuffle=False)
    