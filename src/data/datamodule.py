import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(
        self, data_path, batch_size,
        num_workers, image_size,
        mean_value=None, std_value=None,
    ):
        super().__init__()
        
        self._mean_value = mean_value
        self._std_value = std_value

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train = None
        self.val = None
        self.test = None

        stats_dataset = Dataset(
            self, split='train',
            mean_value=self._mean_value,
            std_value=self._std_value,
        )
        self.num_classes = stats_dataset.num_classes
        self.class2id = stats_dataset.class2id
        self.id2class = stats_dataset.id2class
        self.class_weights = stats_dataset.class_weights

    def setup(self, stage: str):
        if stage == 'fit':
            self.train = Dataset(
                self, split='train',
                mean_value=self._mean_value,
                std_value=self._std_value,
            )
            self.val = Dataset(
                self, split='val',
                mean_value=self._mean_value,
                std_value=self._std_value,
            )
        if stage == 'test':
            self.test = Dataset(
                self, split='test',
                mean_value=self._mean_value,
                std_value=self._std_value,
            )
            
    def collate_fn(self, examples):
        pixel_values = torch.stack([example[0] for example in examples])
        labels = torch.tensor([example[1] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, collate_fn=self.collate_fn)
