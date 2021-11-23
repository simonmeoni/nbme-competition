from typing import Optional

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer


class NBMEDataset(Dataset):
    def __init__(self, df, tokenizer, tokenizer_config):
        self.df = df
        df.annotation = df.annotation.apply(lambda x: eval(x))
        self.location = df.location.apply(
            lambda x: [
                [int(numeric) for numeric in offset.split(" ")]
                for loc in eval(x)
                for offset in loc.split(";")
            ]
        )
        if "disease_loc" in df.keys():
            self.diseases_loc = df.diseases_loc.apply(lambda x: eval(x))
            self.chemicals_loc = df.chemicals_loc.apply(lambda x: eval(x))
        else:
            self.diseases_loc = self.location
            self.chemicals_loc = self.location
        self.tokenizer_config = tokenizer_config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.feature_text = df.feature_text.values
        self.pn_history = df.pn_history.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            "feature_text": self.feature_text[index],
            "pn_history": self.pn_history[index],
            "location": self.location[index],
            "chemicals_loc": self.chemicals_loc[index],
            "diseases_loc": self.diseases_loc[index],
        }

    @staticmethod
    def create_labels(location, offset_mapping):
        labels = torch.zeros(offset_mapping.shape[0])
        for loc in location:
            start, end = loc
            end_mapping = offset_mapping[end <= offset_mapping[:, 1]]
            if end_mapping.shape[0] != 0:
                end_offset = (offset_mapping == end_mapping[0]).nonzero()[0]
                start_mapping = offset_mapping[(start < offset_mapping[:, 0])]
                start_offset = (
                    (offset_mapping == start_mapping[0]).nonzero()[0]
                    if start_mapping.shape[0] != 0
                    else end_offset
                )
                labels[start_offset[0] - 1 : end_offset[0] + 1] = 1
        return labels

    def collate_fn(self, batch):
        text = [[example["pn_history"], example["feature_text"]] for example in batch]
        tokens = self.tokenizer(text, **self.tokenizer_config)
        locations = [example["location"] for example in batch]
        diseases_loc = [example["diseases_loc"] for example in batch]
        chemicals_loc = [example["chemicals_loc"] for example in batch]
        diseases_labels = []
        chemicals_labels = []
        labels = []
        for offsets_mapping, location, disease_loc, chemical_loc in zip(
            tokens["offset_mapping"],
            locations,
            diseases_loc,
            chemicals_loc,
        ):
            labels.append(self.create_labels(location, offsets_mapping))
            chemicals_labels.append(self.create_labels(disease_loc, offsets_mapping))
            diseases_labels.append(self.create_labels(chemical_loc, offsets_mapping))
        del tokens["offset_mapping"]
        return (
            tokens,
            torch.stack(labels).unsqueeze(2),
            torch.stack(diseases_labels).unsqueeze(2),
            torch.stack(chemicals_labels).unsqueeze(2),
        )


class DataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    splits, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_length: int = 512,
        k_fold: int = 5,
        current_fold: int = 0,
        tokenizer: str = "",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.full_dataset: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.tokenizer_config = {
            "max_length": self.hparams.max_length,
            "padding": "longest",
            "return_offsets_mapping": True,
            "return_tensors": "pt",
        }

    def prepare_data(self):
        """Download data if needed. This method is called ony from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.full_dataset = NBMEDataset(
            pd.read_csv(self.hparams.data_dir).dropna(),
            self.hparams.tokenizer,
            tokenizer_config=self.tokenizer_config,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`,
        so be careful if you do a random splits!
        The `stage` can be used to differentiate whether it's called before
        trainer.fit()` or `trainer.test()`."""
        k_fold = KFold(n_splits=self.hparams.k_fold, shuffle=True)
        data_train_ids, data_val_ids = list(
            k_fold.split(self.full_dataset),
        )[self.hparams.current_fold]
        self.data_train = Subset(self.full_dataset, data_train_ids)
        self.data_val = Subset(self.full_dataset, data_val_ids)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.full_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.full_dataset.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
