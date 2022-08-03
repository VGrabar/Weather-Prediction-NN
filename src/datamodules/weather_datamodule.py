import pathlib
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from utils.data_utils import create_celled_data


class Dataset_RNN(Dataset):
    def __init__(
        self,
        celled_data,
        start_date,
        end_date,
        periods_to_predict,
    ):
        self.data = celled_data[start_date:end_date]
        self.size = self.data.shape[0]
        self.periods_to_predict = periods_to_predict

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.data[(idx) : (idx + self.periods_to_predict)],
        )


class WeatherDataModule(LightningDataModule):
    """LightningDataModule for Weather dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data",
        dataset_name: str = "dataset_name",
        n_cells_hor: int = 200,
        n_cells_ver: int = 250,
        left_border: int = 0,
        down_border: int = 0,
        right_border: int = 2000,
        up_border: int = 2500,
        time_col: str = "time",
        event_col: str = "value",
        valid_periods: int = 365,
        test_periods: int = 365,
        periods_to_predict: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.left_border = left_border
        self.right_border = right_border
        self.down_border = down_border
        self.up_border = up_border
        self.time_col = time_col
        self.event_col = event_col

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.valid_periods = valid_periods
        self.test_periods = test_periods
        self.periods_to_predict = periods_to_predict

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        celled_data_path = pathlib.Path(
            self.data_dir,
            "celled",
            self.dataset_name,
            "_" + str(self.n_cells_hor) + "x" + str(self.n_cells_ver),
        )
        if not celled_data_path.is_file():
            celled_data = create_celled_data(
                self.data_path,
                self.dataset_name,
                self.n_cells_hor,
                self.n_cells_ver,
                self.left_border,
                self.down_border,
                self.right_border,
                self.up_border,
                self.time_col,
                self.event_col,
            )
            torch.save(celled_data, celled_data_path)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            celled_data_path = pathlib.Path(
                self.data_dir,
                "celled",
                self.dataset_name,
                "_" + str(self.n_cells_hor) + "x" + str(self.n_cells_ver),
            )
            celled_data = torch.load(celled_data_path)
            train_start = 0
            train_end = celled_data.shape[0] - self.valid_periods - self.test_periods
            self.data_train = Dataset_RNN(
                celled_data, train_start, train_end, self.periods_to_predict
            )
            valid_start = celled_data.shape[0] - self.valid_periods - self.test_periods
            valid_end = celled_data.shape[0] - self.test_periods
            self.data_val = Dataset_RNN(
                celled_data, valid_start, valid_end, self.periods_to_predict
            )
            test_start = celled_data.shape[0] - self.test_periods
            test_end = celled_data.shape[0]
            self.data_test = Dataset_RNN(
                celled_data, test_start, test_end, self.periods_to_predict
            )

    def train_dataloader(self):

        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
