from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.autograd import Variable

from src.models.components.conv_block import ConvBlock
from src.utils.metrics import rmse, rsquared, smape
from sklearn.metrics import r2_score
from src.utils.plotting import make_heatmap, make_pred_vs_target_plot

import seaborn as sns
import matplotlib.pyplot as plt



class Conv1dModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        n_cells_hor: int = 200,
        n_cells_ver: int = 250,
        history_length: int = 1,
        periods_forward: int = 1,
        batch_size: int = 1,
        lr: float = 0.003,
        weight_decay: float = 0.0,
    ):
        super(self.__class__, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.history_length = history_length
        self.periods_forward = periods_forward
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay


        self.conv1x1 = nn.Conv2d(
                self.batch_size,
                self.periods_forward,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),

        # loss
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        
        prediction = self.conv1x1(x)

        return prediction

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        all_preds = outputs[0]["preds"]
        all_targets = outputs[0]["targets"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)

        # log metrics
        r2table = rsquared(all_targets, all_preds, mode="mean")
        self.log("train/R2_std", np.std(r2table), on_epoch=True, prog_bar=True)
        self.log("train/R2", np.median(r2table), on_epoch=True, prog_bar=True)
        self.log("train/R2_min", np.min(r2table), on_epoch=True, prog_bar=True)
        self.log("train/R2_max", np.max(r2table), on_epoch=True, prog_bar=True)
        self.log("train/MSE", rmse(all_targets, all_preds), on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        all_preds = outputs[0]["preds"]
        all_targets = outputs[0]["targets"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)

        # log metrics
        r2table = rsquared(all_targets, all_preds, mode="mean")
        self.log("val/R2_std", np.std(r2table), on_epoch=True, prog_bar=True)
        self.log("val/R2", np.median(r2table), on_epoch=True, prog_bar=True)
        self.log("val/R2_min", np.min(r2table), on_epoch=True, prog_bar=True)
        self.log("val/R2_max", np.max(r2table), on_epoch=True, prog_bar=True)
        self.log("val/MSE", rmse(all_targets, all_preds), on_epoch=True, prog_bar=True)


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        all_preds = outputs[0]["preds"]
        all_targets = outputs[0]["targets"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)

        # log metrics
        test_r2table = rsquared(all_targets, all_preds, mode="full")
        self.log("test/R2_std", np.std(test_r2table), on_epoch=True, prog_bar=True)
        self.log(
            "test/R2_median", np.median(test_r2table), on_epoch=True, prog_bar=True
        )
        self.log("test/R2_min", np.min(test_r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_max", np.max(test_r2table), on_epoch=True, prog_bar=True)
        self.log("test/MSE", rmse(all_targets, all_preds), on_epoch=True, prog_bar=True)

        # log graphs
        mse_conv = []
        r2_conv = []

        for i in range(1, self.periods_forward+1):
            
            preds_i = all_preds[:, :i, :, :]
            targets_i = all_targets[:, :i, :, :]
            mse_conv.append(rmse(preds_i, targets_i))
            mean_preds = torch.mean(preds_i, axis=(2, 3))
            mean_targets = torch.mean(targets_i, axis=(2, 3))
            r2_conv.append(r2_score(mean_targets.cpu().numpy(), mean_preds.cpu().numpy()))

        h = [i for i in range(1, self.periods_forward+1)]
        fig1 = plt.figure(figsize=(7, 7))
        ax = fig1.add_subplot(1, 1, 1)
        sns.lineplot(x=h, y=mse_conv, ax = ax)
        ax.legend(['conv1d'])
        ax.set_xlabel("horizon (in months)")
        ax.set_title("MSE")
        fig1.savefig("conv_mse.png")

        fig2 = plt.figure(figsize=(7, 7))
        ax = fig2.add_subplot(1, 1, 1)
        sns.lineplot(x=h, y=r2_conv, ax = ax)
        ax.legend(['conv1d'])
        ax.set_xlabel("horizon (in months)")
        ax.set_title("R2")
        fig2.savefig("r2_mse.png")


    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
