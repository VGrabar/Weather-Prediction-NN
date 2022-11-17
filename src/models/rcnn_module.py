from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.autograd import Variable

from src.models.components.conv_block import ConvBlock
from src.utils.metrics import rmse, rsquared, smape
from src.utils.plotting import make_heatmap, make_pred_vs_target_plot


class RCNNModule(LightningModule):
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
        embedding_size: int = 16,
        hidden_state_size: int = 32,
        kernel_size: int = 3,
        n_cells_hor: int = 200,
        n_cells_ver: int = 250,
        history_length: int = 1,
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
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.emb_size = embedding_size
        self.hid_size = hidden_state_size
        self.kernel_size = kernel_size

        self.embedding = nn.Sequential(
            ConvBlock(
                history_length,
                self.emb_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.ReLU(),
            ConvBlock(
                self.emb_size,
                self.emb_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
        )

        self.f_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Sigmoid(),
        )
        self.i_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Sigmoid(),
        )
        self.c_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Tanh(),
        )
        self.o_t = nn.Sequential(
            ConvBlock(
                self.hid_size + self.emb_size,
                self.hid_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
            nn.Sigmoid(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                self.hid_size,
                self.hid_size,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(self.hid_size, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.register_buffer(
            "prev_state_h",
            torch.zeros(
                self.batch_size,
                self.hid_size,
                self.n_cells_hor,
                self.n_cells_ver,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "prev_state_c",
            torch.zeros(
                self.batch_size,
                self.hid_size,
                self.n_cells_hor,
                self.n_cells_ver,
                requires_grad=False,
            ),
        )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = rsquared
        self.val_metric = rsquared
        self.test_metric = rsquared

        # loss
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        prev_c = self.prev_state_c
        prev_h = self.prev_state_h
        x_emb = self.embedding(x)
        x_and_h = torch.cat([prev_h, x_emb], dim=1)

        f_i = self.f_t(x_and_h)
        i_i = self.i_t(x_and_h)
        c_i = self.c_t(x_and_h)
        o_i = self.o_t(x_and_h)

        next_c = prev_c * f_i + i_i * c_i
        next_h = torch.tanh(next_c) * o_i

        assert prev_h.shape == next_h.shape
        assert prev_c.shape == next_c.shape

        prediction = self.final_conv(next_h)
        self.prev_state_c = next_c
        self.prev_state_h = next_h

        return prediction

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def on_after_backward(self) -> None:
        self.prev_state_c.detach_()
        self.prev_state_h.detach_()

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
        all_preds = torch.squeeze(all_preds, 1)
        train_metric = self.train_metric(all_preds, all_targets)
        self.log("train/R2", train_metric, on_epoch=True, prog_bar=True)
        pass

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
        all_preds = torch.squeeze(all_preds, 1)
        val_metric = self.val_metric(all_preds, all_targets)
        self.log("val/R2", val_metric, on_epoch=True, prog_bar=True)

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
        all_preds = torch.squeeze(all_preds, 1)
        test_metric = self.test_metric(all_preds, all_targets)
        self.log("test/R2", test_metric, on_epoch=True, prog_bar=True)

        # log table
        r2table = np.zeros((all_preds.shape[1], all_preds.shape[2]))
        for x in range(all_preds.shape[1]):
            for y in range(all_preds.shape[2]):
                r2table[x][y] = self.test_metric(
                    all_preds[:, x, y], all_targets[:, x, y]
                )

        self.log("test/R2_std", np.std(r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_mean", np.mean(r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_min", np.min(r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_max", np.max(r2table), on_epoch=True, prog_bar=True)
        
        # log R2 table
        image_name = "confusion_matrix.png"
        image_path = make_heatmap(r2table, image_name)
        # self.logger.experiment.log_image(image_path, name="R2 Spatial Distribution")

        # log plots
        image_name = "preds_targets.png"
        make_pred_vs_target_plot(
            all_preds,
            all_targets,
            title="Forecasting",
            size=(8, 6),
            xlabel="periods",
            xlabel_rotate=45,
            ylabel="Value",
            ylabel_rotate=0,
            filename=image_name,
        )
        # self.logger.Experiment.log_image("img", image_name, 0)

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
