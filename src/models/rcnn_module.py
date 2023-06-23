import sys
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import r2_score, roc_auc_score
from torch.autograd import Variable

from src.models.components.conv_block import ConvBlock
from src.utils.metrics import metrics_celled
from src.utils.plotting import make_heatmap, make_cf_matrix


class ScaledTanh(nn.Module):
    def __init__(self, coef: int = 10):
        super().__init__()
        self.c = coef

    def forward(self, x):
        output = torch.mul(torch.tanh(x), self.c)
        return output


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
        mode: str = "regression",
        embedding_size: int = 16,
        hidden_state_size: int = 32,
        kernel_size: int = 3,
        groups: int = 1,
        dilation: int = 1,
        n_cells_hor: int = 200,
        n_cells_ver: int = 250,
        history_length: int = 1,
        periods_forward: int = 1,
        batch_size: int = 1,
        num_of_additional_features: int = 0,
        boundaries: List[int] = [-2],
        values_range: int = 10,
        dropout: float = 0.0,
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

        self.num_of_features = num_of_additional_features + 1
        self.tanh_coef = values_range
        # number of bins for pdsi
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.num_class = len(boundaries) + 1
        self.boundaries = torch.tensor(boundaries).cuda()

        self.emb_size = embedding_size
        self.hid_size = hidden_state_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups

        self.saved_predictions = None
        self.saved_targets = None

        self.embedding = nn.Sequential(
            ConvBlock(
                self.num_of_features * history_length,
                self.emb_size,
                self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                dilation=self.dilation,
                groups=self.groups,
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
                self.periods_forward,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            ScaledTanh(self.tanh_coef),
            # nn.Tanh(),
            # nn.Conv2d(
            #    self.hid_size,
            #    self.periods_forward,
            #    kernel_size=1,
            #    stride=1,
            #    padding=0,
            #    bias=False,
            # ),
        )

        self.final_classify = nn.Sequential(
            ConvBlock(
                self.hid_size,
                self.num_class,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            ),
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

        self.mode = mode
        # loss
        if self.mode == "regression":
            self.criterion = nn.MSELoss()
            self.loss_name = "MSE"
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.loss_name = "CrossEntropy"

    def forward(self, x: torch.Tensor):
        prev_c = self.prev_state_c
        prev_h = self.prev_state_h
        x = self.dropout(x)
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

        if self.mode == "regression":
            prediction = self.final_conv(next_h)
        elif self.mode == "classification":
            prediction = self.final_classify(next_h)
        self.prev_state_c = next_c
        self.prev_state_h = next_h

        return prediction

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        # atm checking last prediction
        loss = self.criterion(preds, y[:, -1, :, :])
        return loss, preds, y[:, -1, :, :]

    def rolling_step(self, batch: Any):
        x, y = batch
        # x -> B*Hist*W*H or B*(Hist*Feat)*W*H
        # pdsi is first feature in tensor
        x = x[:, : self.history_length, :, :]
        rolling = torch.mean(x, dim=1)
        rolling_forecast = rolling[:, None, :, :]

        for i in range(1, self.periods_forward):
            x = torch.cat((x[:, 1:, :, :], rolling[:, None, :, :]), dim=1)
            rolling = torch.mean(x, dim=1)
            rolling_forecast = torch.cat(
                (rolling_forecast, rolling[:, None, :, :]), dim=1
            )

        return rolling_forecast

    def class_baseline(self, batch: Any):
        x, y = batch
        # return most frequent class along history_dim
        x_binned = torch.bucketize(x, self.boundaries)
        most_freq_values, most_freq_indices = torch.mode(x_binned, dim=1)
        # transform from class to prob
        if self.num_class > 2:
            new_most_freq_values = torch.zeros(most_freq_values.shape[0], self.num_class, most_freq_values.shape[1], most_freq_values.shape[2])
            for b in range(most_freq_values.shape[0]):
                for h in range(most_freq_values.shape[1]):
                    for w in range(most_freq_values.shape[2]):
                        c = most_freq_values[b, h, w]
                        new_most_freq_values[b, c, h, w] = 100.0 
            most_freq_values = new_most_freq_values
            most_freq_values = most_freq_values.to(torch.device('cuda:0'))
        return most_freq_values

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
        all_targets = outputs[0]["targets"]
        all_preds = outputs[0]["preds"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
        all_preds = torch.softmax(all_preds, dim=1)
        if self.num_class == 2:
            all_preds = all_preds[:, 1, :, :]

        # log metrics
        if self.mode == "classification" and self.num_class == 2:
            rocauc_table, ap_table, f1_table, thr = metrics_celled(
                all_targets, all_preds, self.num_class
            )
            self.log(
                "train/f1_median",
                torch.median(f1_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train/ap_median",
                torch.median(ap_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "train/rocauc_median",
                torch.median(rocauc_table),
                on_epoch=True,
                prog_bar=True,
            )
        elif self.mode == "classification" and self.num_class > 2:
            acc_table, ap_table, f1_table, thr = metrics_celled(
                all_targets, all_preds, self.num_class
            )
            self.log(
                "train/accuracy_median",
                torch.median(acc_table),
                on_epoch=True,
                prog_bar=True,
            )

        # log metrics
        # r2table = rsquared(all_targets, all_preds, mode="mean")
        # self.log("train/R2_std", np.std(r2table), on_epoch=True, prog_bar=True)
        # self.log("train/R2", np.median(r2table), on_epoch=True, prog_bar=True)
        # self.log("train/R2_min", np.min(r2table), on_epoch=True, prog_bar=True)
        # self.log("train/R2_max", np.max(r2table), on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        all_targets = outputs[0]["targets"]
        all_preds = outputs[0]["preds"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
        
        all_preds = torch.softmax(all_preds, dim=1)
        if self.num_class == 2:
            all_preds = all_preds[:, 1, :, :]

        # log metrics
        if self.mode == "classification" and self.num_class == 2:
            rocauc_table, ap_table, f1_table, thr = metrics_celled(
                all_targets, all_preds, self.num_class
            )
            self.log(
                "val/f1_median",
                torch.median(f1_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/ap_median",
                torch.median(ap_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/rocauc_median",
                torch.median(rocauc_table),
                on_epoch=True,
                prog_bar=True,
            )
        elif self.mode == "classification" and self.num_class > 2:
            acc_table, ap_table, f1_table, thr = metrics_celled(
                all_targets, all_preds, self.num_class
            )
            self.log(
                "val/accuracy_median",
                torch.median(acc_table),
                on_epoch=True,
                prog_bar=True,
            )

        # log metrics
        # r2table = rsquared(all_targets, all_preds, mode="mean")
        # self.log("val/R2_std", np.std(r2table), on_epoch=True, prog_bar=True)
        # self.log("val/R2", np.median(r2table), on_epoch=True, prog_bar=True)
        # self.log("val/R2_min", np.min(r2table), on_epoch=True, prog_bar=True)
        # self.log("val/R2_max", np.max(r2table), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        if self.mode == "regression":
            baseline = self.rolling_step(batch)
        elif self.mode == "classification":
            baseline = self.class_baseline(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "baseline": baseline}

    def test_epoch_end(self, outputs: List[Any]):
        all_targets = outputs[0]["targets"]
        all_baselines = outputs[0]["baseline"]
        all_preds = outputs[0]["preds"]

        for i in range(1, len(outputs)):
            all_preds = torch.cat((all_preds, outputs[i]["preds"]), 0)
            all_targets = torch.cat((all_targets, outputs[i]["targets"]), 0)
            all_baselines = torch.cat((all_baselines, outputs[i]["baseline"]), 0)

        all_preds = torch.softmax(all_preds, dim=1)
        if self.num_class == 2:
            all_preds = all_preds[:, 1, :, :]

        self.saved_predictions = all_preds
        self.saved_targets = all_targets

        if self.mode == "classification" and self.num_class == 2:
            rocauc_table, ap_table, f1_table, thr = metrics_celled(
                all_targets, all_preds, self.num_class, "test"
            )
            (
                rocauc_table_baseline,
                ap_table_baseline,
                f1_table_baseline,
                thresholds,
            ) = metrics_celled(all_targets, all_baselines, self.num_class, "test")
            # log confusion matrix
            cf_path = make_cf_matrix(
                all_targets, all_preds, thresholds, "cf_matrix.png"
            )
            self.logger.experiment[0].log_image(cf_path)
            # log metrics
            self.log(
                "test/f1_median",
                torch.median(f1_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/ap_median",
                torch.median(ap_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/rocauc_median",
                torch.median(rocauc_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/baseline/f1_median",
                torch.median(f1_table_baseline),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/baseline/ap_median",
                torch.median(ap_table_baseline),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/baseline/rocauc_median",
                torch.median(rocauc_table_baseline),
                on_epoch=True,
                prog_bar=True,
            )

            rocauc_path = make_heatmap(rocauc_table, filename="rocauc_spatial.png")
            torch.save(rocauc_table, "rocauc_table.pt")
            self.logger.experiment[0].log_image(rocauc_path)

        elif self.mode == "classification" and self.num_class > 2:
            acc_table, ap_table, f1_table, thr = metrics_celled(
                all_targets, all_preds, self.num_class, "test"
            )
            #transform baselines from class to probability 100

            (
                acc_table_baseline,
                ap_table_baseline,
                f1_table_baseline,
                thresholds,
            ) = metrics_celled(all_targets, all_baselines, self.num_class, "test")

            # log metrics
            self.log(
                "test/accuracy_median",
                torch.median(acc_table),
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test/baseline/accuracy_median",
                torch.median(acc_table_baseline),
                on_epoch=True,
                prog_bar=True,
            )

            acc_path = make_heatmap(acc_table, filename="acc_spatial.png")
            torch.save(acc_table, "acc_table.pt")
            self.logger.experiment[0].log_image(acc_path)

        # log metrics
        # test_r2table = rsquared(all_targets, all_preds, mode="full")
        # self.log("test/R2_std", np.std(test_r2table), on_epoch=True, prog_bar=True)
        # self.log(
        #     "test/R2_median", np.median(test_r2table), on_epoch=True, prog_bar=True
        # )
        # self.log("test/R2_min", np.min(test_r2table), on_epoch=True, prog_bar=True)
        # self.log("test/R2_max", np.max(test_r2table), on_epoch=True, prog_bar=True)
        if self.mode == "regression":
            self.log(
                "test/baseline_MSE",
                self.criterion(all_baselines, all_targets),
                on_epoch=True,
                prog_bar=True,
            )

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
