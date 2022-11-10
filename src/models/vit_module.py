from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.autograd import Variable

from src.models.components.conv_block import ConvBlock
from src.utils.metrics import rmse, rsquared, smape
from src.utils.plotting import make_confusion_matrix, make_pred_vs_target_plot
from components.attention_block import AttentionBlock


def cell_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x


class ViTModule(LightningModule):
    """
    Vision Transformer for Weather Prediction
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout: float = 0.0,
        lr: float = 0.003,
        weight_decay: float = 0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = rsquared
        self.val_metric = rsquared
        self.test_metric = rsquared

        # loss
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, x):
        # Preprocess input
        x = cell_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

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
            all_targets = torch.cat((all_targets, outputs[i]["preds"]), 0)

        # log metrics
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
            all_targets = torch.cat((all_targets, outputs[i]["preds"]), 0)

        # log metrics
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
            all_targets = torch.cat((all_targets, outputs[i]["preds"]), 0)

        # log metrics
        test_metric = self.test_metric(all_preds, all_targets)
        self.log("test/R2", test_metric, on_epoch=True, prog_bar=True)

        # log table
        r2table = np.zeros((all_preds.shape[2], all_preds.shape[3]))
        for x in range(all_preds.shape[2]):
            for y in range(all_preds.shape[3]):
                r2table[x][y] = self.test_metric(
                    all_preds[:, 0, x, y], all_targets[:, 0, x, y]
                )
                print(r2table[x][y])

        self.log("test/R2_std", np.std(r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_mean", np.mean(r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_min", np.min(r2table), on_epoch=True, prog_bar=True)
        self.log("test/R2_max", np.max(r2table), on_epoch=True, prog_bar=True)

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
        # self.logger.experiment.add_image("img", image_name, 0)

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
