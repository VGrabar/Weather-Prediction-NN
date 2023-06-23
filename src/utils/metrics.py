import numpy as np
import torch
from torchmetrics.classification import AUROC, AveragePrecision, F1Score, ROC, Accuracy


def metrics_celled(all_targets, all_preds, mode: str = "train"):
    acc_table = torch.zeros(all_preds.shape[2], all_preds.shape[3])
    acc = Accuracy(
        task="multiclass", num_classes=3, top_k=1, average="micro"
    )#.to(torch.device("cuda", 0))
    acc_table = torch.tensor(
        [
            [
                acc(all_preds[:, :, x, y], all_targets[:, x, y])
                    for x in range(all_preds.shape[2])
            ]
            for y in range(all_preds.shape[3])
        ]
    )
    acc_table = torch.nan_to_num(acc_table, nan=0.0)
    ap_table = torch.zeros(all_preds.shape[2], all_preds.shape[3])
    f1_table = torch.zeros(all_preds.shape[2], all_preds.shape[3])
    thresholds = torch.zeros(all_preds.shape[2], all_preds.shape[3])

    return acc_table, ap_table, f1_table, thresholds

