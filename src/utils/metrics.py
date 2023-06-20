import numpy as np
import torch
from torchmetrics.classification import AUROC, AveragePrecision, F1Score, ROC


def metrics_celled(all_targets, all_preds, mode: str = "train"):
    rocauc_table = torch.zeros(all_preds.shape[1], all_preds.shape[2])
    rocauc = AUROC(task="binary", num_classes=1)
    rocauc_table = torch.tensor(
        [
            [
                rocauc(all_preds[:, x, y], all_targets[:, x, y])
                for x in range(all_preds.shape[1])
            ]
            for y in range(all_preds.shape[2])
        ]
    )
    rocauc_table = torch.nan_to_num(rocauc_table, nan=0.0)

    ap_table = torch.zeros(all_preds.shape[1], all_preds.shape[2])
    f1_table = torch.zeros(all_preds.shape[1], all_preds.shape[2])
    thresholds = torch.zeros(all_preds.shape[1], all_preds.shape[2])

    if mode == "test":
        ap = AveragePrecision(task="binary")
        roc = ROC(task="binary")
        for x in range(all_preds.shape[1]):
            for y in range(all_preds.shape[2]):
                ap_table[x][y] = ap(all_preds[:, x, y], all_targets[:, x, y])
                fpr, tpr, thr = roc(all_preds[:, x, y], all_targets[:, x, y])
                j_stat = tpr - fpr
                ind = torch.argmax(j_stat).item()
                thresholds[x][y] = thr[ind]
                f1 = F1Score(task="binary", threshold=thresholds[x][y]).to(
                    torch.device("cuda", 0)
                )
                f1_table[x][y] = f1(all_preds[:, x, y], all_targets[:, x, y])

        ap_table = torch.nan_to_num(ap_table, nan=0.0)
        f1_table = torch.nan_to_num(f1_table, nan=0.0)

    return rocauc_table, ap_table, f1_table, thresholds
