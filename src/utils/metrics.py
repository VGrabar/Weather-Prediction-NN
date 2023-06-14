import numpy as np
import torch
from torchmetrics.classification import AUROC, AveragePrecision, F1Score, ROC, Accuracy


def metrics_celled(all_targets, all_preds, n_classes, mode: str = "train"):
    if n_classes > 2:
        acc_table = torch.zeros(all_preds.shape[2], all_preds.shape[3])
        print(all_preds.shape)
        print(all_targets.shape)
        acc = Accuracy(
            task="multiclass", num_classes=n_classes, top_k=1, average="micro"
        ).to(torch.device("cuda", 0))
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

    else:
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
                    f1_table = f1(all_preds[:, x, y], all_targets[:, x, y])

            ap_table = torch.nan_to_num(ap_table, nan=0.0)
            f1_table = torch.nan_to_num(f1_table, nan=0.0)

    return rocauc_table, ap_table, f1_table, thresholds
