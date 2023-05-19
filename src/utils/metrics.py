import numpy as np
import torch
from sklearn.metrics import r2_score, roc_auc_score, f1_score, average_precision_score
# from torchmetrics.classification import BinaryAUROC


def rmse(target, pred):

    return torch.mean((pred - target) ** 2)


def smape(target, pred):
    return 100 * torch.mean(
        (torch.abs(pred - target)) / (torch.abs(pred) + torch.abs(target))
    )


def rsquared(target, pred, mode: str = "mean"):

    if mode == "full":
        # log table
        r2table = np.zeros((pred.shape[2], pred.shape[3]))
        for x in range(pred.shape[2]):
            for y in range(pred.shape[3]):
                for b in range(pred.shape[0]):
                    r2table[x][y] += r2_score(
                        target[b, :, x, y].cpu().numpy(), pred[b, :, x, y].cpu().numpy()
                    )
                r2table[x][y] /= pred.shape[0]
        return r2table

    elif mode == "mean":
        pred_mean = torch.mean(pred, (2, 3)).cpu().numpy()
        target_mean = torch.mean(target, (2, 3)).cpu().numpy()

        r2_raw = np.zeros(pred_mean.shape[0])

        for b in range(pred_mean.shape[0]):
            r2_raw[b] = r2_score(target_mean[b, :], pred_mean[b, :])

        return r2_raw


def metrics_celled(all_targets, all_preds):

    rocauc_table = torch.zeros(all_preds.shape[1], all_preds.shape[2])
    f1_table = torch.zeros(all_preds.shape[1], all_preds.shape[2])
    ap_table = torch.zeros(all_preds.shape[1], all_preds.shape[2])
    for x in range(all_preds.shape[1]):
        for y in range(all_preds.shape[2]):
            try:
                rocauc_table[x][y] = roc_auc_score(
                    all_targets[:, x, y].cpu().numpy(), all_preds[:, x, y].cpu().numpy()
                )
            except:
                rocauc_table[x][y] = 0
            
            try:
                ap_table[x][y] = average_precision_score(
                    all_targets[:, x, y].cpu().numpy(), all_preds[:, x, y].cpu().numpy()
                )
            except:
                ap_table[x][y] = 0
            
            try:
                f1_table[x][y] = f1_score(
                    all_targets[:, x, y].cpu().numpy(), np.where(all_preds[:, x, y].cpu().numpy() > 0.5, 1, 0)
                )
            except:
                f1_table[x][y] = 0
            # rocauc_table[x][y] = BinaryAUROC(all_preds[:,x,y], all_targets[:,x,y])

    return rocauc_table, ap_table, f1_table
