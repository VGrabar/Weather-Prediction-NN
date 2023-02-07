import numpy as np
import torch
from sklearn.metrics import r2_score


def rmse(target, pred):

    return torch.mean((pred - target) ** 2)


def smape(target, pred):
    return 100 * torch.mean(
        (torch.abs(pred - target)) / (torch.abs(pred) + torch.abs(target))
    )


def rsquared(target, pred, mode: str = "mean"):

    if (mode == "full"):
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

    elif (mode == "mean"):
        pred_mean = torch.mean(pred, (2, 3)).cpu().numpy()
        target_mean = torch.mean(target, (2, 3)).cpu().numpy()

        r2_raw = np.zeros(pred_mean.shape[0])
        
        for b in range(pred_mean.shape[0]):
            r2_raw[b] = r2_score(target_mean[b, :], pred_mean[b, :])
        #r2_median = np.median(r2_raw)

        return r2_raw
