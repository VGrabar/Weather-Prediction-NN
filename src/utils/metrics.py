import torch


def rmse(output, target):
    return torch.mean((output - target) ** 2)


def smape(output, target):
    return 100 * torch.mean(
        (torch.abs(output - target)) / (torch.abs(output) + torch.abs(target))
    )


def rsquared(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2
