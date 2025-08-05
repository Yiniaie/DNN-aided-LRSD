import torch

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))


def mape(y_true, y_pred, threshold=0.1):
    v = torch.clip(torch.abs(y_true), threshold, None)
    diff = torch.abs((y_true - y_pred) / v)
    return 100.0 * torch.mean(diff, axis=-1).mean()


def rse(y_true, y_pred):
    batch_num, j = y_true.shape
    return torch.sqrt(torch.square(y_pred - y_true).sum() / (batch_num - 2))