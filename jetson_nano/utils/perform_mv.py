import torch


def perform_mv(preds):
    final_pred = torch.mode(preds).values

    return final_pred
