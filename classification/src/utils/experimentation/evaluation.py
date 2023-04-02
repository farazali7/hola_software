from statistics import mode
import torch


def majority_vote_transform(preds, targets, voters, drop_last=False):
    """
    Transform input predictions and targets vectors to a majority-vote setting based on number of voters
    :param preds: Tensor of predictions
    :param targets: Tensor of targets/labels
    :param voters: Integer for number of voters in majority vote
    :param drop_last: Boolean to drop last group
    """

    all_preds = []
    all_targets = []
    for x in torch.unique(targets):
        idxs = torch.where(targets == x)[0]

        rel_preds = preds[idxs]
        rel_targets = targets[idxs]

        grouped_preds = [rel_preds[i:i+voters].detach().numpy() for i in range(0, len(rel_preds), voters)]
        grouped_targets = [rel_targets[i:i + voters].detach().numpy() for i in range(0, len(rel_targets), voters)]

        if drop_last:
            if len(grouped_preds[-1]) < voters:
                grouped_preds = grouped_preds[:-1]
            if len(grouped_targets[-1]) < voters:
                grouped_targets = grouped_targets[:-1]

        mode_preds = [mode(pred) for pred in grouped_preds]
        all_preds += mode_preds
        mode_targets = [mode(target) for target in grouped_targets]
        all_targets += mode_targets

    preds_tensor = torch.tensor(all_preds, dtype=torch.int64)
    targets_tensor = torch.tensor(all_targets, dtype=torch.int64)

    return preds_tensor, targets_tensor
