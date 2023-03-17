from typing import List

import torch
from torch import Tensor

from classification.utils.data_pipeline import load_data


class Dataset(torch.utils.data.Dataset):
    """
    A custom Dataset to interface with PyTorch.
    """
    def __init__(self, data_ids):
        super(Dataset, self).__init__()
        self.data_ids = data_ids

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        X, y = load_data(data_id + '.pkl')
        # all_data = torch.load(data_id+'.pkl')
        # X = all_data[:, :-1]
        # y = all_data[:, -1:]

        return X, y


def custom_collate(batch):
    data: list[Tensor] = [torch.tensor(item[0]) for item in batch]
    target = [torch.tensor(item[1]) for item in batch]
    stacked = torch.stack(data, dim=1)

    return stacked
