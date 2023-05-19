import torch
import torch.distributed as dist

import numpy as np
from sklearn.metrics import average_precision_score

from .base_metric import BaseMetric
from one_peace.utils.data_utils import all_gather


class MAP(BaseMetric):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.logits = torch.FloatTensor([]).cuda()
        self.targets = torch.FloatTensor([]).cuda()

    def compute(self, logits, targets):
        self.logits = torch.cat([self.logits, logits], dim=0)
        self.targets = torch.cat([self.targets, targets], dim=0)

    def merge_results(self):
        if dist.is_initialized():
            pred = all_gather(self.logits)
            target = all_gather(self.targets)
        else:
            pred = self.logits
            target = self.targets

        pred = torch.sigmoid(pred).cpu().numpy()
        target = target.cpu().numpy()
        return {
            'map': np.mean(average_precision_score(target, pred, average=None)),
            'map_cnt': len(target)
        }