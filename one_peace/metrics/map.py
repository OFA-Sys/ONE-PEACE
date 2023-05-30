import torch
import torch.distributed as dist

import numpy as np
from sklearn.metrics import average_precision_score

from .base_metric import BaseMetric
from ..utils.data_utils import all_gather


class MAP(BaseMetric):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.logits = torch.FloatTensor([]).cuda()
        self.targets = torch.FloatTensor([]).cuda()
        self.ids = torch.LongTensor([]).cuda()

    def compute(self, ids, logits, targets):
        self.ids = torch.cat([self.ids, ids], dim=0)
        self.logits = torch.cat([self.logits, logits], dim=0)
        self.targets = torch.cat([self.targets, targets], dim=0)

    def merge_results(self, output_predict=False):
        if dist.is_initialized():
            ids = all_gather(self.ids)
            preds = all_gather(self.logits)
            targets = all_gather(self.targets)
        else:
            ids = self.ids
            preds = self.logits
            targets = self.targets
        preds = torch.sigmoid(preds).cpu().numpy()
        targets = targets.cpu().numpy()

        predict_results = {}
        if output_predict:
            for id, pred in zip(ids.cpu().tolist(), preds.cpu().tolist()):
                predict_results[id] = pred

        return {
            'map': np.mean(average_precision_score(targets, preds, average=None)),
            'map_cnt': len(targets),
            'predict_results': predict_results
        }