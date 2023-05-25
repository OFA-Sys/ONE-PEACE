import torch
import torch.distributed as dist

from .base_metric import BaseMetric
from ..utils.data_utils import all_gather


class Accuracy(BaseMetric):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.score_sum = torch.FloatTensor([0]).cuda()
        self.score_cnt = torch.IntTensor([0]).cuda()
        self.ids = torch.LongTensor([]).cuda()
        self.hyps = torch.LongTensor([]).cuda()

    def compute(self, ids, logits, targets):
        predict_labels = logits.argmax(1)
        if targets.dim() == 2:
            n_correct = targets.gather(1, predict_labels.unsqueeze(1)).sum()
        else:
            n_correct = predict_labels.eq(targets).sum()

        self.score_sum += n_correct
        self.score_cnt += logits.size(0)
        self.ids = torch.cat([self.ids, ids], dim=0)
        self.hyps = torch.cat([self.hyps, predict_labels], dim=0)

    def merge_results(self, output_predict=False):
        if dist.is_initialized():
            dist.all_reduce(self.score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.score_cnt, op=dist.ReduceOp.SUM)
            ids = all_gather(self.ids)
            hyps = all_gather(self.hyps)
        else:
            ids = self.ids
            hyps = self.hyps

        predict_results = {}
        if output_predict:
            for id, hyp in zip(ids.cpu().tolist(), hyps):
                predict_results[id] = hyp

        score_sum = self.score_sum.item()
        score_cnt = self.score_cnt.item()
        return {
            'accuracy': score_sum / score_cnt,
            'score_sum': score_sum,
            'score_cnt': score_cnt,
            'predict_results': predict_results
        }