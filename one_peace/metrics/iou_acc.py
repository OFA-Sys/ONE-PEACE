import torch
import torch.distributed as dist

from .base_metric import BaseMetric
from one_peace.utils.data_utils import all_gather


class IouAcc(BaseMetric):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.ids = torch.Tensor([]).cuda()
        self.hyps = torch.Tensor([]).cuda()
        self.score_sum = torch.FloatTensor([0]).cuda()
        self.score_cnt = torch.IntTensor([0]).cuda()

    def compute(self, ids, hyps, refs):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts.float() / (area_predictions + area_targets - area_interacts)

        self.score_sum += ((ious >= 0.5) & (interacts_w > 0) & (interacts_h > 0)).float().sum().item()
        self.score_cnt += hyps.size(0)
        self.ids = torch.cat([self.ids, ids], dim=0)
        self.hyps = torch.cat([self.hyps, hyps], dim=0)

    def merge_results(self):
        if dist.is_initialized():
            dist.all_reduce(self.score_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.score_cnt, op=dist.ReduceOp.SUM)
            ids = all_gather(self.ids)
            hyps = all_gather(self.hyps)
        else:
            ids = self.ids
            hyps = self.hyps
        score_sum = self.score_sum.item()
        score_cnt = self.score_cnt.item()
        refcoco_result = []
        for id, hyp in zip(ids.tolist(), hyps.tolist()):
            refcoco_result.append((id, hyp))
        return {
            'iou_acc': score_sum / score_cnt,
            'score_sum': score_sum,
            'score_cnt': score_cnt,
            'refcoco_result': refcoco_result
        }