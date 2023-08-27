import torch
import torch.distributed as dist

from .base_metric import BaseMetric
from ..utils.data_utils import all_gather


class Recall(BaseMetric):
    def __init__(self):
        super().__init__()

    def initialize(self, text_ids, text_logits):
        self.text_ids = text_ids
        self.text_logits = text_logits
        self.image_ids_list = []
        self.image_logits_list = []

    def compute(self, image_ids, image_logits):
        self.image_ids_list.append(image_ids)
        self.image_logits_list.append(image_logits)

    def merge_results(self, output_predict=False):
        image_ids = torch.cat(self.image_ids_list, dim=0)
        image_logits = torch.cat(self.image_logits_list, dim=0)
        if dist.is_initialized():
            self.image_ids = all_gather(image_ids)
            self.image_logits = all_gather(image_logits)
        else:
            self.image_ids = image_ids
            self.image_logits = image_logits

        sim_i2t = self.image_logits @ self.text_logits.t()
        sim_t2i = sim_i2t.t()
        eval_log = self.retrieval_eval(sim_i2t, sim_t2i, output_predict)
        return eval_log

    def retrieval_eval(self, scores_i2t, scores_t2i, output_predict=False):
        # Image->Text
        _, rank_txt = scores_i2t.topk(k=10, dim=1)
        predict_txt = self.text_ids[None, :].expand(rank_txt.size(0), -1).gather(1, rank_txt)
        i2t_corrects = [predict_txt[:, :r].eq(self.image_ids[:, None]).any(1).sum().item() for r in [1, 5, 10]]

        tr_r1 = 100.0 * i2t_corrects[0] / scores_i2t.size(0)
        tr_r5 = 100.0 * i2t_corrects[1] / scores_i2t.size(0)
        tr_r10 = 100.0 * i2t_corrects[2] / scores_i2t.size(0)
        tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3

        # Text->Image
        _, rank_img = scores_t2i.topk(k=10, dim=1)
        predict_img = self.image_ids[None, :].expand(rank_img.size(0), -1).gather(1, rank_img)
        t2i_corrects = [predict_img[:, :r].eq(self.text_ids[:, None]).any(1).sum().item() for r in [1, 5, 10]]

        ir_r1 = 100.0 * t2i_corrects[0] / scores_t2i.size(0)
        ir_r5 = 100.0 * t2i_corrects[1] / scores_t2i.size(0)
        ir_r10 = 100.0 * t2i_corrects[2] / scores_t2i.size(0)
        ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3

        predict_txt_results = {}
        predict_img_results = {}
        if output_predict:
            for img_id, predict_txt_ in zip(self.image_ids.cpu().tolist(), predict_txt.cpu().tolist()):
                predict_txt_results[img_id] = predict_txt_
            for txt_id, predict_img_ in zip(self.text_ids.cpu().tolist(), predict_img.cpu().tolist()):
                predict_img_results[txt_id] = predict_img_

        eval_log = {'txt_r1': tr_r1,
                    'txt_r5': tr_r5,
                    'txt_r10': tr_r10,
                    'txt_r_mean': tr_mean,
                    'img_count': scores_i2t.size(0),
                    'img_r1': ir_r1,
                    'img_r5': ir_r5,
                    'img_r10': ir_r10,
                    'img_r_mean': ir_mean,
                    'r_mean': (tr_mean + ir_mean) / 2,
                    'txt_count': scores_t2i.size(0),
                    'predict_txt': predict_txt_results,
                    'predict_img': predict_img_results}
        return eval_log