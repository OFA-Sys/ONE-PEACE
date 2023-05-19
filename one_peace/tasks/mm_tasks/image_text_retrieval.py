# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
from typing import Optional
import logging
import torch
import torch.distributed as dist

from fairseq.tasks import register_task
from fairseq.utils import move_to_cuda

from one_peace.tasks.base_task import BaseTask, BaseTaskConfig
from one_peace.data.vl_data.image_text_retrieval_dataset import ImageTextRetrievalDataset
from one_peace.metrics import Recall
from one_peace.utils.data_utils import new_islice, all_gather

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig(BaseTaskConfig):
    valid_file: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    min_scale: float = field(
        default=0.9,
        metadata={"help": ""}
    )


@register_task("image_text_retrieval", dataclass=RetrievalConfig)
class ImageTextRetrievalTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Recall()
        self.caption_ids = None
        self.captions = None

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        if self.caption_ids is None:
            file_path = self.cfg.valid_file

            self.caption_ids = []
            self.captions = []
            for caption_id, caption_list in torch.load(file_path).items():
                if not isinstance(caption_list, list):
                    caption_list = eval(caption_list)
                for caption in caption_list:
                    self.caption_ids.append(caption_id)
                    self.captions.append(caption)
            self.caption_ids = torch.tensor(self.caption_ids).cuda()

        self.datasets[split] = ImageTextRetrievalDataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_src_length=self.cfg.max_src_length,
            patch_image_size=self.cfg.patch_image_size,
            min_scale=self.cfg.min_scale
        )

    @torch.no_grad()
    def begin_valid_epoch(self, epoch, model, subset):
        model.eval()

        dataset = self.datasets[subset]
        text_cnt = len(self.caption_ids)
        text_ids = self.caption_ids

        if dist.is_initialized():
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        else:
            slice_id = 0
            slice_count = 1
        batch_sampler = new_islice(range(text_cnt), slice_id, text_cnt, slice_count)
        start_idx = batch_sampler[0]
        end_idx = batch_sampler[-1] + 1

        text_logits_list = []
        for i in range(start_idx, end_idx, 50):
            samples_list = []
            for text in self.captions[i:min(i+50, end_idx)]:
                item_tuple = (0, None, text)
                sample = dataset.__getitem__(0, item_tuple)
                samples_list.append(sample)
            samples = dataset.collater(samples_list)
            samples = move_to_cuda(samples)
            src_tokens = samples["net_input"]["src_tokens"]
            text_logits = model(src_tokens=src_tokens, encoder_type='text')
            text_logits_list.append(text_logits)

        text_logits = torch.cat(text_logits_list, dim=0)
        text_logits = all_gather(text_logits) if dist.is_initialized() else text_logits
        self.metric.initialize(text_ids, text_logits)

    @torch.no_grad()
    def valid_step(self, sample, model, criterion, is_dummy_batch):
        loss = 0
        sample_size = len(sample['id'])
        logging_output = {'nsentences': 1, 'ntokens': 1}
        if not is_dummy_batch:
            model.eval()
            self.eval_step(model, sample)
        return loss, sample_size, logging_output

    @torch.no_grad()
    def eval_step(self, model, sample):
        src_images = sample["net_input"]["src_images"]
        image_ids = torch.tensor(sample['id']).to(src_images.device)
        image_logits = model(src_images=src_images, encoder_type='image')
        self.metric.compute(image_ids, image_logits)