# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import logging
import os
import math
import re
import numpy as np
from typing import Optional
from omegaconf import II

import torch
from fairseq.data import FairseqDataset, data_utils
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.tasks import FairseqTask, register_task
from omegaconf import DictConfig

from data.tsv_reader import TSVReader
from data.iterators import EpochBatchIterator

logger = logging.getLogger(__name__)


@dataclass
class BaseTaskConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated path to training data list, will be iterated upon during epochs"
        },
    )
    valid_data: Optional[str] = field(
        default=None,
        metadata={"help": "validation data"},
    )
    selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "tsv headers"},
    )
    bpe_dir: Optional[str] = field(
        default=None,
        metadata={"help": "bpe directory"},
    )

    max_positions: int = field(
        default=1024, metadata={"help": "max tokens per example"},
    )
    max_src_length: int = field(
        default=70, metadata={"help": "the maximum text sequence length"}
    )
    patch_image_size: int = field(
        default=256, metadata={"help": "the image resolution"}
    )
    max_duration: int = field(
        default=15, metadata={"help": "the maximum audio duration"}
    )

    reader_separator: str = field(
        default='\t',
        metadata={"help": ""},
    )
    feature_encoder_spec: str = II("model.encoder.audio_adapter.feature_encoder_spec")

    head_type: ChoiceEnum(["text", "image", "audio", "vl", "al", "val"]) = field(
        default='vl',
        metadata={"help": "classifier head types"}
    )
    num_classes: Optional[int] = field(
        default=None,
        metadata={"help": "number of classes"}
    )
    use_two_images: bool = field(
        default=False,
        metadata={"help": "use for nlvr2"}
    )

    zero_shot: bool = field(
        default=False,
        metadata={"help": "whether perform zero-shot evaluation"}
    )


@register_task("base_task", dataclass=BaseTaskConfig)
class BaseTask(FairseqTask):
    def __init__(self, cfg: BaseTaskConfig, dictionary):
        super().__init__(cfg)
        self.dict = dictionary

        bpe_dict = {
            "_name": "gpt2",
            "gpt2_encoder_json": os.path.join(self.cfg.bpe_dir, "encoder.json"),
            "gpt2_vocab_bpe": os.path.join(self.cfg.bpe_dir, "vocab.bpe")
        }
        bpe_dict = DictConfig(bpe_dict)
        self.bpe = self.build_bpe(bpe_dict)
        self.metric = None

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""

        # load dictionary
        dictionary = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    _PATH_ALT = re.compile(r'(\[\d+-\d+\])')
    def _parse_dataset_paths(self):
        paths = []
        for path in self.cfg.data.split(','):
            mat = self._PATH_ALT.findall(path)
            if len(mat) == 0:
                paths.append(path)
            elif len(mat) == 1:
                start, end = tuple(map(int, mat[0].strip('[]').split('-')))
                for i in range(start, end + 1):
                    paths.append(self._PATH_ALT.sub(str(i), path))
            else:
                raise ValueError(f"only one expansion is supported, get {path}")
        return paths

    def load_dataset(self, split, epoch=1, **kwargs):
        if split == 'valid':
            file_path = self.cfg.valid_data
        else:
            paths = self._parse_dataset_paths()
            file_path = paths[(epoch - 1) % len(paths)]

        dataset = TSVReader(file_path, self.cfg.selected_cols, self.cfg.reader_separator)
        return dataset

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
        disable_shuffling=False,
        ensure_equal_batch=False,
        persistent_workers=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # create mini-batches with given size constraints
        global_num_batches = int(math.ceil(len(dataset) / max_sentences))
        total_row_count = len(dataset)
        sample_ids = list(range(total_row_count))
        if not disable_shuffling:
            with data_utils.numpy_seed(seed + epoch):
                np.random.shuffle(sample_ids)
        if skip_remainder_batch and global_num_batches % num_shards != 0:
            global_num_batches -= global_num_batches % num_shards
            total_row_count = global_num_batches * max_sentences
            sample_ids = sample_ids[:total_row_count]
        if ensure_equal_batch and global_num_batches % num_shards != 0:
            assert not skip_remainder_batch
            global_num_batches += num_shards - global_num_batches % num_shards
            total_row_count = global_num_batches * max_sentences
            sample_ids = sample_ids + sample_ids[:total_row_count-len(sample_ids)]

        global_batch_sampler = [
            [sample_ids[j] for j in range(i, min(i + max_sentences, total_row_count))]
            for i in range(0, total_row_count, max_sentences)
        ]

        # return a reusable, sharded iterator
        epoch_iter = EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=global_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            disable_shuffling=disable_shuffling,
            persistent_workers=persistent_workers
        )

        return epoch_iter

    @torch.no_grad()
    def begin_valid_epoch(self, epoch, model, subset):
        """Hook function called before the start of each validation epoch."""
        if self.metric is not None:
            self.metric.initialize()

    @torch.no_grad()
    def valid_step(self, sample, model, criterion, is_dummy_batch):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if not is_dummy_batch:
            self.eval_step(model, sample)
        return loss, sample_size, logging_output

    @torch.no_grad()
    def eval_step(self, models, sample):
        raise NotImplementedError

    @torch.no_grad()
    def merge_results(self, output_predict=False):
        if self.metric is not None:
            return self.metric.merge_results(output_predict=output_predict)

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dict

    @property
    def target_dictionary(self):
        return self.dict