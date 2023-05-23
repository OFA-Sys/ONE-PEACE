#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import os
import sys
import json

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from fairseq import distributed_utils, checkpoint_utils, tasks, utils
from fairseq.logging import progress_bar
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import omegaconf_no_object_check
from fairseq.dataclass.initialize import add_defaults
from omegaconf import DictConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("one_peace.evaluate")


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def apply_bfloat16(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.bfloat16)
    return t


def main(cfg: DictConfig):
    utils.import_user_module(cfg.common)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_bf16 = cfg.common.bf16
    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if cfg.task.zero_shot:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count
        )
    logger.info(saved_cfg)

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.common_eval.path)):
        # if kwargs['ema_eval']:
        #     logger.info("loading EMA weights from {}".format(ckpt_path))
        #     model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_bf16:
            logger.info('evaluate with bf16')
            model.bfloat16()
        if use_fp16:
            logger.info('evaluate with fp16')
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    task.begin_valid_epoch(epoch=1, model=models[0], subset=cfg.dataset.gen_subset)
    for sample in progress:
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)
        elif cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)
        task.eval_step(models[0], sample)
        progress.log({"sentences": sample["nsentences"]})

    if task.metric is not None:
        stats = task.metric.merge_results(output_predict=True)
        if (not dist.is_initialized() or dist.get_rank() == 0) and cfg.common_eval.results_path is not None:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            with open(output_path, 'w') as fw:
                json.dump(stats, fw)
        if 'predict_results' in stats:
            del stats['predict_results']
        if 'predict_txt' in stats:
            assert 'predict_img' in stats
            del stats['predict_txt']
            del stats['predict_img']
        logger.info(stats)


def cli_main():
    from hydra._internal.utils import get_args

    cfg_dir = get_args().config_dir
    cfg_name = '{}.yaml'.format(get_args().config_name)
    cfg_path = os.path.join(cfg_dir, cfg_name)
    command_overrides = get_args().overrides

    cfg_default = OmegaConf.structured(FairseqConfig)
    cfg_yaml = OmegaConf.load(cfg_path)
    cfg_command = OmegaConf.from_cli(command_overrides)
    if cfg_command.common_eval is not None and cfg_command.common_eval.model_overrides is not None:
        for command in command_overrides:
            k, v = command.split('=', 1)
            if k == 'common_eval.model_overrides':
                cfg_command.common_eval.model_overrides = v
                break

    if 'default_yaml' in cfg_yaml:
        cfg_default_yaml = OmegaConf.load(cfg_yaml.default_yaml)
        del cfg_yaml['default_yaml']
        cfg = OmegaConf.merge(cfg_default, cfg_default_yaml, cfg_yaml, cfg_command)
    else:
        cfg = OmegaConf.merge(cfg_default, cfg_yaml, cfg_command)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()