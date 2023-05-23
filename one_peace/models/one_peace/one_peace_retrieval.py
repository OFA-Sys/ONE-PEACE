# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

"""
One-Piece Retrieval
"""
from typing import Optional
from dataclasses import dataclass

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from models.unify_model_config import UnifyModelConfig
from models.components import Linear
from models.one_peace.one_peace_base import ModelWrapper, OnePeaceBaseModel, init_one_peace_params

logger = logging.getLogger(__name__)


@dataclass
class OnePeaceRetrievalConfig(UnifyModelConfig):
    pass


@register_model("one_peace_retrieval", dataclass=OnePeaceRetrievalConfig)
class OnePeaceRetrievalModel(OnePeaceBaseModel):

    def __init__(self, cfg: OnePeaceRetrievalConfig, src_dict, head_type):
        super().__init__(cfg, src_dict)

        embed_dim = self.cfg.encoder.embed_dim
        self.head_type = head_type

        cfg.encoder.use_text_moe = False
        cfg.encoder.use_image_moe = False
        cfg.encoder.use_audio_moe = False
        if self.head_type in ('text', 'vl', 'al', 'val'):
            cfg.encoder.use_text_moe = True
        if self.head_type in ('image', 'vl', 'val'):
            cfg.encoder.use_image_moe = True
        if self.head_type in ('audio', 'al', 'val'):
            cfg.encoder.use_audio_moe = True

        self.encoder_wrapper = ModelWrapper(
            cfg.encoder,
            src_dict,
            use_text_norm=cfg.encoder.use_text_moe,
            use_image_norm=cfg.encoder.use_image_moe,
            use_audio_norm=cfg.encoder.use_audio_moe
        )
        if cfg.encoder.use_text_moe:
            self.text_proj = Linear(embed_dim, embed_dim)
        if cfg.encoder.use_image_moe:
            self.image_proj = Linear(embed_dim, embed_dim)
        if cfg.encoder.use_audio_moe:
            self.audio_proj = Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self.apply(init_one_peace_params)

        for i, layer in enumerate(self.encoder_wrapper.fusion_model.layers):
            if (i + 1) % cfg.encoder.fsdp_checkpoint_wrap_layer_preserve_frequency != 0:
                continue
            if (i + 1) % cfg.encoder.fsdp_checkpoint_wrap_layer_skip_frequency == 0:
                continue
            if cfg.encoder.checkpoint_activations:
                self.encoder_wrapper.fusion_model.layers[i] = fsdp_wrap(checkpoint_wrapper(layer))
            else:
                self.encoder_wrapper.fusion_model.layers[i] = fsdp_wrap(layer)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(
        self,
        src_tokens: Optional[torch.Tensor] = None,
        src_images: Optional[torch.Tensor] = None,
        src_audios: Optional[torch.Tensor] = None,
        audio_padding_masks: Optional[torch.Tensor] = None,
        return_logit_scale: bool = False,
        encoder_type: Optional[str] = None
    ):

        if return_logit_scale:
            with torch.no_grad():
                self.logit_scale.clamp_(0, math.log(100))
            logit_scale_exp = self.logit_scale.exp()
            return logit_scale_exp
        else:
            enc_text_features, enc_image_features, enc_audio_features = self.encoder_wrapper(
                src_tokens=src_tokens,
                src_images=src_images,
                src_audios=src_audios,
                audio_padding_masks=audio_padding_masks,
                encoder_type=encoder_type
            )

            if encoder_type == 'text':
                text_cls_logits = enc_text_features[:, 0, :]
                text_logits = F.normalize(self.text_proj(text_cls_logits), dim=1)
                return text_logits
            elif encoder_type == 'image':
                image_cls_logits = enc_image_features[:, 0, :]
                image_logits = F.normalize(self.image_proj(image_cls_logits), dim=1)
                return image_logits
            elif encoder_type == 'audio':
                audio_cls_logits = enc_audio_features[:, 0, :]
                audio_logits = F.normalize(self.audio_proj(audio_cls_logits), dim=1)
                return audio_logits
            else:
                raise NotImplementedError

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        cfg.encoder.image_adapter.rel_bucket_size = task.cfg.patch_image_size // 16
        src_dict = task.source_dictionary
        head_type = task.cfg.head_type
        return cls(cfg, src_dict, head_type)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        self.remove_pretraining_modules(state_dict)

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]

    def remove_pretraining_modules(self, state_dict):
        for param_name in list(state_dict.keys()):
            if self.head_type not in ('text', 'vl', 'al', 'val') and 'text_' in param_name:
                del state_dict[param_name]
            elif self.head_type not in ('image', 'vl', 'val') and 'image_' in param_name:
                del state_dict[param_name]
            elif self.head_type not in ('audio', 'al', 'val') and 'audio_' in param_name:
                del state_dict[param_name]
