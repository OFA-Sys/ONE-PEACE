# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

"""
ONE-PEACE Classify
"""
from typing import Optional
from dataclasses import dataclass, field

import contextlib
import logging

import torch
from fairseq.models import register_model
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from ..unify_model_config import UnifyModelConfig
from .one_peace_base import ModelWrapper, OnePeaceClassifyHead, OnePeaceBaseModel, init_one_peace_params

logger = logging.getLogger(__name__)


@dataclass
class OnePeaceClassifyConfig(UnifyModelConfig):
    head_scale_ratio: int = field(
        default=1,
        metadata={"help": ""}
    )
    use_pooler: bool = field(
        default=False,
        metadata={"help": ""}
    )
    pooler_dropout: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    attn_pooling: bool = field(
        default=False,
        metadata={"help": ""}
    )

    use_image_features: bool = False
    freeze_finetune_updates: int = 0


@register_model("one_peace_classify", dataclass=OnePeaceClassifyConfig)
class OnePeaceClassifyModel(OnePeaceBaseModel):

    def __init__(
        self, cfg: OnePeaceClassifyConfig, src_dict, head_type,
        num_classes=None, use_two_images=False
    ):
        super().__init__(cfg, src_dict)

        embed_dim = self.cfg.encoder.embed_dim
        self.head_type = head_type
        self.classify_head = OnePeaceClassifyHead(
            attn_pooling=self.cfg.attn_pooling,
            use_pooler=self.cfg.use_pooler,
            pooler_dropout=self.cfg.pooler_dropout,
            input_dim=embed_dim,
            num_heads=self.cfg.encoder.attention_heads,
            head_scale_ratio=self.cfg.head_scale_ratio,
            num_classes=num_classes,
            use_two_images=use_two_images
        )

        cfg.encoder.use_text_moe = False
        cfg.encoder.use_image_moe = False
        cfg.encoder.use_audio_moe = False
        if self.head_type in ('text', 'vl', 'al'):
            cfg.encoder.use_text_moe = True
        if self.head_type in ('image', 'vl'):
            cfg.encoder.use_image_moe = True
        if self.head_type in ('audio', 'al'):
            cfg.encoder.use_audio_moe = True

        use_text_norm = self.head_type in ('text', 'vl', 'al')
        use_image_norm = self.head_type in ('image', 'vl')
        use_audio_norm = self.head_type in ('audio', 'al')
        self.encoder_wrapper = ModelWrapper(
            cfg.encoder,
            src_dict,
            use_text_norm=use_text_norm,
            use_image_norm=use_image_norm,
            use_audio_norm=use_audio_norm,
            num_layers=cfg.encoder.layers
        )

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
        src_images_2: Optional[torch.Tensor] = None,
        src_audios: Optional[torch.Tensor] = None,
        audio_padding_masks: Optional[torch.Tensor] = None
    ):
        encoder_type = self.head_type

        ft = self.cfg.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            enc_text_features, enc_image_features, enc_audio_features, \
            text_padding_masks, image_padding_masks, audio_padding_masks = self.encoder_wrapper(
                src_tokens=src_tokens,
                src_images=src_images,
                src_audios=src_audios,
                audio_padding_masks=audio_padding_masks,
                encoder_type=encoder_type,
                return_padding_mask=True
            )

        if src_images_2 is not None:
            enc_text_features_2, enc_image_features_2, enc_audio_features_2 = self.encoder_wrapper(
                src_tokens=src_tokens,
                src_images=src_images_2,
                is_second_image=True,
                src_audios=src_audios,
                audio_padding_masks=audio_padding_masks,
                encoder_type=encoder_type
            )
        else:
            enc_text_features_2 = None
            enc_image_features_2 = None
            enc_audio_features_2 = None

        if enc_text_features is not None and not self.cfg.use_image_features:
            logits = self.classify_head(enc_text_features, enc_text_features_2, text_padding_masks)
        elif enc_image_features is not None:
            logits = self.classify_head(enc_image_features, enc_image_features_2, image_padding_masks)
        elif enc_audio_features is not None:
            logits = self.classify_head(enc_audio_features, enc_audio_features_2, audio_padding_masks)
        else:
            raise NotImplementedError
        return logits

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        cfg.encoder.image_adapter.rel_bucket_size = task.cfg.patch_image_size // 16
        src_dict = task.source_dictionary
        head_type = task.cfg.head_type
        num_classes = task.cfg.num_classes
        use_two_images = task.cfg.use_two_images
        return cls(cfg, src_dict, head_type=head_type, num_classes=num_classes, use_two_images=use_two_images)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        self.remove_pretraining_modules(state_dict)

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]

    def remove_pretraining_modules(self, state_dict):
        if 'encoder_wrapper.fusion_model.text_layer_norm.weight' not in self.state_dict():
            if 'encoder_wrapper.fusion_model.text_layer_norm.weight' in state_dict:
                del state_dict['encoder_wrapper.fusion_model.text_layer_norm.weight']
                del state_dict['encoder_wrapper.fusion_model.text_layer_norm.bias']
        if 'encoder_wrapper.fusion_model.image_layer_norm.weight' not in self.state_dict():
            if 'encoder_wrapper.fusion_model.image_layer_norm.weight' in state_dict:
                del state_dict['encoder_wrapper.fusion_model.image_layer_norm.weight']
                del state_dict['encoder_wrapper.fusion_model.image_layer_norm.bias']
        if 'encoder_wrapper.fusion_model.audio_layer_norm.weight' not in self.state_dict():
            if 'encoder_wrapper.fusion_model.audio_layer_norm.weight' in state_dict:
                del state_dict['encoder_wrapper.fusion_model.audio_layer_norm.weight']
                del state_dict['encoder_wrapper.fusion_model.audio_layer_norm.bias']
        if 'text_proj.weight' in state_dict:
            del state_dict['text_proj.weight']
            del state_dict['text_proj.bias']
        if 'image_proj.weight' in state_dict:
            del state_dict['image_proj.weight']
            del state_dict['image_proj.bias']
        if 'audio_proj.weight' in state_dict:
            del state_dict['audio_proj.weight']
            del state_dict['audio_proj.bias']

        for param_name in list(state_dict.keys()):
            if self.head_type not in ('text', 'vl', 'al', 'val') and 'text_' in param_name:
                del state_dict[param_name]
            elif self.head_type not in ('image', 'vl', 'val') and 'image_' in param_name:
                del state_dict[param_name]
            elif self.head_type not in ('audio', 'al', 'val') and 'audio_' in param_name:
                del state_dict[param_name]
