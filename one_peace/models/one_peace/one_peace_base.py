# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

"""
ONE-PEACE Base Model Wrapper
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import math
from fairseq.models import register_model, BaseFairseqModel
from fairseq import utils

from ..unify_model_config import UnifyModelConfig
from ..components import trunc_normal_
from ..adapter.text import TextAdapter
from ..adapter.image import ImageAdapter
from ..adapter.audio import AudioAdapter
from ..transformer.transformer_encoder import TransformerEncoder
from ..components import Linear, LayerNorm

logger = logging.getLogger(__name__)

try:
    import xformers
    from xformers.ops.fmha import memory_efficient_attention
    has_xformers = True
    logger.info('****** use memory_efficient_attention ******')
except ImportError:
    has_xformers = False
    logger.info('****** Import memory_efficient_attention fail, please install xFormers ******')


class ModelWrapper(nn.Module):
    def __init__(
        self,
        cfg,
        src_dict=None,
        use_text_norm=True,
        use_image_norm=True,
        use_audio_norm=True,
        num_layers=None
    ):
        super(ModelWrapper, self).__init__()

        embed_dim = cfg.embed_dim
        attention_heads = cfg.attention_heads
        if cfg.use_text_moe:
            self.text_adapter = TextAdapter(cfg.text_adapter, embed_dim, attention_heads, src_dict, num_layers)
        if cfg.use_image_moe:
            self.image_adapter = ImageAdapter(cfg.image_adapter, embed_dim, attention_heads, num_layers)
        if cfg.use_audio_moe:
            self.audio_adapter = AudioAdapter(cfg.audio_adapter, embed_dim, attention_heads, num_layers)

        self.fusion_model = TransformerEncoder(
            cfg,
            src_dict,
            use_text_norm=use_text_norm,
            use_image_norm=use_image_norm,
            use_audio_norm=use_audio_norm
        )

    def forward(
        self,
        src_tokens: Optional[torch.Tensor] = None,
        text_preserve_ids: Optional[torch.Tensor] = None,
        text_preserve_embed: Optional[torch.Tensor] = None,
        text_mask_token: Optional[torch.Tensor] = None,
        src_images: Optional[torch.Tensor]= None,
        image_preserve_ids: Optional[torch.Tensor] = None,
        image_preserve_embed: Optional[torch.Tensor] = None,
        image_mask_token: Optional[torch.Tensor] = None,
        is_second_image: bool = False,
        src_audios: Optional[torch.Tensor] = None,
        audio_padding_masks: Optional[torch.Tensor] = None,
        audio_preserve_ids: Optional[torch.Tensor] = None,
        audio_preserve_embed: Optional[torch.Tensor] = None,
        audio_mask_token: Optional[torch.Tensor] = None,
        encoder_type: Optional[str] = None,
        return_padding_mask: bool = False
    ):

        text_info, image_info, audio_info = None, None, None
        if encoder_type in ('text', 'vl', 'al', 'val'):
            text_info = self.text_adapter(
                src_tokens, text_preserve_ids, text_preserve_embed, text_mask_token
            )
        if encoder_type in ('image', 'vl', 'val'):
            image_info = self.image_adapter(
                src_images, image_preserve_ids, image_preserve_embed, image_mask_token, is_second_image
            )
        if encoder_type in ('audio', 'al', 'val'):
            audio_info = self.audio_adapter(
                src_audios, audio_padding_masks,
                preserve_ids=audio_preserve_ids,
                preserve_embed=audio_preserve_embed,
                mask_token=audio_mask_token
            )

        model_out = self.fusion_model(
            text_info,
            image_info,
            audio_info,
            encoder_type=encoder_type
        )

        model_logits = model_out['encoder_out'][0].transpose(0, 1)
        encoder_padding_mask = model_out['encoder_padding_mask']
        text_features, image_features, audio_features = None, None, None
        text_padding_masks, image_padding_masks, audio_padding_masks = None, None, None
        if encoder_type in ('text', 'vl', 'al', 'val'):
            text_features = model_logits[:, :text_info[0].size(1), :]
            text_padding_masks = encoder_padding_mask[:, :text_info[0].size(1)]
        if encoder_type in ('image', 'vl', 'val'):
            image_features = model_logits[:, -image_info[0].size(1):, :]
            image_padding_masks = encoder_padding_mask[:, -image_info[0].size(1):]
        if encoder_type in ('audio', 'al', 'val'):
            audio_features = model_logits[:, -audio_info[0].size(1):, :]
            audio_padding_masks = encoder_padding_mask[:, -audio_info[0].size(1):]
        if return_padding_mask:
            return text_features, image_features, audio_features, \
                   text_padding_masks, image_padding_masks, audio_padding_masks
        else:
            return text_features, image_features, audio_features


class MultiheadAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.q = nn.Parameter(torch.zeros(1, 1, self.num_heads, self.head_dim))
        trunc_normal_(self.q)

    def forward(self, x, key_padding_mask=None):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        x = x.transpose(0, 1)

        seq_len, bsz, embed_dim = x.size()
        q = self.q.expand(1, bsz, -1, -1).reshape(1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = attn_weights.view(bsz, self.num_heads, 1, seq_len)
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, seq_len).contiguous()
        attn_weights.masked_fill_(key_padding_mask.expand(bsz, self.num_heads, 1, seq_len), -math.inf)
        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_probs = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_probs.view(bsz * self.num_heads, 1, seq_len)
        attn = torch.bmm(attn_probs, v)

        attn = attn.reshape(bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn


class OnePeaceClassifyHead(nn.Module):
    """Head for classify tasks."""

    def __init__(
        self,
        attn_pooling: bool,
        use_pooler: bool,
        pooler_dropout: float,
        input_dim: int,
        num_heads: int,
        head_scale_ratio: float,
        num_classes: int,
        use_two_images: bool = False
    ):
        super().__init__()
        self.attn_pooling = attn_pooling
        self.norm = LayerNorm(input_dim)

        self.attn_pooling_func = None
        if self.attn_pooling:
            self.attn_pooling_func = MultiheadAttentionPooling(input_dim, num_heads)

        if use_pooler:
            self.pooler = nn.Sequential(
                nn.Dropout(p=pooler_dropout),
                Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Dropout(p=pooler_dropout)
            )
        else:
            self.pooler = None

        classifier_input_dim = input_dim * 2 if use_two_images else input_dim
        inner_dim = int(input_dim * head_scale_ratio)
        self.classifier = nn.Sequential(
            Linear(classifier_input_dim, inner_dim),
            LayerNorm(inner_dim),
            nn.GELU(),
            Linear(inner_dim, num_classes)
        )

    def forward_features(self, features, padding_masks):
        if self.attn_pooling:
            other_logits = features[:, 1:, :]
            padding_masks = padding_masks[:, 1:]
            x = self.attn_pooling_func(other_logits, padding_masks)
            x = self.norm(x)
        else:
            x = features[:, 0, :]

        x = self.pooler(x) if self.pooler is not None else x
        return x

    def forward(self, features_1, features_2, padding_masks):
        x = self.forward_features(features_1, padding_masks)
        if features_2 is not None:
            x_2 = self.forward_features(features_2, padding_masks)
            x = torch.cat([x, x_2], dim=1)

        x = self.classifier(x)
        return x


@register_model("one_peace_base", dataclass=UnifyModelConfig)
class OnePeaceBaseModel(BaseFairseqModel):
    def __init__(self, cfg: UnifyModelConfig, src_dict):
        super().__init__()
        self.cfg = cfg
        self.src_dict = src_dict

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        src_dict = task.source_dictionary
        return cls(cfg, src_dict)

    def no_weight_decay(self):
        return {
            'encoder_wrapper.text_adapter.embed_positions.weight', 'encoder_wrapper.text_adapter.cls_embedding',
            'encoder_wrapper.image_adapter.pos_embed', 'encoder_wrapper.image_adapter.cls_embedding',
            'encoder_wrapper.audio_adapter.cls_embedding',
            'decoder_wrapper.text_adapter.embed_positions.weight', 'decoder_wrapper.text_adapter.cls_embedding'
            'decoder_wrapper.image_adapter.pos_embed', 'decoder_wrapper.image_adapter.cls_embedding',
            'decoder_wrapper.audio_adapter.embed_positions.weight', 'decoder_wrapper.audio_adapter.cls_embedding'
        }


def init_one_peace_params(module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        if not(hasattr(module, 'elementwise_affine') and module.elementwise_affine == False):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv2d):
        # trunc_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)