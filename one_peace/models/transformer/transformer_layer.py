# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Optional
import logging

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.modules.fairseq_dropout import FairseqDropout

from one_peace.models.components import Linear, LayerNorm
from .multihead_attention import MultiheadAttention

logger = logging.getLogger(__name__)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (1, x.shape[1], 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GeGLU(nn.Module):
    """ GeGLU """

    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.wi_0 = Linear(embed_dim, ffn_dim, bias=False)
        self.wi_1 = Linear(embed_dim, ffn_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x_gelu = self.act(self.wi_0(x))
        x_linear = self.wi_1(x)
        x = x_gelu * x_linear
        return x

@torch.jit.script
def fused_dropout_res(
    x: torch.Tensor,
    gamma: Optional[torch.nn.parameter.Parameter],
    residual: torch.Tensor,
    dropout_prob: float,
    drop_path_prob: float
) -> torch.Tensor:
    a = torch.nn.functional.dropout(x, dropout_prob) if dropout_prob > 0.0 else x
    b = gamma * a if gamma is not None else a
    if drop_path_prob > 0.0:
        keep_prob = 1 - drop_path_prob
        shape = (1, x.shape[1], 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        c = b * random_tensor
        return torch.add(c, residual)
    else:
        return torch.add(b, residual)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, drop_path_rate=0.0):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.ffn_embed_dim = cfg.ffn_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            float(cfg.activation_dropout), module_name=self.__class__.__name__
        )

        self.dropout_prob = cfg.dropout
        self.drop_path_prob = drop_path_rate

        ln = None
        if cfg.share_ln:
            ln = LayerNorm(self.ffn_embed_dim) if cfg.scale_fc else nn.Identity()

        if cfg.use_text_moe:
            self.text_ffn = self.build_ffn(cfg, ln) if not cfg.use_geglu else self.build_geglu_ffn(cfg, ln)
        if cfg.use_image_moe:
            self.image_ffn = self.build_ffn(cfg, ln) if not cfg.use_geglu else self.build_geglu_ffn(cfg, ln)
        if cfg.use_audio_moe:
            self.audio_ffn = self.build_ffn(cfg, ln) if not cfg.use_geglu else self.build_geglu_ffn(cfg, ln)

        self.attn_ln = LayerNorm(self.embed_dim) if cfg.scale_attn else None
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.gamma_1 = None
        self.gamma_2 = None
        if cfg.use_layer_scale:
            self.gamma_1 = nn.Parameter(cfg.layer_scale_init_value * torch.ones((self.embed_dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(cfg.layer_scale_init_value * torch.ones((self.embed_dim)), requires_grad=True)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.attention_heads,
            dropout=cfg.attention_dropout,
            scale_heads=cfg.scale_heads,
            magneto_scale_attn=cfg.magneto_scale_attn
        )

    def build_ffn(self, cfg, ln=None):
        if ln is None:
            ln = LayerNorm(self.ffn_embed_dim) if cfg.scale_fc else nn.Identity()

        return nn.Sequential(
            *[
                Linear(self.embed_dim, self.ffn_embed_dim),
                nn.GELU(),
                self.activation_dropout_module,
                ln,
                Linear(self.ffn_embed_dim, self.embed_dim)
            ]
        )

    def build_geglu_ffn(self, cfg, ln=None):
        if ln is None:
            ln = LayerNorm(self.ffn_embed_dim) if cfg.scale_fc else nn.Identity()

        return nn.Sequential(
            *[
                GeGLU(self.embed_dim, self.ffn_embed_dim),
                self.activation_dropout_module,
                ln,
                Linear(self.ffn_embed_dim, self.embed_dim)
            ]
        )

    def residual_connection(self, x, residual, gamma=None):
        if gamma is not None:
            return residual + self.drop_path(gamma * x)
        else:
            return residual + self.drop_path(x)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        self_attn_bias: Optional[Tensor] = None,
        encoder_type: Optional[str] = None,
        text_seq_len: Optional[int] = None,
        image_seq_len: Optional[int] = None,
        audio_seq_len: Optional[int] = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=self_attn_bias
        )
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = fused_dropout_res(
            x, self.gamma_1, residual,
            self.dropout_prob if self.training else 0.0,
            self.drop_path_prob if self.training else 0.0
        )
        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual, gamma=self.gamma_1)

        residual = x
        x = self.final_layer_norm(x)
        if encoder_type == 'text':
            x = self.text_ffn(x)
        elif encoder_type == 'image':
            x = self.image_ffn(x)
        elif encoder_type == 'audio':
            x = self.audio_ffn(x)
        elif encoder_type == 'vl':
            text_x = self.text_ffn(x[:text_seq_len, :, :])
            image_x = self.image_ffn(x[-image_seq_len:, :, :])
            x = torch.cat([text_x, image_x], dim=0)
        elif encoder_type == 'al':
            text_x = self.text_ffn(x[:text_seq_len, :, :])
            audio_x = self.audio_ffn(x[-audio_seq_len:, :, :])
            x = torch.cat([text_x, audio_x], dim=0)
        else:
            raise NotImplementedError
        x = fused_dropout_res(
            x, self.gamma_2, residual,
            self.dropout_prob if self.training else 0.0,
            self.drop_path_prob if self.training else 0.0
        )
        # x = self.dropout_module(x)
        # x = self.residual_connection(x, residual, gamma=self.gamma_2)

        return x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]
