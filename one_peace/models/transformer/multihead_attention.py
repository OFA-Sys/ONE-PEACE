# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging

import torch
import torch.nn.functional as F
from torch import nn

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout

from models.components import Linear, LayerNorm

logger = logging.getLogger(__name__)

try:
    import xformers
    from xformers.ops.fmha import memory_efficient_attention
    has_xformers = True
    logger.info('****** use memory_efficient_attention ******')
except ImportError:
    has_xformers = False
    logger.info('****** Import memory_efficient_attention fail, please install xFormers ******')


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        scale_heads=False,
        magneto_scale_attn=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if has_xformers:
            assert dropout == 0.0, "xformers doesn't support dropout"

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.c_attn = nn.Parameter(torch.ones((self.num_heads,)), requires_grad=True) if scale_heads else None
        self.ln = LayerNorm(self.embed_dim) if magneto_scale_attn else None

        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (FloatTensor, optional): equal to relative position bias
        """

        seq_len, bsz, embed_dim = x.size()
        if has_xformers:
            x = x.transpose(0, 1)
            new_seq_len = seq_len
            if seq_len % 8 != 0 and attn_mask is not None:
                pad_len = 8 - seq_len % 8
                new_seq_len = seq_len + pad_len
                x = F.pad(x, (0, 0, 0, pad_len), 'constant', 0)
                attn_mask = F.pad(attn_mask, (0, 0, 0, pad_len), 'constant', 0)
                attn_mask = F.pad(attn_mask, (0, pad_len), 'constant', float("-inf"))

            qkv_proj_weight = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0)
            qkv_proj_bias = torch.cat(
                (self.q_proj.bias, torch.zeros_like(self.v_proj.bias, requires_grad=False), self.v_proj.bias)
            )
            qkv = F.linear(x, qkv_proj_weight, qkv_proj_bias)
            qkv = qkv.reshape(bsz, new_seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = xformers.ops.unbind(qkv, 2)
            attn = memory_efficient_attention(
                q, k, v, attn_mask,
                self.dropout_p if self.training else 0.0,
                scale=self.scaling
            )
            attn = attn[:, :seq_len, :, :]
        else:
            q = self.q_proj(x).view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = self.k_proj(x).view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = self.v_proj(x).view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            if attn_mask is not None:
                attn_weights += attn_mask.view(-1, seq_len, seq_len)

            attn_weights_float = utils.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)
            attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim)
        if self.c_attn is not None:
            attn = attn.view(seq_len, bsz, self.num_heads, self.head_dim)
            attn = torch.einsum('nbhd,h->nbhd', attn, self.c_attn)
            attn = attn.reshape(seq_len, bsz, self.embed_dim)
        if self.ln is not None:
            attn = self.ln(attn)
        attn = self.out_proj(attn)

        return attn
