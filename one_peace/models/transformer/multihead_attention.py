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

from one_peace.models.components import Linear, LayerNorm
from einops import rearrange

logger = logging.getLogger(__name__)

try:
    import xformers
    from xformers.ops.fmha import memory_efficient_attention
    has_xformers = True
    logger.info('****** use memory_efficient_attention ******')
except ImportError:
    has_xformers = False
    logger.info('****** Import memory_efficient_attention fail ******')

try:
    from apex.transformer.functional.fused_softmax import generic_scaled_masked_softmax
    has_fused = True
    logger.info('****** use generic_scaled_masked_softmax ******')
except ImportError:
    has_fused = False
    logger.info('****** Import generic_scaled_masked_softmax fail ******')


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    embed_dim = hidden_states.size(2)
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    hidden_states = hidden_states.reshape(-1, embed_dim)[indices]
    return  hidden_states, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, '(b s) ... -> b s ...', b=batch)


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
            assert dropout == 0.0, "xformers cant support dropout"

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

            if has_fused and attn_weights.dtype != torch.float32:
                attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
                key_padding_mask = key_padding_mask.view(bsz, 1, 1, seq_len).expand(-1, -1, seq_len, -1).contiguous()
                attn_weights = generic_scaled_masked_softmax(attn_weights, key_padding_mask, 1.0)
                attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, seq_len)
            else:
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
