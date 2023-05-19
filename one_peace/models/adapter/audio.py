# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import math
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout
from fairseq import utils

from one_peace.models.components import Embedding, trunc_normal_, LayerNorm, Linear

logger = logging.getLogger(__name__)


def make_token_bucket_position(bucket_size, max_position):
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    assert bucket_size < max_position
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid-1, torch.abs(relative_pos))
    log_pos = mid + torch.ceil(
        torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid-1)
    ).long()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos*sign).long()
    return bucket_pos + bucket_size - 1


class AudioAdapter(torch.nn.Module):
    def __init__(self, cfg, embed_dim, attention_heads, num_layers=None):
        super().__init__()
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.alpha = cfg.shrink_alpha

        if cfg.feature_encoder_spec is not None:
            feature_encoder_spec = eval(cfg.feature_encoder_spec)
            feature_embed_dim = feature_encoder_spec[-1][0]
            self.embed_audios = torch.nn.Sequential(
                ConvFeatureExtractionModel(
                    conv_layers=feature_encoder_spec,
                    dropout=0.0,
                    conv_bias=cfg.conv_bias
                ),
                TransposeLast(),
                LayerNorm(feature_embed_dim),
                Linear(feature_embed_dim, embed_dim),
            )

        if cfg.abs_pos_type == 'conv':
            num_pos_layers = cfg.conv_pos_depth
            k = max(3, cfg.conv_pos_width // num_pos_layers)
            self.embed_positions = nn.Sequential(
                TransposeLast(),
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            embed_dim,
                            embed_dim,
                            kernel_size=k,
                            padding=k // 2,
                            groups=cfg.conv_pos_groups,
                        ),
                        SamePad(k),
                        TransposeLast(),
                        nn.LayerNorm(embed_dim, elementwise_affine=False),
                        TransposeLast(),
                        nn.GELU(),
                    )
                    for _ in range(num_pos_layers)
                ],
                TransposeLast(),
            )
            if cfg.conv_pos_pre_ln:
                self.embed_positions = nn.Sequential(LayerNorm(embed_dim), self.embed_positions)
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_pos_embed)
        elif cfg.abs_pos_type == 'fixed':
            self.embed_positions = Embedding(1024 + 2, embed_dim)
        else:
            raise NotImplementedError

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cls_embedding = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        if cfg.add_type_embedding:
            self.type_embedding = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.type_embedding = None

        if cfg.use_attn_bias:
            num_rel_dis = 2 * cfg.bucket_size - 1
            rp_bucket = make_token_bucket_position(cfg.bucket_size, max_position=1024)
            rp_bucket[0, :] = num_rel_dis
            rp_bucket[:, 0] = num_rel_dis + 1
            rp_bucket[0, 0] = num_rel_dis + 2
            self.register_buffer("rp_bucket", rp_bucket)
            self.rel_pos_table_list = torch.nn.ModuleList(
                [
                    Embedding(num_rel_dis + 3, attention_heads, zero_init=True)
                    for _ in range(num_layers if num_layers is not None else 1)
                ]
            )
        else:
            self.rel_pos_table_list = None

        self.mask_embedding = torch.nn.Parameter(torch.zeros(1, embed_dim))

        trunc_normal_(self.cls_embedding)
        trunc_normal_(self.mask_embedding)
        if cfg.abs_pos_type == 'fixed':
            trunc_normal_(self.embed_positions.weight)

    def get_rel_pos_bias(self, bsz, seq_len):
        rel_pos_bias_list = []
        for rel_pos_table in self.rel_pos_table_list:
            rp_bucket = self.rp_bucket[:seq_len, :seq_len]
            values = rel_pos_table(rp_bucket).unsqueeze(0).expand(bsz, -1, -1, -1)
            values = values.permute(0, 3, 1, 2)
            rel_pos_bias_list.append(values)
        return rel_pos_bias_list

    def gather_features(self, adapter_embedding, self_attn_bias_list, position_ids):
        seq_len, embed_dim = adapter_embedding.shape[-2:]
        adapter_embedding = adapter_embedding.gather(1, position_ids[:, 1:, None].expand(-1, -1, embed_dim) - 1)

        gather_seq_len = position_ids.size(1)
        if self_attn_bias_list is not None:
            new_self_attn_bias_list = []
            for self_attn_bias in self_attn_bias_list:
                self_attn_bias = self_attn_bias.gather(
                    2, position_ids[:, None, :, None].expand(-1, self_attn_bias.size(1), -1, seq_len+1)
                ).gather(3, position_ids[:, None, None, :].expand(-1, self_attn_bias.size(1), gather_seq_len, -1))
                new_self_attn_bias_list.append(self_attn_bias)
        else:
            new_self_attn_bias_list = None

        return adapter_embedding, new_self_attn_bias_list

    def forward(
        self,
        src_audios,
        padding_mask,
        preserve_ids=None,
        preserve_embed=None,
        mask_token=None
    ):
        """
        Args:
            src_audios List of audios
        Returns:
                - **x** (Tensor): the processed embedding of
                  shape `(batch, src_len, embed_dim)`
                - **padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **self_attn_bias_list** (Tensor): list of attention bias in self attention
                 of shape `(bsz, num_attention_heads, src_len, src_len)`.
        """

        bsz = padding_mask.size(0)
        position_ids = utils.new_arange(padding_mask)
        if self.rel_pos_table_list is not None:
            self_attn_bias_list = self.get_rel_pos_bias(bsz, position_ids.size(1))
        else:
            self_attn_bias_list = None

        if preserve_embed is not None:
            pos_embed = self.embed_positions(position_ids)
            seq_len, embed_dim = pos_embed.size(1), pos_embed.size(2)
            adapter_embedding = mask_token.repeat(bsz * seq_len, 1)
            right_preserve_indices = torch.nonzero(preserve_ids.ne(-1).flatten(), as_tuple=False).flatten()
            left_preserve_indices = preserve_ids + (torch.arange(bsz) * seq_len).unsqueeze(1).type_as(preserve_ids)
            left_preserve_indices = left_preserve_indices.view(-1)[right_preserve_indices]
            adapter_embedding[left_preserve_indices] = preserve_embed.reshape(-1, embed_dim)[right_preserve_indices]
            adapter_embedding = adapter_embedding.reshape(bsz, seq_len, embed_dim)
        else:
            adapter_embedding = self.embed_audios(src_audios)
            if preserve_ids is not None:
                padding_mask = preserve_ids.eq(-1)
                position_ids = preserve_ids.masked_fill(preserve_ids.eq(-1), preserve_ids.size(1) - 1)
                adapter_embedding, self_attn_bias_list = self.gather_features(
                    adapter_embedding, self_attn_bias_list, position_ids
                )
            pos_embed = self.embed_positions(adapter_embedding)
            pos_embed = torch.cat([self.cls_pos_embed.expand(bsz, -1, -1), pos_embed], dim=1)
            cls_embedding = self.cls_embedding.expand(bsz, -1, -1)
            adapter_embedding = torch.cat([cls_embedding, adapter_embedding], dim=1)

            if self.layernorm_embedding is not None:
                adapter_embedding = self.layernorm_embedding(adapter_embedding)
            if self.alpha != 1.0:
                adapter_embedding = adapter_embedding * self.alpha + adapter_embedding.detach() * (1 - self.alpha)

        x = adapter_embedding + pos_embed

        if self.type_embedding is not None:
            x += self.type_embedding.expand_as(x)
        x = self.dropout_module(x)

        return x, padding_mask, self_attn_bias_list

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        if prefix + 'rel_pos_table.weight' in state_dict:
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table.weight']
            state_dict[prefix + 'rel_pos_table_list.0.weight'] = rel_pos_table_weight
            del state_dict[prefix + 'rel_pos_table.weight']
        if self.rel_pos_table_list is not None and len(self.rel_pos_table_list) > 1 \
                and prefix + 'rel_pos_table_list.1.weight' not in state_dict:
            logger.info('copy rel_pos_weight to each layer')
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table_list.0.weight']
            for i in range(len(self.rel_pos_table_list)):
                state_dict[prefix + 'rel_pos_table_list.{}.weight'.format(i)] = rel_pos_table_weight.clone()

        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        return state_dict


class TransposeLast(nn.Module):
    def __init__(self, tranpose_dim=-2):
        super().__init__()
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        return x.transpose(self.tranpose_dim, -1)


class SamePad(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        conv_bias: bool = False
    ):
        super().__init__()

        def block(
            n_in,
            n_out,
            k,
            stride,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(),
                    LayerNorm(dim),
                    TransposeLast(),
                ),
                nn.GELU(),
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x
