# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import numpy as np
import logging
from scipy import interpolate

import torch
import torch.nn.functional as F
from fairseq.modules import FairseqDropout

from one_peace.models.components import Embedding, trunc_normal_, LayerNorm

logger = logging.getLogger(__name__)


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class LayerNorm2D(torch.nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ImageAdapter(torch.nn.Module):
    def __init__(self, cfg, embed_dim, attention_heads, num_layers=None):
        super().__init__()
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.alpha = cfg.shrink_alpha

        if cfg.vision_encoder_type == 'mlp':
            self.embed_images = torch.nn.Conv2d(
                in_channels=3,
                out_channels=embed_dim,
                kernel_size=16,
                stride=16,
                bias=False
            )
        elif cfg.vision_encoder_type == 'hmlp':
            self.embed_images = torch.nn.Sequential(
                *[torch.nn.Conv2d(3, embed_dim//4, kernel_size=4, stride=4),
                LayerNorm2D(embed_dim//4),
                torch.nn.GELU(),
                torch.nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                LayerNorm2D(embed_dim//4),
                torch.nn.GELU(),
                torch.nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2)
            ])
        else:
            self.embed_images = None

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cls_embedding = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        if cfg.add_type_embedding:
            self.type_embedding = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.type_embedding_2 = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.type_embedding = None
            self.type_embedding_2 = None

        self.bucket_size = cfg.bucket_size
        self.pos_embed = torch.nn.Parameter(torch.zeros(self.bucket_size ** 2 + 1, embed_dim))

        position_idx = torch.arange(self.bucket_size ** 2 + 1)
        self.register_buffer("position_idx", position_idx)

        if cfg.use_attn_bias:
            self.rel_bucket_size = cfg.rel_bucket_size
            num_rel_dis = (2 * self.rel_bucket_size - 1) * (2 * self.rel_bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(self.rel_bucket_size, num_rel_dis)
            self.register_buffer("rp_bucket", rp_bucket)
            self.rel_pos_table_list = torch.nn.ModuleList(
                [
                    Embedding(num_rel_dis, attention_heads, zero_init=True)
                    for _ in range(num_layers if num_layers is not None else 1)
                ]
            )
        else:
            self.rel_pos_table_list = None

        trunc_normal_(self.cls_embedding)
        trunc_normal_(self.pos_embed)

    def _geometric_sequence_interpolation(self, src_size, dst_size, sequence, num):
        """Get new sequence via geometric sequence interpolation.
        Args:
            src_size (int): Pos_embedding size in pre-trained model.
            dst_size (int): Pos_embedding size in the current model.
            sequence (tensor): The relative position bias of the pretrain
                model after removing the extra tokens.
            num (int): Number of attention heads.
        Returns:
            new_sequence (tensor): Geometric sequence interpolate the
                pre-trained relative position bias to the size of
                the current model.
        """

        def geometric_progression(a, r, n):
            return a * (1.0 - r ** n) / (1.0 - r)

        # Here is a binary function.
        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        # The position of each interpolated point is determined
        # by the ratio obtained by dichotomy.
        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q ** (i + 1)
        r_ids = [-_ for _ in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        # Interpolation functions are being executed and called.
        new_sequence = []
        for i in range(num):
            z = sequence[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            new_sequence.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(sequence))
        new_sequence = torch.cat(new_sequence, dim=-1)
        return new_sequence

    def get_rel_pos_bias(self, bsz):
        rel_pos_bias_list = []
        for rel_pos_table in self.rel_pos_table_list:
            rp_bucket = self.rp_bucket
            values = rel_pos_table(rp_bucket).unsqueeze(0).expand(bsz, -1, -1, -1)
            values = values.permute(0, 3, 1, 2)
            rel_pos_bias_list.append(values)
        return rel_pos_bias_list

    def get_embed_positions(self, bsz, window_size):
        if window_size != self.bucket_size:
            cls_pos_emebd = self.pos_embed[:1]
            old_pos_embed = self.pos_embed[1:]
            old_pos_embed = old_pos_embed.reshape(1, self.bucket_size, self.bucket_size, -1).permute(0, 3, 1, 2)
            old_pos_embed_float = old_pos_embed.float()
            pos_embed = F.interpolate(old_pos_embed_float, size=(window_size, window_size), mode='bicubic')
            pos_embed = pos_embed.type_as(old_pos_embed)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(window_size ** 2, -1)
            pos_embed = torch.cat([cls_pos_emebd, pos_embed], dim=0)
        else:
            pos_embed = self.pos_embed
        pos_embed = pos_embed.unsqueeze(0).expand(bsz, -1, -1)
        return pos_embed

    def gather_features(self, adapter_embedding, pos_embed, self_attn_bias_list, position_ids):
        seq_len, embed_dim = adapter_embedding.shape[-2:]
        gather_seq_len = position_ids.size(1)
        adapter_embedding = adapter_embedding.gather(1, position_ids[:, :, None].expand(-1, -1, embed_dim))
        pos_embed = pos_embed.gather(1, position_ids[:, :, None].expand(-1, -1, embed_dim))

        if self_attn_bias_list is not None:
            new_self_attn_bias_list = []
            for self_attn_bias in self_attn_bias_list:
                self_attn_bias = self_attn_bias.gather(
                    2, position_ids[:, None, :, None].expand(-1, self_attn_bias.size(1), -1, seq_len)
                ).gather(3, position_ids[:, None, None, :].expand(-1, self_attn_bias.size(1), gather_seq_len, -1))
                new_self_attn_bias_list.append(self_attn_bias)
        else:
            new_self_attn_bias_list = None

        return adapter_embedding, pos_embed, new_self_attn_bias_list

    def forward(self, src_images, preserve_ids=None, preserve_embed=None, mask_token=None, is_second_image=False):
        """
        Args:
            src_images (Tensor): images of shape
                `(batch, channels, h, w)`
        Returns:
                - **x** (Tensor): the processed embedding of
                  shape `(batch, src_len, embed_dim)`
                - **padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **self_attn_bias_list** (Tensor): list of attention bias in self attention
                 of shape `(bsz, num_attention_heads, src_len, src_len)`.
        """

        bsz = src_images.size(0)
        window_size = (src_images.size(2) // 16)
        padding_mask = src_images.new_zeros((bsz, window_size**2+1)).bool()
        pos_embed = self.get_embed_positions(bsz, window_size)
        if self.rel_pos_table_list is not None:
            self_attn_bias_list = self.get_rel_pos_bias(bsz)
        else:
            self_attn_bias_list = None

        if preserve_embed is not None:
            seq_len, embed_dim = pos_embed.size(1), pos_embed.size(2)
            adapter_embedding = mask_token.repeat(bsz * seq_len, 1)
            right_preserve_indices = torch.nonzero(preserve_ids.ne(-1).flatten(), as_tuple=False).flatten()
            left_preserve_indices = preserve_ids + (torch.arange(bsz) * seq_len).unsqueeze(1).type_as(preserve_ids)
            left_preserve_indices = left_preserve_indices.view(-1)[right_preserve_indices]
            adapter_embedding[left_preserve_indices] = preserve_embed.reshape(-1, embed_dim)[right_preserve_indices]
            adapter_embedding = adapter_embedding.reshape(bsz, seq_len, embed_dim)
        else:
            adapter_embedding = self.embed_images(src_images).flatten(2).transpose(1, 2)
            cls_embedding = self.cls_embedding.expand(bsz, -1, -1)
            adapter_embedding = torch.cat([cls_embedding, adapter_embedding], dim=1)
            if preserve_ids is not None:
                padding_mask = preserve_ids.eq(-1)
                position_ids = preserve_ids.masked_fill(preserve_ids.eq(-1), preserve_ids.size(1) - 1)
                adapter_embedding, pos_embed, self_attn_bias_list = self.gather_features(
                    adapter_embedding, pos_embed, self_attn_bias_list, position_ids
                )
            if self.layernorm_embedding is not None:
                adapter_embedding = self.layernorm_embedding(adapter_embedding)
            if self.alpha != 1.0:
                adapter_embedding = adapter_embedding * self.alpha + adapter_embedding.detach() * (1 - self.alpha)

        x = adapter_embedding + pos_embed

        if self.type_embedding is not None:
            x += self.type_embedding.expand_as(x)
        if is_second_image and self.type_embedding_2 is not None:
            x += self.type_embedding_2.expand_as(x)
        x = self.dropout_module(x)

        return x, padding_mask, self_attn_bias_list

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        if prefix + 'rel_pos_table.weight' in state_dict:
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table.weight']
            state_dict[prefix + 'rel_pos_table_list.0.weight'] = rel_pos_table_weight
            del state_dict[prefix + 'rel_pos_table.weight']

        if prefix + 'rel_pos_table_list.0.weight' in state_dict and \
                (2 * self.rel_bucket_size - 1) ** 2 + 3 > state_dict[prefix + 'rel_pos_table_list.0.weight'].size(0):
            logger.info('interpolate relative position embedding')
            num_extra_tokens = 3
            num_attn_heads = state_dict[prefix + 'rel_pos_table_list.0.weight'].size(-1)
            src_size = int((state_dict[prefix + 'rel_pos_table_list.0.weight'].size(0) - num_extra_tokens) ** 0.5)
            dst_size = 2 * self.rel_bucket_size - 1

            extra_tokens = state_dict[prefix + 'rel_pos_table_list.0.weight'][-num_extra_tokens:, :]
            rel_pos_bias = state_dict[prefix + 'rel_pos_table_list.0.weight'][:-num_extra_tokens, :]
            new_rel_pos_bias = self._geometric_sequence_interpolation(
                src_size, dst_size, rel_pos_bias.cpu(), num_attn_heads)
            new_rel_pos_bias = new_rel_pos_bias.to(extra_tokens)
            new_rel_pos_bias = torch.cat((new_rel_pos_bias, extra_tokens), dim=0)
            state_dict[prefix + 'rel_pos_table_list.0.weight'] = new_rel_pos_bias
            state_dict[prefix + 'rp_bucket'] = self.state_dict()['rp_bucket']

        if self.rel_pos_table_list is not None and len(self.rel_pos_table_list) > 1 \
                and prefix + 'rel_pos_table_list.1.weight' not in state_dict:
            logger.info('copy rel_pos_weight to each layer')
            rel_pos_table_weight = state_dict[prefix + 'rel_pos_table_list.0.weight']
            for i in range(len(self.rel_pos_table_list)):
                state_dict[prefix + 'rel_pos_table_list.{}.weight'.format(i)] = rel_pos_table_weight.clone()

        if prefix + 'pos_embed' in state_dict and \
                self.bucket_size ** 2 + 1 > state_dict[prefix + 'pos_embed'].size(0):
            logger.info('interpolate absolute position embedding')
            cls_pos_emebd = state_dict[prefix + 'pos_embed'][:1]
            old_pos_embed = state_dict[prefix + 'pos_embed'][1:]
            orig_bucket_size = int(old_pos_embed.size(0) ** 0.5)
            old_pos_embed = old_pos_embed.reshape(1, orig_bucket_size, orig_bucket_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(old_pos_embed, size=(self.bucket_size, self.bucket_size), mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(self.bucket_size ** 2, -1)
            pos_embed = torch.cat([cls_pos_emebd, pos_embed], dim=0)
            state_dict[prefix + 'pos_embed'] = pos_embed
            state_dict[prefix + 'position_idx'] = self.state_dict()['position_idx']

        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]

        return state_dict
