# Copyright 2023 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Dict, Optional
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from scipy import interpolate

from mmcv.cnn.utils.weight_init import constant_init, kaiming_init, trunc_normal_
from mmcv.runner import BaseModule, _load_checkpoint, get_dist_info
from mmcv.cnn import build_norm_layer

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES


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
    relative_position_index = torch.zeros(
      size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
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
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


class LayerNorm2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GeGLU(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.wi_0 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.wi_1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x_gelu = self.act(self.wi_0(x))
        x_linear = self.wi_1(x)
        x = x_gelu * x_linear
        return x


class ImageAdaptor(nn.Module):
    def __init__(
        self,
        attention_heads: int = 24,
        bucket_size: int = 16,
        embed_dim: int = 1536,
        dropout: float = 0.0,
        shared_rp_bias: bool = True,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.dropout = nn.Dropout(dropout)

        self.embed_images = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=4, stride=4),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
        )

        scale = embed_dim ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(scale * torch.randn(bucket_size ** 2 + 1, embed_dim))

        self.shared_rp_bias = shared_rp_bias
        if shared_rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(bucket_size, num_rel_dis)
            self.rel_pos_table = Embedding(num_rel_dis, attention_heads, zero_init=True)
            self.register_buffer("rp_bucket", rp_bucket)

    def get_rel_pos_bias(self):
        rp_bucket = self.rp_bucket
        values = F.embedding(rp_bucket, self.rel_pos_table.weight)
        values = values.permute(2, 0, 1)
        return values

    def get_embed_positions(self, src_images):
        pos_embed = self.pos_embed
        window_size = src_images.size(2) // 16
        if window_size ** 2 > pos_embed.size(0):
            cls_pos_embed = pos_embed[:1]
            old_pos_embed = pos_embed[1:]
            old_pos_embed = old_pos_embed.reshape(
                1, self.bucket_size, self.bucket_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(old_pos_embed, size=(
                window_size, window_size), mode="bicubic")
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(self.bucket_size ** 2, -1)
            pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=0)
        return pos_embed

    def forward(self, src_images: Tensor):
        """
        Args:
            src_images (Tensor): BxCxHxW
        """
        batch_size = src_images.size(0)
        pos_embed = self.get_embed_positions(src_images)

        x = self.embed_images(src_images)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_embedding, x], dim=1)

        x += pos_embed.unsqueeze(0)
        x = self.dropout(x)

        self_attn_bias = self.get_rel_pos_bias() if self.shared_rp_bias else None

        return x, self_attn_bias, H, W


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
          self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.ln = nn.LayerNorm(embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        """
        query: LxBxC
        """
        tgt_len, bsz, _ = query.size()

        q = self.q_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q *= self.scaling

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights += attn_bias

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().reshape(tgt_len, bsz, self.embed_dim)
        attn = self.ln(attn)
        attn = self.out_proj(attn)

        return attn


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 24,
        bucket_size: int = 16,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_dim: Optional[int] = 1536,
        ffn_embed_dim: int = 6144,
        layer_scale_init_value: float = 1e-2,
        norm_cfg: Dict = dict(type='LN'),
        use_checkpoint: bool = False,
        rp_bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.use_checkpoint = use_checkpoint
        self.rp_bias = rp_bias
        self.self_attn = MultiheadAttention(
          embed_dim,
          attention_heads,
          dropout=attention_dropout,
        )
        self.self_attn_layer_norm = build_norm_layer(norm_cfg, self.embed_dim)[1]
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.activation_dropout = nn.Dropout(float(activation_dropout))

        self.final_layer_norm = build_norm_layer(norm_cfg, self.embed_dim)[1]

        self.image_ffn = self.build_ffn(norm_cfg)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))

        if rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(bucket_size, num_rel_dis)
            self.rel_pos_table = Embedding(num_rel_dis, attention_heads, zero_init=True)
            self.register_buffer("rp_bucket", rp_bucket)

    def build_ffn(self, norm_cfg):
        return nn.Sequential(
            GeGLU(self.embed_dim, self.ffn_embed_dim),
            self.activation_dropout,
            build_norm_layer(norm_cfg, self.ffn_embed_dim)[1],
            nn.Linear(self.ffn_embed_dim, self.embed_dim)
        )

    def get_rel_pos_bias(self):
        values = F.embedding(self.rp_bucket, self.rel_pos_table.weight)
        values = values.permute(2, 0, 1).contiguous()
        return values

    def forward(self, x, H=None, W=None, attn_bias: Optional[Tensor] = None):
        def _inner_forward(x, attn_bias=None):
            if self.rp_bias:
                L, B = x.shape[:2]
                attn_bias = self.get_rel_pos_bias().unsqueeze(0).expand(B, -1, -1, -1).flatten(0, 1)
            x = x + self.drop_path(self.gamma_1 *
                                   self.dropout(self.self_attn(self.self_attn_layer_norm(x), attn_bias)))
            x = x + self.drop_path(self.gamma_2 *
                                   self.dropout(self.image_ffn(self.final_layer_norm(x))))
            return x

        if self.use_checkpoint and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, attn_bias)
        else:
            x = _inner_forward(x, attn_bias)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 24,
        bucket_size: int = 16,
        dropout: float = 0.0,
        embed_dim: Optional[int] = 1536,
        drop_path_rate: float = 0.0,
        ffn_embed_dim: int = 6144,
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        norm_cfg: Dict = dict(type='LN'),
        use_checkpoint: bool = False,
        rp_bias: bool = False,
    ):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.attention_heads = attention_heads

        self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        for i in range(layers):
            layer = TransformerEncoderLayer(
              activation_dropout=activation_dropout,
              attention_dropout=attention_dropout,
              attention_heads=attention_heads,
              bucket_size=bucket_size,
              dropout=dropout,
              drop_path_rate=dpr[i],
              embed_dim=embed_dim,
              ffn_embed_dim=ffn_embed_dim,
              layer_scale_init_value=layer_scale_init_value,
              norm_cfg=norm_cfg,
              use_checkpoint=use_checkpoint,
              rp_bias=rp_bias,
            )
            self.layers.append(layer)
        self.num_layers = len(self.layers)

    def forward(self, image_info):
        x, attn_bias = image_info

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(0).expand(x.size(0), -1, -1, -1).flatten(0, 1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1).contiguous()

        # encoder layers
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)

        return x


@BACKBONES.register_module(name="OnePeace")
class OnePeace(BaseModule):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 24,
        bucket_size: int = 16,
        dropout: float = 0.0,
        embed_dim: Optional[int] = 1536,
        drop_path_rate: float = 0.0,
        ffn_embed_dim: int = 6144,
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        norm_cfg: Dict = dict(type='LN'),
        pretrained: Optional[str] = None,
        use_checkpoint: bool = False,
        rp_bias: bool = True,
        shared_rp_bias: bool = False,
    ):
        super().__init__()
        self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            embed_dim=embed_dim,
            dropout=dropout,
            shared_rp_bias=shared_rp_bias,
        )
        self.encoder = TransformerEncoder(
            activation_dropout=activation_dropout,
            attention_dropout=attention_dropout,
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            dropout=dropout,
            embed_dim=embed_dim,
            drop_path_rate=drop_path_rate,
            ffn_embed_dim=ffn_embed_dim,
            layers=layers,
            layer_scale_init_value=layer_scale_init_value,
            norm_cfg=norm_cfg,
            use_checkpoint=use_checkpoint,
            rp_bias=rp_bias,
        )

        self.rp_bias = rp_bias
        self.shared_rp_bias = shared_rp_bias

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
            return a * (1.0 - r**n) / (1.0 - r)

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
            cur += q**(i + 1)
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

    def resize_abs_pos_embed(self, checkpoint):
        pos_embed_checkpoint = checkpoint['image_adapter.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        bucket_size = self.image_adapter.bucket_size
        num_patches = bucket_size ** 2
        num_extra_tokens = self.image_adapter.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        rank, _ = get_dist_info()
        if orig_size != new_size:
            if rank == 0:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            checkpoint['image_adapter.pos_embed'] = new_pos_embed

    def resize_rel_pos_embed(self, checkpoint):
        """Resize relative pos_embed weights.
        This function is modified from
        https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/checkpoint.py.  # noqa: E501
        Copyright (c) Microsoft Corporation
        Licensed under the MIT License
        Args:
            checkpoint (dict): Key and value of the pretrain model.
        Returns:
            state_dict (dict): Interpolate the relative pos_embed weights
                in the pre-train model to the current model size.
        """
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        rank, _ = get_dist_info()

        if self.rp_bias and "image_adapter.rel_pos_table_list.0.weight" in state_dict:
            if rank == 0:
                print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = self.encoder.num_layers
            rel_pos_bias = state_dict["image_adapter.rel_pos_table_list.0.weight"]
            for i in range(num_layers):
                state_dict["encoder.layers.%d.rel_pos_table.weight" % i] = rel_pos_bias.clone()
            state_dict.pop("image_adapter.rel_pos_table_list.0.weight")

        all_keys = list(state_dict.keys())
        for key in all_keys:
            if 'image_adapter.rp_bucket' in key:
                state_dict.pop(key)
            # In order to keep the center of pos_bias as consistent as
            # possible after interpolation, and vice versa in the edge
            # area, the geometric sequence interpolation method is adopted.
            if 'rel_pos_table.weight' in key:
                rel_pos_bias = state_dict[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.state_dict()[key].size()
                # Count the number of extra tokens.
                num_extra_tokens = 3
                src_size = int((src_num_pos - num_extra_tokens)**0.5)
                dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                    new_rel_pos_bias = self._geometric_sequence_interpolation(
                        src_size, dst_size, rel_pos_bias, num_attn_heads)
                    new_rel_pos_bias = torch.cat(
                        (new_rel_pos_bias, extra_tokens), dim=0)
                    state_dict[key] = new_rel_pos_bias

        return state_dict

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            if self.init_cfg['checkpoint'].endswith(".pkl"):
                with open(self.init_cfg['checkpoint'], "rb") as f:
                    checkpoint = pickle.load(f, encoding="latin1")
            else:
                checkpoint = _load_checkpoint(
                    self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            model = checkpoint['model']
            state_dict = self.resize_abs_pos_embed(model)
            state_dict = self.resize_rel_pos_embed(model)
            msg = self.load_state_dict(state_dict, False)
            rank, _ = get_dist_info()
            if rank == 0:
                print(msg)
        elif self.init_cfg is not None:
            super(OnePeace, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            # Copyright 2019 Ross Wightman
            # Licensed under the Apache License, Version 2.0 (the "License")
            trunc_normal_(self.image_adapter.cls_embedding, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'image_ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward(self, src_images: Optional[Tensor] = None):
        image_info = self.image_adapter(src_images)
        x = self.encoder(image_info)
        return x
