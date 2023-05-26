# Copyright 2023 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Optional, Tuple
import math
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from scipy import interpolate

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_

from detectron2.modeling import Backbone
from detectron2.modeling.backbone.utils import window_partition, window_unpartition, add_decomposed_rel_pos
from detectron2.utils.comm import get_rank


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
      bucket_size: int = 64,
      embed_dim: int = 1536,
      dropout: float = 0.1,
      pretrain_bucket_size: int = 16,
      shared_rp_bias: bool = True,
      window_size: int = 0,
    ):
        super().__init__()
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
        self.pretrain_bucket_size = pretrain_bucket_size
        self.bucket_size = bucket_size
        self.pos_embed = nn.Parameter(scale * torch.randn(bucket_size ** 2 + 1, embed_dim))

        self.window_size = window_size

        self.shared_rp_bias = shared_rp_bias
        if shared_rp_bias:
            num_rel_dis_pretrain = (2 * pretrain_bucket_size - 1) * \
                (2 * pretrain_bucket_size - 1) + 3
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(bucket_size, num_rel_dis)[1:, 1:].contiguous()
            num_rel_dis_window = (2 * window_size - 1) * (2 * window_size - 1) + 3
            rp_bucket_window = make_image_bucket_position(
                window_size, num_rel_dis_window)[1:, 1:].contiguous()
            self.rel_pos_table = Embedding(num_rel_dis_pretrain, attention_heads, zero_init=True)

            self.register_buffer("rp_bucket", rp_bucket)
            self.register_buffer("rp_bucket_window", rp_bucket_window)

    def get_rel_pos_bias(self, window=False):
        num_extra_tokens = 3
        src_size = 2 * self.pretrain_bucket_size - 1
        dst_size = 2 * self.window_size - 1 if window else 2 * self.bucket_size - 1
        rp_bucket = self.rp_bucket_window if window else self.rp_bucket

        if src_size != dst_size:
            extra_tokens = self.rel_pos_table.weight[-num_extra_tokens:, :]
            rel_pos_bias = self.rel_pos_table.weight[:-num_extra_tokens,
                                                     :].view(1, src_size, src_size, -1).permute(0, 3, 1, 2)
            new_rel_pos_bias = F.interpolate(
                rel_pos_bias,
                size=(dst_size, dst_size),
                mode="bicubic",
            )
            new_rel_pos_bias = torch.cat((new_rel_pos_bias.permute(
                0, 2, 3, 1).reshape(dst_size ** 2, -1), extra_tokens), dim=0)
            values = F.embedding(rp_bucket, new_rel_pos_bias)
        else:
            values = F.embedding(rp_bucket, self.rel_pos_table.weight)
        values = values.permute(2, 0, 1).contiguous()
        return values

    def forward(self, x):
        x = self.embed_images(x).permute(0, 2, 3, 1)
        h, w = x.shape[1:3]

        pos_embed = self.pos_embed[1:]
        x = x + pos_embed.unsqueeze(0).reshape(1, h, w, -1)
        x = self.dropout(x)

        self_attn_bias = self.get_rel_pos_bias() if self.shared_rp_bias else None
        self_attn_bias_window = self.get_rel_pos_bias(True) if self.shared_rp_bias else None

        return x, self_attn_bias, self_attn_bias_window


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_decomposed_rel_pos: bool = False,
        input_size: Tuple[int] = None,
    ):
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

        self.use_decomposed_rel_pos = use_decomposed_rel_pos
        if self.use_decomposed_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, query, attn_bias: Optional[Tensor] = None) -> Tensor:
        B, H, W, _ = query.size()

        q = self.q_proj(query).reshape(B, H * W, self.num_heads, -1).permute(0,
                                                                             2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        k = self.k_proj(query).reshape(B, H * W, self.num_heads, -1).permute(0,
                                                                             2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        v = self.v_proj(query).reshape(B, H * W, self.num_heads, -1).permute(0,
                                                                             2, 1, 3).reshape(B * self.num_heads, H * W, -1)

        attn = (q * self.scaling) @ k.transpose(-2, -1)
        attn += attn_bias
        if self.use_decomposed_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1).type_as(query)

        x = (attn @ v).view(B, self.num_heads, H * W, -1)

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.ln(x)
        x = self.out_proj(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 24,
        bucket_size: int = 64,
        pretrain_bucket_size: int = 16,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_dim: Optional[int] = 1536,
        ffn_embed_dim: int = 6144,
        layer_scale_init_value: float = 1e-2,
        rp_bias: bool = False,
        use_decomposed_rel_pos: bool = False,
        window_size: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.rp_bias = rp_bias
        self.self_attn = MultiheadAttention(
          embed_dim,
          attention_heads,
          dropout=attention_dropout,
          use_decomposed_rel_pos=use_decomposed_rel_pos,
          input_size=(bucket_size, bucket_size) if window_size == 0 else (window_size, window_size),
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(float(activation_dropout))

        self.image_ffn = self.build_ffn()

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))

        self.window_size = window_size
        self.bucket_size = window_size if window_size > 0 else bucket_size
        self.pretrain_bucket_size = window_size if window_size > 0 else pretrain_bucket_size

        if rp_bias:
            pretrain_bucket_size = window_size if window_size > 0 else pretrain_bucket_size
            num_rel_dis_pretrain = (2 * pretrain_bucket_size - 1) * \
                (2 * pretrain_bucket_size - 1) + 3
            bucket_size = window_size if window_size > 0 else bucket_size
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(bucket_size, num_rel_dis)[1:, 1:].contiguous()
            self.register_buffer("rp_bucket", rp_bucket)
            self.rel_pos_table = Embedding(num_rel_dis_pretrain, attention_heads, zero_init=True)

    def build_ffn(self):
        return nn.Sequential(
            GeGLU(self.embed_dim, self.ffn_embed_dim),
            self.activation_dropout,
            nn.LayerNorm(self.ffn_embed_dim),
            nn.Linear(self.ffn_embed_dim, self.embed_dim)
        )

    def get_rel_pos_bias(self):
        src_size = 2 * self.pretrain_bucket_size - 1
        dst_size = 2 * self.bucket_size - 1
        rp_bucket = self.rp_bucket

        if src_size != dst_size:
            num_extra_tokens = 3
            extra_tokens = self.rel_pos_table.weight[-num_extra_tokens:, :]
            rel_pos_bias = self.rel_pos_table.weight[:-num_extra_tokens,
                                                     :].view(1, src_size, src_size, -1).permute(0, 3, 1, 2)
            new_rel_pos_bias = F.interpolate(
                rel_pos_bias,
                size=(dst_size, dst_size),
                mode="bicubic",
            )
            new_rel_pos_bias = torch.cat((new_rel_pos_bias.permute(
                0, 2, 3, 1).reshape(dst_size ** 2, -1), extra_tokens), dim=0)
            values = F.embedding(rp_bucket, new_rel_pos_bias)
        else:
            values = F.embedding(rp_bucket, self.rel_pos_table.weight)
        values = values.permute(2, 0, 1)
        return values

    def forward(self, x, attn_bias: Optional[Tensor] = None, attn_bias_window: Optional[Tensor] = None):
        shortcut = x
        x = self.self_attn_layer_norm(x)
        # Window partition
        B, H, W, _ = x.shape
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
            attn_bias = attn_bias_window

        if self.rp_bias:
            if self.window_size > 0:
                Ws = self.window_size
                numsH = H // Ws
                numsW = W // Ws
                S = B * numsH * numsW
            else:
                S = B
            attn_bias = self.get_rel_pos_bias().unsqueeze(0).expand(S, -1, -1, -1).flatten(0, 1)
        x = self.self_attn(x, attn_bias)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(self.gamma_1 * self.dropout(x))
        x = x + self.drop_path(self.gamma_2 *
                               self.dropout(self.image_ffn(self.final_layer_norm(x))))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 24,
        bucket_size: int = 64,
        pretrain_bucket_size: int = 16,
        dropout: float = 0.0,
        embed_dim: Optional[int] = 1536,
        drop_path_rate: float = 0.0,
        ffn_embed_dim: int = 6144,
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        rp_bias: bool = False,
        use_decomposed_rel_pos: bool = False,
        use_checkpoint: bool = False,
        window_size: int = 0,
        window_block_indexes: Tuple = (),
    ):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.attention_heads = attention_heads
        self.window_size = window_size

        self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        for i in range(layers):
            layer = TransformerEncoderLayer(
                activation_dropout=activation_dropout,
                attention_dropout=attention_dropout,
                attention_heads=attention_heads,
                bucket_size=bucket_size,
                pretrain_bucket_size=pretrain_bucket_size,
                dropout=dropout,
                drop_path_rate=dpr[i],
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                layer_scale_init_value=layer_scale_init_value,
                rp_bias=rp_bias,
                use_decomposed_rel_pos=use_decomposed_rel_pos,
                window_size=window_size if i in window_block_indexes else 0
            )
            if use_checkpoint:
                layer = checkpoint_wrapper(layer)
            self.layers.append(layer)
        self.num_layers = len(self.layers)

    def forward(self, image_info):
        x, image_attn_bias, image_attn_bias_window = image_info

        if image_attn_bias is not None and image_attn_bias_window is not None:
            L = x.shape[1] * x.shape[2]
            W = self.window_size
            LW = W * W

            abs_pos_bias = x.new_zeros(self.attention_heads, L, L)
            abs_pos_bias_window = x.new_zeros(self.attention_heads, LW, LW)

            abs_pos_bias += image_attn_bias
            attn_bias = abs_pos_bias.unsqueeze(0).expand(
              x.shape[0], -1, -1, -1).reshape(-1, L, L)  # BxH L L

            abs_pos_bias_window += image_attn_bias_window
            num_window = math.ceil(x.shape[1] / W)
            attn_bias_window = abs_pos_bias_window.unsqueeze(0).expand(
              x.shape[0] * num_window * num_window, -1, -1, -1).reshape(-1, LW, LW)  # BxH LW LW
        else:
            attn_bias = attn_bias_window = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, attn_bias_window=attn_bias_window)

        return x


class OnePeace(Backbone):
    def __init__(
      self,
      activation_dropout: float = 0.0,
      attention_dropout: float = 0.0,
      attention_heads: int = 24,
      bucket_size: int = 64,
      pretrain_bucket_size: int = 16,
      dropout: float = 0.0,
      embed_dim: Optional[int] = 1536,
      drop_path_rate: float = 0.0,
      ffn_embed_dim: int = 6144,
      layers: int = 40,
      layer_scale_init_value: float = 1e-2,
      out_feature: str = "last_feat",
      rp_bias: bool = False,
      use_decomposed_rel_pos: bool = False,
      shared_rp_bias: bool = True,
      use_checkpoint: bool = False,
      window_size: int = 0,
      window_block_indexes: Tuple = (),
      pretrained: str = None,
    ):
        super().__init__()
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            embed_dim=embed_dim,
            dropout=dropout,
            shared_rp_bias=shared_rp_bias,
            window_size=window_size,
        )
        self.encoder = TransformerEncoder(
            activation_dropout=activation_dropout,
            attention_dropout=attention_dropout,
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            pretrain_bucket_size=pretrain_bucket_size,
            dropout=dropout,
            embed_dim=embed_dim,
            drop_path_rate=drop_path_rate,
            ffn_embed_dim=ffn_embed_dim,
            layers=layers,
            layer_scale_init_value=layer_scale_init_value,
            rp_bias=rp_bias,
            use_decomposed_rel_pos=use_decomposed_rel_pos,
            use_checkpoint=use_checkpoint,
            window_size=window_size,
            window_block_indexes=window_block_indexes,
        )

        self.rp_bias = rp_bias
        self.shared_rp_bias = shared_rp_bias

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: 16}
        self._out_features = [out_feature]

        self.apply(self._init_weights)

        if pretrained:
            if pretrained.endswith(".pkl"):
                with open(pretrained, "rb") as f:
                    checkpoint_model = pickle.load(f, encoding="laten1")["model"]
            else:
                checkpoint_model = torch.load(pretrained, map_location="cpu")["model"]
            self.resize_abs_pos_embed(checkpoint_model)
            checkpoint_model.pop("image_adapter.rp_bucket")
            self.resize_rel_pos_embed(checkpoint_model)
            if get_rank() == 0:
                print(self.load_state_dict(checkpoint_model, strict=False))
                print(f"Loading OFA Encoder pretrained weights from {pretrained}.")

    def _geometric_sequence_interpolation(self, src_size, dst_size, sequence,
                                          num):
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
        if orig_size != new_size:
            if get_rank() == 0:
                print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
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

        if self.rp_bias and "image_adapter.rel_pos_table_list.0.weight" in state_dict:
            if get_rank() == 0:
                print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = self.encoder.num_layers
            rel_pos_bias = state_dict["image_adapter.rel_pos_table_list.0.weight"]
            for i in range(num_layers):
                state_dict["encoder.layers.%d.rel_pos_table.weight" % i] = rel_pos_bias.clone()
            state_dict.pop("image_adapter.rel_pos_table_list.0.weight")

        if self.shared_rp_bias and "image_adapter.rel_pos_table_list.0.weight" in state_dict:
            state_dict["image_adapter.rel_pos_table.weight"] = state_dict["image_adapter.rel_pos_table_list.0.weight"]

        all_keys = list(state_dict.keys())
        for key in all_keys:
            if 'image_adapter.rp_bucket' in key:
                state_dict.pop(key)
            # In order to keep the center of pos_bias as consistent as
            # possible after interpolation, and vice versa in the edge
            # area, the geometric sequence interpolation method is adopted.
            if 'rel_pos_table.weight' in key and 'encoder' in key:
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.image_adapter(x)
        x = self.encoder(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2).contiguous()}
        return outputs


def get_onepeace_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=40):
    """
    Calculate lr decay rate for different OnePeace blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of OnePeace blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".image_adapter" in name:
            layer_id = 0
        elif ".layers." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layers."):].split(".")[2]) + 1

        if get_rank() == 0:
            print(f"{name} layer_id: {layer_id}")

    return lr_decay_rate ** (num_layers + 1 - layer_id)
