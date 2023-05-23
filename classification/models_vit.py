# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Optional
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn, Tensor
from timm.models.layers import trunc_normal_


def make_image_bucket_position(bucket_size: int, num_relative_distance: int):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
    coords_flatten = torch.flatten(coords, 1)  # 2, h*w
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(
        size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # h*w, h*w
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index  # h*w+1, h*w+1


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
        dropout: float = 0.0,
        embed_dim: int = 1536,
        shared_rp_bias: bool = True,
    ):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
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

        self.bucket_size = bucket_size
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
        values = values.permute(2, 0, 1).contiguous()
        return values

    def get_embed_positions(self, src_images):
        pos_embed = self.pos_embed
        window_size = src_images.size(2) // 16
        if window_size ** 2 > pos_embed.size(0):
            cls_pos_embed = pos_embed[:1]
            old_pos_embed = pos_embed[1:]
            old_pos_embed = old_pos_embed.reshape(
                1, window_size, window_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(old_pos_embed, size=(
                self.bucket_size, self.bucket_size), mode="bicubic")
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(self.bucket_size ** 2, -1)
            pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=0)
        return pos_embed

    def forward(self, src_images: Tensor):
        batch_size = src_images.size(0)
        pos_embed = self.get_embed_positions(src_images)

        x = self.embed_images(src_images).flatten(2).transpose(1, 2)
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_embedding, x], dim=1)

        x += pos_embed.unsqueeze(0)
        x = self.dropout_module(x)

        self_attn_bias = self.get_rel_pos_bias() if self.shared_rp_bias else None

        return x, self_attn_bias


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

        q = self.q_proj(query).view(tgt_len, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        k = self.k_proj(query).view(tgt_len, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        v = self.v_proj(query).view(tgt_len, bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        q *= self.scaling

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights += attn_bias

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
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
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        embed_dim: int = 1536,
        ffn_embed_dim: int = 6144,
        layer_scale_init_value: float = 1e-2,
        rp_bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.rp_bias = rp_bias
        self.heads = attention_heads
        self.bucket_size = bucket_size
        self.self_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(float(activation_dropout))

        self.image_ffn = self.build_ffn()

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))

        if rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(bucket_size, num_rel_dis)
            self.rel_pos_table = Embedding(num_rel_dis, attention_heads, zero_init=True)
            self.register_buffer("rp_bucket", rp_bucket)

    def build_ffn(self):
        return nn.Sequential(
            GeGLU(self.embed_dim, self.ffn_embed_dim),
            self.activation_dropout_module,
            nn.LayerNorm(self.ffn_embed_dim),
            nn.Linear(self.ffn_embed_dim, self.embed_dim)
        )

    def get_rel_pos_bias(self):
        values = F.embedding(self.rp_bucket, self.rel_pos_table.weight)
        values = values.permute(2, 0, 1).contiguous()
        return values

    def residual_connection(self, x, residual, gamma=None):
        if gamma is not None:
            return residual + self.drop_path(gamma * x)
        else:
            return residual + self.drop_path(x)

    def forward(self, x: Tensor, attn_bias: Optional[Tensor] = None):
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.rp_bias:
            _, B = x.shape[:2]
            attn_bias = self.get_rel_pos_bias().unsqueeze(0).expand(B, -1, -1, -1).flatten(0, 1)
        x = self.self_attn(x, attn_bias)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual, gamma=self.gamma_1)

        residual = x
        x = self.dropout_module(self.image_ffn(self.final_layer_norm(x)))
        x = self.residual_connection(x, residual, gamma=self.gamma_2)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
      self,
      activation_dropout: float = 0.0,
      attention_dropout: float = 0.0,
      attention_heads: int = 12,
      bucket_size: int = 16,
      dropout: float = 0.0,
      drop_path_rate: float = 0.0,
      embed_dim: int = 1536,
      ffn_embed_dim: int = 6144,
      global_pool: bool = True,
      layers: int = 40,
      layer_scale_init_value: float = 1e-2,
      rp_bias: bool = False,
      use_checkpoint: bool = False,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.num_attention_heads = attention_heads
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        for i in range(layers):
            self.layers.append(
                TransformerEncoderLayer(
                    activation_dropout=activation_dropout,
                    attention_dropout=attention_dropout,
                    attention_heads=attention_heads,
                    bucket_size=bucket_size,
                    dropout=dropout,
                    drop_path_rate=dpr[i],
                    embed_dim=embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    rp_bias=rp_bias,
                )
            )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(embed_dim) if not self.global_pool else nn.Identity()

    def forward(self, image_info):
        x, attn_bias = image_info

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(0).expand(x.size(0), -1, -1, -1).flatten(0, 1)

        # BxLxC -> LxBxC
        x = x.transpose(0, 1)

        for layer in self.layers:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(layer, x, attn_bias)
            else:
                x = layer(x, attn_bias)

        x = self.layer_norm(x)

        return x


class OnePieceViT(nn.Module):
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
      global_pool: bool = True,
      init_scale: float = 0.001,
      layers: int = 40,
      layer_scale_init_value: float = 1e-2,
      num_classes: int = 1000,
      rp_bias: bool = False,
      shared_rp_bias: bool = True,
      use_checkpoint: bool = False,
    ):
        super().__init__()
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            dropout=dropout,
            embed_dim=embed_dim,
            shared_rp_bias=shared_rp_bias,
        )
        self.encoder = TransformerEncoder(
            activation_dropout=activation_dropout,
            attention_dropout=attention_dropout,
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            global_pool=global_pool,
            layers=layers,
            layer_scale_init_value=layer_scale_init_value,
            rp_bias=rp_bias,
            use_checkpoint=use_checkpoint,
        )

        self.global_pool = global_pool
        self.fc_norm = nn.LayerNorm(embed_dim) if global_pool else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'image_adapter.pos_embed',
            'image_adapter.cls_embedding',
        }

    def get_num_layers(self):
        return len(self.encoder.layers)

    def forward_features(self, src_images: Tensor):
        image_info = self.image_adapter(src_images)
        encoder_out = self.encoder(image_info)
        return encoder_out

    def forward_head(self, x):
        x = x[:, 1:, :].mean(dim=1) if self.global_pool else x[:, 0]
        x = self.fc_norm(x)
        return self.head(x)

    def forward(self, src_images: Tensor):
        x = self.forward_features(src_images)
        x = x.transpose(0, 1)
        return self.forward_head(x)


def one_piece_g_256(**kwargs):
    model = OnePieceViT(bucket_size=16, rp_bias=False, shared_rp_bias=True, **kwargs)
    return model


def one_piece_g_384(**kwargs):
    model = OnePieceViT(bucket_size=24, rp_bias=False, shared_rp_bias=True, **kwargs)
    return model


def one_piece_g_448(**kwargs):
    model = OnePieceViT(bucket_size=28, rp_bias=False, shared_rp_bias=True, **kwargs)
    return model


def one_piece_g_512(**kwargs):
    model = OnePieceViT(bucket_size=32, rp_bias=False, shared_rp_bias=True, **kwargs)
    return model
