# Copyright 2023 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Optional
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn, Tensor
from scipy import interpolate
from timm.models.layers import trunc_normal_
from mmcv.runner import get_dist_info
from mmaction.utils import get_root_logger
from einops import rearrange
from mmaction.models.builder import BACKBONES


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


def make_image_bucket_position(bucket_size, num_relative_distance):
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
        num_frames: int = 32,
        dropout: float = 0.1,
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
        self.num_frames = num_frames
        self.pos_embed = nn.Parameter(scale * torch.randn(bucket_size ** 2 + 1, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

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

    def forward(self, src_images):
        batch_size = src_images.size(0)
        pos_embed = self.get_embed_positions(src_images)

        x = self.embed_images(src_images).flatten(2).transpose(1, 2)  # BxLxC
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_embedding, x], dim=1)

        x += pos_embed.unsqueeze(0)
        x = self.dropout_module(x)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

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

    def forward(self, query, attn_bias: Optional[Tensor] = None) -> Tensor:
        """
        input shape: LxBxC
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
        num_tadapter: int = 1,
        num_frames: int = 8,
        scale: float = 1.,
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

        self.MLP_Adapter = Adapter(embed_dim, skip_connect=False)
        self.S_Adapter = Adapter(embed_dim)
        self.scale = scale
        self.T_Adapter = Adapter(embed_dim, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(embed_dim)
        self.num_tadapter = num_tadapter
        self.num_frames = num_frames

    def build_ffn(self):
        return nn.Sequential(
            GeGLU(self.embed_dim, self.ffn_embed_dim, False),
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

    def forward(self, x, attn_bias: Optional[Tensor] = None):
        n, bt, d = x.shape
        residual = x
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        # temporal adaptation
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.self_attn(self.T_Adapter_in(self.self_attn_layer_norm(xt))))
        else:
            xt = self.T_Adapter(self.self_attn(self.self_attn_layer_norm(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)
        # spatial adaptation
        if self.rp_bias:
            b = x.shape[1]
            attn_bias = self.get_rel_pos_bias().unsqueeze(0).expand(b, -1, -1, -1).flatten(0, 1)
        x = self.S_Adapter(self.self_attn(self.self_attn_layer_norm(x), attn_bias))
        x = self.residual_connection(x, residual, self.gamma_1)
        # joint adaptation
        residual = x
        xn = self.final_layer_norm(x)
        x = residual + self.gamma_2 * \
            self.dropout_module(self.image_ffn(xn)) + \
            self.drop_path(self.scale * self.MLP_Adapter(xn))

        return x


class TransformerEncoder(nn.Module):
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
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        num_tadapter: int = 1,
        num_frames: int = 8,
        scale: float = 0.5,
        rp_bias: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
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
                    num_tadapter=num_tadapter,
                    num_frames=num_frames,
                    scale=scale,
                    rp_bias=rp_bias,
                )
            )
        self.num_layers = len(self.layers)

        self.image_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_info):
        x, attn_bias = image_info

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(0).expand(x.size(0), -1, -1, -1).flatten(0, 1)

        # (BT)xLxC -> Lx(BT)xC
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x, attn_bias)
            else:
                x = layer(x, attn_bias)

        x = self.image_layer_norm(x)

        return x


@BACKBONES.register_module()
class OnePieceViT(nn.Module):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_heads: int = 12,
        adapter_scale: float = 0.5,
        bucket_size: int = 16,
        num_tadapter: int = 1,
        num_frames: int = 32,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        layers: int = 12,
        layer_scale_init_value: float = 1e-2,
        rp_bias: bool = False,
        shared_rp_bias: bool = True,
        use_checkpoint: bool = False,
        pretrained: str = None,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            num_frames=num_frames,
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
            layers=layers,
            layer_scale_init_value=layer_scale_init_value,
            num_tadapter=num_tadapter,
            num_frames=num_frames,
            scale=adapter_scale,
            rp_bias=rp_bias,
            use_checkpoint=use_checkpoint,
        )

        self.rp_bias = rp_bias
        self.shared_rp_bias = shared_rp_bias

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
        if self.shared_rp_bias and "image_adapter.rel_pos_table_list.0.weight" in state_dict:
            rel_pos_bias = state_dict["image_adapter.rel_pos_table_list.0.weight"]
            state_dict["image_adapter.rel_pos_table.weight"] = rel_pos_bias.clone()
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

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            model = pickle.load(open(self.pretrained, "rb"), encoding="latin1")["model"]
            state_dict = self.resize_abs_pos_embed(model)
            state_dict = self.resize_rel_pos_embed(model)
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        # initialize S_Adapter
        for n, m in self.encoder.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        # initialize T_Adapter
        for n, m in self.encoder.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        # initialize MLP_Adapter
        for n, m in self.encoder.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'image_adapter.pos_embed',
            'image_adapter.cls_embedding',
            'image_adapter.temporal_embedding',
        }

    def get_num_layers(self):
        return len(self.encoder.layers)

    def forward_features(self, x: Optional[Tensor] = None):
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        image_info = self.image_adapter(x)
        encoder_out = self.encoder(image_info)
        return encoder_out

    def forward(self, x: Optional[Tensor] = None):
        B, C, T, H, W = x.shape
        x = self.forward_features(x)
        x = x.transpose(0, 1)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t', b=B, t=T)
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head
        return x
