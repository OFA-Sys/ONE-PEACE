import logging

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

logger = logging.getLogger(__name__)

fast_layernorm_valid_sizes = set(
    [768, 1024, 1536, 2048, 2304, 3072, 3840, 4096, 5120, 6144, 8192, 10240, 12288]
)

try:
    from flash_attn.ops.layer_norm import layer_norm
    has_flash = True
    logger.info('****** use FlashNorm ******')
except Exception as e:
    has_flash = False

try:
    from apex.contrib.layer_norm import FastLayerNorm
    from apex.normalization import FusedLayerNorm
    has_fast_layernorm = True
    has_fused_layernorm = True
    logger.info('****** use FastLayerNorm or FusedLayerNorm ******')
except ImportError:
    try:
        from apex.normalization import FusedLayerNorm
        has_fast_layernorm = False
        has_fused_layernorm = True
        logger.info('****** use FusedLayerNorm ******')
    except ImportError:
        has_fast_layernorm = False
        has_fused_layernorm = False
        logger.info('****** use torch LayerNorm ******')


def trunc_normal_(tensor, mean=0., std=0.02):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if has_flash:
        return FlashNorm(normalized_shape, eps, elementwise_affine)
    if torch.cuda.is_available() and has_fast_layernorm and normalized_shape in fast_layernorm_valid_sizes:
        return FastLayerNorm(normalized_shape, eps)
    if torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False, init_values=None, requires_grad=True):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    if init_values is not None:
        m.weight.data.copy_(init_values)
    if not requires_grad:
        m.weight.requires_grad = False
    return m


class FlashNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps, elementwise_affine):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)