import logging

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

logger = logging.getLogger(__name__)

try:
    from flash_attn.ops.layer_norm import layer_norm
    has_flash = True
    logger.info('****** use FlashNorm ******')
except Exception as e:
    has_flash = False
    logger.info('****** Import flash_attn.ops.layer_norm fail, please install flash_attn ******')


def trunc_normal_(tensor, mean=0., std=0.02):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if has_flash:
        return FlashNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


class FlashNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps, elementwise_affine):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)