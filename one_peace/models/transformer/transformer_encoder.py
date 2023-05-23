# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from typing import Optional
import logging

import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList
)

from models.components import LayerNorm
from .transformer_layer import TransformerEncoderLayer

logger = logging.getLogger(__name__)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, use_text_norm, use_image_norm, use_audio_norm):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = cfg.layerdrop

        embed_dim = cfg.embed_dim
        self.max_positions = cfg.max_positions
        self.num_attention_heads = cfg.attention_heads

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.layers)]
        self.layers.extend(
            [self.build_encoder_layer(cfg, drop_path_rate=dpr[i]) for i in range(cfg.layers)]
        )
        self.num_layers = len(self.layers)

        self.text_layer_norm = None
        self.image_layer_norm = None
        self.audio_layer_norm = None
        if cfg.use_text_moe and use_text_norm:
            self.text_layer_norm = LayerNorm(embed_dim)
        if cfg.use_image_moe and use_image_norm:
            self.image_layer_norm = LayerNorm(embed_dim)
        if cfg.use_audio_moe and use_audio_norm:
            self.audio_layer_norm = LayerNorm(embed_dim)

    def build_encoder_layer(self, cfg, drop_path_rate=0.0):
        layer = TransformerEncoderLayer(cfg, drop_path_rate=drop_path_rate)
        return layer

    def forward(
        self,
        text_info,
        image_info,
        audio_info,
        return_all_hiddens: bool = False,
        encoder_type: Optional[str] = None
    ):
        """
        Args:
            text_info (tuple): containing the results returned by TextAdapter
            image_info (tuple): containing the results returned by ImageAdapter
            audio_info (tuple): containing the results returned by AudioAdapter
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            encoder_type (str): encoder type

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        text_x, text_padding_mask, text_self_attn_bias_list, text_seq_len = None, None, None, 0
        image_x, image_padding_mask, image_self_attn_bias_list, image_seq_len = None, None, None, 0
        audio_x, audio_padding_mask, audio_self_attn_bias_list, audio_seq_len = None, None, None, 0
        if text_info is not None:
            text_x, text_padding_mask, text_self_attn_bias_list = text_info
            text_seq_len = text_x.size(1)
        if image_info is not None:
            image_x, image_padding_mask, image_self_attn_bias_list = image_info
            image_seq_len = image_x.size(1)
        if audio_info is not None:
            audio_x, audio_padding_mask, audio_self_attn_bias_list = audio_info
            audio_seq_len = audio_x.size(1)

        if encoder_type == 'text':
            x = text_x
            encoder_padding_mask = text_padding_mask
            attn_bias_num = len(text_self_attn_bias_list) if text_self_attn_bias_list is not None else 0
        elif encoder_type == 'image':
            x = image_x
            encoder_padding_mask = image_padding_mask
            attn_bias_num = len(image_self_attn_bias_list) if image_self_attn_bias_list is not None else 0
        elif encoder_type == 'audio':
            x = audio_x
            encoder_padding_mask = audio_padding_mask
            attn_bias_num = len(audio_self_attn_bias_list) if audio_self_attn_bias_list is not None else 0
        elif encoder_type == 'vl':
            x = torch.cat([text_x, image_x], dim=1)
            encoder_padding_mask = torch.cat([text_padding_mask, image_padding_mask], dim=1)
            attn_bias_num = len(text_self_attn_bias_list) if text_self_attn_bias_list is not None else 0
        elif encoder_type == 'al':
            x = torch.cat([text_x, audio_x], dim=1)
            encoder_padding_mask = torch.cat([text_padding_mask, audio_padding_mask], dim=1)
            attn_bias_num = len(text_self_attn_bias_list) if text_self_attn_bias_list is not None else 0
        else:
            raise NotImplementedError

        has_pads = encoder_padding_mask.any()
        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        bsz, seq_len, embed_dim = x.shape
        self_attn_bias_list = []
        attn_bias_pad_mask = encoder_padding_mask.view(
            bsz, 1, 1, seq_len).expand(-1, self.num_attention_heads, seq_len, -1)
        for i in range(attn_bias_num):
            self_attn_bias = x.new_zeros(bsz, self.num_attention_heads, seq_len, seq_len)
            if text_info is not None and text_self_attn_bias_list is not None:
                start_idx, end_idx = 0, text_seq_len
                self_attn_bias[:, :, start_idx:end_idx, start_idx:end_idx] += text_self_attn_bias_list[i]
            if image_info is not None and image_self_attn_bias_list is not None:
                start_idx, end_idx = text_seq_len, text_seq_len + image_seq_len
                self_attn_bias[:, :, start_idx:end_idx, start_idx:end_idx] += image_self_attn_bias_list[i]
            if audio_info is not None and audio_self_attn_bias_list is not None:
                start_idx, end_idx = text_seq_len + image_seq_len, text_seq_len + image_seq_len + audio_seq_len
                self_attn_bias[:, :, start_idx:end_idx, start_idx:end_idx] += audio_self_attn_bias_list[i]
            if has_pads:
                self_attn_bias.masked_fill_(attn_bias_pad_mask, float("-inf"))
            # self_attn_bias = self_attn_bias.reshape(-1, x.size(1), x.size(1))
            self_attn_bias_list.append(self_attn_bias)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        text_encoder_states = []
        image_encoder_states = []
        audio_encoder_states = []

        # encoder layers
        for idx, layer in enumerate(self.layers):
            if len(self_attn_bias_list) == 0:
                self_attn_bias = None
            elif len(self_attn_bias_list) == 1:
                self_attn_bias = self_attn_bias_list[0]
            else:
                self_attn_bias = self_attn_bias_list[idx]

            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_bias=self_attn_bias,
                encoder_type=encoder_type,
                text_seq_len=text_seq_len,
                image_seq_len=image_seq_len,
                audio_seq_len=audio_seq_len
            )

            if return_all_hiddens:
                if text_info is not None:
                    start_idx, end_idx = 0, text_seq_len
                    text_encoder_states.append(x[start_idx:end_idx, :, :])
                if image_info is not None:
                    start_idx, end_idx = text_seq_len, text_seq_len + image_seq_len
                    image_encoder_states.append(x[start_idx:end_idx, :, :])
                if audio_info is not None:
                    start_idx, end_idx = text_seq_len + image_seq_len, text_seq_len + image_seq_len + audio_seq_len
                    audio_encoder_states.append(x[start_idx:end_idx, :, :])

        if encoder_type == 'text':
            x = self.text_layer_norm(x) if self.text_layer_norm is not None else x
        elif encoder_type == 'image':
            x = self.image_layer_norm(x) if self.image_layer_norm is not None else x
        elif encoder_type == 'audio':
            x = self.audio_layer_norm(x) if self.audio_layer_norm is not None else x
        elif encoder_type == 'vl':
            text_x = x[:text_seq_len, :, :]
            image_x = x[-image_seq_len:, :, :]
            text_x = self.text_layer_norm(text_x) if self.text_layer_norm is not None else text_x
            image_x = self.image_layer_norm(image_x) if self.image_layer_norm is not None else image_x
            x = torch.cat([text_x, image_x], dim=0)
        elif encoder_type == 'al':
            text_x = x[:text_seq_len, :, :]
            audio_x = x[-audio_seq_len:, :, :]
            text_x = self.text_layer_norm(text_x) if self.text_layer_norm is not None else text_x
            audio_x = self.audio_layer_norm(audio_x) if self.audio_layer_norm is not None else audio_x
            x = torch.cat([text_x, audio_x], dim=0)
        else:
            raise NotImplementedError

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": encoder_padding_mask,
            "text_encoder_states": text_encoder_states,  # List[T x B x C]
            "image_encoder_states": image_encoder_states,  # List[T x B x C]
            "audio_encoder_states": audio_encoder_states,  # List[T x B x C]
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]
        return state_dict
