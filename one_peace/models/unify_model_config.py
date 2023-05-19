# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models.transformer import EncDecBaseConfig
from fairseq import utils


@dataclass
class TextAdapterConfig(FairseqDataclass):
    bucket_size: int = field(
        default=256,
        metadata={"help": "text bucket size"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    add_type_embedding: bool = field(
        default=False, metadata={"help": "add type embedding"}
    )
    shrink_alpha: float = field(
        default=1.0,
        metadata={"help": ""},
    )
    dropout: float = field(default=0.0, metadata={"help": "dropout probability"})
    use_attn_bias: bool = field(
        default=False,
        metadata={"help": ""},
    )


@dataclass
class ImageAdapterConfig(FairseqDataclass):
    bucket_size: int = field(
        default=16,
        metadata={"help": "image bucket size"},
    )
    rel_bucket_size: int = field(
        default=16,
        metadata={"help": "image relative bucket size"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    add_type_embedding: bool = field(
        default=False, metadata={"help": "add type embedding"}
    )
    vision_encoder_type: ChoiceEnum(["mlp", "hmlp", "none"]) = field(
        default="hmlp",
        metadata={"help": "vision encoder type"},
    )
    shrink_alpha: float = field(
        default=1.0,
        metadata={"help": ""},
    )
    dropout: float = field(default=0.0, metadata={"help": "dropout probability"})
    use_attn_bias: bool = field(
        default=False,
        metadata={"help": ""},
    )


@dataclass
class AudioAdapterConfig(FairseqDataclass):
    feature_embed_dim: int = field(
        default=512,
        metadata={"help": ""},
    )
    feature_encoder_spec: Optional[str] = field(
        default='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]',
        metadata={"help": ""},
    )

    abs_pos_type: str = field(
        default='conv',
        metadata={"help": ""},
    )
    conv_pos_depth: int = field(
        default=5,
        metadata={"help": ""},
    )
    conv_pos_width: int = field(
        default=95,
        metadata={"help": ""},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": ""},
    )
    conv_pos_pre_ln: bool = field(
        default=False,
        metadata={"help": ""},
    )

    bucket_size: int = field(
        default=256,
        metadata={"help": "audio bucket size"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    add_type_embedding: bool = field(
        default=False, metadata={"help": "add type embedding"}
    )
    shrink_alpha: float = field(
        default=1.0,
        metadata={"help": ""},
    )
    dropout: float = field(default=0.0, metadata={"help": "dropout probability"})
    use_attn_bias: bool = field(
        default=False,
        metadata={"help": ""},
    )

    conv_bias: bool = False
    freeze_extractor: bool = False


@dataclass
class AdjustEncDecConfig(EncDecBaseConfig):
    text_adapter: TextAdapterConfig = TextAdapterConfig()
    image_adapter: ImageAdapterConfig = ImageAdapterConfig()
    audio_adapter: AudioAdapterConfig = AudioAdapterConfig()

    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "drop path rate"},
    )

    magneto_scale_attn: bool = field(
        default=False,
        metadata={"help": "magneto scale attn"},
    )
    scale_attn: bool = field(
        default=True,
        metadata={"help": "scale attn"},
    )
    scale_fc: bool = field(
        default=True,
        metadata={"help": "scale fc"},
    )
    scale_heads: bool = field(
        default=True,
        metadata={"help": "scale heads"},
    )

    use_text_moe: bool = field(
        default=True,
        metadata={"help": "use text moe"},
    )
    use_image_moe: bool = field(
        default=True,
        metadata={"help": "use image moe"},
    )
    use_audio_moe: bool = field(
        default=True,
        metadata={"help": "use image moe"},
    )

    use_layer_scale: bool = field(
        default=True,
        metadata={"help": "use layer scale"},
    )
    layer_scale_init_value: float = field(
        default=1e-2,
        metadata={"help": "layer scale init value"},
    )

    use_geglu: bool = field(
        default=True,
        metadata={"help": ""},
    )
    share_ln: bool = field(
        default=False,
        metadata={"help": ""},
    )

    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu",
        metadata={"help": "activation function to use"},
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN.",
            "alias": "--relu-dropout",
        },
    )
    max_positions: int = field(
        default=1024,
        metadata={"help": "Maximum length"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, which saves GPU memory usage at the cost of some additional compute"
        },
    )
    fsdp_checkpoint_wrap_layer_preserve_frequency: Optional[int] = field(
        default=1,
        metadata={"help": ""},
    )
    fsdp_checkpoint_wrap_layer_skip_frequency: Optional[int] = field(
        default=1000,
        metadata={"help": ""},
    )
    offload_activations: bool = field(
        default=False,
        metadata={
            "help": "checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations."
        },
    )


@dataclass
class UnifyModelConfig(FairseqDataclass):
    encoder: AdjustEncDecConfig = AdjustEncDecConfig()
    decoder: AdjustEncDecConfig = AdjustEncDecConfig()