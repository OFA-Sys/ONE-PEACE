import re
import json
import logging

logger = logging.getLogger(__name__)


def get_num_layer(var_name, num_max_layer):
    if var_name.startswith("text_adapter") or var_name.startswith("image_adapter") or var_name.startswith("audio_adapter"):
        var_name = re.sub('^(text_adapter|image_adapter|audio_adapter).', '', var_name)
        if var_name.startswith("rel_pos_table"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return 0
    elif var_name.startswith("fusion_model.layers"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer(var_name, len(self.values))


def get_parameter_groups(
    model,
    weight_decay=1e-5,
    skip_list=(),
    get_num_layer=None,
    get_layer_scale=None
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        name = re.sub('^module.module.', '', name)
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            var_name = re.sub('^encoder_wrapper.', '', name)
            layer_id = get_num_layer(var_name)
            scale = get_layer_scale(layer_id)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            scale = 1

        if group_name not in parameter_group_vars:
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    logger.info("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
