import json

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info

from mmseg.utils import get_root_logger
from mmseg.core.builder import OPTIMIZER_BUILDERS


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name.startswith('backbone.image_adapter'):
        return 0
    elif name.startswith('decode_head.mask_embed'):
        return 0
    elif name.startswith('decode_head.cls_embed'):
        return 0
    elif name.startswith('decode_head.level_embed'):
        return 0
    elif name.startswith('decode_head.query_embed'):
        return 0
    elif name.startswith('decode_head.query_feat'):
        return 0
    elif name.startswith('backbone.encoder.layers'):
        return int(name.split('.')[3]) + 1
    else:
        return num_layers - 1


@OPTIMIZER_BUILDERS.register_module()
class OnePeaceLearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.
    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = get_root_logger()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build OnePeaceLearningRateDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'image_adapter.pos_embed', 'image_adapter.cls_embedding'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            layer_id = get_layer_id_for_vit(name, num_layers)

            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
