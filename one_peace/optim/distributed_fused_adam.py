# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)


def get_distributed_fused_adam_class():
    """ . """
    try:
        from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam as DistributedFusedAdam_
        return DistributedFusedAdam
    except ImportError:
        pass
    return None

try:
    from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam as DistributedFusedAdam_

    class DistributedFusedAdam(DistributedFusedAdam_):
        """ . """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @property
        def supports_memory_efficient_fp16(self):
            return True

        @property
        def supports_flat_params(self):
            return True

except ImportError:
    pass
