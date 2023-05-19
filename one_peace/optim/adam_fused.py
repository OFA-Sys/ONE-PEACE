# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

logger = logging.getLogger(__name__)


def get_fused_adam_class():
    """
    Look for the FusedAdam optimizer from apex. We first try to load the
    "contrib" interface, which is a bit faster than the main interface,
    but is technically deprecated.
    """
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
        from apex.optimizers import FusedAdam as FusedAdam_
        if multi_tensor_applier.available:
            return FusedAdam
    except ImportError:
        pass
    return None

try:
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.optimizers import FusedAdam as FusedAdam_

    class FusedAdam(FusedAdam_):
        """
        Compared to the original version in Apex, the fairseq version casts grads
        and params to FP32 internally to support ``--memory-efficient-fp16``.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not hasattr(self, "multi_tensor_adam"):
                raise Exception(
                    "Apex installation is outdated. Please install an updated version of apex."
                )

            self.p_data_fp32 = []
            for group in self.param_groups:
                group_params = []
                for p in group["params"]:
                    group_params.append(p.data.float())
                self.p_data_fp32.append(group_params)

        @property
        def supports_memory_efficient_fp16(self):
            return True

        @property
        def supports_flat_params(self):
            return True

        def step(
            self,
            closure=None,
            grads=None,
            output_params=None,
            scale=None,
            grad_norms=None,
        ):
            """Performs a single optimization step."""
            loss = None
            if closure is not None:
                loss = closure()

            for i, group in enumerate(self.param_groups):
                bias_correction = 1 if group["bias_correction"] else 0
                beta1, beta2 = group["betas"]

                # assume same step across group now to simplify things
                # per parameter step can be easily support by making it tensor, or pass list into kernel
                if "step" in group:
                    group["step"] += 1
                else:
                    group["step"] = 1

                # create lists for multi-tensor apply
                g_16, orig_p_16, m_16, v_16 = [], [], [], []

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "FusedAdam does not support sparse gradients, "
                            "please consider SparseAdam instead"
                        )

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p.data, dtype=torch.float
                        )
                    else:
                        state["exp_avg"] = state["exp_avg"].to(
                            device=p.data.device, dtype=torch.float
                        )
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(
                            device=p.data.device, dtype=torch.float
                        )

                    g_16.append(p.grad.data.float())
                    orig_p_16.append(p.data)
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])

                with torch.cuda.device(p.device):
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, self.p_data_fp32[i], m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )
                    for orig_p, p in zip(orig_p_16, self.p_data_fp32[i]):
                        orig_p.copy_(p.data)

            return loss

except ImportError:
    pass
