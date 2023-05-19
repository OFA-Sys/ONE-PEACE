
from collections import defaultdict

import torch
from omegaconf import DictConfig

from fairseq import optim

from .dynamic_loss_scaler import DynamicLossScaler
from .base_optimizer import BaseOptimizer


class _FP16OptimizerMixin(object):
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in MRO (method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    @property
    def has_flat_params(self):
        return torch.is_tensor(self.fp32_params) or (
            isinstance(self.fp32_params, dict)
            and all(torch.is_tensor(t) for t in self.fp32_params.values())
        )

    @classmethod
    def build_fp32_params(cls, args, params, flatten=True):
        # create FP32 copy of parameters and grads
        if flatten:
            is_pipeline_parallel = getattr(
                args, "pipeline_model_parallel", False
            ) and getattr(args, "distributed_no_spawn", False)
            total_param_size = sum(p.data.numel() for p in params)
            devices = [torch.cuda.current_device()]
            if is_pipeline_parallel:
                devices = list(set(args.pipeline_devices))
            fp32_params = {}
            for device in devices:
                if is_pipeline_parallel:
                    device_param_size = sum(
                        p.data.numel() for p in params if p.device.index == device
                    )
                    device_params = [p for p in params if p.device.index == device]
                else:
                    device_param_size = total_param_size
                    device_params = params
                fp32_params[device] = (
                    device_params[0].new(0).float().new(device_param_size)
                )
                offset = 0
                for p in device_params:
                    numel = p.data.numel()
                    fp32_params[device][offset : offset + numel].copy_(p.data.view(-1))
                    offset += numel
                fp32_params[device] = torch.nn.Parameter(fp32_params[device])
                fp32_params[device].grad = fp32_params[device].data.new(
                    device_param_size
                )
            return fp32_params
        else:
            fp32_params = []
            for p in params:
                p32 = torch.nn.Parameter(p.data.float())
                if hasattr(p, "expert"):
                    p32.expert = True
                elif hasattr(p, "base_expert"):
                    p32.base_expert = True
                p32.grad = torch.zeros_like(p32.data)
                if hasattr(p, "param_group"):
                    p32.param_group = p.param_group
                fp32_params.append(p32)
            return fp32_params

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.fp32_optimizer.state_dict()
        if self.scaler is not None:
            state_dict["loss_scale"] = self.scaler.loss_scale
            state_dict["scale_window"] = self.scaler.scale_window_interval * (self.scaler.loss_scale / 128)
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if "loss_scale" in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict["loss_scale"]
            self.scaler.scale_window = self.scaler.scale_window_interval * (self.scaler.loss_scale / 128)
        self.fp32_optimizer.load_state_dict(state_dict, optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self):
        if self._needs_sync:
            # copy FP16 grads to FP32
            if self.has_flat_params:
                devices = list(self.fp32_params.keys())
                device_params_dict = defaultdict(list)
                for p in self.fp16_params:
                    if p.requires_grad:
                        device_params_dict[p.device.index].append(p)
                for device in devices:
                    device_params = device_params_dict[device]
                    offset = 0
                    for p in device_params:
                        grad_data = (
                            p.grad.data
                            if p.grad is not None
                            else p.data.new_zeros(p.data.shape)
                        )
                        numel = grad_data.numel()
                        self.fp32_params[device].grad.data[
                            offset : offset + numel
                        ].copy_(grad_data.view(-1))
                        offset += numel
            else:
                for p, p32 in zip(self.fp16_params, self.fp32_params):
                    if not p.requires_grad:
                        continue
                    if p.grad is not None:
                        if p32.grad is None:
                            p32.grad = p.grad.data.float()
                        else:
                            p32.grad.data.copy_(p.grad.data)
                    else:
                        p32.grad = torch.zeros_like(p.data, dtype=torch.float)

            self._needs_sync = False

    def _sync_fp32_params_to_fp16(self):
        # copy FP32 params back into FP16 model
        if self.has_flat_params:
            devices = list(self.fp32_params.keys())
            device_params_dict = defaultdict(list)
            for p in self.fp16_params:
                device_params_dict[p.device.index].append(p)
            for device in devices:
                device_params = device_params_dict[device]
                offset = 0
                for p in device_params:
                    numel = p.data.numel()
                    p.data.copy_(
                        self.fp32_params[device]
                        .data[offset : offset + numel]
                        .view_as(p.data)
                    )
                    offset += numel
        else:
            for p, p32 in zip(self.fp16_params, self.fp32_params):
                if not p.requires_grad:
                    continue
                p.data.copy_(p32.data)

    def _unscale_grads(self):
        self._sync_fp16_grads_to_fp32()
        if (
            # Skip the multiplication if it's a no-op (i.e., if _multiply_factor
            # is 1.0). At the same time, we want to avoid the device-to-host
            # transfer by comparing it to 1.0. Since _multiply_factor starts as
            # a Python float, we roughly assume that if it's a tensor then it's
            # probably not =1.0 anymore and we do the multiplication. Otherwise
            # we can safely check the value without a D2H transfer.
            torch.is_tensor(self._multiply_factor)
            or self._multiply_factor != 1.0
        ):
            self.fp32_optimizer.multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0

    def multiply_grads(self, c):
        """Multiplies grads by a constant ``c``."""
        self._multiply_factor *= c

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""
        self._sync_fp16_grads_to_fp32()

        max_norm = float(max_norm)
        grad_norm = self._multiply_factor * self.fp32_optimizer.clip_grad_norm(
            0, aggregate_norm_fn
        )

        if self.scaler is not None:
            grad_norm_cpu = float(grad_norm)
            if grad_norm_cpu > max_norm > 0.0:
                self._multiply_factor *= max_norm / grad_norm_cpu

            self.scaler.check_overflow(grad_norm_cpu)
        elif max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
            self._multiply_factor *= clip_coef

        return grad_norm

    def step(self, closure=None, groups=None):
        """Performs a single optimization step."""
        self._sync_fp16_grads_to_fp32()

        if getattr(self, "supports_step_with_scale", False):
            self.fp32_optimizer.step(
                closure, scale=(1.0 / self._multiply_factor), groups=groups
            )
        else:
            self._unscale_grads()
            self.fp32_optimizer.step(closure, groups=groups)

        if self.scaler is not None:
            self.scaler.update()

        self._sync_fp32_params_to_fp16()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.fp16_params:
            p.grad = None
        if self.has_flat_params:
            if torch.is_tensor(self.fp32_params):
                self.fp32_params.grad.zero_()
            elif isinstance(self.fp32_params, dict):
                for fp32_params in self.fp32_params.values():
                    fp32_params.grad.zero_()
            else:
                raise RuntimeError("self.fp32_params must be a tensor or dict")
        else:
            for p32 in self.fp32_params:
                if p32.grad is not None:
                    p32.grad.zero_()
        self._needs_sync = False

        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)
        else:
            self._multiply_factor = 1.0

    @property
    def supports_flat_params(self):
        return self.fp32_optimizer.supports_flat_params


class FP16Optimizer(_FP16OptimizerMixin, BaseOptimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.
    """

    def __init__(self, cfg: DictConfig, params, fp32_optimizer, fp32_params, **kwargs):
        super().__init__(cfg.optimizer)
        self.fp16_params = params
        self.fp32_optimizer = fp32_optimizer
        self.fp32_params = fp32_params

        if getattr(cfg.common, "fp16_scale_window", None) is None:
            if len(cfg.optimization.update_freq) > 1:
                raise ValueError(
                    "--fp16-scale-window must be given explicitly when using a "
                    "custom --update-freq schedule"
                )
            data_parallel_size = int(
                cfg.distributed_training.distributed_world_size
                / cfg.common.model_parallel_size
            )
            scale_window = int(
                2**14 / data_parallel_size / cfg.optimization.update_freq[0]
            )
        else:
            scale_window = cfg.common.fp16_scale_window

        if not getattr(cfg.common, "bf16", False):
            self.scaler = DynamicLossScaler(
                init_scale=cfg.common.fp16_init_scale,
                scale_window=scale_window,
                tolerance=cfg.common.fp16_scale_tolerance,
                threshold=cfg.common.threshold_loss_scale,
                min_loss_scale=cfg.common.min_loss_scale,
            )
        else:
            # disable loss scaling for bfloat16
            self.scaler = None

    @classmethod
    def build_optimizer(cls, cfg: DictConfig, params, **kwargs):
        """
        Args:
            cfg (omegaconf.DictConfig): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        flatten = not getattr(cfg.common, "fp16_no_flatten_grads", False)
        fp32_params = cls.build_fp32_params(cfg.optimizer, params, flatten=flatten)
        if flatten:
            fp32_optimizer = optim.build_optimizer(cfg.optimizer, [fp32_params])
        else:
            fp32_optimizer = optim.build_optimizer(cfg.optimizer, fp32_params)
        if flatten and not fp32_optimizer.supports_flat_params:
            raise RuntimeError(
                f"chosen optimizer {fp32_optimizer.__class__.__name__} does not support flat params, please set --fp16-no-flatten-grads"
            )
        return cls(cfg, params, fp32_optimizer, fp32_params, **kwargs)

    @property
    def optimizer(self):
        return self.fp32_optimizer.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.fp32_optimizer.optimizer = optimizer

    @property
    def optimizer_config(self):
        return self.fp32_optimizer.optimizer_config

    @property
    def lr_scheduler(self):
        return getattr(self.fp32_optimizer, "lr_scheduler", None)

    def get_lr(self):
        return self.fp32_optimizer.get_lr()

    def set_lr(self, lr):
        self.fp32_optimizer.set_lr(lr)

    def all_reduce_grads(self, module):
        self.fp32_optimizer.all_reduce_grads(module)