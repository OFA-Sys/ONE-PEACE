from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq.optim import _build_optimizer

from .dynamic_loss_scaler import DynamicLossScaler
from .base_optimizer import BaseOptimizer


class _MemoryEfficientFP16OptimizerMixin(object):
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in MRO (method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    @property
    def has_flat_params(self):
        return False

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.wrapped_optimizer.state_dict()
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

        self.wrapped_optimizer.load_state_dict(state_dict, optimizer_overrides)

        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. But with --memory-efficient-fp16 the
        # params are FP16 while the optimizer state is FP32 and we don't want
        # to cast. A workaround is to manually copy back the original state
        # after the optimizer has been loaded.
        if not getattr(self.optimizer, "disable_mem_eff_fp16_loading_hack", False):
            groups = self.optimizer.param_groups
            saved_groups = state_dict["param_groups"]
            id_map = {
                old_id: p
                for old_id, p in zip(
                    chain(*(g["params"] for g in saved_groups)),
                    chain(*(g["params"] for g in groups)),
                )
            }
            for k, v in state_dict["state"].items():
                if k in id_map:
                    param = id_map[k]
                    self.optimizer.state[param] = v

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()

    def _unscale_grads(self):
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
            if self.cfg.use_distributed_fused_adam:
                self.wrapped_optimizer.optimizer.unscale_grads(self._multiply_factor)
            else:
                self.wrapped_optimizer.multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        self._multiply_factor *= c

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""
        max_norm = float(max_norm)
        if hasattr(self.wrapped_optimizer.optimizer, 'clip_grad_norm'):
            grad_norm = self._multiply_factor * self.wrapped_optimizer.optimizer.clip_grad_norm(float('inf'))
        else:
            grad_norm = self._multiply_factor * self.wrapped_optimizer.clip_grad_norm(
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
        if getattr(self, "supports_step_with_scale", False):
            # NOTE(msb) optimizer divides by scale factor
            self.wrapped_optimizer.step(
                closure, scale=(1.0 / self._multiply_factor), groups=groups
            )
        else:
            self._unscale_grads()
            self.wrapped_optimizer.step(closure, groups=groups)

        if self.scaler is not None:
            self.scaler.update()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.wrapped_optimizer.zero_grad()

        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)
        else:
            self._multiply_factor = 1.0

    @property
    def supports_flat_params(self):
        return self.wrapped_optimizer.supports_flat_params


class MemoryEfficientFP16Optimizer(
    _MemoryEfficientFP16OptimizerMixin, BaseOptimizer
):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.

    Compared to :class:`fairseq.optim.FP16Optimizer`, this version does not
    maintain an FP32 copy of the model. We instead expect the optimizer to
    convert the gradients to FP32 internally and sync the results back to the
    FP16 model params. This significantly reduces memory usage but slightly
    increases the time spent in the optimizer.

    Since this wrapper depends on specific functionality in the wrapped
    optimizer (i.e., on-the-fly conversion of grads to FP32), only certain
    optimizers can be wrapped. This is determined by the
    *supports_memory_efficient_fp16* property.
    """

    def __init__(self, cfg: DictConfig, params, optimizer, **kwargs):
        super().__init__(cfg.optimizer)
        self.wrapped_optimizer = optimizer

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
        fp16_optimizer = _build_optimizer(cfg.optimizer, params)
        return cls(cfg, params, fp16_optimizer, **kwargs)

    @property
    def optimizer(self):
        return self.wrapped_optimizer.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.wrapped_optimizer.optimizer = optimizer

    @property
    def optimizer_config(self):
        return self.wrapped_optimizer.optimizer_config

    @property
    def lr_scheduler(self):
        return getattr(self.wrapped_optimizer, "lr_scheduler", None)

    def get_lr(self):
        return self.wrapped_optimizer.get_lr()

    def set_lr(self, lr):
        self.wrapped_optimizer.set_lr(lr)

    def all_reduce_grads(self, module):
        self.wrapped_optimizer.all_reduce_grads(module)