from fairseq.optim import FairseqOptimizer


class BaseOptimizer(FairseqOptimizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group['lr_scale']
            else:
                param_group["lr"] = lr