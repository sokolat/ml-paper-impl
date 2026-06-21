from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

class LinearWarmupLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int) -> None:
        self.warmup_factor = warmup_steps ** -1.5
        self.scale = d_model ** -0.5
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        step = self.last_epoch + 1  # last_epoch is the step counter (step() called per batch)
        lr = self.scale * min(step ** -0.5, step * self.warmup_factor)

        return [lr for _ in self.optimizer.param_groups]
