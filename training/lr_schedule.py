"""The learning rate schedules a training leg can declare: one LRSchedule
subclass per schedule, each declaring its parameters and computing the rate
of a step. The schedule is the only place a leg states its learning rate; the
optimizer is handed the rate before every step.
"""

import math

from artifacts.core.artifact import ParamConfig


class LRSchedule(ParamConfig):
    """The base of every learning rate schedule a leg declares: a subclass's
    fields are its parameters, and `rate` is the learning rate of a step.
    Independent of the Optimizer: any schedule can drive any optimizer.

    A field annotated `LRSchedule` holds any subclass, and its manifest names
    that subclass under "class" (artifacts.core.manifest). A custom schedule is
    a subclass in an importable module, so a manifest can name it.

        WarmupCosine(max_learning_rate=1e-3, min_learning_rate=1e-4, warmup_iters=10, cosine_cycle_iters=100).rate(5)
          -> 0.0005
    """

    def rate(self, step: int) -> float:
        """Returns the learning rate of step number `step`."""
        raise NotImplementedError


class Constant(LRSchedule):
    """A linear warmup from 0 to learning_rate over warmup_iters steps, then
    learning_rate at every step. No warmup by default.

        Constant(learning_rate=3e-4, warmup_iters=10).rate(5) -> 0.00015
    """

    learning_rate: float
    warmup_iters: int = 0

    def rate(self, step: int) -> float:
        if step < self.warmup_iters:
            return step * self.learning_rate / self.warmup_iters
        return self.learning_rate


class WarmupCosine(LRSchedule):
    """A linear warmup from 0 to max_learning_rate over warmup_iters steps, then
    a cosine decay to min_learning_rate at cosine_cycle_iters, and
    min_learning_rate after it."""

    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int

    def rate(self, step: int) -> float:
        if step < self.warmup_iters:
            return step * self.max_learning_rate / self.warmup_iters
        if step > self.cosine_cycle_iters:
            return self.min_learning_rate
        return (
            self.min_learning_rate
            + (1 + math.cos((step - self.warmup_iters) * math.pi / (self.cosine_cycle_iters - self.warmup_iters)))
            * (self.max_learning_rate - self.min_learning_rate)
            / 2.0
        )
