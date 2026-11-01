"""The optimizers a training leg can declare: one Optimizer subclass per torch
optimizer, each declaring the keyword arguments it is built with, defaulting
to torch's. Every one is built over a model's parameters and stepped once per
backward, which is what an SGD-family optimizer is; this module stays free of
torch.
"""

from typing import ClassVar

from artifacts.core.artifact import ParamConfig
from artifacts.core.locate import locate


class Optimizer(ParamConfig):
    """The base of every optimizer a leg declares: a subclass's fields are the
    keyword arguments its `torch_class` is built with, after the parameters,
    each defaulting to torch's; torch checks their values when it builds it.
    A field left out is validated to its default, so the manifest names it.
    The learning rate is not one of them: the leg's lr_schedule sets it before
    every step.

    A field annotated `Optimizer` holds any subclass, and its manifest names that
    subclass under "class" (artifacts.core.manifest).

        AdamW(weight_decay=0.1).model_dump()
          -> {'betas': (0.9, 0.999), 'weight_decay': 0.1, 'eps': 1e-08}
    """

    torch_class: ClassVar[str]  # resolved worker-side via artifacts.core.locate.locate

    def build(self, parameters):
        """Returns the torch optimizer over `parameters`, built with this one's fields."""
        return locate(self.torch_class)(parameters, **self.model_dump())

    def measure_size(self, optimizer) -> dict[str, int]:
        """Returns the element count and bytes of the state tensors of the built torch `optimizer`, which
        exist only after its first step.

            AdamW().measure_size(optimizer) -> {'parameters': 69228, 'bytes': 276912}
        """
        state = [t for per_parameter in optimizer.state.values() for t in per_parameter.values() if hasattr(t, "numel")]
        return {"parameters": sum(t.numel() for t in state), "bytes": sum(t.numel() * t.element_size() for t in state)}


class AdamW(Optimizer):
    torch_class = "torch.optim.AdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    eps: float = 1e-8

    def estimate_size(self, model_size: dict[str, int]) -> dict[str, int]:
        """Returns the state for weights of `model_size` (as measure_size returns it): two moment buffers
        (exp_avg, exp_avg_sq) per weight. The per-parameter step counters are not counted.

            AdamW().estimate_size({'parameters': 34608, 'bytes': 138432}) -> {'parameters': 69216, 'bytes': 276864}
        """
        return {key: 2 * value for key, value in model_size.items()}


class SGD(Optimizer):
    """Vanilla SGD: no momentum, no weight decay, each step subtracts the
    scheduled rate times the gradient.

        SGD().model_dump() -> {}
    """

    torch_class = "torch.optim.SGD"

    def estimate_size(self, model_size: dict[str, int]) -> dict[str, int]:
        """Returns the state for weights of `model_size`: none, since SGD keeps no momentum.

            SGD().estimate_size({'parameters': 34608, 'bytes': 138432}) -> {'parameters': 0, 'bytes': 0}
        """
        return {"parameters": 0, "bytes": 0}
