"""The base of every architecture: Model, which a leg holds in its `model`
parameter. Each architecture is a package under models/ with one subclass; this
module stays free of torch so declaring a leg never imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from artifacts.core.artifact import ParamConfig

if TYPE_CHECKING:
    import torch


class Model(ParamConfig):
    """The base of every architecture under models/: a subclass's fields are
    what a leg declares, and `build` makes the network they describe. What
    stages and jobs read off a model (`vocab_size`, `sequence_length`, `dtype`,
    `step_flops`) each architecture declares itself.

    A field annotated `Model` holds any subclass, and its manifest names that
    subclass under "class" (artifacts.core.manifest).
    """

    def build(self, device: torch.device):
        """Returns a freshly initialized torch module on `device`.

            Transformer(...).build(torch.device("cpu")) -> TransformerLM(...) with every parameter on the cpu
        """
        raise NotImplementedError

    def measure_size(self, network: torch.nn.Module) -> dict[str, int]:
        """Returns the parameter count and bytes of the built `network`, read from its weights.

            Transformer(...).measure_size(network) -> {'parameters': 34608, 'bytes': 138432}
        """
        weights = list(network.parameters())
        return {"parameters": sum(w.numel() for w in weights),
                "bytes": sum(w.numel() * w.element_size() for w in weights)}

    def measure_gradient_size(self, network: torch.nn.Module) -> dict[str, int]:
        """Returns the element count and bytes of the gradients `network` holds, which exist from a backward
        pass until the gradients are zeroed to None.

            Transformer(...).measure_gradient_size(network) -> {'parameters': 34608, 'bytes': 138432}
        """
        gradients = [w.grad for w in network.parameters() if w.grad is not None]
        return {"parameters": sum(g.numel() for g in gradients),
                "bytes": sum(g.numel() * g.element_size() for g in gradients)}
