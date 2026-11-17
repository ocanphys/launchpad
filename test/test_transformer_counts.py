import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from models.transformer import Transformer
from models.transformerFA import TransformerFA

SHAPES = [
    dict(batch=4, vocab_size=1000, sequence_length=8, num_layers=1, d_model=16, d_ff=32, num_heads=2),
    dict(batch=3, vocab_size=500, sequence_length=32, num_layers=4, d_model=64, d_ff=128, num_heads=4),
]


@pytest.mark.parametrize("cls", [Transformer, TransformerFA])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_counts_match_the_built_network(cls, shape, tie_embeddings):
    shape = dict(shape)
    batch = shape.pop("batch")
    model = cls(**shape, rope_theta=10000, tie_embeddings=tie_embeddings)
    network = model.build(torch.device("cpu"))
    assert (network.linear.W is network.embedding.weight) == tie_embeddings
    # parameters() and named_parameters() list a tied matrix once, as embedding.weight
    with FlopCounterMode(display=False) as counter:
        network(torch.randint(0, model.vocab_size, (batch, model.sequence_length)))
    assert model.parameter_count(embedding=True) == sum(p.numel() for p in network.parameters())
    assert model.parameter_count() == sum(p.numel() for name, p in network.named_parameters() if name != "embedding.weight")
    assert model.forward_flops(batch) == counter.get_total_flops()


def test_estimated_size_matches_measured_size():
    model = Transformer(**{k: v for k, v in SHAPES[0].items() if k != "batch"}, rope_theta=10000, dtype="torch.bfloat16")
    assert model.estimate_size() == model.measure_size(model.build(torch.device("cpu")))


def test_gradients_are_the_size_of_the_weights_after_a_backward_pass():
    model = Transformer(**{k: v for k, v in SHAPES[0].items() if k != "batch"}, rope_theta=10000)
    network = model.build(torch.device("cpu"))
    assert model.measure_gradient_size(network) == {"parameters": 0, "bytes": 0}
    network(torch.randint(0, model.vocab_size, (2, model.sequence_length))).sum().backward()
    assert model.measure_gradient_size(network) == model.measure_size(network)
