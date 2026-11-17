"""TransformerFA is Transformer with flash attention: the same weights from the same seed, the same
output wherever it falls back to the dense scaled dot product, and close to it where flash runs."""

import pytest
import torch

from models.transformer import Transformer
from models.transformerFA import TransformerFA

SHAPE = dict(vocab_size=500, sequence_length=32, num_layers=2, d_model=64, d_ff=128, num_heads=4, rope_theta=10000)
flash_gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
                               reason="flash attention needs an Ampere or later GPU")


def built(cls, device, **fields):
    torch.manual_seed(0)
    return cls(**SHAPE, **fields).build(torch.device(device))


def tokens(device):
    return torch.randint(0, SHAPE["vocab_size"], (2, SHAPE["sequence_length"]), device=device)


@pytest.mark.parametrize("tie_embeddings", [False, True])
@pytest.mark.parametrize(("dtype", "autocast"), [("torch.float32", False), ("torch.bfloat16", False), ("torch.float32", True)])
def test_the_same_seed_builds_the_same_weights_and_on_the_cpu_the_same_output(dtype, autocast, tie_embeddings):
    fields = {"dtype": dtype, "tie_embeddings": tie_embeddings}
    reference, flash = built(Transformer, "cpu", **fields), built(TransformerFA, "cpu", **fields)
    assert reference.state_dict().keys() == flash.state_dict().keys()
    assert all(torch.equal(a, b) for a, b in zip(reference.state_dict().values(), flash.state_dict().values()))
    batch = tokens("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        logits = reference(batch), flash(batch)
    assert logits[0].dtype == (torch.bfloat16 if autocast or dtype == "torch.bfloat16" else torch.float32)
    assert torch.equal(*logits)


@flash_gpu
def test_flash_attention_matches_the_dense_product_under_autocast():
    network, batch = built(TransformerFA, "cuda"), tokens("cuda")

    def logits(capture_internals):  # capturing internals takes the dense path
        for block in network.tblocks:
            block.attention.capture_internals = capture_internals
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return network(batch)

    torch.testing.assert_close(logits(False), logits(True), atol=2e-2, rtol=2e-2)


@flash_gpu
def test_flash_attention_refuses_fp32():
    with pytest.raises(RuntimeError):
        built(TransformerFA, "cuda")(tokens("cuda"))
