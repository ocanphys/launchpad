import pytest
import torch

from models.transformerV2.src.model import SPDA

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and triton")


@cuda
@pytest.mark.parametrize("seq", [64, 100, 257])
def test_matches_reference_forward_and_backward(seq):
    from models.transformerV2.src.flash_attention import flash_attention

    torch.manual_seed(0)
    shape = (2, 3, seq, 64)
    qkv = [torch.randn(shape, device="cuda", requires_grad=True) for _ in range(3)]
    ref = [t.detach().clone().requires_grad_() for t in qkv]
    mask = torch.tril(torch.ones(seq, seq, device="cuda", dtype=torch.bool))
    do = torch.randn(shape, device="cuda")

    out = flash_attention(*qkv)
    out.backward(do)
    expected, _ = SPDA(*ref, mask)
    expected.backward(do)

    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
    for got, want in zip(qkv, ref):
        torch.testing.assert_close(got.grad, want.grad, atol=2e-2, rtol=2e-2)
