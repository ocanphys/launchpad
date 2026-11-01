# Model Repository

Every architecture a leg can train, one page each. [Models](models.md) is the
contract they share; these pages are the details of each implementation.

| Model | Class | Mixer | Notes |
| --- | --- | --- | --- |
| [Transformer](model-repository/transformer.md) | `models.transformer.Transformer` | Causal attention with RoPE | Dense attention; can capture scores. |
| [Transformer FA](model-repository/transformerFA.md) | `models.transformerFA.TransformerFA` | Causal attention with RoPE | Same shapes; PyTorch's flash attention on CUDA. |
| [Mamba](model-repository/mamba.md) | `models.mamba.Mamba` | Selective state space scan | No FLOP count yet. |

A model's page covers its architecture, its fields, and what it can count.
Adding an architecture ([Models](models.md#adding-an-architecture)) also means
a page under `docs/model-repository/`, a row above, and a nav entry.
