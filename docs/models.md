# Models

A model is not an artifact. It is a param config of the leg that trains it:
validated and frozen like the rest of the leg's parameters, part of the leg's
identity, and with no folder, manifest or job of its own. Each architecture is
a package under `models/` (`transformer`, `transformerFA`, `mamba`) holding one
`Model` subclass, which declares the architecture's shapes and builds the
network from them. The package imports no torch until `build()` runs, so
declaring a leg stays cheap.

## What readers expect of a model

`Model` declares only `build(device)`, which returns the freshly initialized
torch module on `device`. It has no fields: each architecture declares its own,
and the code that reads a model asks for exactly the names it needs.

| Name | Read by |
| --- | --- |
| `vocab_size` | `Pretraining`, which refuses a model whose vocabulary is not its tokenizer's. |
| `sequence_length` | The pretraining job, for the shape of every batch it draws; chat, for its context window. |
| `dtype` | `build()`, as a dotted torch dtype (`"torch.float32"`) resolved on the worker; profiling reports it. |
| `step_flops(batch_size)` | `gpus/monitor.py`, which turns it into MFU, and the profiler. |

Nothing enforces the table. A model that lacks a name breaks only the reader
that needs it: `Mamba` has no FLOP counts yet, so training it works and MFU
reporting does not. All three shipped models declare the three fields, in this
order, and the Transformers add `parameter_count`, `forward_flops`,
`backward_flops` and `step_flops` (twice the forward, and the sum of the two).

Sizes come in two kinds, named for how they are obtained. `measure_size` reads the
tensors of a built network (on `Model`) or of a built optimizer that has stepped (on `Optimizer`),
and the profiler calls both. `estimate_size` is a theoretical value from the declaration alone,
with its approximations stated in its docstring, and exists only where a class can give one: the
Transformers (from `parameter_count` and `dtype`), `AdamW` (two moment buffers per weight) and
`SGD` (none).

Fields carry their constraints (`num_layers: int = Field(gt=0)`). A value that
breaks one is refused when the model is built, at that line, with a
`ValidationError` naming the field.

The device is not a field. It is where a leg runs, not what it trains, so the
training job picks it, `cuda` when there is a GPU and `cpu` otherwise, and
passes it to `build` and to the GPU monitor; batches go to the same device.
A leg has one path whichever device trains it, and the profiler builds on
`cuda`.

## Which class a manifest names

A leg's `model` field is annotated `Model` and holds a subclass. JSON carries
field values, not types, so the leg's manifest records the subclass's import
path under `"class"`, beside its fields:

```json
"model": {"class": "models.transformer.Transformer", "d_model": 128, "num_heads": 16, ...}
```

The codec writes that key for any param config held in a field annotated with
one of its base classes, at any depth, and reads it back; `Model` declares
nothing for it. The import path is part of the stored format: moving or
renaming an architecture class leaves every manifest that names it unreadable
until they are rewritten. It is also part of a leg's identity, since
`lineage_hash` digests the model as encoded ([Training](training.md#identity)),
so the rewrite moves the legs as well.

## Adding an architecture

Add a package under `models/` whose `__init__.py` defines a `Model` subclass:
its fields and their constraints, including the names in the table above,
and `build(device)` importing torch inside it. `Pretraining` needs no change,
since it reads only those names.

## Why a subclass per architecture

Each architecture owns three things: its fields; how they map onto its
module's constructor (`sequence_length` becomes `context_length`, the `dtype`
string becomes a torch dtype, `Mamba` ignores `sequence_length`); and its FLOP
count. A class per architecture keeps all three in one place. A single
`ModelSpec` holding a class reference and a config would separate the fields
from the class, and the mapping and the FLOP count would still need a home per
architecture. A union of the architectures annotated on the leg would make the
stage import and list every architecture, and could not tell apart two with
the same fields, as `Transformer` and `TransformerFA` have.

## Reference

::: models.Model

::: models.transformer.Transformer

::: models.transformerFA.TransformerFA

::: models.mamba.Mamba
