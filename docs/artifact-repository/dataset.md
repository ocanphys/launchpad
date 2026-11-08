# Dataset

`artifacts.dataset.DataSet`: a training set that physically concatenates its
tokenized sources into `datasets/<uid>/train.bin` and `valid.bin`, with
`<|endoftext|>` between sources. Shared, not run-scoped: its bytes are a pure
function of its fields, so runs asking for the same sources reuse one folder.
[Mapped dataset](mappeddataset.md) is the variant that copies nothing.

| Field | Meaning |
| --- | --- |
| `train_set` | Ordered [`TokenizedSource`s](tokenized.md) for training. |
| `valid_set` | Ordered `TokenizedSource`s for validation. |

```python
DataSet.from_sources(tokenizer, train_sources=(odyssey,), valid_sources=(iliad,))
```

Order identifies it, because it changes the bytes. All sources must share one
tokenizer, and a split with several sources needs `<|endoftext|>` in the
tokenizer's special tokens; both are refused at construction.

## Bound view

After `bind(root)`, `train` and `valid` are each a `Split`: the split's
`tokens`, a `uint16` memmap (an empty split is an empty array), and its `tags`,
the tag of every token position. A dataset serves tokens and tags only; how
batches are drawn from them is the leg's [sampler](samplers.md).

```python
split = dataset.bind(root).train
split.tokens[100:108], split.tags[100:108]
```

Positions belong to one split, so train and valid carry their own tags. Tags
are stored as ranges (the position each starts at, and its tag id), one entry
per range rather than one per token, and read as one tag per position. No
dataset is tagged yet, so every position reads tag 0.

::: artifacts.core.split
    options:
      heading_level: 3

## The job

`DataSetJob` appends each tokenized source's `tokens.bin` to the split file,
encoding the separator with the first source's tokenizer.
