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

After `bind(root)`, `train_tokens` and `valid_tokens` are `uint16` memmaps
(an empty split is an empty array). Two methods serve the training loop:

- `train_batches(batch_size, sequence_length, seed)` returns a function from a
  step to `(inputs, targets)`. The batch depends on `(seed, step)` alone, so a
  resumed run draws the batches the interrupted one would have.
- `valid_batches(batch_size, sequence_length)` returns the batch count and a
  function from a batch id to consecutive, non-overlapping sequences. A split
  shorter than one sequence raises.

## The job

`DataSetJob` appends each tokenized source's `tokens.bin` to the split file,
encoding the separator with the first source's tokenizer.
