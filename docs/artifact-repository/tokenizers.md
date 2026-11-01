# Tokenizers

`artifacts.tokenizers`: the contract every tokenizer family satisfies, and the
only thing code outside this folder depends on. A consumer annotates
`tokenizer: Tokenizer` and never names a family.

| Field | Meaning |
| --- | --- |
| `vocab_size` | Number of ids. Expected to stay under 2**16, because tokens are stored as `uint16`. |
| `special_tokens` | Strings that always encode to a single id. |

A tokenizer lives at `tokenizers/<uid>/` and owns `tokenizer.json`. The trained
state is not a parameter: it is what the job wrote, and arrives through
`bind(root)`. `encode` and `decode` raise on an unbound tokenizer.
`encode_iterable` encodes texts one at a time so a source of any size is never
held whole.

## BPE

`artifacts.tokenizers.bpe.Tokenizer`, trained by `TokenizerJob`. Byte-level
BPE with the GPT-style pretokenizer regex; byte strings are stored in
`tokenizer.json` as latin-1 text, which round-trips any byte sequence.

| Field | Meaning |
| --- | --- |
| `sources` | The `Source`s it trains on, a set: order does not identify it. |

The uid is `bpe-<vocab_size>-<digest>`, the digest covering `vocab_size`,
`special_tokens` and the sources' uids. Training and encoding share `PAT` and
`mergebpairs`, the pair that must not drift apart.

## Template

`artifacts.tokenizers.template.TemplateTokenizer` is the shape of a new family,
with a deliberately simple word-level vocabulary of the most frequent words.
To add a family, copy the folder, rename the classes, write the full dotted
path of your job in `producer`, and replace the algorithm in `train` and in
`encode`/`decode`. Keep `uid` injective over the parameters: two parameter sets
sharing a uid would overwrite one another's folder.
