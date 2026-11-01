# Tokenized

`artifacts.tokenized.TokenizedSource`: one source run through one tokenizer, at
`tokenized/<tokenizer uid>/<source uid>/tokens.bin`. It is its own family
because neither side owns it: a tokenizer family does not own what it is run
over, and a dataset only consumes these. Grouping by tokenizer means one `ls`
shows every source a tokenizer has been run over.

| Field | Meaning |
| --- | --- |
| `tokenizer` | Any [tokenizer](tokenizers.md). |
| `source` | The [source](sources.md) to encode. |

`tokens.bin` is raw `uint16` ids with no header, so it can be copied or
memory-mapped as it is.

## The job

`TokenizeSourceJob` binds the tokenizer, then encodes `body.txt` a line at a
time and flushes ids to disk every 2**20 tokens, so neither text nor ids are
held whole (a list of ids costs about ten times its file). Progress reports
bytes read.
