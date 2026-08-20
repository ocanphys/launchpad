# Job spec — v2

Extends the barebones spec with a job-type system and pydantic-mediated parameters, and binds the ML training schema to it. The core stays abstract; the training jobs are one instantiation, given in the appendix.

---

## 1. Job

```text
Job(parameters, dependencies)
```

**parameters** — a pydantic model instance, typed per job type (§2).

**dependencies** — named lists of references to other jobs. Every value in every list is a job object or the manifest path of a job. Nothing else is admissible — not a parameter dict, not a data file path — and this is enforced at initiation.

Every dependency name maps to a **list**, always. A single dependency is a list of one. There is no scalar form; code never branches on arity.

Dependency names are declared by the job type and bound at initiation; the job refers to dependencies only by these names.

Every job has a deterministic **id** derived from its parameters and dependency ids. Derivation is deferred; two jobs with the same id are interchangeable.

A job may therefore be initialized either by constructing it explicitly or by pointing to an existing manifest:

```python
tokenizer = TrainTokenizer(
    {...},
    run_id,
    sources=[src(name) for name in TOKEN_SET],
)
```

or:

```python
tokenizer = Job.from_manifest(
    "tokenizers/llama-64/manifest.json"
)
```

The latter loads the existing job as a frozen dependency and does not rebuild its dependencies.

---

## 2. Job type

A job type is a class defining:

* **Parameters** — a pydantic model. Construction of the job validates the parameters; an invalid job cannot be initiated. Field types, defaults, and per-field constraints live here and nowhere else.

* **dependency slots** — the named dependency lists the type accepts: each slot's name and expected job type. Every slot holds a list; expected length (exactly one, one-or-more, zero-or-more) is a constraint on the list, not a change of shape.

* **output declarations** — the named output lists the type produces.

* **input bindings** — for each input name, which dependency slot and which of that slot's output names it consumes.

All four are declared on the type. The caller supplies only parameter values and the dependency lists to fill the slots.

Job types are registered by name; the manifest's `job_type` string resolves back to the class, and thereby to the Parameters model used to re-validate a loaded manifest.

---

## 3. Files

**outputs** — declared by the type, materialized per instance as a named list of paths. Paths may interpolate parameters, e.g. a uid. A single file is a list of one; a glob-like family (`text/*`, `checkpoints/*`) is a list resolved at materialization. Path layout is deferred.

**inputs** — named lists of paths, never supplied by the caller. Each input resolves through a dependency slot: `self.dep("tokenizer")` yields a list of jobs, and an input binding concatenates the named outputs across that list. Inputs are therefore lists by construction.

A job knows its dependencies' output **names**, never their paths. This remains the only coupling between jobs.

---

## 4. Dependency given as a manifest

A dependency list entry that is a manifest path is loaded and taken as given: a **frozen leaf**.

Its parameters are re-validated against its job type's pydantic model on load. A manifest that no longer validates is an error surfaced at load, not at run.

Its own dependencies are not rebuilt.

Live jobs and frozen jobs may be mixed within one slot's list; a job wiring against the slot cannot tell them apart.

---

## 5. Manifest

```json
{
  "id": "<string>",
  "job_type": "<string>",
  "parameters": { "...": "..." },
  "dependencies": {
    "<name>": ["<manifest>", "..."]
  },
  "inputs": {
    "<name>": ["<path>", "..."]
  },
  "outputs": {
    "<name>": ["<path>", "..."]
  }
}
```

Every value under `dependencies`, `inputs`, and `outputs` is a list — no scalar variants exist in the format.

`parameters` round-trips through the pydantic model: dumped on save, re-parsed and validated on load.

`dependencies` nests recursively; the manifest is a recipe, not a graph.

A manifest corresponds to one **job initialization**.

---

## 6. DAG

Unchanged: walk depth-first through every dependency list, dedupe by id, topo-sort.

Frozen nodes enter as nodes but are not walked into.

---

## 7. Status and scheduling

Unchanged in structure, restated over lists:

* **done** — every path in every output list exists.
* **runnable** — not done, and every path in every input list exists.
* **blocked** — not done, and some input path is missing.

A node is scheduled if not done or if anything upstream is scheduled; frozen nodes are never scheduled.

---

## 8. Validation boundaries

**Per-job** (now):

* pydantic Parameters model at initiation and at manifest load;
* dependency admissibility (job or manifest, nothing else);
* slot list-length constraints at initiation.

**Cross-job** (deferred):

* consistency between a job's parameters and its dependencies' parameters;
* slot type enforcement beyond duck typing.

---

# Appendix: training job types

The training workflow consists of six job types.

Every dependency slot is a list. The length constraint is noted. Paths shown are the intended layout but remain non-normative here.

| job type           | parameters (pydantic)                                                | dependency slots (list, length)                                    | inputs (via slot)                                           | outputs                                                                               |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **DownloadSource** | `source_uid, url, type, config`                                      | —                                                                  | —                                                           | `text` → `sources/{source_uid}/text/*`                                                |
| **TrainTokenizer** | `tokenizer_uid, vocab_size, kind, special_tokens, token_source_uids` | `sources: [DownloadSource], 1+`                                    | `texts` ← `sources.text`                                    | `tokenizer`, `config` → `tokenizers/{tokenizer_uid}/…`                                |
| **TokenizeSource** | `tokenizer_uid, source_uid`                                          | `source: [DownloadSource], =1`; `tokenizer: [TrainTokenizer], =1`  | `text` ← `source.text`; `tokenizer`, `config` ← `tokenizer` | `bin` → `tokenizers/{tokenizer_uid}/bin/{source_uid}/*`                               |
| **BuildSplit**     | `train_source_uids, valid_source_uids, tokenizer_uid`                | `tokenizer: [TrainTokenizer], =1`; `sources: [TokenizeSource], 1+` | tokenized source bins; tokenizer metadata                   | `train`, `valid` → `datasets/{dataset-id}/train.bin`, `datasets/{dataset-id}/val.bin` |
| **Pretrain**       | `run_id, model_config, training_config`                              | `dataset: [BuildSplit], =1`                                        | `train`, `valid` ← dataset                                  | `checkpoints`, `logs`, `progress` → `runs/{run_id}/pretrain/…`                        |
| **SFT**            | `run_id, base_run_id, sft_training_config`                           | `base: [Pretrain], =1`                                             | `checkpoints` ← base.checkpoints                            | `checkpoints`, `logs`, `progress` → `runs/{run_id}/sft/…`                             |

### BuildSplit

`BuildSplit` is the boundary between **source-level data preparation** and **training-level data consumption**.

Its parameters specify the logical source selection:

```python
BuildSplit(
    {
        "train_source_uids": ["odyssey", "mobydick"],
        "valid_source_uids": ["montecristo", "romeojuliet"],
        "tokenizer_uid": "llama-64",
    },
    run_id,
    tokenizer=[tokenizer],
    sources=[
        tokenize_odyssey,
        tokenize_mobydick,
        tokenize_montecristo,
        tokenize_romeojuliet,
    ],
)
```

It depends on the tokenized representations of the requested sources and materializes the two concrete datasets:

```text
datasets/{dataset-id}/train.bin
datasets/{dataset-id}/val.bin
```

`BuildSplit` is responsible for determining which tokenized sources belong to each split and constructing the corresponding files. `Pretrain` does not know which individual sources produced those files.

---

# Appendix: notebook initialization

A complete workflow can now be expressed as:

```python
VOCAB_SIZE = 64

TRAIN_SET = ["odyssey", "mobydick"]
VALID_SET = ["montecristo", "romeojuliet"]
TOKEN_SET = ["odyssey", "mobydick"]

RUN_ID = "gpt2-small-v1"


def src(name, run_id=RUN_ID):
    return DownloadSource(
        {
            "source_uid": name,
            "url": ...,
            "type": ...,
            "config": {
                "normalizer": "lowercase",
            },
        },
        run_id,
    )


tokenizer = TrainTokenizer(
    {
        "tokenizer_uid": "llama-64",
        "vocab_size": VOCAB_SIZE,
        "kind": "bpe",
        "special_tokens": [
            "<|endoftext|>",
            "<|begin|>",
            "<|end|>",
        ],
        "token_source_uids": TOKEN_SET,
    },
    RUN_ID,
    sources=[
        src(name)
        for name in TOKEN_SET
    ],
)


tokenized = [
    TokenizeSource(
        {
            "tokenizer_uid": "llama-64",
            "source_uid": name,
        },
        RUN_ID,
        source=[src(name)],
        tokenizer=[tokenizer],
    )
    for name in set(TRAIN_SET + VALID_SET)
]


dataset = BuildSplit(
    {
        "train_source_uids": TRAIN_SET,
        "valid_source_uids": VALID_SET,
        "tokenizer_uid": "llama-64",
    },
    RUN_ID,
    tokenizer=[tokenizer],
    sources=tokenized,
)


pretrain = Pretrain(
    {
        "run_id": RUN_ID,
        "model_config": {
            "n_layer": 2,
            "n_embd": 64,
        },
        "training_config": {
            "lr": 3e-4,
            "epochs": 1,
        },
    },
    RUN_ID,
    dataset=[dataset],
)


sft = SFT(
    {
        "run_id": RUN_ID,
        "base_run_id": RUN_ID,
        "sft_training_config": {
            "lr": 1e-5,
            "epochs": 2,
        },
    },
    RUN_ID,
    base=[pretrain],
)
```

The resulting dependency graph is:

```text
DownloadSource
      │
      ├───────────────┐
      │               │
      ▼               ▼
TrainTokenizer   TokenizeSource
      │               │
      │               │
      └───────┬───────┘
              │
              ▼
          BuildSplit
              │
              │
       ┌──────┴──────┐
       ▼             ▼
    train.bin      val.bin
       │             │
       └──────┬──────┘
              ▼
           Pretrain
              │
              ▼
             SFT
```

More precisely, `BuildSplit` fans out over the requested sources:

```text
                         ┌── Tokenize(odyssey) ────┐
                         │                         │
                         ├── Tokenize(mobydick) ───┤
                         │                         │
TrainTokenizer ──────────┼── Tokenize(montecristo) ┤
                         │                         │
                         └── Tokenize(romeojuliet) ┘
                                                   │
                                                   ▼
                                              BuildSplit
                                             /          \
                                            ▼            ▼
                                        train.bin      val.bin
                                            \            /
                                             ▼          ▼
                                              Pretrain
```

The same workflow can reuse existing manifests rather than reconstructing the upstream jobs:

```python
tokenizer = Job.from_manifest(
    "tokenizers/llama-64/manifest.json"
)

tokenized = [
    Job.from_manifest(
        f"tokenizers/llama-64/bin/{name}/manifest.json"
    )
    for name in TRAIN_SET + VALID_SET
]

dataset = BuildSplit(
    {
        "train_source_uids": TRAIN_SET,
        "valid_source_uids": VALID_SET,
        "tokenizer_uid": "llama-64",
    },
    RUN_ID,
    tokenizer=[tokenizer],
    sources=tokenized,
)

pretrain = Pretrain(
    {
        "run_id": RUN_ID,
        "model_config": {
            "n_layer": 2,
            "n_embd": 64,
        },
        "training_config": {
            "lr": 3e-4,
            "epochs": 1,
        },
    },
    RUN_ID,
    dataset=[dataset],
)
```

An existing dataset can also be referenced directly:

```python
dataset = Job.from_manifest(
    "datasets/{dataset-id}/manifest.json"
)

pretrain = Pretrain(
    {...},
    RUN_ID,
    dataset=[dataset],
)
```

Thus `Pretrain` has a deliberately small interface: it receives **one dataset job** and trains on its `train` and `valid` outputs. Source selection, tokenizer application, and split construction are all upstream concerns handled by the data DAG.

SFT remains even simpler: it depends only on `Pretrain` and consumes the resulting pretrained checkpoints.

---

# Resulting conceptual layers

```text
SOURCE LAYER

DownloadSource
    │
    ▼
raw text


TOKENIZER LAYER

TrainTokenizer
    │
    ▼
tokenizer


TOKENIZATION LAYER

TokenizeSource
    │
    ▼
tokenized source


DATASET LAYER

BuildSplit
    │
    ├── train.bin
    └── val.bin


TRAINING LAYER

Pretrain
    │
    ▼
pretrained checkpoints


FINETUNING LAYER

SFT
    │
    ▼
SFT checkpoints
```

This makes each layer have a single responsibility: `DownloadSource` acquires data, `TrainTokenizer` creates a tokenizer, `TokenizeSource` converts a source into the tokenizer's representation, `BuildSplit` constructs the actual train/validation dataset, `Pretrain` trains a model from that dataset, and `SFT` fine-tunes the resulting model.
