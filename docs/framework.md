# Framework

Launchpad turns a training pipeline into a graph of immutable, described
things, and runs the graph on Modal. This page explains the idea: what the
repo is for, how a training run maps onto it, why dependencies are resolved,
and how to think about stages.

## The goal

A training run is a chain of expensive steps: fetch text, train a tokenizer,
tokenize, assemble a dataset, train a model, train it further. Done by hand,
each step is a script with its own paths, and nothing records which output
came from which input. Two experiments that share a tokenizer rebuild it, or
worse, quietly use the wrong one.

Launchpad's answer is to describe each step as an **artifact**: a frozen
Python object whose parameters fully define what it is. Identical parameters
give an identical folder on a shared Modal Volume, so the same description is
built once, whoever asks for it.

```mermaid
flowchart LR
  D["Describe<br/>artifacts as Python objects"] --> R["Resolve<br/>work out what must exist, in order"]
  R --> L["Launch<br/>one job per missing artifact"]
  L --> V[("Volume<br/>one folder per artifact")]
  V --> B["Bind<br/>load finished artifacts for use"]
  V -. "finished artifacts are skipped" .-> R
```

What this buys:

- **No rebuilds.** An artifact whose output files exist is done, and is never
  produced again.
- **Reproducibility.** The manifest next to the outputs records exactly what
  was asked for, down to every upstream artifact.
- **Conflicts are caught.** Two different definitions that map to the same
  folder are an error, not a silent overwrite.
- **Resumable work.** Each step is its own job and its own folder, so a
  failure costs one step, not the pipeline.

## How a training run maps onto artifacts

Each step of the training process is an artifact family. An artifact names the
artifacts it is built from as constructor parameters, and those names are the
edges of the graph.

| Training step | Artifact | Output | Built from |
| --- | --- | --- | --- |
| Raw text | `SourceURL` | `body.txt` | nothing |
| Tokenizer training | `Tokenizer` (BPE, template) | `tokenizer.json` | sources |
| Tokenization | `TokenizedSource` | `tokens.bin` | a tokenizer and a source |
| Dataset assembly | `DataSet` or `MappedDataSet` | `train.bin`, `valid.bin` (`MappedDataSet` owns no bytes) | tokenized sources |
| Training | `Pretraining` (one leg) | `model.pt` | a dataset, and optionally the previous leg |

The model architecture is not an artifact. It is a parameter of the training
leg, an instance of a `Model` class from `models/`, so one `Pretraining` class
trains every architecture.

```mermaid
flowchart LR
  SRC["SourceURL<br/>body.txt"] --> TOK["Tokenizer<br/>tokenizer.json"]
  SRC --> TS["TokenizedSource<br/>tokens.bin"]
  TOK --> TS
  TS --> DS["DataSet<br/>train.bin, valid.bin"]
  DS --> L1["Pretraining leg 1<br/>steps 0 to N<br/>model.pt"]
  L1 --> L2["Pretraining leg 2<br/>steps N to 2N<br/>model.pt"]
  DS --> L2
```

Arrows point from an artifact to what is built from it. A leg continues the
previous one by naming it as its `starting_checkpoint`, so a leg's
dependencies include every leg before it.

## Why dependencies are resolved

Describing the last artifact in the chain is enough, because its parameters
already name everything it needs. Resolution turns that nested description
into a work list.

```python
import lab

lab.declare(leg2)  # reports every artifact leg2 needs, dependencies first
```

Resolution does three jobs that nothing else can.

**Order.** `resolve(artifact)` returns the artifact and everything beneath it,
dependencies before dependents. The launcher only runs an artifact when all
its direct dependencies are done, so every artifact reports one of `blocked`,
`runnable`, `starting`, `running`, `failed` or `done`.

```mermaid
flowchart LR
  B["blocked<br/>a dependency is not done"] --> R["runnable<br/>every dependency is done"]
  R --> S["starting<br/>lease granted"]
  S --> G["running<br/>job is beating"]
  G --> D["done<br/>output files exist"]
  G --> F["failed<br/>runnable again"]
```

**Sharing.** Two runs rarely differ everywhere. Resolution keys on
`artifact_path`, so the source, tokenizer and dataset that two experiments
have in common appear once and are built once.

```mermaid
flowchart LR
  subgraph shared ["Built once, shared by both runs"]
    S["SourceURL"] --> T["Tokenizer"]
    S --> TS["TokenizedSource"]
    T --> TS
    TS --> DS["DataSet"]
  end
  DS --> A["Pretraining<br/>transformer"]
  DS --> M["Pretraining<br/>mamba"]
```

**Safety.** A path reached twice must carry the same definition, or the graph
is rejected. A path that reaches itself is a cycle and is rejected.

```mermaid
flowchart TD
  V["Visit an artifact"] --> C{"Path on the active stack?"}
  C -->|yes| E1["Raise: cycle"]
  C -->|no| S{"Path already in the list?"}
  S -->|yes| A{"Definitions equal?"}
  A -->|no| E2["Raise: conflict"]
  A -->|yes| K["Skip: already in the list"]
  S -->|no| D["Visit each dependency, in order"]
  D --> M["Add this artifact to the list"]
```

Resolution reads no storage and launches nothing. Whether an artifact is
already done is a separate question, answered by looking at its folder.

## Stages

A **stage** is a family of artifacts that consumes the outputs of earlier
artifacts and produces outputs for later ones. It is a way of reading the
graph, not a type in the code.

- **Data stage.** Sources, tokenizers, tokenized sources and datasets. These
  are shared across experiments, which is why they are identified by their
  parameters alone and carry no run name.
- **Pretraining stage.** `Pretraining` legs. A leg trains for a fixed number
  of steps and is complete when its `model.pt` exists. A long run is a chain
  of legs rather than one long job, so each leg is resumable, inspectable and
  a valid starting point for another run. Within a leg, periodic failsafe
  checkpoints let an interrupted job pick up where it stopped. They are not
  artifacts, and nothing may depend on them.

```mermaid
flowchart LR
  subgraph data ["Data stage: shared across runs"]
    SRC["Sources"] --> TOK["Tokenizer"]
    TOK --> TS["Tokenized sources"]
    SRC --> TS
    TS --> DS["Dataset"]
  end
  subgraph pre ["Pretraining stage: one chain per run"]
    L1["Leg 1"] --> L2["Leg 2"] --> L3["Leg 3"]
  end
  DS --> L1
  DS --> L2
  DS --> L3
  L3 --> OUT["model.pt<br/>the trained model"]
```

Only pretraining exists today. The pattern for another stage is the same: a
new artifact family whose parameters name the artifacts it consumes, with a
job that writes its output files. Resolution, launching and the dashboard
need no changes, because they only ever see artifacts and their dependencies.

## Where to go next

- [Artifacts](artifacts.md): how an artifact is defined and identified.
- [Manifests](manifests.md): how a definition is written to disk and read back.
- [Jobs](jobs.md): the code that builds one artifact.
- [Launcher](launcher.md): how the graph is launched and watched.
