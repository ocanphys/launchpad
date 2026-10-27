# Experiments

*Draft.* Tags do not exist in the code yet. This page records what an
experiment is meant to be, so the design can be made precise before it is
built.

## What an experiment is

An **experiment** is a group of artifacts that belong together because they
answer one question: does this architecture beat that one, what does a larger
tokenizer buy, how does the learning rate change the loss. It is identified by
**tags** on its artifacts. An experiment is not an artifact, and it has no
folder of its own.

## Why a label and not a structure

The artifact graph is shared by design. Two experiments that use the same
tokenizer and dataset build them once, and that sharing is the point of
identifying artifacts by their parameters. So an experiment cannot be a
subtree or a folder, because the same artifact belongs to several experiments.
It has to be a label laid over the graph.

```mermaid
flowchart LR
  subgraph graph ["The shared artifact graph"]
    SRC["Source"] --> TOK["Tokenizer"]
    TOK --> TS["TokenizedSource"]
    SRC --> TS
    TS --> DS["DataSet"]
    DS --> A["Leg: transformer"]
    DS --> B["Leg: mamba"]
    DS --> C["Leg: transformer, higher lr"]
  end
  E1(["Experiment: architecture comparison"]) -.-> A
  E1 -.-> B
  E2(["Experiment: learning rate sweep"]) -.-> A
  E2 -.-> C
```

The transformer leg is in both experiments and is trained once.

## What a tag must not do

A tag cannot be part of an artifact's identity. If it were, adding a tag would
change an artifact's path and the sharing above would break. Tags sit beside
the definition the way `declaring_commit` and `allocated_resources` do: recorded, and
excluded from identity.

## Where the closest thing lives today

- **`run_id`.** Training legs carry a `run_id`, and their folders sit under
  `runs/<run_id>/`. This groups the legs of one training run, but it names
  nothing above that level, and it applies to training artifacts only.
- **The dashboard.** The table filters by artifact type, then by any parameter
  or dependency ([Dashboard](dashboard.md#filtering)). It has no notion of a
  group.

## To be decided

- Where tags are stored: inside each artifact's manifest, or in a separate
  index that can change without touching a manifest.
- Whether tags can change after an artifact is declared.
- The shape of a tag: a plain string, or a key and a value.
- Whether tagging an artifact tags what it depends on.
- How the dashboard and the lab select by tag.
