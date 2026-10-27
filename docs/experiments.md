# Experiments

*Draft.* Tags exist on every artifact (`tags`, a dict of `str` to `list[str]`),
and the dashboard filters by them. Experiments as a thing in their own right do
not: this page records what one is meant to be.

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
the definition the way `declaring_commit` and `allocated_resources` do: recorded at
the root of the artifact's manifest, and excluded from identity
([Manifests](manifests.md#recorded-fields)).

## Where the closest thing lives today

- **`run_id`.** Training legs carry a `run_id`, and their folders sit under
  `runs/<run_id>/`. This groups the legs of one training run, but it names
  nothing above that level, and it applies to training artifacts only.
- **The dashboard.** The table filters by tag, by artifact type, then by any
  parameter or dependency ([Dashboard](dashboard.md#filtering)). It has no
  notion of a group beyond a tag value.

## To be decided

- Whether tagging an artifact tags what it depends on. Today it does not: a
  tag labels only the artifact that carries it.
- How the lab selects by tag.
