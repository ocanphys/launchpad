
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

## Invariants

- Every artifact has one path, and one definition at that path.
- Dependencies are acyclic.
- A manifest's definition and declaring commit never change once written.
- Completed outputs are never modified in place.
- An artifact is `done` only when every completion file exists at its final
  path.
- A job is launched only when every direct dependency is `done`.
- At most one execution publishes to a path, and a failed execution never
  corrupts completed outputs.
- Declaring or launching an already complete artifact does no work.
- Definitions, jobs, the launcher and storage stay separately understandable:
  resolution reads no storage, status reads no ownership, and binding launches
  nothing.

What these guarantee in practice:

| Case | Result |
| --- | --- |
| Shared dependency reached twice | One resolved entry; conflicting definitions raise before any write. |
| Tokenizer sources supplied in reverse order | One definition, equal and equally hashed: the same path, manifest and training input order. |
| Two sources named alike with different URLs in one set | Refused at construction, not deduplicated. |
| Manifest round trip | The same definition and field types, and the root's recorded fields; the same manifest text; the result unbound. |
| Equal definitions written differently (`1` and `1.0` in a float field) | One artifact, one path, one manifest. |
| Preview against empty storage | A report only; no folders or files created. |
| Redeclare from a later commit, with different resources or more tags | The existing manifest kept; drift on every node declared at an older commit, blocking only in strict mode; resources never rewritten; missing tags added on commit, none removed. |
| Redeclaring an already-published path | The existing manifest is read, not overwritten; a different definition is a conflict. |
| Interrupted declaration | A retry completes the missing manifests without rewriting existing ones. |
| Interrupted output write | Temporary files do not make an artifact complete; partial outputs are reused only after the job validates them. |
| Mapped dataset with unfinished inputs | No owned outputs and no job; completion follows its dependencies' files. |
| Bind a completed tokenizer | A distinct usable object; the original stays unbound. |
| Bind an incomplete or conflicting definition | A descriptive error before any object is returned. |
| Standalone tokenizer or corrupt manifest | The state map finds it; corruption affects its row and the work that depends on it, not unrelated rows. |
| Two launch requests or a stale heartbeat | At most one worker publishes for that path. |
| Two different definitions rendering to one path | They compare unequal; declaration reports a conflict and writes nothing. |
| The whole flow run against an explicit `root`, with no mount present | Declaring, status, binding and a job's own dependency reads all stay under that root; nothing falls back to `STORAGE`. |
| Commit with open files on a Modal mount | The volume-reload error propagates; declaration does not proceed from stale state. |

## Extending

- **A new artifact family** is a package under `artifacts/` with its artifact
  class (fields, path, owned files, optionally a `_load` for what binding reads)
  and its job. The core does not change, and families depend on one another
  only by naming each other's classes as dependency types
  ([Artifacts](artifacts.md#defining-a-family)).
- **A new architecture** is a new `Model` subclass under `models/`; one
  training artifact type trains every architecture ([Models](models.md)).
- **A parameter** may only have a type whose values survive storage exactly,
  which is checked when the type is first used.
- **Code changes** take effect after `modal deploy`, since jobs run the
  deployed code.

## Deliberate limits

- **Identity** is parameters and dependency definitions only. Commit and
  resources are recorded, never identity. External content is not identity
  either: a source behind a URL may change without changing its path, and the
  first completed download is treated as fixed until it is explicitly deleted.
- **Outputs are immutable.** There are no mutable or versioned artifacts.
  Deleting outputs is an explicit operation outside jobs, and invalidates
  nothing downstream: rebuilding one can change its bytes without invalidating
  what was built from it.
- **One producer** per artifact type, or none for a virtual artifact.
- **Nothing is guaranteed to run.** At most one execution publishes to a path,
  and launching and relaunching are by hand.
- **Validation is file presence.** Content is not checked, and the same
  identity does not promise the same bytes: GPU training is not bit
  reproducible, and recorded commits do not make execution reproducible.
- **Graphs are static**: fully declared before anything runs. A job cannot add
  dependencies.
- **One platform, one store**, with no pluggable execution or storage backends.

Not built, each needing its own design when a pipeline needs it: automatic
scheduling, retry policies, content verification, automatic invalidation,
executing historical code, graph-wide transactions, mutable annotations,
garbage collection, and cleanup of the temporary files a hard-killed execution
leaves behind. A change to what is stored is applied to the whole volume at
once, by hand.

## Where to go next

- [Artifacts](artifacts.md): how an artifact is defined and identified.
- [Manifests](manifests.md): how a definition is written to disk and read back.
- [Jobs](jobs.md): the code that builds one artifact.
- [Launcher](launcher.md): how the graph is launched and watched.
- [Repository](repository.md): what lives where.
