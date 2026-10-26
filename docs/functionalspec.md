# Launchpad: Functional Specification

Launchpad lets a user define, declare, build, monitor and use pipelines made of
interdependent artifacts. It is built for ML pipelines (sources, tokenizers,
datasets, training runs), but its core assumes nothing about what an artifact
contains.

The user describes what should exist. Launchpad works out what that depends on,
records the plan in storage as manifests, runs one job per artifact when the
user launches it, and shows the state of every artifact on a dashboard.

## 1. Artifacts

- An artifact is an immutable definition of something that should exist: a
  type and its typed parameters.
- An artifact may own any number of output files, or none.
- Parameters are every input that determines the result. A parameter whose type
  is an artifact, or a collection of artifacts, is a dependency. Whether a
  parameter is a dependency is decided by its declared type, never by its value.
- Every artifact has a storage path derived from its type and parameters. The
  path is its identity: deduplication, ownership and manifest locations key on
  it.
- Two artifacts are the same definition when their type, parameters and
  dependency definitions agree recursively. A shared path is never evidence of a
  shared definition: two definitions at one path are a conflict, never merged.
- Changing a parameter or any upstream definition yields a different
  definition, and normally a different path, so it declares a new artifact.
  The old one remains.
- The defining code revision and the requested compute resources are recorded
  with every artifact and excluded from identity and equality.
- Parameter values are normalized when an artifact is defined: a value is kept
  only if its stored form reads back equal to it, so `1` and `1.0` for a
  real-valued parameter are one artifact, and a value that would not survive
  storage is rejected.
- A dependency collection is either ordered, where order and repeats are part of
  the definition, or a set, where neither is. A set with two members at one path
  is rejected.
- Defining an artifact reads no storage and does not require the artifact to
  exist.
- A virtual artifact owns no files and has no job. It is complete when the
  dependency files it reads exist, and it is never launched.

## 2. Jobs

- Each artifact type has at most one job that produces it. Each job produces
  exactly one artifact type. There are no alternative producers.
- A job is given its artifact and nothing else; its inputs are that artifact's
  direct dependencies. They are derived, never declared a second time.
- A job writes only inside its artifact's location.
- Every file that marks an artifact complete is written whole under a temporary
  name, then published by an atomic rename, and only while the job still owns
  the artifact.
- A job never appends to, replaces or deletes a published output. It may write
  auxiliary files in its own location (checkpoints, a step log); these never
  count toward completion.
- Jobs execute the currently deployed code. The recorded code revision does not
  select which code runs.
- Defining, resolving and inspecting artifacts never loads job code.
- A job can be run by hand, outside the launcher, with no ownership and console
  logging.

## 3. Dependencies and resolution

- Resolving a request returns every artifact it needs, dependencies before
  dependents, each path once. Resolution reads no storage.
- A cycle is rejected.
- A repeated path with a disagreeing definition is rejected. Agreeing duplicates
  keep the first one encountered, recorded fields included.
- An artifact shared by several requests or runs is one path, declared once and
  built once.
- An artifact's manifest embeds its dependencies' manifests recursively, so its
  whole dependency graph can be rebuilt from that one manifest.

## 4. Manifests and provenance

- Every declared artifact owns a manifest at its path, holding its type,
  recorded code revision, requested resources, parameters and dependencies.
- A manifest's contents are a pure function of the definition.
- A manifest reads back on its own to an equal artifact, without loading any
  outputs. Reading rejects a manifest whose definition does not belong at the
  location it was found, and reports an unreadable one naming its location.
- Once written, a manifest's definition and code revision never change.
  Redeclaring may update its requested resources and nothing else.
- The recorded code revision marks whether the working tree had uncommitted
  changes. Revisions may be mixed within one graph.
- Declaring a new training run keeps a copy of the notebook that declared it.
- Every execution ever granted for an artifact is recorded, each with its logs.

## 5. Declaration

- A declaration previews by default: it resolves the request, inspects each path
  once against storage, and returns one report row per path. A preview writes
  nothing.
- Each row has a state:

  | State | Meaning | Blocks commit? |
  | --- | --- | --- |
  | `new` | No manifest, no owned outputs | No |
  | `undeclared` | Owned outputs exist without a manifest | Yes |
  | `conflict` | The stored manifest disagrees or cannot be read | Yes |
  | `declared` | Matching manifest, no completion files | No |
  | `partial` | Matching manifest, some completion files | No |
  | `done` | Matching manifest, every completion file | No |

- Drift in the recorded code revision is reported, and blocks only in strict
  mode. Resource differences never block.
- Committing a declaration refuses the whole request if any row blocks.
  Otherwise it writes manifests for `new` artifacts in dependency order and
  updates the requested resources of matching manifests whose resources differ.
- An existing manifest is never overwritten with a different definition.
- Declaration is not a transaction. An interrupted one is completed by repeating
  it; what was already written is left alone.
- Declaration runs no jobs.

## 6. Status and state

- An artifact's status is a storage observation: whether its manifest exists,
  which owned outputs exist, which completion files exist. An artifact is
  complete when every completion file exists. Only finished files at their final
  paths count; temporary files never do.
- `done` means the files are present. Their bytes are not verified.
- The launcher keeps one state view with an entry per declared artifact. Each
  entry has a verdict, first match wins:

  | Verdict | Meaning |
  | --- | --- |
  | `done` | Complete |
  | `running` | Its execution is alive |
  | `starting` | An execution was granted and has not yet reported in, within its startup grace |
  | `failed` | The manifest cannot be read, or its execution stopped reporting or never started |
  | `runnable` | Has a job, and every direct dependency is done |
  | `blocked` | Otherwise; the entry lists the dependencies not done |

- A manifest that cannot be read is an error row at its own path and affects no
  other row.
- The state view is recomputed when something changed: an execution started or
  ended, a launch or cancel was accepted, or the user asked for a refresh. A
  manifest declared outside the launcher appears after a refresh.
- A running execution's liveness and progress update between recomputes.

## 7. Launching and execution

- Launching is per artifact and by hand. Nothing launches a dependency
  automatically: the user launches dependencies first.
- A launch is refused unless the artifact has a job, is not complete, every
  direct dependency is declared and complete, and no execution is running or
  starting on its path.
- An accepted launch starts one execution with the artifact's requested
  resources and grants it ownership of the path. It returns without waiting for
  the job. A resource left unspecified takes the platform default.
- Different paths run concurrently, including a path nested under another's
  location.
- An execution confirms its ownership, reads the manifest at its own path
  (authoritative over any copy embedded in a parent), runs the job, persists its
  writes whatever the outcome, and reports how it ended: done, failed, or lost
  ownership.
- A failed attempt is never reported as success. Errors are logged under the
  execution.

## 8. Concurrency and ownership

- Each artifact path has at most one owner, the execution currently granted it.
  Ownership is per exact path; owning a location grants nothing beneath it.
- Launch decisions are serialized.
- Ownership prevents two executions from publishing to one path, not from
  running at once. Only the owner publishes a file, and concurrent executions
  never share a temporary file.
- Liveness comes from heartbeats. An execution with no heartbeat past its
  startup grace, or whose heartbeat goes stale, is `failed`, and its path can be
  launched again.
- Cancelling revokes ownership first, then asks for the execution to stop. The
  execution's next ownership check fails, so it cannot complete the artifact even
  while it keeps running.
- Declaration assumes one writer per storage root at a time; this is not
  enforced.

## 9. Failure and recovery

- A failed, cancelled or superseded execution leaves completed files untouched,
  and the artifact stays launchable.
- There are no automatic retries. The user relaunches.
- Publication is per file, so an interrupted job may leave `partial` state. A job
  validates partial state before reusing it, and fails if it cannot resume.
- A training run writes checkpoints at a fixed step interval. A relaunch resumes
  from the newest resumable one. Batches are a function of the seed and the step,
  so a resumed attempt sees the batches the interrupted one would have.
- A finished training leg is never retrained: once its output exists, its job
  returns without touching anything.
- Every attempt's per-step training metrics are appended to one log, tagged with
  the attempt, so continuous executions can be told apart.

## 10. Using results

- Reading a declared artifact back from storage returns it unbound, whether or
  not it is built.
- Binding an artifact checks that the stored manifest agrees with it and that it
  is complete, then returns a distinct copy with its outputs loaded. It fails
  otherwise, and the original stays unbound.
- Binding loads only what the artifact and its direct dependencies name; it never
  resolves a whole pipeline.

## 11. Monitoring

- The dashboard has one row per declared artifact (verdict, progress, launch or
  cancel), filters by type and status, and a live launcher log.
- An artifact's page shows its parameters, links to its dependencies, training
  curves where it has them, and the logs of every execution ever granted for it.
- Logs are attributed to the job, the launcher, or the surrounding environment.
  They are visible live and persisted alongside the artifact.
- Progress is live while an execution runs, and durable (read from storage, such
  as a training leg's furthest checkpoint) after it ends.
- The dashboard is password-protected and links to a notebook environment with
  access to the same storage.

## 12. Storage

- One store is the source of truth. Every other reader works from a snapshot of
  it.
- Every operation that touches storage accepts a storage root, defaulting to the
  shared store. Any other location works as a root, such as a test directory or
  a local folder. Nothing that holds a root falls back to the default.
- Artifacts not scoped to a run are shared and reused across runs. Run-scoped
  artifacts are grouped under their run.
- Path encodings are a storage contract. Changing one requires an explicit
  migration.

## 13. Extending

- A new artifact type is defined with its parameters, path, owned files, its job
  and, optionally, how to load its outputs when bound. The core does not change.
- Artifact types depend on one another only by naming each other as dependency
  types.
- A parameter may only have a type whose values survive storage exactly. This is
  checked when the type is first used.
- A new model architecture is a new model definition. One training artifact type
  trains every architecture.
- Jobs run the deployed code, so a code change takes effect after redeploying.

## 14. Invariants

- Every artifact has one path, and one definition at that path.
- Dependencies are acyclic.
- A manifest's definition and code revision never change once written.
- Completed outputs are never modified in place.
- An artifact is `done` only when every completion file exists at its final
  path.
- A job is launched only when every direct dependency is `done`.
- At most one execution publishes to a path.
- Failed executions never corrupt completed outputs.
- Declaring or launching an already complete artifact does no work.
- Definitions, jobs, the launcher and storage stay separately understandable:
  resolution reads no storage, status reads no ownership, and binding launches
  nothing.

## 15. Decisions and limits

1. **Identity** is parameters and dependency definitions only. Code revision and
   resources are recorded, never identity. External content is not identity: a
   source behind a URL may change without changing its path, and the first
   completed download is treated as fixed.
2. **Immutability** applies to every output. There are no mutable or versioned
   artifacts. Deleting outputs is an explicit operation outside jobs, and
   invalidates nothing downstream.
3. **Producers**: exactly one job per artifact type, or none for a virtual
   artifact.
4. **Execution guarantees**: at most one execution publishes to a path. Nothing
   is guaranteed to run: launching and relaunching are by hand.
5. **Validation** is file presence. Content is not checked, and identical
   identity does not promise identical bytes (GPU training is not bit
   reproducible).
6. **Graphs are static**: fully declared before anything runs. A job cannot add
   dependencies.
7. **One platform, one store.** There are no pluggable execution or storage
   backends.

Not built: automatic scheduling, retry policies, content verification,
automatic invalidation, executing historical code, graph-wide transactions,
garbage collection, and cleanup of temporary files a hard-killed execution leaves
behind. Each needs its own design when a pipeline needs it.
