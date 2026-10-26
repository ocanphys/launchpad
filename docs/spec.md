# Artifact Framework Backbone Spec

The contract every artifact, job, and the code that declares, resolves, launches and binds them is written against. [README.md](../../README.md) is the map of where things live; [docs/workflows.md](../../docs/workflows.md) traces each flow function by function.

## 1. Purpose

Build ML pipelines by composing immutable artifact definitions and their dependencies. Declare those definitions, run jobs to produce their outputs, and bind completed artifacts for use. Everything runs on Modal, against one mounted volume.

An artifact defines what should exist. Its manifest records the declared definition. The presence of its required completion files determines whether it is complete.

## 2. Artifact definitions

Moved to [Artifacts](artifacts.md).

## 3. Manifests and loading

Every declared artifact owns `root / artifact_path / "manifest.json"`: a self-contained definition, with dependency manifests embedded recursively. Embedding is what lets a manifest be decoded and compared on its own, since definition equality recurses through dependencies. A tokenizer trained on one source:

```json
{
  "artifact": "artifacts.tokenizers.bpe.Tokenizer",
  "commit": "<defining-revision>",
  "allocated_resources": {"cpu": null, "gpu_count": null, "gpu_type": null},
  "parameters": {"special_tokens": ["<pad>"], "vocab_size": 1000},
  "dependencies": {
    "sources": [
      {
        "artifact": "artifacts.sources.SourceURL",
        "commit": "<defining-revision>",
        "allocated_resources": {"cpu": null, "gpu_count": null, "gpu_type": null},
        "parameters": {"name": "odyssey", "url": "https://example.org/odyssey.txt"},
        "dependencies": {}
      }
    ]
  }
}
```

The `"artifact"` key is the concrete class's dotted path. The `parameters` and `dependencies` maps partition that class's constructor fields by annotation, so `sources=(source,)` becomes an array of manifests while `special_tokens`, also an array, stays a parameter. Dependency values are manifests, arrays of manifests, or `null`. Declaration also writes the source's manifest at its own `sources/odyssey/manifest.json`.

Manifest bytes are a pure function of the definition: the five top-level keys in the order above, every nested map sorted by key, so reordering a dataclass's fields changes nothing on disk.

A parameter is encoded and decoded by pydantic against its field's annotation, so the annotation alone rebuilds it. A field type is allowed exactly when that holds, and the field schema (`artifacts/core/schema.py`) checks it when a class is first used: no `Any`, no tuple or set without item types, no mutable containers, no artifact inside a parameter. Dependencies are annotated as artifacts, never hidden in a parameter.

A manifest stores a record's fields and not its class, so the annotation is the schema: a record encodes as plain data, and its class can move or be renamed without migrating manifests. It follows that a record field holds exactly its annotated class, since a subclass would decode as the base, and that a union holds at most one record type, since nothing on disk tells two apart. The one exception is declared by the base class: a record class with `polymorphic = True`, such as `Model`, may stand in a field as any subclass, and that field's manifest carries the subclass's dotted path under `"class"`, beside its fields; `class` is a Python keyword, so no field collides with it. Decoding reads it and rebuilds the record as `locate(class)`. That is how one `Pretraining` class trains every architecture: `"model": {"class": "models.transformer.Transformer", "d_model": 64, ...}`.

**A constructor keeps a value only if its manifest decodes back equal to it, field types included.** Every constructor runs each parameter through the codec, so an int in a float field becomes that float, `-0.0` becomes `0.0`, and records are rebuilt through their constructors; anything that would not come back exactly raises instead of being coerced. Coercion would let an artifact differ from what its manifest decodes to, breaking the `==` that declaration and `bind` depend on; normalizing means two equal artifacts encode, and so hash and land, identically. Decoding is equally strict and refuses unknown and missing fields. [LESSONS.md](../../LESSONS.md) records the pydantic settings this takes.

Conversion is separate from filesystem access:

```python
artifact.to_manifest() -> dict
artifact.parameters() -> dict
Artifact.from_manifest(data: dict, memo=None) -> Artifact
Artifact.load(artifact_path, root=None, memo=None) -> Artifact
artifact.bind(root=None) -> Self
```

`to_manifest` and `from_manifest` touch no storage; the manifest names the concrete class `from_manifest` and `load` build. `load` reads the manifest at `root / artifact_path` and passes it to `from_manifest`. A missing manifest raises `FileNotFoundError`; anything else that keeps it from rebuilding an artifact (unreadable JSON, an unknown class, a field that does not fit) raises `ValueError` naming the folder, so a scan can turn one bad manifest into one bad row. It also rejects a rebuilt artifact whose path differs from the folder: a path is a pure function of the definition, so a mismatch means the folder was copied or renamed.

**`load` and `from_manifest` always return an unbound instance, even when every output exists.** Call `.bind()` before using output-dependent methods.

An artifact is a pure function of its manifest, so decoding is memoizable. `memo` is one dict a caller keeps across calls: each distinct node is constructed once, so a run's legs, each embedding every leg before it, share their subtrees as single instances. Instances are frozen, so sharing them is safe; the one thing a shared instance must not be is bound, which is why `bind` never passes a memo.

A definition at a path never changes once declared, so a reader may decode each path once and keep it; the launcher does, for its whole life. What a redeclaration may change, the resources, is read fresh by launching.

The round-trip contract is definition agreement, plus preservation of recorded commit and resources. Path equality alone is too weak to test serialization.

### Recorded fields

New definitions capture the process's defining revision once per process: the HEAD hash, with `-dirty` when the working tree has uncommitted changes, since what ships is what is on disk; a container without git reads the revision baked into its image. Reconstruction always retains the manifest's recorded value. Mixed revisions within a graph are allowed.

`commit` describes the definition. It neither selects execution code nor proves which code produced the outputs. Execution uses deployed code. Resources configure that execution; an unspecified dimension is recorded as `null`, and Modal supplies its platform default for it.

Once written, a manifest's definition and commit are immutable. A matching redeclaration retains both; only its resources follow the latest committed declaration, since they configure the next execution and nothing else. The manifest at an artifact's own path is authoritative for execution; an embedded copy in a parent's manifest cannot override it. General annotations, mutable labels, and execution histories are outside this manifest schema.

## 4. Storage, status, and binding

Declaration, loading, status, and binding default to one fixed root, where the volume is mounted in a container:

```python
STORAGE = Path("/storage")  # config.py
```

Every storage-touching call takes `root=None`, meaning `STORAGE`. Any other store is reached by passing another absolute `Path`: a second volume mounted elsewhere, a test's `tmp_path`, a folder on a development machine. All are ordinary roots, and nothing distinguishes them.

**Nothing internal calls `bind()` or `load()` without its root.** A job, a `_load` hook, or anything else already holding a root threads it through every call it makes; a hook that bound a dependency bare would silently read `STORAGE` while everything around it reads another store. The default is consumed only at the top: a notebook, the launcher.

**Every root is an absolute `Path`, and every `artifact_path` relative.** Every join depends on that pairing: pathlib's `/` silently discards the left side when the right is absolute, so either mistake would resolve to the wrong file rather than raise. A root given as a string or a relative path raises at the call, so one type reaches every join, glob and `relative_to`.

### Status is a filesystem observation

`artifact.status(root=None)` returns a small footprint: manifest presence, which owned outputs exist, and which required completion files exist. It does not parse manifests, compare definitions, inspect leases, import jobs, or load output contents. `footprint.complete` reads the completion files, never the owned outputs: the two name the same files for every artifact that owns its bytes, and differ exactly where confusing them is silent.

Ordinarily, `completion_paths(root)` is the owned output paths. A virtual artifact such as `MappedDataSet` owns no outputs and has no producer; it names the files it consumes from direct dependencies instead. An empty completion set is satisfied immediately, so a type whose completion is borrowed must reject, in its own parameter validation, a definition that borrows from nothing: it would read as `done` against an empty volume.

Only regular files at final paths count. Temporary files, logs, and nested artifact folders do not. Declaration classifies a footprint as follows:

| Observation | State | Blocks declaration? |
| --- | --- | --- |
| No manifest and no owned outputs | `new` | No |
| No manifest and at least one owned output | `undeclared` | Yes |
| Manifest cannot be decoded or its definition disagrees | `conflict` | Yes |
| Matching manifest; required files absent | `declared` | No |
| Matching manifest; some required files present | `partial` | No |
| Matching manifest; all required files present | `done` | No |

For zero required files, a matching manifest means `done`. A virtual artifact can be `new` even when all its dependency files exist: undeclared-output detection checks ownership, not borrowed completion files.

Commit drift is an additional observation, not a replacement state. Running, failed, and cancelled describe execution attempts, not this footprint. `done` establishes file presence, not byte correctness.

### Binding always returns a usable copy or raises

`artifact.bind(root=None)` reads the stored declaration from the root, checks definition agreement and completion, creates a distinct copy, and invokes its type's `_load(root)` hook. Checking the stored definition prevents binding a conflicting request to somebody else's outputs at the same path. A failed check or loader raises without changing the original. The copy carries the stored declaration's provenance and resources. Running a job by hand follows the same declare, produce, bind sequence.

The default hook has nothing to load. A tokenizer loads vocabulary and merges; a mapped dataset opens its dependency token streams. Hooks read only outputs named by the artifact or its direct dependencies, validate the formats they consume, and may bind those dependencies with the same root. Ordinary binding does not resolve an entire pipeline.

```python
declared = Artifact.load(tokenizer.artifact_path)  # unbound
bound = declared.bind()
bound.encode("hello")                              # declared.encode would raise

assert bound == declared and bound is not declared
```

Every bind builds fresh runtime state; runtime caches and open handles never enter parameters, identity, or manifests. All loading is local to the executing process.

## 5. Pure dependency resolution

`resolve(artifact) -> list[Artifact]` traverses the artifact's direct dependencies depth-first in `deps()` order. It emits dependencies before dependents and each path once; the requested artifact is last.

- A path revisited on the active traversal stack is a cycle and raises.
- A repeated path must have an agreeing definition before it can be deduplicated, so a shared path cannot conceal a conflicting leaf.
- For agreeing duplicates, retain the first encountered object, including its non-identifying recorded fields.

Resolution reads no storage and makes no launch decisions; its traversal state lives within the call.

```python
for artifact in resolve(tokens):
    print(artifact.artifact_path)
# sources/odyssey
# tokenizers/<tokenizer-uid>
# tokenized/<tokenizer-uid>/odyssey
```

The source occurs once even though both the tokenizer and tokenized source reference it.

## 6. `lab`: the notebook-facing API

`lab` is the notebook API, imported with `import lab`, and declaration is its one entry point. Construction, loading, and binding stay on the artifact classes and instances (`Tokenizer(...)`, `Artifact.load(path)`, `tokenizer.bind()`). There is nothing to initialize. Beside `declare`, `lab` exports the two volume verbs a notebook in the lab container needs by hand, `refresh` (see files a job wrote elsewhere) and `save` (make a saved notebook outlive the container), and `worker`, a stand-in execution context for running one job from a cell with no lease and console logging. The `lab.py` module docstring owns the details.

A committed declaration that creates the requested artifact, when it belongs to a run (a `Training`, with a `run_id`), snapshots the declaring notebook as `runs/<run_id>/declare-<uid>.ipynb`. Preview, a redeclaration, and dependencies write no copy.

Declaration establishes the stored plan. The executor runs its jobs separately. Completing a declaration does not produce outputs or bind the supplied instance.

### Declaration

```python
lab.declare(
    artifact,
    *,
    root=STORAGE,
    commit=False,
    strict_commit=False,
    verbose=False,
    notebook=None,
) -> DeclarationReport
```

`verbose` controls only how the result is printed. `root` is for tests and scripts; a notebook gets `STORAGE`. `notebook` is the declaring notebook's bytes, for a caller with no kernel: a laptop declares onto the volume through `local.declare_on_volume`, one explicit remote call that ships them.

1. Resolve the requested artifact into its graph.
2. Inspect each path once and compare existing manifests using definition agreement.
3. Record commit drift and resource differences per graph node. Drift blocks only with `strict_commit=True`; resource differences never block.
4. Collect one row per path with state, drift, and differences. Preview returns these in a report without writes.
5. With `commit=True`, refuse before writing if any row blocks. Otherwise publish manifests for `new` artifacts in dependency order, and rewrite an existing matching manifest whose resources differ with the requested resources, its definition and commit kept. Against the mounted volume, reload before inspecting and commit after writing; this is the one place declaration touches volume mechanics.
6. Return a report recording created and updated manifests and the resulting states, one row per graph path. Reuse the resolved graph; do not resolve again.

Example: `sources/odyssey` and `tokenizers/bpe-1000-feeeeefa90` are declared and built (the tokenizer's manifest recorded an older commit than `HEAD`); the `TokenizedSource` built from them is not yet declared.

```python
>>> report = lab.declare(tokens)   # preview
sources/odyssey                                done
tokenizers/bpe-1000-feeeeefa90                 done   (drift: a1b2c3d -> e4f5a6b)
tokenized/bpe-1000-feeeeefa90/odyssey          new

2 done, 1 new
ok -- 1 to declare

>>> report.rows   # commit and resources are [stored, requested] pairs
[{"path": "sources/odyssey", "state": "done", "drift": False,
  "differences": {}, "created": False, "updated": False, ...},
 {"path": "tokenizers/bpe-1000-feeeeefa90", "state": "done", "drift": True,
  "differences": {}, "created": False, "updated": False, ...},
 {"path": "tokenized/bpe-1000-feeeeefa90/odyssey", "state": "new",
  "drift": False, "differences": {}, "created": False, "updated": False, ...}]

>>> lab.declare(tokens, commit=True)   # same rows; the new one gets published
sources/odyssey                                done
tokenizers/bpe-1000-feeeeefa90                 done   (drift: a1b2c3d -> e4f5a6b)
tokenized/bpe-1000-feeeeefa90/odyssey          done   (created)

3 done
1 manifest written: tokenized/bpe-1000-feeeeefa90/odyssey
```

Requesting the tokenizer again with `allocated_resources=Resources(cpu=4.0)` reports the resource difference in preview and writes it on commit, definition and commit unchanged.

`DeclarationReport` is serializable data: rows include path, state, drift, differences, the stored and requested commit and resources, and whether a manifest was created or updated. Counts and the verdict are derived from the rows. Storage conflicts return a report in preview and raise with that report attached when committing. Structurally invalid input graphs raise before a report exists. Infrastructure failures propagate with the affected operation and path.

Preview never creates folders or temporary files.

### Publication

One writer at a time per root, assumed rather than enforced. Once a manifest reaches disk, it is the truth.

Write each manifest directly to `root / artifact_path / "manifest.json"`, with no temporary file or rename: a manifest is small, written in one call, and written before any job runs, so there is no half-written manifest to mistake for a complete one. Never overwrite an existing manifest; read it and apply the same definition and drift rules as any other declared path. Job outputs are published differently (§7), because a partially written output at a completion path would read as done.

Publication across a graph is not a transaction. Interruption can leave some manifests written and others not; repeating the declaration completes what is missing and leaves what is there untouched.

## 7. State inspection and execution

### One state entry per declared path

`launcher.state.state(root)` maps every declared path under `root` to its entry, over one glob of `manifest.json` files: each manifest loaded once, its footprint read, its direct dependency paths listed, and one snapshot of leases and heartbeats for the whole scan. There is no dependency resolution and no graph walk; a dependency is a path in `depends_on`, and whatever else is true of it lives on its own entry, so no record goes stale when another changes. `parameters` is the artifact's own fields only, never a dependency's manifest. The `state` docstring owns the entry's fields.

`verdict` is the one word a row shows, first match wins: `done`; `running` while the leased call beats; `starting` while a lease has no heartbeat yet and is within its startup grace; `failed` for a manifest that would not load, or a lease whose call stopped beating or outlived its grace; `runnable` when ready; `blocked` otherwise. An artifact failed by its lease stays ready, so it can be run again. Every verdict comes from one snapshot of leases and heartbeats and one view of the volume, with no call-status request to Modal; launching refuses a second call during the same grace.

`blocked_by` is every direct dependency not `done`, missing manifests included; `ready` is not done, not blocked, and produced by something, so a virtual artifact is never ready. A scan has no requested artifact to compare against, so it has no drift, never reports `new` or `undeclared`, and reports `conflict` only for a manifest it cannot decode, as an error entry at its own path that leaves every other path's discovery intact.

A state result is an observation over a scan interval, not a transactional snapshot: rebuild it whole, publish the completed map at once.

### One recompute, taken when something changed

The launcher recomputes the map, reloading the volume, only when something has changed what it would say: a call announced that it started or ended, the launcher granted or released a lease, or the page asked (a manifest declared from the lab is announced by no call). Between recomputes the map is the authority on the volume, lagged by one message but complete, because an artifact's files never leave the volume once they land. The live half of an entry (which call holds it, its last beat, its reported progress, and the verdict those imply) depends on a lease and a heartbeat alone, so it is reread on every request for the calls under way.

A reload and a read on one mount must never overlap, which one lock in the launcher enforces; the [README](../../README.md#readwrite-traffic) and LESSONS.md ("A background thread makes the mount's reload race its readers") have why and how. One container owns the map. Nothing reaches the dashboard from a worker except what the worker committed under its lease, and the log rows and heartbeats it publishes to the Dicts; the launcher container writes nothing to the volume.

### Minimal execution contract

A job is eligible when its declaration and dependency graph are valid, it has a producer, its state is `declared` or `partial`, every direct dependency is `done`, and no execution owns its exact path. Eligibility permits an attempt; the producer must validate any partial state before resuming. Virtual artifacts become complete through their inputs and are never launched.

For an eligible artifact, the executor ([docs/workflows.md](../../docs/workflows.md) §6 and §8 trace each step):

1. Refuses while a call is active or starting on its path.
2. Refreshes storage, checks readiness against the manifest at the path and each direct dependency's manifest and completion files, spawns a worker with resources from that manifest, and records the grant naming that call.
3. In the worker, refreshes storage and confirms the grant names this call, loads the authoritative manifest, and resolves its producer.
4. Calls `Job(artifact).run(root, worker)`. Inputs come from the artifact's direct dependencies; the job binds what it consumes.
5. Confirms the grant again before committing, then commits whatever the outcome, and logs the outcome under the call. Exceptions propagate; a failed attempt is never reported as success.
6. Announces itself when it starts and again when it has ended, each after what it announces, so a launcher that reloaded earlier finds nothing new. A message carries no state; it only says the volume and the Dicts are worth reading again.

Different artifact paths may run concurrently, including nested paths when dependencies permit it.

A job creates only its own outputs. Every path in the artifact's `completion_paths` is fully written and closed at a temporary path in its destination directory, then atomically renamed into place, through `worker.publishing`, the one context manager that confirms the lease between the two. The temporary name is the final filename plus the call id, so neither two owned files nor two calls share one. A job writes completion files only through `publishing`; naming, opening or renaming a temporary path in a job is itself the defect. Final outputs are never streamed into, appended to, replaced, or deleted. Temporary files never satisfy completion. Publication is per file, so a crash between files may leave `partial` state. Jobs may also write auxiliary files in locations their type owns (checkpoints, a training step log); these never count toward completion or overlap nested artifacts.

On `partial` artifacts, a producer verifies that it can reuse existing outputs and checkpoints consistently with the definition before publishing missing outputs. It preserves existing final files and raises if resumption is unsupported. Cleanup and deletion remain explicit operations outside jobs. Temporary-only failures leave no completed output and may be retried.

Execution ownership must prevent two workers from **publishing** to the same path; it cannot prevent two from running. Launch decisions are serialized, and ownership is released either by the call ending or by a cancel, which drops the lease before asking the platform to stop the call, so a second call may be granted the path while the first still writes. The fence is the confirm inside `publishing`: only the lease holder renames, and overlapping calls never write the same temporary bytes. The rename is the step that publishes, since the mount commits in the background whether or not a call reaches its own commit, so a cancelled or superseded call leaves its checkpoints and no finished artifact. A check before publishing cannot unmake a rename already made, so a call also reports how it ended; see LESSONS.md ("A cancel is a request, and it arrives late"). Log handling must never become a path for publishing unfinished outputs.

## 8. Delivery and acceptance

Implement in this order, with a complete source-to-tokenizer-to-tokenized-source example working end to end before connecting execution:

1. Immutable definitions, normalization, manifest round trips, and pure resolution.
2. Declaration, status, and binding.
3. Generic state discovery and dependency validation, including virtual datasets.
4. Execution ownership and atomic output publication.

The backbone is ready when these cases hold:

| Case | Required result |
| --- | --- |
| Shared dependency reached twice | One resolved entry; conflicting definitions raise before writes. |
| Tokenizer sources supplied in reverse order | One definition, equal and equally hashed; same identity, manifest, and training input order. |
| Two sources named alike with different URLs in one set | Rejected at construction, not deduplicated. |
| Manifest round trip | Same definition, field types and recorded fields; same manifest bytes; reconstructed value is unbound. |
| Equal definitions written differently (`1` and `1.0` in a float field) | One artifact, one path, one manifest. |
| Preview against empty storage | Report only; no directories or files created. |
| Redeclare with different commit or resources | Existing manifest retained; commit drift blocks only in strict mode. |
| Redeclaring an already-published path | Existing manifest is read, not overwritten; incompatible definitions raise as a conflict. |
| Interrupted declaration | Retry completes missing manifests without rewriting existing ones. |
| Interrupted output write | Temporary files do not make an artifact complete; partial outputs are reused only after producer validation. |
| Mapped dataset with unfinished inputs | No owned outputs and no producer; completion follows dependency files. |
| Bind a completed tokenizer | Distinct usable object; original remains unbound. |
| Bind an incomplete or conflicting definition | Descriptive error before returning an object. |
| Standalone tokenizer or corrupt manifest | Generic discovery finds it; corruption affects its row and dependent work, not unrelated rows. |
| Two launch requests or a stale heartbeat | At most one worker may publish for that path. |
| Two different definitions rendering to one path | They compare unequal; declaration reports a conflict and writes nothing. |
| Whole flow run against an explicit `root`, with no mount present | Declaring, status, binding, and a job's own dependency reads all stay under that root; nothing falls back to `STORAGE`. |
| Commit with open files on a Modal mount | The volume-reload error propagates; declaration does not proceed from stale state. |

## 9. Deliberate limits

Identity describes declared inputs, not output bytes. A source URL may change without changing its path. The first completed download is treated as fixed until explicit deletion; deleting and rebuilding it can change bytes without invalidating downstream artifacts. Recorded commits do not make execution reproducible.

The volume is reachable only from Modal. A laptop works against a folder of its own as an ordinary root, and declares onto the volume only through `local.declare_on_volume`, never through a default root that varies per process.

Content verification, automatic invalidation, historical-code execution, graph-wide transactions, general retry policies, mutable annotations, multiple producers, and garbage collection require separate designs. Schema changes are migrated explicitly, one script per change. Add the rest when a concrete pipeline needs them; the backbone keeps declaration, observation, execution, and use independently understandable.
