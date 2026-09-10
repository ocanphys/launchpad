# Artifact Framework Backbone Spec: Astra Version

Experimental target design, based on the [original backbone spec](artifact_framework_backbone_spec.md). This specifies the intended contract; implementation may differ today.

## 1. Purpose

Build ML pipelines by composing immutable artifact definitions and their dependencies. Declare those definitions, run jobs to produce their outputs, and bind completed artifacts for use. Everything runs on Modal, against one mounted volume.

An artifact defines what should exist. Its manifest records the declared definition. The presence of its required completion files determines whether it is complete.

## 2. Artifact definitions

The notation distinguishes classes from instances throughout this spec:

| Notation | Meaning |
| --- | --- |
| `Artifact` | The abstract base class. Never instantiated directly. |
| `Source`, `Tokenizer`, `TokenizedSource` | Concrete classes that inherit from `Artifact`. |
| `artifact`, `source`, `tokenizer`, `tokens` | Variables holding instances of concrete artifact classes. |
| `bound` | A bound instance of the same concrete class as its unbound definition. |

An artifact instance is an immutable definition with typed constructor parameters. Calling a concrete class constructs one:

```python
source = Source(name="odyssey", url="https://example.org/odyssey.txt")
tokenizer = Tokenizer(
    vocab_size=1000,
    special_tokens=("<pad>",),
    sources=(source,),
)
tokens = TokenizedSource(tokenizer=tokenizer, source=source)
```

A concrete type defines:

| Member | Contract |
| --- | --- |
| Parameters | Every declared input that determines the result, including artifact-valued inputs. |
| `uid` | A type-specific label derived from parameters. |
| `artifact_path` | The unique path relative to storage, derived from parameters, usually through `uid`. |
| `files` | A mapping from output names to required completion filenames owned by this artifact. |
| `producer` | A class-level dotted path to its `Job`, or `None` for an artifact that writes no outputs. |
| `commit` | Recorded definition provenance, excluded from identity. |
| `allocated_resources` | Recorded execution allocation, excluded from identity. |

`deps()` returns only direct artifact-valued parameters. Support individual artifacts, optional artifacts, and tuples of artifacts. Parameter annotations distinguish empty dependency tuples from ordinary empty tuples. Other nested configuration contains ordinary values, never hidden artifact dependencies.

Paths, files, producers, and dependency lists are derived. They are not constructor arguments or separately persisted state. Artifact modules remain lightweight; resolving a producer imports its implementation only when executing it.

### Identity and agreement

Keep three questions distinct:

1. **Same identity:** two artifacts compute the same `artifact_path`. Identity is always written as `artifact.artifact_path`, never as `==`. It is what deduplication, lease keys, manifest locations, and every map of artifacts key on.
2. **Same definition:** concrete type, normalized parameters, and dependency definitions agree recursively. Commit, resources, and bound state are excluded at every node. **This is what `==` and `hash()` mean.** The frozen dataclass generates both: `commit` and `allocated_resources` are declared `compare=False`, and bound state is set outside the declared fields, so all three exclusions hold with no hand-written comparison. Binding does not change equality.
3. **Same provenance or allocation:** the recorded non-identifying fields agree.

Declaration and resolution check definition agreement with `==`, and always between two artifacts already known to share a path — a manifest read from that path, or a second route reaching it during traversal. Sharing a path is therefore never evidence of agreement: two sources named `odyssey` with different URLs land on one path, compare unequal, and must conflict.

Normalization happens at construction and reconstruction. A tokenizer's `sources` are ordered canonically by artifact path because their supplied order is non-identifying. Repeated source paths are rejected. Order remains identifying where it changes the result, such as a mapped dataset's concatenated source streams. Path calculation, definition comparison, serialization, and traversal all use these normalized values.

Parameters are immutable, including nested configuration. A result-changing setting belongs among the parameters even if it also affects execution. Resource allocation never silently changes scientific configuration.

### Location and ownership

`artifact_path` is a nonempty, normalized relative path. It cannot escape storage. Owned filenames are unique bare filenames, excluding `manifest.json` and the reserved temporary-file namespace. This keeps nested artifacts independent:

```text
root/
  sources/odyssey/
    manifest.json
    body.txt
  tokenizers/<tokenizer-uid>/
    manifest.json
    tokenizer.json
    bin/odyssey/
      manifest.json
      tokens.bin
```

`TokenizedSource.artifact_path` is `tokenizer.artifact_path / "bin" / source.uid`. A parent's outputs never include its children's files. Leases use exact artifact paths; owning a directory does not grant ownership of artifacts beneath it.

Path encodings are a storage compatibility contract. Changing them requires an explicit migration. Readable names and short hashes may collide; consistency checks detect conflicting definitions at a shared path rather than assuming the encoding is collision-free.

## 3. Manifests and loading

Every declared artifact owns `root / artifact_path / "manifest.json"`. Its manifest is a self-contained definition, with dependency manifests embedded recursively. **The manifest when loaded to Python is a dictionary:**

```json
{
  "artifact": "artifacts.sources.Source",
  "commit": "<defining-revision>",
  "allocated_resources": {},
  "parameters": {
    "name": "odyssey",
    "url": "https://example.org/odyssey.txt"
  },
  "dependencies": {}
}
```

A tokenizer trained on that source embeds its full manifest under `dependencies.sources`:

```json
{
  "artifact": "artifacts.tokenizers.bpe.Tokenizer",
  "commit": "<defining-revision>",
  "allocated_resources": {},
  "parameters": {
    "special_tokens": [
      "<pad>"
    ],
    "vocab_size": 1000
  },
  "dependencies": {
    "sources": [
      {
        "artifact": "artifacts.sources.Source",
        "commit": "<defining-revision>",
        "allocated_resources": {},
        "parameters": {
          "name": "odyssey",
          "url": "https://example.org/odyssey.txt"
        },
        "dependencies": {}
      }
    ]
  }
}
```

Here, `sources=(source,)` becomes an array of embedded manifests. `special_tokens` remains an ordinary parameter even though it also encodes as an array. Declaration also writes the source's manifest at its own `sources/odyssey/manifest.json` path.

The manifest's `"artifact"` key contains the concrete class's dotted path as a string. It is a schema field, distinct from the Python variable `artifact` used for an instance. The `parameters` and `dependencies` maps partition that class's constructor fields. Dependency values are manifests, arrays of manifests, or `null`, according to their annotations. Reconstruction merges both maps and applies the same validation and normalization as construction.

Use one canonical JSON encoding: UTF-8, two-space indentation, a trailing newline, the five top-level keys in the order above, and sorted keys in ordinary nested maps. Embedded manifests repeat the five-key order. Tuples encode as arrays; typed configuration records encode as maps and reconstruct from annotations. Reject unsupported values and non-finite numbers. Field reordering must not change manifest bytes.

The in-memory manifest is a dictionary. `manifest.json` is its JSON representation on disk. Separate conversion from filesystem access:

```python
artifact.to_manifest() -> dict
Artifact.from_manifest(data: dict) -> Artifact
Artifact.load(artifact_path, root=None) -> Artifact
artifact.bind(root=None) -> Self
```

`artifact.to_manifest()` and `artifact.bind()` are instance methods: they operate on an existing object. `Artifact.from_manifest(...)` and `Artifact.load(...)` are class-level entry points: the manifest identifies which concrete class to instantiate. Their `-> Artifact` annotation means a concrete subclass instance, never the abstract class itself. `-> Self` means an instance of the receiver's concrete class.

Omitting `root` uses `STORAGE`; an explicit root overrides it for that call. Normal calls are `artifact.bind()` and `Artifact.load(artifact_path)`.

`to_manifest` and `from_manifest` convert between an artifact and a dictionary without storage access. `load` reads `root / artifact_path / "manifest.json"`, parses JSON into a dictionary, and passes it to `from_manifest`. It rejects a reconstructed path that differs from the requested folder. Invalid schemas, unknown types, and missing or unreadable manifests raise descriptive errors. This keeps decoding in one place and gives filesystem access its own name.

**`Artifact.load(...)` always returns an unbound instance, even when all outputs exist.** It never loads output state or calls the binding hook. Call `.bind()` explicitly on the returned instance before using output-dependent methods. `Artifact.from_manifest(...)` also always returns an unbound instance.

The round-trip contract is definition agreement, plus preservation of recorded commit and resources. Path equality alone is too weak to test serialization.

### Recorded fields

New definitions capture the process's defining revision lazily: local HEAD, or the revision supplied with deployed code. A process may cache this stamp; an unavailable revision is recorded explicitly as unknown. Reconstruction always retains the manifest's recorded value. Mixed revisions within a graph are allowed.

`commit` describes the definition. It neither selects execution code nor proves which code produced the outputs. Execution uses deployed code. Resources configure that execution; unspecified dimensions are omitted so Modal supplies its platform defaults.

Once written, a manifest is immutable. A matching redeclaration retains its existing commit and resources. The manifest at an artifact's own path is authoritative for execution; an embedded copy in a parent's manifest cannot override it. General annotations, mutable labels, and execution histories are outside this manifest schema.

## 4. Storage, status, and binding

Declaration, loading, status, and binding all read one fixed path:

```python
STORAGE = Path("/storage")  # config.py -- where the volume is mounted in the container
```

`STORAGE` is a plain constant, not configuration: there is no per-session
setup, no environment detection, and no second root that varies by process.

Every storage-touching call takes `root=None` and resolves it to `STORAGE`
when omitted. **Passing an explicit absolute path in its place is how you
reach any other store** — a second volume mounted elsewhere in the container,
a `tmp_path` in a test, a working folder on a development machine. All of
these are ordinary roots; the code does not distinguish them and nothing
detects which kind it was handed. This is not a corner case, it is the whole
mechanism, and it is why no local mount ever has to be arranged for.

That leaves one rule, which the whole arrangement depends on: **nothing
internal may call `bind()` or `load()` bare.** A job, a `_load` hook, or
anything else already holding a root threads it through every call it makes.
A hook that binds a dependency with no argument would silently reach for
`STORAGE` while everything around it reads a different store — no error,
just the wrong file or a missing one. `root` is therefore a parameter on
every internal call, and the `STORAGE` default is consumed only at the top:
a notebook in the lab container, or the launcher.

**`STORAGE` and every `root` are always absolute. `artifact_path` is always
relative (§2).** Every join in this contract (`root / artifact_path`, §3,
§6, §7) depends on that pairing — pathlib's `/` silently discards the left
side if the right side is itself absolute, so a relative root or an
absolute `artifact_path` would not raise, it would resolve to the wrong
file. Nothing else in this spec makes storage location a matter of the
current working directory.

### Status is a filesystem observation

`artifact.status(root=None)` reads the given root, defaulting to `STORAGE`, and returns a small footprint: manifest presence, which owned outputs exist, and which required completion files exist. It does not parse manifests, compare definitions, inspect leases, import jobs, or load output contents.

Ordinarily, `completion_paths(root)` is the set of owned output paths. A virtual artifact such as `MappedDataSet` owns no outputs and overrides this to name the files it consumes from direct dependencies. It has no producer; completion follows those files. An empty completion set is satisfied immediately, subject to the type's parameter validation.

Only regular files at final paths count. Temporary files, logs, and nested artifact folders do not count. Declaration classifies a footprint as follows:

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

`artifact.bind(root=None)` reads the stored declaration from the given root, defaulting to `STORAGE`, checks definition agreement and completion, creates a distinct copy, and invokes its type's `_load(root)` hook. Checking the stored definition prevents binding a conflicting request to somebody else's outputs at the same path. A failed check or loader raises without changing the original. The copy retains the stored declaration's provenance and resources. Running a job by hand follows the same declare, produce, bind sequence.

The default hook has nothing to load. A tokenizer loads vocabulary and merges; a mapped dataset opens its dependency token streams. Hooks read only outputs named by the artifact or its direct dependencies and validate the formats they consume. They may bind those dependencies as needed, using the same root. Ordinary binding does not resolve an entire pipeline.

```python
declared = Artifact.load(tokenizer.artifact_path)  # always unbound
# declared.encode("hello") raises, even if tokenizer.json exists.
bound = declared.bind()  # explicit binding is required
bound.encode("hello")

assert bound == declared
assert bound is not declared
# declared remains unbound.
```

An unbound artifact's output-dependent methods raise. Every bind builds fresh runtime state; runtime caches and open handles never enter parameters, identity, or manifests. All loading is local to the executing process. There is no remote binding target or conditional class-level `bind`.

## 5. Pure dependency resolution

```python
resolve(artifact: Artifact) -> list[Artifact]
```

Traverse direct dependencies depth-first in deterministic field-name order and normalized tuple order. Emit dependencies before dependents, include the requested artifact last, and emit each path once.

- A path revisited on the active traversal stack is a cycle and raises.
- A repeated path must have an agreeing definition before it can be deduplicated. Validate repeated dependency definitions too, so a shared path cannot conceal a conflicting leaf.
- For agreeing duplicates, retain the first encountered object, including its non-identifying recorded fields.

Resolution reads no storage and makes no launch decisions. Keep visited nodes and traversal state within the call. A persistent topology cache is unnecessary for this backbone.

```python
artifacts = resolve(tokens)  # [source, tokenizer, tokens]: artifact instances
for artifact in artifacts:
    print(artifact.artifact_path)

# Displayed paths only; <tokenizer-uid> stands for the computed UID:
# sources/odyssey
# tokenizers/<tokenizer-uid>
# tokenizers/<tokenizer-uid>/bin/odyssey
```

The source occurs once even though both the tokenizer and tokenized source reference it.

## 6. `lab`: the notebook-facing API

`lab` is the notebook API, imported with `import lab`. It has one entry point: declaration. Construction, loading, and binding stay on the artifact classes and instances — `Tokenizer(...)`, `Artifact.load(path)`, `tokenizer.bind()` — and all read `STORAGE` directly (§4). There is nothing to initialize.

```python
import lab

report = lab.declare(tokenizer)               # preview
report = lab.declare(tokenizer, commit=True)   # declare

# After jobs have produced the outputs:
bound = tokenizer.bind()
```

Declaration establishes the stored plan. The executor runs its jobs separately. Completing a declaration does not produce outputs or bind the supplied instance.

### Declaration

There is one `declare` function, defined in `lab.py`. No separate core module,
no wrapper calling through to it:

```python
lab.declare(
    artifact,
    *,
    root=STORAGE,
    commit=False,
    strict_commit=False,
    verbose=False,
) -> DeclarationReport
```

`verbose` controls how the result is printed; it does not change resolution,
inspection, or writes. `root` is for tests and manual scripts — a notebook
never passes it, and gets `STORAGE`.

The algorithm performs one operation against its root:

1. Resolve the requested graph.
2. Inspect each path once and compare existing manifests using definition agreement.
3. Record commit drift per retained graph node. It blocks only with `strict_commit=True`. Report resource differences without blocking; stored resources continue to apply.
4. Collect one row per path with state, drift, and differences. Preview returns these observations in a report without writes.
5. With `commit=True`, refuse before writing if any row blocks. Otherwise publish manifests for `new` artifacts in dependency order, retaining every existing matching manifest. Reload the volume before inspection and commit successful writes before returning — the one place this contract touches volume mechanics, internal to this function, not exposed as a separate call.
6. Return a report recording created manifests and the resulting observed states. Reuse the resolved artifact list and recheck changed paths as needed; do not resolve the graph again.

Example: `sources/odyssey` and `tokenizers/bpe-1.0k-feeeeefa90` are already
declared and built (the tokenizer's manifest recorded an older commit than
`HEAD`); the `TokenizedSource` built from them is not yet declared.

```python
>>> report = lab.declare(tokens)   # preview -- commit defaults to False
sources/odyssey                                      done
tokenizers/bpe-1.0k-feeeeefa90                        done   (drift: a1b2c3d -> e4f5a6b)
tokenizers/bpe-1.0k-feeeeefa90/bin/odyssey            new

2 done, 1 new
ok -- 1 to declare

>>> report.rows
[{"path": "sources/odyssey", "state": "done", "drift": False,
  "differences": {}, "created": False},
 {"path": "tokenizers/bpe-1.0k-feeeeefa90", "state": "done", "drift": True,
  "differences": {}, "created": False},
 {"path": "tokenizers/bpe-1.0k-feeeeefa90/bin/odyssey", "state": "new",
  "drift": False, "differences": {}, "created": False}]

>>> lab.declare(tokens, commit=True)   # same rows; the new one gets published
sources/odyssey                                      done
tokenizers/bpe-1.0k-feeeeefa90                        done   (drift: a1b2c3d -> e4f5a6b)
tokenizers/bpe-1.0k-feeeeefa90/bin/odyssey            done   (created)

1 manifest published: tokenizers/bpe-1.0k-feeeeefa90/bin/odyssey
```

`drift` alone never blocks (only `strict_commit=True` would); `verbose=False`
still shows drift, since it isn't a blocker reason, only `conflict` and
`undeclared` rows hide their detail without `verbose`.

`DeclarationReport` is serializable data: rows include path, state, drift, differences, and whether a manifest was created. Counts and the verdict are derived from the rows. `declare` prints a concise summary; `verbose` expands details. Storage conflicts return a report in preview and raise with that report attached when committing. Structurally invalid input graphs raise before a report exists. Infrastructure failures propagate with the affected operation and path.

Preview never creates folders or temporary files. Definition and completion checks remain part of the artifact contract.

### Implementation plan

Five stages, each testable before the next.

1. **`artifacts/core/artifact.py` — the definition contract.** Normalize artifact-valued tuples in `__post_init__`, rejecting repeated paths. Rename `manifest()` to `to_manifest()` and make the encoder canonical (trailing newline, `allow_nan=False`). Change `load(path)` to `load(artifact_path, root=None)`, absorbing the path-agreement check. Delete `Artifact.at` and `Artifact.declare`. Add `status(root=None)`, folding in `exists()`. Give `bind()` the stored-declaration read and definition check it currently lacks. Allow `producer = None`. Leave `__eq__`/`__hash__` exactly as they are — the generated definition comparison is the contract (§2).
2. **`artifacts/core/resolve.py` — shrink to one pure function.** `resolve(artifact) -> list[Artifact]`, with definition agreement checked on every repeated path. Delete `Node`, `Dag`, `_check`, `plan`, and `declared`; move `declare(dag)` and `conflict_diff` out.
3. **`lab.py` — one declaration operation.** `lab.declare` holds what `_check` and `declare(dag)` used to do: resolve once, inspect all paths, reject blockers, publish missing manifests in dependency order, return a `DeclarationReport`. Delete `environment`, `target`, `_PERMISSIONS`, `init`, `_root_for`, `bind`, and `plan`.
4. **`state()` — one glob, one entry per path.** Replace `read_state`'s three-prefix discovery and its `resolve` over every root. Keep the per-artifact shape, the lease and heartbeat snapshot, and both progress kinds; drop only the runs/sources/datasets grouping.
5. **Callers.** `main.py`: delete the deployed `declare` function, update `run_job`, `declared_artifact`, and `attempt_launch`. Delete `artifacts/mappeddataset/jobs.py`. Update the visualizer, which imports `Dag` and `Status`, and the tests, which import `Node`, `conflict_diff`, and `declare`. Then the notebooks.

The final API has one `declare` function, one `resolve` function, one `state` function, and no session configuration.

### Publication

One writer at a time per root -- assumed, not enforced. Once a manifest
reaches disk, it is the truth.

Write each manifest directly to `root / artifact_path / "manifest.json"`. No
temporary file, no rename: a manifest is small, written in one call, and
declaration writes it before any job runs, so there is no half-written
manifest to mistake for a complete one. Never overwrite an existing manifest
-- if one's already there, read it and apply the same definition/drift rules
as any other declared path.

Job *outputs* are published differently, through a temporary file and a
rename (§7). That protects against a crashed or failed job leaving a
partially written file at a completion path, where its mere presence would
read as done. Manifests have no such exposure.

Publication across a graph is not a transaction. Interruption can leave
some manifests written and others not; repeating the declaration completes
what's missing and leaves what's already there untouched.

## 7. State inspection and execution

### One state entry per declared path

```python
state(root=STORAGE) -> dict[Path, dict]
```

Replaces `main.py`'s `read_state()`. Computes the current state of every
artifact under `root` over one glob -- no dependency resolution, no DAG walk:

1. `glob(root/**/manifest.json)` -- every declared artifact, by folder.
2. Per manifest, once: `Artifact.from_manifest(...)`, then `artifact.status(root)`.
3. Per artifact, `deps()` for its direct dependency paths -- one level, every
   path already a key in this map, no traversal.
4. One lease and heartbeat snapshot for the whole scan.

Each entry keeps the shape the dashboard reads today:

```python
{"type": "Tokenizer", "status": "done", "drift": False,
 "depends_on": ["sources/odyssey"], "blocked_by": [], "ready": True,
 "call_id": None, "active": False, "last_heartbeat": None, "live_progress": None,
 "durable_progress": {"phase": "files", "done": 1, "total": 1}}
```

`blocked_by` and `ready` come from looking up `depends_on` in this same map.
`durable_progress` is the artifact's own file-based report; `live_progress` is
what a running worker reports about itself. What changes is only the grouping:
one flat map of every artifact on the volume, with no runs/sources/datasets
split. Launching, leases, and heartbeats are unaffected.

These are not declaration's states (§4), deliberately. Declaration classifies a
*requested* artifact against disk and can report `new`, `undeclared`, or a
definition `conflict`. A scan has no request to compare against and only finds
folders that already hold a manifest, so it never reports the first two and
reports `conflict` only for a manifest it cannot decode.

An unreadable manifest becomes an error entry at its own path; discovery of
every other path continues.

A state result is an observation over a scan interval, not a transactional
snapshot: rebuild it on refresh, publish the completed map at once.

### Minimal execution contract

A job is eligible when its declaration and dependency graph are valid, it has a producer, its state is `declared` or `partial`, every direct dependency is `done` with a valid dependency graph, and no execution owns its exact path. Eligibility permits an attempt; the producer must validate any partial state before resuming. Virtual artifacts become complete through their inputs and are never launched.

For an eligible artifact, the executor:

1. Establishes exclusive execution ownership for its path.
2. Refreshes storage, rechecks readiness, and spawns a worker using resources from the authoritative manifest.
3. In the worker, refreshes and rechecks readiness again, loads the authoritative manifest, and resolves its producer.
4. Calls `Job(artifact).run(root, worker)`. Inputs come from the artifact's direct dependencies; the job binds what it consumes.
5. Confirms ownership at publication boundaries, verifies completion, commits remote changes, and records the attempt's outcome. Exceptions propagate; failed attempts are not reported as success.

Different artifact paths may run concurrently, including nested paths when dependencies permit it.

A job creates only its own outputs. Each output is fully written and closed at a unique temporary path in the destination directory before atomic publication. Final outputs are never streamed into, appended to, replaced, or deleted. Temporary files never satisfy completion. Output publication is per file; a crash between files may leave `partial` state. Jobs may also publish auxiliary checkpoints in locations owned by their type; these do not count toward completion or overlap nested artifacts.

On `partial` artifacts, a producer verifies that it can reuse existing outputs and checkpoints consistently with the definition before publishing missing outputs. It preserves existing final files and raises if resumption is unsupported. Cleanup and deletion remain explicit operations outside jobs. Temporary-only failures leave no completed output and may be retried.

Execution ownership must prevent two workers from publishing to the same path. Serialize launch decisions and retain ownership until worker termination and its publication outcome are known, including failures. A missed heartbeat or lease check immediately before writing cannot fence off a still-running worker. Log handling must not serve as an unconditional path for publishing unfinished outputs.

## 8. Delivery and acceptance

Implement in this order, with a complete source-to-tokenizer-to-tokenized-source example working end to end before connecting execution. Everything is developed and tested on Modal, against the mounted volume — there is no local path to keep working.

1. Immutable definitions, normalization, manifest round trips, and pure resolution.
2. Declaration, status, and binding.
3. Generic state discovery and dependency validation, including virtual datasets.
4. Execution ownership and atomic output publication.

The backbone is ready when these cases hold:

| Case | Required result |
| --- | --- |
| Shared dependency reached twice | One resolved entry; conflicting definitions raise before writes. |
| Tokenizer sources supplied in reverse order | Same normalized definition, identity, manifest, and training input order. |
| Manifest round trip | Same definition and recorded fields; reconstructed value is unbound. |
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

`STORAGE` is the mount path of the volume inside the container, and it is the default root for every storage-touching call. Any other store is reached by passing an alternative absolute path in its place: a second volume mounted elsewhere in the container, or a plain folder on a development machine. Both are ordinary roots — the code does not distinguish them, and nothing detects which kind it was handed.

What is not supported is reaching the volume itself from outside Modal. The mount exists only in the container, so there is no environment detection and no remote-declaration call; a laptop works against a folder of its own, not against `/storage` over the network. If declaring onto the volume from a laptop is wanted later, it returns as one explicit remote call, never as a default root that varies per process.

Content verification, automatic invalidation, historical-code execution, graph-wide transactions, general retry policies, mutable annotations, schema migrations, multiple producers, and garbage collection require separate designs. Add them when a concrete pipeline needs them; the backbone keeps declaration, observation, execution, and use independently understandable.
