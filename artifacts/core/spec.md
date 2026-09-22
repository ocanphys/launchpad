# Artifact Framework Backbone Spec

The contract every artifact, job, and the code that declares, resolves, launches and binds them is written against. [README.md](../../README.md) is the map of where things live.

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

Each concrete type belongs to a family: a package under `artifacts/` holding the artifact class and, in a sibling `jobs.py`, the one job that produces it. Families never import functions or constants from one another; the only thing one family takes from another is an artifact class to name as a dependency type.

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

`deps()` returns only direct artifact-valued parameters, in a deterministic order: fields by name, sets by artifact path. Support individual artifacts, optional artifacts, and tuples or sets of artifacts. Parameter annotations distinguish an empty dependency container from an ordinary empty tuple. Other nested configuration contains ordinary values, never hidden artifact dependencies.

Paths, files, producers, and dependency lists are derived. They are not constructor arguments or separately persisted state. Artifact modules remain lightweight; resolving a producer imports its implementation only when executing it.

### Identity and agreement

Keep three questions distinct:

1. **Same identity:** two artifacts compute the same `artifact_path`. Identity is always written as `artifact.artifact_path`, never as `==`. It is what deduplication, lease keys, manifest locations, and every map of artifacts key on.
2. **Same definition:** concrete type, normalized parameters, and dependency definitions agree recursively. Commit, resources, and bound state are excluded at every node. **This is what `==` and `hash()` mean.** The frozen dataclass generates both: `commit` and `allocated_resources` are declared `compare=False`, and bound state is set outside the declared fields, so all three exclusions hold with no hand-written comparison. Binding does not change equality.
3. **Same provenance or allocation:** the recorded non-identifying fields agree.

Declaration and resolution check definition agreement with `==`, and always between two artifacts already known to share a path -- a manifest read from that path, or a second route reaching it during traversal. Sharing a path is therefore never evidence of agreement: two sources named `odyssey` with different URLs land on one path, compare unequal, and must conflict.

A dependency field's declared type says whether its order identifies the artifact, and nothing else does. A `tuple` is a sequence: its order and its repeats are part of what the artifact is, as with a mapped dataset's concatenated source streams. A `frozenset` is a set: neither is, as with a tokenizer's `sources`. Equality and hashing then follow from the type, so no value is ever reordered to make two definitions compare equal.

Construction accepts any iterable for a set-valued field and rejects two members that share an artifact path, which is a conflicting definition wherever it appears. A set has no order, so anything rendering one as a sequence -- a digest, a JSON array, a traversal, a job's input stream -- picks one at that point. Those orders are renderings, never comparisons.

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
  tokenized/<tokenizer-uid>/odyssey/
    manifest.json
    tokens.bin
```

`TokenizedSource.artifact_path` is `tokenized/<tokenizer-uid>/<source-uid>`: its own root, grouped by tokenizer, never inside the tokenizer's folder. Leases use exact artifact paths; owning a directory does not grant ownership of artifacts beneath it.

Path encodings are a storage compatibility contract. Changing them requires an explicit migration. Readable names and short hashes may collide; consistency checks detect conflicting definitions at a shared path rather than assuming the encoding is collision-free.

## 3. Manifests and loading

Every declared artifact owns `root / artifact_path / "manifest.json"`. Its manifest is a self-contained definition, with dependency manifests embedded recursively. **The manifest when loaded to Python is a dictionary:**

```json
{
  "artifact": "artifacts.sources.SourceURL",
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
        "artifact": "artifacts.sources.SourceURL",
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
artifact.parameters() -> dict
Artifact.from_manifest(data: dict, memo=None) -> Artifact
Artifact.load(artifact_path, root=None, memo=None) -> Artifact
artifact.bind(root=None) -> Self
```

`artifact.to_manifest()` and `artifact.bind()` are instance methods: they operate on an existing object. `Artifact.from_manifest(...)` and `Artifact.load(...)` are class-level entry points: the manifest identifies which concrete class to instantiate. Their `-> Artifact` annotation means a concrete subclass instance, never the abstract class itself. `-> Self` means an instance of the receiver's concrete class.

Omitting `root` uses `STORAGE`; an explicit root overrides it for that call. Normal calls are `artifact.bind()` and `Artifact.load(artifact_path)`.

`to_manifest` and `from_manifest` convert between an artifact and a dictionary without storage access. `load` reads `root / artifact_path / "manifest.json"`, parses JSON into a dictionary, and passes it to `from_manifest`. It rejects a reconstructed path that differs from the requested folder. Invalid schemas, unknown types, and missing or unreadable manifests raise descriptive errors. This keeps decoding in one place and gives filesystem access its own name.

**`Artifact.load(...)` always returns an unbound instance, even when all outputs exist.** It never loads output state or calls the binding hook. Call `.bind()` explicitly on the returned instance before using output-dependent methods. `Artifact.from_manifest(...)` also always returns an unbound instance.

An artifact is a pure function of its manifest, so decoding is memoizable by the manifest. `memo` is one dict a caller keeps across calls: a subtree seen before, nested inside another manifest, comes back as the same instance instead of being decoded again, which is what keeps a run's legs, each embedding every leg before it, linear to decode. Instances are frozen, so sharing them is safe; the one thing a shared instance must not be is bound, which is why `bind` never passes a memo.

A definition at a path is immutable once declared, and a scan of the volume shows nothing from a manifest but its definition, so the launcher keeps what each path decoded to for its whole life (`main.resolved`) and reads a manifest only the first time it sees the path. What may change on a redeclaration, the resources, is read fresh by launching, which loads the manifest itself.

The round-trip contract is definition agreement, plus preservation of recorded commit and resources. Path equality alone is too weak to test serialization.

### Recorded fields

New definitions capture the process's defining revision lazily: local HEAD, or the revision supplied with deployed code. A process may cache this stamp; an unavailable revision is recorded explicitly as unknown. Reconstruction always retains the manifest's recorded value. Mixed revisions within a graph are allowed.

`commit` describes the definition. It neither selects execution code nor proves which code produced the outputs. Execution uses deployed code. Resources configure that execution; unspecified dimensions are omitted so Modal supplies its platform defaults.

Once written, a manifest's definition and commit are immutable. A matching redeclaration retains both; only its resources follow the latest committed declaration, since they configure the next execution and nothing else. The manifest at an artifact's own path is authoritative for execution; an embedded copy in a parent's manifest cannot override it. General annotations, mutable labels, and execution histories are outside this manifest schema.

## 4. Storage, status, and binding

Declaration, loading, status, and binding all read one fixed path:

```python
STORAGE = Path("/storage")  # config.py -- where the volume is mounted in the container
```

`STORAGE` is a plain constant, not configuration: there is no per-session
setup, no environment detection, and no second root that varies by process.

Every storage-touching call takes `root=None` and resolves it to `STORAGE`
when omitted. **Passing an explicit absolute path in its place is how you
reach any other store** -- a second volume mounted elsewhere in the container,
a `tmp_path` in a test, a working folder on a development machine. All of
these are ordinary roots; the code does not distinguish them and nothing
detects which kind it was handed. This is not a corner case, it is the whole
mechanism, and it is why no local mount ever has to be arranged for.

That leaves one rule, which the whole arrangement depends on: **nothing
internal may call `bind()` or `load()` bare.** A job, a `_load` hook, or
anything else already holding a root threads it through every call it makes.
A hook that binds a dependency with no argument would silently reach for
`STORAGE` while everything around it reads a different store -- no error,
just the wrong file or a missing one. `root` is therefore a parameter on
every internal call, and the `STORAGE` default is consumed only at the top:
a notebook in the lab container, or the launcher.

**`STORAGE` and every `root` are always absolute. `artifact_path` is always
relative (§2).** Every join in this contract (`root / artifact_path`, §3,
§6, §7) depends on that pairing -- pathlib's `/` silently discards the left
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
# tokenized/<tokenizer-uid>/odyssey
```

The source occurs once even though both the tokenizer and tokenized source reference it.

## 6. `lab`: the notebook-facing API

`lab` is the notebook API, imported with `import lab`. It has one entry point: declaration. Construction, loading, and binding stay on the artifact classes and instances -- `Tokenizer(...)`, `Artifact.load(path)`, `tokenizer.bind()` -- and all read `STORAGE` directly (§4). There is nothing to initialize.

A committed declaration whose requested artifact belongs to a run (a `Training`, with a `run_id`) and was created by that commit also copies the notebook it ran from to `runs/<run_id>/declare-<uid>.ipynb`, before the commit, so the copy and the manifests land together. The notebook is the kernel's own file (`lab.current_notebook()`, from jupyter_server's `JPY_SESSION_NAME`); `local.declare_on_volume` reads it on the laptop and ships it along. Preview, a redeclaration, and a shared artifact write no copy.

`lab.refresh()` reloads the volume and `lab.save()` commits it, the two mount operations a notebook in the lab container needs by hand: see files a job wrote elsewhere, and make a saved notebook outlive the container. Both are refused outside a container with the mount.

The one other name it exports is `lab.worker`, a stand-in execution context for running a single job by hand from a cell, `job.run(root, lab.worker)`: no lease, no heartbeat, logging to the console. It is not configuration, and nothing in `lab` reads it.

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
    notebook=None,
) -> DeclarationReport
```

`verbose` controls how the result is printed; it does not change resolution,
inspection, or writes. `root` is for tests and manual scripts -- a notebook
never passes it, and gets `STORAGE`. `notebook` is the bytes of the declaring
notebook when the caller has them and no kernel; a notebook never passes it
either.

The algorithm performs one operation against its root:

1. Resolve the requested graph.
2. Inspect each path once and compare existing manifests using definition agreement.
3. Record commit drift and resource differences per retained graph node. Drift blocks only with `strict_commit=True`; resource differences never block.
4. Collect one row per path with state, drift, and differences. Preview returns these observations in a report without writes.
5. With `commit=True`, refuse before writing if any row blocks. Otherwise publish manifests for `new` artifacts in dependency order, and rewrite an existing matching manifest whose resources differ with the requested resources, its definition and commit kept. Reload the volume before inspection and commit successful writes before returning -- the one place this contract touches volume mechanics, internal to this function, not exposed as a separate call.
6. Return a report recording created and updated manifests and the resulting observed states. Reuse the resolved artifact list and recheck changed paths as needed; do not resolve the graph again.

Example: `sources/odyssey` and `tokenizers/bpe-1.0k-feeeeefa90` are already
declared and built (the tokenizer's manifest recorded an older commit than
`HEAD`); the `TokenizedSource` built from them is not yet declared.

```python
>>> report = lab.declare(tokens)   # preview -- commit defaults to False
sources/odyssey                                      done
tokenizers/bpe-1.0k-feeeeefa90                        done   (drift: a1b2c3d -> e4f5a6b)
tokenized/bpe-1.0k-feeeeefa90/odyssey            new

2 done, 1 new
ok -- 1 to declare

>>> report.rows   # commit and resources are [stored, requested] pairs
[{"path": "sources/odyssey", "state": "done", "drift": False,
  "differences": {}, "created": False, "updated": False, ...},
 {"path": "tokenizers/bpe-1.0k-feeeeefa90", "state": "done", "drift": True,
  "differences": {}, "created": False, "updated": False, ...},
 {"path": "tokenized/bpe-1.0k-feeeeefa90/odyssey", "state": "new",
  "drift": False, "differences": {}, "created": False, "updated": False, ...}]

>>> lab.declare(tokens, commit=True)   # same rows; the new one gets published
sources/odyssey                                      done
tokenizers/bpe-1.0k-feeeeefa90                        done   (drift: a1b2c3d -> e4f5a6b)
tokenized/bpe-1.0k-feeeeefa90/odyssey            done   (created)

1 manifest written: tokenized/bpe-1.0k-feeeeefa90/odyssey
```

Requesting the tokenizer again with `allocated_resources=Resources(cpu=4.0)`
reports the difference in preview and writes it on commit:

```python
tokenizers/bpe-1.0k-feeeeefa90                        done   (drift: a1b2c3d -> e4f5a6b; resources differ, requested apply on commit: {} -> {"cpu": 4.0})
...
tokenizers/bpe-1.0k-feeeeefa90                        done   (drift: a1b2c3d -> e4f5a6b; resources updated: {} -> {"cpu": 4.0})
```

`drift` alone never blocks (only `strict_commit=True` would); `verbose=False`
still shows drift, since it isn't a blocker reason, only `conflict` and
`undeclared` rows hide their detail without `verbose`.

`DeclarationReport` is serializable data: rows include path, state, drift, differences, the stored and requested commit and resources, and whether a manifest was created or updated. Counts and the verdict are derived from the rows. `declare` prints a concise summary; `verbose` expands details. Storage conflicts return a report in preview and raise with that report attached when committing. Structurally invalid input graphs raise before a report exists. Infrastructure failures propagate with the affected operation and path.

Preview never creates folders or temporary files. Definition and completion checks remain part of the artifact contract.

The API is one `declare` function, one `resolve` function, one `state` function, the two volume verbs `refresh` and `save`, and no session configuration.

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
state(root=STORAGE) -> dict[str, dict]
```

`main.py`'s whole-volume read, keyed by artifact path as a string since the
map is what `/state` serves. Computes the current state of every artifact
under `root` over one glob -- no dependency resolution, no DAG walk:

1. `glob(root/**/manifest.json)` -- every declared artifact, by folder.
2. Per manifest, once: `Artifact.load(...)`, then `artifact.status(root)`.
3. Per artifact, `deps()` for its direct dependency paths -- one level, every
   path a key in this map or missing from it, no traversal.
4. One lease and heartbeat snapshot for the whole scan.

Each entry keeps the shape the dashboard reads:

```python
{"type": "Tokenizer", "status": "done", "error": None,
 "depends_on": ["sources/odyssey"], "parameters": {"vocab_size": 1000},
 "blocked_by": [], "done": True, "ready": False, "verdict": "done",
 "call_id": None, "active": False, "last_heartbeat": None, "live_progress": None,
 "durable_progress": {"phase": "files", "done": 1, "total": 1}}
```

`verdict` is the one word a row shows, first match wins: `done`; `running`
while the leased call beats; `starting` while a lease has no heartbeat and
`now - granted_ts < STARTUP_GRACE_SECONDS` (60 seconds, configured in
`config.py`); `failed` for a manifest that would not load or a lease whose
call stopped beating or exhausted its startup grace period; `runnable` when
`ready`; `blocked` otherwise. A failed artifact is still `ready`, so it can be run again. Every
field is computed from one snapshot of the leases and heartbeats and one reload
of the volume. Startup classification uses that snapshot's clock and grant
timestamp, with no call-status request to Modal. A lease without a grant
timestamp cannot qualify as starting. The launch check uses the same grace
period to refuse a second call while the first is starting.

`parameters` is the artifact's own fields only -- the same half of the
annotation-driven split `to_manifest` puts under `"parameters"`, so a
dependency never appears here as a nested manifest. A dependency is a path in
`depends_on` and nothing more; whatever else is true of it lives on its own
entry in this same map, which is what keeps one artifact's record from going
stale when another's changes.

`blocked_by` is every direct dependency not `done` on this same map, a
dependency with no manifest included; `ready` is not done, not blocked, and
produced by something -- a virtual artifact is never ready. There is no
`drift`: a scan has no requested commit to compare against. `durable_progress`
is the artifact's own file-based report; `live_progress` is what a running
worker reports about itself. The dashboard slices the map per view itself,
following `depends_on` for a run's or a dataset's closure. Launching, leases,
and heartbeats are unaffected.

These are not declaration's states (§4), deliberately. Declaration classifies a
*requested* artifact against disk and can report `new`, `undeclared`, or a
definition `conflict`. A scan has no request to compare against and only finds
folders that already hold a manifest, so it never reports the first two and
reports `conflict` only for a manifest it cannot decode.

An unreadable manifest becomes an error entry at its own path; discovery of
every other path continues.

A state result is an observation over a scan interval, not a transactional
snapshot: rebuild it whole, publish the completed map at once.

### One reload, taken when something changed

The launcher container keeps no clock. It reloads the volume and recomputes
the `state` map -- manifests, leases, heartbeats, what launching and the
table need -- when something has changed what that map would say: a call
has started or has committed and exited (it announces each on the
`refreshes` Queue, which a listener thread in the launcher blocks on), this
container granted or released a lease, or the page asked (`POST /refresh`,
for a manifest declared from the lab, which no call announces). The map is then
the authority on the volume: lagged by one message, but complete, because
an artifact's files never leave the volume once they land. Nothing else the
container serves is cached. A call's log channels are read from the Dicts on
every request, so they are live; a leg's step log is read off the mount,
which means off the last reload's image of it, and only inside a block that
closes the descriptor before the request returns (`open_jsonl`).

A reload and a read cannot both be in flight on the same mount: a read
landing mid-reload sees a path that is briefly absent, and a file descriptor
still open when a reload starts fails the reload. Neither presents as an
error. What keeps them apart is one lock (`main.mount_lock`), taken by
everything in that container which reloads the mount or opens a file on it,
and the fact that no request holds a descriptor past its own block. The lock
is never held across the listener's blocking read of the Queue, and the
routes that take it stay sync `def`, so it is only ever held on a threadpool
thread. `leasebook` must never be given `@modal.concurrent`: one container,
one map, one writer of it.

The volume is the source of truth, and nothing reaches the dashboard from a
worker that the worker has not committed under its lease, except the log
rows and heartbeats it publishes to the Dicts. Writing to the volume is the
worker's (its own artifact, on commit) and `persist_logs`'s (log files and
`call_history.json`, on a schedule, in its own container); the launcher
container writes nothing there.

### Minimal execution contract

A job is eligible when its declaration and dependency graph are valid, it has a producer, its state is `declared` or `partial`, every direct dependency is `done` with a valid dependency graph, and no execution owns its exact path. Eligibility permits an attempt; the producer must validate any partial state before resuming. Virtual artifacts become complete through their inputs and are never launched.

For an eligible artifact, the executor:

1. Refuses while a call is active or starting on its path (`main.attempt_launch`, reading the `leases` and `beats` Dicts).
2. Refreshes storage and checks readiness (`main.attempt_launch`: the manifest at the path, each direct dependency's manifest and completion files), spawns a worker using resources from the authoritative manifest (`main.resource_options`, `run_job.spawn`), and records the grant naming that call (`leases.put`, `call_history`).
3. In the worker, refreshes storage and confirms the grant names this call (`system.runtime.initialize_worker`, `Lease.confirm("boot")`), loads the authoritative manifest (`Artifact.load`) and resolves its producer (`Artifact.job`), all in `main.run_job`.
4. Calls `Job(artifact).run(root, worker)`. Inputs come from the artifact's direct dependencies; the job binds what it consumes.
5. Confirms the grant again after the run and before the commit (`Lease.confirm("pre vol commit")`, `("commit")`), commits (`volume.commit` in `initialize_worker`), and logs the outcome under the call (`call_logs["{call_id}:container"]`). Exceptions propagate; failed attempts are not reported as success.
6. Announces itself twice on the `refreshes` Queue, each message naming the artifact, the call and the event: `started`, once it holds the lease and has published its first heartbeat, and one naming how it ended, after the commit and after the last heartbeat, which is marked `exited`. Each message trails what it announces, never leads it: a launcher that reloaded earlier would find what it already had. A message carries no state of its own -- it says only that the volume and the Dicts are worth reading again, which the launcher then does for itself.

Different artifact paths may run concurrently, including nested paths when dependencies permit it.

A job creates only its own outputs. Each output is fully written and closed at a unique temporary path in the destination directory before atomic publication. Final outputs are never streamed into, appended to, replaced, or deleted. Temporary files never satisfy completion. Output publication is per file; a crash between files may leave `partial` state. Jobs may also publish auxiliary checkpoints in locations owned by their type; these do not count toward completion or overlap nested artifacts.

On `partial` artifacts, a producer verifies that it can reuse existing outputs and checkpoints consistently with the definition before publishing missing outputs. It preserves existing final files and raises if resumption is unsupported. Cleanup and deletion remain explicit operations outside jobs. Temporary-only failures leave no completed output and may be retried.

Execution ownership must prevent two workers from publishing to the same path. Serialize launch decisions and retain ownership until worker termination and its publication outcome are known, including failures. A missed heartbeat or lease check immediately before writing cannot fence off a still-running worker. Log handling must not serve as an unconditional path for publishing unfinished outputs.

## 8. Delivery and acceptance

Implement in this order, with a complete source-to-tokenizer-to-tokenized-source example working end to end before connecting execution. Everything is developed and tested on Modal, against the mounted volume -- there is no local path to keep working.

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

`STORAGE` is the mount path of the volume inside the container, and it is the default root for every storage-touching call. Any other store is reached by passing an alternative absolute path in its place: a second volume mounted elsewhere in the container, or a plain folder on a development machine. Both are ordinary roots -- the code does not distinguish them, and nothing detects which kind it was handed.

What is not supported is reaching the volume itself from outside Modal. The mount exists only in the container, so there is no environment detection and no remote-declaration call; a laptop works against a folder of its own, not against `/storage` over the network. If declaring onto the volume from a laptop is wanted later, it returns as one explicit remote call, never as a default root that varies per process.

Content verification, automatic invalidation, historical-code execution, graph-wide transactions, general retry policies, mutable annotations, schema migrations, multiple producers, and garbage collection require separate designs. Add them when a concrete pipeline needs them; the backbone keeps declaration, observation, execution, and use independently understandable.
