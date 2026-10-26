# The artifact codec, end to end

How an artifact goes from a constructor call to a `manifest.json` on the volume and back into a Python object, which function runs at each step, what it does, and why it does it. [spec.md](spec.md) states the rules; this file shows the machinery that enforces them. [LESSONS.md](../../LESSONS.md) records the pydantic behaviour the machinery works around, and [docs/workflows.md](../../docs/workflows.md) traces the flows that call into it.

The code lives in four files:

| File | Owns |
| --- | --- |
| [artifact.py](artifact.py) | `Artifact`: construction checks, identity, `load`, `bind`, `status` |
| [schema.py](schema.py) | Reading field annotations: which fields are dependencies, which types are allowed, which are polymorphic |
| [manifest.py](manifest.py) | The codec itself: Python value to JSON and back, `to_manifest`, `from_manifest` |
| [locate.py](locate.py) | `dotted(cls)` and `locate(path)`: a class to its import path and back |

## 1. The lifecycle at a glance

```text
 class definition    __init_subclass__                  producer is a str or None
        |
 construction        Tokenizer(...)  ->  __post_init__   every field checked and normalized
        |
 identity            uid, artifact_path, ==, hash        derived from the fields, nothing stored
        |
 graph               deps(), resolve()                   dependencies before dependents
        |
 encoding            to_manifest() -> manifest_json()    dict, then the bytes on disk
        |
 declaration         lab.declare                         compare with disk, write what is new
        |
 decoding            Artifact.load -> from_manifest      bytes back to the same artifact
        |
 running             main.run_job -> artifact.job()      the producer, imported only here
        |
 binding             artifact.bind(root) -> _load        the stored definition, files attached
```

The rest of the file has one section per row, then worked examples, then the polymorphic `Model` field, then the design choices and what could be simpler.

## 2. Class definition: `__init_subclass__`

When Python creates a subclass of `Artifact`, `Artifact.__init_subclass__` checks that `producer`, if the class sets it, is a string or `None`. It does not check that `producer` is set at all, because intermediate bases like `Source` and `tokenizers.Tokenizer` leave it to their subclasses. A missing producer is caught later, at the first instantiation.

**Why a string:** `producer = "artifacts.tokenizers.bpe.jobs.TokenizerJob"` names the job without importing it. A family's `jobs.py` imports torch or numpy; declaring, resolving and drawing a graph must not pay for that. `artifact.job()` is the only place the string becomes a class.

## 3. Construction: `__post_init__`

The dataclass `__init__` assigns the fields, then calls `__post_init__`. A subclass that defines its own `__post_init__` calls `super().__post_init__()` first, so `Artifact.__post_init__` always runs before any family-specific check.

`Artifact.__post_init__` ([artifact.py:165](artifact.py#L165)) does this:

1. **Producer present.** `hasattr(cls, "producer")`, or `TypeError`.
2. **Read the schema.** `schema.dependencies(cls)` and `schema.annotations(cls)`, both cached per class, so the work below happens once per class, at its first instantiation:
   - `dependencies(cls)` ([schema.py:39](schema.py#L39)) returns the fields whose annotation is an artifact type: `A`, `A | B | None`, `tuple[A, ...]` or `frozenset[A]`, mapped to their container (`None`, `tuple` or `frozenset`). It reads annotations, never values, so an empty `sources=frozenset()` is still a dependency.
   - `annotations(cls)` ([schema.py:66](schema.py#L66)) returns every field's resolved annotation, and runs `_check_parameter` on each one that is not a dependency. That refuses any type the codec cannot rebuild exactly: `Any`, a bare `tuple`, a `list`, a `dict`, an artifact hidden inside a parameter, a union of two record types. For a frozen dataclass it recurses into that record's own fields.
3. **Each parameter is normalized.** For every field that is not a dependency: `manifest.normalize(value, annotation, path)`, and the result replaces the field's value (via `object.__setattr__`, since the dataclass is frozen).
4. **Each dependency is checked.** A `frozenset` field accepts any iterable and is converted. A `tuple` field must already be a tuple. Every member must be an instance of one of the annotated artifact types. A `frozenset` may not hold two members with the same `artifact_path`, since that is two definitions of one folder.

### `normalize`: the round trip every constructor runs

`normalize` ([manifest.py:100](manifest.py#L100)) makes a constructed artifact hold exactly what its manifest will decode to. Three steps, none redundant (LESSONS.md, "pydantic is strict only where it is told to be"):

| Step | Call | Catches |
| --- | --- | --- |
| Strict Python validation | `_field(record).model_validate({"value": value})` | Wrong types: a `bool` in an `int` field, a `list` in a `tuple` field, a `str` in an `int` field |
| JSON round trip | `_decode(_encode(validated, annotation), annotation, path)` | Whatever the disk would give back differently: an `int` in a `float` field becomes a `float`, `-0.0` becomes `0.0`, records are rebuilt through their constructors |
| Equality check | `restored != value` raises | Anything the first two silently narrowed, such as a subclass serialized as its base |

The pieces it uses:

- **`_field(annotation)`** ([manifest.py:35](manifest.py#L35)) builds, once per annotation, a pydantic model whose one field `value` has that annotation and the strict config `_STRICT`. The wrapper exists because pydantic accepts no config for a bare dataclass, but a model's config reaches every dataclass nested inside its field.
- **`_STRICT`**: `strict=True` (no coercion except int to float), `extra="forbid"` (unknown fields refused), `allow_inf_nan=False` (JSON has no NaN), `revalidate_instances="always"` (a dataclass instance is checked field by field, never trusted for its type).
- **`_dump(value, annotation)`** ([manifest.py:57](manifest.py#L57)) serializes with pydantic, then re-parses the JSON with every object's keys sorted and every float passed through `float(s) + 0.0`, which turns `-0.0` into `0.0`. Sorted keys make the bytes independent of field order; the float trick makes two equal values encode identically.
- **`_load(data, annotation, path)`** ([manifest.py:71](manifest.py#L71)) validates JSON back into a value, strictly.
- **`_encode` / `_decode`** ([manifest.py:78-97](manifest.py#L78-L97)) are `_dump` / `_load` for one field, plus the polymorphic case (section 11). For every other field they simply call `_dump` and `_load`.
- **`_named(path)`** ([manifest.py:44](manifest.py#L44)) turns a pydantic `ValidationError` into a `TypeError` (or a `ValueError` where a validator refused the value), naming the field: `Tokenizer.vocab_size: Input should be a valid integer`.

Family checks run after this, in each subclass's `__post_init__`: `DataSet` refuses mixed tokenizers, `Training` fills `model` in from its starting checkpoint, `Pretraining` checks the model's vocabulary against the tokenizer's. They see already-normalized values.

## 4. Identity: `uid`, `artifact_path`, `==`

Nothing about identity is stored; all of it is computed from the fields.

- **`==` and `hash`** come from the frozen dataclass. `commit` and `allocated_resources` are declared `compare=False`, so they never affect either. Two artifacts are the same definition when their type, parameters and dependencies agree, recursively.
- **`uid`** is a readable label each family writes, usually with `_digest(...)` ([artifact.py:59](artifact.py#L59)): the first 10 hex characters of a SHA-256 over key-sorted JSON.
- **`artifact_path`** is the folder the artifact owns, usually built from `uid`. It must be a pure function of the identifying fields, so equal artifacts share a folder and `load` can reject a manifest found anywhere else.

**Why normalization matters here:** `uid` digests the field values. Without normalization, a leg declared with `max_norm=1` and one with `max_norm=1.0` would compare equal (`1 == 1.0`) but digest differently (`1` and `1.0` are different JSON), so two equal definitions would land in two folders. Normalizing at construction means equal definitions have equal fields, hence equal digests, paths and manifests.

## 5. The graph: `deps()` and `resolve()`

- **`deps()`** ([artifact.py:203](artifact.py#L203)) lists the direct dependencies: fields in name order, and each field's members through `manifest.members` ([manifest.py:118](manifest.py#L118)), which sorts a `frozenset` by `artifact_path`. A set has no order, so the traversal picks one that never depends on iteration order.
- **`resolve(artifact)`** ([resolve.py](resolve.py)) walks `deps()` depth first, emits dependencies before dependents, each path once. A path reached twice must hold an equal definition, which compares the whole subtree, so a conflicting leaf cannot hide behind a shared path. A path on the current descent again is a cycle and raises.

## 6. Encoding: `to_manifest` and `manifest_json`

`to_manifest` ([manifest.py:143](manifest.py#L143)) builds the dict, five keys in fixed order:

| Key | Built by | Contents |
| --- | --- | --- |
| `"artifact"` | `dotted(type(artifact))` | The concrete class's import path |
| `"commit"` | the field | The revision the definition was made at |
| `"allocated_resources"` | `_encode(...)` | CPU and GPU request; `null` means the platform default |
| `"parameters"` | `parameters(artifact)` | Every non-dependency field except the two above, `_encode`d, sorted by name |
| `"dependencies"` | `to_manifest(dep)`, recursively | One full manifest per dependency, a list for a container, `null` for an empty optional |

`parameters(artifact)` ([manifest.py:126](manifest.py#L126)) is also used on its own, by the dashboard, to show an artifact's fields without its dependencies' manifests.

`manifest_json(manifest)` ([manifest.py:204](manifest.py#L204)) turns the dict into the bytes that go on disk: indented, UTF-8, `allow_nan=False` because Python would otherwise write `NaN` and `Infinity`, which no other JSON reader accepts.

**Why dependencies are embedded:** a manifest can be decoded and compared on its own. `==` recurses through dependencies, so comparing a stored manifest with a request needs the whole subtree, and having it inline means no other folder has to be read, or even exist, to decode one.

## 7. Declaration: `lab.declare`

`lab.declare(artifact, commit=...)` is the only writer of manifests. For each node of `resolve(artifact)`, `lab._inspect` reads `artifact.status(root)` (which files exist) and, if a manifest is there, `Artifact.load(path, root, memo)` and compares it with the request using `==`. A new node gets `path.write_text(manifest_json(node.to_manifest()))`. An existing, equal node is left alone, except that changed resources are written into it, keeping its stored definition and commit. An unequal one is a conflict and nothing is written.

## 8. Decoding: `Artifact.load` and `from_manifest`

`Artifact.load(artifact_path, root, memo)` ([artifact.py:245](artifact.py#L245)):

1. Refuses an absolute path or one containing `..`.
2. Reads and parses `root / artifact_path / manifest.json`.
3. Calls `from_manifest(data, memo)`. Any `ImportError`, `AttributeError`, `KeyError`, `TypeError` or `ValueError` from it is reraised as one `ValueError` naming the folder, so a scan can show one bad manifest as one bad row.
4. Refuses the result if its `artifact_path` is not the folder it was read from. A path is a function of the definition, so a mismatch means the folder was copied or renamed.

`from_manifest(data, memo)` ([manifest.py:172](manifest.py#L172)):

1. **Memo lookup.** With a memo, the key is the manifest's sorted JSON; a subtree decoded before comes back as the same instance.
2. **The class.** `locate(data["artifact"])`, which must be an `Artifact` subclass.
3. **Parameters.** `commit`, `allocated_resources` and every entry of `"parameters"` go through `_decode(value, hints[name], path)`, against the class's annotations.
4. **Dependencies.** Each nested manifest is decoded recursively; a container field is rebuilt with its container (`tuple(...)` or `frozenset(...)`).
5. **Construction.** `cls(**inputs)`. This runs `__post_init__` again, normalization and family checks included, so a manifest that breaks an invariant, such as a vocabulary mismatch, cannot be loaded either.

The result is always unbound: no files are attached, even when they all exist.

## 9. Running and binding

`main.run_job(artifact_path)` loads the artifact from the volume, calls `artifact.job()`, which runs `locate(self.producer)(self)`, and then `job.run(STORAGE, worker)`. This is the only place a producer string is imported.

`artifact.bind(root)` ([artifact.py:289](artifact.py#L289)):

1. `Artifact.load(self.artifact_path, root)`, deliberately without a memo, so the instance it gets is fresh and nobody else holds it.
2. Refuses if the stored definition is not `==` to `self`: you cannot bind a request to someone else's outputs at the same path.
3. Refuses if any completion file is missing.
4. Calls the stored instance's `_load(root)` hook, which attaches runtime state (a tokenizer's vocabulary, a dataset's memmaps) with `object.__setattr__`. That state is not a field, so it never enters `==`, `hash` or the manifest.
5. Returns the stored instance, with the stored commit and resources. The object `bind` was called on is never modified.

## 10. Worked examples

All outputs below are real, produced by the current code.

### A source: scalars, no dependencies

```python
SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
```

`Source` and `SourceURL` declare `name: str` and `url: str`. In `__post_init__`, both fields are parameters; `normalize` validates each as a strict `str`, round-trips it unchanged, and keeps it. There are no dependencies. `uid` is the name, `artifact_path` is `sources/odyssey`.

```json
{
  "artifact": "artifacts.sources.SourceURL",
  "commit": "abc1234",
  "allocated_resources": {"cpu": null, "gpu_count": null, "gpu_type": null},
  "parameters": {"name": "odyssey", "url": "https://example.org/odyssey.txt"},
  "dependencies": {}
}
```

### A tokenizer: a parameter tuple and a dependency set

```python
Tokenizer(vocab_size=1000, special_tokens=("<pad>", "<|endoftext|>"), sources=[source])
```

`artifacts.tokenizers.bpe.Tokenizer` has three fields, and the schema splits them by annotation alone:

| Field | Annotation | Kind | What `__post_init__` does |
| --- | --- | --- | --- |
| `vocab_size` | `int` | parameter | strict `int`; `vocab_size=True` raises `Tokenizer.vocab_size: Input should be a valid integer` |
| `special_tokens` | `tuple[str, ...]` | parameter | strict tuple; `["<pad>"]` raises `Tokenizer.special_tokens: Input should be a valid tuple` |
| `sources` | `frozenset[Source]` | dependency | any iterable becomes a `frozenset`; the list above is accepted |

`special_tokens` and `sources` both look like sequences in Python, but one is an array in `"parameters"` and the other an array of manifests in `"dependencies"`:

```json
{
  "artifact": "artifacts.tokenizers.bpe.Tokenizer",
  "commit": "abc1234",
  "allocated_resources": {"cpu": null, "gpu_count": null, "gpu_type": null},
  "parameters": {"special_tokens": ["<pad>", "<|endoftext|>"], "vocab_size": 1000},
  "dependencies": {
    "sources": [
      {
        "artifact": "artifacts.sources.SourceURL",
        "commit": "abc1234",
        "allocated_resources": {"cpu": null, "gpu_count": null, "gpu_type": null},
        "parameters": {"name": "odyssey", "url": "https://example.org/odyssey.txt"},
        "dependencies": {}
      }
    ]
  }
}
```

**Why `frozenset`:** which sources trained the tokenizer matters, not their order. With a set, `sources=[a, b]` and `sources=[b, a]` are one definition by construction: equal, equally hashed, one path, one manifest (`members` sorts them by path when writing the array). Nothing has to reorder values to make them compare equal.

**Why `special_tokens` must be a tuple and not a list:** the field is part of identity, so it must be hashable and immutable. Strict validation refuses a list rather than converting it, because a converted value would no longer be the value the caller passed.

`uid` is `bpe-1000-<digest>`, the digest covering `vocab_size`, `special_tokens` and the sorted source uids, so `artifact_path` is `tokenizers/bpe-1000-12db0597a5`.

### A tokenized source and a dataset: ordered dependencies and shared subtrees

```python
tokens = TokenizedSource(tokenizer=tokenizer, source=source)   # tokenized/bpe-1000-12db0597a5/odyssey
data = DataSet(train_set=(tokens,), valid_set=(tokens,))        # datasets/dataset-7c8413c945
```

`TokenizedSource` has two single dependencies, `tokenizer: Tokenizer` and `source: Source`, and no parameters at all. Its `tokenizer` field is annotated with the abstract base `artifacts.tokenizers.Tokenizer` and holds a BPE tokenizer: that is fine for an artifact, because every nested manifest names its own class under `"artifact"`. Section 11 comes back to this.

`DataSet` has `train_set: tuple[TokenizedSource, ...]`. A tuple is a sequence, so order and repeats are part of identity: the job concatenates the sources in that order, and `DataSet.uid` digests the uids in order. A `list` here raises; only a tuple is accepted.

The dataset's manifest embeds the tokenized source twice (once per split), and each copy embeds the tokenizer and the source. Decoding with a shared `memo` builds each distinct node once and hands back the same instance wherever it recurs. The instances are frozen, so sharing them is safe. `bind` never uses a memo, because it attaches state to the instance it loads.

## 11. Training, Pretraining and the polymorphic `Model`

### Nested records

`Training` ([SGD/training.py](SGD/training.py)) holds three plain records as parameters: `training_parameters: TrainingParameters` (which itself holds `LRSchedule` and `OptimizerParameters`), `loop_config: LoopConfig`, and `model: Model | None`. Records are frozen dataclasses, not artifacts: they have no folder or manifest of their own, and they encode as nested JSON objects inside `"parameters"`.

`normalize` treats a whole record as one value. With `revalidate_instances="always"`, pydantic rebuilds it through its constructor, so a record's own `__post_init__` (such as `LoopConfig`'s positivity checks) runs during normalization too. The JSON round trip turns `max_norm=1` into `1.0`, since the field is a `float`:

```json
"training_parameters": {
  "batch_size": 4,
  "lr_schedule": {"cosine_cycle_iters": 100, "max_learning_rate": 0.001, "min_learning_rate": 0.0001, "warmup_iters": 1},
  "lr_schedule_fn": "artifacts.core.SGD.lr_schedule.lr_cosine_schedule",
  "max_norm": 1.0,
  "optimizer": "torch.optim.AdamW",
  "optimizer_parameters": {"betas": [0.9, 0.95], "eps": 1e-08, "lr": 0.001, "weight_decay": 0.1},
  "seed": 0,
  "total_steps": 4
}
```

`optimizer` and `lr_schedule_fn` are dotted strings, not the objects, for the same reason `producer` is: declaring must not import torch. The job resolves them with `locate` on the worker.

### Training and Pretraining

`Training` and every subclass are `@dataclass(frozen=True, kw_only=True)`. `Pretraining` redeclares `starting_checkpoint` to narrow its type to `Pretraining | None`, and a redeclared field moves to the base's position; without `kw_only`, a defaulted base field followed by required subclass fields fails at class creation (LESSONS.md, "Dataclass inheritance with defaults needs `kw_only`").

`Pretraining` has two dependencies and several parameters:

| Field | Kind | Note |
| --- | --- | --- |
| `dataset: DataSet \| MappedDataSet` | dependency | a union of two artifact types is fine: the nested manifest's `"artifact"` key says which |
| `starting_checkpoint: Pretraining \| None` | dependency | the previous leg, so a run is a chain of manifests, each embedding all earlier legs |
| `run_id`, `training_parameters`, `loop_config` | parameters | |
| `model: Model \| None` | parameter, polymorphic | the architecture |

`Training.__post_init__` calls `super().__post_init__()` first, so every field is normalized before it runs. Then, if `starting_checkpoint` is set and `model` is `None`, it copies the checkpoint's model; if both are set, they must be equal. `Pretraining.__post_init__` then checks `model.vocab_size` against the dataset's tokenizer. Because the model is written into the manifest either way, a decoded leg passes the model explicitly and the equality check holds.

### The schema issue: a record field holds exactly its annotation

Records and artifacts differ in one thing that matters here. **An artifact's manifest names its class (`"artifact"`); a record's JSON does not.** A record is rebuilt as whatever class its field's annotation names. That is what lets `dataset: DataSet | MappedDataSet` work and what would break `model: Model | None`, because the value is a `models.transformer.Transformer` (or a `models.mamba.Mamba`, or a `models.transformerV2.Transformer`), and the annotation names only the base.

This is what the codec does with `Model` when the polymorphic flag is turned off (real outputs):

```python
>>> normalize(transformer, Model | None, "Pretraining.model")
TypeError: Pretraining.model: expected exactly artifacts.core.SGD.training.Model | None, got Transformer(device='cuda', vocab_size=1000, ...)

>>> _dump(transformer, Model | None)          # what would be written
{'device': 'cuda'}

>>> _load({"device": "cuda", "d_model": 16}, Model | None, "Pretraining.model")
TypeError: Pretraining.model.d_model: Unexpected keyword argument
```

The second line is the dangerous one: pydantic serializes a subclass as its base, silently dropping every field `Transformer` adds. Only the equality check in `normalize` notices, which is why constructing such a `Pretraining` fails at once rather than writing `{"device": "cuda"}` to disk. Without an exception to the rule, a field can never hold more than one record type.

The alternatives each fail on a different point:

- **A union of records** (`Transformer | TransformerV2 | Mamba | None`): `_check_parameter` refuses it, because two records with overlapping fields cannot be told apart on disk. It would also make `artifacts/core` import every package under `models/`, reversing the dependency direction.
- **One stage per architecture** (`TransformerPretraining`, `MambaPretraining`, ...): duplicates the stage, its job and its lineage logic, when the loop is the same for every model.
- **Store the class of every record**: works, but then renaming or moving any record class, `LoopConfig` included, breaks every old manifest.

### How `polymorphic = True` solves it

`Model` declares `polymorphic: ClassVar[bool] = True` ([SGD/training.py:28](SGD/training.py#L28)). A `ClassVar` is not a dataclass field, so it never enters a manifest. Its subclasses inherit it. The codec reads it in one place:

- **`schema.polymorphic(annotation)`** ([schema.py:76](schema.py#L76)) returns the record class in an annotation that declares the flag, as a one-tuple, or `()`. For `Model | None` that is `(Model,)`; for `LoopConfig` it is `()`. The tuple goes straight into `isinstance` and `issubclass`.
- **`normalize`** validates a value that is an instance of that class against its own class (`type(value)`), not the annotation, so pydantic does not drop its fields.
- **`_encode`** dumps it against its own class and adds its dotted path under `"class"`. `class` is a Python keyword, so no dataclass field can collide with it.
- **`_decode`** pops `"class"`, imports it with `locate`, checks that it is a subclass of the field's class (`TypeError: Drawing.shape: ... is not a Shape` otherwise), and loads the remaining fields as that class.

The result in a `Pretraining` manifest:

```json
"model": {
  "class": "models.transformer.Transformer",
  "d_ff": 32,
  "d_model": 16,
  "device": "cuda",
  "dtype": "torch.float32",
  "num_heads": 2,
  "num_layers": 1,
  "rope_theta": 10000,
  "sequence_length": 8,
  "vocab_size": 1000
}
```

and `Artifact.from_manifest(leg.to_manifest())` gives back a leg equal to the original, whose `model` is a `models.transformer.Transformer`.

**The cost, and why it is opt-in:** the class path is now part of the stored format. Moving or renaming `models/transformer` breaks old manifests (`locate` cannot find the path) and needs a migration script. `Pretraining.lineage_hash` also includes `type(self.model).__module__`, so a moved architecture gets a new uid and folder, which is intended: a different architecture is a different trajectory. Only `Model`, the one field that genuinely varies by type, takes on that coupling; every other record stays plain data that can be renamed freely.

## 12. The choices, and why

| Choice | Why | What it costs |
| --- | --- | --- |
| The annotation is the schema; records store fields only | Record classes can move or be renamed without migrating manifests; the JSON reads as plain data | A record field holds exactly its annotated class, hence the polymorphic exception |
| Normalize at construction | A constructed artifact equals what its manifest decodes to, so `==`, digests, paths and bytes all agree for equal definitions | Every construction pays a pydantic round trip; a decode pays it twice (section 13) |
| Strict validation, no coercion except int to float | A coerced value would differ from what the caller passed, so the caller's object and the stored one could disagree | Callers must pass tuples, not lists, and exact types |
| Sorted keys, fixed top-level order, `-0.0` to `0.0` | Manifest bytes are a pure function of the definition; reordering a dataclass changes nothing on disk | None worth noting |
| Dependency manifests embedded | A manifest decodes and compares on its own, without reading other folders | Large manifests for long chains (each leg embeds every earlier leg), mitigated by `memo` |
| Containers declare order: `tuple` vs `frozenset` | Equality follows from the type, so no value is ever reordered to make two definitions agree | A caller must pick the right container when designing a family |
| `commit` and `allocated_resources` are `compare=False` | Recorded provenance and execution settings, not identity; an artifact rebuilt from an old manifest is still the same artifact | Two code paths (`parameters`, `from_manifest`) handle them by name |
| `producer`, `optimizer`, `dtype` as dotted strings | Declaring and resolving never import torch, numpy or a family's jobs | Typos surface only on the worker, when `locate` runs |
| `load` checks the path matches the folder | A copied or renamed folder cannot pass itself off as another artifact | None |
| `bind` loads fresh, without a memo | `_load` mutates the instance it returns; a shared instance would leak state | One extra decode per bind |

## 13. What could be simpler

These are suggestions, in order of how much they would help a reader.

1. **Move dependency checking into the codec.** `Artifact.__post_init__` normalizes a parameter with one call (`manifest.normalize`) but checks a dependency inline, in about 15 lines: container conversion, the `isinstance` against the annotation's arms, the duplicate-path check. A `manifest.normalize_dependency(value, container, annotation, path)` beside `normalize` would make `__post_init__` one loop with two calls, and put every rule about what a field may hold in one module, behind the same seam.
2. **Derive identity digests from the encoded parameters.** `Pretraining.lineage_hash` hand-picks `type(self.model).__module__`, `asdict(self.model)` and `asdict(self.training_parameters)`, which overlap with what `parameters()` already encodes, but not exactly: the hash uses the model's module while the manifest records module and class name. Hashing the encoded `"model"` and `"training_parameters"` values would make the folder and the manifest unable to drift apart, and would remove the need to know which `asdict` calls are equivalent to the encoding.
3. **Treat `commit` and `allocated_resources` as ordinary fields of the encoding.** `parameters()` excludes them by name, `to_manifest` encodes them separately, and `from_manifest` merges them back into the parameters before decoding. One tuple, `RECORDED = ("commit", "allocated_resources")`, used by all three would at least name the rule once.
4. **Decoding normalizes twice.** `from_manifest` validates each parameter with `_decode`, then `cls(**inputs)` runs `normalize` on the same values again. It is correct and idempotent, and the second pass is also what runs the family checks, so it cannot simply be skipped. Leave it unless decoding a long run chain becomes measurably slow; a way around it would need a second construction path, which costs more readability than it saves time.

What should stay as it is: the three steps of `normalize`, the strict config, sorted keys, embedded dependencies and the polymorphic opt-in. Each one is there because a specific failure got through without it (LESSONS.md has the list).

## Appendix: source

A snapshot of every file involved in declaring, encoding, decoding and binding an artifact, in lifecycle order. The files themselves are authoritative; when this copy and a file disagree, the file is right.

### artifacts/core/artifact.py

`artifact`: construction checks, identity, `load`, `bind`, `status`.

```python
"""The base artifact model: parameters plugged together with the artifacts
they're built from, a folder each artifact owns, and a manifest that rebuilds
it. Concrete types live in sources/, tokenizers/, tokenized/, dataset/,
mappeddataset/, stages/*.

An artifact has three lives, and this class carries all three:

    recipe    Artifact(params)         parameters, a folder, a manifest
    job       Job(artifact).run(root)  writes the files into that folder
    bound     artifact.bind(root)      the stored declaration, files loaded

`load`, `status` and `bind` take a root, defaulting to STORAGE. That default
is for the top of a notebook only. Anything already holding a root (a job, a
`_load` hook) passes it on, or it silently reads the volume while everything
around it reads a test folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, get_args

from artifacts.core import manifest, schema
from artifacts.core.locate import locate

if TYPE_CHECKING:
    # Type-only: job.py imports Artifact from here, so a real import would be
    # circular. job() needs only `producer` (a string) at runtime.
    from artifacts.core.job import Job

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)  # config.py is at the repo root; artifacts/core/artifact.py is two levels down
from config import LAB_COMMIT_ENV, STORAGE, get_git_commit

MANIFEST = "manifest.json"


@cache
def _head() -> str:
    """Returns the commit stamped on every artifact built in this process.

    Cached, so one session cannot stamp two commits onto one tree; restart the
    kernel to pick up new work. The lab container has no git binary and no
    .git, so there the commit comes from LAB_COMMIT_ENV, which main.py bakes in.
    """
    override = os.environ.get(LAB_COMMIT_ENV)
    return override if override else get_git_commit()


def _digest(*parts: object) -> str:
    """Returns a 10-character hash of `parts`, which subclasses build their
    `uid` and `artifact_path` from.

    Hashes key-sorted JSON, so equal parameters give an equal id in every
    process. Every part must be JSON-serializable.

    _digest(("<pad>",), ["a", "b"]) -> "9f2b1c40a7"
    """
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _root(root: Path | None) -> Path:
    """Returns `root`, or STORAGE when it is None, checked to be an absolute Path.

    Every public entry point goes through this so a bad root fails at the
    call. A string is rejected rather than converted, so one type reaches every
    join, glob and relative_to downstream; a relative root would be discarded
    by `root / artifact_path` and surface as a missing file somewhere else.
    """
    resolved = root if root is not None else STORAGE
    if not isinstance(resolved, Path):
        raise TypeError(f"root must be a Path, got {type(resolved).__name__} {resolved!r}")
    if not resolved.is_absolute():
        raise ValueError(f"root must be an absolute path, got {resolved}")
    return resolved


@dataclass(frozen=True)
class Resources:
    """What a job should be given to run this artifact; None means Modal's
    platform default for that dimension. Never identity.

    Resources(gpu_type="A100", gpu_count=2)
    """

    cpu: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


@dataclass(frozen=True)
class Footprint:
    """What one root holds for one artifact, as files: its manifest, its own
    outputs by description, and the files that make it complete by path.

    For most artifacts the last two name the same files. A virtual artifact
    owns no outputs and borrows completion from its dependencies, which is why
    "is anything of mine written" and "am I done" are separate questions.
    """

    manifest: bool
    outputs: dict[str, bool]
    completion: dict[Path, bool]

    @property
    def complete(self) -> bool:
        """Returns whether every file that makes this artifact done is there.

        Ask this, not `all(outputs.values())`: the two agree for everything
        that owns its bytes and differ exactly for virtual artifacts, where
        asking the wrong one is silently wrong.
        """
        return all(self.completion.values())


# frozen: identity is its fields, so it must be hashable/immutable
@dataclass(frozen=True)
class Artifact(ABC):
    # Neither of these says which artifact this is: compare=False keeps both
    # out of ==/hash, so an artifact rebuilt from an old manifest is still the
    # same artifact. kw_only, or a defaulted base field would force defaults
    # on every subclass.
    commit: str = field(default_factory=_head, compare=False, repr=False, kw_only=True)
    allocated_resources: Resources = field(
        default_factory=Resources, compare=False, repr=False, kw_only=True
    )

    # The dotted path of the Job class that produces this artifact
    # ("artifacts.sources.jobs.SourceURLJob"), or None for an artifact nothing
    # produces, whose completion is its dependencies' files.
    #
    # A string, never the class: an artifact knowing its producer must not
    # mean importing it, since a family's jobs.py pulls in torch or numpy and
    # resolving, inspecting or drawing a graph must stay free of that. `job()`
    # is the one place the string becomes a class.
    producer: ClassVar[str | None]

    def __init_subclass__(cls, **kwargs):
        """Rejects a subclass that sets `producer` to anything but a string or None.

        Catches a wrong value when the class is defined, not when its job is
        first located. It cannot catch a missing one: bases like sources.Source
        leave `producer` for their subclasses to set, so __post_init__ checks
        presence once a class is actually instantiated.
        """
        super().__init_subclass__(**kwargs)
        producer = cls.__dict__.get("producer")
        if producer is not None and not isinstance(producer, str):
            raise TypeError(
                f"{cls.__name__} must set `producer` to the dotted path of the Job "
                f'class that produces it -- producer: ClassVar[str] = "artifacts.'
                f'sources.jobs.SourceURLJob" -- or to None if nothing does'
            )

    def __post_init__(self) -> None:
        """Checks the class sets `producer`, holds every field to the field
        schema, and replaces each parameter with what its manifest decodes to.

        Instantiation is the first point where a missing `producer` is an
        error. A set-valued dependency accepts any iterable, and two members
        surviving it with one artifact_path are conflicting definitions of one
        folder. A subclass with its own __post_init__ must call super(), or all
        of this is skipped.
        """
        cls = type(self)
        if not hasattr(cls, "producer"):
            raise TypeError(
                f"{cls.__name__} must set `producer` to the dotted path of "
                f"the Job class that produces it, or to None if nothing does"
            )
        deps = schema.dependencies(cls)
        for name, annotation in schema.annotations(cls).items():
            value, path = getattr(self, name), f"{cls.__qualname__}.{name}"
            if name not in deps:
                object.__setattr__(self, name, manifest.normalize(value, annotation, path))
                continue
            container = deps[name]
            if container is frozenset:
                value = frozenset(value)
                object.__setattr__(self, name, value)
            elif container is tuple and not isinstance(value, tuple):
                raise TypeError(f"{path}: expected {annotation}, got {type(value).__name__}")
            arms = schema.arms(get_args(annotation)[0] if container else annotation)
            for item in value if container else (value,):
                if not isinstance(item, arms):
                    raise TypeError(f"{path}: expected {annotation}, got {type(item).__name__}")
            if container is frozenset:
                paths = Counter(item.artifact_path for item in value)
                repeated = sorted(str(p) for p, n in paths.items() if n > 1)
                if repeated:
                    raise ValueError(f"{cls.__name__}.{name} holds more than one definition at {', '.join(repeated)}")

    def deps(self) -> list[Artifact]:
        """Returns the artifacts this one is built from, one level deep:
        dependency fields in name order, set members sorted by artifact path.

        resolve walks this to build the graph, so the order must not depend
        on set iteration; sorting is what makes it deterministic.
        """
        return [
            dep
            for name, container in schema.dependencies(type(self)).items()
            for dep in manifest.members(getattr(self, name), container)
        ]

    def to_manifest(self) -> dict:
        """Returns this artifact as a manifest dict: type, commit, resources,
        own parameters, and a nested manifest per dependency.

        It is what manifest.json at artifact_path holds, and from_manifest
        rebuilds the artifact from it alone (see artifacts.core.manifest).
        """
        return manifest.to_manifest(self)

    def parameters(self) -> dict:
        """Returns this artifact's own fields as JSON: the `"parameters"` half
        of `to_manifest`, without the dependencies' manifests.

        For display (launcher/state.py), where every dependency's nested
        manifest would bury the fields that tell one artifact from another.
        """
        return manifest.parameters(self)

    @staticmethod
    def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
        """Returns the artifact a manifest describes and the tree under it, unbound.

        `memo`, one dict shared across calls, hands back the same instance for
        a manifest decoded before, so a subtree many artifacts share is decoded
        once (see artifacts.core.manifest.from_manifest).
        """
        return manifest.from_manifest(data, memo)

    @staticmethod
    def load(
        artifact_path: Path | str, root: Path | None = None, memo: dict[str, Artifact] | None = None
    ) -> Artifact:
        """Returns the artifact declared at `root / artifact_path`, always
        unbound, even when every one of its files is there; bind it to use them.

        Raises unless the manifest rebuilds an artifact whose artifact_path is
        the folder it was read from. A path is a pure function of parameters,
        so a disagreeing manifest was copied or renamed, and nothing beside it
        is what it claims. `memo` is `from_manifest`'s.
        """
        wanted = Path(artifact_path)
        if wanted.is_absolute() or ".." in wanted.parts:
            raise ValueError(f"artifact_path must stay under root, got {wanted}")
        path = _root(root) / wanted / MANIFEST
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"nothing declared at {path}") from None
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not readable JSON: {error}") from error
        try:
            artifact = Artifact.from_manifest(data, memo)
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} is not a readable manifest: {error!r}") from error
        if artifact.artifact_path != wanted:
            raise ValueError(f"{path} describes {artifact.artifact_path}, not {wanted}")
        return artifact

    def status(self, root: Path | None = None) -> Footprint:
        """Returns a Footprint of what `root` holds for this artifact, as files.

        A filesystem observation and nothing else: no manifest parsed, no
        definition compared, no lease read. What a footprint means (declared,
        partial, done, conflicting) is lab.declare's to say, against a request
        this method never sees.
        """
        at = _root(root)
        return Footprint(
            manifest=(at / self.artifact_path / MANIFEST).is_file(),
            outputs={desc: path.is_file() for desc, path in self.paths(at).items()},
            completion={path: path.is_file() for path in self.completion_paths(at)},
        )

    def bind(self, root: Path | None = None) -> Self:
        """Returns this artifact rebuilt from the manifest stored at `root`,
        with its files loaded onto it, ready to be used rather than just named.

        The stored manifest is the authority on what lives at a path: a
        definition disagreeing with it raises rather than being handed someone
        else's outputs, and the result carries the stored commit and resources.
        The object bind is called on is never modified.
        """
        at = _root(root)
        stored = Artifact.load(self.artifact_path, at)
        if stored != self:
            raise ValueError(
                f"{self.artifact_path} holds a different definition -- bind what is "
                f"declared there, or declare this one at its own path"
            )
        completion = stored.status(at).completion
        missing = [str(path) for path, there in completion.items() if not there]
        if missing:
            raise FileNotFoundError(f"{self.uid} is not built -- missing {missing}")
        stored._load(at)
        return stored

    def _load(self, root: Path) -> None:
        """Populates the in-memory state this artifact's own methods need from
        its files; a subclass hook that does nothing by default.

        Only bind calls it, after confirming every completion file is there,
        so an override may assume its files exist.
        """

    @property
    @abstractmethod
    def uid(self) -> str:
        """Returns a readable id derived from this artifact's own parameters,
        for naming it to people: logs, errors, the dashboard."""

    @property
    @abstractmethod
    def artifact_path(self) -> Path:
        """Returns the folder this artifact owns, relative to root.

        Must be a pure function of the identifying fields: equal artifacts
        share a folder, and load rejects a manifest found at any other path.
        """

    @property
    @abstractmethod
    def files(self) -> dict[str, str]:
        """Returns the filenames inside artifact_path, keyed by a short
        description of each (e.g. "weights", "config").

        Names, not paths, so paths() can join them to any root: the volume or
        a test folder.
        """

    def paths(self, root: Path) -> dict[str, Path]:
        """Returns each of `files` as a full path under `root`, keyed the same way.

        These are where this artifact's job may write; completion_paths, not
        this, says when it is done.
        """
        return {
            desc: root / self.artifact_path / name for desc, name in self.files.items()
        }

    def job(self) -> Job:
        """Returns the Job that produces this artifact, imported from `producer`.

        The one place that dotted path becomes a class. Its only caller is
        main.run_job, inside the container about to run it, so nothing that
        resolves, inspects or draws a graph imports a family's jobs.py.
        """
        if self.producer is None:
            raise ValueError(
                f"{type(self).__name__} declares no producer -- nothing runs it, its "
                f"completion follows the files it depends on"
            )
        return locate(self.producer)(self)

    def completion_paths(self, root: Path) -> list[Path]:
        """Returns the paths whose presence means this artifact is done.

        Its own files, for almost everything. A virtual artifact owning no
        bytes (mappeddataset.MappedDataSet) overrides this to point at its
        dependencies' files, which is why it is separate from `paths()`, the
        question of where this artifact may write.
        """
        return list(self.paths(root).values())

    def durable_progress(self, root: Path) -> dict | None:
        """Returns how far along the volume says this artifact is, as how many
        of its completion paths exist, or None when there is nothing to count.

        The dashboard shows this when no call is running, and the call's own
        self-report (system.runtime.Worker.progress) when one is. A subclass
        whose job writes intermediate state overrides it: SGD.Training reports
        its furthest checkpoint, which says more than "0 of 1 files".
        """
        completion = self.status(root).completion
        if not completion:
            return None
        return {
            "phase": "files",  # says what is being counted, as a job's own does
            "done": sum(completion.values()),
            "total": len(completion),
        }
```

### artifacts/core/schema.py

Reading field annotations.

```python
"""The field schema: what each field of an artifact or parameter record may
hold, read off its annotations.

A parameter is a scalar, a Literal, a tuple with item types, a frozen
dataclass of these, or a union of them with at most one dataclass; a
dependency is A, A | B | None, tuple[A, ...] or frozenset[A] of artifacts. A
field annotated with a record class that declares `polymorphic = True` (such
as `Model`) may hold any subclass of it, and its manifest carries the
subclass's dotted path under "class". manifest.py encodes against this schema,
and Artifact.__post_init__ checks every field against it.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import cache
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints


@cache
def is_artifact(cls: object) -> bool:
    """Returns whether `cls` is an Artifact subclass.

    Artifact is imported at call time, because artifact.py imports this module.
    """
    from artifacts.core.artifact import Artifact

    return isinstance(cls, type) and issubclass(cls, Artifact)


def arms(annotation: object) -> tuple:
    """Returns the arms of a union annotation, None included, or the
    annotation alone."""
    return get_args(annotation) if get_origin(annotation) in (UnionType, Union) else (annotation,)


@cache
def dependencies(cls: type) -> dict[str, type | None]:
    """Returns which of `cls`'s fields hold artifacts, and what holds them:
    `tuple`, `frozenset`, or None for a single artifact. Sorted by field name;
    a parameter record has none.

    The container declares whether order identifies the artifact: a tuple is a
    sequence, so its order and its repeats are part of what the artifact is; a
    frozenset is a set, so neither is. Read off annotations, not values, so an
    empty `sources=frozenset()` is still a dependency.
    """
    if not is_artifact(cls):
        return {}
    hints = get_type_hints(cls)
    found: dict[str, type | None] = {}
    for f in sorted(fields(cls), key=lambda f: f.name):
        annotation, container = hints[f.name], get_origin(hints[f.name])
        if container in (tuple, frozenset):
            annotation = get_args(annotation)[0]
        else:
            container = None
        held = [arm for arm in arms(annotation) if arm is not type(None)]
        if held and all(map(is_artifact, held)):
            found[f.name] = container
    return found


@cache
def annotations(cls: type) -> dict[str, object]:
    """Returns `cls`'s field annotations, each checked to be a dependency or a
    parameter type the codec rebuilds exactly (see `_check_parameter`)."""
    hints, deps = get_type_hints(cls), dependencies(cls)
    for f in fields(cls):
        if f.name not in deps:
            _check_parameter(hints[f.name], f"{cls.__qualname__}.{f.name}")
    return {f.name: hints[f.name] for f in fields(cls)}


def polymorphic(annotation: object) -> tuple[type, ...]:
    """Returns the record class in `annotation` that declares `polymorphic = True`,
    as a one-tuple, or () if it names none: a field so annotated may hold any
    subclass of it."""
    return tuple(arm for arm in arms(annotation) if is_dataclass(arm) and getattr(arm, "polymorphic", False))


def _check_parameter(annotation: object, where: str) -> None:
    """Raises unless `annotation` is a scalar, a Literal, a tuple with item
    types, a frozen dataclass of these, or a union with at most one dataclass."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (bool, int, float, str, type(None)) or origin is Literal:
        return
    if len(arms(annotation)) > 1:
        if sum(map(is_dataclass, args)) > 1:
            raise TypeError(
                f"{where}: a union holds at most one record type; a manifest "
                f"stores a union's record as its fields alone"
            )
        for arm in args:
            _check_parameter(arm, where)
        return
    if origin is tuple and args:
        for arg in args:
            if arg is not Ellipsis:
                _check_parameter(arg, where)
        return
    if is_artifact(annotation):
        raise TypeError(
            f"{where}: artifact hidden in a parameter; a dependency is annotated "
            f"A, A | B | None, tuple[A, ...] or frozenset[A]"
        )
    if is_dataclass(annotation) and annotation.__dataclass_params__.frozen:
        annotations(annotation)
        return
    raise TypeError(
        f"{where}: annotated {annotation}, but the codec needs a concrete type: "
        f"a scalar, Literal, tuple[T, ...], union, or frozen dataclass"
    )
```

### artifacts/core/manifest.py

The codec: `to_manifest`, `from_manifest`.

```python
"""Artifact <-> manifest dict <-> bytes on disk.

A parameter is encoded and decoded by pydantic against its field's annotation
(artifacts.core.schema), so its annotation alone says how to rebuild it, except
in a polymorphic field, whose record names its own class under "class"; a
dependency is a nested manifest naming its own class. `normalize` is the round
trip every constructor runs, so a concrete artifact already equals what its
manifest decodes to. Nothing here touches storage.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from artifacts.core.locate import dotted, locate
from artifacts.core.schema import annotations, dependencies, is_artifact, polymorphic

if TYPE_CHECKING:
    from artifacts.core.artifact import Artifact

# strict: no str as int, bool as int or list as tuple, though an int still
# becomes a float. revalidate_instances: a record is checked field by field,
# never trusted for its type.
_STRICT = ConfigDict(
    strict=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
)


@cache
def _field(annotation: object) -> type[BaseModel]:
    """Returns a model whose one field, `value`, is annotated `annotation`.

    pydantic takes no config for a bare dataclass, while a model's config
    reaches every record nested in its field.
    """
    return create_model("Field", __config__=_STRICT, value=(annotation, ...))


@contextmanager
def _named(path: str):
    """Reraises a ValidationError as a TypeError, or a ValueError where a
    validator refused the value, naming `path` and the field inside it."""
    try:
        yield
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        inside = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in first["loc"][1:])
        kind = ValueError if first["type"] in ("value_error", "finite_number") else TypeError
        raise kind(f"{path}{inside}: {first['msg']}") from error


def _dump(value: object, annotation: object) -> object:
    """Returns `value`, annotated `annotation`, as JSON data with keys sorted.

    Adding 0.0 turns -0.0 into 0.0: the two compare equal, so they must encode
    equal.
    """
    text = _field(annotation).model_construct(value=value).model_dump_json()
    return json.loads(
        text,
        parse_float=lambda s: float(s) + 0.0,
        object_pairs_hook=lambda pairs: dict(sorted(pairs)),
    )["value"]


def _load(data: object, annotation: object, path: str) -> object:
    """Returns the value JSON `data` encodes as `annotation`, validated
    strictly; `path` names it in an error."""
    with _named(path):
        return _field(annotation).model_validate_json(json.dumps({"value": data})).value


def _encode(value: object, annotation: object) -> object:
    """Returns a field's `value` as JSON data; a record in a polymorphic field
    is dumped as its own class, which it names under "class"."""
    if isinstance(value, polymorphic(annotation)):
        # "class" is a Python keyword, so no field can be named it
        return dict(sorted({"class": dotted(type(value)), **_dump(value, type(value))}.items()))
    return _dump(value, annotation)


def _decode(data: object, annotation: object, path: str) -> object:
    """Returns the field value JSON `data` encodes; a polymorphic field's record
    is loaded as the class it names, which must subclass the field's."""
    bases = polymorphic(annotation)
    if not bases or data is None:
        return _load(data, annotation, path)
    data = dict(data)
    cls = locate(data.pop("class"))
    if not issubclass(cls, bases):
        raise TypeError(f"{path}: {dotted(cls)} is not a {bases[0].__qualname__}")
    return _load(data, cls, path)


def normalize(value: object, annotation: object, path: str) -> object:
    """Returns `value` as its manifest decodes it: ints in float fields as
    floats, -0.0 as 0.0, records rebuilt through their constructors.

    Raises unless `value` is exactly an `annotation`: a bool is not an int, a
    list is not a tuple, and a subclass is not the record its field names
    unless that record is polymorphic.
    """
    record = type(value) if isinstance(value, polymorphic(annotation)) else annotation
    with _named(path):
        validated = _field(record).model_validate({"value": value}).value
    restored = _decode(_encode(validated, annotation), annotation, path)
    if restored != value:
        expected = getattr(annotation, "__qualname__", annotation)
        raise TypeError(f"{path}: expected exactly {expected}, got {value!r}")
    return restored


def members(value: object, container: type | None) -> list[Artifact]:
    """Returns a dependency field's artifacts in manifest order: a set's
    sorted by artifact path, never by set iteration."""
    if container is None:
        return [] if value is None else [value]
    return sorted(value, key=lambda a: a.artifact_path.as_posix()) if container is frozenset else list(value)


def parameters(artifact: Artifact) -> dict:
    """An artifact's own fields as JSON, sorted by name: the `"parameters"`
    half of its manifest, without the dependencies' manifests.

    The parameters/dependencies split is read off the annotations, so it is
    the same split regardless of what a field happens to hold. `commit` and
    `allocated_resources` are neither, being about running and not state.
    """
    cls = type(artifact)
    deps = dependencies(cls)
    return {
        name: _encode(getattr(artifact, name), annotation)
        for name, annotation in sorted(annotations(cls).items())
        if name not in deps and name not in ("commit", "allocated_resources")
    }


def to_manifest(artifact: Artifact) -> dict:
    """An artifact as a dictionary, enough to rebuild it from nothing else:
    its type, its commit, its resources, its own parameters, and a manifest
    apiece for the artifacts it's built from.

    Key order is fixed -- the five here as written, everything below sorted
    -- so the file stays a pure function of the artifact even when a
    dataclass's fields are reordered. A set of artifacts is an array in
    `members` order, since JSON has no set type.
    """
    cls = type(artifact)
    deps = {}
    for name, container in dependencies(cls).items():
        value = getattr(artifact, name)
        if container is None:
            deps[name] = None if value is None else to_manifest(value)
        else:
            deps[name] = [to_manifest(item) for item in members(value, container)]
    return {
        "artifact": dotted(cls),
        "commit": artifact.commit,
        "allocated_resources": _encode(
            artifact.allocated_resources, annotations(cls)["allocated_resources"]
        ),
        "parameters": parameters(artifact),
        "dependencies": deps,
    }


def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
    """The artifact a manifest describes and the tree under it, unbound.

    Each node keeps the commit recorded for it, so a subtree produced by
    older code stays visibly older.

    `memo` maps a manifest's JSON to the artifact it decoded to. One dict
    across many calls hands back the same instance for a subtree they share
    (the dataset under every leg of a run, each leg under the next) instead
    of decoding it again; the instances are frozen, so sharing them changes
    nothing but the work.
    """
    key = json.dumps(data, sort_keys=True) if memo is not None else None
    if key is not None and key in memo:
        return memo[key]
    cls = locate(data["artifact"])
    if not is_artifact(cls):
        raise TypeError(f"manifest target must be an Artifact, got {cls}")
    hints, deps = annotations(cls), dependencies(cls)
    own = {"commit": data["commit"], "allocated_resources": data["allocated_resources"], **data["parameters"]}
    inputs = {name: _decode(value, hints[name], f"{cls.__qualname__}.{name}") for name, value in own.items()}
    for name, value in data["dependencies"].items():
        if deps[name]:
            inputs[name] = deps[name](from_manifest(item, memo) for item in value)
        else:
            inputs[name] = None if value is None else from_manifest(value, memo)
    artifact = cls(**inputs)
    if key is not None:
        memo[key] = artifact
    return artifact


def manifest_json(manifest: dict) -> str:
    """One manifest as the bytes that belong on disk.

    allow_nan=False because Python writes NaN and Infinity as bare words and
    reads them back, while no other JSON reader accepts either.
    """
    return json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
```

### artifacts/core/locate.py

A class to its import path and back.

```python
"""Resolve a dotted path (e.g. "torch.float32", "torch.optim.AdamW",
"numpy.float32") to the live object it names, importing modules along the
way as needed.

Lets artifact definitions store framework types -- torch dtypes, optimizer
classes, numpy dtypes, and so on -- as plain strings, so an artifact module
never imports torch/numpy just to type a field. The actual import only
happens where the string gets resolved, which is always worker-side code
where the framework is guaranteed installed.
"""

import importlib
from typing import Any


def locate(path: str) -> Any:
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        prefix = ".".join(parts[:split])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as exc:
            # exc.name is the specific module Python couldn't find -- this
            # prefix itself, or (when a middle segment is missing) a shorter
            # one Python needed just to look for this prefix. Either way,
            # nothing of this prefix exists -- try a shorter one. Anything
            # else means a module along the way was found and failed on its
            # own terms, which must surface, not get silently treated as
            # "keep shrinking the prefix".
            if exc.name == prefix or prefix.startswith(exc.name + "."):
                continue
            raise
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"could not resolve {path!r}")


def dotted(cls: type) -> str:
    """Returns the dotted path `locate` resolves back to `cls`."""
    return f"{cls.__module__}.{cls.__qualname__}"
```

### artifacts/core/resolve.py

Dependency resolution.

```python
"""Pure dependency resolution: the artifacts a request is built from, in the
order they have to exist. Reads no storage and decides nothing about running.
"""

from __future__ import annotations

from pathlib import Path

from artifacts.core.artifact import Artifact


def resolve(artifact: Artifact) -> list[Artifact]:
    """Every artifact `artifact` is built from and then itself, dependencies
    before dependents, each path once.

    A path reached twice must hold an agreeing definition, and `==` compares
    the whole subtree, so a shared path cannot hide a conflicting leaf. The
    first object reaching a path is the one kept, recorded fields and all.
    """
    visited: dict[Path, Artifact] = {}
    active: list[Path] = []  # the current descent, for cycle detection

    def visit(node: Artifact) -> None:
        path = node.artifact_path
        if path in visited:
            if visited[path] != node:
                raise ValueError(f"different definitions at {path}")
            return
        if path in active:
            chain = " -> ".join(str(p) for p in (*active, path))
            raise ValueError(f"cycle: {chain}")
        active.append(path)
        for dep in node.deps():
            visit(dep)
        active.pop()
        visited[path] = node

    visit(artifact)
    return list(visited.values())
```

### lab.py

Declaration: `declare`, `_inspect`, the report.

```python
"""The notebook API. One entry point, declaration:

    import lab
    report = lab.declare(tokenizer)               # preview, writes nothing
    report = lab.declare(tokenizer, commit=True)  # publish missing manifests

A committed declaration of a run also copies the notebook it ran from to
`runs/<run_id>/declare-<uid>.ipynb`, so the volume keeps what declared it.

Construction, loading and binding stay on the artifact classes --
`Tokenizer(...)`, `Artifact.load(path)`, `tokenizer.bind()` -- and every one
of them reads STORAGE unless handed another root. Nothing to initialize.

Two volume verbs for the lab container: `refresh()` reloads the mount so
files written elsewhere (a job's outputs, a notebook saved in another lab)
appear; `save()` commits it so a notebook saved in JupyterLab outlives the
container. `worker` is a stand-in execution context for running one job by
hand from a cell, `job.run(root, lab.worker)`: no lease, no heartbeat, log to
the console.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import modal

from artifacts.core.artifact import MANIFEST, Artifact
from artifacts.core.manifest import manifest_json
from artifacts.core.resolve import resolve
from artifacts.core.SGD.training import Training
from config import STORAGE, VOLUME_NAME
from system.runtime import Worker

# States a person has to look at before anything is written.
BLOCKING = ("conflict", "undeclared")


class DeclarationError(ValueError):
    """A commit refused over blocking rows; `report` holds them.

    `args` is (report, verbose) rather than the rendered text so the error
    pickles: it crosses back from a container when declaring remotely.
    """

    def __init__(self, report: DeclarationReport, verbose: bool = False):
        super().__init__(report, verbose)
        self.report = report

    def __str__(self) -> str:
        return self.report.render(self.args[1])


@dataclass
class DeclarationReport:
    """One row per resolved path: path, state, drift, differences, created,
    updated, and the stored and requested commits and resources. Counts and
    the verdict are derived."""

    rows: list[dict]
    strict_commit: bool = False

    @property
    def blockers(self) -> list[dict]:
        return [
            row
            for row in self.rows
            if row["state"] in BLOCKING or (self.strict_commit and row["drift"])
        ]

    def render(self, verbose: bool = False) -> str:
        """The report as printed; `verbose` adds each blocker's differences."""
        width = max(len(row["path"]) for row in self.rows) + 2
        blocked = {row["path"] for row in self.blockers}
        lines = []
        for row in self.rows:
            notes = []
            if row["drift"]:
                stored, requested = row["commit"]
                notes.append(f"drift: {stored[:7]} -> {requested[:7]}")
            stored, requested = row["resources"]
            if stored is not None and stored != requested:
                verb = "updated" if row["updated"] else "differ, requested apply on commit"
                notes.append(f"resources {verb}: {_set(stored)} -> {_set(requested)}")
            if row["created"]:
                notes.append("created")
            line = f"{row['path']:<{width}}{row['state']:<10}"
            lines.append(line.rstrip() + (f"  ({'; '.join(notes)})" if notes else ""))
            if verbose and row["path"] in blocked:
                for key, (on_disk, requested) in sorted(row["differences"].items()):
                    lines.append(
                        f"    {key}: on disk {json.dumps(on_disk)}, "
                        f"requested {json.dumps(requested)}"
                    )
        counts = Counter(row["state"] for row in self.rows)
        summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
        written = [row["path"] for row in self.rows if row["created"] or row["updated"]]
        if self.blockers:
            verdict = f"BLOCKED -- {len(self.blockers)} to resolve"
            if not verbose:
                verdict += " (verbose=True to see why)"
        elif written:
            plural = "s" if len(written) != 1 else ""
            verdict = f"{len(written)} manifest{plural} written: {', '.join(written)}"
        else:
            verdict = f"ok -- {counts['new']} to declare"
        return "\n".join([*lines, "", summary, verdict])

    __str__ = render


def _set(resources: dict) -> str:
    """The dimensions a Resources dict sets, as JSON: `{"gpu_type": "A100"}`."""
    return json.dumps({k: v for k, v in resources.items() if v is not None})


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """A manifest flattened to dotted leaf paths, commits and resources left
    out at every level: neither is definition."""
    if isinstance(value, dict) and value:
        out: dict[str, object] = {}
        for key, item in value.items():
            if key in ("commit", "allocated_resources"):
                continue
            out.update(_leaves(item, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, item in enumerate(value):
            out.update(_leaves(item, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def _inspect(artifact: Artifact, root: Path, memo: dict[str, Artifact]) -> dict:
    """One report row: how what `root` holds at this artifact's path compares
    with the artifact requested. Every difference is `key: [on disk, requested]`.
    `memo` is `Artifact.load`'s, shared across one declaration's rows so a
    subtree many manifests embed decodes once."""
    footprint = artifact.status(root)
    row = {
        "path": artifact.artifact_path.as_posix(),
        "state": "new",
        "drift": False,
        "differences": {},
        "created": False,
        "updated": False,
        "commit": [None, artifact.commit],
        "resources": [None, asdict(artifact.allocated_resources)],
    }
    if not footprint.manifest:
        present = [desc for desc, there in footprint.outputs.items() if there]
        if present:
            row["state"] = "undeclared"
            row["differences"] = {f"outputs.{desc}": ["present", None] for desc in present}
        return row
    try:
        stored = Artifact.load(artifact.artifact_path, root, memo)
    except ValueError as error:
        row["state"] = "conflict"
        row["differences"] = {"manifest": [str(error), None]}
        return row
    row["commit"] = [stored.commit, artifact.commit]
    row["drift"] = stored.commit != artifact.commit
    row["resources"] = [asdict(stored.allocated_resources), asdict(artifact.allocated_resources)]
    if stored != artifact:
        row["state"] = "conflict"
        on_disk, wanted = _leaves(stored.to_manifest()), _leaves(artifact.to_manifest())
        row["differences"] = {
            key: [on_disk.get(key), wanted.get(key)]
            for key in sorted(on_disk.keys() | wanted.keys())
            if on_disk.get(key) != wanted.get(key)
        }
        return row
    present = list(footprint.completion.values())
    row["state"] = "done" if all(present) else "partial" if any(present) else "declared"
    return row


def _volume():
    """The volume handle, built on demand: at module scope it would be a
    network call the moment anything imports `lab`, image staging included."""
    return modal.Volume.from_name(VOLUME_NAME)


def _outside_volume(op):
    """Run a volume operation with cwd stepped outside the mount first.

    Modal counts a cwd under the volume as an open file, and a notebook rooted
    there would otherwise fail every reload and commit.
    """
    cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        return op()
    finally:
        os.chdir(cwd)


def refresh() -> None:
    """Reloads the volume, so files written outside this container appear.
    Only a container with the volume mounted can do this; Modal refuses
    anywhere else."""
    _outside_volume(_volume().reload)


def save() -> None:
    """Commits the volume, so what is on disk here outlives this container:
    save the notebook in JupyterLab first, then `lab.save()`. Only a
    container with the volume mounted can do this; Modal refuses anywhere
    else."""
    _outside_volume(_volume().commit)


def current_notebook() -> bytes | None:
    """The notebook file this kernel runs, or None outside a notebook.

    jupyter_server starts every kernel with JPY_SESSION_NAME set to its
    notebook's path; VS Code leaves `__vsc_ipynb_file__` in the namespace
    instead. A notebook renamed since its kernel started is a missing file
    here, which raises rather than being skipped.
    """
    path = os.environ.get("JPY_SESSION_NAME")
    if path is None:
        try:
            from IPython import get_ipython
        except ImportError:  # a worker image: no notebook, by construction
            return None
        kernel = get_ipython()
        path = kernel.user_ns.get("__vsc_ipynb_file__") if kernel else None
    if path is None:
        return None
    return (STORAGE / path).read_bytes()  # absolute `path` wins the join


def declare(
    artifact: Artifact,
    *,
    root: Path = STORAGE,
    commit: bool = False,
    strict_commit: bool = False,
    verbose: bool = False,
    notebook: bytes | None = None,
) -> DeclarationReport:
    """The declaration report for `artifact` and everything it is built from,
    printed and returned; with `commit`, missing manifests are published first
    and existing ones whose resources differ from those requested take the
    requested ones, everything else in them kept.

    Preview writes nothing and returns blockers as rows. Commit refuses over
    any blocker -- a conflict, an undeclared folder, or drift under
    `strict_commit` -- and raises DeclarationError with the report attached
    before touching disk. `root` is
    for tests and scripts; a notebook gets STORAGE. The volume is reloaded
    before inspection and committed after writes only when this runs inside
    a container and `root` is its mount: anywhere else `/storage` is a plain
    folder with nothing behind it to reload.

    A commit that creates a run's artifact (one with a `run_id`) also writes
    the declaring notebook to `runs/<run_id>/declare-<uid>.ipynb`: the
    kernel's own by default, or the bytes handed in as `notebook` by a
    caller that has them and no kernel (`local.declare_on_volume`).
    """
    on_volume = root == STORAGE and not modal.is_local()
    graph = resolve(artifact)
    if commit and notebook is None:
        notebook = current_notebook()
    if on_volume:
        refresh()
    memo: dict[str, Artifact] = {}
    rows = [_inspect(node, root, memo) for node in graph]
    report = DeclarationReport(rows, strict_commit)
    if commit:
        if report.blockers:
            raise DeclarationError(report, verbose)
        for node, row in zip(graph, rows):
            stored, requested = row["resources"]
            if row["state"] == "new":
                manifest = node.to_manifest()
            elif stored != requested:
                on_disk = Artifact.load(node.artifact_path, root, memo)
                manifest = replace(on_disk, allocated_resources=node.allocated_resources).to_manifest()
            else:
                continue
            path = root / node.artifact_path / MANIFEST
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(manifest_json(manifest))
            if row["state"] == "new":
                row.update(_inspect(node, root, memo), created=True)
            else:  # keep the pair, so the report shows what changed to what
                row.update(_inspect(node, root, memo), updated=True, resources=[stored, requested])
        if rows[-1]["created"] and notebook is not None and isinstance(artifact, Training):
            copy = root / "runs" / artifact.run_id / f"declare-{artifact.uid}.ipynb"
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(notebook)
        if on_volume:
            save()
    print(report.render(verbose))
    return report


# A job run by hand from a cell is not a Modal call: nothing races it for a
# lease, and its log has nowhere to be filed but the console.
_log = logging.getLogger("lab")
_log.setLevel(logging.INFO)
if not _log.handlers:  # module-level, so this only ever runs once per process
    _log.addHandler(logging.StreamHandler())

worker = Worker(
    artifact_path="lab",
    call_id="lab",
    log=_log,
    confirm_lease=lambda *_, **__: None,
    progress={},
)
```
