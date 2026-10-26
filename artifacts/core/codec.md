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
