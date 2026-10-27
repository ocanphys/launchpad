# Launchpad: Unified Artifact Architecture and API Specification

**Date:** October 26, 2026
**Status:** Design checkpoint, revised against the codebase (branch `pydantic-artifact-baseclass`, commit `ce7a2d0`)
**Scope:** Artifact and model validation with pydantic, dependency resolution, the manifest codec, storage, declaration, the implementation order, and folding every description of the code into `docs/`.
**Companion:** `notes-scratch/launchpad_identity_migration_spec.md`, which starts after this one has landed and changes identity and the manifest format on purpose.

This document began as a proposal written without the codebase in view. This revision checks each proposal against the current code. Where the two disagree, the section says what the code does today, what this document recommends, and why. Every pydantic behavior marked **(verified)** was checked against the installed pydantic 2.13.4 in a scratch script. The baseline before any change is 142 passing tests across `test_codec`, `test_artifact`, `test_resolve`, `test_declare`, `test_dataset`, `test_mappeddataset` and `test_state`.

Part I specifies architecture and behavior. Part II gives the API contracts. Nothing in this spec changes what is on the volume: every artifact path, every manifest byte, and the report and commit semantics stay identical, and golden fixtures and a volume scan prove it after every stage. Changing identity on purpose (taking `device` out of `Model`, deriving `lineage_hash` from the encoded parameters, renaming `commit` to `declaring_commit` and recording it at the root only) is the companion identity spec's job, with the migration that moves the volume to match.

---

# Part I: Architecture and Behavior

## 1. Goal

Make pydantic the one place where what an artifact or model may hold is declared and checked, and delete the code that exists only to apply pydantic to dataclasses. Keep deterministic identities, embedded dependency provenance, and the current declaration report and commit semantics.

### 1.1 Where the code stands

Validation already runs at instantiation, and pydantic already performs it:

- Artifacts (`Source`, `SourceURL`, the tokenizers, `TokenizedSource`, `DataSet`, `MappedDataSet`, `Training`, `Pretraining`) and param configs (`Resources`, `OptimizerParameters`, `LRSchedule`, `TrainingParameters`, `LoopConfig`, `Model` and its subclasses under `models/`) are frozen dataclasses.
- `Artifact.__post_init__` sends every parameter through `manifest.normalize`, which wraps the field's annotation in a one-field pydantic model (`manifest._field`) carrying a strict config. The wrapper exists because pydantic accepts no config for a stdlib dataclass (LESSONS.md, "pydantic is strict only where it is told to be").
- `schema.py` reads annotations to split parameters from dependencies, refuse types the codec cannot rebuild, and find the one polymorphic param config (`Model`).

So the change is not *when* validation runs but *where its rules are written*: on the classes, as pydantic declarations. What gets deleted is the layer in between: all of `schema.py` (114 lines), most of `manifest.py` (from 210 lines to roughly 80), and `Artifact.__post_init__` (37 lines).

## 2. Concepts and invariants

- **Artifact:** a pydantic model for an independently identified and tracked object. Its fields annotated with artifact types are dependency edges.
- **ParamConfig:** a pydantic model nested in an artifact as a parameter, not a dependency node: `Resources`, `TrainingParameters`, `LRSchedule`, `OptimizerParameters`, `LoopConfig`, and the `Model` family. A param config has no folder and no manifest of its own.
- **Manifest:** the persisted JSON form of an artifact, with every dependency's manifest embedded. Its recorded fields, `commit` and `allocated_resources`, are on every node, as today.
- **Definition equality:** equality of concrete class, parameters and dependencies, recursively. Recorded fields do not take part. In this design it is `==`.
- **Artifact path:** the folder an artifact owns, a pure function of its definition, computed by each family's hand-written `uid` and `artifact_path`.

**Invariants:**

- **Round trip:** `from_manifest(to_manifest(a)) == a`, with the same path, and a decoded manifest re-encodes to exactly its own bytes, recorded fields included.
- **Equal artifacts have equal paths and equal definition encodings.** The *definition encoding* is the manifest with the recorded fields (`commit`, `allocated_resources`) left out at every node. Two `==` artifacts can still have different complete manifests, differing in exactly those fields: `==` ignores them, and the manifest records them. So "equal artifacts write equal bytes" is not an invariant, and no code may compare manifest bytes to decide whether two definitions agree. Declaration compares objects with `==`, and when it renders a conflict it already strips the recorded fields at every level (`lab._leaves`), which is the definition encoding.
- A set of dependencies encodes in one order regardless of iteration order; conflicting definitions at one path are refused; recorded fields never change identity.

### 2.1 Definition equality is `==`, not a separate method

The original proposal added `same_definition(other)` and left pydantic's `==` as it is. **Recommendation: make `==` and `hash` on `Artifact` mean definition equality, by overriding both on the base class.**

Why:

- `==` already is definition equality today: the dataclass fields `commit` and `allocated_resources` are declared `compare=False`. Four places rely on it: `resolve` (a path reached twice must hold an equal definition), `Artifact.bind` (the stored definition must equal the requested one), `lab._inspect` (an unequal stored manifest is a conflict), and the tests. A second method would mean every one of those call sites has to remember to use it, and the one that forgets turns a commit difference into a conflict.
- Set-valued dependencies (`frozenset[Source]`) need a `__hash__` that agrees with `==`. Pydantic's generated hash for a frozen model covers every field, `commit` included.
- Pydantic has no per-field `compare=False` (**verified**: `Field` has no `compare` parameter in 2.13.4), so the exclusion has to be an override.

```python
RECORDED = ("commit", "allocated_resources")

class Artifact(BaseModel):
    def _definition(self) -> tuple:
        names = sorted(n for n in type(self).model_fields if n not in RECORDED)
        return type(self), tuple((n, getattr(self, n)) for n in names)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Artifact) and self._definition() == other._definition()

    def __hash__(self) -> int:
        return hash(self._definition())
```

**(verified)** Subclasses inherit the override: pydantic installs its generated `__hash__` only when no base defines one. Two artifacts differing only in `commit` compare equal, hash equal, and collapse to one member of a `frozenset`. Pydantic's own `__eq__` would also compare private attributes, which matters because `bind` attaches runtime state to instances (§4.6); the override compares fields only.

`RECORDED` names the rule once. The codec uses the same tuple to keep recorded fields out of `"parameters"`, which resolves item 3 of codec.md §13 (today the rule is spelled out separately in `parameters`, `to_manifest` and `from_manifest`).

Param configs keep pydantic's default `==`: they have no recorded fields.

## 3. Module structure and contracts

| Module | Today | Contract after this spec | Does not do |
|---|---|---|---|
| `artifacts/core/artifact.py` | `Artifact` dataclass, `Resources`, `Footprint`, `load`, `bind`, `status` | `STRICT`; the `ParamConfig` and `Artifact` bases; `==` and `hash` as definition equality; `dependencies()`; `load`, `status`, `bind` | Encode manifests; write anything |
| `artifacts/core/SGD/training.py`, `models/*/__init__.py` | `Model` and the training param configs as dataclasses | The same classes as `ParamConfig` subclasses, constraints on their fields | Import torch at module scope |
| `artifacts/core/resolve.py` | `resolve(root)` | Unchanged | Storage, encoding |
| `artifacts/core/manifest.py` (the codec) | `normalize`, `_field`, `_encode`, `_decode`, `to_manifest`, `from_manifest`, `parameters`, `manifest_json` | `to_manifest`, `from_manifest`, `parameters`, `manifest_json`, `canonical` | File I/O |
| `artifacts/core/storage.py` | (new; today inside `Artifact.load` and `lab.declare`) | `MANIFEST`, `read_manifest`, `write_manifest` | Decode or encode; decide whether a write is allowed |
| `artifacts/core/schema.py` | Annotation reading and checking | **Deleted** | |
| `artifacts/core/locate.py` | `locate`, `dotted` | Unchanged | |
| `lab.py` | `declare`, `DeclarationReport` | Unchanged in behavior | Encode JSON; validate fields |

Of the original proposal's new modules, `storage.py` is created and the others are not:

- **`storage.py`:** created, small: reading and writing manifest files (§7).
- **`codec.py`:** `manifest.py` already is the codec, behind the seam CLAUDE.md names (`to_manifest`, `from_manifest`). Renaming it is churn with no capability.
- **`config.py`:** the model param configs stay where they are (§4.5).
- **`declare.py`:** `lab.declare` is the declaration service already (§8); a module holding only it adds an import and nothing else.

Where `schema.py`'s responsibilities go:

| `schema.py` today | After |
|---|---|
| `dependencies(cls)`: which fields hold artifacts, and in what container | A cached classmethod on `Artifact` reading `model_fields` (§4.3) |
| `annotations(cls)` and `_check_parameter`: refuse annotations the codec cannot rebuild | Mostly pydantic's own schema building; one small guard remains (§4.4) |
| `polymorphic(annotation)` and the `polymorphic` ClassVar | A codec rule with no flag (§6.3) |
| `is_artifact`, `arms` | `issubclass`, and a two-line union helper shared by `dependencies()` and the codec |

## 4. Pydantic models

### 4.1 One strict config

```python
STRICT = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

class ParamConfig(BaseModel):
    model_config = STRICT

class Artifact(BaseModel):
    model_config = STRICT

class LoopConfig(ParamConfig): ...
class Model(ParamConfig): ...
```

Every param config and every artifact gets the same config, from one constant. A config does not propagate into nested models (each pydantic model validates with its own), so each of the two bases sets it and every subclass inherits it: a new param config, or a new architecture under `models/`, cannot end up with pydantic's lax defaults by forgetting a line.

Pydantic's metaclass derives from `ABCMeta`, so `@abstractmethod` on `uid`, `artifact_path`, `files`, `encode` and `decode` keeps working without `ABC` in the bases (**verified**: instantiating a class with an abstract `uid` raises `TypeError`).

**Recommendation: `strict=True` in the config, not `StrictInt` and `StrictFloat` per field.** The original proposal asked for strict types on each field. Under a strict config, plain `int`, `float`, `bool` and `str` annotations are already strict; per-field types would write the same rule on every field. **(verified)** under this config:

- `True` in an `int` field, `"0.5"` in a `float` field, and `None` in a `str` field are refused.
- An `int` in a `float` field is accepted and stored as a `float`. This keeps today's rule that `max_norm=1` and `max_norm=1.0` are one artifact with one path.
- NaN and infinity are refused (`allow_inf_nan=False`), since JSON cannot carry them.

**Gotcha: an int too large for a float is rounded silently.** `rate=2**53 + 1` is stored as `9007199254740992.0` (**verified**). Today `normalize` refuses it, because the value would not come back exactly (the test_codec case `"expected exactly Shapes"`). **Decided: accept the rounding.** The artifact holds the float it converted to, and equality, path and bytes all follow that float consistently; no hyperparameter comes near 2**53. Drop that test case.

### 4.2 Constraints are declared on fields

Every range or cross-field check moves out of a `__post_init__` and into the class declaration:

| Check | Today | After |
|---|---|---|
| `total_steps > 0` | `Training.__post_init__` | `TrainingParameters.total_steps: int = Field(gt=0)` |
| `checkpoint_every > 0`, `gpu_check_every > 0` | `LoopConfig.__post_init__` | `Field(gt=0)` on each |
| `d_model > 0`, `num_heads > 0` | `Transformer.__post_init__`, in both `models/transformer` and `models/transformerV2` | `Field(gt=0)` |
| `d_model % num_heads == 0` | same | `@model_validator(mode="after")` |
| `model` required without a starting checkpoint, inherited from it otherwise, and must match it | `Training.__post_init__`, with `object.__setattr__` | a `mode="before"` validator and a required field (below) |
| model vocab equals tokenizer vocab; tokenizer equals the starting checkpoint's | `Pretraining.__post_init__` | `@model_validator(mode="after")` |
| one tokenizer across sources; separator token present; `MappedDataSet` not empty | `DataSet.__post_init__`, `MappedDataSet.__post_init__` | `@model_validator(mode="after")` |
| the class sets `producer` | `Artifact.__post_init__` | `@model_validator(mode="after")` on `Artifact` |

**Decided: add positivity constraints wherever a value must be positive, written in the class definition and nowhere else.** That covers every `Transformer` and `Mamba` dimension (`vocab_size`, `sequence_length`, `num_layers`, `d_model`, `d_ff`, `num_heads`, `rope_theta`, `d_state`, `d_conv`, `expand`), `batch_size` and `val_every`. The rule for what qualifies: a constraint is one `Field(...)` argument (`gt`, `ge`, `lt`), or, for a relation between fields such as `d_model % num_heads`, one `after` validator of a few lines. Anything that needs more than that stays out. The list above, plus the existing checks in the table, is the complete set for this spec: optimizer arguments (`lr`, `betas`, `eps`, `weight_decay`) and `max_norm` are left to torch, which validates its own arguments when the optimizer is built. These are new refusals, so the volume scan rerun after stage 1 (§10.2) must confirm that no stored manifest violates them; the stage 0 run cannot, since the constraints do not exist yet.

**`Training.model` becomes a required field, filled in by a before-validator.** Today the field is `model: Model | None = None`, and `__post_init__` copies the starting checkpoint's model in with `object.__setattr__`. After:

```python
class Training(Artifact):
    model: Model
    starting_checkpoint: Training | None = None

    @model_validator(mode="before")
    @classmethod
    def _inherit_model(cls, data):
        if isinstance(data, dict) and data.get("model") is None and data.get("starting_checkpoint") is not None:
            return {**data, "model": data["starting_checkpoint"].model}
        return data

    @model_validator(mode="after")
    def _model_matches_checkpoint(self) -> Self:
        if self.starting_checkpoint is not None and self.model != self.starting_checkpoint.model:
            raise ValueError("model must match starting_checkpoint.model")
        return self
```

**(verified)** A leg built from a starting checkpoint alone gets its model. A leg with neither is refused by pydantic as a missing required field, so the explicit "model is required" check disappears, and `model` is typed as what it always holds after construction. Manifests already write `model` for every leg, so decoding is unaffected.

**Gotcha: pydantic never calls `__post_init__`.** A `BaseModel` runs `model_post_init` instead. Every subclass's `__post_init__` must be converted in the same change that makes `Artifact` a `BaseModel` (stage 2), or its checks silently stop running. Stage 2 renames them mechanically to `model_post_init`; stage 3 turns them into the validators above.

### 4.3 Dependencies and parameters

A field whose annotation holds artifact types is a dependency: an edge `resolve` walks, embedded as a nested manifest under `"dependencies"`, and declared at its own path. Every other field is a parameter. `dependencies()` replaces `schema.dependencies` and reads `cls.model_fields[name].annotation`, which pydantic has already resolved, self-references and `| None` included (**verified**).

Rules, each with the failure it prevents:

1. **A dependency is annotated `A`, `A | B | None`, `tuple[A, ...]` or `frozenset[A]`, and an artifact type anywhere else is refused when the class is created (§4.4).**
   Why: only these shapes are followed by `resolve`, `to_manifest` and `lab.declare`. An artifact inside a param config or a `dict` would be dumped as plain fields with no `"artifact"` key: it would get no manifest or folder of its own, its job would never be offered, and the parent's manifest could not be decoded.

2. **An unordered collection of dependencies is a `frozenset`.**
   Why: identity. Pydantic compares field values in order, so `sources=(a, b)` and `sources=(b, a)` held as tuples are unequal. The same tokenizer declared twice with its sources listed in a different order would then be reported by `lab.declare` as a conflict at its own path. As a `frozenset` the two are `==` and hash equal: one definition, one manifest, written in one order (sorted by path).

3. **An ordered collection is a `tuple`, never a `list`.**
   Why: identity again, the other way round. `DataSet` concatenates its sources in order, so a different order is a different `train.bin`; a tuple makes order part of `==`, and `DataSet.uid` digests the sources in order. A `list` is unhashable, so `Artifact.__hash__` (§2.1) would fail as soon as the artifact landed in a `frozenset`. A list is also mutable, so it could change underneath a frozen artifact after its path was computed.

4. **Classify a field by its annotation, never by its value.**
   Why: `sources=frozenset()` and `starting_checkpoint=None` are still dependencies, written as `[]` and `null` under `"dependencies"`. Classifying by value would move an empty field between `"dependencies"` and `"parameters"`, so the same class would write manifests of two shapes, and decoding would look for the field in the wrong half.

5. **`Artifact` converts any iterable to a `frozenset` before validation (one `mode="before"` validator for every `frozenset` field).**
   Why: strict mode accepts only a `frozenset` there and refuses a tuple or a list (**verified**), while today's notebooks pass `sources=(valid,)`. Without the conversion, every existing declaration notebook stops working. One validator on the base does it for every family; `Field(strict=False)` on each field would have to be repeated in each one, and would also accept dicts as members (rule 7).

6. **A set may not hold two members at one path (an `after` validator, as `__post_init__` does today).**
   Why: `SourceURL(name="odyssey", url=a)` and `SourceURL(name="odyssey", url=b)` are unequal, so a set keeps both, yet both own `sources/odyssey`. `resolve` would reject the graph later, at `lab.declare`; the validator refuses it at the line that built the set, naming the field. Members differing only in `commit` collapse to one (§2.1), and the first one passed survives.

7. **`from_manifest` builds each dependency as an instance first, from its own `"artifact"` key, and passes instances to `model_validate`.**
   Why: given a dict, pydantic validates it as the annotated class. The abstract `Tokenizer` cannot be instantiated, and `DataSet | MappedDataSet` have the same fields, so pydantic would pick one by guessing. An instance of a subclass passes through as itself (**verified**: `tokenizer: Tokenizer` keeps its `bpe.Tokenizer`).

8. **Pydantic never serializes a dependency; `parameters()` dumps with `include=<parameter names>`, and `to_manifest` encodes each dependency by calling itself.**
   Why: pydantic serializes by annotation, so a `bpe.Tokenizer` in a `Tokenizer` field would be written with the base's fields only, silently (§6.2). Its `sources` and its class would be missing from the manifest, which then cannot be decoded.

9. **Never change a definition with a call that skips validation: `model_construct`, `model_copy(update=)`, or `object.__setattr__` (which only test_resolve uses, to close a cycle).**
   Why: an instance passed into a field is not revalidated (`revalidate_instances="never"`). That is what makes construction cheap and is safe only because every instance was validated when built. An unvalidated value would be trusted by every artifact built on top of it, and `model_copy` also keeps cached properties, so a copied leg would keep its old `lineage_hash` and live at the wrong path (§8).

### 4.4 What a parameter may hold

`schema._check_parameter` refuses, when a class is first used, any parameter annotated `Any`, `object`, a bare `tuple`, a `list`, a `dict`, a `frozenset` of non-artifacts, a mutable dataclass, a union of two param config types, or an artifact hidden inside a parameter. Pydantic accepts most of these annotations without complaint.

**Recommendation: keep one small guard, run when an `Artifact` subclass is created** (`__pydantic_init_subclass__`, which sees the finished `model_fields`; **verified**). It refuses, in a parameter annotation and in the fields of any param config it names:

- `list`, `dict`, `set`, `frozenset`, `Any`, `object`;
- an artifact type anywhere outside the four dependency shapes.

It drops the "at most one param config type per union" rule: the class key rule (§6.3) writes the class for every value in a union of two param configs, so nothing on disk is ambiguous.

The guard is load-bearing in the new design, for two reasons. Decoding turns every JSON array in `"parameters"` into a tuple (§6.4), which is only correct if tuples are the only parameter container. And `Artifact.__hash__` hashes field values, so a `list` or `dict` field would make the artifact unhashable, failing only when it lands in a `frozenset`, far from the declaration. The guard lives next to `dependencies()`, which walks the same annotations.

### 4.5 The model: a `Model` subclass per architecture, not `ModelSpec`

The original proposal introduced `ModelSpec(class_ref: str, config: GPTConfig)`: a reference to an `nn.Module` class, a typed config (eventually a tagged union of per-architecture configs), and a trusted runtime registry that resolves `class_ref`.

Today `Model` (in `artifacts/core/SGD/training.py`) is a frozen dataclass with a `device` field, `build()` and `step_flops()`. Its subclasses `models.transformer.Transformer`, `models.transformerV2.Transformer` and `models.mamba.Mamba` hold their constructor fields and import torch only inside `build()`.

**Recommendation: keep the subclasses and make `Model` a `ParamConfig`.** A model is defined like an artifact (validated fields, frozen, part of identity) but is not one: it has no folder, no manifest and no job. The reasons for not adopting `ModelSpec`:

1. Each architecture owns three things: its fields; how they map onto its module's constructor (`sequence_length` becomes `context_length`, the `dtype` string becomes a torch dtype, `Mamba` ignores `sequence_length`); and its analytic FLOP count (`step_flops`, which `gpus/monitor.py` uses for MFU and `profiling/util.py` reports). `ModelSpec` separates the fields from the class reference, but the mapping and the FLOP count still need a per-architecture home, and that home is the subclass that exists today.
2. A tagged union of configs has to be enumerated where the field is declared, so `artifacts/core` would import every package under `models/`, reversing the dependency direction. codec.md §11 rejected this for the same reason.
3. The registry would restate what the class's import path already says.
4. The runtime factory the proposal asks for already exists: `Model.build()`, called only on the worker.

What changes: `@dataclass(frozen=True)` becomes `class Transformer(Model)`; the `__post_init__` checks become `Field(gt=0)` and a model validator (§4.2); the `polymorphic` ClassVar goes (§6.3); and `asdict(model)` becomes `model.model_dump()` in `profiling/util.py` and `Pretraining.lineage_hash`. `device` stays on `Model` here; the identity spec takes it out. The two `Transformer` classes stay separate: they are distinct architectures with distinct class paths, and so distinct identities.

**`Model` is an ordinary `ParamConfig`: no flag, no validator, no special code.** `Pretraining` declares `model: Model` and pydantic validates whatever it is given against that annotation, like any other param config field. A subclass instance passes as itself (**verified**), and its own field constraints ran when it was built. Two things are particular to this field, and neither belongs on `Model`:

- **The manifest has to name the subclass.** JSON carries field values, not types. Decoding `{"vocab_size": 1000, "d_model": 64, ...}` needs to know whether to build a `Transformer` or a `Mamba`, because only the class knows its `build()` and `step_flops()`. Two architectures can also have the same fields; the two `Transformer` classes do today. So a field holding a subclass of its annotation writes `"class"`. That is a codec rule for any param config field (§6.3), not something `Model` declares. Today's `polymorphic = True` flag exists only because the dataclass codec refused subclasses by default; the generic rule makes it unnecessary.
- **`Pretraining` relies on fields the base does not declare.** It checks `model.vocab_size` against the tokenizer; `PretrainJob` reads `model.sequence_length`; `profiling/util.py` reads both and `dtype`. Today only the subclasses declare them, so `Pretraining` works by every subclass happening to have them. **Recommendation: declare `vocab_size`, `sequence_length` and `dtype` on `Model`** (CLAUDE.md: if two modules rely on the same field, declare it). Then `Pretraining`'s validation is a check against the `Model` contract, and a new architecture missing one of them is refused when its class is created, not when a leg is first trained.

The alternative that avoids a class key, annotating `model: Transformer | Mamba` and letting pydantic tell the arms apart by their fields, was rejected. `artifacts/stages/pretraining` would have to import and list every architecture, adding one would mean editing the stage, and two architectures with the same fields could not be told apart.

### 4.6 Runtime state attached by `bind`

The `_load` hooks (the BPE and template tokenizers, `DataSet`, `MappedDataSet`) attach state that is not a field (`_vocab`, `_merges`, `_train_tokens`) with `object.__setattr__`, and `bound` asks `hasattr(self, "_vocab")`.

**Recommendation: keep this as it is.** **(verified)** On a frozen `BaseModel`, `object.__setattr__` stores an underscore name on the instance, the value stays out of `model_dump`, `==` is unaffected, and `hasattr` is `False` before the hook runs. Declaring these as `PrivateAttr` gains nothing: pydantic's default `__eq__` would compare them, and the override in §2.1 ignores them anyway. The comments in the `_load` hooks that justify `object.__setattr__` "because the dataclass is frozen" must say "model" instead.

Artifacts also cross process boundaries by pickle: `local.declare_on_volume` sends the requested artifact to a Modal function with `.remote(...)`. **(verified)** A frozen pydantic artifact pickles and unpickles equal, keeping its subclass param configs as their own classes, its cached properties, and any bound state.

`functools.cached_property` works on frozen pydantic models (**verified**), so `Training.start_step` and `Pretraining.lineage_hash` stay as they are. §8 covers the one way a cached value goes stale.

### 4.7 Identity and paths do not change

**No stage changes a path.** Every `uid` and `artifact_path` comes out identical, so the golden fixtures and the volume scan can prove each stage did nothing to identity. The digests are hand-written per family; the dataclass-specific inputs are `asdict(self.model)` and `asdict(self.training_parameters)` in `Pretraining.lineage_hash`. **(verified)** `_digest(asdict(config))` equals `_digest(config.model_dump())` for the same values: `model_dump()` keeps tuples as tuples, `json.dumps` writes both as arrays, and `_digest` sorts keys, so field order does not matter.

**Gotcha: a `"class"` key inside `model_dump()` would move every run.** If `Model` serialized its own class name (a `model_serializer` that adds `"class"`, the natural pydantic way to do polymorphism), `self.model.model_dump()` in `lineage_hash` would include it, and every pretraining path would change. The codec writes the class key, never the param config (§6.3), so `model.model_dump()` stays exactly today's `asdict(model)`.

Two identity questions found during this review are deliberately left to the identity spec, because each moves paths: `device` is part of a leg's identity (the same leg declared for CPU gets a different folder), and `lineage_hash` hand-picks its inputs instead of digesting what the manifest records.

### 4.8 Construction becomes keyword-only

`BaseModel.__init__` takes keyword arguments only (**verified**: a positional argument raises `TypeError`). Positional call sites to change:

- Production code: `TokenizedSource(tokenizer, s)` in `DataSet.from_sources` and `MappedDataSet.from_sources`.
- Tests: `LRSchedule(1e-3, 1e-4, 1, 100)` and `TokenizedSource(...)` in test_pretrain.py, test_declare.py and test_dataset.py; `Link("a")` in test_resolve.py; `Leaf(...)`, `Shapes(...)`, `Square(4, 2)`, `Polygon(...)` and `Shape(0)` in test_codec.py.
- The two tracked notebooks (`declarations/tinystories.ipynb`, `profiling/profile_model.ipynb`) already use keywords. Notebooks stored only on the volume were not checked.

Two LESSONS.md entries change. "Dataclass inheritance with defaults needs `kw_only`" becomes obsolete and is deleted. "Artifact annotations must be real imports" still holds, with new wording: pydantic resolves annotations when the class is created, and an import under `TYPE_CHECKING` leaves the class incompletely defined, so it raises at first instantiation.

## 5. Resolver

Unchanged. `deps()` reads `dependencies()`, and the order is still dependency fields by name, with set members sorted by path, so traversal never depends on set iteration. A path reached twice must hold an `==` definition, which compares the whole subtree.

Cycle detection stays, even though a frozen model cannot form a cycle without `object.__setattr__` (which is how test_resolve builds one); it costs nothing. Errors stay `ValueError` with today's messages, rather than the proposal's `CycleError` and `IdentityConflictError`: nothing catches them by type.

## 6. Codec and canonical manifests

### 6.1 The format

Manifests keep today's shape and bytes: five top-level keys in fixed order (`artifact`, `commit`, `allocated_resources`, `parameters`, `dependencies`), every nested map sorted by key, `indent=2`, `ensure_ascii=False`, `allow_nan=False`, and a trailing newline. Every embedded dependency carries its own `commit` and `allocated_resources`, and the manifest at an artifact's own path is authoritative for it.

The proposal's step 6 ("compatibility handling for existing manifests") is therefore verification only: the golden fixtures and the volume scan show that no stage changes a byte. The format change the identity spec makes comes with its own migration.

**No schema version and no formal canonical JSON (JCS) for now.** One Python program reads and writes manifests, and identity hashes are computed from chosen fields, not from manifest bytes. Byte canonicalization matters for two properties (§2): equal definitions give equal definition encodings, and a manifest re-encodes to its own bytes. Sorted keys and the float rule (§6.5) already give both. A version field is itself a format change; add one the first time a format change needs a migration script (docs/spec.md: "Schema changes are migrated explicitly, one script per change").

### 6.2 Encoding

```python
def to_manifest(artifact):
    return {
        "artifact": dotted(type(artifact)),
        "commit": artifact.commit,
        "allocated_resources": canonical(artifact.allocated_resources.model_dump(mode="json")),
        "parameters": parameters(artifact),
        "dependencies": {name: <each member's to_manifest, in members() order> for each dependency field},
    }
```

`parameters(artifact)`, which is also what the dashboard shows, is `artifact.model_dump(mode="json", include=<parameter names>, serialize_as_any=True)`, plus class keys (§6.3), passed through `canonical` (§6.5).

Neither encoding nor decoding runs `resolve`: `to_manifest` recurses through dependencies directly. The proposal's `resolved=` parameter, meant to avoid resolving twice, has nothing to save.

**Gotcha: pydantic serializes a field by its annotation, not by its value.** A `Transformer` in a field annotated `Model` dumps as `Model`'s fields only, silently dropping every field the subclass adds (**verified**: a subclass instance in a base-annotated field dumped as `{}`). This is the same trap LESSONS.md records for the dataclass codec. `serialize_as_any=True` dumps each value as its own class (**verified**). **Recommendation: pass it at the one dump call in `parameters()`,** rather than annotating each polymorphic field `SerializeAsAny[...]`.

### 6.3 The class key for subclass param configs

**Rule, at every depth:** a param config value is written with `"class": dotted(type(value))` beside its fields unless its class is exactly the one param config class its annotation names. That covers a top-level parameter (`Pretraining.model`), a param config nested inside another (a field of `TrainingParameters`), and a param config inside a tuple (`tuple[A, ...]`). A param config whose class is exactly its annotation carries no class key.

- **Encoding.** `parameters()` dumps with `serialize_as_any=True` (§6.2), then walks the dump alongside the instance: for every param config it meets, it compares `type(value)` with the class its field annotation names (the field's `model_fields` annotation, or a tuple's item type), and adds the key when they differ.
- **Decoding.** `from_manifest` walks the parameters JSON bottom-up, in the same pass that turns arrays into tuples (§6.4): any object carrying `"class"` is located, its remaining fields are validated as that class, and the instance replaces the object. Pydantic then checks each instance against the field it lands in, and strict validation refuses an instance that is not that field's class or a subclass of it. So "must subclass the annotated class" is pydantic's check, not a second one in the codec (**verified**: a `Square` nested in a tuple inside a param config comes back a `Square`, and an unrelated class in its place is refused with `model_type`).
- **Unions.** A union naming two param config classes (`A | B`), which nothing uses today, carries the key on every value, since no single annotated class could be meant. Nothing on disk is ambiguous, which is why the guard no longer needs the "at most one param config per union" rule (§4.4).

This replaces the `polymorphic = True` ClassVar and `schema.polymorphic`. Today only `Model` fields carry a class key, and under this rule still only they do, since nothing else holds a subclass, so manifests stay byte-identical. Param configs like `LoopConfig` still store plain fields and can be renamed freely.

**Why the codec and not `Model`:** pydantic's idiomatic polymorphism would put a wrap validator and a serializer on `Model`. That fails here twice. A serializer that adds `"class"` changes `model_dump()`, and so every lineage hash (§4.7). And a wrap or before validator breaks strict JSON decoding (the next gotcha). Keeping both directions in the codec also keeps the manifest format in one module, behind its seam.

Two behaviors change. Today a non-polymorphic param config field refuses a subclass instance ("expected exactly"); under this rule it accepts the instance and records its class, so test_codec's `Narrower` case flips from a refusal to a round trip. And today a polymorphic field writes a class key even for an instance of the base class itself; under this rule it does not, so the `Shape(0)` case in `test_a_polymorphic_field_survives_as_its_own_class` loses its class key. No stored manifest holds a bare `Model`, since `Model.build()` raises.

**No exception to the round-trip guarantee.** Today's `polymorphic` flag works at the top level only, and a subclass nested deeper would be dumped with all its fields, no class key, and refused when decoded. Applying the rule at every depth removes that exception, at the cost of a few lines in a walk the decoder needs anyway: every param config comes back as its own class wherever it sits. Nothing below the top level holds a subclass today, so manifests stay byte-identical; test_codec gains a fixture with a subclass two levels down, inside a tuple, to keep the guarantee tested.

**Gotcha: under `strict=True`, any `wrap` or `before` validator breaks JSON-mode decoding of tuples.** Once such a validator has run, pydantic continues with Python objects, and strict validation then refuses a JSON array for a `tuple` field, even inside `model_validate_json` (**verified**: `betas: tuple[float, ...]` under a `Model` with a wrap validator fails with "Input should be a valid array"). `Training` needs a before-validator (§4.2), and `Artifact` needs one for `frozenset` fields (§4.3), so JSON-mode decoding would break on every tuple parameter beneath them (`betas`, `special_tokens`). This is why decoding runs in Python mode (§6.4), and one more reason the class dispatch is not a validator.

### 6.4 Decoding

`from_manifest(data, memo)`:

1. Locate the class named by `data["artifact"]`, and require an `Artifact` subclass.
2. Decode each dependency recursively, sharing `memo` as today, and build the `tuple` or `frozenset` its field declares.
3. Walk `"parameters"` once, bottom-up: every JSON array becomes a tuple, and every object carrying `"class"` becomes an instance of that class (§6.3). About ten lines.
4. Call `cls.model_validate({**parameters, **dependencies, "commit": data["commit"], "allocated_resources": data["allocated_resources"]})` in Python mode.

**Why Python mode, with arrays turned into tuples:** strict Python mode refuses a list for a `tuple` field (**verified**), and JSON mode is ruled out by the gotcha in §6.3. Turning arrays into tuples is correct because the guard in §4.4 makes tuples the only parameter container. Param configs inside parameters arrive as dicts and validate into their classes in strict Python mode (**verified**).

All validators run on decode, so a manifest that breaks an invariant (a vocabulary mismatch, for example) cannot be loaded, as today. Decoding validates each node once, which removes codec.md §13 item 4 ("decoding normalizes twice").

**Errors:** `_named`, which rewrites pydantic errors as `TypeError` or `ValueError` naming the field path, is deleted. Pydantic's `ValidationError` already names the model and the location (`1 validation error for Transformer / d_model`), and it subclasses `ValueError` (**verified**), so `Artifact.load`'s existing `except (..., ValueError)` still turns it into "not a readable manifest" naming the folder. Tests that expect a `TypeError` from a bad field expect `ValidationError` instead.

**Trusted classes: no registry and no allow-list (decided); round-trip tests instead.** The proposal required decoding to resolve classes through an approved registry, so that a manifest cannot import arbitrary code. Neither a registry nor a prefix allow-list is added:

- The volume is written only by `lab.declare`, from your own notebooks, so the threat a registry addresses is not present.
- A class path that is wrong for any other reason (a typo, a moved module, a corrupted file) already fails safely: `locate` raises `ImportError` or `AttributeError`, and `Artifact.load` reports the manifest as unreadable, one row, naming the folder. A located class that is not an `Artifact` (or, for a class key, not a subclass of the annotated param config) is refused.
- An allow-list of `artifacts.` and `models.` would refuse every artifact the tests define: pytest imports test modules at the top level, so their classes are `test_codec.Plain` and `test_resolve.Link`.

What the allow-list would have protected, that every manifest names a class that rebuilds it, is checked instead by round-trip tests over every class in the repository (§9).

Dotted strings inside parameters (`optimizer`, `lr_schedule_fn`, `dtype`) stay strings through validation; only the job resolves them, on the worker that runs it.

### 6.5 Canonical bytes and `-0.0`

`manifest_json` is unchanged.

**Gotcha: `-0.0` survives strict validation** (**verified**). `-0.0 == 0.0`, but JSON writes `-0.0`, so two equal artifacts would get different digests and different paths. Today `normalize` turns `-0.0` into `0.0` at construction. **Decided: canonicalize at encoding instead, in one function** used by both `_digest` and `parameters()`:

```python
def canonical(value: object) -> object:
    """Returns `value` as JSON data with keys sorted and -0.0 written as 0.0."""
    return json.loads(
        json.dumps(value),
        parse_float=lambda s: float(s) + 0.0,
        object_pairs_hook=lambda pairs: dict(sorted(pairs)),
    )
```

The consequence: an artifact built with `-0.0` holds `-0.0`, and its decoded copy holds `0.0`. The two are `==`, hash equal (`hash(-0.0) == hash(0.0)`), and have the same path and the same bytes; only their `repr` differs. The hypothesis test that asserts `repr(artifact) == repr(back)` compares manifests instead.

## 7. Storage

**Recommendation (revised): a small `artifacts/core/storage.py` owns manifest files, and `Artifact.load` stays as the convenience that combines it with the codec.**

An earlier revision rejected `storage.py` because its functions would each have one caller. That was the wrong test, and it is no longer true either. The question is whether separating file I/O from reconstruction makes each part easier to understand and to test. Here it does:

- **The codec stays pure.** `to_manifest` and `from_manifest` take and return dicts, and nothing in `manifest.py` touches a file, so codec tests need no folder. Today `Artifact.load` mixes the two in one method: path safety, reading, JSON parsing, error wording, decoding and the path check.
- **One place owns the file layout and its errors:** `root / artifact_path / "manifest.json"`, the refusal of an absolute path or one containing `..`, `_root`'s check that a root is an absolute `Path`, and the messages "nothing declared at ..." and "... is not readable JSON". Today each place that reads or writes a manifest spells these out itself.
- **Several callers.** Reading: `Artifact.load`, `migrations/scan.py`, and the identity spec's migration. Writing: `lab.declare`, the identity migration, and the tests that write manifests by hand today (test_artifact, test_codec, test_dataset, test_mappeddataset, test_pretrain). `launcher/state.py` keeps globbing for `MANIFEST` and loading through `Artifact.load`. The `/artifact` route in `launcher/leasebook.py` keeps its own tolerant `open_json` read under `mount_lock`, because it shows whatever is on disk, a corrupt file included, rather than refusing it; that is a different contract from `read_manifest`'s.

The module is two functions and the constant:

```python
MANIFEST = "manifest.json"

def read_manifest(root: Path, artifact_path: Path | str) -> dict:
    """Returns the parsed manifest at root / artifact_path.

    Raises FileNotFoundError when nothing is declared there, and ValueError
    for a path that leaves root or a file that is not JSON, naming the file.

    read_manifest(Path("/storage"), "sources/odyssey")
      -> {"artifact": "artifacts.sources.SourceURL", "commit": "abc1234", ...}
    """

def write_manifest(root: Path, artifact_path: Path | str, manifest: dict) -> None:
    """Writes `manifest` to root / artifact_path / manifest.json as
    manifest_json's bytes, creating the folder.

    Overwrites whatever is there: whether a write is allowed is lab.declare's
    decision, made before it calls this.

    write_manifest(Path("/tmp/root"), "sources/odyssey", odyssey.to_manifest())  -> None
    """
```

`Artifact.load` becomes `from_manifest(read_manifest(root, path), memo)` plus the check that the decoded artifact's path is the folder it was read from. That check is about identity, not files, so it stays on the artifact. Its contract is unchanged: `FileNotFoundError` for a missing manifest, a `ValueError` naming the folder for anything that will not decode, and a `ValueError` when the decoded path is not the folder. `status` keeps its own file-presence checks, since it looks at an artifact's outputs as well as its manifest. `MANIFEST` and `_root` move from artifact.py to storage.py, and their importers (artifact.py, lab.py, launcher/state.py, launcher/leasebook.py, tests) import them from there. `storage.py` imports `manifest_json` from the codec, which owns the byte format; the codec imports nothing from storage.

Manifests are still written directly, without a temporary file and a rename: they are small, written in one call, and written before any job runs (docs/spec.md §6).

## 8. Declaration

`lab.declare` keeps its behavior, its report, and its commit and refusal semantics. The only changes are mechanical: `asdict(resources)` becomes `resources.model_dump()`, and the resources update uses `model_copy`:

```python
manifest = on_disk.model_copy(update={"allocated_resources": node.allocated_resources}).to_manifest()
```

**Gotcha: `model_copy(update=...)` neither validates nor clears cached properties.** **(verified)** It accepts `True` into an `int` field, and a copy whose `steps` changed still returned the original's cached value. For `Pretraining`, a copy with new `training_parameters` would keep the old `lineage_hash`, and so the old `uid` and path. `dataclasses.replace`, which the code uses today, runs `__init__` and has neither problem. For the resources update both are harmless: the new `Resources` is already validated, and no cached property depends on `allocated_resources`.

**Rule: use `model_copy` only for recorded fields. Any change to a definition field constructs a new instance** (`type(a)(**{n: getattr(a, n) for n in type(a).model_fields} | changes)`), so validators run and cached identity is recomputed. The tests that use `replace()` to provoke a refusal follow the same rule.

Commit drift keeps today's rule: a row's drift compares the stored `commit` at the node's own path with the request object's `commit`. The identity spec changes it.

Comparison stays at the object level (`stored != artifact`); `_leaves` over the two manifests only renders the differences for the report. The proposal's separate `inspect_declaration` is `declare(commit=False)`, and its `DeclarationConflictError` is today's `DeclarationError`. Concurrent writers are out of scope: docs/spec.md assumes one writer per root.

## 9. Side-effect boundaries and tests

Pure: validation, resolution, `==`, encoding, decoding, `canonical`. Reads files: `storage.read_manifest`, and through it `load`; `status`; `declare(commit=False)`. Writes files: `storage.write_manifest`, called by `declare(commit=True)`. Building and running models: jobs only.

Test changes, beyond the positional constructors of §4.8:

- **New in stage 0:** golden fixtures pinning `artifact_path` and `manifest_json` bytes (§10).
- **New, round trip for every class (decided, replacing the allow-list):** a test that imports every package under `artifacts/` and `models/` (their `__init__.py` modules only, which are torch-free; `jobs.py` modules are not imported), collects every concrete `Artifact` subclass and every `ParamConfig` subclass (concrete: `inspect.isabstract` is false and, for an artifact, `producer` is set, which excludes bases like `Source`, `Tokenizer` and `Training`), and requires an example instance for each in one fixture table. For every example it asserts that `from_manifest(json.loads(manifest_json(a.to_manifest())))` is `==` to `a`, has the same `artifact_path`, re-encodes to the same bytes, and holds every param config as its own class (a `Mamba` comes back a `Mamba`). A class added without a fixture fails the test, so a new artifact family or architecture cannot ship without proof that its manifests rebuild it. The hypothesis tests over parameter shapes in test_codec.py stay alongside it.
- **New:** the `frozenset` conversion; a commit-only difference is `==`, hash-equal and one set member, with equal definition encodings and different complete manifests (§2); a param config subclass nested two levels down, inside a tuple, round-trips as its own class (§6.3); the §4.4 guard refuses `list`, `dict` and a hidden artifact when the class is created; `read_manifest` and `write_manifest` on a temporary folder, including a missing manifest, unreadable JSON and a path that leaves the root (§7).
- **Removed:** the annotation refusals the guard no longer makes (`Mutable`, `Leaf | Two`, a bare `tuple`); `test_normalize_is_idempotent_and_exact`; the `2**53 + 1` refusal.
- **Changed:** `Narrower` round-trips with a class key, and `Shape(0)` has none (§6.3); the `repr` property compares manifests (§6.5); field errors are `ValidationError` (§6.4); the test_codec.py fixtures become `ParamConfig` and `Artifact` subclasses; definition changes construct new instances instead of `replace()` (§8); hypothesis strategies mostly stay as they are, since `st.builds` generates values that respect `Field(gt=0)` by itself (**verified**).

## 10. Implementation plan

### 10.0 Rules for every stage

**Gates.** A stage lands only when all three hold:

- the whole test suite passes (`pytest test/`);
- `mkdocs build --strict` passes;
- the golden fixtures and the volume scan are unchanged.

**One description of the code, in `docs/`.** Everything that describes what the code does lives in the pages under `docs/`, served by mkdocs, and nowhere else: each thing is described once, on one page, so no two texts can disagree. Each stage updates the pages for what it changes, in the same commit as the code, and folds in every other text about the same subject, deleting it in that commit. A page reaches the code through mkdocstrings (`::: artifacts.core.artifact`), so the API reference on a page *is* the docstrings, and a docstring change shows up there with no separate edit.

- **When sources disagree,** the code decides. The page describes what the code does, and each disagreement found while folding is listed in that stage's report to you, so a wrong description is fixed, not copied.
- **Not covered by the rule,** because they do not describe the code: `CLAUDE.md` and `.claude/skills/` (instructions for writing it), `LESSONS.md` (what was learned and how) and `TODO.md`. A LESSONS.md entry may point at a page; it does not restate it.
- **README.md** shrinks to what a repository front page needs: one paragraph on what Launchpad is, how to serve the docs (`mkdocs serve`), and a link to them.
- **Design rationale** goes on the page of the subject, under a "Why" heading, next to the behavior it explains. `docs/spec.md` describes the same subjects as the topic pages, so it is dissolved into them, page by page, and removed from the nav when it is empty. CLAUDE.md's "design rationale goes in `spec.md`" is reworded to match in stage 0.

Every descriptive file outside `docs/` today, and where it goes:

| Source | Folded into | Stage |
|---|---|---|
| `notes-scratch/artifact_workflow.md` | Nothing: an earlier version of this spec, superseded by it | 0 (deleted) |
| `docs/spec.md` §2, §5; `notes-scratch/functionalspec.md` §1, §3; `notes-scratch/workflows.md` §2, §3, §12 | `docs/artifacts.md` | 2 |
| `artifacts/core/codec.md` (without its source appendix); `docs/spec.md` §3; `functionalspec.md` §4; `workflows.md` §5 | `docs/manifests.md` | 2 |
| `artifacts/stages/pretraining/training.md`; `workflows.md` §13 | `docs/training.md`, `docs/models.md` | 3 |
| `docs/spec.md` §6; `functionalspec.md` §5, §10; `workflows.md` §4; README "The lab" | `docs/lab.md` | 4 |
| This spec | The pages above, as each stage lands | 4 (deleted) |
| README "Where things live"; `docs/spec.md` §4; `functionalspec.md` §12 | `docs/repository.md` | 4 |
| README "Jobs", "One job, traced"; `functionalspec.md` §2, §9; `workflows.md` §8, §9 | `docs/jobs.md`, `docs/workers.md` | 4 |
| README "Read/write traffic", "Scheduling, deferred"; `docs/spec.md` §7; `functionalspec.md` §6 to §8; `workflows.md` §1, §6, §7, §10 | `docs/launcher.md` | 4 |
| `notes-scratch/UI.md`; README "Dashboard"; `functionalspec.md` §11 | a new `docs/dashboard.md` | 4 |
| `notes-scratch/LOGGING.md`; `workflows.md` §11 | a new `docs/logging.md` | 4 |
| `notes-scratch/profiling.md` | `docs/profiling.md` | 4 |
| `notes-scratch/guide.md` | a new `docs/getting-started.md` (setup, first workflow, end-to-end example, common questions) | 4 |
| `docs/spec.md` §1, §8, §9 | `docs/framework.md` | 4 |

**Docstrings on everything a user calls.** "User-facing" means every public name reachable from a notebook or a job: artifact and param config classes, their public methods and properties, `lab.declare`, `lab.refresh`, `lab.save`, `resolve`, and the codec's public functions. Names starting with `_` are excluded, and mkdocs already filters them out (`filters: ["!^_"]`). Each user-facing docstring has, in this order:

1. a first line saying what it returns, in the present tense;
2. at most one paragraph, only for the thing a caller can get wrong;
3. one input/output example with real values, in the form the code-style skill uses.

```python
def resolve(artifact: Artifact) -> list[Artifact]:
    """Returns every artifact `artifact` is built from and then itself,
    dependencies before dependents, each path once.

    resolve(TokenizedSource(tokenizer=bpe, source=odyssey))
      -> [SourceURL(name='odyssey'), Tokenizer(vocab_size=1000), TokenizedSource(...)]
    """
```

A class docstring's example is a construction, followed by the derived values a reader most needs (`uid`, `artifact_path`). Those properties are overridden in every family, so their examples live once, on the class, not on each override; the base class's abstract property keeps its contract docstring. CLAUDE.md's docstring rule ("then at most one paragraph... Stop there") is amended in stage 0 to allow the example line. For user-facing functions this rule takes precedence over the code-style skill's "skip the docstring if the name makes it obvious"; internal functions keep following the skill.

**Who runs what on the volume.** The volume scan runs from a local checkout with `modal run local.py::scan_volume`. Its logic is `migrations/scan.py`'s `scan(root)`, which imports only `artifacts` and so is tested on a temporary folder; `local.py` runs it in an ephemeral Modal app that builds its image from the checkout and mounts the volume, like `declare_on_volume`, so it runs the checkout's code without deploying it. The runner stays in `local.py` because Modal ships the function by value but the names it calls by reference: a runner in `migrations/` that imported `main` would make the container import `main`, which the worker image does not carry (**verified** offline: the serialized function loads with `local` and `main` unimportable). I write it; you run it and review its output.

### 10.1 Stage 0: safety net and doc build

| | |
|---|---|
| Code | None in the library. |
| Tests | `test/test_golden.py`: pins `artifact_path` and `manifest_json` bytes for a `SourceURL`, a two-source BPE tokenizer, a template tokenizer, a `TokenizedSource`, a `DataSet`, a `MappedDataSet`, a two-leg `Pretraining` chain on a `Transformer`, a leg on the `transformerV2` `Transformer`, and a `Mamba` leg. Every fixture sets `commit` explicitly, so the bytes do not depend on HEAD. Expected values are generated with today's code and committed as literals, under `test/fixtures/golden/`. |
| Tools | `migrations/scan.py`'s `scan(root)`: loads every manifest under a root, re-encodes it, and reports any manifest that does not load (including one whose decoded path is not its folder) or whose bytes change; `test/test_scan.py` covers it on a temporary folder. `modal run local.py::scan_volume` runs it on the volume. Rerun after stages 1 and 2; stage 2 switches its reads to `storage.read_manifest`; the identity spec reuses it to verify its migration. `migrations/cpu_floats.py`, covered by `test/test_cpu_floats.py`, is the volume's first migration (next row). |
| Docs | Make `mkdocs build --strict` pass: `docs/spec.md` has 5 broken links today (to `README.md`, `LESSONS.md` and a missing `docs/workflows.md`). Fix README's links to the missing `docs/guide.md` and `artifacts/core/spec.md`. In CLAUDE.md: point its links at `docs/`, amend the docstring rule for the example line, and reword "design rationale goes in `spec.md`" to the page of the subject. Delete `notes-scratch/artifact_workflow.md`. |
| Volume | The first scan found 41 manifests, all loading at their own paths, and 32 that re-encode differently: each stores `allocated_resources.cpu` as an int (`4`), written by code older than the int-to-float rule, either at its root (the three tokenizers) or in an embedded tokenizer copy. `migrations/cpu_floats.py` rewrites exactly those in today's bytes, keeping every value, and refuses if any manifest differs in anything else; `modal run local.py::canonicalize_manifests` runs it as a dry run, `--apply` writes. |
| Done when | Suite and strict build pass; the cpu-floats migration is applied, and the scan on the volume reports 0 problems, the baseline for stages 1 to 3. |

### 10.2 Stage 1: param configs and models become `ParamConfig`s

| | |
|---|---|
| Code | `artifacts/core/artifact.py`: `STRICT`, `ParamConfig`, `Resources`. `artifacts/core/SGD/training.py`: `Model` (declaring `vocab_size`, `sequence_length`, `dtype`, §4.5), `OptimizerParameters`, `LRSchedule`, `TrainingParameters`, `LoopConfig`, with constraints as `Field(gt=0)` and validators (§4.2). `models/transformer`, `models/transformerV2`, `models/mamba`: the architectures, with `Field(gt=0)` and the `d_model % num_heads` validator. `asdict` becomes `model_dump` in `SGD/job.py`, `profiling/util.py`, `Pretraining.lineage_hash` and `lab.py`. `schema.py`: about five lines so the old codec accepts frozen `BaseModel` param configs (deleted in stage 2). |
| Tests | Positional param config constructors become keyword (§4.8). New: a refusal per new constraint. The hypothesis strategies need few changes: `st.builds` respects `Field(gt=0)` by itself (**verified**). |
| Docs | `docs/models.md`: the `Model` contract (the fields a stage reads, `build`, `step_flops`) and each architecture through `:::`. `docs/training.md`: the param configs a leg is declared with, through `::: artifacts.core.SGD.training`. Docstrings with examples on every class touched. |
| Done when | Gates pass; goldens unchanged; you rerun the scan, which now also reports any stored value a new constraint refuses. |

### 10.3 Stage 2: `Artifact` becomes a `BaseModel`, and the codec shrinks

This is the large stage, and it cannot be split: `__post_init__` is what calls `normalize`, so the base class and the codec change together.

| | |
|---|---|
| Code | `artifact.py`: `Artifact(BaseModel)` with `model_config = STRICT`; `RECORDED` and `__eq__`/`__hash__` (§2.1); `dependencies()` and the guard (§4.4); the `frozenset` before-validator, the duplicate-path check and the `producer` check (§4.3). `manifest.py`: `to_manifest`, `parameters`, `from_manifest`, `manifest_json`, `canonical`, the class key rule (§6); delete `normalize`, `_field`, `_STRICT`, `_named`, `_dump`, `_load`, `_encode`, `_decode`. New `storage.py` (§7): `MANIFEST`, `_root`, `read_manifest`, `write_manifest`; `Artifact.load` and `lab.declare` call it, and `MANIFEST`'s importers move to it. Delete `schema.py`. Every family's `__init__.py` (sources, tokenizers, bpe, template, tokenized, dataset, mappeddataset, stages/pretraining): drop `@dataclass`, rename `__post_init__` to `model_post_init(self, context)` with its body unchanged, and replace each `super().__post_init__()` with `super().model_post_init(context)` (the base has no `__post_init__` any more, so a leftover call raises `AttributeError`); fix the "dataclass is frozen" comments. Pydantic runs `model_post_init` after field validation and before the `after` validators, so `Training` still fills in `model` before `Pretraining` checks it, and the base's producer and duplicate-path validators run last. `DataSet.from_sources` and `MappedDataSet.from_sources`: keyword constructors. `lab.py`: `replace` becomes `model_copy` for the resources update, `_leaves` skips `RECORDED`. |
| Tests | Everything in §9, including the new round-trip test over every class (`test/test_roundtrip.py`). |
| Docs | `docs/artifacts.md`: definition equality (§2.1), construction, and the dependency rules (§4.3), replacing its dataclass text. `docs/manifests.md`: the codec. Both pages fold in the stage 2 rows of §10.0's table (`codec.md` without its source appendix, `docs/spec.md` §2, §3 and §5, and the matching `functionalspec.md` and `workflows.md` sections), which are then deleted. LESSONS.md: delete the `kw_only` entry, rewrite "pydantic is strict only where it is told to be" around the gotchas in §4.3, §6.2, §6.3, §6.5 and §8, and reword "Artifact annotations must be real imports". |
| Done when | Gates pass; goldens unchanged; you rerun the scan and it reports nothing. |

### 10.4 Stage 3: family checks become validators

| | |
|---|---|
| Code | `Training` (the before-validator and the required `model`, removing its `object.__setattr__`), `Pretraining`, `DataSet` and `MappedDataSet` move from `model_post_init` to the validators in §4.2. |
| Tests | A leg built from a starting checkpoint alone gets its model; a leg with neither is refused; a mismatched model is refused. The existing family refusals keep passing. |
| Docs | `docs/training.md`: declaring a leg from a starting checkpoint, folded in from `artifacts/stages/pretraining/training.md`, which is then deleted. Docstrings that describe `__post_init__` are rewritten. |
| Done when | Gates pass; goldens unchanged. |

### 10.5 Stage 4: the rest of the descriptions

No code changes. The descriptions not yet folded by stages 2 and 3 go into their pages (the stage 4 rows of the table in §10.0): the lab, repository layout, jobs and workers, the launcher, the dashboard, logging, profiling, getting started, and the framework overview. README shrinks to its front page, and the new pages join the mkdocs nav.

| | |
|---|---|
| Docs | Every stage 4 row of §10.0's table. Then delete `notes-scratch/` (its remaining files have all been folded in), except the identity spec if it has not landed yet, and `docs/spec.md` (empty by now). CLAUDE.md's "the map of where things live and how traffic flows is README.md" points at `docs/repository.md` and `docs/launcher.md` instead. |
| Done when | `mkdocs build --strict` passes, and no `.md` file outside `docs/` describes the code: only `CLAUDE.md`, `.claude/skills/`, `LESSONS.md`, `TODO.md` and the short README remain, plus the identity spec until it lands. |

### 10.6 Noticed, out of scope

- `DataSet` and `MappedDataSet` duplicate `require_separator`, `from_sources`, `tokenizer`, `train_batches` and `valid_batches`.

## 11. Decisions

The original "decisions to pin down", resolved, along with the choices this revision makes:

| Question | Decision |
|---|---|
| Identity and path algorithm; who owns metadata | Unchanged: hand-written `uid` and `artifact_path` per family, with byte-identical digest inputs (§4.7). `commit` and `allocated_resources` are fields of every artifact, named in `RECORDED`, and excluded from `==` and `hash` (§2.1). Changing identity is the identity spec's. |
| Metadata on every dependency, or only the root | Every node, as today (§6.1); the identity spec moves it to the root only. |
| Existing runs on the volume | Untouched: no path or byte changes, proven by the golden fixtures and the scan (§10). |
| Schema version and canonical JSON rules | No version, since the format does not change. Sorted keys, fixed top-level order, `-0.0` written as `0.0`, NaN refused (§6.1, §6.5). |
| Approved type registry | Neither a registry nor an allow-list; round-trip tests over every class instead (§6.4, §9). |
| New field constraints | Positivity on every dimension and count, as `Field(gt=0)` in the class definition; nothing that needs more than one argument or a short validator (§4.2). |
| Subclass param configs | Allowed in any param config field at any depth, recorded with a class key; `Model` has no flag and declares `vocab_size`, `sequence_length` and `dtype` (§4.5, §6.3). |
| What equality guarantees about manifests | Equal paths and equal definition encodings; complete manifests may differ in the recorded fields (§2). |
| A storage module | Yes: `storage.py` with `read_manifest` and `write_manifest`; `Artifact.load` stays as the convenience over it and the codec (§7). |
| Merge the two `Transformer` classes | No: they stay separate architectures (§4.5). |
| Floats too large to be exact, and `-0.0` | Accepted: rounding is kept; `-0.0` is written as `0.0` (§4.1, §6.5). |
| Resource updates, commit drift, leases | Unchanged from `lab.declare` and docs/spec.md (§8). |
| `same_definition()` or `==` | `==` (§2.1). |
| `ModelSpec` or `Model` subclasses | Subclasses (§4.5). |
| `StrictInt` per field or a strict config | A strict config (§4.1). |
| How artifacts and param configs share the config | One `STRICT` constant set on both bases; `Artifact` does not subclass `ParamConfig` (§4.1). |
| Where the class key is written and read | The codec, both directions (§6.3). |
| JSON-mode or Python-mode decoding | Python mode, arrays turned into tuples (§6.4). |

**Architectural rule:** pydantic declares and validates; the resolver checks graph semantics; the codec owns the manifest format, class keys included; `storage.py` reads and writes manifest files, and `Artifact.load` and `lab.declare` reach files only through it; declaration enforces consistency; `Model.build()` makes the network, on the worker.

---

# Part II: API Contracts

These are the signatures as they will be when this spec has landed. The identity spec then changes a few of them (its §11). Docstrings follow §10.0: the first line says what the function returns, at most one paragraph covers what a caller can get wrong, and a user-facing function adds one input/output example (omitted here for brevity). Methods not shown keep their current signatures and behavior.

## `artifacts/core/artifact.py`

```python
RECORDED = ("commit", "allocated_resources")


# Strict: no str as int, bool as int or list as tuple, though an int in a
# float field becomes a float.
STRICT = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class ParamConfig(BaseModel):
    """A validated, frozen parameter of an artifact: never a dependency, and
    never a folder or manifest of its own."""

    model_config = STRICT


class Resources(ParamConfig):
    """What a job should be given to run an artifact; None means Modal's
    platform default for that dimension. Never identity."""

    cpu: float | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


class Artifact(BaseModel):
    model_config = STRICT

    commit: str = Field(default_factory=_head, repr=False)
    allocated_resources: Resources = Field(default_factory=Resources, repr=False)
    producer: ClassVar[str | None]

    def __eq__(self, other: object) -> bool:
        """Returns whether `other` is the same definition: class, parameters and dependencies.

        `commit` and `allocated_resources` never take part, so an artifact
        rebuilt from an older manifest is still the same artifact.
        """

    def __hash__(self) -> int:
        """Returns a hash that agrees with `==`."""

    @classmethod
    def dependencies(cls) -> dict[str, type | None]:
        """Returns the fields that hold artifacts, sorted by name, each mapped
        to its container: tuple, frozenset, or None for a single artifact."""

    # Unchanged: deps, to_manifest, parameters, from_manifest, load, status,
    # bind, _load, paths, completion_paths, durable_progress, job, and the
    # abstract uid, artifact_path and files.
```

`Footprint` stays a dataclass, and so does `DeclarationReport` in lab.py: neither is validated or encoded.

## `artifacts/core/SGD/training.py` and `models/`

```python
class Model(ParamConfig):
    """The base of every architecture under models/: a subclass's fields are
    what a leg declares, and `build` makes the network they describe.

    A field annotated `Model` holds any subclass, and its manifest names the
    subclass under "class". The fields declared here are the ones stages and
    jobs read off any model.
    """

    vocab_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    dtype: str = "torch.float32"  # resolved worker-side via artifacts.core.locate.locate
    device: str = "cuda"

    def build(self):
        """Returns a freshly initialized torch module on this model's device."""
        raise NotImplementedError

    def step_flops(self, batch_size: int) -> int:
        """Returns the FLOPs of one forward and backward pass over batch_size
        sequences: an analytic estimate, not a measurement."""
        raise NotImplementedError


class Transformer(Model):  # in models/transformer and, separately, models/transformerV2
    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_ff: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    rope_theta: int = Field(gt=0)

    @model_validator(mode="after")
    def _heads_divide_d_model(self) -> Self: ...


class Mamba(Model):
    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_state: int = Field(default=16, gt=0)
    d_conv: int = Field(default=4, gt=0)
    expand: int = Field(default=2, gt=0)


class TrainingParameters(ParamConfig):
    total_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_norm: float
    lr_schedule: LRSchedule
    optimizer: str  # a torch.optim.Optimizer subclass, resolved by the job
    optimizer_parameters: OptimizerParameters
    lr_schedule_fn: str = "artifacts.core.SGD.lr_schedule.lr_cosine_schedule"
    seed: int = 0


class LoopConfig(ParamConfig):
    val_every: int = Field(gt=0)
    gpu_check_every: int = Field(gt=0)
    checkpoint_every: int = Field(default=100, gt=0)
    optimizer_checkpoint_policy: Literal["all", "latest"] = "latest"


class Training(Artifact):
    run_id: str
    training_parameters: TrainingParameters
    loop_config: LoopConfig
    model: Model  # filled in from starting_checkpoint when omitted (§4.2)
    starting_checkpoint: Training | None = None
```

## `artifacts/core/storage.py`

`MANIFEST`, `read_manifest(root, artifact_path) -> dict` and `write_manifest(root, artifact_path, manifest) -> None`, as in §7.

## `artifacts/core/resolve.py`

```python
def resolve(artifact: Artifact) -> list[Artifact]:
    """Returns every artifact `artifact` is built from and then itself,
    dependencies before dependents, each path once.

    A path reached twice must hold an `==` definition; the first object
    reaching a path is the one kept, recorded fields and all.
    """
```

## `artifacts/core/manifest.py`

```python
def parameters(artifact: Artifact) -> dict:
    """Returns the artifact's own fields as canonical JSON: the "parameters"
    half of its manifest, without the dependencies' manifests.

    A param config holding a subclass of its annotated class, at any depth,
    carries that subclass's dotted path under "class".
    """

def to_manifest(artifact: Artifact) -> dict:
    """Returns the artifact as a manifest dict: its class, commit, resources,
    parameters, and a manifest per dependency."""

def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
    """Returns the artifact a manifest describes and the tree under it, unbound.

    Each node keeps the commit and resources its manifest records. `memo`, one
    dict shared across calls, hands back the same instance for a subtree
    decoded before.
    """

def canonical(value: object) -> object:
    """Returns `value` as JSON data with keys sorted and -0.0 written as 0.0."""

def manifest_json(manifest: dict) -> str:
    """Returns one manifest as the text that belongs on disk."""
```

## `lab.py`

`declare` keeps its signature and docstring:

```python
def declare(artifact, *, root=STORAGE, commit=False, strict_commit=False,
            verbose=False, notebook=None) -> DeclarationReport: ...
```

The proposal's `Lab` facade class is not added: `lab` is a module of functions a notebook imports, with nothing to initialize.

## Contract test checklist

1. `Artifact.from_manifest(a.to_manifest()) == a`, with the same path, and the decoded artifact re-encodes to the same `manifest_json` bytes, for nested artifacts and param configs, including a `Model` subclass and a subclass nested inside another param config.
2. Two artifacts that differ only in `commit` or `allocated_resources` are `==` and hash equal, have one path and equal definition encodings, and have different complete manifests.
3. Golden fixtures: paths and bytes equal the values recorded in stage 0, before any code changed.
4. Two orders of a `frozenset` dependency give identical bytes; a tuple dependency keeps its order, and a different order is a different artifact.
5. Pydantic refuses wrong types, out-of-range values, unknown fields and NaN, both at construction and at decode.
6. `resolve` refuses a conflicting definition behind a shared path, and a cycle, without touching disk.
7. `load` refuses a manifest whose decoded path is not its folder, and names the folder for anything that will not decode.
8. `declare(commit=False)` writes nothing and reports every node.
9. `declare(commit=True)` refuses over blockers before writing, and keeps today's report behavior.
10. Every concrete `Artifact` and `ParamConfig` subclass in the repository has an example that round-trips (item 1), and a class without one fails the test.
11. `read_manifest` raises `FileNotFoundError` for a missing manifest and `ValueError` naming the file for unreadable JSON or a path that leaves the root; `write_manifest` writes exactly `manifest_json`'s bytes.
