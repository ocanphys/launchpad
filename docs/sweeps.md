# Artifact sweeps

Status: implemented runtime extension to the [artifact backbone](../artifacts/core/spec.md). Constructor dispatch, expansion, graph resolution, declaration, and transport are covered by [sweep tests](../test/test_sweep.py). Static constructor overloads for lifted input types are not generated; declaration inputs are validated at runtime.

A concrete `Artifact` is fully resolved: all parameters are concrete values and all dependencies are concrete artifacts. Users keep defining artifacts through their normal constructors. If any sweepable input contains a `Sweep` or a nested unresolved declaration, the framework transparently returns an internal `ArtifactSweep[A]` representing that declaration. It constructs actual instances of `A` only after their inputs have been resolved.

> **Sweeps exist only in the declaration layer. The execution DAG contains only concrete `Artifact` objects.**

Here, fully resolved describes the definition, not whether a job has produced its output files. A concrete artifact can still be undeclared, unbuilt, or unbound.

```text
User declaration
      ↓
Artifact or invisible ArtifactSweep
      ↓
expand children recursively
      ↓
Cartesian product of independent alternatives
      ↓
construct concrete Artifacts
      ↓
resolve the combined dependency graph and deduplicate
      ↓
final concrete Artifact DAG
```

The central rule is **every sweep is its own choice: Cartesian product across sweeps at every level, concatenation across alternatives within a sweep**. Reusing a `Sweep` object in two positions makes two choices. An artifact declaration used in several places is one node: the sweeps inside it are chosen once per combination, so every place that uses it sees the same artifact.

**Expansion** removes sweeps and produces concrete root variants. **Resolution** retains its existing codebase meaning: traverse dependencies and return a deduplicated list of artifacts in dependency order. The public `resolve` function performs expansion internally before resolving the combined graph.

## 1. Input model

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Sweep[T]:
    values: tuple[T, ...]  # an alternative may itself be a Sweep

Sweep((64, 128, 256))                 # three scalar alternatives
Sweep((books_dataset, code_dataset)) # two dependency alternatives
Sweep((pretraining_a, pretraining_b)) # two root alternatives
```

The declaration above describes the public data shape. Construction accepts a finite, ordered sequence, including `range`, and snapshots it into a tuple. Strings, unordered collections, generators, and empty sequences are rejected. A singleton sweep is valid. Duplicate alternatives are permitted; their concrete artifact results are deduplicated during resolution.

`Sweep[T]` is a choice among values of the declared field type, and a type checker reads `Sweep((64, 128))` as an `int`, so it fits a field annotated `int`. It is not an artifact, has no producer or storage path, and never appears in an artifact manifest. Alternatives may themselves contain sweeps. A directly nested sweep has the same concatenation semantics: `Sweep((Sweep((1, 2)), 3))` expands to `1, 2, 3`.

`ArtifactSweep[A]` is an immutable internal declaration with the same field schema as artifact type `A`: field names, concrete types, defaults, and parameter/dependency roles come from `A`. Its values may additionally contain `Sweep[T]`, nested `ArtifactSweep`s, and unresolved parameter records. It records its target artifact type and constructor inputs, including captured metadata defaults.

`ArtifactSweep[A]` is not an `Artifact` subclass. It has no artifact identity, outputs, producer job, or manifest. Users neither instantiate nor wrap it explicitly. Reading a declared field is supported for composition; accessing properties or methods that require a concrete artifact is an unresolved-declaration error. The target class's identity properties and semantic validators are never invoked on the wrapper.

```python
# Illustrative constructors: every other required field is supplied.
concrete = A(size=64, **other_inputs)
declaration = A(size=Sweep((64, 128)), **other_inputs)

assert isinstance(concrete, A)
assert not isinstance(declaration, Artifact)
# Internally: declaration is ArtifactSweep[A].
```

A parent declaration containing an `ArtifactSweep` becomes an `ArtifactSweep` too, even if it contains no direct `Sweep`. Parameter records are ordinary frozen dataclasses: `Transformer(d_model=Sweep(...), ...)` remains a `Transformer` instance containing choices. No registration is required, and a base class (`Record`) only where the record has a `__post_init__` check. An artifact containing such a record becomes an `ArtifactSweep`; concrete artifacts contain only concrete records. Here, **template** means an unresolved declaration.

Supported positions in the first version:

| Position | Meaning |
| --- | --- |
| Entire request | Alternatives among root artifact definitions. |
| Scalar or optional field | Choose one value, including `None` if the field permits it. |
| Frozen parameter dataclass | Sweep the whole record or fields nested inside it. |
| Direct dependency field | Accept a concrete artifact, an `ArtifactSweep` for an assignable artifact type, or a sweep of either; recursively resolve the selected declaration. |
| Tuple field | Sweep the whole tuple or individual elements; preserve order and repeats. |
| Set-valued dependency field | Sweep the whole collection or members supplied through an ordered tuple; normalize to `frozenset` after expansion. |

Ordinary collections remain collections: `(64, 128)` is one tuple value, not two alternatives. For example, `Sweep(((a, b), (c,)))` chooses one of two dependency collections, whereas `(Sweep((a, b)), c)` generates `(a, c)` and `(b, c)`.

Nested parameter records cannot hide artifact dependencies: artifacts must remain in declared dependency fields, as required by the backbone. Arbitrary mappings, mutable parameter structures, dictionary keys, and opaque objects containing sweeps are unsupported in this version; report their field paths rather than treating them as concrete leaves.

`commit` and `allocated_resources` remain concrete recorded metadata and cannot contain sweeps. They do not define distinct artifacts, so sweeping them would not create distinct jobs. Result-changing settings, such as a random seed, belong in identifying parameters and can be swept.

## 2. Recursive expansion

Let `E(x)` denote the concrete alternatives produced by a value or template, including repeats. These equations specify which combinations exist and their multiplicities when every artifact declaration appears once; a declaration used in several places is one node (see "Shared declarations are one node" below), and enumeration order is defined separately below:

```text
E(concrete scalar) = [scalar]

E(concrete Artifact) = [Artifact]

E(Sweep((a, b, ...))) = E(a) ++ E(b) ++ ...

E(ArtifactSweep[A](f1=v1, ..., fn=vn)) =
    [construct_and_validate(A, c1, ..., cn)
     for (c1, ..., cn) in product(E(v1), ..., E(vn))]

E(parameter_record[P](f1=v1, ..., fn=vn) containing sweeps) =
    [construct_and_validate(P, c1, ..., cn)
     for (c1, ..., cn) in product(E(v1), ..., E(vn))]

E(tuple(v1, ..., vn)) =
    [tuple(c1, ..., cn)
     for (c1, ..., cn) in product(E(v1), ..., E(vn))]
```

Concrete parameter records also expand to themselves. A dependency's expansion supplies alternatives to its parent's product, rather than leaving a list in the resolved dependency field. Each resulting parent refers to exactly one concrete choice per dependency occurrence.

Operationally, at an artifact level:

1. Recursively expand every parameter and dependency field to its concrete alternatives, including nested parameter records and nested `ArtifactSweep`s.
2. Form the Cartesian product of those independent field alternatives at this node.
3. For each combination, invoke the target artifact's normal constructor with concrete inputs and validate the result.
4. Return those concrete alternatives to the parent. Continue upward until the root is a collection of fully concrete artifacts. Dependency resolution then traverses and deduplicates their combined graph before declaration or execution.

The equations and these steps define the same recursive process when no declaration is shared. The implementation enumerates top-down instead: it finds the next sweep without a choice, following the choices already made, branches on its alternatives, and builds each complete assignment through the normal constructors. A choice is keyed by the declaration it sits in and its position there, which gives the same combinations and order and makes a shared declaration one node. A dependency's choice domain never depends on the parent selection.

Only selected branches contribute axes. If a dependency sweep chooses between a books template with two expansions and a code template with three, that field has `2 + 3 = 5` alternatives. It does not have `2 × 3` alternatives. Other independent fields multiply those five alternatives.

With no sweeps, expansion yields exactly the supplied concrete artifact. Empty dependency tuples are legal where the artifact type allows them and contribute one empty collection, not zero combinations. An empty `Sweep` is an input error, preventing an accidental request that silently resolves to nothing.

Ordering is deterministic: visit record fields by name, recursively enumerate each field's concrete alternatives, and vary the last factor fastest in the node's Cartesian product. Sweep alternatives and tuple elements retain their input order; a sweep exhausts each alternative before the next. Apply the same rules recursively. Concrete dependency sets are traversed by artifact path. Unresolved members of a set-valued field must be supplied as an ordered tuple; do not hash them or sort them by a path that does not exist yet. Normalize and check set members only after they are concrete.

### Shared declarations are one node

```python
width = Sweep((64, 128))
pair = Pair(left=width, right=width)  # illustrative parameter record
```

This produces all four pairs: `width` in two positions is two choices, and object identity never links parameters. An artifact declaration is different, because it is a node: a tokenizer declaration that a dataset's train and valid splits both use is one tokenizer per combination, never two. Two separately written declarations are two nodes, each with its own choices. Equal concrete dependencies still share one graph node.

## 3. Resolve and declare APIs

Every request is a sweep: a concrete artifact is a sweep of one, and a `Sweep` of artifacts is a group. `expand`, `resolve`, and `declare` all accept the same `Declaration`:

```python
Declaration = Artifact | ArtifactSweep[Artifact] | Sweep[Artifact | ArtifactSweep[Artifact]]

def expand(declaration: Declaration) -> list[Artifact]: ...   # the flat sweep of roots
def resolve(declaration: Declaration) -> list[Artifact]: ...  # their combined graph
```

`ArtifactSweep` appears in this signature to state the contract; callers never import or construct it. `expand` returns the concrete roots in product order, duplicates kept, and `[a]` for a concrete `a`. `resolve` expands its request and walks the roots' dependencies with one shared visited-path registry and the cycle and agreement checks, returning every concrete artifact needed, dependencies before dependents, one entry per path across the whole request.

A root may also be another root's dependency. Its graph entry is emitted once, so roots cannot be inferred from the graph; a caller that needs them calls `expand`. Because a group of concrete artifacts is itself a sweep, `resolve(Sweep(roots))` walks already-expanded roots without rebuilding them.

```python
roots = expand(pretraining)       # every concrete run the declaration stands for
artifacts = resolve(pretraining)  # everything they are built from, each path once

report = lab.declare(pretraining)              # preview every distinct path
print(report.root_paths)                      # requested concrete outputs
report = lab.declare(pretraining, commit=True) # publish after full validation
```

A concrete request goes through the same code as a sweep, with one root.

`lab.declare` and `local.declare_on_volume` accept the same request shape. Their existing options retain their meaning. The declaration report adds `root_paths`, a list of distinct requested root paths as strings in first-requested order. Its existing rows and state counts remain one per unique artifact path. It stays serializable without carrying templates or graph objects.

The number of distinct candidate jobs is the number of graph entries whose `producer` is not `None`; actual launch eligibility still depends on storage and execution state.

Declaration performs these steps once for the whole request:

1. Expand the request into its distinct roots, then resolve that group (`resolve(Sweep(roots))`), completing concrete validation, graph deduplication, and collision checks before storage inspection or writes. Keep the roots for reporting and notebook copies.
2. Inspect each unique graph artifact once and build the existing report rows. Preview writes nothing.
3. On commit, reject the entire request if any row blocks; otherwise publish in dependency order with the existing commit and resource-update rules.
4. For each newly created concrete root that is a `Training`, write its notebook copy using that root's `run_id` and `uid`. Do not use the original template's identity or `rows[-1]`.
5. Return and print one combined report. Never implement this as a loop of independently committed `declare` calls.

An invalid combination or in-request path conflict raises before a declaration report exists, as structurally invalid graphs do today. Storage conflicts remain report rows in preview and block commit. Preflight prevents known invalid requests from partially publishing; it does not add a filesystem transaction or roll back I/O failures during publication.

Manifests retain their current concrete format. Loading, binding, job construction, launching, leases, and dashboard entries continue to address one concrete artifact path. Declaration does not execute jobs. The remote wrapper serializes internal declarations as target types and field data, with no callbacks or expansion context, and uses the same resolver remotely. Declaration transport is separate from an artifact manifest; `ArtifactSweep` is never published to storage as an artifact.

## 4. Deduplication and identity

Use one path registry for the entire resolution, including all roots and all depths. Preserve the backbone's distinction between identity and agreement:

| Repeated concrete result | Behavior |
| --- | --- |
| Same path, same recursive definition | Reuse the graph entry. |
| Same path, different type, parameters, or dependency definitions | Raise a conflict naming the path and originating choices. |
| Different paths | Keep distinct entries, even if the jobs might produce equivalent bytes. |

Commit and resource metadata remain outside definition equality. Agreeing duplicates keep the first encountered concrete graph object and its recorded metadata, following the current resolver. Root entries reference that path's retained graph object. Sharing is by path and definition; references inside the retained objects do not have to be the same Python instance.

Expansion position and sweep position never enter `uid`, `artifact_path`, or a manifest. Resolving a value chosen from a sweep must produce the same definition and path as constructing that value directly. The same artifact declared outside a sweep is therefore reused automatically.

Do not deduplicate an unresolved template using `uid`, a partial hash, or its path. Validate recursive definition agreement when a concrete path repeats; a repeated parent must not conceal a conflicting dependency. For a set-valued dependency field, check same-path disagreement before collapsing agreeing members into a `frozenset`.

For example, `SourceURL(name="books", url=Sweep((url_a, url_b)))` yields conflicting definitions at `sources/books`, since a source's current path uses its name. Resolution must fail. Use a sweep of separately named `SourceURL` definitions when requesting both. Sweeps do not automatically change existing path encodings or make their hashes collision-free.

Built-in deduplication means one graph entry and one declaration row per artifact path. It does not create an exactly-once execution guarantee; launching retains the existing completion and lease protocol.

## 5. Independence: contract and enforcement

Independence here is about **choice domains**, not whether a parent can read a dependency's declared configuration. Define it as follows:

- A subtree's expansion is a pure function of that subtree's explicit input definition. It cannot consult an ancestor selection, a sibling selection, storage contents, runtime job outputs, or global expansion state.
- Every product combination must be a valid concrete definition. Validators report violations; they never silently filter combinations or select a replacement dependency.
- Normalization can derive fixed fields from selected concrete inputs, but cannot introduce sweeps, add dependencies, or alter another occurrence's domain.

This permits varying numbers of alternatives in different branches of a `Sweep`. Each selected branch still has a fixed domain independent of the context in which it appears.

### What can be enforced

| Mechanism | Guarantee and limit |
| --- | --- |
| Immutable, finite data-only sweeps | No generators, parent references, predicates, or callbacks whose output depends on an enclosing choice. Freezing is recursive, not just a frozen outer dataclass. |
| Schema-aware traversal | Every choice is in an admitted parameter or dependency position; hidden dependencies and unsupported containers fail with a field path. |
| A choice per sweep position, a node per declaration | Every sweep multiplies; a declaration reached through several fields is built once per combination and shared. |
| Concrete validation of every combination | Establishes that the supplied finite product is valid according to the declared validators. One failure rejects resolution. |
| Validation of normalization output | No unresolved values remain; normalization has not inserted dependencies or changed independently chosen fields. Explicit derived/default fields are the only permitted filled values. |
| Review and conformance tests for artifact hooks | Needed because arbitrary Python validators and properties can still read globals or do I/O. Types alone cannot prove purity or scientific independence. |

The framework can enforce the data model and validate a finite request. It cannot prove that arbitrary artifact implementations, model code, or external data obey all semantic assumptions. Compatibility checks must live in lightweight artifact or parameter validation, not only in job execution, if resolution is expected to catch them.

### Constraints already present in this repository

A dataset's sources share one tokenizer, which is the leg's tokenizer, and `Pretraining` requires the model vocabulary size to match it and a starting checkpoint to use it too. Sweeping one tokenizer declaration that every source uses is valid, since a shared declaration is one node; mixing unrelated datasets and tokenizers is not automatically valid. The main example below fixes one compatible tokenizer for both datasets.

`Training` inherits `model` when it is omitted and a checkpoint is provided. Retain this as deterministic normalization **after selecting a concrete checkpoint**. Omission is not a second independent axis. An explicitly supplied model or model-parameter sweep must agree with every selected checkpoint combination or resolution fails.

Model constraints such as `d_model % num_heads == 0` are another case: a product containing an invalid pair is an invalid request. Adding the sweep framework does not itself add all missing model validators; the relevant parameter types must declare them.

### Expressing correlated choices

When values must move together, put complete compatible configurations into a single sweep at their nearest common enclosing record:

```python
models = Sweep((
    Transformer(d_model=64, num_heads=4, **common_transformer_fields),
    Transformer(d_model=128, num_heads=8, **common_transformer_fields),
))
```

This chooses two model configurations. Independent sweeps in `d_model` and `num_heads` would choose four. Here `common_transformer_fields` supplies the remaining concrete fields and excludes those two keys.

For a dataset/tokenizer/model relationship, use `Sweep((compatible_pretraining_a, compatible_pretraining_b))`. Common independent settings may still contain their own sweeps within each branch. A convenience factory can construct these finite definitions before resolution, but the resolver never invokes user-provided factories to generate domains. Sharing one declaration is the only way two positions see the same choice; there is no implicit zip, constraint solver, or invalid-combination pruning.

## 6. Examples

### Parameter and dependency sweeps: 12 pretraining roots

Assume `books_dataset` and `code_dataset` are distinct concrete datasets tokenized with the same concrete `tokenizer`. `training_parameters` and `loop_config` are concrete and valid. Under the proposed constructors:

```python
from artifacts.core.sweep import Sweep
from artifacts.core.resolve import resolve
from artifacts.stages.pretraining import Pretraining
from models.transformer import Transformer

pretraining = Pretraining(
    run_id="architecture-sweep",
    dataset=Sweep((books_dataset, code_dataset)),
    model=Transformer(
        vocab_size=tokenizer.vocab_size,
        sequence_length=512,
        d_model=Sweep((64, 128, 256)),
        num_layers=Sweep((4, 8)),
        d_ff=128,
        num_heads=4,
        rope_theta=10_000,
    ),
    training_parameters=training_parameters,
    loop_config=loop_config,
)

artifacts = resolve(pretraining)
# In this example, only the requested roots are Pretraining artifacts.
runs = [artifact for artifact in artifacts if isinstance(artifact, Pretraining)]
assert len(runs) == 2 * 3 * 2
```

`pretraining` itself is internally an `ArtifactSweep[Pretraining]`, because both a dependency and model parameters are unresolved. Only the 12 resolved roots are `Pretraining` artifacts.

The root combinations are:

| Dataset | `(d_model, num_layers)` |
| --- | --- |
| books | `(64, 4)`, `(64, 8)`, `(128, 4)`, `(128, 8)`, `(256, 4)`, `(256, 8)` |
| code | `(64, 4)`, `(64, 8)`, `(128, 4)`, `(128, 8)`, `(256, 4)`, `(256, 8)` |

Every root has a single dataset, integer dimensions, and concrete dependencies all the way down. The graph contains 12 distinct pretraining artifacts plus the union of both datasets' dependency graphs. Each dataset and the shared tokenizer appear once. The total graph size is not necessarily 12, and virtual artifacts do not have jobs.

### New sweeps encountered inside a dependency: 18 roots

```python
from artifacts.dataset import DataSet

books_variants = DataSet(
    train_set=Sweep((
        (books_tokens,),
        (books_tokens, extra_books_tokens),
    )),
    valid_set=(books_validation_tokens,),
)

common_pretraining_inputs = dict(
    run_id=pretraining.run_id,
    training_parameters=pretraining.training_parameters,
    loop_config=pretraining.loop_config,
)

nested = Pretraining(
    dataset=Sweep((books_variants, code_dataset)),
    model=pretraining.model,
    **common_pretraining_inputs,
)

artifacts = resolve(nested)
runs = [artifact for artifact in artifacts if isinstance(artifact, Pretraining)]
assert len(runs) == (2 + 1) * 3 * 2
```

All tokenized inputs here use the same tokenizer, whose special tokens include the separator required for a multi-source dataset. Assume the two books variants and the code dataset have distinct paths. The books branch adds two dataset alternatives; the code branch contributes one. Each combines with the six architecture settings. Common token streams and their raw sources are deduplicated across all 18 roots.

### Duplicate choices do not create duplicate jobs

```python
import lab

duplicates = Pretraining(
    dataset=books_dataset,
    model=Transformer(
        vocab_size=tokenizer.vocab_size,
        sequence_length=512,
        d_model=Sweep((64, 64, 128)),
        num_layers=4,
        d_ff=128,
        num_heads=4,
        rope_theta=10_000,
    ),
    **common_pretraining_inputs,
)

report = lab.declare(duplicates)  # preview; writes nothing
assert len(report.root_paths) == 2
```

The two requests for width 64 agree and share one pretraining artifact. That artifact also has exactly the path of an otherwise identical request with `d_model=64` written directly.

### A sweep below a resumed training leg

```python
from dataclasses import replace

continuation = Pretraining(
    run_id="continuation-sweep",
    starting_checkpoint=pretraining,  # the template with 12 concrete roots
    dataset=books_dataset,
    training_parameters=replace(
        training_parameters,
        seed=Sweep((10, 20)),
    ),
    loop_config=loop_config,
    # model inherits from each concrete checkpoint
)

artifacts = resolve(continuation)
# Both starting legs and continuation roots are Pretraining artifacts here.
continuations = [
    artifact for artifact in artifacts
    if isinstance(artifact, Pretraining)
    and artifact.run_id == "continuation-sweep"
]
assert len(continuations) == 12 * 2
```

The 24 continuation roots inherit model settings from their selected checkpoints. Their graph includes the 12 starting legs once each, plus shared inputs. The parent's seed choices do not change how the checkpoint template expands. The `replace` call starts with a concrete `TrainingParameters` record and produces another ordinary parameter record containing the new sweep. This illustrates why concrete artifact construction must wait until the selected dependencies are resolved.

## 7. Construction, typing, and implementation

Simply adding `Sweep` to `resolve` is insufficient. Today, `Artifact.__post_init__` hashes set members and reads their paths, `Training.__post_init__` inspects and inherits checkpoint fields, and `Pretraining.__post_init__` reads dataset and tokenizer attributes. These operations cannot run on unresolved choices.

`Artifact` uses `DeclarationMeta`, an `ABCMeta`-compatible constructor dispatcher. It binds constructor inputs, captures defaults, and recursively checks field types, alternatives, dependency roles, and immutability before invoking artifact initialization. If inputs contain choices, it returns `ArtifactSweep`; otherwise it runs the ordinary artifact constructor and validates the concrete result.

Parameter records use normal dataclass constructors:

```python
from dataclasses import dataclass
from artifacts.core.sweep import Record

@dataclass(frozen=True)
class Parameters(Record):
    width: int

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("width must be positive")
```

Annotations alone do not enforce runtime types or defer Python code. Framework schema validation happens when a record enters an artifact constructor. A parameter `__post_init__` runs as soon as the record is written, sweeps and all, so a record with a check subclasses `Record`, which skips the check while any field holds a `Sweep`. Expansion constructs each concrete record normally and runs these domain checks there, then checks its concrete schema and that normalization did not rewrite selected values. Already concrete records have already run their constructors. Fields can disable sweeping with `field(metadata={"sweepable": False})`; `init=False` fields cannot contain sweeps.

Artifact semantic checks run only with concrete inputs. Checkpoint inheritance therefore sees a concrete checkpoint. `lab.declare` expands and validates the complete graph before publishing manifests, so an invalid combination fails without partial publication. Manifest loading also reconstructs through ordinary constructors and validates the resulting artifacts.

The declaration representation exposes the target schema's fields for composition, but does not inherit `A`'s behavior. Requests for `uid`, `artifact_path`, `deps()`, `parameters()`, `to_manifest()`, binding, status, or job construction on an `ArtifactSweep` raise an unresolved-declaration error with a field location and a direction to call `resolve`. Wrappers are unhashable and their equality is never an artifact-agreement test. Concrete artifacts keep their existing equality, hashing, properties, and methods. The base `Artifact` class owns the concrete invariant: its constructor dispatcher validates the complete parameter structure, including after subclass initialization. Execution and manifest entry points call `Artifact.require_concrete(value)`, a lightweight type check that rejects wrappers without repeating recursive validation. Individual artifact subclasses need no concrete checks. Frozen instances must not be mutated after construction.

### Schema and static typing

Fields carry their concrete annotations only, `lr: float` rather than `lr: float | Sweep[float]`, and the constructor dispatcher accepts a `Sweep` in any sweepable field. The annotations are then exactly the manifest codec's schema, with nothing to strip before validating or decoding. For static checkers, `Sweep.__new__` is declared (under `TYPE_CHECKING`) to return `T`, so `lr=Sweep((1e-3, 1e-4))` checks as a float and `lr=Sweep(("x",))` is flagged where it is written. Pyright honours a `__new__` that returns another type; mypy rejects the declaration.

Artifact constructor dispatch is not automatically understood by static type checkers; generated overloads are not implemented, so a checker types a swept constructor call as the concrete artifact. Runtime validation guarantees that the execution graph contains no sweeps. `resolve` retains its `list[Artifact]` return type.

### Numeric JSON round trips

Python float parameters such as `lr=0.003` pass unchanged through expansion, manifest encoding, JSON storage, and loading, as the same finite binary float and artifact identity. Constructors normalize each parameter through the codec ([backbone spec §3](../artifacts/core/spec.md#3-manifests-and-loading)), so `lr=1` and `lr=1.0` declare one artifact at one path. Tests cover `0.003`, its adjacent distinct float, small/subnormal values, generated values of every supported shape (test/test_codec.py), and actual training manifests on disk.

This preserves the Python value, not the original spelling: `0.0030` and `0.003` are the same float. Binary floats do not provide exact decimal arithmetic, and `decimal.Decimal` is not supported by this parameter codec. Use a separately designed decimal representation if decimal scale or exact decimal arithmetic becomes a requirement.

Implementation boundaries:

| Location | Change |
| --- | --- |
| `artifacts/core/sweep.py` | `Sweep`, internal `ArtifactSweep`, sweep detection and constructor dispatch, `check` (constructor inputs against the field schema, returned with every concrete parameter normalized) and `verify` (a built instance keeps its chosen inputs), recursive expansion. |
| `artifacts/core/schema.py` | The field schema: which annotations a field may carry (`annotations`, `dependencies`), checked when a class is first used. |
| `artifacts/core/artifact.py` and parameter records | Dispatch artifact declarations before concrete initialization; reconstruct parameter records and rerun validators during expansion. |
| `artifacts/core/manifest.py` | The pydantic codec over that schema, including `normalize`, which every constructor runs; reject unresolved values at serialization/loading boundaries. |
| `artifacts/core/resolve.py` | Preserve `resolve(...) -> list[Artifact]`; share internal preparation with declaration; traverse all expanded roots with one graph registry. |
| `lab.py` | Consume one internal preparation result and inspect its graph once; report roots; copy notebooks for newly created training roots. |
| `local.py` | Accept and transport the same request shape and report. |

Declarations are frozen, so they cannot form cycles; only concrete graph traversal checks for path cycles and disagreements. Expansion makes one choice per sweep position within a declaration.

Expansion and graph ingestion are separate concerns. Enumerate complete concrete root candidates, then traverse their dependencies in the existing order and register agreeing paths. Caching or eagerly expanding a child must not preemptively register it and change which encountered object's metadata wins. Subtree caching must not collapse metadata-distinct candidates using definition equality before deterministic graph ingestion.

Generate products incrementally where practical; the final unique graph still needs materialization before declaration can publish. Cartesian growth is intrinsic. If an implementation adds an expansion limit, it must fail explicitly before writes, never truncate or silently sample. Retain choice locations for diagnostic errors without storing them in identity or manifests; for example, `root.dataset.values[0].train_set.values[1]`.

## 8. Acceptance criteria

| Case | Required result |
| --- | --- |
| Request without sweeps | One root; existing graph order, identity, validation, and metadata behavior. |
| Public `resolve` with concrete or swept input | Always returns `list[Artifact]`, including dependencies; existing iteration and indexing remain valid. |
| Declaration preparation | Expands and resolves once; retains requested roots independently from graph order and artifact type. |
| Normal constructor with concrete inputs | Returns the concrete artifact type and runs its ordinary validation immediately. |
| Normal constructor with a sweep in a nested parameter record | Returns internal `ArtifactSweep[A]` before concrete initialization. |
| Parent containing only a nested `ArtifactSweep` and concrete local parameters | Returns an `ArtifactSweep` even without a direct `Sweep` field. |
| Wrapper target does not match the dependency's concrete type | Declaration type error before expansion or concrete construction. |
| Constructor defaults and metadata | Captured once and preserved across all resolved variants. |
| Concrete construction or normalization introduces an unresolved value | Construction fails; no partially concrete artifact escapes. |
| Main example | 12 concrete roots; common dependencies emitted once. |
| Nested books example | 18 roots, proving alternatives add within a branch and multiply across independent fields. |
| Resumed training example | 24 roots, concrete checkpoint inheritance, 12 distinct starting legs. |
| Same sweep or dependency template in two fields | Full Cartesian product, regardless of shared object identity. |
| Sweep of complete paired configurations | Only the stated configurations, with any internal sweeps expanded recursively. |
| Duplicate choices, including duplicates in separate branches | Correct logical root count and one graph entry per agreeing path. |
| Root also requested as another root's dependency | Present once in the graph and explicitly retained among internal roots and the report's `root_paths`. |
| Sweep-free concrete dependency shared at many depths | One declaration row and existing first-encounter metadata rule. |
| Same path with conflicting parameter or nested dependency | Error before storage writes; report both originating locations. |
| Invalid Cartesian combination | Entire resolution fails with field/choice context; no silent filtering. |
| Empty sweep, wrong alternative type, hidden sweep, or hidden dependency | Input error with field location. |
| Whole tuple versus tuple element sweep | Correct collection shape; tuple order and repeated members preserved. |
| Set-valued dependency members | Expand before normalization; collapse agreement and reject same-path disagreement. |
| Optional dependency with `None` | Concrete variants with and without the dependency. |
| Template passed to a concrete-only operation | Clear unresolved-input error; no partial identity or manifest. |
| Final DAG and all fields below it | Only concrete `Artifact` objects and concrete parameter values; no `Sweep`, `ArtifactSweep`, or unresolved parameter record. |
| Concrete manifest round trip | Original definition and path; no sweep tags or schema change. |
| Declaration preview and blocked commit | One combined report; no writes for preview or any blocking request. |
| Successful commit and redeclaration | One manifest per path; existing files reused and metadata policy retained. |
| Multiple new training roots | One notebook copy per newly created root; no copy for roots merely reused. |
| Local and remote declaration | Equivalent roots, graph paths, and report semantics. |

Automatic scheduling, conditional search, persisted experiment definitions, and changes to artifact path encodings are outside this extension. Remote transport is tested through Modal serialization and the declaration wrapper locally; these tests do not deploy or launch jobs.
