1
## NOTES ON SIMPLIFICATION

# Launchpad: Artifact Validation, Serialization, and Declaration — Architecture Review

## 1. Objective

Simplify Launchpad's artifact lifecycle by separating four responsibilities:

1. **Pydantic:** Define and validate artifacts.
2. **Resolver:** Validate the dependency graph and produce a topological ordering.
3. **Codec:** Convert artifacts to canonical JSON and reconstruct artifacts from JSON.
4. **Declaration:** Compare requested artifacts against stored manifests, detect conflicts, and persist changes.

The goal is to eliminate redundant validation, serialization, and comparison logic while preserving deterministic artifact identity, dependency integrity, and reproducibility.

---

## 2. Artifact Definition — Pydantic

Every artifact should extend a common Pydantic `BaseModel`.

The artifact class defines parameters and dependencies using typed fields.

```python
from pydantic import BaseModel, ConfigDict, Field

class Artifact(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )

class Tokenizer(Artifact):
    vocab_size: int
    sources: frozenset[Source] = Field(
        default_factory=frozenset
    )
```

Pydantic handles type validation, nested model validation, defaults, and immutable model instances.

**Important distinction:** Constructing a nested artifact validates its fields and dependencies through Pydantic, but does not establish graph-wide consistency, such as conflicting definitions sharing the same artifact path.

### Ordered versus unordered fields

The field type expresses whether ordering affects identity.

- `tuple[T, ...]`: order and duplicates are meaningful.
- `frozenset[T]`: order is irrelevant and duplicates collapse under equality.

For example, a tokenizer built from English and literature sources should have the same definition regardless of the order in which those sources were supplied.

Pydantic can validate and construct `frozenset` fields, but deterministic JSON serialization still requires canonical ordering.

### Removing `schema.py`

The existing `schema.py` duplicates parts of Pydantic's type introspection and validation capabilities.

Most of this can be eliminated by using Pydantic field metadata (`model_fields`) and standard field annotations.

However, Launchpad still needs small, explicit rules for distinguishing artifact dependencies from ordinary parameters and supporting polymorphic types.

The objective is not to eliminate these rules, but to centralize them rather than maintain a second validation framework.

---

## 3. Dependency Resolution

The resolver accepts a validated artifact and produces a dependency-first, topologically ordered list.

```python
resolve(artifact) -> list[Artifact]
```

Its responsibilities are to detect cycles, identify conflicting definitions sharing an artifact path, deduplicate repeated dependency nodes, and establish a deterministic traversal order.

For example:

```text
Source A ───┐
            ├── Tokenizer ── TokenizedSource ── Dataset
Source B ───┘
```

The resolver returns dependencies before their dependents, with each distinct artifact appearing once.

**The resolver should remain independent of serialization and storage.** It operates entirely on validated Python objects.

Although Pydantic validates individual nested models, graph-level consistency checks are still necessary.

---

## 4. Codec — Canonical Serialization

The codec owns the transformation between Python artifacts and their serialized representation.

Its public interface is:

```python
to_manifest(artifact: Artifact) -> dict

from_manifest(manifest: dict) -> Artifact
```

The codec should not read or write files.

### Encoding

The forward flow is:

```text
Validated Artifact
       ↓
    resolve()
       ↓
Topologically ordered artifacts
       ↓
Canonicalize each artifact
       ↓
Embed dependency representations
       ↓
Canonical manifest
```

The codec may call `resolve()` internally through `to_manifest()`, while also supporting an already-resolved graph to avoid redundant traversal.

Each artifact is serialized once. The resulting representations are composed into nested manifests, reusing the previously computed dependency representations.

### Decoding

The reverse flow is:

```text
Stored manifest JSON
       ↓
Parse nested declarations
       ↓
Reconstruct dependencies first
       ↓
Instantiate parent artifacts
       ↓
Validated Artifact
```

`from_manifest()` recursively reconstructs dependencies and instantiates their parent artifacts.

Pydantic validates the reconstructed objects.

A separate global `resolve()` operation is not normally necessary during decoding, provided the decoder itself guarantees structural validity and prevents malformed or cyclic input from being accepted.

### Canonicalization

Canonical serialization must guarantee:

- Deterministic ordering of object keys.
- Deterministic ordering of `frozenset` members.
- Preservation of order for tuples.
- Consistent numeric representations.
- Explicit identification of concrete polymorphic artifact types.
- Stable round-trip behavior.

The key invariant is:

```python
from_manifest(to_manifest(artifact)) == artifact
```

Here equality means artifact-definition equality, ignoring explicitly designated nonidentity metadata.

---

## 5. Separate Artifact Identity from Stored Metadata

This was an important distinction in our discussion.

Currently, manifests contain:

```json
{
  "artifact": "path.to.Tokenizer",
  "commit": "abc123",
  "allocated_resources": {
    "gpu_count": 1
  },
  "parameters": {
    "vocab_size": 32000
  },
  "dependencies": {}
}
```

However, `commit` and `allocated_resources` are not identifying fields under Launchpad's existing rules.

Changing an artifact's GPU allocation should not change its identity.

We therefore distinguish two concepts.

**Canonical definition:** The artifact class, identifying parameters, and recursively embedded dependency definitions.

**Stored manifest:** The canonical definition plus provenance and execution metadata such as Git commit and allocated resources.

One clean implementation is to generate the canonical definition first and then construct the stored manifest by adding metadata.

The codec should remain the single authority for both representations. It should not require two unrelated serialization mechanisms.

### Equality versus hashing

We initially considered hashing canonical JSON to compare declarations.

However, hashing is unnecessary when both canonical definitions are already available.

```python
requested_definition == stored_definition
```

Python directly compares nested dictionaries and lists.

Hashing remains useful when a compact, stable identity is needed, such as an artifact UID, content-addressed path, cache key, or external reference.

For declaration equality, direct comparison is simpler and avoids hash collisions.

---

## 6. Declaration — Comparing Against Storage

`lab.declare()` becomes the outer orchestration boundary.

It should receive an artifact and use the codec's canonical representations to inspect the corresponding storage locations.

The workflow is:

```text
Validated Artifact
       ↓
Codec + Resolver
       ↓
Canonical definitions and manifests
       ↓
Read existing manifests from storage
       ↓
Compare definitions
       ↓
DeclarationReport
       ↓
Optional persistence
```

Each artifact can be classified as:

- **New:** No declaration exists at the expected path.
- **Declared:** A matching definition exists but outputs are not yet present.
- **Partial:** Some required outputs exist.
- **Done:** All required completion files exist.
- **Conflict:** The existing declaration differs from the requested definition.
- **Undeclared:** Outputs exist without an accompanying manifest.

Git commit drift and allocated-resource differences remain separate from definition conflicts.

This preserves the behavior already supported by the current `DeclarationReport`.

### Comparing declarations with DeepDiff

Rather than maintaining custom recursive difference-reporting logic, consider using the Python package `DeepDiff`.

```python
from deepdiff import DeepDiff

diff = DeepDiff(
    stored_definition,
    requested_definition,
)
```

A nonempty difference indicates disagreement.

DeepDiff can also report which nested fields changed, making it useful for the verbose declaration report.

Since the comparison occurs on canonical definitions that already exclude nonidentity metadata, the diff does not need complex exclusion rules.

DeepDiff is optional. Python's `==` remains sufficient for equality checks.

### Persistence

After comparison, `declare()` decides whether writing is permitted.

For a committed declaration:

1. Inspect every requested artifact before making changes.
2. Reject the operation if any blocking conflicts exist.
3. Write missing manifests in dependency order.
4. Apply permitted metadata changes while preserving existing definitions and provenance.
5. Commit the storage changes.

For a preview, return the same report without writing.

The codec never needs to know whether the manifest will be stored locally, on a Modal volume, or in another storage backend.

---

## 7. Recommended Module Boundaries

| Component | Responsibility |
|---|---|
| `artifact.py` | Pydantic artifact base class, identity-related API, artifact-specific behavior |
| `resolve.py` | Dependency traversal, topological ordering, cycle and conflict detection |
| `codec.py` | Canonicalization, `to_manifest`, `from_manifest`, definition representation |
| `declare.py` or `lab.py` | Storage inspection, declaration comparison, reports, persistence |
| Storage interface | Read/write operations and storage backend integration |

`schema.py` can potentially disappear, with its necessary semantic rules absorbed into artifact definitions and codec helpers.

`locate.py` can remain as a small utility for resolving dotted class paths.

## 8. Important Design Invariants

The refactoring should explicitly test the following properties.

**Round-trip correctness**

```python
from_manifest(to_manifest(a)) == a
```

**Canonical identity**

```python
a == b  # definition equality

# implies
canonical_definition(a) == canonical_definition(b)
```

Ideally the converse holds as well.

**Order independence for sets**

```python
Tokenizer(sources=frozenset({a, b}))
==
Tokenizer(sources=frozenset({b, a}))
```

**Metadata independence**

Changing the Git commit or GPU allocation does not change artifact-definition equality.

**Graph consistency**

Two unequal artifacts must not claim the same artifact path within one resolved graph.

**Storage consistency**

A stored manifest must reconstruct an artifact matching the storage path in which it appears.

**Determinism**

Canonical representations must not depend on Python set iteration order, dictionary insertion order, or the machine executing the declaration.

---

## 9. What Becomes Cleaner

The main simplifications relative to the current implementation are:

| Existing complexity | Proposed simplification |
|---|---|
| Custom annotation validation in `schema.py` | Pydantic model definitions and minimal semantic introspection |
| Dataclass normalization through repeated Pydantic round trips | Native Pydantic model validation |
| Recursive manifest construction repeated for shared subtrees | Resolve once, serialize once per distinct node |
| Comparison by reconstructing stored artifacts into Python objects | Compare canonical definitions directly, after validating stored input as needed |
| Custom `_leaves()` comparison and diff generation | DeepDiff for diagnostics |
| Identity digest logic implemented independently in artifact families | Derive digests from one canonical definition representation |
| Serialization, identity, and storage concerns mixed across modules | Explicit artifact, resolver, codec, and declaration boundaries |

## 10. Remaining Design Decisions

There are a few details worth resolving before implementation.

**First, Pydantic equality.** Pydantic compares fields by default, including metadata. Launchpad must decide whether to override artifact equality or rely exclusively on canonical-definition equality for identity-sensitive operations. Both approaches must produce consistent results.

**Second, canonical JSON versus stored manifest.** The codec should clearly define which representation determines identity and which contains additional metadata. This distinction should be explicit in its API.

**Third, backward compatibility.** Existing manifests currently encode specific class paths, field structures, and numeric representations. Changes to these formats may require migrations or a versioned codec.

**Fourth, declaration concurrency.** Inspecting all manifests before writing prevents known conflicts but does not make the write operation atomic. Concurrent declarations can change storage between inspection and persistence. Locks, leases, or transactional storage semantics remain a separate concern.

---

## Conclusion

The clean architecture is:

**Artifact (Pydantic) → Resolver → Codec → Declaration → Storage**

And the reverse path is:

**Storage → Codec → Artifact (Pydantic)**

Pydantic validates the models, the resolver establishes graph consistency, the codec guarantees canonical and reversible serialization, and declaration enforces compatibility with existing storage.

The central principle is that **an artifact's identity should have one canonical definition, reused for comparison, hashing, serialization, and consistency checking**.

This reduces duplicated logic while preserving Launchpad's core requirements: deterministic artifacts, reproducible manifests, explicit dependencies, and safe declarations.