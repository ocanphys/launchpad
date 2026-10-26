# Artifacts

*Draft.* Declaring and identifying artifacts. Binding and status (spec section 4) are still in the spec.

An artifact instance is an immutable definition with typed constructor parameters. `Artifact` is the abstract base; calling a concrete subclass constructs one:

```python
source = SourceURL(name="odyssey", url="https://example.org/odyssey.txt")
tokenizer = Tokenizer(          # artifacts.tokenizers.bpe.Tokenizer
    vocab_size=1000,
    special_tokens=("<pad>",),
    sources=(source,),
)
tokens = TokenizedSource(tokenizer=tokenizer, source=source)
```

Each concrete type belongs to a family: a package under `artifacts/` holding the artifact class and, in a sibling `jobs.py`, the job that produces it. Families never import functions or constants from one another; the only thing one family takes from another is an artifact class to name as a dependency type.

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

`deps()` returns only direct artifact-valued parameters, in a deterministic order: fields by name, sets by artifact path. Whether a field is a dependency is read off its annotation ([Manifests](manifests.md)), never its value, so an empty dependency container is still a dependency and an empty tuple of strings is still a parameter.

Paths, files, producers, and dependency lists are derived. They are not constructor arguments or separately persisted state. Artifact modules remain lightweight; resolving a producer imports its implementation only when executing it.

## Identity and agreement

Keep three questions distinct:

1. **Same identity:** two artifacts compute the same `artifact_path`. Identity is always written as `artifact.artifact_path`, never as `==`. It is what deduplication, lease keys, manifest locations, and every map of artifacts key on.
2. **Same definition:** concrete type, parameters, and dependency definitions agree recursively. Commit, resources, and bound state are excluded at every node. **This is what `==` and `hash()` mean.** The frozen dataclass generates both: `commit` and `allocated_resources` are declared `compare=False`, and bound state is set outside the declared fields. Binding does not change equality.
3. **Same provenance or allocation:** the recorded non-identifying fields agree.

Declaration and resolution check definition agreement with `==`, and always between two artifacts already known to share a path: a manifest read from that path, or a second route reaching it during traversal. Sharing a path is therefore never evidence of agreement: two sources named `odyssey` with different URLs land on one path, compare unequal, and must conflict.

A dependency field's declared type says whether its order identifies the artifact, and nothing else does. A `tuple` is a sequence: its order and its repeats are part of what the artifact is, as with a mapped dataset's concatenated source streams. A `frozenset` is a set: neither is, as with a tokenizer's `sources`. Equality and hashing then follow from the type, so no value is ever reordered to make two definitions compare equal.

Construction accepts any iterable for a set-valued field and rejects two members that share an artifact path, which is a conflicting definition wherever it appears. Anything rendering a set as a sequence (a digest, a JSON array, a traversal, a job's input stream) picks an order at that point. Those orders are renderings, never comparisons.

Parameters are immutable, including nested configuration. A result-changing setting belongs among the parameters even if it also affects execution; `commit` and `allocated_resources` are not identity. Resource allocation never silently changes scientific configuration.

## Location and ownership

`artifact_path` is a nonempty, normalized relative path, and `load` refuses one that is absolute or climbs out of its root. Owned filenames are unique bare filenames, excluding `manifest.json` and the temporary names publication derives from them ([spec §7](spec.md#7-state-inspection-and-execution)). This keeps nested artifacts independent:

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

## REPO

::: artifacts.core.artifact

