# Launchpad: Identity Changes and Volume Migration

**Date:** October 26, 2026
**Status:** Design checkpoint. Starts after the pydantic artifact spec ("the pydantic spec" below) has landed. That spec was folded into `docs/` and deleted when it finished; it is in git history at commit `2acb192`.
**Scope:** Three deliberate changes to artifact identity and to the manifest format, and the one-off script that migrates the volume to match.

The pydantic spec changes how artifacts are written in code and leaves everything on the volume as it is: every path and every manifest byte stays identical, proven by golden fixtures and a volume scan. This spec does the opposite: it changes paths and manifest bytes on purpose, once, and moves the volume to match. Keeping the two apart means the refactor is checked against an unchanged volume, and this change is reviewed and rolled out on its own.

References such as "pydantic §6.3" point into the pydantic spec. That spec is folded into `docs/` and deleted as its last stage lands, so by the time this one starts its material lives elsewhere: §2 and §4 in `docs/artifacts.md` and `docs/models.md`, §6 in `docs/manifests.md`, §8 in `docs/lab.md`, Part II in the code's docstrings, and §10.1's fixtures and scan in `test/fixtures/golden/` and `migrations/scan.py`. §1 below restates every fact this spec relies on, so it reads on its own either way.

## 1. Prerequisites

Everything below assumes the pydantic spec's end state:

- `Artifact` and `ParamConfig` are pydantic models with the shared `STRICT` config; `==` and `hash` on `Artifact` ignore the fields named in `RECORDED` (pydantic §2.1).
- `manifest.py` holds the codec: `to_manifest`, `parameters`, `from_manifest`, `canonical`, the class key rule at every depth, and Python-mode decoding (pydantic §6).
- `storage.py` reads and writes manifest files (`read_manifest`, `write_manifest`, `MANIFEST`), and `Artifact.load` is `from_manifest(read_manifest(...))` plus the path check (pydantic §7).
- Equal artifacts have equal paths and equal *definition encodings* (the manifest without recorded fields at any node), while complete manifests may differ in the recorded fields (pydantic §2).
- `Model` is an ordinary `ParamConfig` that declares `vocab_size`, `sequence_length` and `dtype`, and still has `device` (pydantic §4.5).
- `test/fixtures/golden/` pins today's paths and bytes, and `migrations/scan.py` checks the volume (pydantic §10.1). Those golden manifests are exactly the format this spec migrates from.

## 2. The three changes (decided)

All three land in one commit, with one migration. Doing them together means one script and one move of the volume, not three.

### 2.1 `device` leaves `Model`

Today `Model` has `device: str = "cuda"`, and `Pretraining.lineage_hash` hashes `self.model.model_dump()`, which includes it. So the same leg declared for CPU gets a different folder, and a leg cannot be tested on a CPU machine at the path it trains at on a GPU. Device is an execution setting, like `allocated_resources`, not part of what is trained.

After: `Model` has no `device` field. `Model.build(device)` takes it as an argument. `TrainingJob` picks `cuda` when it is available and `cpu` otherwise, and passes the same device to `build` and to `GpuMonitor`; `PretrainJob` keeps using the job's device for batches and pinned memory. `profiling/util.py`, which runs only on CUDA, passes `"cuda"`. Nothing can force `cpu` on a GPU machine any more; nothing does today, and tests run on machines where the automatic choice is already `cpu`.

**Gotcha: legs that collapse onto one path.** A CPU leg and a CUDA leg that are otherwise identical become one definition with one path. The migration script refuses that rather than merge two folders of trained weights (§8, step 3); you choose which one to keep before running it again.

### 2.2 `lineage_hash` digests the encoded parameters

Today `lineage_hash` digests `type(self.model).__module__`, `self.model.model_dump()` and `self.training_parameters.model_dump()`. That overlaps with what the manifest records, but not exactly: the hash uses the model's module, while the manifest records module and class under `"class"`.

After:

```python
@cached_property
def lineage_hash(self) -> str:
    params = self.parameters()  # the canonical JSON written under "parameters"
    return _digest(
        self.starting_checkpoint.uid if self.starting_checkpoint else None,
        self.dataset.uid,
        self.tokenizer.uid,
        params["model"],
        params["training_parameters"],
    )
```

The folder is then a function of what its manifest says, and the two cannot drift apart. `params["model"]` includes the class key (pydantic §6.3), so moving or renaming an architecture class changes the path, which is intended: a different architecture is a different trajectory. The two `Transformer` classes, which have identical fields, keep distinct paths for the same reason. `loop_config` and `run_id` stay out, as today: the first controls observation, not the trajectory, and the second is already in the path.

**Gotcha: changes cascade down a run.** A leg's `uid` is part of the next leg's `lineage_hash` (through `starting_checkpoint.uid`), so every leg of a run moves, not only the first. The script decodes each manifest with the new code, which computes the whole chain from the embedded definitions, so it does not need to process legs in order; it does need every leg of a run to be migrated in the same pass.

### 2.3 `commit` becomes `declaring_commit`, recorded at the root only

Paths do not change; manifest bytes do.

- **The rename.** The field is renamed `declaring_commit` in `Artifact`, in `RECORDED`, and in every manifest on the volume (the migration renames the key). The name says what the value is: the commit of the process that declared the manifest. That is not necessarily the code that runs the job, since execution uses whatever is deployed. Recording the executed commit is deferred to a later spec.
- **Stamped when a manifest is written.** `lab.declare` is the only writer of manifests (besides this one-off migration, which keeps every stored value). It stamps `declaring_commit = _head()`, the commit of the process declaring now, on every manifest it creates. On a fresh object the field defaults to `None`: a request has no declaring commit until it is declared. So every `manifest.json` on disk records the commit that wrote it, whatever the object in the notebook held. A resources update rewrites an existing manifest but keeps its stored `declaring_commit`: once written, a definition and its declaring commit are immutable.
- **Where `_head()` runs.** In the process that runs `lab.declare`. From a laptop, `local.declare_on_volume` runs it in an ephemeral lab container whose image bakes in the laptop's `get_git_commit()` (`main.py`, `LAB_COMMIT_ENV`), so the stamp is the laptop's commit, the same value the request object carries today.
- **Root only.** Today every embedded dependency copy carries the `commit` and `allocated_resources` of the request object at declaration time. Because `_head()` is cached per process, every artifact built in one notebook session carries the same commit, so an embedded commit is usually the parent's own commit repeated, not the commit the dependency was declared at: a tokenizer declared at commit A, then a dataset on it at commit B, embeds the tokenizer at B. The copies look like a record of what each dependency was built at, mostly are not one, and duplicate what each dependency's own manifest already holds. After: only the root carries the recorded fields, and an embedded copy decodes with `declaring_commit=None` and `allocated_resources=None`, where `None` means "recorded at its own path". Every reader that needs the recorded fields reads the artifact's own manifest: `lab._inspect` loads each node from its path, the launcher reads the manifest it is about to run, and `bind` loads the stored artifact.
- **Drift.** For every node in the requested graph, `lab._inspect` compares the `declaring_commit` at the root of the manifest at that node's own path with `_head()`, not with the request object's field (which is `None` for anything fresh). Declaring a new artifact on top of dependencies declared at an older commit is therefore drift, reported on each of those dependencies' rows (stored A, declaring now B), and `strict_commit=True` refuses to commit over it. Each declaring commit exists once, at its own path, and history is read by walking the graph's own-path manifests, never from copies.

**Behavior change:** today a dependency obtained through `Artifact.load` carries its stored commit and never shows drift. Under this rule it does, when it was declared at an older commit than the one declaring now.

## 3. The manifest format

A root manifest has the five top-level keys in fixed order: `artifact`, `declaring_commit`, `allocated_resources`, `parameters`, `dependencies`. An embedded copy has three: `artifact`, `parameters`, `dependencies`. An embedded copy is then exactly a definition encoding (pydantic §2), so two `==` dependencies embed as identical bytes in any parent. The canonical rules are unchanged (pydantic §6.1, §6.5): nested maps sorted by key, `-0.0` written as `0.0`, NaN refused, `indent=2`, `ensure_ascii=False`, a trailing newline.

```python
def to_manifest(artifact):
    return {
        "artifact": dotted(type(artifact)),
        "declaring_commit": artifact.declaring_commit,
        "allocated_resources": canonical(artifact.allocated_resources.model_dump(mode="json")),
        **_definition_manifest(artifact),
    }

def _definition_manifest(artifact):  # what an embedded copy holds
    return {
        "parameters": parameters(artifact),
        "dependencies": {name: <each member's _definition_manifest, in members() order> for each dependency field},
    }
```

**Decoding: an absent recorded key decodes as `None`.** A root always has both keys, since `lab.declare` writes them; an embedded copy never does. So one rule covers both, and the decoder needs no flag saying whether it is decoding a root or a copy: `data.get("declaring_commit")` and `data.get("allocated_resources")` are passed explicitly, so the field defaults (`None`, `Resources()`) never stand in for a value the manifest did not record.

`memo` keeps working unchanged: an embedded copy and the same artifact's own manifest are different JSON (the root has recorded keys), so they decode to different instances, the embedded one with `None` and the own-path one with its stored values.

**No format version field.** The migration converts every manifest in one pass, so old and new manifests never coexist, and no reader has to handle both.

## 4. Declaration

A new manifest is written as:

```python
manifest = node.model_copy(update={"declaring_commit": _head()}).to_manifest()
```

`model_copy` is allowed here because `declaring_commit` is a recorded field (pydantic §8). A resources update keeps the stored `declaring_commit`. A row's drift compares the stored root `declaring_commit` at the node's own path with `_head()`. The row's `"commit"` key is renamed `"declaring_commit"` and keeps the pair `[stored, declaring now]` for the report.

**Gotcha: a requested node can carry `allocated_resources=None`.** A dependency inside an artifact obtained with `Artifact.load` was decoded from an embedded copy, so its resources are `None`. The commit loop would treat `None != stored` as a resources change and write `None` over the stored request. **Rule: `None` in a request means "no resources requested": the row reports the stored value, nothing is compared, and nothing is written for it.** A fresh object always requests (its default is `Resources()`), so a redeclaration's resources still replace the stored ones for everything built in a notebook. A node that is new at its path and carries `None` (possible only if a dependency's folder was deleted) is written with `Resources()`, the platform defaults. When one path is reached twice in a graph, once fresh and once loaded, `resolve` keeps the first object, so which request applies follows the deterministic traversal order.

## 5. Code changes

| Change | Files |
|---|---|
| `device` out of `Model` | `artifacts/core/SGD/training.py` (`Model`), `models/transformer`, `models/transformerV2`, `models/mamba` (`build(device)`), `artifacts/core/SGD/job.py` (picks the device), `artifacts/stages/pretraining/jobs.py`, `gpus/monitor.py` (takes the job's device), `profiling/util.py` |
| `lineage_hash` over encoded parameters | `artifacts/stages/pretraining/__init__.py` |
| `declaring_commit`, root only | `artifacts/core/artifact.py` (field, `RECORDED`), `artifacts/core/manifest.py` (`to_manifest`, `from_manifest`), `lab.py` (stamping, drift against `_head()`, the `None` resources rule, the row key), the comment in `web/artifactview.js` (the dashboard lists a manifest's top-level keys generically, so its label follows the key) |
| The migration | `migrations/identity.py` (§8) |

## 6. Tests

- **Contract:** every manifest `declare` writes records `declaring_commit` equal to the declaring process's commit, at the root only, and no embedded copy carries `declaring_commit` or `allocated_resources`. A resources update keeps the stored `declaring_commit`. Declaring a new artifact over a dependency declared at an older commit reports drift on that dependency's row, read from its own manifest, and `strict_commit` refuses the commit. A requested node carrying `allocated_resources=None` writes nothing over the stored resources.
- **Contract:** `migrations/identity.py` plans without writing by default, refuses collapsing paths and occupied targets before any move, and plans nothing on a second run. Its input is a temporary root built from the pydantic spec's golden manifests, which are exactly the old format: they move to `test/fixtures/old_format/` before the goldens are regenerated.
- **Changed:** the goldens are regenerated from the new code, in the same commit, and pin the new paths from then on. `test/test_routes.py`'s manifest fixture has a `"commit"` key; `test_resolve.py` and `test_declare.py` pass `commit=`; test models pass `device="cpu"`. All follow the new names.

## 7. Docs

In the same commit: `docs/manifests.md` (the format in §3), `docs/lab.md` (drift and the declaring commit), `docs/training.md` and `docs/models.md` (`build(device)` and the job choosing the device), and the round-trip and identity text in `docs/artifacts.md`. This spec is then folded into those pages in full and deleted.

## 8. The migration script

`migrations/identity.py` is the second of the "one script per change", after `migrations/cpu_floats.py`. Like the scan, its logic imports only `artifacts` and `storage`, so it is tested on a temporary folder, and `local.py` runs it on the volume in an ephemeral Modal app built from the checkout (`modal run local.py::migrate_identity`, with `--apply` to write), so it runs the new code without deploying it.

1. **Preconditions, checked before anything else.** No call is running: for every entry in the `leases` Dict, the liveness that `launcher.state` computes for the dashboard says the call is neither running nor starting. Leases are not empty in normal operation (a finished call's grant stays until the path is granted again or cancelled), so the check is on liveness, not presence. A job writing into a folder while it moves would publish into the old path. The volume is reloaded first.
2. **Plan, without writing.** For every `manifest.json` under the root: read it with `storage.read_manifest` (raw JSON, not decoded, since the old format no longer decodes) and rewrite it into the new format. Drop `device` from every `model` parameter; rename the root's `commit` key to `declaring_commit`, keeping its stored value (the script never restamps: the value says who declared the manifest, and that did not change); and drop `commit` and `allocated_resources` from every embedded copy. Decode the result with the new code, so every validator runs, and compute its new path. The old path is the folder it was read from.
3. **Refusals, before any move.** Stop and print the reasons if any manifest fails to decode, if two old folders map to one new path (the CPU and CUDA twins of §2.1), or if a new path is already occupied by a folder that is not itself moving. Nothing is written in this case.
4. **Dry run by default.** Without `--apply`, print the plan (old path, new path) and stop. Also write it to `migrations/identity.plan.json` on the volume, so the mapping from old to new paths is kept.
5. **Apply, with `--apply`.**
   - Move each folder whose path changed with one `rename` of the whole directory. A leg's folder holds everything that belongs to it, so `model.pt`, `checkpoints/`, the step log and `logs/<call_id>.jsonl` move with it. Only `runs/<run_id>/pretraining/...` folders move; the other families' paths do not change.
   - Write the new manifest at every manifest's new location with `storage.write_manifest`, from the decoded artifact's `to_manifest()` with its stored `declaring_commit` and `allocated_resources` kept. Every manifest has new bytes, moved or not, since its root key is renamed.
   - Rename the notebook copies `runs/<run_id>/declare-<uid>.ipynb` to the new `uid`.
   - Rewrite the keys of `call_history.json` at the root and of the `call_history` and `leases` Modal Dicts from old paths to new, so the dashboard keeps each leg's call history and last grant. `beats` and `call_logs` are keyed by call id and do not change.
   - Commit the volume.
6. **Verify.** Run `modal run local.py::scan_volume` from the same checkout: every manifest loads with the new code, its decoded path is its folder, and it re-encodes to the bytes on disk. Running the migration a second time must plan nothing.

## 9. Rollout

The new code cannot read an unmigrated manifest (it has `commit`, not `declaring_commit`, and a leg's decoded path is not its folder), and the old code cannot read a migrated one, so the order matters:

1. Wait until no call is running, and launch nothing until step 5.
2. From the checkout of this change, run the dry run (`modal run local.py::migrate_identity`). The deployed app is untouched, so a refusal costs nothing: fix the cause (for example, choose which of two colliding legs to keep) and run it again.
3. Run it with `--apply`.
4. Run the scan from the same checkout; it must report nothing.
5. `modal deploy`, so the dashboard, launcher and workers run the new code. Between steps 3 and 5 the old deployed dashboard shows migrated manifests as unreadable; nothing is lost, and step 5 ends it.

I write the script; you run each step and review its output before the next.

**Done when:** the test suite and `mkdocs build --strict` pass, the migration has been applied, the scan on the migrated volume reports nothing, and the new code is deployed.

## 10. Decisions

| Question | Decision |
|---|---|
| Identity and paths | `device` leaves `Model`; `lineage_hash` digests the encoded `"model"` and `"training_parameters"` (§2.1, §2.2). |
| The commit a manifest records | `declaring_commit`: stamped by `lab.declare` with the declaring process's commit on every manifest it writes, kept on resource updates, renamed from `commit` by the migration. The executed commit is not recorded yet (deferred). |
| Recorded fields on embedded copies | None: the root only (§2.3, §3). |
| Drift | Each node's own-path root `declaring_commit` against the commit declaring now; a new artifact over older dependencies is drift on their rows (§2.3). |
| A request with `allocated_resources=None` | No resources requested: nothing compared or written (§4). |
| Existing runs on the volume | Moved and rewritten once, by `migrations/identity.py`, run from the checkout before deploying (§8, §9). |
| Format version | None: the migration converts everything in one pass (§3). |
| Merge the two `Transformer` classes | No: distinct architectures, distinct class paths, distinct identities (§2.2). |

## 11. API contract deltas

Everything not listed keeps the pydantic spec's contract (pydantic Part II).

```python
RECORDED = ("declaring_commit", "allocated_resources")


class Artifact(BaseModel):
    model_config = STRICT

    # declaring_commit is None until lab.declare stamps it on write; both are
    # None on an embedded copy, which records them at its own path instead.
    declaring_commit: str | None = Field(default=None, repr=False)
    allocated_resources: Resources | None = Field(default_factory=Resources, repr=False)


class Model(ParamConfig):
    vocab_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    dtype: str = "torch.float32"

    def build(self, device: str):
        """Returns a freshly initialized torch module on `device`."""
        raise NotImplementedError


def from_manifest(data: dict, memo: dict[str, Artifact] | None = None) -> Artifact:
    """Returns the artifact a manifest describes and the tree under it, unbound.

    Embedded dependencies come back with declaring_commit and
    allocated_resources None: those are recorded at each dependency's own path.
    """
```
