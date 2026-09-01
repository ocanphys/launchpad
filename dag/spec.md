# Architecture — artifacts, jobs, resolution

Companion to the artifact & job spec. That document says what an artifact and a
job *are*. This one says how the pieces are arranged, what each layer is allowed
to know, and where the world is allowed to intrude.

---

## 1. Layers

Four kinds of thing, in order of increasing knowledge:

| | knows |
|---|---|
| **Artifact** | its own parameters, where it lives, whether it exists |
| **Job** | which artifacts it produces, which it consumes, how to run |
| **Registry** | artifact type to producing job type |
| **Resolver / Scheduler** | how a request becomes a plan, and what of that plan runs |

Nothing points upward. An artifact cannot see the registry; a job cannot see the
resolver. Each layer is usable, and testable, without the ones above it.

## 2. Artifact

```
Artifact(parameters, commit)

  parameters        identifying; may contain other artifacts
  commit            the code its job runs; recorded, never identifying
  metadata          descriptive, never identifying
  refs()            the artifact-valued parameters, one level deep
  relpath()         readable(parameters) / digest(refs())
  path(root)        root / relpath()
  check_status()    the file is there
  identity          (root_kind, relpath())

  manifest()        this artifact as JSON, recursively
  from_manifest()   the inverse
```

An artifact is a value: a set of parameters and a location derived from them.

Parameters are the complete definition. They are recursive — a parameter may
itself be an artifact — because state nests. A supervised-finetuning checkpoint
is not described by its own hyperparameters alone but by those together with the
pretraining checkpoint it started from, which is described in turn by its splits,
its tokenized sources, its tokenizer. Declaring the ancestry as parameters is
what makes two checkpoints trained from different bases different artifacts
without anyone enforcing it.

Metadata is descriptive and never identifying. It may record anything — where a
copied run came from, when something was produced, a human-readable name — and
it changes nothing about which artifact this is.

**An artifact is identified by where it lives on disk.** Its parameters are what
its path is computed from; the path is what identity is. Two artifacts are the
same artifact when they would write to the same file. This makes deduplication,
cycle detection and conflict detection all path comparisons, and it makes them
inspectable — the identity of an artifact is something you can look at in a
directory listing.

The path has two halves. A root, which is ambient, and a relative path, which is
a pure function of the parameters. The relative path is composed of a readable
part, so the tree can be navigated by hand, and a digest of the artifact-valued
parameters, so that a full ancestry does not become a full path.

**The relative path must be injective over parameters.** Because identity is
location, two distinct parameter sets that render to one path are not a
confusing directory tree — they are one artifact silently overwriting another.
The digest half of the encoding is therefore not the subclass's business; only
the readable half is.

`check_status()` asks whether the file is there. It is the artifact's only
contact with the world, and it is never consulted during resolution.

An artifact still does not know how it is made. Carrying what it is made *from*
is not the same as carrying a producer.

### The manifest

```
{ artifact:     type name
  commit:       the code its job runs
  parameters:   the fields that aren't artifacts
  dependencies: the fields that are, each a manifest }
```

Because parameters are the complete definition and nest, writing them out is
writing the artifact out. `manifest()` does exactly that and nothing more:
`from_manifest()` rebuilds an equal object, so a `manifest.json` on disk and a
constructor call in a notebook are two spellings of one artifact. **The file is
the source of truth.** Everything else about an artifact — where it lives, what
files it comprises, which job produces it, whether it is done — is derived from
the type and the parameters, so none of it is recorded. A recorded derivation is
a second answer that can disagree with the first.

The parameters/dependencies split is presentational: it separates what was
plugged in from what was plugged in *from*, and reconstruction merges them back
into the one argument set they were.

**Every dict in the tree is key-ordered** — the four above by the order written,
everything below them by sorting. Dict equality ignores key order, so no
comparison in Python can see this drifting, while every byte on disk can:
without it, reordering a dataclass's fields silently rewrites every manifest
mentioning that type. Ordered, the file is a pure function of the artifact, and
so diffable, and so hashable.

Commit is the exception to *parameters are everything*. It is what pins the code
the producing job runs, without which identical parameters do not imply identical
bytes — and it cannot be a parameter, because then every commit would rename
every artifact. So it is recorded per node and excluded from identity: an
artifact rebuilt from a manifest written months ago is the *same* artifact.

Recording it per node rather than per file is what makes drift legible. An
artifact loaded from disk keeps the commit it was produced under; a new artifact
built on top of it is stamped with HEAD; and the resulting tree carries both, so
asking whether the graph was built by one version of the code is a walk over it.
Nothing enforces agreement — a mixed tree is normal, and usually fine.

## 3. Scope

```
run_id in parameters   ->  runs_root / run_id / relpath()
run_id absent          ->  data_root / relpath()
```

```
Tokenizer(corpus, vocab_size)                   shared
TokenizedSource(source, tokenizer)              shared
Split(tokenized, scheme, seed, role)            shared
PretrainCkpt(run_id, split, arch, schedule)     run
SFTCkpt(run_id, base, dataset, schedule)        run
RLCkpt(run_id, base, reward, schedule)          run
```

There are two roots: one for artifacts shared across runs, one for artifacts
belonging to a single run. Which applies is determined by whether the artifact's
parameters include a run identifier — nothing is declared, so nothing can drift.

A run identifier is identifying but not content-determining. It changes where
bytes live, not what they are. That is what buys run isolation, and it means
types whose contents cannot vary between runs — tokenizers, tokenized sources,
splits — must not carry one, or every run redoes work it could have shared.

The membrane is one-way: a run-scoped artifact may depend on a shared one, never
the reverse. A shared artifact whose contents depended on a run would collide
between runs with no error anywhere, so this is checked, not assumed.

Forking a run is a file copy performed outside the system. Copy the state into a
directory under a new run identifier and request an artifact under that
identifier; resolution builds the same complete graph as always, and scheduling
finds the copied artifacts already produced and starts below them. Nothing in
the model needs to know a fork happened. If the lineage is worth keeping, it is
metadata.

## 4. Job

```
Job

  produces          [ ArtifactType, ... ]     declared
  discriminator     parameter name | none     for sibling output sets

  for_output(artifact) -> Job
  outputs           [ Artifact, ... ]         full set, incl. the requested one
  inputs            union of refs() over outputs
  run(execution)

  identity          the set of its outputs
```

A job is where the wiring lives.

It declares the artifact types it produces. Its inputs are derived, not
declared: they are the artifact-valued parameters of its outputs. Because
artifacts describe state completely, this derivation is usually the whole story,
and a job that needs to override it is worth a second look — it is claiming a
dependency its outputs do not record.

A job is completely determined by its outputs. Given one artifact to produce,
everything else follows: the rest of the output set, the inputs, the work.

Most jobs produce one artifact. The honest multi-output case is a set of
siblings that differ only by a discriminating parameter and come from one
indivisible operation — the train, validation and test parts of a single
partitioning. Reconstructing the job from any one of them means dropping the
discriminator and expanding it across its domain, which is symmetric by
construction. An output set that is not of this shape is more likely a modelling
error than a case to generalize for.

Execution parameters configure the run, never the result. They are excluded from
identity and from the plan, and they arrive only when something is scheduled.

## 5. Registry

```
defining a Job subclass:
    for each type in produces:
        register(type, subclass)      collision -> error, here and now

producer_for(artifact) -> Job
    subclass = lookup(type(artifact))
    subclass.for_output(artifact)
```

The mapping from artifact to producing job cannot live on the artifact, so it
lives on the job — and is collected automatically. Defining a job subclass
registers it as the producer of the artifact types it declares. Two subclasses
claiming one artifact type is an error where the second is defined, not where a
plan is built.

Lookup takes an artifact and returns the job that would produce it. This is the
one seam that will move: today it matches on artifact type alone, and the
extension to dispatching on parameters as well changes this function and nothing
on either side of it.

## 6. Resolution

```
resolve(artifact, stack):
    artifact in stack -> cycle, error
    job = producer_for(artifact)
    plan {
        artifact
        job      { type }
        outputs  [ artifact, ... ]
        inputs   [ resolve(a, stack + [artifact]) for a in job.inputs ]
    }
```

```
job_list(plan):
    walk inputs depth first
    collect jobs, deduplicated by output set
    sort topologically
```

Requesting an artifact yields a plan: the artifact, the job that produces it,
that job's full output set, and a plan for each input, recursively to leaf jobs.

A plan is the manifest with the jobs attached, and it is the layer that may be
thrown away. It is never written; what lands in each folder is the artifact's
own manifest, which names no job at all. The job is looked up from the artifact's
type and the commit says which version of it — recording the job would fix in a
file something the registry already decides.

Resolution is a pure function of the request. It does not touch the filesystem,
prunes nothing that already exists, and cuts no branch short. The same request
resolves identically on a machine where nothing has ever been produced.

Because ancestry is carried in parameters, the shape of the graph is already
latent in the request; the registry supplies the implementation at each node
rather than the structure between them.

A manifest on disk and a live artifact are interchangeable as a starting point —
one is a tree written down, the other a tree in memory, and resolution takes
either.

Flattening a plan into an ordered job list is separate: walk it, collect jobs
deduplicated by their output set, sort topologically. An artifact reappearing on
the recursion stack is a cycle and an error.

## 7. Declaration

```
check(artifact, root, strict_commit) -> Declaration

  per artifact in the plan, dependency-ordered:

    new         nothing there; write will declare it
    declared    manifest agrees, no outputs yet
    partial     manifest agrees, some outputs
    done        manifest agrees, every output present
    conflict    a manifest is there and describes something else   BLOCKS
    undeclared  outputs nobody declared                            BLOCKS

  drift       declared under another commit; blocks only if strict_commit

Declaration.write()  -> the `new` manifests, dependencies first
```

Manifests are written **ahead of the work**, for the whole graph, and are
immutable once written. That is the pivot the rest of this section turns on: a
manifest stops being a record of what happened and becomes a declaration of what
is meant to happen, and the disk becomes the interface between planning and
execution. A notebook declares; a launcher discovers.

Because they are immutable, a manifest that disagrees with the one being
requested is never a re-declaration. It means the history recorded there was
edited, and it blocks. Changing parameters is a new run, and `write` only ever
adds — never modifies, never deletes. Correcting a declaration means removing it
by hand, which is deliberate: it is the one operation that discards history.

This also settles the coherence question the duplicated subtrees raise. The same
artifact is written once at its own path and again inside every dependent, and
those copies can disagree only if they were written at different times. Checking
every node in the plan against the manifest at its own path catches exactly that,
because the plan being checked is internally coherent by construction — so no
separate reconciliation pass is needed.

Nothing about it needs a lease. Manifests are canonically serialized, so two
processes declaring the same artifact write identical bytes; `write` creates
exclusively and, on losing the race, compares — equal is someone else declaring
the same thing, different is a conflict. Declaring is therefore local, offline
and idempotent, and only running needs to coordinate.

`check` does not police run scope. Each artifact already carries its own
`run_id` where one is needed, and `Declaration`'s header reads it off the
requested artifact for display — but nothing compares it against the rest of
the plan. A plan that quietly mixes two runs (a dependency built for one run,
reused as another's input) reconciles cleanly against disk like any other; see
Limitations.

**Status is not scheduling.** These six say what is on disk, not what to run
next. Runnability — a job whose dependencies are satisfied — is the launcher's
question, answered by walking the declared manifests under a run, and it is
where leasing and preflight belong.

## 8. Launching

```
declared_artifact(artifact_path) -> (Artifact, status)   -- inside a container, volume mounted

attempt_launch(artifact_path):
    active?                             -> refuse
    declared_artifact(artifact_path)    -> not declared, or blocked -> refuse
    call = run_job.spawn(artifact_path)
    leases.put(artifact_path, grant)

run_job(artifact_path):
    lease.confirm("boot")
    job = producer_for(load(artifact_path))     # registry, looked up only here
    job.run(root)
    lease.confirm("commit")
    volume.commit()
```

A lease is granted per `artifact_path`, not per run — the key is identity
itself, so two artifacts in one run's DAG (a dataset and a sibling
tokenizer) contend for nothing and run concurrently. This is the Extensions
section's "Leasing" entry; it now exists, scoped narrower than first sketched
there (per artifact, not per run).

The job is never looked up before `run_job` needs it. `attempt_launch` only
ever asks the same question `status` already answers — ready or blocked —
which needs the artifact, not its producer. `producer_for` (the registry) is
called exactly once, inside the container that's about to run it.

`declared_artifact` runs the manifest read and status check inside an actual
container, deliberately: `Path(STORAGE)` is only a real mount inside one, and
`volume.reload()` refuses to run anywhere else. `attempt_launch` itself has no
fixed home — called from a web request (already inside a container) or a
local CLI entrypoint (never inside one) — so the one part that needs a mount
is pulled into its own function and called with `.remote()` either way,
rather than assumed to already have one.

A launch races on a read-then-write of the lease, not a compare-and-swap.
Two near-simultaneous launches can both pass the check and both spawn. This
is bounded, not prevented: each call confirms the lease at boot and again
before its final commit, and confirm re-reads the grant fresh each time — so
whichever call's write didn't win the race raises `LeaseLost` and discards
its own output instead of landing it. The cost is wasted compute, not a
corrupted artifact — but it can be paid in full: nothing here re-checks the
lease *during* a run, so a job that loses the race at boot still runs to
completion before its final confirm discovers it lost, however long that
took. Cheap for today's jobs; a real training loop should confirm between
steps, not just at the two ends.

## 9. Reading

```
declared_under(run_id, root) -> [Artifact, ...]
```

The dashboard's own book: walk the manifests actually written under
`runs/{run_id}`, plus every dependency reachable from them (which may live
under a shared root), deduplicated by identity. Not a `resolve()` from one
requested artifact — a run's declared state is however many manifests exist,
not one tree computed top-down.

Per artifact this discovers, status and readiness are the same `inspect`/
`blocked_by` `attempt_launch` uses to decide whether to launch — the
dashboard and the launcher never disagree about what's runnable, because
they ask the same question of the same disk state. Lease and heartbeat are
read per artifact too, from the one `leases`/`beats` snapshot taken for the
whole request: a lease's key is an artifact_path, so that's the granularity
at which "is this active" means anything now.

---

## Limitations

These follow from the design rather than from anything left undone. Some are
prices worth paying; all are worth knowing before they are discovered.

**Existence is not correctness.** Status asks whether a file is there, not
whether it is right. A truncated write or a file produced by an earlier version
of the code both report produced. Declaring an explicit completion marker last,
after the real outputs are durable, narrows this — the marker cannot appear
before the bytes it vouches for — but it does not close it.

**Interruption is no longer visible in the file set.** Writing manifests ahead of
the work means "manifest present, outputs absent" is the normal declared state,
not evidence a job died. Only `partial` — some declared outputs, not all —
suggests interruption, and an artifact declaring a single file cannot even be
partial. Distinguishing running from abandoned needs the lease, which status does
not consult.

**Declaration is one-way.** `write` only adds. A declaration made in error can be
undone only by deleting the file, outside the system and unrecorded. This is the
deliberate price of immutability, but it means the recovery path for the most
likely mistake is the one operation nothing checks.

**Run scope is not checked.** A run-scoped artifact can be handed as a
dependency to a request from a different run — a `toy2` `Pretraining` built on
`toy`'s `DataSet`, say — and `check` will not object: the mismatched artifact
already reconciles against disk on its own path, which is all `check` looks at.
The invariant that a run-scoped job's run-scoped inputs share one run
identifier is therefore a modeling discipline, not something enforced in code.

**A change at a leaf renames everything below it.** Because ancestry is carried
in parameters and parameters determine the path, retokenizing with a different
vocabulary gives every downstream checkpoint a new path and marks it not
produced. This is correct — that is what state-complete means — but there is no
escape hatch for a change that did not really matter, and no way to say so.

**Paths are one-way.** The digest half of the encoding cannot be read back into
the parameters that made it. A directory listing shows what kind of thing is
where, not what it was made from. The manifest in each folder answers that, but
only for folders that have one — a path alone still says nothing.

**Leg decomposition is identifying.** A run is a chain of legs, each continuing
the one named by `base`, and where it is cut is operational — a container does
not live forever — not scientific: with seeds carried deterministically, one leg
to 2000 and two legs through 1000 produce the same weights. But they are
different artifacts at one path, so declaring one and later wanting the other is
a conflict. Re-cutting a run's legs means a new run, even though the science is
unchanged. The alternative — making `base` non-identifying — would remove the
ordering from the manifest, which is worse.

**The recovery checkpoint is invisible, and that is correct.** A leg resumes from
mutable training state that appears in no manifest. It stays out because its
contents are determined by parameters that *are* declared — seed, step, config,
the base chain — so it is a cache, and deleting it costs recomputation rather
than correctness. That reasoning holds only as long as the training loop derives
its RNG stream and data order from the seed. A loop that carried genuinely
unreproducible state would silently make every leg boundary an undeclared input,
and nothing here would detect it.

**Manifests repeat themselves.** The tree is written as a tree, so an artifact
reached by several routes — a tokenizer under each of its tokenized sources —
is written out once per route. It is correct, self-contained and readable, and
it grows with the square of a wide graph. Interning repeated nodes behind a
reference is the escape hatch, at the cost of a file you can no longer read
top to bottom.

**The commit is recorded, not honoured.** Nothing checks a manifest's commit
before running its job; a job always runs as the code currently on disk. The
commit says what produced an artifact, and comparing it to HEAD is available to
anyone who asks — but it is a report, not a gate, and it cannot make an old
manifest run under old code.

**A session stamps one commit.** HEAD is read once per process, so committing
mid-session doesn't change what subsequent artifacts record. Because the read
includes a `-dirty` marker, an artifact built from uncommitted work says so —
which is honest and also means the commit does not always identify code that
exists anywhere but that working tree.

**The encoding is load-bearing forever.** Identity is location, so changing how
paths are rendered orphans everything already produced. The encoding cannot be
improved in place; it can only be versioned, with a migration.

**Injectivity is asserted, not proved.** Nothing checks that two parameter sets
cannot render to one path. A collision is silent overwriting, and the design
gives no point at which it would surface.

**Run isolation duplicates work.** A run-scoped artifact is distinct per run
even when its bytes would be identical, so anything carrying a run identifier is
recomputed for every run that wants it. Which types carry one is a judgement
made per type, and getting it wrong is expensive in one direction and unsafe in
the other.

**A fork is unverified.** Copying state into a new run directory is done outside
the system, so nothing checks that what was copied is what the new run's
parameters describe. A copy from a differently-configured run produces a plan
that looks satisfied and is not.

**Derived inputs assume state-completeness.** Inputs come from the outputs'
artifact-valued parameters. Anything a job reads that is not named there —
a config file, an environment variable, a dataset addressed by mutable path —
is invisible to the graph and unversioned by it.

**Execution parameters are policed by convention.** The rule that they cannot
change the result is stated, not enforced. A parameter that changes numerics is
indistinguishable from one that does not until the outputs differ.

**One producer per artifact.** An artifact type has exactly one job that makes
it. The same type arriving by different routes — an initialized checkpoint
versus a trained one — has no expression.

**Resolution is unbounded.** The graph is built complete every time, with no
pruning and no memoization across requests. This is what makes it deterministic,
and it means the cost of asking about one artifact scales with the whole history
behind it.

**A lease outlives its job.** Nothing pops `leases[artifact_path]` on success
— only staleness (a heartbeat that stops arriving) makes a grant stop reading
as active. Harmless for correctness (a fresh launch overwrites it, and `done`
outranks it in the dashboard's own read), but the Dict grows without bound,
and a raw read of `leases` shows finished work as still "held."

**"Failed" is two different problems wearing one color.** The dashboard folds
"a call held the lease and went stale before finishing" and "the manifest
itself conflicts with what's declared" into the same status. Both mean a
human should look, but not at the same thing.

**A fresh launch can read as a stale one.** A grant is written the instant a
job is spawned, before its first heartbeat lands — and a cold container can
take several seconds to boot. In that window, an artifact with a real,
healthy job starting up looks identical to one whose call already died: a
`call_id` with no heartbeat yet.

**One unreadable top-level manifest blacks out its whole run.** `declared_under`
doesn't guard its own `Artifact.load` calls for the manifests it finds
directly under `runs/{run_id}` — only *shared* dependencies pulled in
transitively get `inspect`'s `_Unreadable`/`conflict` handling. A corrupt
`runs/{id}/dataset/manifest.json` therefore hides every other artifact in
that run too, not just itself.

**Renamed fields don't migrate what's already written.** `leases`/`beats` are
persistent, named, and outlive any one version of this code. An entry written
under an older shape (a different key, a differently-named field) stays in
whatever shape it was written in — silently ignored by lookups that expect
the current shape, but still visible, and confusing, in a raw read of either
Dict.

## Extensions

Roughly in order of how soon each is likely to be wanted.

**Conflict diffs.** `check()` labels a mismatched artifact `conflict` but says
nothing about what disagrees -- the row carries a status, not the two
manifests it compared. Diffing the recorded manifest against the requested one,
field by field, would turn "something's wrong here" into "here's what
changed," without opening both `manifest.json` files by hand.

**Verification alongside existence.** A sidecar recording size and digest turns
`check_status()` from "a file is there" into "the right file is there", and makes
partial output detectable. The manifest is already beside the file; what is
missing is anything about the bytes.

**Commit checking.** A walk comparing each manifest node's commit against HEAD,
and a policy for what to do when they differ — report, refuse, or rebuild. The
data is recorded; only the question is unasked.

**Atomic production.** Writing to a temporary location and moving into place on
success makes existence mean completion, and removes most of the need for the
previous item.

**Parameter dispatch.** Choosing the producer on the artifact's parameters as
well as its type, so one type can arrive by several routes. The uniqueness check
becomes mutual exclusion of predicates, which is not decidable in general and
would have to be approximated or checked at plan time.

**Execution records.** The manifest is already written beside the artifact; the
execution parameters a run was given are not. They are excluded from identity by
design, which is exactly why nothing currently records that they happened.

**A path index.** A record mapping rendered paths back to parameters, which
restores readability of the tree and gives a place where collisions would be
caught rather than silently taken.

**Forking as a graph node.** Making the copy a job whose input is an artifact
under one run identifier and whose output is the same artifact under another.
The fork becomes visible in the plan and checkable, at the cost of a real file
operation and a relaxed invariant.

**Content-addressed runs.** Deriving the run identifier from the parameter tree
rather than choosing it. Forking stops being an operation — configurations that
share a prefix share its artifacts automatically — and the shared/run split
largely dissolves. It costs human-chosen run names, which would have to survive
as metadata and a name index, and it changes identity, so it is a migration
rather than a swap.

**Garbage collection.** Nothing currently removes anything. A rule for what is
reachable from a set of held manifests would give one.

---

## Invariants

- An artifact never references a job, a producer, or a dependency list.
- Resolution never touches the filesystem.
- Roots and execution parameters never appear in a manifest.
- A manifest records only parameters and commit; anything derivable is derived.
- `from_manifest(a.manifest()) == a`, for every artifact.
- Every dict in a manifest is key-ordered, so its bytes depend on nothing but
  the artifact.
- A manifest is never modified once written; declaration only adds.
- An artifact declares only files it is certain to write.
- Declaring needs no lease, and no network. Running needs both: a lease keyed
  by artifact_path, and a container to hold it in.
- `allocated_resources` (what a job should be given to run) is never
  identity, never a parameter, and never defaulted from anything but Modal's
  own platform default -- same reasoning as `commit`, same mechanism.
- A shared artifact never depends on a run-scoped one.
- Distinct parameters render to distinct paths.

## Deferred

- when we create artifacts, save them to a temp path and only when they are finished we actually move them to the path we want.
- the readable half of the path encoding, per artifact type
- dispatching on parameters, and more than one producer per artifact
- parameter validation
- failure, partial output, retry
- `Declaration.check` doesn't verify that an artifact and everything in its
  manifest tree share one run id -- a plan that mixes runs reconciles as clean
  (see Limitations, "Run scope is not checked")
- `Declaration.write` doesn't check for a live lease -- writing a manifest
  while a job is already running against that artifact_path is a race. Guard
  `write`, not just running, against an artifact with a job in flight
- leasing exists now (see section 8) but only confirms at boot and before the
  final commit -- a job that loses the launch race still runs to completion
  before finding out. Real per-step confirmation is what closes this, and is
  worth adding once a job is long enough for it to matter (see Limitations,
  "A lease outlives its job" and "A launch races on a read-then-write" in
  section 8)
- nothing pops a lease on success or garbage-collects `leases`/`beats` --
  see Limitations, "A lease outlives its job"