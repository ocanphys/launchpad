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
Artifact(parameters)

  parameters        identifying; may contain other artifacts
  metadata          descriptive, never identifying
  refs()            the artifact-valued parameters, one level deep
  relpath()         readable(parameters) / digest(refs())
  path(root)        root / relpath()
  check_status()    the file is there
  identity          (root_kind, relpath())
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
    manifest {
        artifact
        job      { type }
        outputs  [ artifact, ... ]
        inputs   [ resolve(a, stack + [artifact]) for a in job.inputs ]
    }
```

```
job_list(manifest):
    walk inputs depth first
    collect jobs, deduplicated by output set
    sort topologically
```

Requesting an artifact yields a manifest: the artifact, the job that produces
it, that job's full output set, and a manifest for each input, recursively to
leaf jobs.

Resolution is a pure function of the request. It does not touch the filesystem,
prunes nothing that already exists, and cuts no branch short. The same request
resolves identically on a machine where nothing has ever been produced.

Because ancestry is carried in parameters, the shape of the graph is already
latent in the request; the registry supplies the implementation at each node
rather than the structure between them.

A manifest and a live artifact are interchangeable as a starting point — one is
a tree to be built, the other a tree already built.

Flattening a manifest into an ordered job list is separate: walk it, collect
jobs deduplicated by their output set, sort topologically. An artifact
reappearing on the recursion stack is a cycle and an error.

## 7. Scheduling

```
Scheduler(data_root, runs_root, execution)

  done       every output reports produced
  runnable   not done, every input reports produced
  blocked    not done, some input does not

  scheduled  not done, or anything earlier in the order is scheduled
```

The only stage that looks at the world. It holds the roots, so it is the only
thing that can turn an artifact into a path; it holds the execution parameters,
so it is the only thing that can start work.

A job is done when all of its outputs report produced, runnable when its inputs
do, blocked otherwise. It is scheduled if it is not done, or if anything before
it in the order is scheduled — reproducing an upstream artifact must rerun what
depends on it, even where those downstream outputs still exist.

A plan whose leaves are all present schedules nothing. That is not a special
case; it is what asking a fully resolved graph about the world returns.

---

## Limitations

These follow from the design rather than from anything left undone. Some are
prices worth paying; all are worth knowing before they are discovered.

**Existence is not correctness.** `check_status()` asks whether a file is there,
not whether it is right. A truncated write, an interrupted job, or a file
produced by an earlier version of the code all report produced. Until failure
and partial output are handled, a job that dies midway leaves the plan claiming
work that did not happen.

**A change at a leaf renames everything below it.** Because ancestry is carried
in parameters and parameters determine the path, retokenizing with a different
vocabulary gives every downstream checkpoint a new path and marks it not
produced. This is correct — that is what state-complete means — but there is no
escape hatch for a change that did not really matter, and no way to say so.

**Paths are one-way.** The digest half of the encoding cannot be read back into
the parameters that made it. A directory listing shows what kind of thing is
where, not what it was made from. Recovering that requires the manifest, which
means a plan is not reconstructible from the tree alone.

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

## Extensions

Roughly in order of how soon each is likely to be wanted.

**Verification alongside existence.** A sidecar recording size, digest and the
manifest that produced it turns `check_status()` from "a file is there" into "the
right file is there", and makes partial output detectable. It is also where a
produced-by-which-code-version check would live.

**Atomic production.** Writing to a temporary location and moving into place on
success makes existence mean completion, and removes most of the need for the
previous item.

**Parameter dispatch.** Choosing the producer on the artifact's parameters as
well as its type, so one type can arrive by several routes. The uniqueness check
becomes mutual exclusion of predicates, which is not decidable in general and
would have to be approximated or checked at plan time.

**Leasing.** A claim on an artifact for the duration of a job, so concurrent
plans touching one graph do not duplicate or corrupt work. Necessary before more
than one scheduler runs at a time.

**Provenance records.** Writing the manifest and the execution parameters beside
the artifact after a run. Cheap, and it answers the questions the one-way path
encoding cannot.

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
- A shared artifact never depends on a run-scoped one.
- A run-scoped job's run-scoped inputs share one run identifier.
- Distinct parameters render to distinct paths.

## Deferred

- the readable half of the path encoding, per artifact type
- dispatching on parameters, and more than one producer per artifact
- leasing, so two jobs cannot work on one artifact concurrently
- parameter validation
- failure, partial output, retry