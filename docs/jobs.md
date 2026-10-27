# Jobs

A job is the code that builds one artifact. Each artifact family
(`artifacts/sources/`, `artifacts/tokenizers/bpe/`, `artifacts/tokenized/`,
`artifacts/dataset/`, `artifacts/stages/pretraining/`, ...) pairs an `Artifact`
subclass, its parameters, where it lives and what files it owns, with exactly
one `Job` subclass that produces it. There are no alternative producers, no
job ids and no per-run config files. A virtual artifact, such as
`MappedDataSet`, has no job at all.

## How a job and its artifact name each other

A `Job` subclass declares what it produces with one class annotation,
`artifact: Tokenizer`. The artifact names its job the other way round, as a
dotted string:

```python
producer: ClassVar[str] = "artifacts.tokenizers.bpe.jobs.TokenizerJob"
```

Only `Artifact.job()` ever imports that string, inside the worker about to run
it, so defining, resolving and inspecting an artifact never load a family's
`jobs.py`, which pulls in torch or numpy. A job is given its artifact and
nothing else. Its inputs are that artifact's direct dependencies
(`artifact.deps()`), derived, never declared a second time, so a job's inputs
cannot drift out of step with what its artifact names. A job with
dependencies binds them in `__init__`, under the names the artifact gives
them, and `run` uses those.

A job's resource request lives on its artifact, as
`allocated_resources=Resources(gpu_type="A100")`, turned into
`Function.with_options(...)` arguments by `resource_options` just before the
launcher spawns it; an artifact that requests none runs on `run_job`'s default
pool. Declaring (writing manifests for a whole dependency tree, [Lab](lab.md))
and launching (granting a lease and spawning the job that fills one in,
[Launcher](launcher.md)) are separate steps.

Jobs run the deployed code: the commit recorded in a manifest does not select
which code runs, so a code change takes effect after `modal deploy`.

## What a job may write

A job creates only its own outputs, inside its artifact's folder; declaration
has already created that folder and written the manifest before `run` is
called.

- **Every completion file is published through `worker.publishing`.** It hands
  the job an open temporary file, `{filename}.{call_id}.tmp` in the
  destination folder; when the job closes it, it confirms the lease and renames
  it into place. A file exists whole or not at all, and only the call holding
  the lease publishes. The temporary name carries the call id, so neither two
  files nor two calls share one. Naming, opening or renaming a temporary path
  in a job is itself the defect.
- **Published outputs are never appended to, replaced or deleted.** Temporary
  files never satisfy completion.
- **Auxiliary files** in locations the type owns (a training leg's
  checkpoints, its step log) are allowed and never count toward completion.
  A checkpoint still goes through `publishing`, since a half-written one would
  poison the resume that reads it; the step log is appended to as the job
  runs.

Publication is per file, so a crash between files may leave the artifact
`partial`. On a `partial` artifact, a job checks that it can reuse what is
there consistently with the definition before publishing what is missing; it
keeps existing final files, and raises if it cannot resume. Deleting outputs is
an explicit operation outside jobs, and invalidates nothing downstream.

## Failure and recovery

- A failed, cancelled or superseded execution leaves completed files untouched,
  and the artifact stays launchable. A failed attempt is never reported as
  success.
- There are no automatic retries: the user relaunches.
- A training leg writes checkpoints at a fixed step interval, and a relaunch
  resumes from the newest one it can; a finished leg is never retrained
  ([Training a Model](training.md#interrupting-and-resuming)).

## Running a job by hand

A job can run outside the launcher, from a cell, with no lease and console
logging: `job.run(root, lab.worker)` ([Lab](lab.md#the-notebook-api)). Declare,
run, bind: the same sequence the launcher follows.

## Reference

::: artifacts.core.job
