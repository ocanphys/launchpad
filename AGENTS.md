# Docs rules

How the code is described in this repo, for any agent working in it. When a rule changes,
change it here, in the same commit as the code that follows it.

Never use em dashes anywhere: code, docstrings, docs, commit messages.

## Where text goes

**`docs/` is the only description of what the code does.** Each subject is described once,
on one page, served with `mkdocs serve` and listed in `mkdocs.yml`'s nav. Any other text
about the same subject (a spec, a README section, a markdown file beside the code) is
folded into its page and deleted, so no two texts can disagree. Exempt, because they
describe no code: [LESSONS.md](LESSONS.md), TODO.md, CLAUDE.md, this file and
`.claude/skills`.

**README.md is a front page:** what the repo is, and how to open the docs.

**Design rationale goes on the subject's docs page, under a "Why" heading.** "Why this and
not the other design" is a document. A docstring that argues it makes the reader read an
argument to find an API.

**A page's reference section is its module's docstrings,** pulled in with `::: module`. The
page explains how the parts fit; it does not restate a signature.

**Code and its docs page change in the same commit.** Removing a mechanism means grepping
its name and clearing every mention in that commit: docstrings, comments, README, docs.

**`mkdocs build --strict` passes before every commit.** A broken link or a missing
reference is an error, not a warning.

## Docstrings

**First line says what it returns, present tense.** Then at most one paragraph, for the
single thing a caller can get wrong. Stop there.

**A user-facing function shows one example with real values, as a code block.** User-facing
is anything a notebook, a job or a shell calls. The example is indented four spaces past
the docstring's text, with a blank line on each side, so mkdocs renders it as code. A
result too long for the line goes on the next line, starting with `->`:

```python
def flatline_age(last_beat: float, now: float) -> float:
    """Returns the seconds since the last heartbeat.

        flatline_age(last_beat=100.0, now=105.0) -> 5.0
    """
    return now - last_beat
```

```python
class Resources(ParamConfig):
    """What a job should be given to run this artifact; None means Modal's
    platform default for that dimension. Never identity.

        Resources(gpu_type="A100", gpu_count=2, cpu=4)
        -> Resources(cpu=4.0, gpu_type='A100', gpu_count=2)
    """
```

The example must be true: run it, and copy the value it returns.

**Never describe what the code used to be.** No "used to," "no longer," "instead of the old."
A reader cannot tell a description of the present from one of the past, so history in a
docstring makes the whole docstring untrustworthy. Git has the history.

**Never write a comment that justifies a defect.** If you find yourself explaining *why* the
duplication is acceptable, that paragraph is the signal to remove the duplication. A
justification reads as settled and stops anyone re-examining it.

**Docstrings are call sites.** Renaming an attribute means the docstrings that use it are
now wrong, not just stale.
