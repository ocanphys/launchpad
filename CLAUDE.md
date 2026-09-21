# Working in this repo

NEVER USE EM DASHES ANYWHERE.

Artifacts are folders on a Modal Volume, each owning a `manifest.json` that names what it
is and what it's built from. The model — what an artifact and a job are, how
declaration/resolution/launching fit together — is [artifacts/core/spec.md](artifacts/core/spec.md).
The map of where things live and how traffic flows is [README.md](README.md).

Design rationale goes in `spec.md`. This file is how to write code here.

Subtleties learned the hard way go in [LESSONS.md](LESSONS.md): what is true,
how it was found, what to do about it. Read it before touching checkpoints,
resume, manifests or dataclass inheritance; add to it whenever a bug or a review
turns up something the code cannot say for itself.

## Abstraction

**A helper needs two call sites, or it isn't a helper.** One caller means inline it. The
exception is a name that replaces something genuinely opaque — a regex, a bit-twiddle — not
a name that restates the three lines below it.

**Never split one computation into functions that must be called in order.** If `f`'s
docstring says "call `g` first," they are one function. That ordering constraint is a cost
you paid *for* splitting them; it did not exist before.

**No function that takes a dict and returns the same dict, mutated.** Build the value once,
in one place.

**Don't add a parameter until a caller passes the non-default.** A flag threaded through four
signatures to answer a question that has one right answer makes four signatures worse and
nothing better.

**One function, one return type.** Don't return a value on one branch and `None` on another —
every caller then carries the branch too.

**"Might be general later" is not a reason.** It reliably produces a helper with one caller
today and one caller forever.

**A complicated part gets its own module, behind a seam you can name.** A codec, a graph
walk, a lease protocol: anything that is a whole problem on its own lives in one file, and
the rest of the code reaches it through a few names (`to_manifest`, `from_manifest`,
`dependencies`) and nothing else. The test is whether you could rewrite that file, or
replace it with a library, without touching a caller. If callers reach past the seam, it
is not a seam. This is not the same as extracting helpers: a helper is a name for a
few lines, a seam is a boundary around a problem you may want to solve differently later.

## Docstrings and comments

**First line says what it returns, present tense.** Then at most one paragraph, for the single
thing a caller can get wrong. Stop there.

**Never describe what the code used to be.** No "used to," "no longer," "instead of the old."
A reader cannot distinguish a description of the present from a description of the past, so
history in a docstring makes the whole docstring untrustworthy. Git has the history.

**Design rationale belongs in `spec.md`, not the docstring.** "Why this and not the other
design" is a document. A docstring that argues it makes the reader read an argument to find
an API.

**Never write a comment that justifies a defect.** If you find yourself explaining *why* the
duplication is acceptable, that paragraph is the signal to remove the duplication. A
justification reads as settled and stops anyone re-examining it.

**Docstrings are call sites.** Renaming an attribute means the docstrings that use it are now
wrong, not just stale.

**Removing a mechanism means grepping its name and clearing every mention in the same commit**
— docstrings, comments, README, spec.

## Deleting

**Supersede by deleting, never by renaming to `*_old`.** The copy is dead the day you make it
and reads as live for months.

**Something is dead the moment nothing imports it.** Don't wait until you're sure.

## Finishing and failing

**A body is finished or it raises.** A stub with undefined names looks complete and fails at
runtime far from where you stopped. `raise NotImplementedError` is the honest placeholder.

**A caught exception is either handled or logged.** `except Exception:` that returns an error
string while a logger sits open in the same container is neither.

**No I/O or network at import time.** A module-scope `Volume.from_name(...)` charges every
importer, including the ones that never touch it.

**If two modules `getattr(x, "field", None)` for the same field, declare the field.**
Twice-guessed is a real property.

** whenever you finish editing a file - do one more pass and ask, is there anything overcomplicating the solution? Aim for the most compact, clean and readable code without too many moving parts and unnecessary ceremony. 


## The instinct underneath

Nearly all of the above is one impulse: doing extra work to be safe. Keeping the old file
safe. Extracting a helper for symmetry. Adding a flag so the caller has the option. Writing
the reasoning down so it isn't lost. Each is defensible alone; together they double the size
of a codebase without adding a capability.

When in doubt, leave less behind, not more.
