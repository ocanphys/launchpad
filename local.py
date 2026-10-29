"""Declaration against the volume from a laptop:

    from local import declare_on_volume
    report = declare_on_volume(pretraining)               # preview
    report = declare_on_volume(pretraining, commit=True)  # publish

The volume's mount exists only inside a container, so this is an ephemeral
Modal app that mounts the volume, runs this checkout's `lab.declare` there,
and hands the report back. It prints and returns as `lab.declare` does, and
raises the same DeclarationError on a refused commit.
"""

import modal

import main
from artifacts.core.artifact import Artifact
from lab import DeclarationReport, _head, current_notebook

app = modal.App("launchpad-local")


# serialized: shipped by value, so the container never imports this module
# and with it `main`, which the worker image does not carry.
@app.function(
    image=main.worker_image.add_local_python_source("lab"),
    volumes={main.STORAGE.as_posix(): main.volume},
    serialized=True,
)
def _declare(artifact: Artifact, declaring_commit: str, **options) -> DeclarationReport:
    import os

    from config import LAB_COMMIT_ENV

    os.environ[LAB_COMMIT_ENV] = declaring_commit  # this container has no git; lab._head() reads it
    import lab

    return lab.declare(artifact, **options)


def declare_on_volume(
    artifact: Artifact,
    *,
    commit: bool = False,
    strict_commit: bool = False,
    verbose: bool = False,
) -> DeclarationReport:
    """`lab.declare(artifact, ...)` against the volume, from anywhere.

    The notebook copy a committed declaration writes is read here, where
    the kernel is, and shipped along, and so is this process's commit, which
    every manifest written is stamped with: the container has neither.
    """
    with app.run():
        report = _declare.remote(
            artifact,
            _head(),
            commit=commit,
            strict_commit=strict_commit,
            verbose=verbose,
            notebook=current_notebook() if commit else None,
        )
    print(report.render(verbose))
    return report
