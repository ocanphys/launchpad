"""Declare on the volume from a laptop: `lab.declare`'s form, run remotely.

    from local import declare_on_volume
    report = declare_on_volume(pretraining)               # preview
    report = declare_on_volume(pretraining, commit=True)  # publish

The volume's mount exists only inside a container, so this is the one
explicit remote call: an ephemeral Modal app that mounts the volume, runs
`lab.declare` there, and hands the report back. Same printing, same return,
same DeclarationError on a refused commit.
"""

import modal

import main
from artifacts.core.artifact import Artifact
from lab import DeclarationReport

app = modal.App("launchpad-declare")


# serialized: shipped by value, so the container never imports this module
# and with it `main`, which the worker image does not carry.
@app.function(
    image=main.worker_image.add_local_python_source("lab"),
    volumes={main.STORAGE: main.volume},
    serialized=True,
)
def _declare(artifact: Artifact, **options) -> DeclarationReport:
    import lab

    return lab.declare(artifact, **options)


def declare_on_volume(
    artifact: Artifact,
    *,
    commit: bool = False,
    strict_commit: bool = False,
    verbose: bool = False,
) -> DeclarationReport:
    """`lab.declare(artifact, ...)` against the volume, from anywhere."""
    with app.run():
        report = _declare.remote(
            artifact, commit=commit, strict_commit=strict_commit, verbose=verbose
        )
    print(report.render(verbose))
    return report
