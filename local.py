"""Run against the volume from a laptop: declaration, and the manifest scan.

    from local import declare_on_volume
    report = declare_on_volume(pretraining)               # preview
    report = declare_on_volume(pretraining, commit=True)  # publish

    modal run local.py::scan_volume                       # every manifest, reread
    modal run local.py::canonicalize_manifests [--apply]  # int cpus written as floats

The volume's mount exists only inside a container, so each of these is an
ephemeral Modal app that mounts the volume, runs this checkout's code there,
and hands the result back. `declare_on_volume` prints and returns as
`lab.declare` does, and raises the same DeclarationError on a refused commit.
"""

import modal

import main
from artifacts.core.artifact import Artifact
from lab import DeclarationReport, current_notebook

app = modal.App("launchpad-local")


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
    """`lab.declare(artifact, ...)` against the volume, from anywhere.

    The notebook copy a committed run declaration writes is read here, where
    the kernel is, and shipped along: the container has no notebook of its own.
    """
    with app.run():
        report = _declare.remote(
            artifact,
            commit=commit,
            strict_commit=strict_commit,
            verbose=verbose,
            notebook=current_notebook() if commit else None,
        )
    print(report.render(verbose))
    return report


@app.function(
    image=main.worker_image.add_local_python_source("migrations"),
    volumes={main.STORAGE: main.volume},
    serialized=True,
)
def _scan() -> tuple[int, list[str]]:
    from config import STORAGE
    from migrations.scan import scan

    return scan(STORAGE)


@app.local_entrypoint()
def scan_volume() -> None:
    """Prints every manifest on the volume that this checkout cannot load or
    would rewrite differently, and exits with status 1 if there is any.

        $ modal run local.py::scan_volume
        42 manifests scanned, 0 problems
    """
    total, problems = _scan.remote()
    for line in problems:
        print(line)
    print(f"{total} manifests scanned, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


@app.function(
    image=main.worker_image.add_local_python_source("migrations"),
    volumes={main.STORAGE: main.volume},
    serialized=True,
)
def _canonicalize(apply: bool) -> list[str]:
    from config import STORAGE
    from launcher.state import volume
    from migrations.cpu_floats import plan

    planned = plan(STORAGE)
    if apply:
        for path, text in planned.items():
            path.write_text(text)
        volume.commit()
    return [path.parent.relative_to(STORAGE).as_posix() for path in planned]


@app.local_entrypoint()
def canonicalize_manifests(apply: bool = False) -> None:
    """Prints the manifests `migrations.cpu_floats` would rewrite, and with
    `--apply` rewrites them; refuses everything if any manifest differs in
    more than an int cpu.

        $ modal run local.py::canonicalize_manifests
        tokenizers/bpe-800-f90570d566
        32 manifests to rewrite (dry run; --apply to write)
    """
    folders = _canonicalize.remote(apply)
    for folder in folders:
        print(folder)
    if apply:
        print(f"{len(folders)} manifests rewritten; now run modal run local.py::scan_volume")
    else:
        print(f"{len(folders)} manifests to rewrite (dry run; --apply to write)")
