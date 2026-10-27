"""Run against the volume from a laptop: declaration, and the manifest scan.

    from local import declare_on_volume
    report = declare_on_volume(pretraining)               # preview
    report = declare_on_volume(pretraining, commit=True)  # publish

    modal run local.py::scan_volume                       # every manifest, reread
    modal run local.py::migrate_identity [--apply]        # the identity change, once

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
    volumes={main.STORAGE.as_posix(): main.volume},
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
    volumes={main.STORAGE.as_posix(): main.volume},
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
    volumes={main.STORAGE.as_posix(): main.volume},
    serialized=True,
)
def _migrate_identity(apply: bool) -> dict[str, str]:
    import json
    import time

    from config import STORAGE
    from launcher.state import is_active, is_starting, volume
    from migrations import identity
    from system.lease_protocol import beats, call_history, leases

    volume.reload()
    now = time.time()
    running = [
        path
        for path, grant in leases.items()
        if is_active(grant, beat := beats.get(grant["call_id"]), now) or is_starting(grant, beat, now)
    ]
    if running:
        raise RuntimeError(f"calls are running, so nothing is migrated: {running}")
    planned = identity.plan(STORAGE)
    if apply:
        identity.apply(STORAGE, planned)
        moved = identity.moves(planned)
        for old, new in moved.items():
            for keyed in (call_history, leases):
                if old in keyed:
                    keyed.put(new, keyed.pop(old))
        record = STORAGE / "migrations" / "identity.plan.json"
        record.parent.mkdir(exist_ok=True)
        record.write_text(json.dumps(moved, indent=2) + "\n")
        volume.commit()
    return {folder.as_posix(): artifact.artifact_path.as_posix() for folder, artifact in planned.items()}


@app.local_entrypoint()
def migrate_identity(apply: bool = False) -> None:
    """Prints every manifest `migrations.identity` rewrites, with its old and
    new path when the folder moves, and with `--apply` moves and rewrites them.
    Refuses everything while a call is running, or if any manifest would not
    decode, collide or land on a taken path.

        $ modal run local.py::migrate_identity
        runs/GOLDEN/pretraining/0-1000-639552a92c -> runs/GOLDEN/pretraining/0-1000-69f057c312
        sources/odyssey
        ...
        11 manifests, 4 of them moving (dry run; --apply to write)
    """
    planned = _migrate_identity.remote(apply)
    for old, new in planned.items():
        print(old if old == new else f"{old} -> {new}")
    summary = f"{len(planned)} manifests, {sum(old != new for old, new in planned.items())} of them moving"
    print(f"{summary}: written; now run modal run local.py::scan_volume" if apply else f"{summary} (dry run; --apply to write)")
