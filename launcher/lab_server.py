"""Starting JupyterLab in the lab container: its settings, then its process.

Named `lab_server`, not `lab`: `lab.py` at the repo root is the module a
notebook imports.
"""

import json
import subprocess
from pathlib import Path

from config import LAB_PORT, STORAGE


def start_jupyterlab() -> None:
    """Writes the settings overrides, then starts `jupyter lab` on LAB_PORT,
    rooted at the volume.

    `root_dir` is the volume itself, not a notebooks folder -- pointing at an
    artifact path is the whole point, and the file browser is where you find
    out what the paths are (`runs/`, `sources/`, `tokenizers/`).

    No `--IdentityProvider.token`: jupyter_server reads JUPYTER_TOKEN out of
    the environment, which LAB_SECRET puts there, and keeping it out of the
    command line keeps it out of every `ps` and traceback.
    """
    # Default-on autocomplete and editor niceties: ONE overrides.json, in
    # JupyterLab's application settings directory -- not the per-user
    # settings tree (~/.jupyter/lab/user-settings/), which holds one
    # <plugin-id>.jupyterlab-settings file per plugin (what the Settings
    # Editor writes when a person changes something by hand), and isn't
    # where overrides.json is read from. Keyed by full plugin id, verified
    # against JupyterLab's own schemas rather than guessed:
    #   - completer-extension:manager's `autoCompletion` (default false) is
    #     what actually turns "press Tab to see suggestions" into
    #     suggestions appearing as you type.
    #   - codemirror-extension:plugin's `defaultConfig` is an open object of
    #     CodeMirror editor options (autoClosingBrackets, lineNumbers, ...),
    #     applied to every editor -- notebook cells included, so there's no
    #     separate notebook-extension setting needed for these.
    # Deliberately not attempting "open the contextual-help/inspector panel
    # by default" here: inspector-extension's own schema declares zero
    # properties (`additionalProperties: false`, nothing in between) --
    # there is no settings key for it. That needs a pre-built default
    # *workspace* (layout state, a different mechanism from settings
    # overrides entirely), not attempted here.
    from jupyterlab.commands import get_app_dir

    settings_dir = Path(get_app_dir()) / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "overrides.json").write_text(
        json.dumps(
            {
                "@jupyterlab/completer-extension:manager": {"autoCompletion": True},
                "@jupyterlab/codemirror-extension:plugin": {
                    "defaultConfig": {
                        "autoClosingBrackets": True,
                        "lineNumbers": True,
                        "codeFolding": True,
                    }
                },
            },
            indent=2,
        )
    )

    subprocess.Popen(
        [
            "jupyter",
            "lab",
            "--ip=0.0.0.0",
            f"--port={LAB_PORT}",
            "--no-browser",
            "--allow-root",
            f"--ServerApp.root_dir={STORAGE}",
            # Modal terminates TLS and forwards to this container under a
            # different host than the browser typed, which jupyter_server reads
            # as a remote/cross-origin access and blocks by default.
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.allow_origin=*",
        ]
    )
