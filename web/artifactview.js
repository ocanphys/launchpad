// artifactview.js — the drill-down page for one artifact: what it is, what it
// was built from, and where to read what its jobs said.
//
// No state and no network of its own: app.js fetches `/manifest/<path>` and
// calls renderArtifactView(container, manifest). A job's own output isn't here
// yet. The worker files it in the `call_logs` Dict and in a file beside the
// artifact (docs/LOGGING.md); the one container serving this dashboard
// reads no log file at all, because reading files on a mount it also reloads
// is how a reload loses a race with a walk (docs/QUEUES.md §3.3).

import { el } from "./el.js";

// A parameter's value as one line: primitives print plain, anything richer
// (a list, a nested config) prints as compact JSON -- enough to show what's
// there without a full nested tree view. Dependency values never reach here
// -- manifest_endpoint's own parameters/dependencies split already keeps
// those out of `parameters`.
function paramValue(value) {
  if (value === null || typeof value !== "object") return String(value);
  return JSON.stringify(value);
}

// Type, own parameters, and one link per direct dependency. main.py's
// artifact_manifest_summary already resolved each dependency to its own
// artifact_path -- this links there rather than inlining that dependency's
// manifest, so drilling further is a click away instead of the whole tree
// being dumped on one page.
function summary(manifest) {
  const params = Object.entries(manifest.parameters || {});
  const deps = manifest.depends_on || [];

  return el("div", { class: "artifact-summary" },
    el("div", { class: "summary-type", text: manifest.type }),
    params.length
      ? el("div", { class: "summary-params" },
          ...params.map(([key, value]) =>
            el("div", { class: "summary-param" },
              el("span", { class: "summary-key", text: key }),
              el("span", { class: "summary-value", text: paramValue(value) }),
            ),
          ),
        )
      : null,
    deps.length
      ? el("p", { class: "summary-deps" },
          "depends on: ",
          ...deps.flatMap((dep, i) => [
            i > 0 ? ", " : null,
            el("a", { href: "#/artifact/" + dep.artifact_path, title: dep.type, text: dep.artifact_path }),
          ]),
        )
      : null,
  );
}

export function renderArtifactView(container, manifest) {
  const nodes = [el("p", { class: "view-back" }, el("a", { href: "#/", text: "← dashboard" }))];

  // `error` covers both "no manifest yet" (declared, not built -- a normal
  // state) and a manifest that wouldn't load, which read the same to a reader
  // of this page: there is nothing to show and the reason is the message.
  if (!manifest || manifest.error) {
    nodes.push(el("p", { class: "empty", text: (manifest && manifest.error) || "loading…" }));
  } else {
    nodes.push(summary(manifest));
  }

  container.replaceChildren(...nodes);
}
