// artifactview.js — the drill-down page for one artifact: what it is, what it
// was built from, and what every call that ever worked on it said.
//
// No state and no network of its own: app.js fetches `/manifest/<path>` and
// `/logs/artifact/<path>` in parallel and calls renderArtifactView(container,
// manifest, logs). The worker files its own output in the `call_logs` Dict
// and in a file beside the artifact (docs/LOGGING.md); `logs` here is
// `/logs/artifact/<path>`'s response, straight from that Dict -- the one
// container serving this dashboard reads no log file at all, because
// reading files on a mount it also reloads is how a reload loses a race
// with a walk (docs/LOGGING.md).

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

// Type, own parameters, and one link per direct dependency. A dependency is
// just its artifact_path -- main.py's computed state holds no nested
// manifests, so drilling further is a click to that artifact's own page
// (where its own type and parameters live) rather than the whole tree being
// dumped on this one.
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
            el("a", { href: "#/artifact/" + dep, text: dep }),
          ]),
        )
      : null,
  );
}

// Call ids are long and only their tail varies day to day -- same
// truncation render.js's callId() uses for the table, so a call looks the
// same wherever it shows up. Full id still reaches the page, via title.
function shortCallId(id) {
  return id.length > 8 ? id.slice(-8) : id;
}

// One call's own output: a labeled header (short id, full id on hover, last
// heartbeat) and its lines below, oldest first, verbatim -- these already
// carry their own `<iso-timestamp> <LEVEL> <message>` prefix
// (docs/LOGGING.md), so nothing here re-parses or re-sorts a line.
function callLogSection(call) {
  const when = call.last_heartbeat
    ? new Date(call.last_heartbeat * 1000).toLocaleString()
    : null;
  return el("div", { class: "call-log" },
    el("div", { class: "call-log-header" },
      el("span", { class: "call-log-id", title: call.call_id, text: shortCallId(call.call_id) }),
      when ? el("span", { class: "call-log-when", text: "last heartbeat " + when }) : null,
    ),
    call.lines.length
      ? el("pre", { class: "call-log-lines", text: call.lines.join("\n") })
      : el("p", { class: "empty call-log-empty", text: "no lines recorded for this call." }),
  );
}

// Every call that has ever worked on this artifact, oldest first (see
// main.py's artifact_call_logs) -- one section per call, so a superseded
// attempt's own output stays legible next to whatever replaced it instead
// of being interleaved into one ambiguous stream.
function logs(logsPayload) {
  const calls = (logsPayload && logsPayload.calls) || [];
  return el("div", { class: "artifact-logs" },
    el("div", { class: "summary-type", text: "logs" }),
    calls.length
      ? el("div", {}, ...calls.map(callLogSection))
      : el("p", { class: "empty", text: "no call has ever worked on this artifact yet." }),
  );
}

export function renderArtifactView(container, manifest, logsPayload) {
  const nodes = [el("p", { class: "view-back" }, el("a", { href: "#/", text: "← dashboard" }))];

  // `error` covers both "no manifest yet" (declared, not built -- a normal
  // state) and a manifest that wouldn't load, which read the same to a reader
  // of this page: there is nothing to show and the reason is the message.
  if (!manifest || manifest.error) {
    nodes.push(el("p", { class: "empty", text: (manifest && manifest.error) || "loading…" }));
  } else {
    nodes.push(summary(manifest));
  }

  // Logs render independently of whether the manifest loaded -- a call can
  // have left output behind even for an artifact that's only declared, not
  // built (a job that ran and failed before writing anything), so an empty
  // manifest is never a reason to hide them.
  nodes.push(logs(logsPayload));

  container.replaceChildren(...nodes);
}
