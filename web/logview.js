// logview.js — the log viewer: one merged timeline of call_id-attributed log
// entries, scoped either to a single artifact or to everything declared
// under a run_id (see main.py's artifact_log_entries / run_log_entries).
//
// Same shape as render.js: a pure function of a payload -> DOM. No network,
// no app state — app.js fetches and calls renderLogView(container, payload,
// opts), which replaces the container's children with what comes back.
//
//   opts = { scope: "run" | "artifact", id: string }   // id is the run_id or artifact_path

import { el } from "./el.js";

const LEVEL_CLASS = { WARNING: "warn", ERROR: "err", CRITICAL: "err" };

// Call ids are long and only their tail varies day to day -- same
// abbreviation render.js uses for the dashboard's own call column.
function shortId(id) {
  return id.length > 8 ? id.slice(-8) : id;
}

// "2026-08-28T12:00:00.000Z" -> "12:00:00.000" -- the date repeats across
// every row in one view, so dropping it keeps the column dense; the full
// stamp is still one hover away.
function timeCell(entry) {
  const stamp = entry.ts;
  const t = stamp.includes("T") ? stamp.slice(stamp.indexOf("T") + 1, -1) : stamp;
  return el("span", { title: stamp, text: t });
}

function callBadge(entry) {
  return el("span", { class: "call-badge", title: entry.call_id, text: shortId(entry.call_id) });
}

// One log line. The artifact column only exists in a run-scoped view --
// that's the whole point of tagging entries with artifact_path: attribution
// only needs showing where the scope could have mixed more than one source.
function logRow(entry, opts) {
  const cls = "log-row" + (LEVEL_CLASS[entry.level] ? " " + LEVEL_CLASS[entry.level] : "");
  return el("tr", { class: cls },
    el("td", { class: "log-ts" }, timeCell(entry)),
    el("td", { class: "log-call" }, callBadge(entry)),
    opts.scope === "run" ? el("td", { class: "log-artifact", text: entry.artifact_path }) : null,
    el("td", { class: "log-msg", text: entry.message }),
  );
}

// Distinct artifacts a run-scoped view pulled entries from, first-seen order
// -- what the "artifacts:" strip above the table links out to.
function contributingArtifacts(entries) {
  const seen = [];
  for (const entry of entries) {
    if (!seen.includes(entry.artifact_path)) seen.push(entry.artifact_path);
  }
  return seen;
}

// A run-scoped artifact's path is always "runs/{run_id}/...", the one shape
// worth linking back from -- a shared artifact (tokenizers/, sources/) has
// no single run to point at, so it gets no link.
function ownerRunId(artifactPath) {
  const parts = artifactPath.split("/");
  return parts[0] === "runs" && parts[1] ? parts[1] : null;
}

export function renderLogView(container, payload, opts) {
  const entries = payload.entries || [];
  const nodes = [el("p", { class: "log-back" }, el("a", { href: "#/", text: "← dashboard" }))];

  if (payload.error) {
    nodes.push(el("p", { class: "log-error", text: payload.error }));
  }

  if (opts.scope === "run") {
    const artifacts = contributingArtifacts(entries);
    if (artifacts.length) {
      nodes.push(el("p", { class: "log-artifacts" },
        "from: ",
        ...artifacts.flatMap((path, i) => [
          i > 0 ? ", " : null,
          el("a", { href: "#/artifact/" + path, text: path }),
        ]),
      ));
    }
  } else {
    const runId = ownerRunId(opts.id);
    if (runId) {
      nodes.push(el("p", { class: "log-artifacts" },
        el("a", { href: "#/run/" + runId, text: "↳ full log for run " + runId }),
      ));
    }
  }

  if (!entries.length) {
    if (!payload.error) nodes.push(el("p", { class: "empty", text: "no log entries yet." }));
  } else {
    nodes.push(
      el("table", { class: "log-table" },
        el("tbody", {}, ...entries.map((entry) => logRow(entry, opts))),
      ),
    );
  }

  container.replaceChildren(...nodes);
}
