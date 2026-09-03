// logview.js — the log viewer: one merged timeline of call_id-attributed log
// entries, scoped either to a single artifact or to everything declared
// under a run_id (see main.py's artifact_log_entries / run_log_entries).
//
// Same shape as render.js: a pure function of a payload -> DOM. No network,
// no app state — app.js fetches and calls renderLogView(container, payload,
// opts), which replaces the container's children with what comes back.
// Level-filter selection is UI state, not data, so it lives in app.js too
// (same ctx-injection pattern render.js uses for onLaunch/onToggleGroup)
// and arrives here read-only, via opts.
//
//   opts = {
//     scope: "run" | "artifact",     // id is the run_id or artifact_path
//     id: string,
//     callId: string | null,         // set only for a call_id-filtered artifact view
//     hiddenLevels: Set<string>,     // levels currently unchecked in the filter bar
//     onToggleLevel: (level: string) => void,
//     manifest: object | null,       // scope "artifact" only -- see main.py's
//                                     // manifest_endpoint: {type, parameters,
//                                     // depends_on, error?}
//   }

import { el } from "./el.js";

const LEVEL_CLASS = { DEBUG: "debug", INFO: "info", WARNING: "warn", ERROR: "err", CRITICAL: "err" };
const LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

// Call ids are long and only their tail varies day to day -- same
// abbreviation render.js uses for the dashboard's own call column.
function shortId(id) {
  return id.length > 8 ? id.slice(-8) : id;
}

// entry.ts is UTC ("2026-08-28T12:00:00.000Z"); every row shows the
// viewer's own local time instead, ISO-shaped but without milliseconds --
// "2026-08-28T05:00:00" -- since a stamp dense enough for a debugging
// session is too dense for a quick scan. Milliseconds (and the date, since
// it repeats down a column) are still one hover away, in the title.
function pad(n, len = 2) {
  return String(n).padStart(len, "0");
}

function localIso(date, withMs) {
  const stamp =
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return withMs ? stamp + "." + pad(date.getMilliseconds(), 3) : stamp;
}

function timeCell(entry) {
  const date = new Date(entry.ts);
  return el("span", { title: localIso(date, true), text: localIso(date, false) });
}

// Links into this same view, narrowed to just this one call -- entry.
// artifact_path, not opts.id, since a run-scoped view mixes entries from
// several artifacts and each call badge has to point at its own.
function callLink(entry) {
  const href = "#/artifact/" + entry.artifact_path + "?call=" + encodeURIComponent(entry.call_id);
  return el("a", { class: "call-badge", href, title: entry.call_id, text: shortId(entry.call_id) });
}

// "jobname@call_id" -- job name hovers to reveal the artifact_path it was
// resolved from (main.py's artifact_job_name), call_id links to that one
// call's log only. Falls back to a bare label when a manifest couldn't be
// resolved (job is None) rather than leaving the cell blank next to a real
// call id.
function jobCell(entry) {
  return el("span", {},
    el("span", { class: "log-jobname", title: entry.artifact_path || "", text: entry.job || "job" }),
    "@",
    callLink(entry),
  );
}

function levelCell(entry) {
  const cls = LEVEL_CLASS[entry.level];
  return el("div", { class: "log-level" + (cls ? " " + cls : ""), text: entry.level });
}

// One log line: time, job@call (hover/link), level, message. Plain divs, not
// <tr>/<td> -- the row is `display: contents` (see index.html's CSS) so its
// four cells drop straight into the log-table grid, which is what lets ts/
// job/level size to `max-content` (as tight as their own text, never
// wrapping) while msg's `1fr` column absorbs whatever's left. A <table>
// can't do that: table-layout: fixed takes widths from the first row only,
// and auto takes them from every row's msg text too, ballooning the whole
// table -- neither gives "tight unless it needs more, rest to msg" for free.
function logRow(entry) {
  return el("div", { class: "log-row" },
    el("div", { class: "log-ts" }, timeCell(entry)),
    el("div", { class: "log-job" }, jobCell(entry)),
    levelCell(entry),
    el("div", { class: "log-msg", text: entry.message }),
  );
}

// Distinct levels actually present, in a fixed severity order (anything
// outside that order, if it ever shows up, falls in behind in first-seen
// order). Computed off the *unfiltered* entries so hiding a level doesn't
// make its own checkbox disappear.
function presentLevels(entries) {
  const seen = [];
  for (const entry of entries) {
    if (!seen.includes(entry.level)) seen.push(entry.level);
  }
  return LEVEL_ORDER.filter((l) => seen.includes(l)).concat(seen.filter((l) => !LEVEL_ORDER.includes(l)));
}

// One checkbox per level present in this view.
function levelFilterBar(entries, opts) {
  const levels = presentLevels(entries);
  if (levels.length < 1) return null;
  return el("p", { class: "log-filter" },
    ...levels.map((level) =>
      el("label", { class: "log-filter-item" },
        el("input", {
          type: "checkbox",
          checked: !opts.hiddenLevels.has(level),
          onchange: () => opts.onToggleLevel(level),
        }),
        " " + level,
      ),
    ),
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

// A parameter's value as one line: primitives print plain, anything richer
// (a list, a nested config) prints as compact JSON -- enough to show what's
// there without a full nested tree view. Dependency values never reach here
// -- manifest_endpoint's own parameters/dependencies split already keeps
// those out of `parameters`.
function paramValue(value) {
  if (value === null || typeof value !== "object") return String(value);
  return JSON.stringify(value);
}

// The artifact drill-down summary: type, own parameters, and one link per
// direct dependency. main.py's artifact_manifest_summary already resolved
// each dependency to its own artifact_path -- this links there rather than
// inlining that dependency's manifest, so drilling further is a click away
// instead of the whole tree being dumped on one page. Nothing to show when
// the artifact hasn't been built yet (manifest.error is set) or opts.manifest
// hasn't loaded (still null on the very first render).
function artifactSummary(manifest) {
  if (!manifest || manifest.error) return null;

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

export function renderLogView(container, payload, opts) {
  const entries = payload.entries || [];
  const nodes = [el("p", { class: "log-back" }, el("a", { href: "#/", text: "← dashboard" }))];

  if (opts.scope === "artifact") {
    const summary = artifactSummary(opts.manifest);
    if (summary) nodes.push(summary);
  }

  if (payload.error) {
    nodes.push(el("p", { class: "log-error", text: payload.error }));
  }

  if (opts.scope === "artifact" && opts.callId) {
    nodes.push(el("p", { class: "log-artifacts" },
      el("a", { href: "#/artifact/" + opts.id, text: "↳ clear call filter, show all of " + opts.id }),
    ));
  } else if (opts.scope === "run") {
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

  const filterBar = levelFilterBar(entries, opts);
  if (filterBar) nodes.push(filterBar);

  const visible = entries.filter((entry) => !opts.hiddenLevels.has(entry.level));

  if (!entries.length) {
    if (!payload.error) nodes.push(el("p", { class: "empty", text: "no log entries yet." }));
  } else if (!visible.length) {
    nodes.push(el("p", { class: "empty", text: "no entries at the selected levels." }));
  } else {
    nodes.push(el("div", { class: "log-table" }, ...visible.map(logRow)));
  }

  container.replaceChildren(...nodes);
}
