// render.js — components.
//
// Each function takes state and returns a DOM node. Nothing here reaches into
// app state or fetches anything: to add a column or a new cell you edit one
// small function, not the poll loop. Interaction is injected via `ctx` so this
// file never needs to know WHERE the launch call or group state live.
//
//   ctx = {
//     onLaunch(artifactPath)      -> void,    // user clicked a launchable artifact
//     isJustClicked(artifactPath) -> boolean, // clicked, not yet refreshed by a poll
//     isGroupOpen(ns, type)       -> boolean, // is this owner's type-group expanded?
//     onToggleGroup(ns, type)     -> void,    // user clicked a group's summary row
//   }
//   `ns` namespaces group open/closed state so two different owners (two
//   runs, or a run and a dataset) with a same-named type-group never
//   collide -- a run id in runRow, a dataset's own artifact_path in
//   datasetRows.

import { el } from "./el.js";

// A dimmed placeholder cell value ("—", "no lease", …).
function dim(text) {
  return el("span", { class: "dim", text });
}

// One status light per artifact, folding declaration status (done/ready/
// blocked_by) and lease/heartbeat (call_id/active) into a single verdict --
// there's one thing worth asking about a row: what would happen if you
// looked at it right now.
//   green  (done)     : status is "done"
//   blue   (running)  : an active call is working on it
//   red    (failed)   : a call held the lease and went stale before
//                        finishing, or the manifest itself conflicts with
//                        what's declared/undeclared
//   amber  (runnable) : not done, not blocked, no call in progress
//   gray   (blocked)  : not done, not runnable -- waiting on a dependency
//
// Takes an artifact's own state now, not a run's -- a lease is granted per
// artifact_path (see main.py's attempt_launch), so this is the granularity
// at which "active" actually means anything.
//
// Exported: app.js also uses this (not just dot()) to tell whether a
// just-clicked artifact's status has actually changed since the click, in
// which case its "just clicked" mark clears early -- see justClicked there.
export function verdict(state) {
  if (state.done) return "done";
  if (state.active) return "running";
  if (state.status === "conflict") return "failed";
  if (state.status === "undeclared") return "failed";
  if (state.call_id) return "failed";
  if (state.ready) return "runnable";
  return "blocked";
}

function dot(state) {
  const cls = verdict(state);
  const title = {
    done: "done",
    running: "running",
    failed:
      state.status === "conflict"
        ? "conflict: manifest disagrees with what's declared"
        : state.status === "undeclared"
        ? "undeclared: outputs exist with no manifest"
        : "failed: call went stale before finishing",
    runnable: "runnable",
    blocked: "blocked on " + (state.blocked_by.join(", ") || "?"),
  }[cls];
  return el("span", { class: "dot " + cls, title });
}

// A type-group's own dot summarizes its members by the same palette, worst
// (most attention-worthy) first: any failed member makes the group red even
// if others are done; any running member makes it blue; only when every
// member is done does it go green; otherwise amber if anything is
// launchable right now, gray if the whole group is just blocked.
function groupVerdict(states) {
  const verdicts = states.map(verdict);
  if (verdicts.includes("failed")) return "failed";
  if (verdicts.includes("running")) return "running";
  if (verdicts.every((v) => v === "done")) return "done";
  if (verdicts.includes("runnable")) return "runnable";
  return "blocked";
}

// Call ids are long and only their tail varies day to day -- show the last
// 8 chars, full value on hover via the native title tooltip.
function callId(state) {
  if (!state.call_id) return dim("—");
  const id = state.call_id;
  const short = id.length > 8 ? id.slice(-8) : id;
  return el("span", { title: id, text: short });
}

function heartbeat(state) {
  if (!state.last_heartbeat) return dim("—");
  // last_heartbeat is unix SECONDS; Date wants ms.
  return document.createTextNode(
    new Date(state.last_heartbeat * 1000).toLocaleTimeString()
  );
}

// One payload (main.py's live_progress or durable_progress -- same shape
// either way, whatever the job's own progress()/durable_progress returned)
// into one line of text. step/total_steps is the one shape a real job type
// (PretrainJob) actually produces today, so that gets a clean "n/total";
// anything else -- a future job type reporting something different -- still
// shows *something* rather than nothing, generically.
function formatProgress(p) {
  if (!p) return null;
  if (typeof p.step === "number" && typeof p.total_steps === "number") {
    return p.step + "/" + p.total_steps;
  }
  return Object.entries(p).map(([k, v]) => k + "=" + v).join(" ");
}

// live_progress first -- a currently-active call's own self-report -- falling
// back to durable_progress (main.py's `_with_progress`: the volume-read
// signal, the only one still meaningful once nothing's active anymore).
function progressCell(state) {
  const text = formatProgress(state.live_progress) || formatProgress(state.durable_progress);
  return text ? el("span", { text }) : dim("—");
}

// One square button per artifact. "Frozen" (shown but inert) when it isn't
// launchable -- keeps its slot in the row from shifting once it does become
// launchable. Frozen when this artifact already has an active call, isn't
// ready (already done, or blocked on a dependency), or was just clicked
// (ctx.isJustClicked) -- that last one is purely local (app.js's
// justClicked), held for a minimum stretch of polls or until this
// artifact's own verdict actually changes, whichever comes first. A
// just-clicked button turns blue rather than showing a spinner.
function launchButton(path, state, ctx) {
  const justClicked = ctx.isJustClicked(path);
  const frozen = justClicked || state.active || !state.ready;

  const title = !state.ready
    ? state.done
      ? "already done"
      : "blocked on " + (state.blocked_by.join(", ") || "?")
    : state.active
    ? "already running"
    : justClicked
    ? "launching…"
    : "launch";

  return el("button", {
    class: "launch-btn" + (justClicked ? " clicked" : ""),
    text: "run",
    title,
    disabled: frozen,
    // Passes `state` along with `path` -- app.js's launchJob snapshots
    // this artifact's verdict at the moment of the click, to compare
    // against on every later poll (see justClicked there).
    onclick: frozen ? undefined : () => ctx.onLaunch(path, state),
  });
}

// One <tr> per declared artifact: status dot + type, path, call/heartbeat,
// and the button that launches it. `depth` indents it further for each
// level of nesting it's under -- 0 for a row sitting directly under its
// owner (a run header, or a dataset's own row), 1 for one nested under a
// type-group's summary row, and so on (a dataset's dependency rows are
// nested two deep: the dataset's own row, then its type-group, then this).
function artifactRow(path, state, ctx, depth = 0) {
  return el("tr", { class: "artifact-row" + (depth > 0 ? " depth-" + depth : "") },
    el("td", {},
      dot(state),
      el("span", { class: "artifact-type", text: state.type }),
      // `mapped` only ever appears on a dataset (main.py's read_state, its
      // `datasets` key) -- MappedDataSet owns no bytes of its own, worth flagging inline
      // rather than making a reader infer it from the type name alone.
      state.mapped ? el("span", { class: "artifact-tag", text: " (Mapped)" }) : null,
    ),
    el("td", { class: "artifact-path dim" },
      el("a", { class: "artifact-link", href: "#/artifact/" + path, text: path, title: path }),
    ),
    el("td", { class: "call" }, callId(state)),
    el("td", {}, heartbeat(state)),
    el("td", {}, progressCell(state)),
    el("td", {}, launchButton(path, state, ctx)),
  );
}

// One <tr> summarizing a type-group ("source (5)") with a triangle that
// reflects (and toggles) whether it's expanded, plus a dot rolling up the
// status of every member (see groupVerdict). Click target is the whole
// row, not just the arrow -- a group can have a lot of artifacts under it,
// and a fiddly hit target for the only way to reach them is a bad trade.
// Same weight as a plain artifact row -- it's standing in for one, not a
// heading, so it shouldn't out-shout the rows around it. `depth` works the
// same as artifactRow's own -- 0 for a group sitting directly under its
// owner, 1 for one nested one level deeper (a dataset's own dependency
// groups, under the dataset's row).
function groupHeaderRow(ns, type, states, open, ctx, depth = 0) {
  return el("tr", { class: "group-header" + (depth > 0 ? " depth-" + depth : ""), onclick: () => ctx.onToggleGroup(ns, type) },
    el("td", { colspan: 6 },
      el("span", { class: "dot " + groupVerdict(states) }),
      el("span", { class: "group-arrow", text: open ? "▾" : "▸" }),
      el("span", { class: "group-type", text: type }),
      el("span", { class: "dim", text: " (" + states.length + ")" }),
    ),
  );
}

// Topological depth of every artifact in `artifacts`, from its own
// `depends_on` list (an edge to another path already present in this same
// dict -- a dependency outside the set, e.g. not yet built, doesn't count).
// 0 for a leaf (a Source, a Tokenizer -- nothing here to build first); 1 +
// its deepest dependency otherwise. `typeGroupedRows` sorts by this,
// descending, so what has to exist before anything else can be built sinks
// to the bottom and what depends on everything above it floats to the top
// -- a topological order, inverted for display. Memoized per call, with a
// zero planted before recursing so a cycle (shouldn't happen -- deps.py
// forbids it -- but this is display code, not the source of truth) reads
// as depth 0 rather than looping forever.
function topoDepth(artifacts) {
  const depths = new Map();
  function depth(path) {
    if (depths.has(path)) return depths.get(path);
    depths.set(path, 0);
    const dependsOn = (artifacts[path].depends_on || []).filter((p) => p in artifacts);
    const d = dependsOn.length === 0 ? 0 : 1 + Math.max(...dependsOn.map(depth));
    depths.set(path, d);
    return d;
  }
  for (const path of Object.keys(artifacts)) depth(path);
  return depths;
}

// A flat {path: state} dict, grouped by artifact type and rendered as
// group-header + artifactRow rows -- the piece runRow and datasetRows both
// need (a run's own artifacts; a dataset's dependency closure), factored
// out so there's one grouping/collapsing implementation, not two drifting
// copies. A type with more than one member collapses behind a summary row
// (closed unless ctx.isGroupOpen says otherwise); a type with just one
// member renders that row directly -- collapsing a group of one would only
// cost a click for no payoff. Groups, and paths within a group, are ordered
// by topological depth (topoDepth), deepest first, so a run reads top to
// bottom the way it was built bottom to top -- ties (same depth) break on
// type name, then path, so ordering stays deterministic poll to poll.
//
// `ns` namespaces the open/closed state (see the ctx doc comment up top).
// `depth` is where the *group header* (and any singleton row) sits;
// members of an expanded group render one level deeper still.
function typeGroupedRows(ns, artifacts, ctx, depth = 0) {
  const paths = Object.keys(artifacts);
  if (paths.length === 0) return [];

  const depths = topoDepth(artifacts);

  const groups = new Map(); // type -> paths
  for (const path of paths) {
    const type = artifacts[path].type;
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type).push(path);
  }
  for (const groupPaths of groups.values()) {
    groupPaths.sort((a, b) => depths.get(b) - depths.get(a) || a.localeCompare(b));
  }

  const groupDepth = (type) => Math.max(...groups.get(type).map((p) => depths.get(p)));
  const orderedTypes = [...groups.keys()].sort(
    (a, b) => groupDepth(b) - groupDepth(a) || a.localeCompare(b)
  );

  const rows = [];
  for (const type of orderedTypes) {
    const groupPaths = groups.get(type);
    if (groupPaths.length === 1) {
      rows.push(artifactRow(groupPaths[0], artifacts[groupPaths[0]], ctx, depth));
      continue;
    }
    const open = ctx.isGroupOpen(ns, type);
    const states = groupPaths.map((path) => artifacts[path]);
    rows.push(groupHeaderRow(ns, type, states, open, ctx, depth));
    if (open) {
      for (const path of groupPaths) rows.push(artifactRow(path, artifacts[path], ctx, depth + 1));
    }
  }
  return rows;
}

// A run is a grouping label, not a data row of its own now -- everything
// that used to be per-run (lease, call, heartbeat) is per-artifact instead
// (see main.py's read_state). One header row, then the run's artifacts
// type-grouped via typeGroupedRows, namespaced by the run's own id.
//
// Returned as an array -- app.js's draw() flattens these into the table
// with the rest.
export function runRow(id, run, ctx) {
  const artifacts = run.artifacts || {};
  const header = el("tr", { class: "run-header" },
    el("td", { colspan: 6 },
      el("a", { class: "run-link", href: "#/run/" + id, text: id }),
      run.notebook
        ? el("a", {
            class: "lab-run-link",
            href: "/lab/run/" + encodeURIComponent(id),
            target: "_blank",
            rel: "noopener",
            title: "open in lab",
            text: "lab ↗",
          })
        : null,
    ),
  );
  if (Object.keys(artifacts).length === 0) {
    return [header, el("tr", {}, el("td", { colspan: 6 }, dim("no artifacts declared")))];
  }
  return [header, ...typeGroupedRows(id, artifacts, ctx)];
}

// The `sources` view: a flat table, one row per Source ever declared
// (main.py's read_state, its `sources` key) -- no grouping, per spec. `[]`
// rather than `null` reads as "genuinely empty" the same way runRow's own
// artifacts dict does, so app.js can drive the shared #empty message off
// length alone regardless of which view it's showing.
export function sourcesRows(payload, ctx) {
  const artifacts = payload.sources || {};
  const paths = Object.keys(artifacts).sort();
  return paths.map((path) => artifactRow(path, artifacts[path], ctx));
}

// The `datasets` view: DataSet + MappedDataSet artifacts (main.py's
// read_state, its `datasets` key), shown the same way runRow shows a run --
// the dataset's own row, then everything it depends on (its full dependency closure:
// TokenizedSource, and through those, Tokenizer and Source), type-grouped
// and collapsible via the same typeGroupedRows runRow uses. Namespaced by
// the dataset's own artifact_path, so two datasets that both have (say) a
// "Source" group never share open/closed state.
export function datasetRows(payload, ctx) {
  const datasets = payload.datasets || {};
  const paths = Object.keys(datasets).sort();
  return paths.flatMap((path) => {
    const entry = datasets[path];
    const row = artifactRow(path, entry.state, ctx);
    return [row, ...typeGroupedRows(path, entry.artifacts || {}, ctx, 1)];
  });
}

// One <tr> in the problem-runs table: a run whose artifact discovery itself
// raised, so there's no artifacts snapshot to show for it -- just which run
// and what the check said.
export function problemRow(id, problem) {
  return el("tr", {},
    el("td", { class: "run", text: id }),
    el("td", { class: "error", text: problem.error }),
  );
}
