// render.js — components.
//
// Each function takes state and returns a DOM node. Nothing here reaches into
// app state or fetches anything: to add a column or a new cell you edit one
// small function, not the poll loop. Interaction is injected via `ctx` so this
// file never needs to know WHERE the pending-set or the launch call live.
//
//   ctx = {
//     isPending(artifactPath)    -> boolean, // is this artifact mid-launch?
//     onLaunch(artifactPath)     -> void,    // user clicked a launchable artifact
//     isGroupOpen(runId, type)   -> boolean, // is this run's type-group expanded?
//     onToggleGroup(runId, type) -> void,    // user clicked a group's summary row
//   }

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
function verdict(state) {
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

// One square button per artifact. "Frozen" (shown but inert) when it isn't
// launchable -- keeps its slot in the row from shifting once it does become
// launchable. Frozen when: mid-launch, this artifact already has an active
// call, or it isn't ready (already done, or blocked on a dependency).
function launchButton(path, state, ctx) {
  const pending = ctx.isPending(path);
  const frozen = pending || state.active || !state.ready;

  const title = !state.ready
    ? state.done
      ? "already done"
      : "blocked on " + (state.blocked_by.join(", ") || "?")
    : state.active
    ? "already running"
    : pending
    ? "launching…"
    : "launch";

  const button = el("button", {
    class: "launch-btn",
    text: "run",
    title,
    disabled: frozen,
    onclick: frozen ? undefined : () => ctx.onLaunch(path),
  });

  if (!pending) return button;

  return el("span", { class: "launch-wrap" },
    button,
    el("span", { class: "launch-spinner" }),
  );
}

// One <tr> per declared artifact: status dot + type, path, call/heartbeat,
// and the button that launches it. `grouped` indents it one level deeper --
// it's nested under a type-group's summary row rather than sitting directly
// under the run header.
function artifactRow(path, state, ctx, grouped = false) {
  return el("tr", { class: "artifact-row" + (grouped ? " grouped" : "") },
    el("td", {},
      dot(state),
      el("span", { class: "artifact-type", text: state.type }),
    ),
    el("td", { class: "artifact-path dim" },
      el("a", { class: "artifact-link", href: "#/artifact/" + path, text: path, title: path }),
    ),
    el("td", { class: "call" }, callId(state)),
    el("td", {}, heartbeat(state)),
    el("td", {}, launchButton(path, state, ctx)),
  );
}

// One <tr> summarizing a type-group ("source (5)") with a triangle that
// reflects (and toggles) whether it's expanded, plus a dot rolling up the
// status of every member (see groupVerdict). Click target is the whole
// row, not just the arrow -- a group can have a lot of artifacts under it,
// and a fiddly hit target for the only way to reach them is a bad trade.
// Same weight as a plain artifact row -- it's standing in for one, not a
// heading, so it shouldn't out-shout the rows around it.
function groupHeaderRow(runId, type, states, open, ctx) {
  return el("tr", { class: "group-header", onclick: () => ctx.onToggleGroup(runId, type) },
    el("td", { colspan: 5 },
      el("span", { class: "dot " + groupVerdict(states) }),
      el("span", { class: "group-arrow", text: open ? "▾" : "▸" }),
      el("span", { class: "group-type", text: type }),
      el("span", { class: "dim", text: " (" + states.length + ")" }),
    ),
  );
}

// A run is a grouping label, not a data row of its own now -- everything
// that used to be per-run (lease, call, heartbeat) is per-artifact instead
// (see main.py's read_state). One header row, then the run's artifacts
// grouped by type (state.type -- the job class name, e.g. "SourceJob"):
// a type with more than one artifact collapses behind a summary row
// (closed unless ctx.isGroupOpen says otherwise), while a type with just
// one artifact renders that row directly -- collapsing a group of one
// would only cost a click for no payoff. Groups are ordered by type name;
// paths within a group keep the path sort `read_state` uses, so a fully
// expanded run still lists in the same order a directory listing under
// runs/{id} would.
//
// Returned as an array -- app.js's draw() flattens these into the table
// with the rest.
export function runRow(id, run, ctx) {
  const artifacts = run.artifacts || {};
  const paths = Object.keys(artifacts).sort();

  const header = el("tr", { class: "run-header" },
    el("td", { colspan: 5 },
      el("a", { class: "run-link", href: "#/run/" + id, text: id }),
    ),
  );
  if (paths.length === 0) {
    return [header, el("tr", {}, el("td", { colspan: 5 }, dim("no artifacts declared")))];
  }

  const groups = new Map(); // type -> paths, in path-sorted order
  for (const path of paths) {
    const type = artifacts[path].type;
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type).push(path);
  }

  const rows = [header];
  for (const type of [...groups.keys()].sort()) {
    const groupPaths = groups.get(type);
    if (groupPaths.length === 1) {
      rows.push(artifactRow(groupPaths[0], artifacts[groupPaths[0]], ctx));
      continue;
    }
    const open = ctx.isGroupOpen(id, type);
    const states = groupPaths.map((path) => artifacts[path]);
    rows.push(groupHeaderRow(id, type, states, open, ctx));
    if (open) {
      for (const path of groupPaths) rows.push(artifactRow(path, artifacts[path], ctx, true));
    }
  }
  return rows;
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
