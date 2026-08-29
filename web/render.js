// render.js — components.
//
// Each function takes state and returns a DOM node. Nothing here reaches into
// app state or fetches anything: to add a column or a new cell you edit one
// small function, not the poll loop. Interaction is injected via `ctx` so this
// file never needs to know WHERE the pending-set or the launch call live.
//
//   ctx = {
//     isPending(artifactPath) -> boolean,   // is this artifact mid-launch?
//     onLaunch(artifactPath)  -> void,      // user clicked a launchable artifact
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
function dot(state) {
  let cls, title;
  if (state.done) {
    cls = "done";
    title = "done";
  } else if (state.active) {
    cls = "running";
    title = "running";
  } else if (state.status === "conflict") {
    cls = "failed";
    title = "conflict: manifest disagrees with what's declared";
  } else if (state.status === "undeclared") {
    cls = "failed";
    title = "undeclared: outputs exist with no manifest";
  } else if (state.call_id) {
    cls = "failed";
    title = "failed: call went stale before finishing";
  } else if (state.ready) {
    cls = "runnable";
    title = "runnable";
  } else {
    cls = "blocked";
    title = "blocked on " + (state.blocked_by.join(", ") || "?");
  }
  return el("span", { class: "dot " + cls, title });
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

  return el("button", {
    class: "launch-btn",
    text: "run",
    title,
    disabled: frozen,
    onclick: frozen ? undefined : () => ctx.onLaunch(path),
  });
}

// One <tr> per declared artifact: status dot + type, path, call/heartbeat,
// and the button that launches it.
function artifactRow(path, state, ctx) {
  return el("tr", { class: "artifact-row" },
    el("td", {},
      dot(state),
      el("span", { class: "artifact-type", text: state.type }),
    ),
    el("td", { class: "artifact-path dim" },
      el("a", { class: "artifact-link", href: "#/artifact/" + path, text: path }),
    ),
    el("td", { class: "call" }, callId(state)),
    el("td", {}, heartbeat(state)),
    el("td", {}, launchButton(path, state, ctx)),
  );
}

// A run is a grouping label, not a data row of its own now -- everything
// that used to be per-run (lease, call, heartbeat) is per-artifact instead
// (see main.py's read_state). One header row plus one real <tr> per
// artifact, sorted by path -- same identity key `read_state` uses, so this
// order matches what a directory listing under runs/{id} would show.
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
  return [header, ...paths.map((path) => artifactRow(path, artifacts[path], ctx))];
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
