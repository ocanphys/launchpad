// render.js — components.
//
// Each function takes state and returns a DOM node. Nothing here reaches into
// app state or fetches anything: to add a column or a new cell you edit one
// small function, not the poll loop. Interaction is injected via `ctx` so this
// file never needs to know WHERE the pending-set or the launch call live.
//
//   ctx = {
//     isPending(runId, jobUid) -> boolean,   // is this job mid-launch?
//     onLaunch(runId, jobUid)  -> void,      // user clicked a launchable job
//   }

import { el } from "./el.js";

// A dimmed placeholder cell value ("—", "no lease", …).
function dim(text) {
  return el("span", { class: "dim", text });
}

// Lease/heartbeat status light.
//   green (active): a lease held by a call whose heartbeat is fresh
//   red   (stale) : a lease, but the holder's heartbeat is stale or missing
//   gray  (none)  : no lease at all — never started, or superseded
function dot(run) {
  const state = !run.call_id ? "none" : run.active ? "active" : "stale";
  return el("span", { class: "dot " + state });
}

function heartbeat(run) {
  if (!run.last_heartbeat) return dim("—");
  // last_heartbeat is unix SECONDS; Date wants ms.
  return document.createTextNode(
    new Date(run.last_heartbeat * 1000).toLocaleTimeString()
  );
}

// One square button per job_uid. A button is "frozen" (shown but inert) when
// the job isn't launchable — this keeps its slot in the row from shifting once
// it does become launchable. Frozen when: mid-launch, the run already has an
// active call, or the job's dependencies aren't met.
function jobButton(runId, jobUid, jobState, run, ctx) {
  const pending = ctx.isPending(runId, jobUid);
  const frozen = pending || run.active || !jobState.ready;

  const title = !jobState.ready
    ? "blocked on " + (jobState.missing_dependencies.join(", ") || "?")
    : run.active
    ? "run already has an active call"
    : pending
    ? "launching…"
    : "launch " + jobUid;

  return el("button", {
    class: "job-btn",
    text: jobUid,
    title,
    disabled: frozen,
    // Only wire the click when it can actually do something.
    onclick: frozen ? undefined : () => ctx.onLaunch(runId, jobUid),
  });
}

function jobButtons(runId, run, ctx) {
  return Object.keys(run.jobs || {})
    .sort()
    .map((jobUid) => jobButton(runId, jobUid, run.jobs[jobUid], run, ctx));
}

// One <tr> for a run. Add/remove a <td> here and nowhere else.
export function runRow(id, run, ctx) {
  return el("tr", {},
    el("td", {}, dot(run)),
    el("td", { class: "run", text: id }),
    el("td", {}, run.job_type || dim("—")),
    el("td", { class: "call" }, run.call_id || dim("no lease")),
    el("td", {}, heartbeat(run)),
    el("td", { class: "job-buttons" }, jobButtons(id, run, ctx)),
  );
}
