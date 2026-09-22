// render.js -- components.
//
// Each function takes state and returns a DOM node. Nothing here reaches into
// app state or fetches anything: to add a column or a new cell you edit one
// small function, not the fetch. Interaction is injected via `ctx` so this
// file never needs to know WHERE the launch call or group state live.
//
//   ctx = {
//     onLaunch(artifactPath) -> void, // run it now
//     onCancel(artifactPath) -> void, // stop the call working on it
//     pending(artifactPath) -> string | undefined, // "starting"/"stopping" until the next refresh
//   }

import { el } from "./el.js";

// A dimmed placeholder cell value ("—", "no lease", …).
function dim(text) {
  return el("span", { class: "dim", text });
}

// One status light per artifact: the server's `verdict` (see main.py's
// `state`) as a color, with the reason on hover.
//   green  (done)     : status is "done"
//   blue   (running)  : an active call is working on it
//   blue   (starting) : waiting for a first heartbeat within the startup grace
//   red    (failed)   : a call held the lease and went stale before
//                        finishing, never beat, or the manifest would not load
//   amber  (runnable) : not done, not blocked, no call in progress
//   gray   (blocked)  : not done, not runnable -- waiting on a dependency
function dot(state) {
  const title = {
    done: "done",
    running: "running",
    starting: "starting: waiting for first heartbeat",
    failed:
      state.status === "conflict"
        ? "conflict: manifest would not load"
        : state.last_heartbeat == null
          ? "failed: call never sent a heartbeat"
          : "failed: call went stale before finishing",
    runnable: "runnable",
    blocked: "blocked on " + (state.blocked_by.join(", ") || "?"),
  }[state.verdict];
  return el("span", { class: "dot " + state.verdict, title });
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

// One payload -- main.py's live_progress (what a running job wrote) or
// durable_progress (what its artifact read off the volume) -- into one compact
// line: "phase: done/total", or just "done/total" for a payload that doesn't
// name a phase.
//
// Nothing is interpreted server-side, so a payload arrives in whatever shape
// the thing that produced it chose. Two count the same way under different
// names -- done/total and step/total_steps -- and anything else still shows
// generically rather than not at all. Counts are never abbreviated: the
// difference between 4741 and 4742 is exactly what this column gets read for.
function formatProgress(p) {
  if (!p) return null;
  const done = p.done ?? p.step;
  const total = p.total ?? p.total_steps;
  if (typeof done === "number" && typeof total === "number") {
    return (p.phase ? p.phase + ": " : "") + done + "/" + total;
  }
  return Object.entries(p).map(([k, v]) => k + "=" + v).join(" ");
}

// live_progress first -- a currently-active call's own self-report -- falling
// back to durable_progress (read off the volume, and the only one still
// meaningful once nothing's active anymore).
function progressCell(state) {
  const text = formatProgress(state.live_progress) || formatProgress(state.durable_progress);
  return text ? el("span", { text }) : dim("—");
}

// One square button per artifact, offering the one thing worth doing to it.
// Asked in this order:
//
//   done                    -> run, frozen
//   a call is running       -> stop
//   a call is starting      -> starting, blue and frozen
//   ready                   -> run
//   blocked                 -> run, frozen
//
// `done` is asked first because it's the one answer that can't be undone.
// Everything else is a claim about right now, and a claim can lag: a lease
// outlives its call, and a beat is only as fresh as the last one written. An
// artifact that is on disk has nothing left worth stopping, whatever the
// liveness signals still say.
//
// "Frozen" (shown but inert) rather than absent, so the slot in the row
// doesn't shift once there is something to do. A button the page has
// already asked something of (ctx.pending) freezes too, blue rather than
// gray and saying what was asked, until a refresh brings a map that knows.
function actionButton(path, state, ctx) {
  const pending = ctx.pending(path);
  const act = (kind, label, title, handler) =>
    el("button", {
      class: "launch-btn " + kind + (pending ? " clicked" : ""),
      text: pending || label,
      title: pending ? "until the next refresh" : title,
      disabled: Boolean(pending) || !handler,
      onclick: pending || !handler ? undefined : () => handler(path),
    });

  if (state.done) return act("run", "run", "already done", null);
  if (state.active) return act("cancel", "stop", "cancel this running job", ctx.onCancel);
  if (state.verdict === "starting") return act("starting", "starting", "waiting for first heartbeat", null);
  if (state.ready) return act("run", "run", "launch", ctx.onLaunch);
  return act("run", "run", "blocked on " + (state.blocked_by.join(", ") || "?"), null);
}


// One <tr> per declared artifact: status dot + type, path, call/heartbeat,
// progress, and the button that launches or stops it.
export function artifactRow(path, state, ctx) {
  return el("tr", { class: "artifact-row" },
    el("td", {},
      dot(state),
      el("span", { class: "artifact-type", text: state.type }),
      // A MappedDataSet owns no bytes of its own, worth flagging inline
      // rather than making a reader infer it from the type name alone.
      state.type === "MappedDataSet" ? el("span", { class: "artifact-tag", text: " (Mapped)" }) : null,
    ),
    el("td", { class: "artifact-path dim" },
      el("a", { class: "artifact-link", href: "#/artifact/" + path, text: path, title: path }),
    ),
    el("td", { class: "call" }, callId(state)),
    el("td", {}, heartbeat(state)),
    el("td", {}, progressCell(state)),
    el("td", {}, actionButton(path, state, ctx)),
  );
}

// One <tr> in the problems table: a manifest state() could not read, so
// there is no artifact to show for it -- just where it is and what the read
// said.
export function problemRow(path, state) {
  return el("tr", {},
    el("td", { class: "run", text: path }),
    el("td", { class: "error", text: state.error }),
  );
}
