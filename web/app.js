// app.js — the runtime: config, app state, the poll loop, and launching.
// This is the only file that talks to the network or holds mutable state.
// Rendering is delegated to render.js; DOM building to el.js.

import { runRow, problemRow } from "./render.js";

// --- config -----------------------------------------------------------------

// How often to re-ask the server. This only bounds how stale the screen can
// look; it is not a correctness knob.
const EVERY_MS = 2000;

// A just-clicked job stays frozen this long even if an in-between poll still
// shows it ready — the spawn is real before the first heartbeat lands, so
// without this a second click in that window could launch a duplicate call.
// Cleared early once a poll shows the run active; this is the backstop.
const PENDING_MS = 15000;

// --- dom refs ---------------------------------------------------------------

const connEl = document.getElementById("conn");
const metaEl = document.getElementById("meta");
const rowsEl = document.getElementById("rows");
const emptyEl = document.getElementById("empty");
const problemsEl = document.getElementById("problems");
const problemCountEl = document.getElementById("problemCount");
const problemRowsEl = document.getElementById("problemRows");

// --- app state --------------------------------------------------------------

let lastPayload = null;
const pending = new Map(); // "runId:jobUid" -> Date.now() when clicked

function isPending(runId, jobUid) {
  const ts = pending.get(runId + ":" + jobUid);
  return ts !== undefined && Date.now() - ts < PENDING_MS;
}

function setConn(text, cls) {
  connEl.textContent = text;
  connEl.className = cls;
}

// --- launching --------------------------------------------------------------

async function launchJob(runId, jobUid) {
  pending.set(runId + ":" + jobUid, Date.now());
  redraw(); // freeze the button immediately, don't wait for the next poll

  try {
    const res = await fetch(
      `launch/${encodeURIComponent(runId)}/${encodeURIComponent(jobUid)}`,
      { method: "POST" }
    );
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.launched) {
      console.warn("launch failed:", body.message || "HTTP " + res.status);
      pending.delete(runId + ":" + jobUid); // refused — let it re-evaluate
      redraw();
    }
    // On success it stays pending until a poll shows the run active
    // (or PENDING_MS elapses).
  } catch (err) {
    console.warn("launch failed:", err);
    pending.delete(runId + ":" + jobUid);
    redraw();
  }
}

// Injected into every row so render.js never sees app state directly.
const ctx = { isPending, onLaunch: launchJob };

// --- draw -------------------------------------------------------------------

// Pure function of the last payload. Full rebuild each time is correct and
// cheap at this scale; don't add reconciliation until something focusable
// needs to survive a poll.
function draw(payload) {
  const runs = payload.runs || {};
  const ids = Object.keys(runs).sort();

  metaEl.textContent =
    ids.length + (ids.length === 1 ? " run" : " runs") +
    " · polled " + new Date().toLocaleTimeString();

  rowsEl.replaceChildren(...ids.map((id) => runRow(id, runs[id], ctx)));
  emptyEl.hidden = ids.length > 0;

  // Only touches #problemRows' children and the count text -- never
  // recreates <details id="problems"> itself, so a poll can't clobber
  // whether the user has it open.
  const problems = payload.problem_runs || {};
  const problemIds = Object.keys(problems).sort();
  problemRowsEl.replaceChildren(...problemIds.map((id) => problemRow(id, problems[id])));
  problemCountEl.textContent = problemIds.length;
  problemsEl.hidden = problemIds.length === 0;
}

// Redraw from the last good payload (used after local state changes).
function redraw() {
  if (lastPayload) draw(lastPayload);
}

// --- poll loop --------------------------------------------------------------

async function poll() {
  try {
    // Same origin as this page — no URL to configure, no CORS to satisfy.
    const res = await fetch("state", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    lastPayload = await res.json();
    draw(lastPayload);
    setConn("live", "ok");
  } catch (err) {
    // Keep the last good screen up: one missed poll is not news, and blanking
    // the page on a blip hides the state someone is watching.
    setConn(String(err.message || err), "bad");
  }
}

poll();
setInterval(poll, EVERY_MS);
