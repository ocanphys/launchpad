// app.js — the runtime: config, app state, the poll loop, launching, and
// routing between the dashboard and the log viewer. This is the only file
// that talks to the network or holds mutable state. Rendering is delegated
// to render.js and logview.js; DOM building to el.js.

import { runRow, problemRow } from "./render.js";
import { renderLogView } from "./logview.js";

// --- config -----------------------------------------------------------------

// How often to re-ask the server. This only bounds how stale the screen can
// look; it is not a correctness knob.
const EVERY_MS = 2000;

// A just-clicked artifact stays frozen this long even if an in-between poll
// still shows it ready — the spawn is real before the first heartbeat
// lands, so without this a second click in that window could launch a
// duplicate call. Cleared early once a poll shows it active; this is the
// backstop.
const PENDING_MS = 15000;

// --- dom refs ---------------------------------------------------------------

const titleEl = document.getElementById("title");
const connEl = document.getElementById("conn");
const metaEl = document.getElementById("meta");
const dashboardEl = document.getElementById("dashboard");
const rowsEl = document.getElementById("rows");
const emptyEl = document.getElementById("empty");
const problemsEl = document.getElementById("problems");
const problemCountEl = document.getElementById("problemCount");
const problemRowsEl = document.getElementById("problemRows");
const logViewEl = document.getElementById("logView");

// --- app state --------------------------------------------------------------

let lastPayload = null;
const pending = new Map(); // artifactPath -> Date.now() when clicked

function isPending(artifactPath) {
  const ts = pending.get(artifactPath);
  return ts !== undefined && Date.now() - ts < PENDING_MS;
}

function setConn(text, cls) {
  connEl.textContent = text;
  connEl.className = cls;
}

// --- launching --------------------------------------------------------------

async function launchJob(artifactPath) {
  pending.set(artifactPath, Date.now());
  redraw(); // freeze the button immediately, don't wait for the next poll

  try {
    // No encodeURIComponent -- artifactPath's /s are meant to stay literal,
    // matching the server's {artifact_path:path} route (a plain path
    // segment can't match a multi-segment path).
    const res = await fetch(`launch/${artifactPath}`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.launched) {
      console.warn("launch failed:", body.message || "HTTP " + res.status);
      pending.delete(artifactPath); // refused — let it re-evaluate
      redraw();
    }
    // On success it stays pending until a poll shows it active (or
    // PENDING_MS elapses).
  } catch (err) {
    console.warn("launch failed:", err);
    pending.delete(artifactPath);
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

  // runRow returns [summaryRow, artifactsRow] per run -- flatMap, not map,
  // so replaceChildren sees a flat list of <tr>s rather than one nested
  // array per run.
  rowsEl.replaceChildren(...ids.flatMap((id) => runRow(id, runs[id], ctx)));
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

// --- dashboard poll ----------------------------------------------------------

async function pollDashboard() {
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

// --- log view poll -----------------------------------------------------------

// One fetcher, parameterized by scope, rather than two near-duplicates --
// "run" and "artifact" differ only in which URL and which id renderLogView
// gets, never in how the fetch/error/render sequence goes.
async function pollLogView(scope, id) {
  try {
    const url = scope === "run" ? `logs/run/${id}` : `logs/artifact/${id}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    renderLogView(logViewEl, payload, { scope, id });
    setConn("live", "ok");
  } catch (err) {
    setConn(String(err.message || err), "bad");
  }
}

// --- routing ------------------------------------------------------------------

// Three routes, all client-side: "" (the dashboard), "run/<id>", and
// "artifact/<path>". No encodeURIComponent when building or reading these --
// an artifact_path's /s are meant to stay literal, the same convention
// launchJob already follows for the POST /launch route.
function parseRoute() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("run/")) return { view: "run", id: hash.slice(4) };
  if (hash.startsWith("artifact/")) return { view: "artifact", id: hash.slice(9) };
  return { view: "dashboard" };
}

let pollTimer = null;

// Re-entered on every hashchange, and once at load. Owns the one active
// interval: switching routes clears whichever poll was running before,
// so navigating away from a run's log view doesn't leave it quietly
// polling in the background.
function route() {
  clearInterval(pollTimer);
  const r = parseRoute();

  if (r.view === "dashboard") {
    titleEl.textContent = "runs";
    dashboardEl.hidden = false;
    logViewEl.hidden = true;
    pollDashboard();
    pollTimer = setInterval(pollDashboard, EVERY_MS);
    return;
  }

  titleEl.textContent = r.view === "run" ? "run " + r.id : r.id;
  dashboardEl.hidden = true;
  logViewEl.hidden = false;
  const tick = () => pollLogView(r.view, r.id);
  tick();
  pollTimer = setInterval(tick, EVERY_MS);
}

window.addEventListener("hashchange", route);
route();
