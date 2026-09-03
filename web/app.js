// app.js — the runtime: config, app state, the poll loop, launching, and
// routing between the three table views (runs/datasets/sources) and the log
// viewer. This is the only file that talks to the network or holds mutable
// state. Rendering is delegated to render.js and logview.js; DOM building
// to el.js.

import { runRow, problemRow, sourcesRows, datasetRows, verdict } from "./render.js";
import { renderLogView } from "./logview.js";

// --- config -----------------------------------------------------------------

// How often to re-ask the server. This only bounds how stale the screen can
// look; it is not a correctness knob.
const EVERY_MS = 2000;

// How long a just-clicked launch button stays blue+disabled at minimum, in
// case its status never visibly changes (e.g. the launch was refused, or a
// spawn just hasn't produced a heartbeat yet). Four times the poll interval
// -- see justClicked below for why this is a floor, not a fixed hold.
const MIN_CLICKED_MS = EVERY_MS * 4;

// --- dom refs ---------------------------------------------------------------

const titleEl = document.getElementById("title");
const navEl = document.getElementById("viewNav");
const connEl = document.getElementById("conn");
const metaEl = document.getElementById("meta");
const dashboardEl = document.getElementById("dashboard");
const rowsEl = document.getElementById("rows");
const emptyEl = document.getElementById("empty");
const problemsEl = document.getElementById("problems");
const problemCountEl = document.getElementById("problemCount");
const problemRowsEl = document.getElementById("problemRows");
const logViewEl = document.getElementById("logView");

// The three table views the nav bar and pollTableView both know about --
// "runs" is the default/existing one, "datasets" and "sources" are its
// siblings, all three sliced from the one payload main.py's read_state
// returns. Log-view routes ("run", "artifact") are a different family
// entirely, handled below.
const TABLE_VIEWS = ["runs", "datasets", "sources"];

// --- app state --------------------------------------------------------------

let lastPayload = null;
let lastView = "runs"; // which of TABLE_VIEWS lastPayload belongs to

// Artifact paths whose launch button was just clicked, each mapped to when
// and what: `since` (Date.now() at click) and `verdict` (render.js's
// verdict() for this artifact at that moment -- the same classification
// that picks the dot's color). A path clears out of here -- goes back to
// reflecting server state -- the first time either becomes true on a poll:
//
//   - its verdict has changed from what it was at click time (the dot
//     would show a different color -- the clearest possible sign the
//     click did something), checked on every poll regardless of how much
//     time has passed;
//   - MIN_CLICKED_MS has elapsed with no such change (a floor: a click
//     shouldn't free the button again after a single poll if nothing has
//     visibly happened yet, but it also shouldn't stay frozen forever on
//     a launch that silently failed).
//
// See reconcileJustClicked, called from pollTableView on every landed poll.
const justClicked = new Map(); // artifactPath -> { since, verdict }

function isJustClicked(artifactPath) {
  return justClicked.has(artifactPath);
}

// Which of a run's own type-groups (render.js's runRow) are expanded. draw()
// does a full rebuild on every poll, so this has to live here rather than
// as local state on a collapsible element -- otherwise a group would snap
// shut every EVERY_MS. Keyed by namespace + key (a run id + its type here)
// rather than just the type, so two different runs' same-named type-groups
// never collide in the one shared Set.
const openGroups = new Set();

function groupKey(ns, key) {
  return ns + " " + key;
}

function isGroupOpen(ns, key) {
  return openGroups.has(groupKey(ns, key));
}

function toggleGroup(ns, key) {
  const groupKeyStr = groupKey(ns, key);
  if (openGroups.has(groupKeyStr)) openGroups.delete(groupKeyStr);
  else openGroups.add(groupKeyStr);
  redraw();
}

function setConn(text, cls) {
  connEl.textContent = text;
  connEl.className = cls;
}

// --- launching --------------------------------------------------------------

// Marks the button clicked (blue, disabled) immediately -- snapshotting
// `state`'s verdict as the baseline reconcileJustClicked compares later
// polls against -- then fires the POST. Doesn't wait for the POST before
// returning, and doesn't clear the mark itself either way; that's entirely
// reconcileJustClicked's job, off real poll results, not this call's own
// outcome (a launch can report success and still not actually change
// anything visible for a beat).
async function launchJob(artifactPath, state) {
  justClicked.set(artifactPath, { since: Date.now(), verdict: verdict(state) });
  redraw(); // reflect the click immediately, don't wait for the next poll

  try {
    // No encodeURIComponent -- artifactPath's /s are meant to stay literal,
    // matching the server's {artifact_path:path} route (a plain path
    // segment can't match a multi-segment path).
    const res = await fetch(`launch/${artifactPath}`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.launched) {
      console.warn("launch failed:", body.message || "HTTP " + res.status);
    }
  } catch (err) {
    console.warn("launch failed:", err);
  }
}

// Injected into every row so render.js never sees app state directly.
const ctx = { onLaunch: launchJob, isJustClicked, isGroupOpen, onToggleGroup: toggleGroup };

// --- draw -------------------------------------------------------------------

// The `runs` view -- unchanged from before `datasets`/`sources` existed.
function drawRuns(payload) {
  const runs = payload.runs || {};
  const ids = Object.keys(runs).sort();

  metaEl.textContent =
    ids.length + (ids.length === 1 ? " run" : " runs") +
    " · polled " + new Date().toLocaleTimeString();

  // runRow returns [summaryRow, artifactsRow] per run -- flatMap, not map,
  // so replaceChildren sees a flat list of <tr>s rather than one nested
  // array per run.
  rowsEl.replaceChildren(...ids.flatMap((id) => runRow(id, runs[id], ctx)));
  emptyEl.textContent = "no run folders on the volume yet.";
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

// The `datasets` view, per main.py's datasets_state -- one row per
// dataset (datasetRows), each followed by its full dependency closure,
// type-grouped and collapsible the same way a run shows its own artifacts.
function drawDatasets(payload) {
  const datasets = payload.datasets || {};
  const count = Object.keys(datasets).length;

  metaEl.textContent =
    count + (count === 1 ? " dataset" : " datasets") +
    " · polled " + new Date().toLocaleTimeString();

  rowsEl.replaceChildren(...datasetRows(payload, ctx));
  emptyEl.textContent = "no datasets tracked yet.";
  emptyEl.hidden = count > 0;
  problemsEl.hidden = true; // problem_runs has no counterpart in this view
}

// The `sources` view -- flat, per main.py's sources_state.
function drawSources(payload) {
  const artifacts = payload.artifacts || {};
  const count = Object.keys(artifacts).length;

  metaEl.textContent =
    count + (count === 1 ? " source" : " sources") +
    " · polled " + new Date().toLocaleTimeString();

  rowsEl.replaceChildren(...sourcesRows(payload, ctx));
  emptyEl.textContent = "no sources tracked yet.";
  emptyEl.hidden = count > 0;
  problemsEl.hidden = true;
}

const DRAW_BY_VIEW = { runs: drawRuns, datasets: drawDatasets, sources: drawSources };

// Pure function of the last payload (plus which view it's for -- the three
// table views have different payload shapes). Full rebuild each time is
// correct and cheap at this scale; don't add reconciliation until something
// focusable needs to survive a poll.
function draw(view, payload) {
  DRAW_BY_VIEW[view](payload);
}

// Redraw from the last good payload (used after a local state change --
// currently just a group toggle; see ctx above).
function redraw() {
  if (lastPayload) draw(lastView, lastPayload);
}

// --- table view poll ----------------------------------------------------------

// Every artifact-state dict in a payload, flattened to one {path: state}
// map regardless of which view's shape it came from -- `runs` nests one
// level under `runs`, `datasets` nests a dataset's own state plus its
// dependency closure under `datasets`, `sources` is already flat.
function statesByPath(view, payload) {
  const out = {};
  if (view === "runs") {
    for (const run of Object.values(payload.runs || {})) {
      Object.assign(out, run.artifacts || {});
    }
  } else if (view === "datasets") {
    for (const [path, entry] of Object.entries(payload.datasets || {})) {
      out[path] = entry.state;
      Object.assign(out, entry.artifacts || {});
    }
  } else {
    Object.assign(out, payload.sources || {});
  }
  return out;
}

// Clears each justClicked entry whose artifact has either changed verdict
// since the click, or sat unchanged past MIN_CLICKED_MS -- see justClicked's
// own comment for the full rule. A path missing from this poll entirely
// (the artifact no longer shows up in this view) clears too -- nothing
// left to compare against.
function reconcileJustClicked(view, payload) {
  if (justClicked.size === 0) return;
  const states = statesByPath(view, payload);
  const now = Date.now();
  for (const [path, { since, verdict: clickedVerdict }] of justClicked) {
    const state = states[path];
    const changed = state && verdict(state) !== clickedVerdict;
    const expired = now - since >= MIN_CLICKED_MS;
    if (!state || changed || expired) justClicked.delete(path);
  }
}

// One fetcher for all three table views -- they're all slices of the one
// /state payload now (main.py's read_state reads the whole volume once and
// cuts it three ways), so there's one URL here regardless of `view`.
// Switching views no longer waits on a fresh round trip either: route()
// redraws from `lastPayload` immediately, before this poll's fetch even
// lands (see route() below) -- this call exists to keep that payload
// fresh, not to fetch it for the first time on every nav click.
//
// `token` guards against a stale poll clobbering a newer view's render: if
// you navigate away from "sources" to "datasets" while a "sources" poll is
// still in flight (more likely right after a launch, since a job actively
// writing to the volume slows every volume.reload()-backed endpoint), that
// old fetch resolves *after* route() has already switched the page over --
// without this check it would silently overwrite the datasets view back to
// sources, with the nav still showing "datasets" as active. route() bumps
// routeToken on every navigation and hands this call the value current at
// the time it was scheduled; a mismatch by the time the fetch resolves
// means a newer navigation has since taken over, so the result is just
// discarded rather than applied.
async function pollTableView(view, token) {
  try {
    // Same origin as this page — no URL to configure, no CORS to satisfy.
    const res = await fetch("state", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    if (token !== routeToken) return; // superseded by a newer navigation
    reconcileJustClicked(view, payload);
    lastPayload = payload;
    lastView = view;
    draw(view, lastPayload);
    setConn("live", "ok");
  } catch (err) {
    if (token !== routeToken) return;
    // Keep the last good screen up: one missed poll is not news, and blanking
    // the page on a blip hides the state someone is watching.
    setConn(String(err.message || err), "bad");
  }
}

// --- log view poll -----------------------------------------------------------

// The log view's own state: the last payload/opts it rendered (so a filter
// toggle can redraw instantly, without waiting on the next poll), and which
// levels are currently unchecked. logview.js stays a pure render function --
// this is the app-state half of that split, same as `openGroups` is for
// the dashboard's group toggles.
let lastLogPayload = null;
let lastLogOpts = null;
const hiddenLevels = new Set();
let lastLogScope = null; // for detecting a genuine scope/id change vs. just a call filter
let lastLogId = null;

function toggleLevel(level) {
  if (hiddenLevels.has(level)) hiddenLevels.delete(level);
  else hiddenLevels.add(level);
  redrawLogView();
}

function redrawLogView() {
  if (lastLogPayload) renderLogView(logViewEl, lastLogPayload, lastLogOpts);
}

// One fetcher, parameterized by scope, rather than two near-duplicates --
// "run" and "artifact" differ only in which URL and which id renderLogView
// gets, never in how the fetch/error/render sequence goes. callId narrows an
// "artifact" fetch server-side to just that one call's log.
//
// An "artifact" scope also fetches its manifest summary (type, parameters,
// dependency links -- see main.py's manifest_endpoint) to show above the
// log table. That fetch gets its own try/catch: a manifest hiccup (or an
// artifact that's declared but not built yet, so it 404s-as-error) should
// never blank out logs that did load.
//
// `token` guards against a stale poll the same way pollTableView's does --
// see its comment for why this matters, same race, same fix.
async function pollLogView(scope, id, callId, token) {
  try {
    let url = scope === "run" ? `logs/run/${id}` : `logs/artifact/${id}`;
    if (scope === "artifact" && callId) url += `?call_id=${encodeURIComponent(callId)}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();

    let manifest = null;
    if (scope === "artifact") {
      try {
        const mRes = await fetch(`manifest/${id}`, { cache: "no-store" });
        manifest = mRes.ok ? await mRes.json() : null;
      } catch {
        manifest = null;
      }
    }

    if (token !== routeToken) return; // superseded by a newer navigation
    lastLogPayload = payload;
    lastLogOpts = { scope, id, callId: callId || null, hiddenLevels, onToggleLevel: toggleLevel, manifest };
    renderLogView(logViewEl, lastLogPayload, lastLogOpts);
    setConn("live", "ok");
  } catch (err) {
    if (token !== routeToken) return;
    setConn(String(err.message || err), "bad");
  }
}

// --- routing ------------------------------------------------------------------

// Five routes, all client-side: the three table views (TABLE_VIEWS -- "" is
// an alias for "runs", the original default), "run/<id>", and
// "artifact/<path>" (optionally "?call=<call_id>", parsed off the hash's own
// query string to narrow it to one call). No encodeURIComponent on id/path
// when building or reading these -- an artifact_path's /s are meant to stay
// literal, the same convention launchJob already follows for the POST
// /launch route; callId, arriving as a query value rather than a path
// segment, is encoded like any other query value.
function parseRoute() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("run/")) return { view: "run", id: hash.slice(4), callId: null };
  if (hash.startsWith("artifact/")) {
    const rest = hash.slice(9);
    const q = rest.indexOf("?");
    if (q === -1) return { view: "artifact", id: rest, callId: null };
    return {
      view: "artifact",
      id: rest.slice(0, q),
      callId: new URLSearchParams(rest.slice(q + 1)).get("call"),
    };
  }
  if (TABLE_VIEWS.includes(hash)) return { view: hash, id: null, callId: null };
  return { view: "runs", id: null, callId: null };
}

// Bold whichever nav link matches the current table view; no-op (nothing
// matches) while a run/artifact log view is showing, which correctly
// leaves all three unbolded rather than guessing an owner.
function updateNavActive(view) {
  for (const a of navEl.querySelectorAll("a[data-view]")) {
    a.classList.toggle("active", a.dataset.view === view);
  }
}

let pollTimer = null; // current setTimeout id, so route() can cancel a not-yet-fired reschedule

// Bumped on every route() call; handed to pollTableView/pollLogView as
// `token` so a poll scheduled under an earlier route can tell, once its
// fetch finally resolves, whether it's still the current one -- see
// pollTableView's own comment for the race this closes. Also doubles as
// schedulePoll's cancellation signal (below).
let routeToken = 0;

// Runs `tick` now, then schedules the next run EVERY_MS after this one
// *finishes* -- not a blind setInterval, which fires on a fixed clock
// regardless of whether the previous call ever returned. A slow response
// (the volume genuinely can take a moment to reload while a job is
// actively writing to it) would otherwise pile up overlapping requests
// faster than they resolve, which is exactly the "lots of GET requests"
// symptom this replaces. Checks `token` against `routeToken` both before
// running and after `tick` resolves, so a poll loop for a view the user
// has since navigated away from stops rescheduling itself rather than
// quietly polling in the background forever.
function schedulePoll(tick, token) {
  clearTimeout(pollTimer);
  const run = async () => {
    if (token !== routeToken) return;
    await tick();
    if (token !== routeToken) return;
    pollTimer = setTimeout(run, EVERY_MS);
  };
  run();
}

// Restarts whichever poll loop route() last set up -- same tick function,
// same token (the route hasn't changed, just whether the tab watching it
// is visible), just kicked off again right now instead of waiting for its
// next scheduled setTimeout. Used by the visibilitychange listener below.
let currentPoll = null;

// Re-entered on every hashchange, and once at load. Owns the one active
// poll loop: switching routes cancels whichever one was running before
// (via the token check in schedulePoll -- clearTimeout alone can't stop a
// fetch already in flight), so navigating away from a run's log view
// doesn't leave it quietly polling in the background.
function route() {
  const token = ++routeToken;
  const r = parseRoute();
  updateNavActive(r.view);

  if (TABLE_VIEWS.includes(r.view)) {
    titleEl.textContent = r.view;
    dashboardEl.hidden = false;
    logViewEl.hidden = true;
    lastLogScope = null;
    lastLogId = null;
    lastView = r.view;
    // All three table views are slices of the same payload (see
    // pollTableView) -- if we already have one, draw it now instead of
    // waiting on a fresh fetch, so switching views is instant. The poll
    // below still runs to keep it current; this just removes the round
    // trip that used to sit between every nav click and a rendered table.
    if (lastPayload) draw(r.view, lastPayload);
    currentPoll = () => schedulePoll(() => pollTableView(r.view, token), token);
    currentPoll();
    return;
  }

  // Reset the level filter on a genuine scope/id change, but not when only
  // the call filter changed -- narrowing the same artifact's log to one call
  // shouldn't silently un-hide levels already hidden. DEBUG starts hidden
  // by default (LOGGING.md) -- it's protocol chatter, not what you open a
  // log for -- but its checkbox still renders whenever DEBUG entries are
  // present, so turning it back on is one click.
  if (r.view !== lastLogScope || r.id !== lastLogId) {
    hiddenLevels.clear();
    hiddenLevels.add("DEBUG");
    lastLogScope = r.view;
    lastLogId = r.id;
  }

  titleEl.textContent = r.view === "run" ? "run " + r.id : r.id;
  dashboardEl.hidden = true;
  logViewEl.hidden = false;
  currentPoll = () => schedulePoll(() => pollLogView(r.view, r.id, r.callId, token), token);
  currentPoll();
}

// A backgrounded browser tab gets its timers throttled hard (Chrome can
// clamp setTimeout to roughly once a minute after a while) -- switching
// back to this tab would otherwise show whatever was last polled before
// that throttling kicked in, stale by however long the tab sat in the
// background, until the throttled timer eventually fires on its own. This
// polls immediately on return instead of waiting that out. Restarting
// `currentPoll` (rather than just calling the fetch once) also replaces
// the still-pending throttled setTimeout with a fresh one on the normal
// cadence, so the tab doesn't stay on a stretched-out schedule afterward.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && currentPoll) currentPoll();
});

window.addEventListener("hashchange", route);
route();
