// app.js — the runtime: config, app state, the poll loop, launching, and
// routing between the three table views (runs/datasets/sources) and the log
// viewer. This is the only file that talks to the network or holds mutable
// state. Rendering is delegated to render.js and artifactview.js; DOM building
// to el.js.

import { runRow, problemRow, sourcesRows, datasetRows, verdict } from "./render.js";
import { renderArtifactView } from "./artifactview.js";

// --- config -----------------------------------------------------------------

// How often to re-ask the server. This only bounds how stale the screen can
// look; it is not a correctness knob.
const EVERY_MS = 2000;

// How long a just-clicked action button stays blue+disabled at minimum, in
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
const artifactViewEl = document.getElementById("artifactView");

// The three table views the nav bar and pollTableView both know about --
// "runs" is the default/existing one, "datasets" and "sources" are its
// siblings, all three sliced from the one payload main.py's read_state
// returns. Log-view routes ("run", "artifact") are a different family
// entirely, handled below.
const TABLE_VIEWS = ["runs", "datasets", "sources"];

// --- app state --------------------------------------------------------------

let lastPayload = null;
let lastView = "runs"; // which of TABLE_VIEWS lastPayload belongs to

// Artifact paths whose action button was just clicked, each mapped to when
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

// The two things a row's button can ask for -- launch it, stop it -- are one
// POST to one route named after the ask.
//
// Marks the button clicked (blue, disabled) immediately -- snapshotting the
// artifact's verdict as the baseline reconcileJustClicked compares later polls
// against -- then fires the POST. Doesn't wait for the POST before returning,
// and doesn't clear the mark itself either way; that's entirely
// reconcileJustClicked's job, off real poll results, not this call's own
// outcome (a launch can report success and still not actually change anything
// visible for a beat).
async function act(route, artifactPath, state) {
  justClicked.set(artifactPath, { since: Date.now(), verdict: verdict(state) });
  redraw(); // reflect the click immediately, don't wait for the next poll

  try {
    // No encodeURIComponent -- artifactPath's /s are meant to stay literal,
    // matching the server's {artifact_path:path} routes (a plain path
    // segment can't match a multi-segment path).
    const res = await fetch(`${route}/${artifactPath}`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.message || "HTTP " + res.status);
    if (body.message) console.info(`${route}:`, body.message);
  } catch (err) {
    console.warn(`${route} failed:`, err);
  }
}

// Injected into every row so render.js never sees app state directly.
const ctx = {
  onLaunch: (path, state) => act("launch", path, state),
  onCancel: (path, state) => act("cancel", path, state),
  isJustClicked,
  isGroupOpen,
  onToggleGroup: toggleGroup,
};

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

// The `sources` view -- flat, per main.py's read_state, its `sources` key.
function drawSources(payload) {
  const count = Object.keys(payload.sources || {}).length;

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

// --- artifact view poll --------------------------------------------------------

// One artifact's drill-down: its manifest summary, and nothing else. A job's
// own output isn't fetched here or anywhere in this app yet -- the worker
// files it, and `leasebook` opens no log files at all (see artifactview.js).
//
// `token` guards against a stale poll the same way pollTableView's does -- see
// its comment for why this matters, same race, same fix.
async function pollArtifactView(path, token) {
  try {
    const res = await fetch(`manifest/${path}`, { cache: "no-store" });
    const manifest = await res.json();
    if (token !== routeToken) return; // superseded by a newer navigation
    renderArtifactView(artifactViewEl, manifest);
    setConn("live", "ok");
  } catch (err) {
    if (token !== routeToken) return;
    setConn(String(err.message || err), "bad");
  }
}

// --- routing ------------------------------------------------------------------

// Four routes, all client-side: the three table views (TABLE_VIEWS -- "" is
// an alias for "runs", the original default) and "artifact/<path>". No
// encodeURIComponent on the path when building or reading these -- an
// artifact_path's /s are meant to stay literal, the same convention act()
// already follows for the POST /launch and /cancel routes.
function parseRoute() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("artifact/")) return { view: "artifact", id: hash.slice(9) };
  if (TABLE_VIEWS.includes(hash)) return { view: hash, id: null };
  return { view: "runs", id: null };
}

// Bold whichever nav link matches the current table view; no-op (nothing
// matches) while an artifact's own page is showing, which correctly
// leaves all three unbolded rather than guessing an owner.
function updateNavActive(view) {
  for (const a of navEl.querySelectorAll("a[data-view]")) {
    a.classList.toggle("active", a.dataset.view === view);
  }
}

let pollTimer = null; // current setTimeout id, so route() can cancel a not-yet-fired reschedule

// Bumped on every route() call; handed to pollTableView/pollArtifactView as
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
// fetch already in flight), so navigating away from an artifact's page
// doesn't leave it quietly polling in the background.
function route() {
  const token = ++routeToken;
  const r = parseRoute();
  updateNavActive(r.view);

  if (TABLE_VIEWS.includes(r.view)) {
    titleEl.textContent = r.view;
    dashboardEl.hidden = false;
    artifactViewEl.hidden = true;
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

  titleEl.textContent = r.id;
  dashboardEl.hidden = true;
  artifactViewEl.hidden = false;
  currentPoll = () => schedulePoll(() => pollArtifactView(r.id, token), token);
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
