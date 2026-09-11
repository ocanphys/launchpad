// app.js — the runtime: config, app state, the poll loop, launching, and
// routing between the artifact table and one artifact's own page. This is
// the only file that talks to the network or holds mutable state. Rendering
// is delegated to render.js and artifactview.js; DOM building to el.js.

import { artifactRow, problemRow, verdict } from "./render.js";
import { renderArtifactPending, renderArtifactView } from "./artifactview.js";

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
const connEl = document.getElementById("conn");
const metaEl = document.getElementById("meta");
const dashboardEl = document.getElementById("dashboard");
const rowsEl = document.getElementById("rows");
const emptyEl = document.getElementById("empty");
const problemsEl = document.getElementById("problems");
const problemCountEl = document.getElementById("problemCount");
const problemRowsEl = document.getElementById("problemRows");
const artifactViewEl = document.getElementById("artifactView");

// --- app state --------------------------------------------------------------

let lastPayload = null;

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
// See reconcileJustClicked, called from pollTable on every landed poll.
const justClicked = new Map(); // artifactPath -> { since, verdict }

function isJustClicked(artifactPath) {
  return justClicked.has(artifactPath);
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
};

// --- draw -------------------------------------------------------------------

// Every artifact on the volume, one row each, by path. Full rebuild each
// time is correct and cheap at this scale; don't add reconciliation until
// something focusable needs to survive a poll.
function draw(states) {
  const paths = Object.keys(states).sort();

  metaEl.textContent =
    paths.length + (paths.length === 1 ? " artifact" : " artifacts") +
    " · polled " + new Date().toLocaleTimeString();

  rowsEl.replaceChildren(...paths.map((path) => artifactRow(path, states[path], ctx)));
  emptyEl.textContent = "nothing declared on the volume yet.";
  emptyEl.hidden = paths.length > 0;

  // Manifests state() could not read. Only touches #problemRows' children
  // and the count text -- never recreates <details id="problems"> itself,
  // so a poll can't clobber whether the user has it open.
  const problems = paths.filter((path) => states[path].error);
  problemRowsEl.replaceChildren(...problems.map((path) => problemRow(path, states[path])));
  problemCountEl.textContent = problems.length;
  problemsEl.hidden = problems.length === 0;
}

// Redraw from the last good payload (used after a local state change --
// a just-clicked button; see act()).
function redraw() {
  if (lastPayload) draw(lastPayload);
}

// --- table poll -------------------------------------------------------------

// Clears each justClicked entry whose artifact has either changed verdict
// since the click, or sat unchanged past MIN_CLICKED_MS -- see justClicked's
// own comment for the full rule. A path missing from this poll entirely
// clears too -- nothing left to compare against.
function reconcileJustClicked(states) {
  if (justClicked.size === 0) return;
  const now = Date.now();
  for (const [path, { since, verdict: clickedVerdict }] of justClicked) {
    const state = states[path];
    const changed = state && verdict(state) !== clickedVerdict;
    const expired = now - since >= MIN_CLICKED_MS;
    if (!state || changed || expired) justClicked.delete(path);
  }
}

// `token` guards against a stale poll clobbering a newer route's render: a
// fetch still in flight when the user navigates into an artifact's page (more
// likely right after a launch, since a job actively writing to the volume
// slows every volume.reload()-backed endpoint) resolves after route() has
// switched the page over, and would otherwise redraw the table underneath
// it. route() bumps routeToken on every navigation and hands this call the
// value current when it was scheduled; a mismatch by the time the fetch
// resolves means a newer navigation has taken over, so the result is
// discarded rather than applied.
async function pollTable(token) {
  try {
    // Same origin as this page — no URL to configure, no CORS to satisfy.
    const res = await fetch("state", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    if (token !== routeToken) return; // superseded by a newer navigation
    reconcileJustClicked(payload);
    lastPayload = payload;
    draw(lastPayload);
    setConn("live", "ok");
  } catch (err) {
    if (token !== routeToken) return;
    // Keep the last good screen up: one missed poll is not news, and blanking
    // the page on a blip hides the state someone is watching.
    setConn(String(err.message || err), "bad");
  }
}

// --- artifact view poll --------------------------------------------------------

// One artifact's drill-down: its manifest summary, plus every call that has
// ever worked on it, aggregated -- `logs/artifact/<path>` (see main.py's
// artifact_call_logs), a Dict read same as everywhere else here, never a
// file (see artifactview.js). Two fetches, in parallel: the manifest can
// 404-shaped-error (not built yet) independently of whether any call has
// ever touched this artifact, so one failing is never a reason to hide the
// other.
//
// `token` guards against a stale poll the same way pollTable's does -- see
// its comment for why this matters, same race, same fix.
async function pollArtifactView(path, token) {
  try {
    const [manifestRes, logsRes] = await Promise.all([
      fetch(`manifest/${path}`, { cache: "no-store" }),
      fetch(`logs/artifact/${path}`, { cache: "no-store" }),
    ]);
    const manifest = await manifestRes.json();
    const logs = await logsRes.json();
    if (token !== routeToken) return; // superseded by a newer navigation
    renderArtifactView(artifactViewEl, manifest, logs);
    setConn("live", "ok");
  } catch (err) {
    if (token !== routeToken) return;
    setConn(String(err.message || err), "bad");
  }
}

// --- routing ------------------------------------------------------------------

// Two routes, both client-side: "" (the table) and "artifact/<path>". No
// encodeURIComponent on the path when building or reading these -- an
// artifact_path's /s are meant to stay literal, the same convention act()
// already follows for the POST /launch and /cancel routes.
function parseRoute() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("artifact/")) return { view: "artifact", id: hash.slice(9) };
  return { view: "table", id: null };
}

let pollTimer = null; // current setTimeout id, so route() can cancel a not-yet-fired reschedule

// Bumped on every route() call; handed to pollTable/pollArtifactView as
// `token` so a poll scheduled under an earlier route can tell, once its
// fetch finally resolves, whether it's still the current one -- see
// pollTable's own comment for the race this closes. Also doubles as
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

  if (r.view === "table") {
    titleEl.textContent = "artifacts";
    dashboardEl.hidden = false;
    artifactViewEl.hidden = true;
    // Draw the last payload now rather than waiting on a fresh fetch, so
    // coming back from an artifact's page is instant; the poll below keeps
    // it current.
    if (lastPayload) draw(lastPayload);
    currentPoll = () => schedulePoll(() => pollTable(token), token);
    currentPoll();
    return;
  }

  titleEl.textContent = r.id;
  dashboardEl.hidden = true;
  artifactViewEl.hidden = false;
  // Clear whatever artifact was on screen before this one. The table branch
  // above can draw its last payload while it waits, because that payload is
  // this view's own data one poll old; here it would be a *different*
  // artifact's, sitting under the name of the one just clicked.
  renderArtifactPending(artifactViewEl, r.id);
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
