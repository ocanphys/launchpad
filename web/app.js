// app.js -- the runtime: app state, fetching, launching, and routing between
// the artifact table and one artifact's own page. This is the only file that
// talks to the network or holds mutable state. Rendering is delegated to
// render.js, artifactview.js and logview.js; DOM building to el.js.
//
// The server serves `/state` out of the map it holds in memory (main.py's
// `latest`); only the refresh button makes it reload the volume and compute
// a new one (POST /refresh), after which the current view is fetched again.
// Only logs are on a clock: the dashboard's launcher stream or an open
// artifact's stream. Both routes read the Dicts alone.

import { artifactRow, problemRow } from "./render.js";
import { renderArtifactPending, renderArtifactView } from "./artifactview.js";
import { logStream, union } from "./logview.js";

// --- config -----------------------------------------------------------------

// How often the current view refetches its log stream. Artifact state
// still changes only on refresh.
const LOG_POLL_MS = 2000;

// --- dom refs ---------------------------------------------------------------

const titleEl = document.getElementById("title");
const refreshEl = document.getElementById("refresh");
const metaEl = document.getElementById("meta");
const dashboardEl = document.getElementById("dashboard");
const rowsEl = document.getElementById("rows");
const emptyEl = document.getElementById("empty");
const problemsEl = document.getElementById("problems");
const problemCountEl = document.getElementById("problemCount");
const problemRowsEl = document.getElementById("problemRows");
const artifactViewEl = document.getElementById("artifactView");
const launcherLogsEl = document.getElementById("launcherLogs");
const launcherLogStatusEl = document.getElementById("launcherLogStatus");

// --- app state --------------------------------------------------------------

let lastPayload = null;
const hiddenLauncherLevels = new Set(["DEBUG"]);

// Artifact paths the page has asked the server to act on, each with what
// the button says meanwhile ("starting", "stopping"). The server's map does
// not know about the request until a refresh recomputes it, so the page
// remembers instead: the button stays blue, disabled and labelled until the
// refresh that follows the server's answer (see act).
const pending = new Map();

function setRefresh(text, cls) {
  refreshEl.textContent = text;
  refreshEl.className = cls;
}

// --- launching --------------------------------------------------------------

// The two things a row's button can ask for -- launch it, stop it -- are one
// POST to one route named after the ask. The button reads "starting" or
// "stopping" from the click until the server answers. An accepted request
// is followed by a refresh, so the button then shows what the map knows: a
// launched call keeps "starting" while it waits for its first heartbeat
// within the startup grace period. A refused one puts the button back.
async function act(route, artifactPath, label) {
  pending.set(artifactPath, label);
  redraw();
  let accepted = false;
  try {
    // No encodeURIComponent -- artifactPath's /s are meant to stay literal,
    // matching the server's {artifact_path:path} routes (a plain path
    // segment can't match a multi-segment path).
    const body = await request(`${route}/${artifactPath}`, { method: "POST" });
    if (body.message) console.info(`${route}:`, body.message);
    accepted = Boolean(body.launched || body.cancelled);
  } catch (err) {
    console.warn(`${route} failed:`, err);
  }
  if (accepted) await load(refresh);
  else {
    pending.delete(artifactPath);
    redraw();
  }
}

// Injected into every row so render.js never sees app state directly.
const ctx = {
  onLaunch: (path) => act("launch", path, "starting"),
  onCancel: (path) => act("cancel", path, "stopping"),
  pending: (path) => pending.get(path),
};

// --- draw -------------------------------------------------------------------

// The line under the header: how many artifacts the server's map holds
// and when this page last fetched. Written on every fetch of either view,
// so a refresh visibly lands on the artifact page too.
function stamp(states) {
  const n = Object.keys(states).length;
  metaEl.textContent = n + (n === 1 ? " artifact" : " artifacts") + " · fetched " + new Date().toLocaleTimeString();
}

// Every artifact on the volume, one row each, by path. Full rebuild each
// time is correct and cheap at this scale; don't add reconciliation until
// something focusable needs to survive a redraw.
function draw(states) {
  const paths = Object.keys(states).sort();
  stamp(states);
  rowsEl.replaceChildren(...paths.map((path) => artifactRow(path, states[path], ctx)));
  emptyEl.textContent = "nothing declared on the volume yet.";
  emptyEl.hidden = paths.length > 0;

  // Manifests state() could not read. Only touches #problemRows' children
  // and the count text -- never recreates <details id="problems"> itself,
  // so a redraw can't clobber whether the user has it open.
  const problems = paths.filter((path) => states[path].error);
  problemRowsEl.replaceChildren(...problems.map((path) => problemRow(path, states[path])));
  problemCountEl.textContent = problems.length;
  problemsEl.hidden = problems.length === 0;
}

// Redraw from the last good payload (used after a local state change --
// a request made or refused; see act()).
function redraw() {
  if (lastPayload) draw(lastPayload);
}

// --- fetching ---------------------------------------------------------------

// Bumped on every navigation. A fetch checks it when its answer lands, so a
// slow answer for a view the user has since left (a refresh while a job is
// writing to the volume can take a moment) is dropped rather than drawn
// under the newer view.
let routeToken = 0;

// Same origin as this page -- no URL to configure, no CORS to satisfy. A
// failed request throws with the server's message when it gave one.
async function request(route, init = {}) {
  const res = await fetch(route, { cache: "no-store", ...init });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || "HTTP " + res.status);
  return body;
}

async function fetchTable() {
  const token = routeToken;
  const payload = await request("state");
  if (token !== routeToken) return;
  lastPayload = payload;
  draw(payload);
}

function drawLauncherLogs(logs, emptyText = "no launcher logs yet.") {
  launcherLogsEl.replaceChildren(logStream(logs, {
    title: "launcher logs",
    className: "launcher-logs",
    hiddenLevels: hiddenLauncherLevels,
    emptyText,
  }));
}

// Independent of table fetches and volume refreshes: start immediately
// when the dashboard opens and keep the last stream through failures.
// Navigation invalidates both a sleeping loop and an in-flight answer.
async function pollLauncherLogs() {
  const token = routeToken;
  if (!launcherLogsEl.hasChildNodes()) drawLauncherLogs([], "loading launcher logs…");
  while (token === routeToken) {
    try {
      const { launcher, volume } = await request("launcher-logs");
      if (token !== routeToken) return;
      drawLauncherLogs(union([volume, launcher]));
      launcherLogStatusEl.textContent = "updates every 2s";
    } catch (err) {
      if (token !== routeToken) return;
      launcherLogStatusEl.textContent = "connection interrupted · retrying…";
      console.warn("launcher log poll failed, keeping the last stream:", err);
    }
    await sleep(LOG_POLL_MS);
  }
}

// Bumped whenever an artifact page's log poll starts, so the loop it
// replaces (a refresh on the same page starts a new one) stops itself.
let logPoll = 0;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// One artifact's drill-down, from three places: its entry in the table's
// state map (type, own parameters, what it depends on; fetched only when
// this page has no map yet -- the first thing loaded, or right after a
// refresh), `artifact/<path>` for its manifest and step log as the
// server's disk holds them, and `logs/<path>` for every call that has ever
// worked on it. Renders,
// then hands the log stream to `pollLogs` and returns.
async function fetchArtifactView(path) {
  const token = routeToken;
  if (!lastPayload) lastPayload = await request("state");
  const [page, logs] = await Promise.all([request(`artifact/${path}`), request(`logs/${path}`)]);
  if (token !== routeToken) return;
  stamp(lastPayload);
  const entry = lastPayload[path];
  renderArtifactView(artifactViewEl, entry, page, logs.calls);
  pollLogs(path, entry, page); // started, not awaited: this fetch is done
}

// Refetches `path`'s log stream every LOG_POLL_MS and redraws the page
// around the entry and disk image it was given, until the route changes
// or another poll for the page starts. A tick that fails keeps the last
// stream up; the next one tries again.
async function pollLogs(path, entry, page) {
  const token = routeToken;
  const mine = ++logPoll;
  while (true) {
    await sleep(LOG_POLL_MS);
    if (token !== routeToken || mine !== logPoll) return;
    try {
      const { calls } = await request(`logs/${path}`);
      if (token !== routeToken || mine !== logPoll) return;
      renderArtifactView(artifactViewEl, entry, page, calls);
    } catch (err) {
      console.warn("log poll failed, keeping the last stream:", err);
    }
  }
}

// Runs `work` with the refresh button showing how it went: "refresh"
// (green) after a good answer, the error (red) after a bad one, the last
// good screen staying up either way.
async function load(work) {
  const token = routeToken;
  refreshEl.disabled = true;
  setRefresh("loading…", "");
  try {
    await work();
    if (token === routeToken) setRefresh("refresh", "ok");
  } catch (err) {
    if (token === routeToken) setRefresh(String(err.message || err), "bad");
  } finally {
    refreshEl.disabled = false;
  }
}

// Fetches and draws the current view. Set by route().
let view = async () => {};

// The server reloads the volume and recomputes its map, then the current
// view is fetched again. The one way the page ever asks the server to look
// at the volume: from the header button, or from act() once a request lands.
async function refresh() {
  await request("refresh", { method: "POST" });
  lastPayload = null;
  pending.clear(); // the new map knows what was asked; the buttons read it
  await view();
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

// Re-entered on every hashchange, and once at load: switches the page over,
// points `view` at the new one and fetches it once.
function route() {
  routeToken++;
  const r = parseRoute();
  document.body.classList.toggle("dashboard-page", r.view === "table");

  if (r.view === "table") {
    titleEl.textContent = "artifacts";
    dashboardEl.hidden = false;
    artifactViewEl.hidden = true;
    // Draw the last payload now rather than waiting on a fresh fetch, so
    // coming back from an artifact's page is instant.
    if (lastPayload) draw(lastPayload);
    view = fetchTable;
    pollLauncherLogs(); // its own loop: a table refresh never restarts it
  } else {
    titleEl.textContent = r.id;
    dashboardEl.hidden = true;
    artifactViewEl.hidden = false;
    // Clear whatever artifact was on screen before this one. The table branch
    // above can draw its last payload while it waits, because that payload is
    // this view's own data one fetch old; here it would be a *different*
    // artifact's, sitting under the name of the one just clicked.
    renderArtifactPending(artifactViewEl, r.id);
    view = () => fetchArtifactView(r.id);
  }
  load(view);
}

refreshEl.addEventListener("click", () => load(refresh));
window.addEventListener("hashchange", route);
route();
