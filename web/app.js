// app.js -- the runtime: app state, fetching, launching, and routing between
// the artifact table and one artifact's own page. This is the only file that
// talks to the network or holds mutable state. Rendering is delegated to
// render.js, artifactview.js and logview.js; DOM building to el.js.
//
// The page mirrors the server and decides nothing. `/state` is the map the
// leasebook container holds in memory (main.py's `latest`), which it
// recomputes when a worker's call exits, when it grants or releases a lease,
// and when the refresh button POSTs /refresh. The page just refetches the
// current view on a clock and redraws what changed.

import { artifactRow, problemRow } from "./render.js";
import { renderArtifactPending, renderArtifactView } from "./artifactview.js";
import { logStream, union } from "./logview.js";

// --- config -----------------------------------------------------------------

// How often the current view refetches. Every route it polls answers out of
// memory or the Dicts; the one that reads a file on the mount
// (`artifact/<path>`) is fetched only when the artifact's entry changed.
const POLL_MS = 2000;

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
const splitEl = document.getElementById("split");
const launcherLogsEl = document.getElementById("launcherLogs");
const launcherLogStatusEl = document.getElementById("launcherLogStatus");

// --- app state --------------------------------------------------------------

let lastPayload = null;
const hiddenLauncherLevels = new Set(["DEBUG"]);
// Empty, so the container's own noise shows with the leasebook's rows until a
// reader turns it off.
const hiddenLauncherSources = new Set();

// Artifact paths the page has asked the server to act on, each with what
// the button says meanwhile ("starting", "stopping"): the click's own
// answer, held only until the fetch that follows the server's (see act).
const pending = new Map();

// What the table on screen was built from -- the server's map and these
// pending marks, as text. A poll that matches it draws nothing, so an idle
// table never rebuilds its rows under the reader.
let drawn = null;

// The artifact page's disk image (`/artifact/<path>`: the manifest and step
// log, a real file read on the server's mount) and the entry it was fetched
// for. Only a change to the artifact's own entry can change it, so that is
// the only thing that refetches it.
let disk = null;

function setRefresh(text, cls) {
  refreshEl.textContent = text;
  refreshEl.className = cls;
}

// --- launching --------------------------------------------------------------

// The two things a row's button can ask for -- launch it, stop it -- are one
// POST to one route named after the ask. The button reads "starting" or
// "stopping" from the click until the server answers; the server recomputes
// its map before it does, so the fetch after it already knows what was
// asked for, whether it was granted or refused.
async function act(route, artifactPath, label) {
  pending.set(artifactPath, label);
  redraw();
  try {
    // No encodeURIComponent -- artifactPath's /s are meant to stay literal,
    // matching the server's {artifact_path:path} routes (a plain path
    // segment can't match a multi-segment path).
    const body = await request(`${route}/${artifactPath}`, { method: "POST" });
    if (body.message) console.info(`${route}:`, body.message);
  } catch (err) {
    console.warn(`${route} failed:`, err);
  }
  pending.delete(artifactPath);
  await load(view);
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
// something focusable needs to survive a redraw. A poll that brings back
// the same map and finds the same pending marks returns without touching
// the DOM at all.
function draw(states) {
  const key = JSON.stringify([states, [...pending]]);
  if (key === drawn) return;
  drawn = key;
  const paths = Object.keys(states).sort();
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

// The bottom frame holds what the container did to the calls it manages as
// well as what it did for itself, so every line names its call; the time
// drops its date, the frame being short rather than wide.
function drawLauncherLogs(logs, emptyText = "no launcher logs yet.") {
  launcherLogsEl.replaceChildren(logStream(logs, {
    title: "launcher logs",
    className: "launcher-logs",
    hiddenLevels: hiddenLauncherLevels,
    hiddenSources: hiddenLauncherSources,
    emptyText,
    compactTime: true,
  }));
}

// The table view: the server's map and the log frame under it, one tick
// apiece. Both answer out of memory and the Dicts, so polling them costs the
// volume nothing. The artifact view fetches neither -- it has its own calls'
// stream, and the frame is not on screen there.
async function fetchTable() {
  const token = routeToken;
  const [states, logs] = await Promise.all([request("state"), request("launcher-logs")]);
  if (token !== routeToken) return;
  lastPayload = states;
  stamp(states);
  draw(states);
  // Every row says who wrote it in its own `source`, so nothing is tagged here.
  drawLauncherLogs(union([logs.volume, logs.livedict]));
  launcherLogStatusEl.textContent = "updates every 2s";
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// One artifact's drill-down, from three places: its entry in the server's
// state map (type, own parameters, what it depends on), `logs/<path>` for
// every call that has ever worked on it, and `artifact/<path>` for its
// manifest and step log as the server's disk holds them -- that last one
// refetched only when the entry it belongs to has changed, since nothing
// else can have changed the files.
async function fetchArtifactView(path) {
  const token = routeToken;
  const [states, logs] = await Promise.all([request("state"), request(`logs/${path}`)]);
  if (token !== routeToken) return;
  lastPayload = states;
  stamp(states);
  const entry = states[path];
  const key = JSON.stringify(entry);
  if (!disk || disk.key !== key) {
    const page = await request(`artifact/${path}`);
    if (token !== routeToken) return;
    disk = { key, page };
  }
  renderArtifactView(artifactViewEl, entry, disk.page, logs.calls);
}

// Refetches the current view every POLL_MS until the route changes: the
// page following the launcher, which is what makes a finished job appear
// without anyone clicking. A tick that fails keeps the last screen up and
// the next one tries again.
async function poll() {
  const token = routeToken;
  while (token === routeToken) {
    await sleep(POLL_MS);
    if (token !== routeToken) return;
    try {
      await view();
    } catch (err) {
      launcherLogStatusEl.textContent = "connection interrupted · retrying…";
      console.warn("poll failed, keeping the last screen:", err);
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

// The header button: the server reloads the volume and recomputes its map,
// then the current view is fetched again. A worker's exit tells the server
// to do this on its own, so what is left for the button is what nothing
// announces -- a manifest declared from the lab.
async function refresh() {
  await request("refresh", { method: "POST" });
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
// points `view` at the new one, fetches it once and starts its poll. The
// old route's loop stops itself on the bumped token.
function route() {
  routeToken++;
  const r = parseRoute();
  document.body.classList.toggle("dashboard-page", r.view === "table");
  disk = null; // another artifact's manifest and step log are not this one's

  if (r.view === "table") {
    titleEl.textContent = "artifacts";
    dashboardEl.hidden = false;
    artifactViewEl.hidden = true;
    // Draw the last payload now rather than waiting on a fresh fetch, so
    // coming back from an artifact's page is instant.
    if (lastPayload) draw(lastPayload);
    view = fetchTable;
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
  poll();
}

// --- layout -------------------------------------------------------------------

// The boundary between the two frames: dragging it writes `--log-height`, the
// grid row the log frame sits in (index.html). The drag listens on the window,
// not on the boundary, because the pointer spends the drag over the frames
// either side and the release can land anywhere -- off the window, or as a
// cancelled pointer. One AbortController unbinds the lot however it ends.
const MIN_LOG_PX = 44; // one row and the status line stay readable
const MIN_PAGE_PX = 140; // the header and a row of the page stay on screen

splitEl.addEventListener("pointerdown", (event) => {
  event.preventDefault(); // no text selection while dragging
  const dragging = new AbortController();
  const { signal } = dragging;
  window.addEventListener("pointermove", (moved) => {
    // A release nothing here saw would leave this bound, and the boundary
    // would follow a bare hover. No button down means the drag is over.
    if (!moved.buttons) return dragging.abort();
    const height = Math.min(Math.max(window.innerHeight - moved.clientY, MIN_LOG_PX), window.innerHeight - MIN_PAGE_PX);
    document.body.style.setProperty("--log-height", `${height}px`);
  }, { signal });
  for (const ended of ["pointerup", "pointercancel", "blur"]) {
    window.addEventListener(ended, () => dragging.abort(), { signal });
  }
});

refreshEl.addEventListener("click", () => load(refresh));
window.addEventListener("hashchange", route);
drawLauncherLogs([], "loading launcher logs…"); // the frame is there before the first answer
route();
