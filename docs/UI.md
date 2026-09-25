# The dashboard

## Mechanism

No build step, no bundler, no framework. `web/index.html` loads
`web/app.js` as an ES module; everything else is `import`ed from there.
Served as static files by the same ASGI app that answers the API
(`main.py`'s `leasebook`), so the page's own fetches (`state`,
`launch/...`) are same-origin relative paths -- no URL to configure, no
CORS.

Files split by responsibility:

- **[app.js](../web/app.js)** -- the only file that touches the network:
  routing, fetching, launching, log polling, pending buttons and launcher
  log filter choices. Rendering modules receive data and give back DOM.
- **[render.js](../web/render.js)** -- the artifact row and its cells.
- **[artifactview.js](../web/artifactview.js)** -- one artifact's own page:
  its type, parameters and dependency links, its training curves when its
  job keeps a step log (one uPlot chart per metric, loss and validation loss
  sharing one with validation dashed, one line per metric per attempt, every
  row of `train.jsonl` as read off the volume; drag to zoom, click a legend
  entry to hide a line), and every call's log. It retains the
  reader's parameter and artifact log filter choices between redraws.
- **[logview.js](../web/logview.js)** -- shared timestamp formatting, level
  filters and safe log rows for artifact streams and the launcher panel.
- **[el.js](../web/el.js)** -- the one DOM-building primitive everything
  else uses (`el(tag, props, ...children)`), so text always goes through
  `textContent`, never `innerHTML`.

## Routing

Hash-based, client-side, two shapes (`app.js`'s `parseRoute`):

- `""` -- the table.
- `#/artifact/<path>` -- one artifact's own page.

`route()` re-runs on every `hashchange` and once at load: it switches the
page over, points `view` at the new one, fetches it once and starts its
poll.

## The table

One `<table>` in `index.html`, one row per artifact on the volume, by path,
built from `artifactRow`/`actionButton`/`dot` in `render.js`. It comes from
one endpoint, `/state`, backed by one function, `main.py`'s `state()`: one
`volume.reload()`, one lease snapshot, one glob of the manifests, and one
flat `{artifact_path: state}` map. There is no grouping by run, source or
dataset; a manifest `state()` could not read lands in the collapsed
"unreadable manifests" table under it.

Four columns: the artifact (its status dot and type), its path, its
progress, and the button. Which call holds it and when that call last beat
are the **hover of the artifact's label** (`callSummary` in `render.js`),
not columns of their own: both are long, and neither is what the table is
read for at a glance. Progress is the only column with no width of its own,
so it takes whatever the others leave, being the one that gets long; what
still overruns it truncates with the whole of it on hover, like the path,
rather than wrapping a row taller than its neighbours.

The dashboard is **two frames**: the table above, the leasebook container's
**launcher logs** below, and a boundary between them that drags (`#split`,
app.js's splitter, writing the `--log-height` the bottom row reads). The log
takes a fifth of the window until someone drags it; each frame scrolls on
its own, and a drag lasts as long as the page does, since nothing stores it.
An artifact's page carries its own calls' stream, so the bottom frame is not
on screen there and the page takes the whole window.

Launcher records are one line each: time, call, source, severity, message,
newest first, with the logger name on the message's hover. Its time drops
the date the artifact stream keeps, the frame being short rather than wide,
and its column labels stay pinned to the top of its scroll area.
Multiline tracebacks preserve their line breaks. Level checkboxes work
like the artifact log filters: DEBUG starts off, and selections survive
polls and navigation. Launcher filters are separate from artifact filters.
The panel is the whole account of what that container did, so it holds what
it did to the calls it manages as well: each line names its call (short id,
`launcher` for the container's own) and its source, which is why a grant here
also reads on that call's artifact page. The noise around them -- uvicorn,
grpc -- is the container's `ambient` source, under the same **ambient** box
the artifact stream has: on by default, and off leaves the leasebook's own
rows.

## Fetching

**The page mirrors the server and decides nothing.** `/state` returns the
map the leasebook container holds, which *it* recomputes when a worker's
call starts or exits (a message on the `launchpad-refreshes` Queue), when
it grants or releases a lease, and when the refresh button POSTs
`/refresh`. The page refetches the current view every `POLL_MS` (2000ms)
and redraws what changed, so a row walks `runnable` -> `starting` ->
`running` -> `done` on its own, a tick behind the server, with nobody
clicking anything.

One loop, `app.js`'s `poll`, calling whatever `view` the route pointed at:

- The table fetches `/state` and `/launcher-logs` per tick. Rows are
  rebuilt only when the map or the page's own pending marks differ from
  what is on screen (`draw` compares them as text), so an idle table never
  rebuilds under the reader -- and a row with a job running rebuilds on
  every tick, because `/state` reads that call's beat off the Dicts per
  request: its progress cell and its label's hover advance between
  recomputes, without anything reloading the volume. The bottom frame unions
  the two storages `/launcher-logs` hands back, `livedict` and `volume`, a
  row counted once (`logview.js`).
- An artifact page fetches `/state` and `/logs/<path>` per tick, and
  `/artifact/<path>` -- the one route that reads a file on the server's
  mount -- only when that artifact's own entry has changed, since nothing
  else can have changed its manifest or step log.

Every other polled route answers out of memory or the Dicts, so polling
costs the volume nothing. A tick that fails keeps the last screen up, shows
a retry message and tries again on the next one. Both the fetch and the
sleeping loop check `routeToken`, so an answer for a view the reader has
left is dropped and the old route's loop stops itself.

The header's refresh button is the manual reload, for what nothing
announces: a manifest declared from the lab. It POSTs `/refresh` and
refetches the view. The button also shows how the last fetch went: green
`refresh` after a good answer, the error in red after a bad one, the last
good screen staying up either way.

- **Coming back from an artifact's page doesn't wait on a fetch.**
  `route()` redraws immediately from `lastPayload` (whatever the last fetch
  landed) and then fetches once to bring it current.
- **`routeToken`, bumped on every navigation, guards against a stale fetch
  clobbering a newer view.** If you navigate away while a fetch for the
  old view is still in flight, that fetch's result is checked against the
  token before being applied -- a mismatch means a newer navigation has
  since taken over, so the result is discarded rather than silently
  overwriting what's now on screen.

## Row buttons

One button per row, offering the one thing worth doing to that artifact
(`render.js`'s `actionButton`). Asked in this order:

| the artifact | button | route |
|---|---|---|
| is done | *run*, frozen | -- |
| has a call running | **stop** | `POST /cancel/<path>` |
| has verdict `starting` | *starting*, blue and disabled | -- |
| is ready | **run** | `POST /launch/<path>` |
| is blocked | *run*, frozen | -- |

`done` is asked first because it's the one answer that can't be undone.
Everything after it is a claim about right now, and a claim can lag -- a
lease outlives its call, and a beat is only as fresh as the last one written
-- so an artifact already on disk never offers to stop anything, whatever
the liveness signals still say. One shape and one slot either way, so
nothing shifts when a row changes hands; only the color says which, matching
the dot palette (green run, red stop, blue starting).

A button reflects server state (`done`/`active`/`verdict`/`ready`/`blocked_by`) directly --
there's no client-side tracking of whether a launch is "in progress" in the
sense of waiting for it to finish. The one piece of local state, `pending`,
covers the round trip and nothing more: a clicked button turns blue and
disabled and reads `starting` (or `stopping`) from the click until the
server answers. The server recomputes its map before answering an accepted
launch or cancel, so the fetch right after already knows what was asked
for, and a refused request lands on a map that says why.

A cancelled row goes back to **run** at once, because the lease it was
holding is what `/cancel` removes and the row's liveness is read off that
grant. The container can still be finishing for seconds afterwards, invisible
here: Modal delivers its cancellation on the container's own heartbeat, and
the call is only stopped for certain at its next lease check. Clicking **run**
in that gap is allowed and grants a second call, which is safe but does mean
two containers briefly working on one artifact (README, `/cancel`).

A call that has been granted but not yet beaten reads as `starting` while
its lease is younger than `STARTUP_GRACE_SECONDS`
(60 seconds in `config.py`) -- what a row shows between the launch and the
call's own "started" message, which is the container booting. The row has a blue dot with a tooltip saying
it is waiting for the first heartbeat, and a blue, disabled `starting`
button even if `ready` is true. At or after that deadline, a call that
still has no heartbeat reads as failed: the dot is red, its tooltip
explains that no first heartbeat arrived, and the button offers `run`
again if the artifact is ready. A heartbeat ends startup: a live one
means `running` and offers `stop`; a stale one means `failed`, as does the
beat a call marks on its way out. These labels are the server's and never
the browser's, but they are not frozen to the recompute: for a call the map
found under way, `/state` reads the grant and the beat again on each
request, so an expired grace period or a call whose beats stopped -- a
container killed hard, which announces nothing -- reaches the row on the
next poll.

## The artifact page

`#/artifact/<path>` shows the artifact's manifest as the server's disk holds
it (`/artifact/<path>`): type, then one row each for its commit, its
allocated resources and its own parameters, and one link per direct
dependency out of its entry in the `/state` map the table already fetched
(fetched only when the page is loaded directly on an artifact URL).
Dependencies are named, not inlined -- clicking one navigates to *its* page
rather than the whole tree being dumped on one screen. A path with no entry
(declared but not built yet) is a message on the page, not an error state.

Below that, what every call that ever worked on the artifact said, as one
stream newest first, each line tagged with its call (short id, full on
hover) and its level under a row of column labels, timestamps as
`DD/MM/YY-HH:mm:ss` in the reader's zone with the full instant on hover. A
call's last heartbeat is a line in the
stream too. The heading holds one checkbox per level present; DEBUG starts
off, and the choice survives refetches. Beside the level boxes is an
**ambient** box, on by default and shown only when the container logged
anything alongside the call: turning it off leaves the call's own account of
itself. Each call arrives with its log in both storages (see
[LOGGING.md](LOGGING.md)) and `artifactview.js` unions them, one row counted
once. A line's `source` says who wrote it: `worker` is the call itself,
`launcher` what the leasebook did to it, `ambient` what the container logged
around it. The stream is polled while the page is
open; the curves are the file as the server's last reload left it, refetched
when the artifact's entry on the state map changes -- which a call's exit
is what causes.

## Known inefficiencies

- **A recompute snapshots every historical beat record.** Nothing prunes
  old ones from that `modal.Dict`, so `state()`'s cost grows with the
  deployment's total call history, not its current active-call count.
