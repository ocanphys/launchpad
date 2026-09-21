# The dashboard

## Mechanism

No build step, no bundler, no framework. `web/index.html` loads
`web/app.js` as an ES module; everything else is `import`ed from there.
Served as static files by the same ASGI app that answers the API
(`main.py`'s `leasebook`), so the page's own fetches (`state`,
`launch/...`) are same-origin relative paths -- no URL to configure, no
CORS.

Three files split by responsibility, each a pure function of its inputs
with no state of its own:

- **[app.js](../web/app.js)** -- the only file that touches the network or
  holds mutable state: routing, fetching, launching, and the one bit of
  client-only UI state (which buttons have a POST in flight). Everything else
  is handed data and gives back DOM.
- **[render.js](../web/render.js)** -- the artifact row and its cells.
- **[artifactview.js](../web/artifactview.js)** -- one artifact's own page:
  its type, parameters and dependency links, its training curves when its
  job keeps a step log (one inline SVG per metric, one line per attempt; the
  rows are `train.jsonl` read off the volume, bucketed to a point budget in
  the browser), and every call's log.
- **[el.js](../web/el.js)** -- the one DOM-building primitive everything
  else uses (`el(tag, props, ...children)`), so text always goes through
  `textContent`, never `innerHTML`.

## Routing

Hash-based, client-side, two shapes (`app.js`'s `parseRoute`):

- `""` -- the table.
- `#/artifact/<path>` -- one artifact's own page.

`route()` re-runs on every `hashchange` and once at load: it switches the
page over, points `refresh` at the new view and fetches it once.

## The table

One `<table>` in `index.html`, one row per artifact on the volume, by path,
built from `artifactRow`/`actionButton`/`dot` in `render.js`. It comes from
one endpoint, `/state`, backed by one function, `main.py`'s `state()`: one
`volume.reload()`, one lease snapshot, one glob of the manifests, and one
flat `{artifact_path: state}` map. There is no grouping by run, source or
dataset; a manifest `state()` could not read lands in the collapsed
"unreadable manifests" table under it.

## Fetching

`/state` is the map the server computed when its container started; the
page fetches it once per navigation. Only the header's refresh button asks
for anything newer: it POSTs `/refresh`, which makes the server reload the
volume and recompute the map, then fetches the current view again
(`app.js`'s `refresh` and `view`). The button shows how the last fetch
went: green `refresh` after a good answer, the error in red after a bad
one, the last good screen staying up either way. A launch or cancel does
not refresh: the row keeps showing the map until you press the button.

The one clock on the page is an open artifact's log stream. The page
fetches `/artifact/<path>` (the step log as the server's disk holds it)
once, then polls `/logs/<path>` every `LOG_POLL_MS` (2000ms), rebuilding
the page with the same entry and curves and the new stream. The server
answers that route from the Dicts alone, so the poll never touches the
mount. The loop checks `routeToken` before and after each fetch and stops
when the route changes; a refresh on the same page starts a new loop and
the old one stops itself (`logPoll`). A tick that fails keeps the last
stream up and tries again next tick.

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
| is ready | **run** | `POST /launch/<path>` |
| is blocked | *run*, frozen | -- |

`done` is asked first because it's the one answer that can't be undone.
Everything after it is a claim about right now, and a claim can lag -- a
lease outlives its call, and a beat is only as fresh as the last one written
-- so an artifact already on disk never offers to stop anything, whatever
the liveness signals still say. One shape and one slot either way, so
nothing shifts when a row changes hands; only the color says which, matching
the dot palette (green run, red stop).

A button reflects server state (`ready`/`active`/`blocked_by`) directly --
there's no client-side tracking of whether a launch is "in progress" in the
sense of waiting for it to finish. The one piece of local state, `pending`,
remembers what the page has asked since the last refresh: a clicked button
turns blue and disabled and reads `starting` (or `stopping`) from the click
on, and stays that way until the next refresh replaces the map with one
that knows about the request. A request the server refused (`launched`/
`cancelled` false, or an error) clears its entry at once, so the button
goes back to what the map says. After the refresh a call that has been
granted but not yet beaten reads as failed (red) until its first heartbeat.

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
hover) and its level, timestamps as `DD/MM/YY-HH:mm:ss` in the reader's zone
with the full instant on hover. A call's last heartbeat is a line in the
stream too. The heading holds one checkbox per level present; DEBUG starts
off, and the choice survives refetches. Each call arrives with all three
channels of its log (see [LOGGING.md](LOGGING.md)) and `artifactview.js`
unions them, one row counted once. The stream is polled while the page is
open; the curves are the file as the last refresh reloaded it.

## Known inefficiencies

- **The `/launch` route pays a full Modal round-trip per click.**
  `attempt_launch` hops to `declared_artifact` to recompute readiness that
  `leasebook`, already inside a volume-mounted container, could read
  in-process. The separation exists so the CLI entrypoint (`launch_job`,
  which runs outside any container) can share the same code path -- but it
  costs every launch click a cold container's worth of latency for an
  answer `state()` already has.
- **A refresh snapshots every historical beat record.** Nothing prunes
  old ones from that `modal.Dict`, so `state()`'s cost grows with the
  deployment's total call history, not its current active-call count.
