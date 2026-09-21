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
  holds mutable state: routing, the poll loop, launching, and the one bit
  of client-only UI state (which button was just clicked). Everything else
  is handed data and gives back DOM.
- **[render.js](../web/render.js)** -- the artifact row and its cells.
- **[artifactview.js](../web/artifactview.js)** -- one artifact's own page:
  its type, parameters and dependency links, its training curves when its
  job keeps a step log (one inline SVG per metric, one line per attempt; the
  rows come out of the `train` Dict, both copies, deduped and bucketed to a
  point budget in the browser), and every call's log.
- **[el.js](../web/el.js)** -- the one DOM-building primitive everything
  else uses (`el(tag, props, ...children)`), so text always goes through
  `textContent`, never `innerHTML`.

## Routing

Hash-based, client-side, two shapes (`app.js`'s `parseRoute`):

- `""` -- the table.
- `#/artifact/<path>` -- one artifact's own page.

`route()` re-runs on every `hashchange` and once at load. It owns the one
active poll loop (see below) and the nav bar's active-link highlighting;
switching routes always tears down whatever polling the previous route had
going, so navigating away never leaves a view quietly polling in the
background.

## The table

One `<table>` in `index.html`, one row per artifact on the volume, by path,
built from `artifactRow`/`actionButton`/`dot` in `render.js`. It comes from
one endpoint, `/state`, backed by one function, `main.py`'s `state()`: one
`volume.reload()`, one lease snapshot, one glob of the manifests, and one
flat `{artifact_path: state}` map. There is no grouping by run, source or
dataset; a manifest `state()` could not read lands in the collapsed
"unreadable manifests" table under it.

## Polling

`EVERY_MS` (2000ms) bounds staleness, not correctness. Two things worth
knowing about how it actually fires (`app.js`):

- **Coming back from an artifact's page doesn't wait on a fetch.**
  `route()` redraws immediately from `lastPayload` (whatever the last poll
  landed) and only then kicks off `pollTable` to keep it current.
- **`schedulePoll` is self-rescheduling, not `setInterval`.** It awaits
  each fetch and only schedules the next one `EVERY_MS` after that one
  *finishes* -- a `setInterval` fires on a fixed clock regardless of
  whether the previous call ever returned, and a slow response (the volume
  can genuinely take a moment to reload while a job is writing to it)
  would otherwise pile up overlapping requests faster than they resolve.
- **`routeToken`, bumped on every navigation, guards against a stale poll
  clobbering a newer view.** If you navigate away while a fetch for the
  old view is still in flight, that fetch's result is checked against the
  token before being applied -- a mismatch means a newer navigation has
  since taken over, so the result is discarded rather than silently
  overwriting what's now on screen.
- **A backgrounded browser tab gets its timers throttled** (Chrome can
  clamp `setTimeout` to roughly once a minute after a while). A
  `visibilitychange` listener re-fires the current poll loop immediately
  when the tab becomes visible again, instead of waiting out however long
  the throttled timer would otherwise take to catch up.

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
there's no client-side tracking of
whether a launch is "in progress" in the sense of waiting for it to
finish. The one piece of local state,
`justClicked`, exists only to stop a double-click before the *next* poll
has had a chance to say anything: a clicked button turns blue and disabled
immediately, and that mark clears the first time either becomes true on a
later poll -- the artifact's own verdict (the same classification that
picks the dot's color) has changed since the click, or `MIN_CLICKED_MS`
(four poll intervals) has passed with no change. The first condition means
"we can see it did something"; the second is a floor so a launch that
silently failed doesn't leave the button frozen forever.

## The artifact page

`#/artifact/<path>` reads its own entry out of the same `/state` map the
table draws from: type, own parameters, and one link per direct dependency.
Dependencies are named, not inlined -- clicking one navigates to *its* page
rather than the whole tree being dumped on one screen. A path with no entry
(declared but not built yet) is a message on the page, not an error state.

Below that, what every call that ever worked on the artifact said, as one
stream in time order, each line tagged with its call (short id, full on
hover) and its level, timestamps as `DD/MM/YY-HH:mm:ss` in the reader's zone
with the full instant on hover. A call's last heartbeat is a line in the
stream too. The heading holds one checkbox per level present; DEBUG starts
off, and the choice survives repolls. Each call arrives with both copies of
its log (`live` and `volume`, see [LOGGING.md](LOGGING.md)) and
`artifactview.js` unions them, one row counted once. The container serving this
opens no log file while it runs: it reloads the volume on a clock, and a
reload cannot run while it has a file open -- see [QUEUES.md](QUEUES.md)
§3.3 for how the previous attempt ended.

## Known inefficiencies

- **The `/launch` route pays a full Modal round-trip per click.**
  `attempt_launch` hops to `declared_artifact` to recompute readiness that
  `leasebook`, already inside a volume-mounted container, could read
  in-process. The separation exists so the CLI entrypoint (`launch_job`,
  which runs outside any container) can share the same code path -- but it
  costs every launch click a cold container's worth of latency for an
  answer `state()` already has.
- **`state()` snapshots every historical beat record on every pass.**
  Nothing prunes old ones from that `modal.Dict`, so the snapshot's cost
  grows with the deployment's total call history, not its current
  active-call count.
