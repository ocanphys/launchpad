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
  holds mutable state: routing, the poll loop, launching, and the small
  bits of client-only UI state (which groups are open, which button was
  just clicked). Everything else is handed data and gives back DOM.
- **[render.js](../web/render.js)** -- the three table views (runs,
  datasets, sources) and the artifact/group row components they share.
- **[artifactview.js](../web/artifactview.js)** -- one artifact's own page:
  its type, parameters and dependency links.
- **[el.js](../web/el.js)** -- the one DOM-building primitive everything
  else uses (`el(tag, props, ...children)`), so text always goes through
  `textContent`, never `innerHTML`.

## Routing

Hash-based, client-side, four shapes (`app.js`'s `parseRoute`):

- `#/runs`, `#/datasets`, `#/sources` -- the three table views (`""` is an
  alias for `runs`).
- `#/artifact/<path>` -- one artifact's own page.

`route()` re-runs on every `hashchange` and once at load. It owns the one
active poll loop (see below) and the nav bar's active-link highlighting;
switching routes always tears down whatever polling the previous route had
going, so navigating away never leaves a view quietly polling in the
background.

## The three table views

All three reuse the same `<table>` markup in `index.html` and the same
`artifactRow`/`actionButton`/`dot` components in `render.js` -- only how
each is sliced out of the payload and grouped differs. All three come from
one endpoint, `/state`, backed by one function, `main.py`'s `read_state`:
it reads the volume once (one `volume.reload()`, one lease snapshot, and
one `core_resolve.resolve` over every declared artifact on the volume) and
computes each distinct artifact's state at most once, however many of the
three views reference it, rather than each view re-inspecting shared
artifacts (a source behind a dozen tokenizers, say) independently. The
single walk is what deduplicates: a tokenizer reachable from three runs and
two datasets is one node in one graph, visited and inspected once.

| view | payload slice | shape | grouping |
|---|---|---|---|
| `runs` | `runs` | `{run_id: {artifacts}}` | per run, by artifact type |
| `datasets` | `datasets` | `{path: {state, artifacts}}` | per dataset, by its dependency closure's type |
| `sources` | `sources` | `{path: state}` | none -- flat |

A run's own artifacts and a dataset's dependency closure are both
type-grouped the same way, via one shared function, `typeGroupedRows`: a
type with more than one member collapses behind a summary row (closed by
default; open/closed state lives in `app.js`'s `openGroups`, namespaced so
two different owners' same-named groups never collide); a type with just
one member renders directly. Groups, and members within a group, are
ordered by topological depth (`render.js`'s `topoDepth`, walking each
artifact's own `depends_on` list), deepest first -- what has to be built
before anything else (a Source, a Tokenizer) sinks to the bottom, what
depends on everything above it floats to the top, so a run reads top to
bottom the way it was built bottom to top. Nesting depth
(`artifactRow`/`groupHeaderRow`'s `depth` parameter, unrelated to
`topoDepth`) controls indentation -- a dataset's dependency rows sit one
level deeper than a run's own, since they're nested under the dataset's
row rather than a run header.

## Polling

`EVERY_MS` (2000ms) bounds staleness, not correctness. Two things worth
knowing about how it actually fires (`app.js`):

- **Switching between the three table views doesn't wait on a fetch.**
  Since all three come from the one `/state` payload, `route()` redraws
  immediately from `lastPayload` (whatever the last poll landed, for
  whichever view) the moment you navigate, and only then kicks off
  `pollTableView` to keep that payload current. The old per-view endpoints
  meant every nav click paid a fresh round trip before anything on screen
  changed; a single shared payload removes that wait entirely.
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

`#/artifact/<path>` fetches one thing, `/manifest/<path>`
(`main.py`'s `artifact_manifest_summary`): type, own parameters, and one link
per direct dependency. Dependencies are named, not inlined -- clicking one
navigates to *its* page rather than the whole tree being dumped on one screen.
A missing manifest (declared but not built yet) is a message on the page, not
an error state.

What a job actually said isn't here yet. The worker files it in two places
(see [LOGGING.md](LOGGING.md)), but this container reads neither: it is the
one that reloads the volume on a clock, and a reload cannot run while it has a
file open -- see [QUEUES.md](QUEUES.md) §3.3 for how the previous attempt
ended. When a log page comes, it reads the Dict entry and never the file.

## Known inefficiencies

- **The `/launch` route pays a full Modal round-trip per click.**
  `attempt_launch` hops to `declared_artifact` to recompute readiness that
  `leasebook`, already inside a volume-mounted container, could read
  in-process. The separation exists so the CLI entrypoint (`launch_job`,
  which runs outside any container) can share the same code path -- but it
  costs every launch click a cold container's worth of latency for an
  answer `read_state` already has.
- **`/state`'s `beats` output is computed and serialized every poll and
  nothing reads it.** `read_state()` builds `held_by` and `beats_out` by
  iterating every historical beat record -- and since nothing prunes old
  ones from that `modal.Dict`, the cost grows with the deployment's total
  call history, not its current active-call count, on every single poll.
  `leases` is in the same position: serialized every poll, read by nothing.
- **`/manifest/<path>` is the last request that touches the mount.** One
  small read, against a volume the refresh thread reloads on its own clock
  (see the README). Small enough to live with; the way to close it for good
  is to fold the summary into what that thread already computes, since it has
  every artifact resolved in front of it already.
