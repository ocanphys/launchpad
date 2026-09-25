// artifactview.js -- the drill-down page for one artifact: what it is, what it
// was built from, its training curves if it has any, and what every call
// that ever worked on it said.
//
// No state and no network of its own: app.js calls
// renderArtifactView(container, entry, page, calls), `entry` being this
// artifact's own entry in the table's state map (or undefined for one with
// no manifest), `page` what the server's disk holds for it
// (`/artifact/<path>`: `manifest`, minus the dependency manifests nested
// in it, and `train`, the leg's step log, only what a worker committed)
// and `calls` every call that has ever worked on it with its log in both
// storages (`livedict`, `volume`) out of the `call_logs` Dict
// (`/logs/<path>`, polled while the page is open; see docs/LOGGING.md).

import { el } from "./el.js";
import { logStream, union } from "./logview.js";

// Whether nested parameter values print indented or on one line. Page
// state rather than DOM state: every fetch rebuilds the view, and the choice
// has to outlive that.
let expandedParams = false;

// A parameter's value: primitives print plain, anything richer (a list, a
// nested config) prints as JSON, compact or indented. Dependency values
// never reach here -- the manifest's own parameters/dependencies split
// keeps those out of `parameters`, and the server drops `dependencies`.
function paramValue(value) {
  if (value === null || typeof value !== "object") return String(value);
  return JSON.stringify(value, null, expandedParams ? 2 : 0);
}

// Type, then one row each for the manifest's commit and allocated
// resources and its own parameters, and one link per direct dependency.
// The rows are the manifest as the disk holds it; a dependency is just its
// artifact_path (the state map's `depends_on`), so drilling further is a
// click to that artifact's own page rather than the whole tree being
// dumped on this one.
function summary(entry, manifest) {
  const { artifact, parameters, ...about } = manifest; // about: commit, allocated_resources
  const params = [...Object.entries(about), ...Object.entries(parameters || {})];
  const deps = entry.depends_on || [];
  const nested = params.some(([, value]) => value !== null && typeof value === "object");
  const list = el("div", { class: "summary-params" });
  const render = () =>
    list.replaceChildren(...params.map(([key, value]) =>
      el("div", { class: "summary-param" },
        el("span", { class: "summary-key", text: key }),
        el("span", { class: "summary-value", text: paramValue(value) }),
      ),
    ));
  render();

  return el("div", { class: "artifact-summary" },
    el("div", { class: "summary-type" },
      artifact.split(".").pop(),
      nested
        ? el("label", { class: "log-filter" },
            el("input", {
              type: "checkbox",
              checked: expandedParams,
              onchange: (event) => { expandedParams = event.target.checked; render(); },
            }),
            "expanded",
          )
        : null,
    ),
    params.length ? list : null,
    deps.length
      ? el("p", { class: "summary-deps" },
          "depends on: ",
          ...deps.flatMap((dep, i) => [
            i > 0 ? ", " : null,
            el("a", { href: "#/artifact/" + dep, text: dep }),
          ]),
        )
      : null,
  );
}

// Which levels the reader has switched off. Page state rather than DOM
// state: every fetch rebuilds the view, and the choice has to outlive that.
const hiddenLevels = new Set(["DEBUG"]);
// Empty, so what the container logged around a call shows with the call's own
// rows until a reader turns it off.
const hiddenSources = new Set();

// Every call's rows in one stream, newest first, so what is happening now is
// at the top while the page polls. A row carries its own call id and its own
// `source` -- worker, launcher, ambient -- so nothing is tagged here. A
// call's last heartbeat is a row too, so where an attempt stopped beating
// reads in sequence with what it last said.
function logRows(calls) {
  return calls
    .flatMap((call) => [
      ...union([call.volume, call.livedict]),
      call.last_heartbeat == null ? null : {
        ts: call.last_heartbeat, level: "INFO", logger: "heartbeat", msg: "last heartbeat",
        call_id: call.call_id, source: "heartbeat",
      },
    ])
    .filter(Boolean);
}

// Every call that has ever worked on this artifact (see launcher/state.py's
// artifact_calls), as one stream with a level filter above it: one
// checkbox per level present, DEBUG off until switched on.
function logs(calls) {
  return logStream(logRows(calls), {
    title: "logs",
    className: "artifact-logs",
    hiddenLevels,
    hiddenSources,
    emptyText: "no call has ever worked on this artifact yet.",
  });
}

// --- training curves ----------------------------------------------------------

// One chart per entry, every metric in it drawn on the same axes: training
// and validation loss share a chart, so the gap between them reads off the
// curves rather than off two charts side by side.
const CHARTS = [["loss", "val_loss"], ["grad_norm"], ["learning_rate"]];
const METRICS = CHARTS.flat();
// One color per attempt, cycling: attempt 1 is always the first color, so a
// leg's history reads the same on every visit.
const ATTEMPT_COLORS = ["#0969da", "#cf222e", "#1a7f37", "#8250df", "#bf8700", "#e16f24"];
const CHART = { width: 305, height: 170 };

// Three significant digits, with the trailing zeros toPrecision keeps dropped.
function fmt(value) {
  return String(Number(Number(value).toPrecision(3)));
}

// uPlot's columnar data off the leg's step rows: `steps` is every step any
// attempt took, `ys[metric][i]` attempt i's value at each of them, null
// where it has no row (before its resume point, after its crash). One x
// for every attempt is what puts a step two attempts both took at one
// place on the chart.
function columns(train) {
  const steps = [...new Set(train.map((r) => r.step))].sort((a, b) => a - b);
  const attempts = [...new Set(train.map((r) => r.attempt))].sort((a, b) => a - b);
  const at = new Map(steps.map((s, i) => [s, i]));
  const ys = Object.fromEntries(METRICS.map((m) => [m, attempts.map(() => new Array(steps.length).fill(null))]));
  for (const row of train) {
    const i = attempts.indexOf(row.attempt), j = at.get(row.step);
    for (const metric of METRICS) ys[metric][i][j] = row[metric] ?? null;
  }
  return { steps, attempts, ys };
}

// One chart's metrics against step, every attempt its own color and every
// metric after the first dashed. Drag to zoom x, double-click to reset,
// click a legend entry to hide that line; the cursor is synced across the
// leg's charts so hovering one reads the step off all three.
function chart(metrics, { steps, attempts, ys }) {
  const node = el("div", { class: "curve" });
  const axis = {
    stroke: getComputedStyle(document.body).color,
    font: "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    grid: { stroke: "rgba(128,128,128,.15)" },
    ticks: { stroke: "rgba(128,128,128,.3)" },
  };
  new uPlot(
    {
      ...CHART,
      title: metrics.join(" / "),
      cursor: { sync: { key: "curves" }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [axis, { ...axis, size: 52, values: (u, vals) => vals.map(fmt) }],
      series: [
        { label: "step" },
        ...metrics.flatMap((metric, m) => attempts.map((a, i) => ({
          label: metrics.length > 1 ? `${metric} ${a}` : `attempt ${a}`,
          stroke: ATTEMPT_COLORS[i % ATTEMPT_COLORS.length],
          dash: m > 0 ? [5, 4] : undefined,
          // validation is measured every val_every steps, so its line has to
          // cross the steps in between rather than break at each of them
          spanGaps: m > 0,
          width: 1.2,
          value: (u, v) => (v == null ? "" : fmt(v)),
        }))),
      ],
    },
    [steps, ...metrics.flatMap((metric) => ys[metric])],
    node,
  );
  return node;
}

// The charts last drawn and the rows they were drawn from. `page.train` is
// one object for as long as the page is open (the log poll redraws around
// it), so the same rows hand back the same nodes and a zoom survives the
// redraw.
let drawnTrain = null, drawnCurves = null;

// The leg's training curves, from the step log's rows (see
// docs/LOGGING.md): one line per metric per attempt, grouped into charts by
// CHARTS. Absent for anything whose job keeps no step log.
function curvesSection(train) {
  if (!train.length) return null;
  if (train !== drawnTrain) {
    const data = columns(train);
    drawnTrain = train;
    drawnCurves = el("div", { class: "artifact-curves" },
      el("div", { class: "summary-type", text: "training" }),
      el("div", { class: "curve-row" }, ...CHARTS.map((metrics) => chart(metrics, data))),
    );
  }
  return drawnCurves;
}

// What the page shows between a click and its first answer: the back link,
// and which artifact is on its way. `route()` calls this the moment it
// switches to an artifact, because `container` still holds the *previous*
// artifact's nodes until a fetch resolves and replaceChildren swaps them --
// and the title has already changed by then, so leaving them up shows one
// artifact's metadata under another's name. Only ever called on navigation,
// never on a refetch: an artifact already on screen keeps what it has until
// its own next answer lands.
export function renderArtifactPending(container, artifactPath) {
  container.replaceChildren(
    el("p", { class: "view-back" }, el("a", { href: "#/", text: "← dashboard" })),
    el("p", { class: "empty", text: "loading " + artifactPath + "…" }),
  );
}

export function renderArtifactView(container, entry, page, calls) {
  const nodes = [el("p", { class: "view-back" }, el("a", { href: "#/", text: "← dashboard" }))];

  // No entry (declared, not built -- a normal state) and a manifest that
  // wouldn't load read the same to a reader of this page: there is nothing
  // to show and the reason is the message.
  if (!entry) {
    nodes.push(el("p", { class: "empty", text: "not built yet -- no manifest" }));
  } else if (entry.error) {
    nodes.push(el("p", { class: "empty", text: entry.error }));
  } else {
    nodes.push(summary(entry, page.manifest));
  }

  // Logs render independently of whether the manifest loaded -- a call can
  // have left output behind even for an artifact that's only declared, not
  // built (a job that ran and failed before writing anything), so an empty
  // manifest is never a reason to hide them.
  nodes.push(curvesSection(page.train), logs(calls));

  container.replaceChildren(...nodes.filter(Boolean)); // curves are null without a step log
}
