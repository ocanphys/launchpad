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
// (`/logs/<path>`, polled while the page is open; see docs/logging.md).

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

// Type, then one row each for the manifest's declaring commit and allocated
// resources and its own parameters, and one link per direct dependency.
// The rows are the manifest as the disk holds it; a dependency is just its
// artifact_path (the state map's `depends_on`), so drilling further is a
// click to that artifact's own page rather than the whole tree being
// dumped on this one.
function summary(entry, manifest) {
  const { artifact, parameters, ...about } = manifest; // about: declaring_commit, allocated_resources
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
// curves rather than off two charts side by side. A plot is its `metric` and
// any uPlot series option to override, as `spanGaps: true` does for
// validation: measured every val_every steps, its line has to cross the steps
// in between instead of breaking at each of them. A metric in the step log that
// is not named here gets a chart of its own after these, its line crossing gaps
// (see `layoutFor`).
const CHARTS = [
  [{ metric: "loss" }, { metric: "val_loss", spanGaps: true }],
  [{ metric: "grad_norm" }],
  [{ metric: "learning_rate" }],
  [{ metric: "mfu", spanGaps: true }],
  [{ metric: "peak_allocated_pct" }, { metric: "peak_reserved_pct", spanGaps: true }],
];
// Columns of a step row that are not metrics.
const ROW_KEYS = ["step", "attempt"];
// The metrics and the attempts the reader has switched off, the latter in
// every chart at once. Page state rather than DOM state: the charts are
// rebuilt when the step log changes, and the choice has to outlive that.
// Everything starts on.
const hiddenMetrics = new Set();
const hiddenAttempts = new Set();
// One color per attempt, cycling: attempt 1 is always the first color, so a
// leg's history reads the same on every visit.
const ATTEMPT_COLORS = ["#0969da", "#cf222e", "#1a7f37", "#8250df", "#bf8700", "#e16f24"];
const CHART = { width: 305, height: 170 };

// Three significant digits, with the trailing zeros toPrecision keeps dropped.
function fmt(value) {
  return String(Number(Number(value).toPrecision(3)));
}

// The charts to draw for a leg's step rows: CHARTS, then one chart per metric
// the rows hold a number for that CHARTS does not name, by name.
function layoutFor(train) {
  const named = new Set(CHARTS.flat().map((plot) => plot.metric));
  const extra = [...new Set(train.flatMap(Object.keys))]
    .filter((key) => !ROW_KEYS.includes(key) && !named.has(key) && train.some((row) => typeof row[key] === "number"))
    .sort();
  return [...CHARTS, ...extra.map((metric) => [{ metric, spanGaps: true }])];
}

// uPlot's columnar data off the leg's step rows: `steps` is every step any
// attempt took, `ys[metric][i]` attempt i's value at each of them, null
// where it has no row (before its resume point, after its crash). One x
// for every attempt is what puts a step two attempts both took at one
// place on the chart.
function columns(train, metrics) {
  const steps = [...new Set(train.map((r) => r.step))].sort((a, b) => a - b);
  const attempts = [...new Set(train.map((r) => r.attempt))].sort((a, b) => a - b);
  const at = new Map(steps.map((s, i) => [s, i]));
  const ys = Object.fromEntries(metrics.map((m) => [m, attempts.map(() => new Array(steps.length).fill(null))]));
  for (const row of train) {
    const i = attempts.indexOf(row.attempt), j = at.get(row.step);
    for (const metric of metrics) ys[metric][i][j] = row[metric] ?? null;
  }
  return { steps, attempts, ys };
}

// One chart's plots (see CHARTS) against step, every attempt its own color
// and every plot after the first dashed. Returns the chart's node, its uPlot
// and its `syncLegend`, which `showLines` calls. Drag to zoom x, double-click
// to reset, click a legend entry to hide that line; the cursor is synced across
// the leg's charts so hovering one reads the step off all of them.
//
// The legend lists a line only while it is switched on at the top, and while
// the cursor is over a step the line has a value at.
function chart(plots, { steps, attempts, ys }) {
  const metrics = plots.map((plot) => plot.metric);
  const node = el("div", { class: "curve" });
  const lines = plots.flatMap(({ metric }) => attempts.map((attempt) => ({ metric, attempt })));
  // uPlot keeps a legend row per series, the step's first; some charts have no
  // row for the step, so the rows are counted from the end.
  const syncLegend = (u) => {
    const rows = u.root.querySelectorAll(".u-legend .u-series");
    const at = u.cursor.idx;
    lines.forEach(({ metric, attempt }, k) => {
      const row = rows[rows.length - lines.length + k];
      if (!row) return;
      const off = hiddenMetrics.has(metric) || hiddenAttempts.has(attempt);
      row.style.display = off || (at != null && u.data[k + 1][at] == null) ? "none" : "";
    });
  };
  const axis = {
    stroke: getComputedStyle(document.body).color,
    font: "9px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    gap: 2,
    grid: { stroke: "rgba(128,128,128,.15)" },
    ticks: { stroke: "rgba(128,128,128,.3)", size: 3 },
  };
  const u = new uPlot(
    {
      ...CHART,
      title: metrics.join(" / "),
      cursor: { sync: { key: "curves" }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      hooks: { setCursor: [syncLegend], setSeries: [syncLegend] },
      padding: [4, 6, 0, 0], // top, right, bottom, left: the axes already keep their own room
      axes: [{ ...axis, size: 22 }, { ...axis, size: 40, values: (u, vals) => vals.map(fmt) }],
      series: [
        { label: "step" },
        ...plots.flatMap(({ metric, ...options }, m) => attempts.map((a, i) => ({
          label: metrics.length > 1 ? `${metric} ${a}` : `attempt ${a}`,
          stroke: ATTEMPT_COLORS[i % ATTEMPT_COLORS.length],
          dash: m > 0 ? [5, 4] : undefined,
          width: 1.2,
          value: (u, v) => (v == null ? "" : fmt(v)),
          ...options,
        }))),
      ],
    },
    [steps, ...metrics.flatMap((metric) => ys[metric])],
    node,
  );
  return { node, u, plots, syncLegend };
}

// Shows the lines whose metric is not in `hiddenMetrics` and whose attempt is
// not in `hiddenAttempts`, and hides a whole chart once none of its metrics or
// none of the attempts is left. A line goes through uPlot's own `setSeries`, so a zoom survives it.
// Series 0 is the step, then each plot's `attempts` in order.
function showLines(charts, attempts) {
  for (const { node, u, plots, syncLegend } of charts) {
    plots.forEach(({ metric }, m) => {
      attempts.forEach((attempt, i) => {
        const shown = !hiddenMetrics.has(metric) && !hiddenAttempts.has(attempt);
        u.setSeries(1 + m * attempts.length + i, { show: shown });
      });
    });
    syncLegend(u);
    node.hidden = plots.every(({ metric }) => hiddenMetrics.has(metric)) || attempts.every((a) => hiddenAttempts.has(a));
  }
}

// The charts last drawn and the rows they were drawn from. `page.train` is
// one object for as long as the page is open (the log poll redraws around
// it), so the same rows hand back the same nodes and a zoom survives the
// redraw.
let drawnTrain = null, drawnCurves = null;

// The leg's training curves, from the step log's rows (see
// docs/logging.md): one line per metric per attempt, grouped into charts by
// CHARTS. Absent for anything whose job keeps no step log.
function curvesSection(train) {
  if (!train.length) return null;
  if (train !== drawnTrain) {
    const layout = layoutFor(train);
    const metrics = layout.flat().map((plot) => plot.metric);
    const data = columns(train, metrics);
    const charts = layout.map((plots) => chart(plots, data));
    // A checkbox that keeps `value` out of `hidden` while ticked.
    const toggle = (hidden, value, text, style) =>
      el("label", { class: "log-filter", style },
        el("input", {
          type: "checkbox",
          checked: !hidden.has(value),
          onchange: (event) => {
            event.target.checked ? hidden.delete(value) : hidden.add(value);
            showLines(charts, data.attempts);
          },
        }),
        text,
      );
    showLines(charts, data.attempts);
    drawnTrain = train;
    drawnCurves = el("div", { class: "artifact-curves" },
      el("div", { class: "summary-type", text: "training" }, ...metrics.map((m) => toggle(hiddenMetrics, m, m))),
      el("div", { class: "summary-type", text: "attempts" },
        ...data.attempts.map((a, i) =>
          toggle(hiddenAttempts, a, a, "color:" + ATTEMPT_COLORS[i % ATTEMPT_COLORS.length])),
      ),
      el("div", { class: "curve-row" }, ...charts.map(({ node }) => node)),
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
