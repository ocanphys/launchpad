// artifactview.js — the drill-down page for one artifact: what it is, what it
// was built from, its training curves if it has any, and what every call
// that ever worked on it said.
//
// No state and no network of its own: app.js fetches `/manifest/<path>` and
// `/logs/artifact/<path>` in parallel and calls renderArtifactView(container,
// manifest, logs). The worker files its own output in a jsonl beside the
// artifact and in the `call_logs` Dict (docs/LOGGING.md); `logs` here is
// `/logs/artifact/<path>`'s response, straight from that Dict -- the one
// container serving this dashboard opens no log file while it runs, because
// reading files on a mount it also reloads is how a reload loses a race
// with a walk (docs/LOGGING.md). Its `train` (the step log, both copies)
// comes out of a Dict the same way, so that holds for the curves too.

import { el } from "./el.js";

// Whether nested parameter values print indented or on one line. Page
// state rather than DOM state: every poll rebuilds the view, and the choice
// has to outlive that.
let expandedParams = false;

// A parameter's value: primitives print plain, anything richer (a list, a
// nested config) prints as JSON, compact or indented. Dependency values
// never reach here -- manifest_endpoint's own parameters/dependencies split
// already keeps those out of `parameters`.
function paramValue(value) {
  if (value === null || typeof value !== "object") return String(value);
  return JSON.stringify(value, null, expandedParams ? 2 : 0);
}

// Type, own parameters, and one link per direct dependency. A dependency is
// just its artifact_path -- main.py's computed state holds no nested
// manifests, so drilling further is a click to that artifact's own page
// (where its own type and parameters live) rather than the whole tree being
// dumped on this one.
function summary(manifest) {
  const params = Object.entries(manifest.parameters || {});
  const deps = manifest.depends_on || [];
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
      manifest.type,
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

// Call ids are long and only their tail varies day to day -- same
// truncation render.js's callId() uses for the table, so a call looks the
// same wherever it shows up. Full id still reaches the page, via title.
function shortCallId(id) {
  return id.length > 8 ? id.slice(-8) : id;
}

// The rows out of every copy of a record -- a call's `launcher`, `container`
// and `volume` channels, a step log's `live` and `volume` -- as one list, a
// row counted once however many copies hold it, `key` saying when two rows
// are the same one. The volume copy trails the others by a persist pass, so
// the union is what is complete.
function union(copies, key) {
  const seen = new Set();
  return copies.flat().filter((row) => {
    const k = key(row);
    return seen.has(k) ? false : seen.add(k);
  });
}

// Levels in severity order, for the filter row; anything else sorts after.
const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
// Which levels the reader has switched off. Page state rather than DOM
// state: every poll rebuilds the view, and the choice has to outlive that.
const hiddenLevels = new Set(["DEBUG"]);

// DD/MM/YY-HH:mm:ss in the reader's zone; the full instant goes on hover.
function shortTs(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getDate())}/${p(d.getMonth() + 1)}/${p(d.getFullYear() % 100)}`
    + `-${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// Every call's rows in one stream, each tagged with its call and with who
// wrote it (`source`: the channel it came out of, or for a row read off the
// file the channel the file recorded), oldest first. A call's last heartbeat
// is a row too, so where an attempt stopped beating reads in sequence with
// what it last said.
function logRows(calls) {
  return calls
    .flatMap((call) => [
      ...union(
        [
          call.volume.map((row) => ({ ...row, source: row.source || "volume" })),
          call.launcher.map((row) => ({ ...row, source: "launcher" })),
          call.container.map((row) => ({ ...row, source: "container" })),
        ],
        (row) => [row.ts, row.level, row.logger, row.msg].join(" "),
      ).map((row) => ({ ...row, call_id: call.call_id })),
      call.last_heartbeat == null ? null : {
        ts: call.last_heartbeat, level: "INFO", logger: "heartbeat", msg: "last heartbeat",
        call_id: call.call_id, source: "heartbeat",
      },
    ])
    .filter(Boolean)
    .sort((a, b) => a.ts - b.ts);
}

function logLine(row) {
  return el("div", { class: "log-line level-" + row.level },
    el("span", { class: "log-ts", title: new Date(row.ts * 1000).toISOString(), text: shortTs(row.ts) }),
    el("span", { class: "log-call", title: row.call_id, text: shortCallId(row.call_id) }),
    el("span", { class: "log-source", text: row.source }),
    el("span", { class: "log-level", text: row.level }),
    el("span", { class: "log-msg", text: row.msg }),
  );
}

// Every call that has ever worked on this artifact (see main.py's
// artifact_call_logs), as one stream with a level filter above it: one
// checkbox per level present, DEBUG off until switched on.
function logs(logsPayload) {
  const rows = logRows((logsPayload && logsPayload.calls) || []);
  const levels = [...new Set(rows.map((row) => row.level))]
    .sort((a, b) => (LEVELS.indexOf(a) + 1 || 99) - (LEVELS.indexOf(b) + 1 || 99));
  const lines = el("div", { class: "log-lines" });
  const render = () =>
    lines.replaceChildren(...rows.filter((row) => !hiddenLevels.has(row.level)).map(logLine));
  render();
  return el("div", { class: "artifact-logs" },
    el("div", { class: "summary-type" },
      "logs",
      ...levels.map((level) =>
        el("label", { class: "log-filter" },
          el("input", {
            type: "checkbox",
            checked: !hiddenLevels.has(level),
            onchange: (event) => {
              if (event.target.checked) hiddenLevels.delete(level); else hiddenLevels.add(level);
              render();
            },
          }),
          level,
        ),
      ),
    ),
    rows.length ? lines : el("p", { class: "empty", text: "no call has ever worked on this artifact yet." }),
  );
}

// --- training curves ----------------------------------------------------------

// `el` builds HTML elements; SVG ones need their own namespace or the
// browser makes an unknown HTML tag that draws nothing.
function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.append(...children);
  return node;
}

const METRICS = ["loss", "grad_norm", "learning_rate"];
const CURVE_POINTS = 500; // at most this many points per attempt per metric are drawn

// Per-attempt series off the leg's step rows: {attempt: {step: [...],
// loss: [...], grad_norm: [...], learning_rate: [...]}}, each attempt
// bucket-averaged down to at most CURVE_POINTS points, `step` being each
// bucket's last. A step redone after a crash is in every attempt that took
// it, so a row is one (attempt, step).
function curves(train) {
  const byAttempt = new Map();
  for (const row of union([train.volume, train.live], (r) => r.attempt + " " + r.step)) {
    if (!byAttempt.has(row.attempt)) byAttempt.set(row.attempt, []);
    byAttempt.get(row.attempt).push(row);
  }
  const series = {};
  for (const [attempt, rows] of byAttempt) {
    rows.sort((a, b) => a.step - b.step);
    const width = Math.ceil(rows.length / CURVE_POINTS);
    const buckets = [];
    for (let i = 0; i < rows.length; i += width) buckets.push(rows.slice(i, i + width));
    series[attempt] = { step: buckets.map((b) => b[b.length - 1].step) };
    for (const metric of METRICS) {
      series[attempt][metric] = buckets.map((b) => b.reduce((sum, r) => sum + r[metric], 0) / b.length);
    }
  }
  return series;
}
// One color per attempt, cycling: attempt 1 is always the first color, so a
// leg's history reads the same on every visit.
const ATTEMPT_COLORS = ["#0969da", "#cf222e", "#1a7f37", "#8250df", "#bf8700", "#e16f24"];
const W = 300, H = 130, PAD = { l: 46, r: 8, t: 8, b: 18 };

// Three significant digits, with the trailing zeros toPrecision keeps dropped.
function fmt(value) {
  return String(Number(Number(value).toPrecision(3)));
}

// One metric against step, every attempt its own line, on a shared scale.
// Only the extremes are labeled -- enough to read the shape and the range,
// which is what a glance at a curve is for.
function chart(metric, attempts, curves) {
  const xs = attempts.flatMap((a) => curves[a].step);
  const ys = attempts.flatMap((a) => curves[a][metric]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x) => PAD.l + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.5) * (W - PAD.l - PAD.r);
  const sy = (y) => PAD.t + (y1 > y0 ? (y1 - y) / (y1 - y0) : 0.5) * (H - PAD.t - PAD.b);
  const label = (x, y, text, anchor) =>
    svg("text", { x, y, "text-anchor": anchor, class: "curve-label" }, text);

  return el("div", { class: "curve" },
    el("div", { class: "curve-title", text: metric }),
    svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "curve-svg" },
      svg("path", {
        class: "curve-axis",
        d: `M${PAD.l},${PAD.t}V${H - PAD.b}H${W - PAD.r}`,
      }),
      label(PAD.l - 4, PAD.t + 4, fmt(y1), "end"),
      label(PAD.l - 4, H - PAD.b, fmt(y0), "end"),
      label(PAD.l, H - 4, String(x0), "start"),
      label(W - PAD.r, H - 4, String(x1), "end"),
      ...attempts.map((a, i) =>
        svg("path", {
          class: "curve-line",
          stroke: ATTEMPT_COLORS[i % ATTEMPT_COLORS.length],
          d: curves[a].step
            .map((x, j) => `${j ? "L" : "M"}${sx(x).toFixed(1)},${sy(curves[a][metric][j]).toFixed(1)}`)
            .join(""),
        }),
      ),
    ),
  );
}

// The leg's training curves, from `/logs/artifact/<path>`'s `train` (both
// copies of its step log, see docs/LOGGING.md): one chart per metric, one
// line per attempt. Absent for anything whose job keeps no step log.
function curvesSection(train) {
  const series = curves(train);
  const attempts = Object.keys(series).sort((a, b) => a - b);
  if (!attempts.length) return null;
  return el("div", { class: "artifact-curves" },
    el("div", { class: "summary-type" },
      "training",
      ...attempts.map((a, i) =>
        el("span", { class: "curve-legend", style: `color: ${ATTEMPT_COLORS[i % ATTEMPT_COLORS.length]}` },
          `● attempt ${a}`,
        ),
      ),
    ),
    el("div", { class: "curve-row" }, ...METRICS.map((metric) => chart(metric, attempts, series))),
  );
}

// What the page shows between a click and its first answer: the back link,
// and which artifact is on its way. `route()` calls this the moment it
// switches to an artifact, because `container` still holds the *previous*
// artifact's nodes until a fetch resolves and replaceChildren swaps them --
// and the title has already changed by then, so leaving them up shows one
// artifact's metadata under another's name. Only ever called on navigation,
// never on a repoll: an artifact already on screen keeps what it has until
// its own next answer lands.
export function renderArtifactPending(container, artifactPath) {
  container.replaceChildren(
    el("p", { class: "view-back" }, el("a", { href: "#/", text: "← dashboard" })),
    el("p", { class: "empty", text: "loading " + artifactPath + "…" }),
  );
}

export function renderArtifactView(container, manifest, logsPayload) {
  const nodes = [el("p", { class: "view-back" }, el("a", { href: "#/", text: "← dashboard" }))];

  // `error` covers both "no manifest yet" (declared, not built -- a normal
  // state) and a manifest that wouldn't load, which read the same to a reader
  // of this page: there is nothing to show and the reason is the message.
  if (!manifest || manifest.error) {
    nodes.push(el("p", { class: "empty", text: (manifest && manifest.error) || "loading…" }));
  } else {
    nodes.push(summary(manifest));
  }

  // Logs render independently of whether the manifest loaded -- a call can
  // have left output behind even for an artifact that's only declared, not
  // built (a job that ran and failed before writing anything), so an empty
  // manifest is never a reason to hide them.
  nodes.push(curvesSection((logsPayload && logsPayload.train) || { live: [], volume: [] }), logs(logsPayload));

  container.replaceChildren(...nodes.filter(Boolean)); // curves are null without a step log
}
