// logview.js -- one log stream, newest first, with a level filter in its
// heading: an artifact's calls (artifactview.js, `callMetadata`: each line
// names its call and which side wrote it) or the launcher's own log
// (app.js: no call to name, the logger on hover). `hiddenLevels` is the
// caller's set, so the choice outlives the rebuild every poll does.

import { el } from "./el.js";

const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

// The rows out of every copy of a record -- a call's `launcher`, `container`
// and `volume` channels, the launcher's `launcher` and `volume` -- as one
// list, a row counted once however many copies hold it. The volume copy
// trails the others by a persist pass, so the union is what is complete.
export function union(copies) {
  const seen = new Set();
  return copies.flat().filter((row) => {
    const key = [row.ts, row.level, row.logger, row.msg].join(" ");
    return seen.has(key) ? false : seen.add(key);
  });
}

// DD/MM/YY-HH:mm:ss in the reader's zone; the full instant goes on hover.
function shortTs(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getDate())}/${p(d.getMonth() + 1)}/${p(d.getFullYear() % 100)}`
    + `-${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function logLine(row, callMetadata) {
  return el("div", { class: "log-line level-" + row.level },
    el("span", { class: "log-ts", title: new Date(row.ts * 1000).toISOString(), text: shortTs(row.ts) }),
    callMetadata
      ? [
          el("span", { class: "log-call", title: row.call_id, text: row.call_id.slice(-8) }),
          el("span", { class: "log-source", text: row.source }),
        ]
      : [],
    el("span", { class: "log-level", text: row.level }),
    el("span", { class: "log-msg", title: callMetadata ? undefined : row.logger, text: row.msg }),
  );
}

// One checkbox per level present; a traceback folded into `msg` keeps its
// line breaks (`.log-msg` is pre-wrap) and, like everything here, goes
// through textContent.
export function logStream(rows, { title, className, hiddenLevels, emptyText, callMetadata = false }) {
  const sorted = [...rows].sort((a, b) => b.ts - a.ts);
  const levels = [...new Set(sorted.map((row) => row.level))]
    .sort((a, b) => (LEVELS.indexOf(a) + 1 || 99) - (LEVELS.indexOf(b) + 1 || 99));
  const lines = el("div", { class: "log-lines" });
  const render = () => {
    const visible = sorted.filter((row) => !hiddenLevels.has(row.level));
    lines.replaceChildren(...visible.map((row) => logLine(row, callMetadata)));
    if (!visible.length) lines.append(el("p", {
      class: "empty",
      text: sorted.length ? "no logs at the selected levels." : emptyText,
    }));
  };
  render();
  return el("div", { class: className },
    el("div", { class: "summary-type log-heading" },
      el("span", { text: title }),
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
    lines,
  );
}
