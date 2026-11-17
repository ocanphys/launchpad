// logview.js -- one log stream, newest first, under a heading of level and
// ambient filters and a row of column labels: an artifact's calls
// (artifactview.js) or the launcher's own log (app.js). Every line names the
// call it is about and who wrote it, with the logger on the message's hover.
// A view is made once and handed each poll's rows: it keeps the lines it has
// drawn, so a poll builds only the lines that are new.

import { el } from "./el.js";

const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

// One cell class per column, with the label that sits over it. A row's call
// and its source are two different things that often read as the same word
// ("launcher launcher": the leasebook's own call id, written by the launcher).
const COLUMNS = [
  ["log-ts", "time"],
  ["log-call", "call"],
  ["log-source", "source"],
  ["log-level", "level"],
  ["log-msg", "message"],
];

// The rows out of both copies of a log -- what the containers have published
// (`livedict`) and what its file holds (`volume`) -- as one list, a row
// counted once however many copies hold it. Every row already says who wrote
// it in its own `source`; a storage is only where it is kept. The volume copy
// trails the live one by a persist pass, so the union is what is complete.
export function union(copies) {
  const seen = new Set();
  return copies.flat().filter((row) => {
    const key = [row.ts, row.level, row.logger, row.msg].join(" ");
    return seen.has(key) ? false : seen.add(key);
  });
}

// DD/MM/YY-HH:mm:ss in the reader's zone, or the time alone where the column
// is too narrow for the date (the launcher panel, whose rows are the running
// container's). The full instant is on hover either way.
function shortTs(ts, compact) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  const time = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  return compact ? time : `${p(d.getDate())}/${p(d.getMonth() + 1)}/${p(d.getFullYear() % 100)}-${time}`;
}

function logLine(row, compactTime) {
  return el("div", { class: "log-line level-" + row.level },
    el("span", { class: "log-ts", title: new Date(row.ts * 1000).toISOString(), text: shortTs(row.ts, compactTime) }),
    el("span", { class: "log-call", title: row.call_id, text: row.call_id.slice(-8) }),
    el("span", { class: "log-source", text: row.source }),
    el("span", { class: "log-level", text: row.level }),
    el("span", { class: "log-msg", title: row.logger, text: row.msg }),
  );
}

// A log view: `node`, mounted once, and `update(rows, emptyText)`, called
// with every poll's rows. One checkbox per level present, and an `ambient`
// one when the container logged anything alongside the call (rows whose
// `source` says so) -- on, so the default view is everything that happened.
// `hiddenLevels` and `hiddenSources` are the caller's sets. A traceback
// folded into `msg` keeps its line breaks (`.log-msg` is pre-wrap) and, like
// everything here, goes through textContent.
export function logView({ title, className, hiddenLevels, hiddenSources, emptyText, compactTime = false }) {
  const heading = el("div", { class: "summary-type log-heading" });
  const columns = el("div", { class: "log-line log-columns" },
    COLUMNS.map(([cell, label]) => el("span", { class: cell, text: label })),
  );
  const lines = el("div", { class: "log-lines" });
  const empty = el("p", { class: "empty" });
  let sorted = [];
  let none = emptyText; // what an empty log says
  let filters = null; // the checkboxes in the heading, as text
  let drawn = new Map(); // each line on screen, by its row

  // Puts the visible rows' lines under the column labels, in order: a line
  // already drawn is kept where it is, so only a new row costs a node.
  const render = () => {
    const visible = sorted.filter((row) => !hiddenLevels.has(row.level) && !hiddenSources.has(row.source));
    const kept = new Map(visible.map((row) => {
      const key = [row.ts, row.call_id, row.source, row.level, row.logger, row.msg].join(" ");
      return [key, drawn.get(key) || logLine(row, compactTime)];
    }));
    drawn = kept;
    empty.textContent = sorted.length ? "no logs at the selected filters." : none;
    let at = lines.firstChild;
    for (const node of visible.length ? [columns, ...kept.values()] : [empty]) {
      if (node === at) at = at.nextSibling;
      else lines.insertBefore(node, at);
    }
    while (at) {
      const next = at.nextSibling;
      at.remove();
      at = next;
    }
  };

  const checkbox = (label, hidden) =>
    el("label", { class: "log-filter" },
      el("input", {
        type: "checkbox",
        checked: !hidden.has(label),
        onchange: (event) => {
          if (event.target.checked) hidden.delete(label); else hidden.add(label);
          render();
        },
      }),
      label,
    );

  const update = (rows, text = emptyText) => {
    sorted = [...rows].sort((a, b) => b.ts - a.ts);
    none = text;
    const levels = [...new Set(sorted.map((row) => row.level))]
      .sort((a, b) => (LEVELS.indexOf(a) + 1 || 99) - (LEVELS.indexOf(b) + 1 || 99));
    const ambient = sorted.some((row) => row.source === "ambient");
    if (filters !== [...levels, ambient].join()) {
      filters = [...levels, ambient].join();
      heading.replaceChildren(
        el("span", { text: title }),
        ...levels.map((level) => checkbox(level, hiddenLevels)),
        ...(ambient ? [checkbox("ambient", hiddenSources)] : []),
      );
    }
    render();
  };

  return { node: el("div", { class: className }, heading, lines), update };
}
