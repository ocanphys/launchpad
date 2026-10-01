// filter.js -- narrows the table level by level down an artifact's manifest,
// as its JSON nests. The top level is the artifact's class (the state map's
// `type`, the manifest's `artifact`). Once one is picked, every entry has the
// same two filterable keys under it: `parameters`, the manifest's own tree,
// and `dependencies`, every dependency's path (`depends_on`) as one set of
// choices. Every key is a level, closed until clicked: one holding an object
// opens onto its keys, any other onto its values, a min/max range when every
// one is a number and a checkbox apiece otherwise. Only an open level
// filters, so closing one drops its choices. Within a leaf any choice
// matches, across leaves every one has to, and an artifact without the leaf
// never does.
//
// Keeps the reader's choices, as artifactview.js keeps its log filters: every
// poll redraws the table, and the choice has to outlive that.

import { el } from "./el.js";

let artifact = ""; // "" is every class, with nothing under it to filter
const open = new Set(); // dotted paths of the levels the reader has opened
const chosen = new Map(); // leaf path -> Set of JSON-encoded values
const ranges = new Map(); // leaf path -> { min, max }, null for an open end

// The bar last handed out and what it was built from (see filterBar).
let built = null;

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

// The values `entry` holds at a dotted `path`: the one value there, or none,
// and every dependency's path at "dependencies". A list anywhere else is one
// value, not several (`betas`).
function held(entry, path) {
  let value = { parameters: entry.parameters, dependencies: entry.depends_on };
  for (const key of path.split(".")) value = isObject(value) ? value[key] : undefined;
  if (value === undefined) return [];
  return path === "dependencies" ? value : [value];
}

// Every distinct value `entries` hold at `path`, numbers ascending and
// anything else by its JSON.
function values(entries, path) {
  const byId = new Map();
  for (const entry of entries) for (const v of held(entry, path)) byId.set(JSON.stringify(v), v);
  const found = [...byId.values()];
  return found.every((v) => typeof v === "number")
    ? found.sort((a, b) => a - b)
    : found.sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

function passes(entry) {
  if (!artifact) return true;
  if (entry.type !== artifact) return false;
  for (const [path, picked] of chosen) {
    if (!held(entry, path).some((v) => picked.has(JSON.stringify(v)))) return false;
  }
  for (const [path, { min, max }] of ranges) {
    if (!held(entry, path).some((v) => (min == null || v >= min) && (max == null || v <= max))) return false;
  }
  return true;
}

// The paths of `states` that pass the reader's choices, sorted.
export function visible(states) {
  return Object.keys(states).filter((path) => passes(states[path])).sort();
}

function choices(path, found, onChange) {
  return found.map((value) => {
    const id = JSON.stringify(value);
    return el("label", { class: "filter-choice" },
      el("input", {
        type: "checkbox",
        checked: Boolean(chosen.get(path)?.has(id)),
        onchange: (event) => {
          const picked = chosen.get(path) || new Set();
          event.target.checked ? picked.add(id) : picked.delete(id);
          picked.size ? chosen.set(path, picked) : chosen.delete(path);
          onChange();
        },
      }),
      typeof value === "string" ? value : id,
    );
  });
}

function range(path, found, onChange) {
  const bounds = ranges.get(path) || { min: null, max: null };
  const bound = (side, placeholder) =>
    el("input", {
      type: "number", step: "any", class: "filter-bound", placeholder, value: bounds[side],
      oninput: (event) => {
        bounds[side] = event.target.value === "" ? null : Number(event.target.value);
        bounds.min == null && bounds.max == null ? ranges.delete(path) : ranges.set(path, bounds);
        onChange();
      },
    });
  return el("div", {}, bound("min", found[0]), " to ", bound("max", found.at(-1)));
}

// A level's key, opening onto `children()` when clicked, with `count` beside
// it when given: how many values a leaf holds. Closing it closes every level
// under it and drops their choices.
function level(key, path, onChange, children, count) {
  const isOpen = open.has(path);
  const toggle = () => {
    if (!isOpen) open.add(path);
    else for (const store of [open, chosen, ranges]) {
      for (const k of [...store.keys()]) if (k === path || k.startsWith(path + ".")) store.delete(k);
    }
    onChange();
  };
  return el("div", {},
    el("button", { type: "button", class: "filter-toggle", onclick: toggle },
      (isOpen ? "▾ " : "▸ ") + key,
      count == null ? null : el("span", { class: "filter-count", text: count }),
    ),
    isOpen ? el("div", { class: "filter-children" }, children()) : null,
  );
}

// What `entries` hold at `path`, as a level under its `key`: onto its keys
// if any of them holds an object there, else onto a leaf's range or choices.
function node(entries, key, path, onChange) {
  const found = values(entries, path);
  const objects = found.filter(isObject);
  if (objects.length) {
    const keys = [...new Set(objects.flatMap(Object.keys))].sort();
    return level(key, path, onChange, () => keys.map((k) => node(entries, k, path + "." + k, onChange)));
  }
  const numeric = found.every((v) => typeof v === "number");
  return level(key, path, onChange, () => (numeric ? range : choices)(path, found, onChange), found.length);
}

// The class picker and, once a class is picked, its `parameters` and
// `dependencies` levels. Hands back the node it built last time until the
// classes, the open levels or the manifests under them change, so a poll
// never rebuilds an input under the reader's cursor: a caller mounts the
// node only when it is a different one.
export function filterBar(states, onChange) {
  const all = Object.values(states);
  const classes = [...new Set(all.map((e) => e.type).filter(Boolean))].sort();
  const entries = artifact ? all.filter((e) => e.type === artifact) : [];
  const key = JSON.stringify([artifact, classes, [...open], entries.map((e) => [e.parameters, e.depends_on])]);
  if (built?.key === key) return built.node;

  const clear = () => { chosen.clear(); ranges.clear(); built = null; };
  const bar = el("div", { class: "filters" },
    el("div", { class: "filter-field" },
      el("span", { class: "filter-key", text: "artifact" }),
      el("select", { onchange: (event) => { artifact = event.target.value; open.clear(); clear(); onChange(); } },
        el("option", { value: "", text: "every artifact" }),
        classes.map((c) => el("option", { value: c, text: c, selected: c === artifact })),
      ),
      artifact ? el("button", { type: "button", text: "clear", onclick: () => { clear(); onChange(); } }) : null,
    ),
    artifact
      ? [
          node(entries, "parameters", "parameters", onChange),
          node(entries, "dependencies", "dependencies", onChange),
        ]
      : null,
  );
  built = { key, node: bar };
  return bar;
}
