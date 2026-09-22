// el.js -- the only DOM-building primitive in the app: one call for
// createElement / className / textContent / appendChild, with the safety
// property that text always goes through textContent, never innerHTML, so
// run ids / call ids chosen by someone else are never parsed as markup.
//
//   el("td", { class: "run", text: id })
//   el("button", { class: "job-btn", disabled: true, onclick: fn }, "label")
//   el("tr", {}, cellA, cellB, [maybeCell, maybeCell])   // arrays are flattened
//
// Props:
//   class      -> className
//   text       -> textContent (safe; use this for untrusted strings)
//   on<event>  -> addEventListener("<event>", fn)   e.g. onclick
//   boolean    -> presence attribute (disabled, hidden…) toggled correctly
//   anything   -> setAttribute (title, id, …); null/undefined values skipped
//
// Children: nodes are appended as-is; strings/numbers become text nodes;
// null / undefined / false are dropped (handy for conditional children).

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);

  for (const [key, value] of Object.entries(props)) {
    if (value == null) continue;                       // skip null/undefined props
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value; // text only -- never HTML
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (typeof value === "boolean") {
      // disabled=false must REMOVE the attribute, not set it to "false"
      // (which the DOM would treat as truthy). toggleAttribute gets this right.
      node.toggleAttribute(key, value);
    } else {
      node.setAttribute(key, value);
    }
  }

  appendChildren(node, children);
  return node;
}

function appendChildren(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
}
