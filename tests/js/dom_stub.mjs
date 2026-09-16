// Minimal DOM stub shared by the shipped-JS interaction harnesses.
//
// It is deliberately small: it supports exactly the node operations the AA
// desktop scripts use (createElement / getElementById / append /
// replaceChildren / remove / querySelectorAll over the descendants / dataset /
// textContent), counts innerHTML writes so a regression is visible, and never
// renders anything. It is not a browser; layout and a real click-through stay a
// separate step.

export function createDom() {
  const elements = new Map();
  const state = {innerHTMLWrites: 0, created: 0};

  function matches(node, selector) {
    return selector.split(",").map(part => part.trim().toLowerCase())
      .includes(String(node.tagName || "").toLowerCase());
  }

  function descendants(node, out = []) {
    for (const child of node.children) {
      out.push(child);
      descendants(child, out);
    }
    return out;
  }

  function makeElement(tag) {
    state.created += 1;
    const node = {
      tagName: tag, children: [], parentNode: null, attributes: {},
      dataset: {}, handlers: {}, style: {}, classList: {add() {}, remove() {}},
      _text: "", className: "", hidden: false, disabled: false, open: false,
      id: "", type: "", value: "", checked: false,
      get textContent() {
        return this._text + this.children.map(child => child.textContent).join("");
      },
      set textContent(value) { this._text = String(value); this.children = []; },
      append(...nodes) {
        this._text = "";
        for (const item of nodes) { item.parentNode = this; this.children.push(item); }
      },
      replaceChildren(...nodes) {
        this._text = ""; this.children = [];
        for (const item of nodes) { item.parentNode = this; this.children.push(item); }
      },
      remove() {
        const parent = this.parentNode;
        if (!parent) return;
        parent.children = parent.children.filter(child => child !== this);
        this.parentNode = null;
      },
      addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); },
      click() { for (const handler of this.handlers.click || []) handler({}); },
      setAttribute(name, value) { this.attributes[name] = String(value); },
      getAttribute(name) { return this.attributes[name] ?? null; },
      querySelectorAll(selector) {
        return descendants(this).filter(node => matches(node, selector));
      },
      querySelector(selector) {
        return this.querySelectorAll(selector)[0] ?? null;
      },
      focus() {}, blur() {},
    };
    Object.defineProperty(node, "innerHTML", {
      get() { return "<stub>"; },
      set() { state.innerHTMLWrites += 1; },
    });
    return node;
  }

  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        const node = makeElement("div");
        node.id = id;
        elements.set(id, node);
      }
      return elements.get(id);
    },
    createElement(tag) { return makeElement(tag); },
    querySelectorAll() { return []; },
  };
  return {document, elements, state, el: id => document.getElementById(id),
          makeElement};
}

// Drive a registered handler list the way a user event would, awaiting each.
export async function fire(el, type, event = {}) {
  let ran = 0;
  for (const handler of el.handlers[type] || []) { ran += 1; await handler(event); }
  return ran;
}

export async function click(el) { return fire(el, "click"); }
