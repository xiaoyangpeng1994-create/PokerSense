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
      type: "", checked: false,
      // Present so the shipped scripts can set them without a guard.
      options: [], maxLength: 0, inputMode: "", placeholder: "", title: "",
      get textContent() {
        return this._text + this.children.map(child => child.textContent).join("");
      },
      set textContent(value) { this._text = String(value); this.children = []; },
      append(...nodes) {
        this._text = "";
        for (const item of nodes) {
          item.parentNode = this;
          this.children.push(item);
          if (String(item.tagName).toLowerCase() === "option") this.options.push(item);
        }
      },
      replaceChildren(...nodes) {
        this._text = ""; this.children = []; this.options = [];
        for (const item of nodes) {
          item.parentNode = this;
          this.children.push(item);
          if (String(item.tagName).toLowerCase() === "option") this.options.push(item);
        }
      },
      remove() {
        const parent = this.parentNode;
        if (!parent) return;
        parent.children = parent.children.filter(child => child !== this);
        parent.options = parent.options.filter(option => option !== this);
        this.parentNode = null;
      },
      addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); },
      removeEventListener(type, handler) {
        this.handlers[type] = (this.handlers[type] || []).filter(item => item !== handler);
      },
      dispatchEvent(event) {
        const type = event && event.type ? event.type : "event";
        for (const handler of this.handlers[type] || []) handler(event);
        return true;
      },
      click() { for (const handler of this.handlers.click || []) handler({}); },
      setAttribute(name, value) { this.attributes[name] = String(value); },
      getAttribute(name) { return this.attributes[name] ?? null; },
      removeAttribute(name) { delete this.attributes[name]; },
      querySelectorAll(selector) {
        return descendants(this).filter(node => matches(node, selector));
      },
      querySelector(selector) {
        return this.querySelectorAll(selector)[0] ?? null;
      },
      showModal() { this.open = true; },
      close() { this.open = false; },
      focus() {}, blur() {},
      getBoundingClientRect() {
        return {top: 0, left: 0, width: 0, height: 0, bottom: 0, right: 0};
      },
    };
    Object.defineProperty(node, "innerHTML", {
      get() { return "<stub>"; },
      set() { state.innerHTMLWrites += 1; },
    });
    // A real input's `value` is always a string, so `.trim()` is always safe.
    let nodeValue = "";
    Object.defineProperty(node, "value", {
      get() { return nodeValue; },
      set(value) { nodeValue = value === null || value === undefined ? "" : String(value); },
    });
    // `element.id = x` must register the node, exactly like a real document: the
    // shipped scripts build inputs and then look them up by id.
    let nodeId = "";
    Object.defineProperty(node, "id", {
      get() { return nodeId; },
      set(value) { nodeId = String(value); elements.set(nodeId, node); },
    });
    return node;
  }

  const document = {
    hidden: false,
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
    addEventListener() {},
    removeEventListener() {},
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
