// Unit tests for Explorer's Meshing card and dilation layer sub-cards.
import assert from "node:assert/strict";
import test from "node:test";

import { build } from "esbuild";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Minimal DOM shim sufficient for Explorer construction, rendering, and events.
function installDomShim() {
  class FakeClassList {
    constructor(el) {
      this.el = el;
    }
    _tokens() {
      return this.el._className ? this.el._className.trim().split(/\s+/).filter(Boolean) : [];
    }
    _write(tokens) {
      this.el._className = tokens.join(" ");
    }
    contains(name) {
      return this._tokens().includes(name);
    }
    add(...names) {
      const tokens = this._tokens();
      for (const name of names) {
        if (!tokens.includes(name)) tokens.push(name);
      }
      this._write(tokens);
    }
    remove(...names) {
      this._write(this._tokens().filter((t) => !names.includes(t)));
    }
    toggle(name, force) {
      const has = this.contains(name);
      const want = force === undefined ? !has : Boolean(force);
      if (want && !has) this.add(name);
      else if (!want && has) this.remove(name);
      return want;
    }
  }

  class FakeElement {
    constructor(tagName) {
      this.tagName = tagName.toUpperCase();
      this._className = "";
      this.classList = new FakeClassList(this);
      this.dataset = {};
      this.attributes = new Map();
      this.children = [];
      this.parentElement = null;
      this.listeners = new Map();
      this.hidden = false;
      this.disabled = false;
      this.checked = false;
      this.type = "";
      this.title = "";
      this.min = "";
      this.max = "";
      this.step = "";
      this._value = "";
      this._textContent = "";
    }
    get className() {
      return this._className;
    }
    set className(val) {
      this._className = String(val ?? "");
    }
    get value() {
      return this._value;
    }
    set value(val) {
      this._value = String(val ?? "");
    }
    get textContent() {
      if (this.children.length === 0) return this._textContent;
      return this._textContent + this.children.map((c) => c.textContent).join("");
    }
    set textContent(val) {
      this.children = [];
      this._textContent = String(val ?? "");
    }
    set innerHTML(val) {
      if (val === "") {
        this.children = [];
        this._textContent = "";
      } else {
        this.children = [];
        this._textContent = String(val);
      }
    }
    get innerHTML() {
      return this._textContent;
    }
    get firstElementChild() {
      return this.children[0] ?? null;
    }
    get options() {
      return this.children.filter((c) => c.tagName === "OPTION");
    }
    append(...nodes) {
      for (const node of nodes) {
        if (typeof node === "string") continue;
        if (node.parentElement) {
          node.parentElement.children = node.parentElement.children.filter((c) => c !== node);
        }
        node.parentElement = this;
        this.children.push(node);
      }
    }
    appendChild(node) {
      this.append(node);
      return node;
    }
    replaceChildren(...nodes) {
      for (const child of this.children) child.parentElement = null;
      this.children = [];
      this._textContent = "";
      this.append(...nodes);
    }
    remove() {
      if (!this.parentElement) return;
      this.parentElement.children = this.parentElement.children.filter((c) => c !== this);
      this.parentElement = null;
    }
    replaceWith(node) {
      if (!this.parentElement) return;
      const parent = this.parentElement;
      const idx = parent.children.indexOf(this);
      if (idx >= 0) {
        this.parentElement = null;
        node.parentElement = parent;
        parent.children[idx] = node;
      }
    }
    setAttribute(name, value) {
      this.attributes.set(name, String(value));
    }
    getAttribute(name) {
      return this.attributes.get(name) ?? null;
    }
    addEventListener(type, fn) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(fn);
    }
    dispatchEvent(event) {
      const ev = {
        type: event.type,
        target: event.target ?? this,
        bubbles: Boolean(event.bubbles),
        _stopped: false,
        stopPropagation() {
          this._stopped = true;
        },
      };
      let cur = this;
      while (cur && !ev._stopped) {
        for (const fn of cur.listeners.get(ev.type) ?? []) fn(ev);
        if (!ev.bubbles) break;
        cur = cur.parentElement;
      }
    }
    click() {
      this.dispatchEvent({ type: "click", bubbles: true, target: this });
    }
    closest(selector) {
      let cur = this;
      while (cur) {
        if (matchesSimple(cur, selector)) return cur;
        cur = cur.parentElement;
      }
      return null;
    }
    querySelectorAll(selector) {
      const out = [];
      const walk = (node) => {
        for (const child of node.children) {
          if (matchesSimple(child, selector)) out.push(child);
          walk(child);
        }
      };
      walk(this);
      return out;
    }
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] ?? null;
    }
  }

  class FakeInputElement extends FakeElement {
    constructor() {
      super("input");
    }
  }
  class FakeSelectElement extends FakeElement {
    constructor() {
      super("select");
    }
  }
  class FakeButtonElement extends FakeElement {
    constructor() {
      super("button");
    }
  }

  function matchesSimple(el, selector) {
    if (selector.startsWith(".")) {
      return el.classList.contains(selector.slice(1));
    }
    if (selector.startsWith("[")) {
      const pairs = [...selector.matchAll(/\[([^=\]]+)="([^"]*)"\]/g)];
      return pairs.every(([, key, val]) => {
        if (key.startsWith("data-")) {
          return el.dataset[key.slice(5)] === val;
        }
        return el.getAttribute(key) === val;
      });
    }
    return el.tagName === selector.toUpperCase();
  }

  const fakeDoc = {
    activeElement: null,
    createElement(tag) {
      const t = tag.toLowerCase();
      if (t === "input") return new FakeInputElement();
      if (t === "select") return new FakeSelectElement();
      if (t === "button") return new FakeButtonElement();
      return new FakeElement(t);
    },
  };
  globalThis.HTMLElement = FakeElement;
  globalThis.HTMLInputElement = FakeInputElement;
  globalThis.HTMLSelectElement = FakeSelectElement;
  globalThis.HTMLButtonElement = FakeButtonElement;
  globalThis.document = fakeDoc;
  globalThis.window = globalThis;
}

installDomShim();

const here = dirname(fileURLToPath(import.meta.url));
const outfile = join(mkdtempSync(join(tmpdir(), "explorer-test-")), "explorer.mjs");
writeFileSync(
  outfile,
  (
    await build({
      entryPoints: [join(here, "explorer.ts")],
      bundle: true,
      format: "esm",
      write: false,
    })
  ).outputFiles[0].text,
);
const { Explorer } = await import(outfile);

function baseMeta(overrides = {}) {
  return {
    levels: [{ name: "core", material: "si", mesh_order: 1, fill: false, zmin: 0, thickness: 0.22, sidewall_angle: 0, bias: 0, width_to_z: 0 }],
    ports: [],
    source_layers: ["LOWER", "UPPER"],
    background: "sio2",
    types: {},
    settings_defaults: {
      wavelength_um: 1.55,
      pulse_width_frac: 0,
      port_padding: [3, 2],
      pml_thickness: 10,
      resolution_um: 0.02,
      time_s: 4e-13,
      boundary_offset: [0, 0, 0, 0, 0, 0],
      log_range: 6,
      averaging: "subpixel",
      keep_partial: false,
      meshing: "uniform",
      pml_um: 1.0,
      ppw: 20.0,
      global_dilation: 3.0,
      global_z_padding: 3.0,
      mesh_growth: 1.25,
      dilation_layers: [],
    },
    ...overrides,
  };
}

function baseState(simOverrides = {}, stateOverrides = {}) {
  return {
    styles: { core: { color: "#ff0000", opacity: 0.9, visible: true } },
    active_tab: "simulation",
    port_visible: {},
    selection: null,
    active_viewport: "3d",
    show_full: false,
    plane: "xy",
    plane_position: 0,
    plane_bounds: [-1, 1],
    plane_fade_frac: 0.05,
    sources: [],
    detectors: [],
    glyphs: [],
    field_settings: {},
    simulation: {
      ...baseMeta().settings_defaults,
      ...simOverrides,
    },
    running: false,
    ...stateOverrides,
  };
}

function makeExplorer({ meta = baseMeta(), startCollapsed = false } = {}) {
  const settingsEvents = [];
  const explorer = new Explorer(
    meta,
    {
      onPatch() {},
      onActiveTab() {},
      onSelect() {},
      onAddSource() {},
      onAddDetector() {},
      onUpdateSpec() {},
      onRemoveSpec() {},
      onSettings(fields) {
        settingsEvents.push(fields);
      },
      onPlay() {},
      onSaveVideo() {},
      onPlayAll() {},
      onRecordAll() {},
      onGroupDownsample() {},
    },
    "simulation",
    startCollapsed,
  );
  return { explorer, settingsEvents };
}

function findFieldRow(container, labelText) {
  for (const row of container.querySelectorAll(".gfx-field")) {
    const nameSpan = row.querySelector(".gfx-field-name");
    if (nameSpan && nameSpan.textContent === labelText) return row;
  }
  return null;
}

test("Meshing card defaults to uniform and toggles adaptive controls", () => {
  const { explorer, settingsEvents } = makeExplorer();
  explorer.render(baseState());

  const simPanel = explorer.root.querySelectorAll(".gfx-tabpanel")[2];
  const meshCard = simPanel.querySelector(".gfx-mesh-card");
  assert.ok(meshCard, "Meshing card exists");
  assert.equal(meshCard.classList.contains("gfx-collapsed"), false);

  const pmlCellsRow = findFieldRow(simPanel, "PML (cells)");
  const pmlUmRow = findFieldRow(simPanel, "PML (µm)");
  const resRow = findFieldRow(meshCard, "resolution");
  const adaptiveBox = meshCard.querySelector(".gfx-mesh-adaptive");

  assert.equal(pmlCellsRow.hidden, false);
  assert.equal(pmlUmRow.hidden, true);
  assert.equal(resRow.hidden, false);
  assert.equal(adaptiveBox.hidden, true);

  // Switch meshing select to adaptive
  const typeSelect = findFieldRow(meshCard, "type").querySelector("select");
  typeSelect.value = "adaptive";
  typeSelect.dispatchEvent({ type: "change" });
  assert.deepEqual(settingsEvents.at(-1), { meshing: "adaptive" });

  // Render adaptive state
  explorer.render(baseState({ meshing: "adaptive" }));
  assert.equal(pmlCellsRow.hidden, true);
  assert.equal(pmlUmRow.hidden, false);
  assert.equal(resRow.hidden, true);
  assert.equal(adaptiveBox.hidden, false);

  assert.ok(findFieldRow(adaptiveBox, "ppw"));
  assert.ok(findFieldRow(adaptiveBox, "global dilation"));
  assert.ok(findFieldRow(adaptiveBox, "global z pad"));
  assert.ok(findFieldRow(adaptiveBox, "max cell growth"));
});

test("Dilation layer sub-cards add, update in place, disable duplicates, and remove", () => {
  const { explorer, settingsEvents } = makeExplorer();
  let state = baseState({ meshing: "adaptive" });
  explorer.render(state);

  const meshCard = explorer.root.querySelector(".gfx-mesh-card");
  const addBtn = meshCard.querySelector(".gfx-mesh-dilations").querySelector(".gfx-section-add");
  assert.equal(addBtn.disabled, false);

  // Click + to add first layer (LOWER)
  addBtn.click();
  assert.deepEqual(settingsEvents.at(-1), { dilation_layers: [{ layer: "LOWER", dilation: 0.1 }] });
  state = baseState({ meshing: "adaptive", dilation_layers: [{ layer: "LOWER", dilation: 0.1 }] });
  explorer.render(state);

  let subcards = meshCard.querySelectorAll(".gfx-subcard");
  assert.equal(subcards.length, 1);
  const firstInput = subcards[0].querySelector("input");
  assert.equal(firstInput.value, "0.1");

  // Edit dilation and re-render: DOM node must stay the same instance (in-place update)
  firstInput.value = "0.25";
  firstInput.dispatchEvent({ type: "input" });
  assert.deepEqual(settingsEvents.at(-1), { dilation_layers: [{ layer: "LOWER", dilation: 0.25 }] });
  state = baseState({ meshing: "adaptive", dilation_layers: [{ layer: "LOWER", dilation: 0.25 }] });
  explorer.render(state);
  assert.equal(meshCard.querySelectorAll(".gfx-subcard")[0].querySelector("input"), firstInput);

  // Click + to add second layer (UPPER); all layers now used so + becomes disabled
  addBtn.click();
  assert.deepEqual(settingsEvents.at(-1), {
    dilation_layers: [
      { layer: "LOWER", dilation: 0.25 },
      { layer: "UPPER", dilation: 0.1 },
    ],
  });
  state = baseState({
    meshing: "adaptive",
    dilation_layers: [
      { layer: "LOWER", dilation: 0.25 },
      { layer: "UPPER", dilation: 0.1 },
    ],
  });
  explorer.render(state);
  subcards = meshCard.querySelectorAll(".gfx-subcard");
  assert.equal(subcards.length, 2);
  assert.equal(addBtn.disabled, true);

  // Sibling select disables the already-used layer
  const firstOpts = subcards[0].querySelector("select").options;
  assert.equal(firstOpts.find((o) => o.value === "LOWER").disabled, false);
  assert.equal(firstOpts.find((o) => o.value === "UPPER").disabled, true);

  // Remove first sub-card
  subcards[0].querySelector(".gfx-remove").click();
  assert.deepEqual(settingsEvents.at(-1), {
    dilation_layers: [{ layer: "UPPER", dilation: 0.1 }],
  });
  state = baseState({
    meshing: "adaptive",
    dilation_layers: [{ layer: "UPPER", dilation: 0.1 }],
  });
  explorer.render(state);
  assert.equal(meshCard.querySelectorAll(".gfx-subcard").length, 1);
  assert.equal(addBtn.disabled, false);
});

test("startCollapsed collapses Meshing card and subcards, and readonly disables all inputs", () => {
  const { explorer } = makeExplorer({ startCollapsed: true });
  explorer.render(
    baseState(
      { meshing: "adaptive", dilation_layers: [{ layer: "LOWER", dilation: 0.2 }] },
      { readonly: { reason: "Precomputed demo" } },
    ),
  );

  const simPanel = explorer.root.querySelectorAll(".gfx-tabpanel")[2];
  const meshCard = simPanel.querySelector(".gfx-mesh-card");
  const subcard = simPanel.querySelector(".gfx-subcard");
  assert.equal(meshCard.classList.contains("gfx-collapsed"), true);
  assert.equal(subcard.classList.contains("gfx-collapsed"), true);

  const inputs = simPanel.querySelectorAll("input").filter((i) => i.type !== "checkbox");
  assert.ok(inputs.length > 0);
  assert.ok(inputs.every((i) => i.disabled === true), "all simulation inputs disabled when readonly");
  assert.ok(simPanel.querySelectorAll("select").every((s) => s.disabled === true));
  assert.equal(meshCard.querySelector(".gfx-section-add").disabled, true);
  assert.equal(subcard.querySelector(".gfx-remove").disabled, true);
});

test("legacy payloads without meshing fields or source_layers render cleanly", () => {
  const legacyMeta = baseMeta({ source_layers: undefined, settings_defaults: {} });
  const { explorer } = makeExplorer({ meta: legacyMeta });
  const legacySim = {
    wavelength_um: 1.55,
    pulse_width_frac: 0,
    port_padding: [3, 2],
    pml_thickness: 10,
    resolution_um: 0.02,
    time_s: 4e-13,
    boundary_offset: [0, 0, 0, 0, 0, 0],
    log_range: 6,
    averaging: "subpixel",
    keep_partial: false,
  };
  explorer.render(baseState(legacySim));
  const simPanel = explorer.root.querySelectorAll(".gfx-tabpanel")[2];
  const meshCard = simPanel.querySelector(".gfx-mesh-card");
  assert.equal(findFieldRow(meshCard, "type").querySelector("select").value, "uniform");
  assert.equal(meshCard.querySelector(".gfx-mesh-adaptive").hidden, true);
  assert.equal(meshCard.querySelector(".gfx-section-add").disabled, true);
});
