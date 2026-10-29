// The static demo's model (static.ts): it plays exported files back through the
// same get / send / on interface the kernel-driven widget has. The tests use an
// in-memory source and a tiny made-up demo; no DOM and no three.js are involved.
import assert from "node:assert/strict";
import test from "node:test";

import { transform } from "esbuild";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "static.ts"), "utf8");
const { code } = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
const { StaticModel, dequantize, nearestStop, FORMAT } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`
);

// ------------------------------------------------------------------ fixtures

const STOPS = { xy: [-1, -0.5, 0, 0.5, 1], xz: [-2, -1, 0, 1, 2], yz: [-3, 0, 3] };

function sectionFile(plane) {
  const stops = STOPS[plane];
  return { stops, slices: stops.map((position) => ({ plane, position, rings: {}, background: [] })) };
}

function worldFile(name) {
  const tab = name === "result" ? "result" : "layout";
  return {
    state: {
      styles: { core: { color: "#808080", opacity: 1, visible: true } },
      active_tab: tab,
      port_visible: { o1: true, o2: true },
      selection: null,
      active_viewport: "3d",
      show_full: false,
      plane: "xy",
      plane_position: 0,
      plane_bounds: [-1, 1],
      plane_fade_frac: 0.5,
      sources: [],
      detectors: name === "result" ? [{ id: "d1", type: "PhasorDetector", port: "None", params: {}, adopted: true }] : [],
      glyphs: name === "result" ? [{ id: "d1", name: "field", visible: true }] : [],
      field_settings: name === "result" ? { d1: { field_component: "Ex", field_transform: "abs", field_freq: 0 } } : {},
      simulation: {},
      running: false,
      readonly: { reason: "Precomputed demo" },
    },
    meta: { levels: [{ name: "core" }], ports: [], background: "clad", types: {}, settings_defaults: {} },
    meshes: {
      origin: [0, 0, 0],
      size: [1, 1, 1],
      levels: [{ name: "core", painted: mesh(0), full: mesh(1) }],
      background: mesh(2),
    },
  };
}

// each mesh: 3 float32 positions (12 bytes) then one uint32 triangle (12 bytes)
function mesh(i) {
  return { count: 1, tris: 1, positions: [i * 24, 12], indices: [i * 24 + 12, 12] };
}

function meshBytes() {
  const buffer = new ArrayBuffer(3 * 24);
  new Float32Array(buffer).forEach((_, k, a) => { a[k] = k; });
  return buffer;
}

function demo({ failing = new Set() } = {}) {
  const requested = [];
  const manifest = {
    format: FORMAT,
    worlds: { editor: { json_bytes: 1, mesh_bytes: 72 }, result: { json_bytes: 1, mesh_bytes: 72 } },
    fields: {
      small: { id: "small", name: "phasor", frames: 1, grid: [2, 2], bytes: { real: 8, imag: 8, json: 1 } },
      big: { id: "big", name: "movie", frames: 2, grid: [2, 2], bytes: { real: 16, json: 1 } },
    },
    sections: Object.fromEntries(
      Object.entries(STOPS).map(([plane, s]) => [plane, { stops: [s[0], s[s.length - 1]], count: s.length, step: s[1] - s[0], default: s[Math.floor(s.length / 2)], bytes: 1 }]),
    ),
    total_bytes: 100,
  };
  const quantized = (values) => new Uint16Array(values).buffer;
  const files = {
    "manifest.json": manifest,
    "editor.json": worldFile("editor"),
    "result.json": worldFile("result"),
    "sections/xy.json": sectionFile("xy"),
    "sections/xz.json": sectionFile("xz"),
    "sections/yz.json": sectionFile("yz"),
    "fields/small.json": { id: "small", name: "phasor", components: ["Ex"], grid: [2, 2], frame_kind: "freq", freqs: [1], quant: { real: [0, 6.5535], imag: [-1, 1] } },
    "fields/small.real.bin": quantized([0, 10000, 20000, 65535]),
    "fields/small.imag.bin": quantized([0, 0, 65535, 65535]),
    "fields/big.json": { id: "big", name: "movie", components: ["Ex"], grid: [2, 2], frame_kind: "time", times: [0, 1], quant: { real: [0, 1] } },
    "fields/big.real.bin": quantized([0, 1, 2, 3, 4, 5, 6, 65535]),
    "editor.meshes.bin": meshBytes(),
    "result.meshes.bin": meshBytes(),
  };
  const src = {
    async json(path) {
      requested.push(path);
      if (failing.has(path)) throw new Error(`could not load ${path} (404)`);
      return structuredClone(files[path]);
    },
    async bytes(path) {
      requested.push(path);
      if (failing.has(path)) throw new Error(`could not load ${path} (404)`);
      return files[path].slice(0);
    },
  };
  return { src, requested };
}

function record(model, ...keys) {
  const seen = [];
  for (const key of keys) model.on(`change:${key}`, () => seen.push(key));
  return seen;
}

// ------------------------------------------------------------------ helpers

test("dequantize maps uint16 back onto the stored range", () => {
  const out = new Float32Array(dequantize(new Uint16Array([0, 65535, 32768]).buffer, [-1, 1]));
  assert.equal(out[0], -1);
  assert.equal(out[1], 1);
  assert.ok(Math.abs(out[2] - 0.0000153) < 1e-4);
});

test("nearestStop snaps to the evenly spaced stored positions and clamps", () => {
  const stops = [-1, -0.5, 0, 0.5, 1];
  assert.equal(nearestStop(stops, 0.26), 3);
  assert.equal(nearestStop(stops, 0.24), 2);
  assert.equal(nearestStop(stops, -50), 0);
  assert.equal(nearestStop(stops, 50), 4);
});

// ------------------------------------------------------------------ opening

test("open loads the result world: geometry, state, section, no fields yet", async () => {
  const { src, requested } = demo();
  const model = await StaticModel.open(src);
  const state = model.get("state");
  assert.equal(state.active_tab, "result");
  assert.deepEqual(state.readonly, { reason: "Precomputed demo" });
  assert.deepEqual(state.plane_stops, STOPS.xy);
  assert.deepEqual(state.plane_bounds, [-1, 1]);
  assert.equal(model.get("section").plane, "xy");
  assert.equal(model.get("geometry").levels[0].name, "core");
  assert.equal(model.get("geometry").levels[0].full.count, 1);
  assert.deepEqual(model.get("fields"), {});
  // only the shown world's meshes and the shown plane's slices are fetched up front
  assert.ok(requested.includes("result.meshes.bin"));
  assert.ok(!requested.includes("editor.meshes.bin"));
  assert.ok(!requested.includes("sections/xz.json"));
  assert.ok(!requested.some((p) => p.startsWith("fields/")));
});

test("open refuses data of another format", async () => {
  const { src } = demo();
  const original = src.json;
  src.json = async (path) => (path === "manifest.json" ? { ...(await original(path)), format: 99 } : original(path));
  await assert.rejects(StaticModel.open(src), /format 99/);
});

test("mesh buffers are cut at their offsets", async () => {
  const model = await StaticModel.open(demo().src);
  const level = model.get("geometry").levels[0];
  assert.equal(level.painted.positions.byteLength, 12);
  assert.deepEqual(Array.from(new Float32Array(level.painted.positions)), [0, 1, 2]);
  assert.deepEqual(Array.from(new Float32Array(level.full.positions)), [6, 7, 8]);
});

// ------------------------------------------------------------------ the slice slider

test("a plane position snaps to a stored one and the slice follows", async () => {
  const model = await StaticModel.open(demo().src);
  const seen = record(model, "state", "section");
  model.send({ event: "set_state", patch: { plane_position: 0.3 } });
  await model.idle();
  assert.equal(model.get("state").plane_position, 0.5);
  assert.equal(model.get("section").position, 0.5);
  assert.deepEqual(seen, ["state", "section"]);
});

test("a position past the ends clamps to the first and last stop", async () => {
  const model = await StaticModel.open(demo().src);
  model.send({ event: "set_state", patch: { plane_position: 99 } });
  await model.idle();
  assert.equal(model.get("state").plane_position, 1);
  model.send({ event: "set_state", patch: { plane_position: -99 } });
  await model.idle();
  assert.equal(model.get("state").plane_position, -1);
});

test("changing the plane loads its slices once and uses that plane's stops", async () => {
  const { src, requested } = demo();
  const model = await StaticModel.open(src);
  model.send({ event: "set_state", patch: { plane: "yz" } });
  await model.idle();
  const state = model.get("state");
  assert.equal(state.plane, "yz");
  assert.deepEqual(state.plane_stops, STOPS.yz);
  assert.equal(state.plane_position, 0); // the plane's own default
  assert.equal(model.get("section").plane, "yz");
  model.send({ event: "set_state", patch: { plane: "xy" } });
  model.send({ event: "set_state", patch: { plane: "yz" } });
  await model.idle();
  assert.equal(requested.filter((p) => p === "sections/yz.json").length, 1);
});

test("each plane remembers its own position", async () => {
  const model = await StaticModel.open(demo().src);
  model.send({ event: "set_state", patch: { plane_position: 1 } });
  model.send({ event: "set_state", patch: { plane: "xz", plane_position: 2 } });
  model.send({ event: "set_state", patch: { plane: "xy" } });
  await model.idle();
  assert.equal(model.get("state").plane_position, 1);
});

// ------------------------------------------------------------------ display edits

test("styles, port visibility, selection, viewport and fade are applied", async () => {
  const model = await StaticModel.open(demo().src);
  model.send({
    event: "set_state",
    patch: {
      styles: { core: { opacity: 0.25, visible: false, bogus: 1 }, nope: { opacity: 1 } },
      port_visible: { o1: false, nope: false },
      selection: { kind: "port", name: "o2" },
      active_viewport: "2d",
      show_full: true,
      plane_fade_frac: 0.9,
    },
  });
  await model.idle();
  const state = model.get("state");
  assert.deepEqual(state.styles.core, { color: "#808080", opacity: 0.25, visible: false });
  assert.ok(!("nope" in state.styles));
  assert.deepEqual(state.port_visible, { o1: false, o2: true });
  assert.deepEqual(state.selection, { kind: "port", name: "o2" });
  assert.equal(state.active_viewport, "2d");
  assert.equal(state.show_full, true);
  assert.equal(state.plane_fade_frac, 0.9);
});

test("a field-display edit from the Result tab updates that detector's settings", async () => {
  const model = await StaticModel.open(demo().src);
  model.send({ event: "update_spec", kind: "detector", spec: { id: "d1", params: { field_transform: "phase", field_freq: 3 } } });
  await model.idle();
  const settings = model.get("state").field_settings.d1;
  assert.equal(settings.field_transform, "phase");
  assert.equal(settings.field_freq, 3);
  assert.equal(settings.field_component, "Ex"); // untouched settings stay
});

test("hiding a detector hides its glyph", async () => {
  const model = await StaticModel.open(demo().src);
  model.send({ event: "update_spec", kind: "detector", spec: { id: "d1", params: { visible: false } } });
  await model.idle();
  assert.equal(model.get("state").glyphs[0].visible, false);
});

test("an edit of anything but display settings is refused and the state is re-sent unchanged", async () => {
  const model = await StaticModel.open(demo().src);
  const before = JSON.stringify(model.get("state"));
  const seen = record(model, "state");
  for (const content of [
    { event: "update_spec", kind: "detector", spec: { id: "d1", params: { port_offset: 1 } } },
    { event: "update_spec", kind: "detector", spec: { id: "nope", params: { visible: false } } },
    { event: "run" },
    { event: "stop" },
    { event: "add_source", port: "o1" },
    { event: "set_settings", settings: { resolution_um: 1 } },
    { event: "save_video", filename: "x.mp4" },
  ]) {
    model.send(content);
  }
  await model.idle();
  assert.equal(JSON.stringify(model.get("state")), before);
  assert.equal(seen.length, 7, "each refusal re-sends the state so the browser's controls snap back");
});

// ------------------------------------------------------------------ worlds

test("switching to a setup tab sends the editor world, loading its meshes then", async () => {
  const { src, requested } = demo();
  const model = await StaticModel.open(src);
  const seen = record(model, "state", "geometry", "meta", "section", "fields");
  model.send({ event: "set_state", patch: { active_tab: "layout" } });
  await model.idle();
  assert.equal(model.get("state").active_tab, "layout");
  assert.deepEqual(model.get("state").detectors, []);
  assert.ok(requested.includes("editor.meshes.bin"));
  assert.deepEqual(seen, ["state", "geometry", "meta", "section", "fields"], "the state goes first");
  model.send({ event: "set_state", patch: { active_tab: "result" } });
  await model.idle();
  assert.equal(model.get("state").detectors.length, 1);
  assert.equal(model.get("state").active_tab, "result");
});

test("moving between setup tabs keeps the editor world and only echoes the tab", async () => {
  const model = await StaticModel.open(demo().src);
  model.send({ event: "set_state", patch: { active_tab: "layout" } });
  await model.idle();
  const seen = record(model, "geometry");
  model.send({ event: "set_state", patch: { active_tab: "sources" } });
  await model.idle();
  assert.equal(model.get("state").active_tab, "sources");
  assert.deepEqual(seen, []);
});

// ------------------------------------------------------------------ fields and status

test("fields load smallest first, each announced, and are decoded to float32", async () => {
  const model = await StaticModel.open(demo().src);
  const seen = record(model, "fields");
  const statuses = [];
  model.on("change:static_status", () => statuses.push([...model.get("static_status").busy]));
  await model.loadFields();
  assert.deepEqual(Object.keys(model.get("fields")), ["small", "big"]);
  assert.equal(seen.length, 2);
  const small = model.get("fields").small;
  assert.deepEqual(small.components, ["Ex"]);
  assert.equal(small.real.byteLength, 16);
  const real = new Float32Array(small.real);
  assert.ok(Math.abs(real[3] - 6.5535) < 1e-4);
  assert.ok(Math.abs(real[1] - 1) < 1e-3);
  assert.ok(small.imag instanceof ArrayBuffer);
  assert.ok(!("quant" in small));
  assert.equal(model.get("fields").big.frame_kind, "time");
  assert.ok(!("imag" in model.get("fields").big));
  assert.ok(statuses.some((busy) => busy.some((label) => label.includes("movie"))), "a chip names the field loading");
  assert.deepEqual(model.get("static_status"), { busy: [], error: null });
});

test("a field that fails to load reports the error and the others still load", async () => {
  const model = await StaticModel.open(demo({ failing: new Set(["fields/small.real.bin"]) }).src);
  await assert.rejects(model.loadFields(), /small.real.bin/);
  assert.match(model.get("static_status").error, /small\.real\.bin/);
});

test("slices that fail to load report the error and can be retried", async () => {
  const failing = new Set(["sections/yz.json"]);
  const { src } = demo({ failing });
  const model = await StaticModel.open(src);
  model.send({ event: "set_state", patch: { plane: "yz" } });
  await model.idle();
  assert.match(model.get("static_status").error, /yz\.json/);
  assert.equal(model.get("state").plane, "xy", "the state did not move to a plane with no slices");
  failing.clear();
  model.send({ event: "set_state", patch: { plane: "yz" } });
  await model.idle();
  assert.equal(model.get("state").plane, "yz");
});

test("off stops a listener", async () => {
  const model = await StaticModel.open(demo().src);
  let calls = 0;
  const handler = () => { calls += 1; };
  model.on("change:state", handler);
  model.off("change:state", handler);
  model.send({ event: "run" });
  await model.idle();
  assert.equal(calls, 0);
});
