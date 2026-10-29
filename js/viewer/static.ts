// A kernel-free stand-in for the anywidget model, for the precomputed demo.
//
// The live viewer asks a Python kernel for everything: geometry, state, 2D
// slices, fields. `bifrost.viewer.export.export_demo` writes what the kernel
// would have sent as files, and `StaticModel` plays them back through the same
// `get` / `send` / `on` interface, so the front end runs unchanged. Anything
// that needs Python (a run, a setup edit, a saved video) is refused, and the
// state carries `readonly` so the browser says so.
//
// This file has no DOM or three.js imports: it only needs a `Source` that reads
// JSON and bytes, which the browser backs with `fetch` and the tests with a map.

import type { AnyModel } from "./index";
import type {
  FieldPayload,
  GeometryPayload,
  MeshPayload,
  MetaPayload,
  SectionPayload,
  StatePayload,
} from "./wire";

export const FORMAT = 1;

type Plane = "xy" | "xz" | "yz";
type Tab = StatePayload["active_tab"];
const PLANES: Plane[] = ["xy", "xz", "yz"];
const TABS: Tab[] = ["layout", "sources", "simulation", "result"];
const DEMO_READONLY = { reason: "Precomputed demo" };

/** Where a mesh's bytes sit in the world's `.meshes.bin`: `[offset, length]`. */
export interface MeshRef {
  count: number;
  tris: number;
  positions: [number, number];
  indices: [number, number];
}

export interface MeshIndex {
  origin: [number, number, number];
  size: [number, number, number];
  levels: { name: string; painted: MeshRef; full: MeshRef }[];
  background: MeshRef;
}

export interface WorldFile {
  state: StatePayload;
  meta: MetaPayload;
  meshes: MeshIndex;
}

export interface SectionIndexEntry {
  //: the first and last stored position
  stops: [number, number];
  count: number;
  step: number;
  //: the position a plane starts at (the box center)
  default: number;
  bytes: number;
}

export interface FieldIndexEntry {
  id: string;
  name: string;
  frames: number;
  grid: [number, number];
  bytes: Record<string, number>;
}

export interface Manifest {
  format: number;
  worlds: Record<string, { json_bytes: number; mesh_bytes: number }>;
  fields: Record<string, FieldIndexEntry>;
  sections: Record<Plane, SectionIndexEntry>;
  total_bytes: number;
  [key: string]: unknown;
}

export interface SectionFile {
  stops: number[];
  slices: SectionPayload[];
}

type FieldMeta = Omit<FieldPayload, "real" | "imag"> & {
  quant: { real?: [number, number]; imag?: [number, number] };
};

/** Reads the exported files. Paths are relative to the data directory. */
export interface Source {
  json<T>(path: string): Promise<T>;
  bytes(path: string): Promise<ArrayBuffer>;
}

/** What the page can show while data loads: a chip per pending load, and the last error. */
export interface DemoStatus {
  busy: string[];
  error: string | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Listener = (...args: any[]) => void;

interface World {
  file: WorldFile;
  geometry?: GeometryPayload;
  //: each plane's current position (the browser's slider)
  positions: Record<Plane, number>;
}

/** Decodes a uint16 buffer written against `[low, high]` into float32 samples. */
export function dequantize(buffer: ArrayBuffer, range: [number, number]): ArrayBuffer {
  const quantized = new Uint16Array(buffer);
  const out = new Float32Array(quantized.length);
  const [low, high] = range;
  const scale = (high - low) / 65535;
  for (let i = 0; i < quantized.length; i += 1) out[i] = low + quantized[i] * scale;
  return out.buffer;
}

/** The index of the stored position nearest `position`, for evenly spaced `stops`. */
export function nearestStop(stops: number[], position: number): number {
  if (stops.length < 2) return 0;
  const step = (stops[stops.length - 1] - stops[0]) / (stops.length - 1);
  const index = Math.round((position - stops[0]) / step);
  return Math.min(stops.length - 1, Math.max(0, index));
}

function slice(buffer: ArrayBuffer, [offset, length]: [number, number]): ArrayBuffer {
  return buffer.slice(offset, offset + length);
}

function toMesh(buffer: ArrayBuffer, ref: MeshRef): MeshPayload {
  return {
    count: ref.count,
    tris: ref.tris,
    positions: slice(buffer, ref.positions),
    indices: slice(buffer, ref.indices),
  };
}

export class StaticModel implements AnyModel {
  private traits = new Map<string, unknown>();
  private listeners = new Map<string, Set<Listener>>();
  private worlds: Record<"editor" | "result", World>;
  private tab: Tab;
  private sections = new Map<Plane, Promise<SectionFile>>();
  private fields: Record<string, FieldPayload> = {};
  private status: DemoStatus = { busy: [], error: null };
  private chain: Promise<void> = Promise.resolve();

  private constructor(
    private readonly source: Source,
    readonly manifest: Manifest,
    worlds: Record<"editor" | "result", WorldFile>,
  ) {
    const positions = (file: WorldFile): Record<Plane, number> => {
      const out = {} as Record<Plane, number>;
      for (const plane of PLANES) out[plane] = manifest.sections[plane].default;
      out[file.state.plane] = file.state.plane_position;
      return out;
    };
    this.worlds = {
      editor: { file: worlds.editor, positions: positions(worlds.editor) },
      result: { file: worlds.result, positions: positions(worlds.result) },
    };
    this.tab = worlds.result.state.active_tab;
  }

  /** Reads the manifest and the initial world, ready for `render`. Fields keep loading afterwards. */
  static async open(source: Source): Promise<StaticModel> {
    const manifest = await source.json<Manifest>("manifest.json");
    if (manifest.format !== FORMAT) {
      throw new Error(`this demo's data is format ${manifest.format}; the viewer reads format ${FORMAT}`);
    }
    const [editor, result] = await Promise.all([
      source.json<WorldFile>("editor.json"),
      source.json<WorldFile>("result.json"),
    ]);
    const model = new StaticModel(source, manifest, { editor, result });
    await model.enter(model.tab, true);
    for (const key of ["phase", "save_dir", "launch_code"]) model.traits.set(key, "");
    for (const key of ["progress", "save_result"]) model.traits.set(key, {});
    model.traits.set("stopping", false);
    model.traits.set("run_result", { status: "done", detectors: Object.values(manifest.fields).map((f) => f.name) });
    model.traits.set("static_status", model.status);
    return model;
  }

  // ---------------------------------------------------------------- the model

  get(key: string): unknown {
    return this.traits.get(key);
  }

  set(key: string, value: unknown): void {
    this.traits.set(key, value);
  }

  save_changes(): void {}

  on(event: "msg:custom", handler: (content: Record<string, unknown>, buffers?: DataView[]) => void): void;
  on(event: string, handler: () => void): void;
  on(event: string, handler: Listener): void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(handler);
  }

  off(event: string, handler?: unknown): void {
    if (handler === undefined) this.listeners.delete(event);
    else this.listeners.get(event)?.delete(handler as Listener);
  }

  /** Messages from the browser: display edits are applied here, everything else needs Python. */
  send(content: Record<string, unknown>): void {
    const event = content?.event;
    this.enqueue(async () => {
      if (event === "set_state") await this.applyPatch((content.patch ?? {}) as Record<string, unknown>);
      else if (event === "update_spec") this.updateSpec(content);
      else this.refuse();
    });
  }

  // ---------------------------------------------------------------- loading

  /** Loads every field in the background, smallest first, announcing each on the status. */
  async loadFields(): Promise<void> {
    const entries = Object.values(this.manifest.fields).sort(
      (a, b) => a.bytes.real + (a.bytes.imag ?? 0) - (b.bytes.real + (b.bytes.imag ?? 0)),
    );
    for (const entry of entries) {
      const mb = ((entry.bytes.real + (entry.bytes.imag ?? 0)) / 1e6).toFixed(1);
      await this.track(`field ${entry.name} (${mb} MB)`, async () => {
        const meta = await this.source.json<FieldMeta>(`fields/${entry.id}.json`);
        const payload = { ...meta } as FieldPayload & { quant?: unknown };
        delete payload.quant;
        payload.real = dequantize(await this.source.bytes(`fields/${entry.id}.real.bin`), meta.quant.real!);
        if (meta.quant.imag) {
          payload.imag = dequantize(await this.source.bytes(`fields/${entry.id}.imag.bin`), meta.quant.imag);
        }
        this.fields = { ...this.fields, [entry.id]: payload };
        this.emitSet("fields", this.fields);
      });
    }
  }

  /** Resolves once everything queued by the browser so far has been applied (for tests). */
  idle(): Promise<void> {
    return this.chain;
  }

  private loadSection(plane: Plane): Promise<SectionFile> {
    let loaded = this.sections.get(plane);
    if (!loaded) {
      loaded = this.track(`${plane} slices`, () => this.source.json<SectionFile>(`sections/${plane}.json`));
      this.sections.set(plane, loaded);
      loaded.catch(() => this.sections.delete(plane)); // a failed load may be retried
    }
    return loaded;
  }

  private async loadGeometry(name: "editor" | "result"): Promise<GeometryPayload> {
    const world = this.worlds[name];
    if (world.geometry) return world.geometry;
    const buffer = await this.track(`${name} geometry`, () => this.source.bytes(`${name}.meshes.bin`));
    const index = world.file.meshes;
    world.geometry = {
      origin: index.origin,
      size: index.size,
      levels: index.levels.map((level) => ({
        name: level.name,
        painted: toMesh(buffer, level.painted),
        full: toMesh(buffer, level.full),
      })),
      background: toMesh(buffer, index.background),
    };
    return world.geometry;
  }

  // ---------------------------------------------------------------- state

  private get activeName(): "editor" | "result" {
    return this.tab === "result" ? "result" : "editor";
  }

  /** The active world's state as the wire carries it: its own, with the tab echoed. */
  private wireState(world: World, plane: Plane, stops: number[]): StatePayload {
    const state = { ...world.file.state };
    state.active_tab = this.tab;
    state.plane = plane;
    state.plane_position = world.positions[plane];
    state.plane_bounds = [stops[0], stops[stops.length - 1]];
    state.plane_stops = stops;
    state.readonly = state.readonly ?? DEMO_READONLY;
    return state;
  }

  /** Makes `tab` current: sends the world's traits, loading its geometry and slices first. */
  private async enter(tab: Tab, first = false): Promise<void> {
    this.tab = tab;
    const name = this.activeName;
    const world = this.worlds[name];
    const plane = world.file.state.plane;
    const [geometry, section] = await Promise.all([this.loadGeometry(name), this.loadSection(plane)]);
    const state = this.wireState(world, plane, section.stops);
    world.file.state = { ...state };
    this.traits.set("static_status", this.status);
    if (first) {
      this.traits.set("state", state);
      this.traits.set("geometry", geometry);
      this.traits.set("meta", world.file.meta);
      this.traits.set("section", section.slices[nearestStop(section.stops, state.plane_position)]);
      this.traits.set("fields", this.fields);
      return;
    }
    // the state goes first, so a rebuild sees the styles the new world ships with
    this.emitSet("state", state);
    this.emitSet("geometry", geometry);
    this.emitSet("meta", world.file.meta);
    this.emitSet("section", section.slices[nearestStop(section.stops, state.plane_position)]);
    this.emitSet("fields", this.fields);
  }

  private async applyPatch(patch: Record<string, unknown>): Promise<void> {
    const rest = { ...patch };
    const tab = rest.active_tab;
    delete rest.active_tab;
    if (typeof tab === "string" && TABS.includes(tab as Tab) && tab !== this.tab) {
      const before = this.activeName;
      this.tab = tab as Tab;
      if (this.activeName !== before) {
        await this.enter(this.tab);
        return;
      }
      // layout, sources and simulation share the editor world: only the tab echo changes
    }
    const world = this.worlds[this.activeName];
    const state: StatePayload = { ...world.file.state };
    let plane: Plane = state.plane;
    let moved = false;

    if (isRecord(rest.styles)) {
      state.styles = { ...state.styles };
      for (const [name, style] of Object.entries(rest.styles)) {
        if (name in state.styles && isRecord(style)) state.styles[name] = { ...state.styles[name], ...pick(style, ["color", "opacity", "visible"]) } as StatePayload["styles"][string];
      }
    }
    if (isRecord(rest.port_visible)) {
      state.port_visible = { ...state.port_visible };
      for (const [name, visible] of Object.entries(rest.port_visible)) {
        if (name in state.port_visible) state.port_visible[name] = Boolean(visible);
      }
    }
    if ("selection" in rest) state.selection = (rest.selection ?? null) as StatePayload["selection"];
    if (rest.active_viewport === "3d" || rest.active_viewport === "2d") state.active_viewport = rest.active_viewport;
    if (typeof rest.show_full === "boolean") state.show_full = rest.show_full;
    if (typeof rest.plane_fade_frac === "number" && rest.plane_fade_frac > 0) state.plane_fade_frac = rest.plane_fade_frac;
    if (typeof rest.plane === "string" && PLANES.includes(rest.plane as Plane) && rest.plane !== plane) {
      plane = rest.plane as Plane;
      moved = true;
    }
    const section = await this.loadSection(plane);
    if (typeof rest.plane_position === "number" && Number.isFinite(rest.plane_position)) {
      world.positions[plane] = section.stops[nearestStop(section.stops, rest.plane_position)];
      moved = true;
    } else if (moved) {
      world.positions[plane] = section.stops[nearestStop(section.stops, world.positions[plane])];
    }
    const stops = section.stops;
    const next: StatePayload = {
      ...state,
      active_tab: this.tab,
      plane,
      plane_position: world.positions[plane],
      plane_bounds: [stops[0], stops[stops.length - 1]],
      plane_stops: stops,
      readonly: state.readonly ?? DEMO_READONLY,
    };
    world.file.state = { ...next };
    this.emitSet("state", next);
    if (moved) this.emitSet("section", section.slices[nearestStop(section.stops, next.plane_position)]);
  }

  /** A display edit from the Result tab: a field setting or a detector's visibility. */
  private updateSpec(content: Record<string, unknown>): void {
    const spec = content.spec;
    const kind = content.kind === "source" ? "sources" : "detectors";
    if (this.tab !== "result" || !isRecord(spec) || typeof spec.id !== "string") return this.refuse();
    const params = isRecord(spec.params) ? spec.params : {};
    if (Object.keys(params).some((key) => key !== "visible" && !key.startsWith("field_"))) return this.refuse();
    const world = this.worlds.result;
    const state: StatePayload = { ...world.file.state };
    const id = spec.id;
    if (!state[kind].some((s) => s.id === id)) return this.refuse();
    state[kind] = state[kind].map((s) => (s.id === id ? { ...s, params: { ...s.params, ...params } } : s));
    const fieldKeys = Object.fromEntries(Object.entries(params).filter(([key]) => key.startsWith("field_")));
    if (Object.keys(fieldKeys).length) {
      state.field_settings = { ...state.field_settings, [id]: { ...state.field_settings?.[id], ...fieldKeys } as StatePayload["field_settings"][string] };
    }
    if ("visible" in params) {
      state.glyphs = state.glyphs.map((g) => (g.id === id ? { ...g, visible: Boolean(params.visible) } : g));
    }
    world.file.state = state;
    this.emitSet("state", this.wireState(world, state.plane, this.currentStops(world, state.plane)));
  }

  private currentStops(world: World, plane: Plane): number[] {
    const stops = world.file.state.plane_stops;
    if (stops && world.file.state.plane === plane) return stops;
    const entry = this.manifest.sections[plane];
    return Array.from({ length: entry.count }, (_, k) => entry.stops[0] + k * entry.step);
  }

  /** A message that needs Python: nothing changes, and the browser's controls snap back. */
  private refuse(): void {
    const world = this.worlds[this.activeName];
    const stops = this.currentStops(world, world.file.state.plane);
    this.emitSet("state", this.wireState(world, world.file.state.plane, stops));
  }

  // ---------------------------------------------------------------- plumbing

  private enqueue(task: () => Promise<void>): void {
    this.chain = this.chain.then(task).catch((error: unknown) => this.fail(error));
  }

  private async track<T>(label: string, task: () => Promise<T>): Promise<T> {
    this.setStatus({ busy: [...this.status.busy, label], error: this.status.error });
    try {
      return await task();
    } catch (error) {
      this.fail(error);
      throw error;
    } finally {
      this.setStatus({ busy: this.status.busy.filter((b) => b !== label), error: this.status.error });
    }
  }

  private fail(error: unknown): void {
    this.setStatus({ busy: this.status.busy, error: error instanceof Error ? error.message : String(error) });
  }

  private setStatus(status: DemoStatus): void {
    this.status = status;
    this.emitSet("static_status", status);
  }

  private emitSet(key: string, value: unknown): void {
    this.traits.set(key, value);
    for (const handler of this.listeners.get(`change:${key}`) ?? []) handler();
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pick(source: Record<string, unknown>, keys: string[]): Record<string, unknown> {
  return Object.fromEntries(Object.entries(source).filter(([key]) => keys.includes(key)));
}

/** The browser's `Source`: JSON and bytes from `base` (a URL ending in a slash) with `fetch`. */
export function fetchSource(base: string, fetchImpl: typeof fetch = fetch): Source {
  const get = async (path: string): Promise<Response> => {
    const response = await fetchImpl(base + path);
    if (!response.ok) throw new Error(`could not load ${path} (${response.status})`);
    return response;
  };
  return {
    json: async <T,>(path: string): Promise<T> => (await get(path)).json() as Promise<T>,
    bytes: async (path: string): Promise<ArrayBuffer> => (await get(path)).arrayBuffer(),
  };
}
