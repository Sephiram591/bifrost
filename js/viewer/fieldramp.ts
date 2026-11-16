// The detector field overlay's color ramp (spec R15, D16).
//
// Color and alpha both come from one map: the transformed component is placed
// on the plane's own range and read through a two-color ramp with a transparent
// region. Signed transforms (real, imag, phase) use color → transparent → color,
// so zero reads as "nothing here"; non-negative transforms (abs, logabs) use
// transparent → color, a single hue fading up from the low end. Alpha is the map
// alone, never a separate function of magnitude (spec D16).
//
// The ramp spans one fixed range for every frame of a detector — the user's
// `field_range`, or by default the min and max over the whole recording — so a
// multi-frame detector (a phasor's frequencies, a FieldDetector's time steps)
// keeps one brightness scale while it plays or is saved as a video.
//
// `logabs` is the one transform whose floor is not the plane's own minimum: it
// floors at the global `log_range` decades below the plane's maximum, so a true
// zero maps to the floor rather than to −∞ (spec D16, Q22).
import type { FieldPayload, FieldSettings } from "./wire";

export type Transform = FieldSettings["field_transform"];

//: A pseudo-component: the total electric-field magnitude, ``|E|`` (spec R15).
//: The result records ``Ex``/``Ey``/``Ez`` as separate complex planes; this
//: offers their magnitude ``sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)`` as one choice, so a
//: detector can be shaded by the field's strength rather than one polarization.
export const TOTAL_COMPONENT = "|E|";

const ELECTRIC_COMPONENTS: ReadonlySet<string> = new Set(["Ex", "Ey", "Ez"]);

/** Whether any electric-field Cartesian component (Ex, Ey, Ez) is recorded. */
export function hasElectricComponent(components: readonly string[]): boolean {
  return components.some((name) => ELECTRIC_COMPONENTS.has(name));
}

/** Component dropdown choices: prepends ``|E|`` only when E components exist and it is not already present. */
export function componentOptions(components: readonly string[]): string[] {
  const prependTotal = hasElectricComponent(components) && !components.includes(TOTAL_COMPONENT);
  return ["", ...(prependTotal ? [TOTAL_COMPONENT] : []), ...components];
}

//: the three transforms whose values straddle zero (spec D16)
const SIGNED_TRANSFORMS: ReadonlySet<Transform> = new Set(["real", "imag", "phase"]);

export function isSigned(transform: Transform): boolean {
  return SIGNED_TRANSFORMS.has(transform);
}

/**
 * One component's complex values, or null if the component is not recorded.
 *
 * ``TOTAL_COMPONENT`` returns the total electric-field magnitude as a real
 * value (imaginary part zero): the Cartesian components are orthogonal, so
 * ``|E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2``. Only ``Ex``, ``Ey``, ``Ez`` contribute;
 * returns null if none are recorded.
 *
 * The buffers are frame-major stacks (spec R17): one frame per recorded
 * frequency or time step. ``frame`` selects one, clamped to what the buffers
 * hold, so a stale `field_freq` from a previous result cannot read out of
 * bounds. A real-valued recording (no ``imag`` buffer) reads as imaginary 0.
 */
export function componentData(
  field: FieldPayload,
  component: string,
  frame = 0,
): { real: Float32Array; imag: Float32Array } | null {
  const [rows, cols] = field.grid;
  const count = rows * cols;
  const perFrame = field.components.length * count;
  const f = Math.min(Math.max(0, Math.floor(frame)), frameCount(field) - 1);
  const base = f * perFrame;
  const imagAt = (offset: number): Float32Array =>
    field.imag ? decodeFloat32(field.imag, offset, count) : new Float32Array(count);
  const index = field.components.indexOf(component);
  if (index >= 0) {
    return {
      real: decodeFloat32(field.real, base + index * count, count),
      imag: imagAt(base + index * count),
    };
  }
  if (component === TOTAL_COMPONENT) {
    const indices: number[] = [];
    for (let i = 0; i < field.components.length; i += 1) {
      if (ELECTRIC_COMPONENTS.has(field.components[i])) indices.push(i);
    }
    if (indices.length === 0) return null;
    const real = new Float32Array(count);
    for (const i of indices) {
      const re = decodeFloat32(field.real, base + i * count, count);
      const im = imagAt(base + i * count);
      for (let k = 0; k < count; k += 1) real[k] += re[k] * re[k] + im[k] * im[k];
    }
    for (let k = 0; k < count; k += 1) real[k] = Math.sqrt(real[k]);
    return { real, imag: new Float32Array(count) };
  }
  return null;
}

/** How many frames a payload's buffers hold (at least 1). */
export function frameCount(field: FieldPayload): number {
  const [rows, cols] = field.grid;
  const perFrame = field.components.length * rows * cols;
  const total = Math.floor(field.real.byteLength / 4);
  return perFrame > 0 ? Math.max(1, Math.floor(total / perFrame)) : 1;
}

//: whole-recording ranges, per payload (a new result is a new object, so its
//: entries go with it) and per component/transform/log range
const RANGE_CACHE = new WeakMap<FieldPayload, Map<string, [number, number]>>();

/**
 * The value range over **every** frame of a payload (spec R15, D16).
 *
 * The default colour scale: one range for the whole recording, so each frame
 * is shaded on the same scale and playback does not flicker. ``logabs`` floors
 * `log_range` decades below the recording's maximum. Cached, since a
 * time-domain recording can hold many frames.
 */
export function frameRange(
  field: FieldPayload,
  component: string,
  transform: Transform,
  logRange: number,
): [number, number] {
  const key = `${component}\u0000${transform}\u0000${logRange}`;
  let cache = RANGE_CACHE.get(field);
  const hit = cache?.get(key);
  if (hit) return hit;
  let min = Infinity;
  let max = -Infinity;
  for (let f = 0; f < frameCount(field); f += 1) {
    const data = componentData(field, component, f);
    if (!data) break;
    const [lo, hi] = extent(transformValues(data.real, data.imag, transform));
    if (lo < min) min = lo;
    if (hi > max) max = hi;
  }
  const range = clampRange(min, max, transform, logRange);
  if (!cache) {
    cache = new Map();
    RANGE_CACHE.set(field, cache);
  }
  cache.set(key, range);
  return range;
}

/** The range a detector is shaded on: the user's fixed one, else the recording's. */
export function shadeRange(
  field: FieldPayload,
  settings: FieldSettings,
  logRange: number,
): [number, number] {
  const fixed = settings.field_range;
  if (fixed && Number.isFinite(fixed[0]) && Number.isFinite(fixed[1]) && fixed[1] > fixed[0]) {
    return [fixed[0], fixed[1]];
  }
  return frameRange(field, settings.field_component ?? "", settings.field_transform, logRange);
}

/** A component's transformed values, in place order (spec R15). */
export function transformValues(
  real: Float32Array,
  imag: Float32Array,
  transform: Transform,
): Float32Array {
  const out = new Float32Array(real.length);
  for (let i = 0; i < real.length; i += 1) {
    const re = real[i];
    const im = imag[i];
    switch (transform) {
      case "real":
        out[i] = re;
        break;
      case "imag":
        out[i] = im;
        break;
      case "phase":
        out[i] = Math.atan2(im, re);
        break;
      case "logabs":
        out[i] = Math.log10(Math.hypot(re, im) + Number.MIN_VALUE);
        break;
      default:
        out[i] = Math.hypot(re, im);
    }
  }
  return out;
}

/** The value range the ramp spans, honoring the logabs floor (spec D16, Q22). */
export function valueRange(values: Float32Array, transform: Transform, logRange: number): [number, number] {
  const [min, max] = extent(values);
  return clampRange(min, max, transform, logRange);
}

/** The min and max of finite values (±Infinity when there are none). */
function extent(values: Float32Array): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return [min, max];
}

/** A raw min/max made into a usable ramp range (symmetric signed zero, logabs floor, empty/zero guard). */
function clampRange(min: number, max: number, transform: Transform, logRange: number): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 0];
  if (isSigned(transform)) {
    // signed ramps place transparent zero at t = 0.5, so keep [-vmax, +vmax]
    // symmetric and return [0, 0] for an all-zero plane so alpha stays 0
    const vmax = Math.max(Math.abs(min), Math.abs(max));
    return vmax === 0 ? [0, 0] : [-vmax, vmax];
  }
  if (transform === "logabs") {
    // the floor is `log_range` decades below the maximum (spec D16); the span
    // never shrinks below it, so a weak plane still reads across the ramp
    const floor = max - Math.abs(logRange);
    if (min < floor) min = floor;
  }
  if (min === 0 && max === 0) return [0, 0];
  if (max <= min) max = min + 1e-12;
  return [min, max];
}

/** Sample one transformed scalar on ``range`` into RGBA, keeping non-finite or zero-span cells transparent. */
export function rampSample(
  value: number,
  range: [number, number],
  transform: Transform,
  low: [number, number, number],
  high: [number, number, number],
): [number, number, number, number] {
  if (!Number.isFinite(value)) return [0, 0, 0, 0];
  const span = range[1] - range[0];
  if (!Number.isFinite(span) || span <= 0) return [0, 0, 0, 0];
  return rampColor((value - range[0]) / span, transform, low, high);
}

/**
 * Builds the RGBA texture bytes for a plane's transformed field (spec D16).
 *
 * `low`/`high` are the two endpoint colors; `low` is unused by a non-negative
 * transform, whose transparent end stands in for it.
 */
export function rampTexture(
  values: Float32Array,
  range: [number, number],
  transform: Transform,
  low: [number, number, number],
  high: [number, number, number],
  opacity: number,
): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  for (let i = 0; i < values.length; i += 1) {
    const [r, g, b, a] = rampSample(values[i], range, transform, low, high);
    bytes[i * 4] = r;
    bytes[i * 4 + 1] = g;
    bytes[i * 4 + 2] = b;
    bytes[i * 4 + 3] = Math.round(a * opacity * 255);
  }
  return bytes;
}

/** `#rrggbb` to an RGB triple; a bad value falls back to mid grey. */
export function parseColor(text: string): [number, number, number] {
  const match = /^#?([0-9a-f]{6})$/i.exec(text.trim());
  if (!match) return [128, 128, 128];
  const value = parseInt(match[1], 16);
  return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff];
}

/** The value at a texture texel's color, for the colorbar (spec R15). */
export function rampColor(
  t: number,
  transform: Transform,
  low: [number, number, number],
  high: [number, number, number],
): [number, number, number, number] {
  if (!Number.isFinite(t)) return [0, 0, 0, 0];
  const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
  if (isSigned(transform)) {
    return [
      mix(low[0], high[0], clamped),
      mix(low[1], high[1], clamped),
      mix(low[2], high[2], clamped),
      Math.abs(2 * clamped - 1),
    ];
  }
  return [high[0], high[1], high[2], clamped];
}

function mix(from: number, to: number, t: number): number {
  return from + (to - from) * t;
}

/** A typed-array view over a DataView/ArrayBuffer, copying if misaligned. */
function decodeFloat32(buffer: DataView | ArrayBuffer, offset: number, count: number): Float32Array {
  const source = buffer instanceof ArrayBuffer ? buffer : (buffer.buffer as ArrayBuffer);
  const base = buffer instanceof ArrayBuffer ? 0 : buffer.byteOffset;
  const start = base + offset * 4;
  if (start % 4 === 0) return new Float32Array(source, start, count);
  return new Float32Array(source.slice(start, start + count * 4));
}
