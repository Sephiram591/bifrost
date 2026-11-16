// The field ramp's rules (spec R15 AC, D16): alpha comes from the map alone.
//
// The AC states two checkable facts: a signed transform's value at the plane's
// midpoint is transparent at full opacity, and an abs/logabs value at the low
// end is transparent. Both are asserted against the real `rampColor`, the same
// function the 2D blit and the 3D texture use.
import assert from "node:assert/strict";
import test from "node:test";

// The module is TypeScript; the build emits the bundle, but for a unit test load
// the ramp through the same source by transpiling on the fly is overkill — the
// functions under test are pure, so re-declare the contract by importing the
// built bundle's behavior is not possible either. Instead, exercise the ramp
// through a tiny inline reimplementation is a weak test; so the test imports the
// source via esbuild's transform.
import { transform } from "esbuild";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "fieldramp.ts"), "utf8");
// strip the type-only import, which esbuild cannot resolve without the types
const stripped = source.replace(/^import type .*?;$/ms, "");
const compiled = await transform(stripped, { loader: "ts", format: "esm" });
const ramp = await import(
  "data:text/javascript;base64," + Buffer.from(compiled.code).toString("base64")
);

const LOW = [59, 111, 214]; // #3b6fd6
const HIGH = [214, 75, 59]; // #d64b3b

test("signed transform: transparent at the midpoint, opaque at the ends", () => {
  const mid = ramp.rampColor(0.5, "real", LOW, HIGH);
  assert.equal(mid[3], 0, "midpoint alpha is 0");
  assert.equal(ramp.rampColor(0, "real", LOW, HIGH)[3], 1, "low end alpha is 1");
  assert.equal(ramp.rampColor(1, "real", LOW, HIGH)[3], 1, "high end alpha is 1");
  // the low end takes the low color, the high end the high color
  assert.deepEqual(ramp.rampColor(0, "real", LOW, HIGH).slice(0, 3), LOW);
  assert.deepEqual(ramp.rampColor(1, "real", LOW, HIGH).slice(0, 3), HIGH);
});

test("non-negative transform: transparent at the low end, opaque at the high end", () => {
  assert.equal(ramp.rampColor(0, "abs", LOW, HIGH)[3], 0, "low end alpha is 0");
  assert.equal(ramp.rampColor(1, "abs", LOW, HIGH)[3], 1, "high end alpha is 1");
  // a non-negative transform uses the high color throughout
  assert.deepEqual(ramp.rampColor(1, "abs", LOW, HIGH).slice(0, 3), HIGH);
  assert.deepEqual(ramp.rampColor(0.5, "logabs", LOW, HIGH).slice(0, 3), HIGH);
});

test("phase is signed, logabs is not", () => {
  assert.equal(ramp.isSigned("phase"), true);
  assert.equal(ramp.isSigned("imag"), true);
  assert.equal(ramp.isSigned("logabs"), false);
  assert.equal(ramp.isSigned("abs"), false);
});

test("logabs floors the range log_range decades below the plane's maximum", () => {
  // values 1e-9 .. 1e0: the raw min is -9 decades, but a 6-decade floor lifts it
  const values = new Float32Array([-9, -6, -3, 0]);
  const [min, max] = ramp.valueRange(values, "logabs", 6);
  assert.equal(max, 0);
  assert.equal(min, -6, "the floor is 6 decades below the maximum");
});

test("|E| is the total E magnitude over the recorded E components", () => {
  // one cell, Ex = 3, Ey = 4 (real), Ez = 0: |E| = 5. Hx is recorded too and
  // must not contribute.
  const components = ["Ex", "Ey", "Ez", "Hx"];
  const real = new Float32Array([3, 4, 0, 100]);
  const imag = new Float32Array([0, 0, 0, 0]);
  const field = {
    id: "d", name: "d", components,
    grid: [1, 1],
    real: real.buffer, imag: imag.buffer,
  };
  const total = ramp.componentData(field, ramp.TOTAL_COMPONENT);
  assert.equal(total.real[0], 5, "sqrt(3^2 + 4^2) = 5");
  assert.equal(total.imag[0], 0, "the magnitude is real");
});

test("|E| carries the imaginary part of each component", () => {
  // Ex = 3j, Ey = 4 (real), Ez = 0: |E| = 5
  const real = new Float32Array([0, 4, 0]);
  const imag = new Float32Array([3, 0, 0]);
  const field = {
    id: "d", name: "d", components: ["Ex", "Ey", "Ez"],
    grid: [1, 1],
    real: real.buffer, imag: imag.buffer,
  };
  const total = ramp.componentData(field, ramp.TOTAL_COMPONENT);
  assert.equal(total.real[0], 5);
});

test("a component the detector did not record is null", () => {
  const field = {
    id: "d", name: "d", components: ["Ex"],
    grid: [1, 1],
    real: new Float32Array([1]).buffer, imag: new Float32Array([0]).buffer,
  };
  assert.equal(ramp.componentData(field, "Ey"), null);
});

// a FieldDetector-like payload: real samples only (no imag), three time frames
// of one component on a 1x2 plane
function timeField(frames) {
  return {
    id: "d", name: "d", components: ["Ey"], grid: [1, 2], frame_kind: "time",
    times: frames.map((_, i) => i * 1e-15),
    real: new Float32Array(frames.flat()).buffer,
  };
}

test("a real-valued recording reads as imaginary zero, frame by frame", () => {
  const field = timeField([[1, 2], [3, -4], [5, 6]]);
  assert.equal(ramp.frameCount(field), 3);
  const second = ramp.componentData(field, "Ey", 1);
  assert.deepEqual([...second.real], [3, -4]);
  assert.deepEqual([...second.imag], [0, 0]);
  // a frame past the end clamps to the last one
  assert.deepEqual([...ramp.componentData(field, "Ey", 99).real], [5, 6]);
});

test("the default scale spans every frame symmetrically for signed transforms", () => {
  const field = timeField([[1, 2], [3, -4], [5, 6]]);
  // signed transforms keep [-vmax, +vmax] symmetric around transparent zero
  assert.deepEqual(ramp.valueRange(new Float32Array([1, 2]), "real", 6), [-2, 2]);
  assert.deepEqual(ramp.frameRange(field, "Ey", "real", 6), [-6, 6]);
  assert.deepEqual(ramp.frameRange(field, "Ey", "abs", 6), [1, 6]);
});

test("logabs floors the whole recording's range below its maximum", () => {
  const field = timeField([[1e-9, 1e-3], [1, 1e-12], [1e-6, 1e-4]]);
  const [min, max] = ramp.frameRange(field, "Ey", "logabs", 6);
  assert.ok(Math.abs(max - 0) < 1e-6, "the maximum is 10^0");
  assert.ok(Math.abs(min - -6) < 1e-6, "the floor is 6 decades below it");
});

test("a fixed field_range overrides the recording's range", () => {
  const field = timeField([[1, 2], [3, -4], [5, 6]]);
  const settings = { field_component: "Ey", field_transform: "real", field_range: [-1, 1] };
  assert.deepEqual(ramp.shadeRange(field, settings, 6), [-1, 1]);
  assert.deepEqual(ramp.shadeRange(field, { ...settings, field_range: null }, 6), [-6, 6]);
  // an unusable range (min >= max) falls back to the recording's
  assert.deepEqual(ramp.shadeRange(field, { ...settings, field_range: [2, 2] }, 6), [-6, 6]);
});

test("rampTexture honors the map's alpha, scaled by opacity", () => {
  const values = new Float32Array([0, 0.5, 1]);
  const range = [0, 1];
  const bytes = ramp.rampTexture(values, range, "abs", LOW, HIGH, 0.5);
  assert.equal(bytes[3], 0, "low end transparent");
  assert.equal(bytes[7], Math.round(0.5 * 0.5 * 255), "midpoint at half opacity");
  assert.equal(bytes[11], Math.round(1 * 0.5 * 255), "high end at half opacity");
});

test("|E| options and componentData reject H-only or non-Ex/Ey/Ez names and never duplicate", () => {
  assert.deepEqual(ramp.componentOptions(["Ex", "Ey", "Ez"]), ["", "|E|", "Ex", "Ey", "Ez"]);
  assert.deepEqual(ramp.componentOptions(["|E|", "Ex", "Ey"]), ["", "|E|", "Ex", "Ey"]);
  assert.deepEqual(ramp.componentOptions(["Hx", "Hy", "Hz"]), ["", "Hx", "Hy", "Hz"]);
  assert.deepEqual(ramp.componentOptions(["Efield", "Energy"]), ["", "Efield", "Energy"]);

  const hOnly = {
    id: "h", name: "h", components: ["Hx", "Hy"], grid: [1, 1],
    real: new Float32Array([3, 4]).buffer, imag: new Float32Array([0, 0]).buffer,
  };
  assert.equal(ramp.componentData(hOnly, ramp.TOTAL_COMPONENT), null);

  const nonCartesianE = {
    id: "e0", name: "e0", components: ["Efield"], grid: [1, 1],
    real: new Float32Array([5]).buffer, imag: new Float32Array([0]).buffer,
  };
  assert.equal(ramp.componentData(nonCartesianE, ramp.TOTAL_COMPONENT), null);

  const precomputedTotal = {
    id: "tot", name: "tot", components: ["|E|", "Ex"], grid: [1, 1],
    real: new Float32Array([9, 3]).buffer, imag: new Float32Array([0, 0]).buffer,
  };
  assert.equal(ramp.componentData(precomputedTotal, ramp.TOTAL_COMPONENT).real[0], 9);
});

test("all-zero signed fields and non-finite samples stay transparent (alpha = 0)", () => {
  for (const transform of ["real", "imag", "phase"]) {
    const zeroRange = ramp.valueRange(new Float32Array([0, 0]), transform, 6);
    assert.deepEqual(zeroRange, [0, 0]);
    const zeroBytes = ramp.rampTexture(new Float32Array([0, 0]), zeroRange, transform, LOW, HIGH, 1);
    assert.equal(zeroBytes[3], 0, `${transform} zero field must have alpha=0 at cell 0`);
    assert.equal(zeroBytes[7], 0, `${transform} zero field must have alpha=0 at cell 1`);
  }

  const signedValues = new Float32Array([0, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 2]);
  assert.deepEqual(ramp.valueRange(signedValues, "real", 6), [-2, 2]);
  const signedBytes = ramp.rampTexture(signedValues, [-2, 2], "real", LOW, HIGH, 1);
  assert.equal(signedBytes[3], 0, "signed zero maps to transparent midpoint");
  assert.equal(signedBytes[7], 0, "NaN maps to transparent");
  assert.equal(signedBytes[11], 0, "+Infinity maps to transparent");
  assert.equal(signedBytes[15], 0, "-Infinity maps to transparent");
  assert.equal(signedBytes[19], 255, "finite peak stays opaque");

  const absValues = new Float32Array([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 1]);
  const absBytes = ramp.rampTexture(absValues, [0, 1], "abs", LOW, HIGH, 1);
  assert.equal(absBytes[3], 0);
  assert.equal(absBytes[7], 0);
  assert.equal(absBytes[11], 0);
  assert.equal(absBytes[15], 255);
});
