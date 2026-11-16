// Camera framing and bounds-equality helpers for Scene3D (E-BF-22).
import assert from "node:assert/strict";
import test from "node:test";

import { build } from "esbuild";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const stubThreePlugin = {
  name: "stub-three",
  setup(pluginBuild) {
    pluginBuild.onResolve({ filter: /^three(\/|$)/ }, (args) => ({
      path: args.path,
      namespace: "stub-three",
    }));
    pluginBuild.onLoad({ filter: /.*/, namespace: "stub-three" }, (args) => {
      if (args.path === "three") {
        return {
          loader: "js",
          contents: `
            export class Vector3 {
              constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
              normalize() { return this; }
              clone() { return new Vector3(this.x, this.y, this.z); }
            }
            export class Scene {}
            export class Group {}
            export class PerspectiveCamera {}
            export class WebGLRenderer {}
            export class EventDispatcher {}
          `,
        };
      }
      return {
        loader: "js",
        contents: `
          export class OrbitControls {}
          export class TrackballControls {}
          export class ArcballControls {}
        `,
      };
    });
  },
};

const outfile = join(mkdtempSync(join(tmpdir(), "scene-helpers-")), "scene.mjs");
writeFileSync(
  outfile,
  (
    await build({
      entryPoints: [join(here, "scene.ts")],
      bundle: true,
      format: "esm",
      write: false,
      plugins: [stubThreePlugin],
    })
  ).outputFiles[0].text,
);
const { fitDistance, sameBounds } = await import(outfile);

test("fitDistance scales distance with zExaggeration and narrow viewport aspect", () => {
  const flatLandscape = fitDistance([10, 10, 2], 1, 45, 16 / 9);
  const tallLandscape = fitDistance([10, 10, 2], 5, 45, 16 / 9);
  assert.ok(tallLandscape > flatLandscape, "larger zExaggeration increases fit distance");

  const square = fitDistance([10, 10, 2], 1, 45, 1);
  const wide = fitDistance([10, 10, 2], 1, 45, 2);
  assert.equal(wide, square, "landscape aspect >= 1 keeps vertical framing unchanged");

  const narrow = fitDistance([10, 10, 2], 1, 45, 0.5);
  assert.ok(
    Math.abs(narrow - square * 2) < 1e-9,
    "aspect < 1 backs camera out inversely with aspect to prevent horizontal crop",
  );
});

test("sameBounds matches identical boxes within tolerance and detects shifts", () => {
  assert.equal(sameBounds([4, 2, 1], [0, 0, 0.5], [4, 2, 1], [0, 0, 0.5]), true);
  assert.equal(sameBounds([4, 2, 1], [0, 0, 0.5], [4 + 1e-8, 2, 1], [0, 0, 0.5]), true);
  assert.equal(sameBounds([4, 2, 1], [0, 0, 0.5], [4.1, 2, 1], [0, 0, 0.5]), false);
  assert.equal(sameBounds([4, 2, 1], [0, 0, 0.5], [4, 2, 1], [0.2, 0, 0.5]), false);
});
