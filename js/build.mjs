// Bundles the viewer frontend into the committed ES module anywidget loads
// (spec D6). The output goes straight into the Python package so flit ships it;
// `dist/` is gitignored, so there is no dist step.
import { build } from "esbuild";

await build({
  entryPoints: ["viewer/index.ts"],
  bundle: true,
  format: "esm",
  target: "es2020",
  platform: "browser",
  outfile: "../bifrost/viewer/static/viewer.js",
  sourcemap: false,
  logLevel: "info",
});

await build({
  entryPoints: ["viewer/style.css"],
  bundle: true,
  outfile: "../bifrost/viewer/static/viewer.css",
  logLevel: "info",
});

// The precomputed demo (docs/demo): the same viewer on the exported files, with
// no Python. Minified, since the docs page loads it from the web.
await build({
  entryPoints: ["viewer/demo.ts"],
  bundle: true,
  format: "esm",
  target: "es2020",
  platform: "browser",
  outfile: "../docs/demo/viewer-demo.js",
  minify: true,
  sourcemap: false,
  logLevel: "info",
});

await build({
  entryPoints: ["viewer/style.css"],
  bundle: true,
  outfile: "../docs/demo/viewer.css",
  minify: true,
  logLevel: "info",
});
