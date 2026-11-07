// The precomputed demo: opens the exported files, mounts the viewer on them and
// reports loading and errors.
//
// Two ways in, one bundle (docs/demo/viewer-demo.js):
//   - the demo page (docs/demo/index.html) holds `<div id="viewer" data-src="data/">`
//     and `<div id="status">`; `start` below fills them when the script loads;
//   - the docs notebook (docs/notebooks/widget_demo.py) imports this module and
//     calls `mountInto` on the element its widget is given.

import viewer from "./index";
import { StaticModel, fetchSource, type DemoStatus } from "./static";

/** Which way the camera looks from at the start: a vector from the device's center. */
const INITIAL_VIEW: [number, number, number] = [0.25, -0.55, 1];

type Say = (text: string, kind?: "info" | "error" | "") => void;

function baseUrl(src: string, relativeTo: string = window.location.href): string {
  const url = new URL(src, relativeTo).href;
  return url.endsWith("/") ? url : `${url}/`;
}

/** Opens the exported files under `base` and shows the viewer in `host`. */
export async function mountReplay(host: HTMLElement, base: string, say: Say): Promise<void> {
  say("Loading the demo…");
  let model: StaticModel;
  try {
    model = await StaticModel.open(fetchSource(base));
  } catch (error) {
    say(`The demo could not load: ${error instanceof Error ? error.message : String(error)}. Reload the page to try again.`, "error");
    return;
  }
  model.set("initial_view", { direction: INITIAL_VIEW });
  viewer.render({ model, el: host });
  say("");

  // chips for slices and fields that are still loading, and the last error
  const showStatus = (): void => {
    const current = model.get("static_status") as DemoStatus;
    if (current.error) say(`${current.error}. Reload the page to try again.`, "error");
    else if (current.busy.length) say(`Loading ${current.busy.join(", ")}…`);
    else say("");
  };
  model.on("change:static_status", showStatus);
  void model.loadFields();
}

/**
 * The notebook's way in: `el` is the element the widget gives us and `src` the folder of
 * exported files, relative to the page (the notebook page, not this module). The viewer
 * fills `el`'s height, so the caller sizes `el`; the loading message sits over it.
 */
export async function mountInto(el: HTMLElement, src: string): Promise<void> {
  const status = document.createElement("div");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.style.cssText =
    "position:absolute;left:12px;bottom:12px;z-index:10;max-width:calc(100% - 24px);" +
    "padding:6px 10px;border-radius:6px;background:rgba(0,0,0,.72);color:#fff;font:14px system-ui,sans-serif";
  const host = document.createElement("div");
  host.style.cssText = "position:absolute;inset:0";
  el.style.position = "relative";
  el.replaceChildren(host, status);
  const say: Say = (text, kind = "info") => {
    status.textContent = text;
    status.style.background = kind === "error" ? "#a1262d" : "rgba(0,0,0,.72)";
    status.hidden = text === "";
  };
  await mountReplay(host, baseUrl(src, document.baseURI), say);
}

async function start(): Promise<void> {
  const host = document.getElementById("viewer");
  const status = document.getElementById("status");
  if (!host || !status) return;
  const say: Say = (text, kind = "info") => {
    status.textContent = text;
    status.dataset.kind = kind;
    status.hidden = text === "";
  };
  await mountReplay(host, baseUrl(host.dataset.src ?? "data/"), say);
}

void start();
