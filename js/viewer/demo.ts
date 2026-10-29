// The precomputed demo's page script: opens the exported files, mounts the
// viewer on them and reports loading and errors on the page.
//
// docs/demo/index.html holds `<div id="viewer" data-src="data/">` and
// `<div id="status">`; this script fills them.

import viewer from "./index";
import { StaticModel, fetchSource, type DemoStatus } from "./static";

/** Which way the camera looks from at the start: a vector from the device's center. */
const INITIAL_VIEW: [number, number, number] = [0.25, -0.55, 1];

function baseUrl(host: HTMLElement): string {
  const url = new URL(host.dataset.src ?? "data/", window.location.href).href;
  return url.endsWith("/") ? url : `${url}/`;
}

async function start(): Promise<void> {
  const host = document.getElementById("viewer");
  const status = document.getElementById("status");
  if (!host || !status) return;
  const say = (text: string, kind: "info" | "error" | "" = "info"): void => {
    status.textContent = text;
    status.dataset.kind = kind;
    status.hidden = text === "";
  };

  say("Loading the demo…");
  let model: StaticModel;
  try {
    model = await StaticModel.open(fetchSource(baseUrl(host)));
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

void start();
