// The 3D viewport: perspective camera, an interchangeable rotation control
// (trackball/arcball/orbit, spec R3), fit-to-view and the simulation region.
import * as THREE from "three";
import { ArcballControls } from "three/examples/jsm/controls/ArcballControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TrackballControls } from "three/examples/jsm/controls/TrackballControls.js";

import { REGION_COLOR, formatTick, niceStep, tickValues } from "./ticks";

//: The three rotation styles the settings screen offers (spec R3). "Trackball"
//: is the free-roll style FDTD software uses; "Orbit" is a turntable.
export type ControlKind = "trackball" | "arcball" | "orbit";
export const CONTROL_KINDS: ControlKind[] = ["trackball", "arcball", "orbit"];

interface ControlHooks {
  onStart: () => void;
  onChange: () => void;
  onEnd: () => void;
}

/** The small surface the scene needs from a three.js control. */
interface Controls {
  enabled: boolean;
  //: drag rotation sensitivity; all three controls spell it `rotateSpeed`
  rotateSpeed: number;
  target: THREE.Vector3;
  update(): boolean;
  //: re-read the canvas's on-screen size. TrackballControls caches it in its
  //: constructor and divides by it when rotating, so a control built while the
  //: canvas is hidden (the settings screen is up) would stay broken forever
  refreshLayout(): void;
  bind(hooks: ControlHooks): () => void;
  dispose(): void;
}

interface ControlDispatcher {
  addEventListener(type: "start" | "change" | "end", listener: () => void): void;
  removeEventListener(type: "start" | "change" | "end", listener: () => void): void;
}

function bindControlEvents(
  dispatcher: ControlDispatcher,
  hooks: ControlHooks,
): () => void {
  dispatcher.addEventListener("start", hooks.onStart);
  dispatcher.addEventListener("change", hooks.onChange);
  dispatcher.addEventListener("end", hooks.onEnd);
  return () => {
    dispatcher.removeEventListener("start", hooks.onStart);
    dispatcher.removeEventListener("change", hooks.onChange);
    dispatcher.removeEventListener("end", hooks.onEnd);
  };
}

/** Whether two bounding boxes match within `tol` (µm). */
export function sameBounds(
  aSize: readonly [number, number, number],
  aOrigin: readonly [number, number, number],
  bSize: readonly [number, number, number],
  bOrigin: readonly [number, number, number],
  tol = 1e-6,
): boolean {
  for (let i = 0; i < 3; i += 1) {
    if (Math.abs(aSize[i] - bSize[i]) > tol || Math.abs(aOrigin[i] - bOrigin[i]) > tol) {
      return false;
    }
  }
  return true;
}

/** Camera distance that frames the z-exaggerated box vertically and horizontally even in narrow viewports. */
export function fitDistance(
  size: readonly [number, number, number],
  zExaggeration: number,
  fovDeg: number,
  aspect: number,
): number {
  const radius = 0.5 * Math.hypot(size[0], size[1], size[2] * zExaggeration);
  const halfFov = (fovDeg * Math.PI) / 360;
  const span = Math.sin(halfFov) * Math.min(1, Math.max(aspect, 1e-6));
  return Math.max(radius, 1e-6) / Math.max(span, 1e-6);
}

//: The default view direction: x horizontal, camera at -y, +z (spec R3).
const VIEW_DIRECTION = new THREE.Vector3(0, -1, 1).normalize();

class OrbitControl implements Controls {
  readonly inner: OrbitControls;
  constructor(camera: THREE.PerspectiveCamera, dom: HTMLElement) {
    this.inner = new OrbitControls(camera, dom);
    this.inner.enableDamping = true;
  }
  get rotateSpeed(): number {
    return this.inner.rotateSpeed;
  }
  set rotateSpeed(value: number) {
    this.inner.rotateSpeed = value;
  }
  get enabled(): boolean {
    return this.inner.enabled;
  }
  set enabled(value: boolean) {
    this.inner.enabled = value;
  }
  get target(): THREE.Vector3 {
    return this.inner.target;
  }
  update(): boolean {
    return this.inner.update();
  }
  refreshLayout(): void {}
  bind(hooks: ControlHooks): () => void {
    return bindControlEvents(this.inner, hooks);
  }
  dispose(): void {
    this.inner.dispose();
  }
}

class TrackballControl implements Controls {
  readonly inner: TrackballControls;
  constructor(camera: THREE.PerspectiveCamera, dom: HTMLElement) {
    this.inner = new TrackballControls(camera, dom);
    // no inertia: the view follows the drag 1:1 and stops when the drag ends
    // (three defaults to staticMoving=false, which glides after release)
    this.inner.staticMoving = true;
  }
  get rotateSpeed(): number {
    return this.inner.rotateSpeed;
  }
  set rotateSpeed(value: number) {
    this.inner.rotateSpeed = value;
  }
  get enabled(): boolean {
    return this.inner.enabled;
  }
  set enabled(value: boolean) {
    this.inner.enabled = value;
  }
  get target(): THREE.Vector3 {
    return this.inner.target;
  }
  update(): boolean {
    this.inner.update();
    return false;
  }
  refreshLayout(): void {
    this.inner.handleResize();
  }
  bind(hooks: ControlHooks): () => void {
    return bindControlEvents(this.inner, hooks);
  }
  dispose(): void {
    this.inner.dispose();
  }
}

class ArcballControl implements Controls {
  readonly inner: ArcballControls;
  constructor(camera: THREE.PerspectiveCamera, dom: HTMLElement, scene: THREE.Scene) {
    this.inner = new ArcballControls(camera, dom, scene);
    // the gizmo rings would fight the region outline / ticks; hide them
    this.inner.setGizmosVisible(false);
  }
  get rotateSpeed(): number {
    return this.inner.rotateSpeed;
  }
  set rotateSpeed(value: number) {
    this.inner.rotateSpeed = value;
  }
  get enabled(): boolean {
    return this.inner.enabled;
  }
  set enabled(value: boolean) {
    this.inner.enabled = value;
  }
  get target(): THREE.Vector3 {
    // ArcballControls has a `target` at runtime, but @types/three omits it
    return (this.inner as unknown as { target: THREE.Vector3 }).target;
  }
  update(): boolean {
    this.inner.update();
    return false;
  }
  refreshLayout(): void {}
  bind(hooks: ControlHooks): () => void {
    return bindControlEvents(this.inner, hooks);
  }
  dispose(): void {
    this.inner.dispose();
  }
}

export class Scene3D {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  //: everything whose z is exaggerated lives under here
  readonly content = new THREE.Group();
  private controls!: Controls;
  private controlKind: ControlKind = "trackball";
  //: whether the renderer is on screen; a control built while the settings
  //: screen is up must not come back enabled
  private viewportVisible = true;
  //: drag rotation sensitivity (spec R3, settings screen); 1 is the control's
  //: own default, so a rebuilt control keeps whatever the user chose
  private rotateSpeed = 1;
  private readonly resizeObserver: ResizeObserver;
  private target = new THREE.Vector3(0, 0, 0);
  private boxSize: [number, number, number] = [1, 1, 1];
  private hasBounds = false;
  //: which way the camera looks from, as a unit vector from the target (spec R3)
  private viewDirection = VIEW_DIRECTION.clone();
  private zExaggeration = 1;
  private frameId = 0;
  private disposed = false;
  private needsRender = true;
  private controlsActive = false;
  private updatingControls = false;
  private unbindControls: (() => void) | null = null;
  //: the simulation region outline (the device box, spec R3); drawn in the
  //: content group so the vertical exaggeration scales it with the solids. Each
  //: edge is a separate line in its axis's color, so the box edges read as the
  //: axes they measure
  private readonly region = new THREE.Group();
  //: axis tick lines + labels; drawn in the content group so they follow the
  //: box when the vertical exaggeration changes. A sprite's size comes from the
  //: world-space lengths of its own x/y axes, so the z-only scale moves the
  //: labels without stretching them
  private readonly ticks = new THREE.Group();
  //: sprite label size in um, set from the box span
  private labelScale = 0.1;
  //: the box center (spec C5); the wire is local, so ticks are labelled with
  //: `local + origin` to read in device (stack) coordinates, like the slider
  private origin: [number, number, number] = [0, 0, 0];

  constructor(private readonly container: HTMLElement) {
    this.scene.background = new THREE.Color(0x1b1b1f);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 2.0));
    // key from the default view's side (-y, +z), fill opposite, so the initial
    // camera sees a lit face (spec R3)
    const key = new THREE.DirectionalLight(0xffffff, 2.5);
    key.position.set(3, -4, 5);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 1.0);
    fill.position.set(-4, 3, -3);
    this.scene.add(fill);

    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1e5);
    // Z-up must be set before constructing OrbitControls so its turntable axis matches stack coordinates
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0, -2, 2);

    // preserveDrawingBuffer keeps the frame readable after compositing, so a
    // whole-widget capture (`captureShell`) can read this canvas without it coming
    // out blank (spec D14)
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    container.appendChild(this.renderer.domElement);

    this.controls = this.attachControls(this.controlKind);
    this.renderer.domElement.addEventListener("pointercancel", this.onPointerCancel);
    document.addEventListener("visibilitychange", this.onVisibilityChange);

    this.scene.add(this.content);
    this.content.add(this.region);
    this.content.add(this.ticks);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.requestRender();
  }

  /** Frames the given box (µm, local origin) and sets up the region + scale bar. */
  setBounds(size: [number, number, number], origin: [number, number, number] = [0, 0, 0]): void {
    const unchanged = this.hasBounds && sameBounds(this.boxSize, this.origin, size, origin);
    this.hasBounds = true;
    this.boxSize = [...size] as [number, number, number];
    this.origin = [...origin] as [number, number, number];
    const span = Math.max(size[0], size[1], size[2]);
    this.labelScale = 0.08 * span;
    this.buildRegion(size);
    if (!unchanged) this.fit();
    else this.requestRender();
  }

  /** Builds the requested rotation control over the renderer's canvas and binds render hooks. */
  private attachControls(kind: ControlKind): Controls {
    if (kind === "orbit") this.camera.up.set(0, 0, 1);
    const dom = this.renderer.domElement;
    let controls: Controls;
    if (kind === "trackball") controls = new TrackballControl(this.camera, dom);
    else if (kind === "arcball") controls = new ArcballControl(this.camera, dom, this.scene);
    else controls = new OrbitControl(this.camera, dom);
    this.unbindControls = controls.bind({
      onStart: () => {
        this.controlsActive = true;
        this.requestRender();
      },
      onChange: () => {
        if (!this.updatingControls) this.requestRender();
      },
      onEnd: () => {
        if (this.controlsActive) {
          this.controlsActive = false;
          this.requestRender();
        }
      },
    });
    return controls;
  }

  /** The active rotation style (spec R3, settings screen). */
  get control(): ControlKind {
    return this.controlKind;
  }

  /** Switches the rotation style, keeping the current camera pose (spec R3). */
  setControl(kind: ControlKind): void {
    if (kind === this.controlKind) return;
    // TrackballControls leaves its event listeners on the document, so build
    // the replacement first and only then tear the old one down
    const previous = this.controls;
    const prevUnbind = this.unbindControls;
    this.target.copy(previous.target);
    this.controls = this.attachControls(kind);
    this.controlKind = kind;
    prevUnbind?.();
    previous.dispose();
    this.controls.target.copy(this.target);
    this.controls.enabled = this.viewportVisible;
    this.controls.rotateSpeed = this.rotateSpeed;
    this.controls.refreshLayout();
    this.requestRender();
  }

  /** Hides the renderer while the settings screen replaces the viewport (spec R3). */
  setViewportVisible(visible: boolean): void {
    this.viewportVisible = visible;
    this.renderer.domElement.style.visibility = visible ? "" : "hidden";
    this.controls.enabled = visible;
    // the canvas has a real size again; trackball caches it, so re-read it
    if (visible) {
      this.controls.refreshLayout();
      if (this.needsRender) this.requestRender();
    } else {
      this.controlsActive = false;
      if (this.frameId !== 0) {
        cancelAnimationFrame(this.frameId);
        this.frameId = 0;
        this.needsRender = true;
      }
    }
  }

  /** Drag rotation sensitivity (spec R3, settings screen). */
  get sensitivity(): number {
    return this.rotateSpeed;
  }

  setSensitivity(factor: number): void {
    this.rotateSpeed = factor;
    this.controls.rotateSpeed = factor;
  }

  /** Outlines the simulation region box and ticks each of its axes (spec R3). */
  private buildRegion(size: [number, number, number]): void {
    const [sx, sy, sz] = size.map((s) => s / 2);
    const corners = [
      new THREE.Vector3(-sx, -sy, -sz),
      new THREE.Vector3(sx, -sy, -sz),
      new THREE.Vector3(sx, sy, -sz),
      new THREE.Vector3(-sx, sy, -sz),
      new THREE.Vector3(-sx, -sy, sz),
      new THREE.Vector3(sx, -sy, sz),
      new THREE.Vector3(sx, sy, sz),
      new THREE.Vector3(-sx, sy, sz),
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];

    this.clearGroup(this.region);
    const points: THREE.Vector3[] = [];
    for (const [a, b] of edges) points.push(corners[a], corners[b]);
    const outline = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: REGION_COLOR, depthTest: false }),
    );
    outline.renderOrder = 9;
    this.region.add(outline);

    this.buildTicks(size);
  }

  /** Tick marks and labels along the box's x, y and z edges (spec R3). */
  private buildTicks(size: [number, number, number]): void {
    this.clearGroup(this.ticks);
    const half = size.map((s) => s / 2);
    const step = niceStep(Math.max(...size));
    const tick = 0.04 * Math.max(...size) * (2 / 3);
    // per axis: `along` is the labelled axis, `base` the corner its ticks run
    // from, and `dir` the unit vector the marks step out along. z sits on the
    // +x/-y edge and steps fully along +x, so it reads as the z axis instead of
    // a second y. The marks and labels stay the plain outline color (spec R3)
    const corner = (sx: number, sy: number): THREE.Vector3 =>
      new THREE.Vector3(sx * half[0], sy * half[1], -half[2]);
    const axes = [
      { along: 0, name: "x", base: corner(-1, -1), dir: new THREE.Vector3(0, -1, 0) },
      { along: 1, name: "y", base: corner(-1, -1), dir: new THREE.Vector3(-1, 0, 0) },
      { along: 2, name: "z", base: corner(1, -1), dir: new THREE.Vector3(1, 0, 0) },
    ];
    for (const { along, name, base, dir } of axes) {
      const color = REGION_COLOR;
      const points: THREE.Vector3[] = [];
      // ticks sit at nice device coordinates (local + box center, spec C5)
      const o = this.origin[along];
      for (const value of tickValues(-half[along] + o, half[along] + o, step)) {
        const at = base.clone();
        at.setComponent(along, value - o);
        const out = at.clone().addScaledVector(dir, tick);
        points.push(at, out); // the tick mark
        // the original (2/3)*labelScale relation, pulled 3x closer to the mark
        this.addLabel(
          formatTick(value),
          out,
          dir,
          color,
          ((2 / 3) * this.labelScale) / 3,
          this.labelScale,
        );
      }
      // name the axis at the middle of its ticked edge, in a larger face and a
      // little further out than the tick labels so the two never collide (a tick
      // often sits at the midpoint too)
      const middle = base.clone();
      middle.setComponent(along, 0);
      // the name sits further out than the tick labels; its gap is 2/3 of the
      // 1.5 * labelScale it started at
      this.addLabel(name, middle, dir, color, this.labelScale * 1.5 * (2 / 3), this.labelScale * 1.5);
      const lines = new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({ color, depthTest: false }),
      );
      lines.renderOrder = 10;
      this.ticks.add(lines);
    }
  }

  /** A canvas-texture sprite label, nudged `gap` clear of the mark it names. */
  private addLabel(
    text: string,
    at: THREE.Vector3,
    dir: THREE.Vector3,
    color: string,
    gap: number,
    size: number,
  ): void {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = color;
    ctx.font = "32px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 64, 32);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }),
    );
    sprite.position.copy(at).addScaledVector(dir, gap);
    sprite.scale.set(size, size * 0.5, 1);
    this.ticks.add(sprite);
  }

  /** Frames the whole scene (spec R3, "fit to view").
   *
   * Uses Z-up for turntable orbiting without gimbal lock; for the default view
   * from (-y, +z) toward the box center, the camera's right axis is +x.
   */
  fit(): void {
    const distance = fitDistance(this.boxSize, this.zExaggeration, this.camera.fov, this.camera.aspect);
    this.target.set(0, 0, 0);
    // keep Z-up for OrbitControls turntable rotation; fall back to +Y up only
    // when looking straight along Z under trackball/arcball to avoid lookAt degeneracy
    if (this.controlKind === "orbit" || Math.abs(this.viewDirection.z) <= 0.999) {
      this.camera.up.set(0, 0, 1);
    } else {
      this.camera.up.set(0, 1, 0);
    }
    this.camera.position.copy(this.target).addScaledVector(this.viewDirection, distance);
    this.camera.near = Math.max(distance / 1000, 1e-3);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(this.target);
    if (this.viewportVisible && !document.hidden) {
      this.renderNow();
    } else {
      this.updatingControls = true;
      try {
        this.controls.update();
      } finally {
        this.updatingControls = false;
      }
      this.needsRender = true;
    }
  }

  /** Sets the direction `fit` places the camera in (a vector from the target; need not be unit). */
  setViewDirection(direction: [number, number, number]): void {
    const next = new THREE.Vector3(...direction);
    if (next.lengthSq() > 0) this.viewDirection.copy(next.normalize());
  }

  /** Scales the z of everything under `content` (spec R3, z-exaggeration). */
  setZExaggeration(factor: number): void {
    this.zExaggeration = factor;
    this.content.scale.set(1, 1, factor);
    this.requestRender();
  }

  get exaggeration(): number {
    return this.zExaggeration;
  }

  /** Schedules a single frame on demand (skipped while hidden until visible again). */
  requestRender(): void {
    if (this.disposed) return;
    if (!this.viewportVisible || document.hidden) {
      this.needsRender = true;
      return;
    }
    if (this.frameId !== 0) return;
    this.frameId = requestAnimationFrame(this.renderLoop);
  }

  /** The current frame as a PNG data URL (spec R3, "save image"). */
  toPng(): string {
    this.renderNow();
    return this.renderer.domElement.toDataURL("image/png");
  }

  /** Renders now and returns the canvas, so a video frame captures this state. */
  captureCanvas(): HTMLCanvasElement {
    this.renderNow();
    return this.renderer.domElement;
  }

  private resize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    // trackball caches the canvas rect, so hand it the new one
    this.controls.refreshLayout();
    if (this.viewportVisible && !document.hidden) this.renderNow();
    else this.needsRender = true;
  }

  private renderLoop = (): void => {
    this.frameId = 0;
    if (this.disposed) return;
    if (!this.viewportVisible || document.hidden) {
      this.needsRender = true;
      return;
    }
    this.renderNow();
  };

  private renderNow(): void {
    if (this.disposed) return;
    if (this.frameId !== 0) {
      cancelAnimationFrame(this.frameId);
      this.frameId = 0;
    }
    this.needsRender = false;
    let moved = false;
    this.updatingControls = true;
    try {
      moved = this.controls.update();
    } finally {
      this.updatingControls = false;
    }
    this.renderer.render(this.scene, this.camera);
    if (moved || this.controlsActive) this.requestRender();
  }

  private readonly onVisibilityChange = (): void => {
    if (!document.hidden && this.needsRender) this.requestRender();
  };

  private readonly onPointerCancel = (): void => {
    if (this.controlsActive) {
      this.controlsActive = false;
      this.requestRender();
    }
  };

  private clearGroup(group: THREE.Group): void {
    for (const child of [...group.children]) {
      group.remove(child);
      if (child instanceof THREE.Sprite) {
        // THREE.Sprite shares a global module-level _geometry: only dispose material and CanvasTexture
        child.material.map?.dispose();
        child.material.dispose();
      } else if (child instanceof THREE.LineSegments || child instanceof THREE.Mesh) {
        child.geometry.dispose();
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) material.dispose();
      }
    }
  }

  dispose(): void {
    this.disposed = true;
    if (this.frameId !== 0) {
      cancelAnimationFrame(this.frameId);
      this.frameId = 0;
    }
    document.removeEventListener("visibilitychange", this.onVisibilityChange);
    this.renderer.domElement.removeEventListener("pointercancel", this.onPointerCancel);
    this.resizeObserver.disconnect();
    this.unbindControls?.();
    this.unbindControls = null;
    this.controls.dispose();
    this.clearGroup(this.region);
    this.clearGroup(this.ticks);
    this.renderer.dispose();
  }
}
